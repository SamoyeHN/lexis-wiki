"""
PedLex Database Packaging, Multi-Mirror Auto-Download (Hugging Face / Cloudflare R2 / GitHub),
and Atomic LZMA Decompression Bootstrapper.
"""

import gzip
import hashlib
import json
import lzma
import os
import shutil
import sqlite3
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_BOOTSTRAP_LOCK = threading.Lock()
_BOOTSTRAP_ATTEMPTED = False
_SQLITE_MAGIC = b"SQLite format 3\x00"


def _sidecar_path(db_path: Path) -> Path:
    return db_path.with_name(db_path.name + ".sha256")


def read_cached_sha256(db_path: Path) -> Optional[str]:
    """Read cached SHA-256 from sidecar if file size and mtime_ns match (<0.1ms)."""
    if not db_path.exists():
        return None
    sidecar = _sidecar_path(db_path)
    if not sidecar.exists():
        return None
    try:
        st = db_path.stat()
        raw = sidecar.read_text(encoding="utf-8").strip().split()
        if len(raw) >= 3:
            cached_hash, cached_size, cached_mtime_ns = raw[0].lower(), int(raw[1]), int(raw[2])
            if cached_size == st.st_size and cached_mtime_ns == st.st_mtime_ns and len(cached_hash) == 64:
                return cached_hash
    except Exception:
        pass
    return None


def write_cached_sha256(db_path: Path, sha256_hex: str) -> None:
    """Persist SHA-256 alongside file size and mtime_ns so startup checks take 0ms."""
    try:
        if db_path.exists():
            st = db_path.stat()
            _sidecar_path(db_path).write_text(
                f"{sha256_hex.lower()} {st.st_size} {st.st_mtime_ns}\n",
                encoding="utf-8",
            )
    except Exception:
        pass


def compute_file_sha256(file_path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    """Compute SHA-256 of a file and update its sidecar cache if it is a .db file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    digest = h.hexdigest().lower()
    if file_path.suffix == ".db":
        write_cached_sha256(file_path, digest)
    return digest


def get_pedlex_distribution_config() -> Dict[str, Any]:
    """Merge default distribution config with wiki_config.json and environment variables."""
    dist_cfg: Dict[str, Any] = {
        "auto_download": True,
        "auto_update": False,
        "expected_sha256": "92f6f4b575cc8f5a1c50f0e6a678723d0c1692839ea2096d5b8c71254172c654",
        "urls": [
            "https://r2.airopen.org/pedlex.db.xz",
            "https://huggingface.co/datasets/huanan88/lexis-asset/resolve/main/pedlex.db.xz",
            "https://hf-mirror.com/datasets/huanan88/lexis-asset/resolve/main/pedlex.db.xz",
        ],
        "auth_token": "",
        "timeout_seconds": 10,
    }
    try:
        from ..config import config
        user_dist = config.get("pedlex_distribution")
        if isinstance(user_dist, dict):
            dist_cfg.update(user_dist)
    except Exception:
        pass

    # Environment variable overrides
    env_url = os.environ.get("PEDLEX_DOWNLOAD_URL", "").strip()
    env_urls = os.environ.get("PEDLEX_URLS", "").strip()
    env_sha = os.environ.get("PEDLEX_SHA256", "").strip()
    env_token = (
        os.environ.get("PEDLEX_TOKEN", "").strip()
        or os.environ.get("HF_TOKEN", "").strip()
    )

    urls: List[str] = []
    if env_url:
        urls.append(env_url)
    if env_urls:
        for u in env_urls.split(","):
            u = u.strip()
            if u and u not in urls:
                urls.append(u)
    for u in dist_cfg.get("urls") or []:
        u = str(u).strip()
        if u and u not in urls:
            urls.append(u)

    # Automatic Hugging Face mirror expansion: if a huggingface.co URL is provided
    # without its hf-mirror.com counterpart, append hf-mirror.com as a fallback candidate
    # (never pre-empting official sources; _rank_mirrors_by_latency selects dynamically).
    expanded_urls: List[str] = []
    for u in urls:
        if u not in expanded_urls:
            expanded_urls.append(u)
        if "huggingface.co/" in u:
            mirror_u = u.replace("https://huggingface.co/", "https://hf-mirror.com/")
            if mirror_u not in expanded_urls and mirror_u not in urls:
                expanded_urls.append(mirror_u)

    dist_cfg["urls"] = expanded_urls
    if env_sha:
        dist_cfg["expected_sha256"] = env_sha.lower()
    elif dist_cfg.get("expected_sha256"):
        dist_cfg["expected_sha256"] = str(dist_cfg["expected_sha256"]).strip().lower()

    if env_token:
        dist_cfg["auth_token"] = env_token

    return dist_cfg


def _build_request(url: str, auth_token: str = "", method: str = "GET") -> urllib.request.Request:
    headers = {
        "User-Agent": "LexisWiki-PedLex-Bootstrapper/2.0",
    }
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"
        headers["X-Lexis-Token"] = auth_token
    if method == "PROBE":
        headers["Range"] = "bytes=0-0"
        return urllib.request.Request(url, headers=headers, method="GET")
    return urllib.request.Request(url, headers=headers, method=method)


def _rank_mirrors_by_latency(urls: List[str], auth_token: str = "", probe_timeout: float = 2.5) -> List[str]:
    """
    Concurrently probe all HTTP(S) mirror URLs with a 1-byte Range request and sort them
    by fastest response time (RTT). Domestic networks automatically select hf-mirror.com,
    while overseas networks automatically select huggingface.co or Cloudflare R2.
    """
    if len(urls) <= 1 or any(u.startswith("file://") for u in urls):
        return list(urls)

    results: Dict[str, float] = {}
    lock = threading.Lock()

    def _probe(u: str, order_idx: int) -> None:
        t0 = time.perf_counter()
        try:
            req = _build_request(u, auth_token=auth_token, method="PROBE")
            with urllib.request.urlopen(req, timeout=probe_timeout) as resp:
                _ = resp.read(1)
                elapsed = time.perf_counter() - t0
                with lock:
                    results[u] = elapsed
        except Exception:
            with lock:
                results[u] = 9999.0 + order_idx

    threads = [
        threading.Thread(target=_probe, args=(u, idx), daemon=True)
        for idx, u in enumerate(urls)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=probe_timeout + 0.5)

    return sorted(urls, key=lambda u: results.get(u, 9999.0))


class _ProgressReader:
    """Wraps an HTTP response stream to report download progress in MB."""

    def __init__(self, resp: Any, total_bytes: int, label: str):
        self.resp = resp
        self.total_bytes = total_bytes
        self.label = label
        self.bytes_read = 0
        self.last_print_time = 0.0
        self.start_time = time.time()

    def read(self, size: int = -1) -> bytes:
        chunk = self.resp.read(size)
        if chunk:
            self.bytes_read += len(chunk)
            now = time.time()
            if now - self.last_print_time >= 0.25 or (self.total_bytes and self.bytes_read >= self.total_bytes):
                self.last_print_time = now
                elapsed = max(0.001, now - self.start_time)
                speed_mb = (self.bytes_read / (1024 * 1024)) / elapsed
                mb_done = self.bytes_read / (1024 * 1024)
                if self.total_bytes > 0:
                    mb_total = self.total_bytes / (1024 * 1024)
                    pct = min(100.0, (self.bytes_read / self.total_bytes) * 100.0)
                    sys.stderr.write(
                        f"\r  [PedLex] Downloading {self.label}: {mb_done:.1f}/{mb_total:.1f} MB ({pct:.0f}%) @ {speed_mb:.1f} MB/s"
                    )
                else:
                    sys.stderr.write(
                        f"\r  [PedLex] Downloading {self.label}: {mb_done:.1f} MB @ {speed_mb:.1f} MB/s"
                    )
                sys.stderr.flush()
        return chunk


def download_and_extract_pedlex(
    urls: List[str],
    target_path: Path,
    expected_sha256: str = "",
    auth_token: str = "",
    timeout_seconds: int = 10,
) -> Tuple[bool, str]:
    """
    Try candidate mirror URLs in order, stream-decompressing (.xz / .gz / raw .db)
    into a temporary file, verifying SQLite magic header and SHA-256, then atomically
    replacing target_path.
    """
    if not urls:
        return False, "No download URLs configured in pedlex_distribution.urls or PEDLEX_DOWNLOAD_URL."

    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_name(target_path.name + f".{os.getpid()}.tmp")

    ranked_urls = _rank_mirrors_by_latency(urls, auth_token=auth_token)
    errors: List[str] = []
    for idx, url in enumerate(ranked_urls, 1):
        host = url.split("/")[2] if "://" in url and len(url.split("/")) > 2 else url
        url_path = url.split("?", 1)[0].lower()
        sys.stderr.write(f"[PedLex] Checking mirror [{idx}/{len(ranked_urls)}] ({host})...\n")
        sys.stderr.flush()

        try:
            req = _build_request(url, auth_token=auth_token)
            with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
                total_bytes = int(resp.headers.get("Content-Length") or 0)
                progress_stream = _ProgressReader(resp, total_bytes, host)

                if url_path.endswith(".xz") or url_path.endswith(".lzma"):
                    reader: Any = lzma.LZMAFile(progress_stream, mode="rb")
                elif url_path.endswith(".gz"):
                    reader = gzip.GzipFile(fileobj=progress_stream, mode="rb")
                else:
                    reader = progress_stream

                hasher = hashlib.sha256()
                first_chunk = True
                total_written = 0

                with open(tmp_path, "wb") as out_f:
                    while True:
                        block = reader.read(2 * 1024 * 1024)
                        if not block:
                            break
                        if first_chunk:
                            first_chunk = False
                            if len(block) < 16 or not block.startswith(_SQLITE_MAGIC):
                                raise ValueError(
                                    "Decompressed stream does not start with a valid SQLite3 header."
                                )
                        out_f.write(block)
                        hasher.update(block)
                        total_written += len(block)

                if hasattr(reader, "close") and reader is not progress_stream:
                    reader.close()

                sys.stderr.write("\n")
                sys.stderr.flush()

                actual_sha256 = hasher.hexdigest().lower()
                if expected_sha256 and actual_sha256 != expected_sha256.lower():
                    raise ValueError(
                        f"SHA-256 mismatch: expected {expected_sha256[:12]}..., got {actual_sha256[:12]}..."
                    )

                try:
                    import sys as _sys
                    _ling_mod = _sys.modules.get("librarian.linguistics")
                    if _ling_mod is not None:
                        _le = getattr(_ling_mod, "LinguisticEngine", None)
                        if _le is not None and getattr(_le, "_ldoce_conn", None) is not None:
                            _le._ldoce_conn.close()
                            _le._ldoce_conn = None
                except Exception:
                    pass

                try:
                    os.replace(tmp_path, target_path)
                except PermissionError:
                    shutil.copyfile(tmp_path, target_path)
                    tmp_path.unlink(missing_ok=True)
                write_cached_sha256(target_path, actual_sha256)
                mb_written = total_written / (1024 * 1024)
                msg = f"Successfully installed pedlex.db ({mb_written:.1f} MB, sha256={actual_sha256[:12]}...) from {host}"
                sys.stderr.write(f"[PedLex] {msg}\n")
                sys.stderr.flush()
                return True, msg

        except Exception as exc:
            sys.stderr.write("\n")
            err_msg = f"{host}: {exc}"
            errors.append(err_msg)
            sys.stderr.write(f"[PedLex] Mirror [{idx}/{len(urls)}] failed ({err_msg}).\n")
            sys.stderr.flush()
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass

    return False, "; ".join(errors)


def ensure_pedlex_database(target_path: Path, force_check: bool = False) -> bool:
    """
    Fast-path runtime check called by PedlexDbMixin._ensure_ldoce_conn().
    - If target_path exists and matches expected SHA-256 (or auto_update is False), returns True in 0ms.
    - If missing (or outdated when auto_update=True), attempts multi-mirror download & atomic decompression.
    """
    global _BOOTSTRAP_ATTEMPTED

    dist_cfg = get_pedlex_distribution_config()
    expected_sha = dist_cfg.get("expected_sha256", "")
    auto_update = bool(dist_cfg.get("auto_update", False))

    # 0ms fast path when file already exists on disk
    if target_path.exists() and target_path.stat().st_size > 1_000_000:
        if not force_check and (not auto_update or not expected_sha):
            return True
        if expected_sha:
            cached_sha = read_cached_sha256(target_path)
            if cached_sha is None:
                cached_sha = compute_file_sha256(target_path)
            if cached_sha == expected_sha:
                return True

    if not dist_cfg.get("auto_download", True) and not force_check:
        return target_path.exists()

    urls = dist_cfg.get("urls") or []
    if not urls:
        return target_path.exists()

    with _BOOTSTRAP_LOCK:
        # Re-check inside lock
        if target_path.exists() and target_path.stat().st_size > 1_000_000:
            if not force_check and (not auto_update or not expected_sha):
                return True
            if expected_sha and read_cached_sha256(target_path) == expected_sha:
                return True

        if _BOOTSTRAP_ATTEMPTED and not force_check:
            return target_path.exists()
        _BOOTSTRAP_ATTEMPTED = True

        ok, _ = download_and_extract_pedlex(
            urls=urls,
            target_path=target_path,
            expected_sha256=expected_sha,
            auth_token=dist_cfg.get("auth_token", ""),
            timeout_seconds=int(dist_cfg.get("timeout_seconds", 10)),
        )
        return ok or target_path.exists()


def pack_pedlex_archive(
    db_path: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    preset: int = 6,
    vacuum_source: bool = True,
    update_config: bool = True,
) -> Dict[str, Any]:
    """
    Compact pedlex.db with SQLite VACUUM + ANALYZE, compress with LZMA (.xz),
    write SHA-256 sidecar and manifest, and update wiki_config.json.
    """
    if db_path is None:
        db_path = Path(__file__).resolve().parent.parent / "data" / "pedlex.db"
    db_path = Path(db_path).resolve()
    if not db_path.exists():
        raise FileNotFoundError(f"Source database not found: {db_path}")

    project_root = db_path.parent.parent.parent
    if out_dir is None:
        out_dir = project_root / "dist"
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    orig_size = db_path.stat().st_size
    print(f"[1/4] Preparing {db_path.name} ({orig_size / (1024 * 1024):.2f} MB)...")

    work_db = out_dir / "pedlex.db.work"
    if work_db.exists():
        work_db.unlink()
    shutil.copy2(db_path, work_db)

    print("[2/4] Running SQLite VACUUM & ANALYZE to strip free-list fragmentation...")
    t0 = time.time()
    conn = sqlite3.connect(str(work_db))
    try:
        conn.execute("VACUUM;")
        conn.execute("ANALYZE;")
        conn.commit()
    finally:
        conn.close()

    vacuumed_size = work_db.stat().st_size
    saved_mb = (orig_size - vacuumed_size) / (1024 * 1024)
    print(
        f"      Vacuumed size: {vacuumed_size / (1024 * 1024):.2f} MB "
        f"(reclaimed {saved_mb:.2f} MB in {time.time() - t0:.1f}s)"
    )

    # Replace source db with vacuumed copy so local pedlex.db matches expected_sha256 identically
    if vacuum_source:
        try:
            from ..linguistics import LinguisticEngine
            if getattr(LinguisticEngine, "_ldoce_conn", None) is not None:
                LinguisticEngine._ldoce_conn.close()
                LinguisticEngine._ldoce_conn = None
        except Exception:
            pass
        try:
            os.replace(work_db, db_path)
            source_for_pack = db_path
        except PermissionError:
            try:
                shutil.copyfile(work_db, db_path)
                work_db.unlink(missing_ok=True)
                source_for_pack = db_path
            except PermissionError:
                source_for_pack = work_db
    else:
        source_for_pack = work_db

    print("[3/4] Computing uncompressed SHA-256 & compressing with LZMA (.xz)...")
    t1 = time.time()
    archive_path = out_dir / "pedlex.db.xz"
    archive_tmp = out_dir / "pedlex.db.xz.tmp"
    if archive_tmp.exists():
        archive_tmp.unlink()

    uncompressed_hasher = hashlib.sha256()
    bytes_processed = 0

    with open(source_for_pack, "rb") as src_f, lzma.open(archive_tmp, mode="wb", preset=preset) as xz_f:
        while True:
            chunk = src_f.read(4 * 1024 * 1024)
            if not chunk:
                break
            uncompressed_hasher.update(chunk)
            xz_f.write(chunk)
            bytes_processed += len(chunk)
            pct = (bytes_processed / vacuumed_size) * 100.0
            sys.stdout.write(
                f"\r      Compressing: {bytes_processed / (1024 * 1024):.1f}/{vacuumed_size / (1024 * 1024):.1f} MB ({pct:.0f}%)"
            )
            sys.stdout.flush()

    sys.stdout.write("\n")
    os.replace(archive_tmp, archive_path)
    if work_db.exists() and source_for_pack != work_db:
        work_db.unlink()

    uncompressed_sha256 = uncompressed_hasher.hexdigest().lower()
    if source_for_pack == db_path:
        write_cached_sha256(db_path, uncompressed_sha256)

    compressed_size = archive_path.stat().st_size
    compressed_sha256 = compute_file_sha256(archive_path)
    ratio = vacuumed_size / max(1, compressed_size)
    elapsed_pack = time.time() - t1

    print(
        f"      Archive ready: {archive_path} "
        f"({compressed_size / (1024 * 1024):.2f} MB, {ratio:.2f}x compression in {elapsed_pack:.1f}s)"
    )

    manifest = {
        "filename": "pedlex.db.xz",
        "uncompressed_filename": "pedlex.db",
        "uncompressed_bytes": vacuumed_size,
        "uncompressed_mb": round(vacuumed_size / (1024 * 1024), 2),
        "uncompressed_sha256": uncompressed_sha256,
        "compressed_bytes": compressed_size,
        "compressed_mb": round(compressed_size / (1024 * 1024), 2),
        "compressed_sha256": compressed_sha256,
        "compression_ratio": round(ratio, 2),
        "lzma_preset": preset,
    }
    manifest_path = out_dir / "pedlex.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if update_config:
        config_path = project_root / "wiki_config.json"
        if config_path.exists():
            try:
                cfg_data = json.loads(config_path.read_text(encoding="utf-8"))
                dist_block = cfg_data.get("pedlex_distribution")
                if not isinstance(dist_block, dict):
                    dist_block = {
                        "auto_download": True,
                        "auto_update": False,
                        "expected_sha256": uncompressed_sha256,
                        "urls": [
                            "https://r2.airopen.org/pedlex.db.xz",
                            "https://huggingface.co/datasets/huanan88/lexis-asset/resolve/main/pedlex.db.xz",
                            "https://hf-mirror.com/datasets/huanan88/lexis-asset/resolve/main/pedlex.db.xz",
                        ],
                        "auth_token": "",
                    }
                else:
                    dist_block["expected_sha256"] = uncompressed_sha256
                cfg_data["pedlex_distribution"] = dist_block
                config_path.write_text(json.dumps(cfg_data, indent=4, ensure_ascii=False), encoding="utf-8")
                print(f"[4/4] Updated expected_sha256 in {config_path.name}: {uncompressed_sha256}")
            except Exception as exc:
                print(f"[4/4] Warning: Could not update {config_path}: {exc}")

    return manifest
