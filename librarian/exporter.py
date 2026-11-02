import os
import re
import shutil
import urllib.parse
import zipfile
from pathlib import Path
from .config import config


def detect_video_path_from_html(html_content: str, html_path: Path):
    """
    Accurately locates the local media file referenced by a video quiz HTML handout.
    Prioritizes:
      1. Exact `video_url` inside `const quizData = {...}`
      2. Stem match inside `<unit_dir>/sources/media/`
      3. Any media file inside `<unit_dir>/sources/media/`
    """
    html_path = Path(html_path)
    unit_dir = html_path.parent.parent
    media_dir = unit_dir / "sources" / "media"
    media_exts = (".mp4", ".webm", ".ogg", ".mp3", ".wav", ".mkv", ".mov", ".avi")

    # 1. Check explicit video_url in quizData JSON
    m_vurl = re.search(r'"video_url"\s*:\s*"([^"]+)"', html_content)
    if m_vurl:
        raw_url = urllib.parse.unquote(m_vurl.group(1).strip())
        if raw_url and not raw_url.startswith(("http://", "https://")):
            rel_candidate = Path(config.project_root) / raw_url.lstrip("/")
            if rel_candidate.exists() and rel_candidate.is_file():
                return rel_candidate
            base_name = os.path.basename(raw_url)
            if media_dir.exists():
                cand = media_dir / base_name
                if cand.exists() and cand.is_file():
                    return cand

    # 2. Match by handout filename stem (e.g. <VideoStem>_video_quiz.html -> <VideoStem>.mp4)
    clean_stem = re.sub(r"_(?:video_quiz|quiz|handout)$", "", html_path.stem, flags=re.IGNORECASE)
    if media_dir.exists() and media_dir.is_dir():
        for ext in media_exts:
            cand = media_dir / f"{clean_stem}{ext}"
            if cand.exists() and cand.is_file():
                return cand
        for f in sorted(media_dir.iterdir()):
            if f.is_file() and f.suffix.lower() in media_exts:
                if clean_stem.lower() in f.stem.lower() or f.stem.lower() in clean_stem.lower():
                    return f
        # 3. Fallback: first valid media file in unit's sources/media/
        for f in sorted(media_dir.iterdir()):
            if f.is_file() and f.suffix.lower() in media_exts:
                return f

    return None


def make_portable_html_content(html_content: str, media_filename: str = None) -> str:
    """
    Rewrites `video_url` in the exported HTML copy to a relative same-folder filename
    so opening the extracted `.html` directly via `file://` on Windows/macOS works immediately.
    """
    if not media_filename:
        return html_content
    escaped_name = media_filename.replace("\\", "\\\\").replace('"', '\\"')
    return re.sub(
        r'("video_url"\s*:\s*)"[^"]*"',
        lambda m: f'{m.group(1)}"{escaped_name}"',
        html_content,
        count=1,
    )


def export_portable_bundle(html_path, video_path=None, output_path=None):
    """
    Packages an interactive video HTML handout and its companion local video/audio file
    into a portable `.zip` archive (or copies them into a target folder if output_path is a directory).
    """
    path_str = str(html_path).replace("\\", "/")
    if "wiki/" in path_str:
        relative_part = path_str.split("wiki/", 1)[1]
        resolved_path = Path(config.project_root) / "wiki" / relative_part
        html_path = resolved_path if resolved_path.exists() else Path(html_path)
    else:
        if path_str.startswith("/"):
            path_str = path_str[1:]
        resolved_path = Path(config.project_root) / path_str
        html_path = resolved_path if resolved_path.exists() else Path(html_path)

    if not html_path.exists():
        print(f"Error: HTML handout not found at '{html_path}'")
        return False

    html_content = html_path.read_text(encoding="utf-8")

    # Resolve companion video file
    resolved_video = None
    if video_path:
        cand = Path(video_path)
        if cand.exists() and cand.is_file():
            resolved_video = cand
    if not resolved_video:
        resolved_video = detect_video_path_from_html(html_content, html_path)

    media_filename = resolved_video.name if resolved_video else None
    portable_html = make_portable_html_content(html_content, media_filename)

    html_stem = html_path.stem
    if output_path:
        out_p = Path(output_path)
        # If output_path is an existing directory or has no suffix, copy files directly into that folder
        if out_p.is_dir() or not out_p.suffix:
            out_p.mkdir(parents=True, exist_ok=True)
            dst_html = out_p / html_path.name
            dst_html.write_text(portable_html, encoding="utf-8")
            if resolved_video and resolved_video.exists():
                shutil.copy2(resolved_video, out_p / resolved_video.name)
            return dst_html
        target_zip_path = out_p if out_p.suffix.lower() == ".zip" else out_p.with_suffix(".zip")
        target_zip_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        target_output_dir = html_path.parent
        target_output_dir.mkdir(parents=True, exist_ok=True)
        target_zip_path = target_output_dir / f"{html_stem}_bundle.zip"

    # Write portable HTML + companion video into ZIP archive
    with zipfile.ZipFile(target_zip_path, "w") as zf:
        zf.writestr(
            zipfile.ZipInfo(html_path.name),
            portable_html.encode("utf-8"),
            compress_type=zipfile.ZIP_DEFLATED,
        )
        if resolved_video and resolved_video.exists():
            # Video files are already compressed; ZIP_STORED packages 50MB in <0.05s with zero CPU overhead
            zf.write(resolved_video, arcname=resolved_video.name, compress_type=zipfile.ZIP_STORED)

    return target_zip_path


# Backward-compatible alias used by dashboard and CLI
export_standalone_exe = export_portable_bundle
