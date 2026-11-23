import http.server
import socketserver
import json
import os
import shutil
import threading
import traceback
import mimetypes
import urllib.parse
import re
import uuid
from pathlib import Path
from .config import config, normalize_name, extract_unit_stem, is_supplement_unit
from .processor import WikiProcessor
from .linter import linter
from .llm import llm
from .dashboard_routes import DashboardRoutesMixin
def get_raw_file_stem(filepath: Path) -> str:
    return filepath.stem


_ORDINAL_EDITION_MAP = {
    "first": "1e", "1st": "1e", "1": "1e", "one": "1e",
    "second": "2e", "2nd": "2e", "2": "2e", "two": "2e",
    "third": "3e", "3rd": "3e", "3": "3e", "three": "3e",
    "fourth": "4e", "4th": "4e", "4": "4e", "four": "4e",
    "fifth": "5e", "5th": "5e", "5": "5e", "five": "5e",
    "sixth": "6e", "6th": "6e", "6": "6e", "six": "6e",
    "seventh": "7e", "7th": "7e", "7": "7e", "seven": "7e",
    "eighth": "8e", "8th": "8e", "8": "8e", "eight": "8e",
    "ninth": "9e", "9th": "9e", "9": "9e", "nine": "9e",
    "tenth": "10e", "10th": "10e", "10": "10e", "ten": "10e",
    "revised": "Rev", "rev": "Rev", "updated": "Upd",
}


def extract_series_edition(raw_series: str | None, raw_edition: str | None = None, raw_abbr: str | None = None) -> tuple[str | None, str | None, str | None, str | None]:
    """
    Cleans series title, extracts normalized edition tag (e.g. '4e', '2e', 'Rev'),
    and derives a friendly short name and combined display tag.
    Returns: (clean_series, edition_tag, series_abbr, series_display)
    """
    if not raw_series:
        return None, None, None, None

    clean_series = raw_series.strip()
    edition_tag = None

    if raw_edition is not None:
        ed_str = str(raw_edition).strip().lower()
        m = re.search(r"\b(\d+|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|1st|2nd|3rd|4th|5th|6th|7th|8th|9th|10th|revised|rev)\b", ed_str)
        if m:
            edition_tag = _ORDINAL_EDITION_MAP.get(m.group(1), f"{m.group(1)}e")
        else:
            edition_tag = ed_str

    # Pattern 1: Parens containing Edition, e.g. "(Fourth Edition)", "(4th Ed.)", "(Revised Edition)"
    m_paren = re.search(r"\s*\(\s*(?:the\s+)?([A-Za-z0-9]+)\s+(?:edition|ed\.?|version)\s*\)", clean_series, re.IGNORECASE)
    if m_paren:
        if not edition_tag:
            key = m_paren.group(1).lower()
            edition_tag = _ORDINAL_EDITION_MAP.get(key, f"{key}e")
        clean_series = re.sub(r"\s*\(\s*(?:the\s+)?([A-Za-z0-9]+)\s+(?:edition|ed\.?|version)\s*\)", "", clean_series, flags=re.IGNORECASE).strip()

    # Pattern 2: Trailing unparenthesized e.g. "4th Edition", "4th Ed.", "Fourth Edition"
    m_trail = re.search(r"\b(?:the\s+)?([A-Za-z0-9]+)\s+(?:edition|ed\.?)\b\.?", clean_series, re.IGNORECASE)
    if m_trail:
        if not edition_tag:
            key = m_trail.group(1).lower()
            edition_tag = _ORDINAL_EDITION_MAP.get(key, f"{key}e")
        clean_series = re.sub(r"\b(?:the\s+)?([A-Za-z0-9]+)\s+(?:edition|ed\.?)\b\.?", "", clean_series, flags=re.IGNORECASE).strip()

    # Pattern 3: e.g. "(Rev.)" or "(Revised)"
    m_rev = re.search(r"\s*\(\s*(revised|rev\.?|updated)\s*\)", clean_series, re.IGNORECASE)
    if m_rev:
        if not edition_tag:
            edition_tag = _ORDINAL_EDITION_MAP.get(m_rev.group(1).lower().rstrip("."), "Rev")
        clean_series = re.sub(r"\s*\(\s*(revised|rev\.?|updated)\s*\)", "", clean_series, flags=re.IGNORECASE).strip()

    clean_series = re.sub(r"[,\.\-\s]+$", "", clean_series)
    clean_series = re.sub(r"\s+", " ", clean_series).strip()

    # Derive short abbreviation name
    series_abbr = raw_abbr.strip() if raw_abbr else None
    if not series_abbr:
        if "New Horizon College English" in clean_series:
            series_abbr = "New Horizon"
        elif "New Practical English" in clean_series:
            series_abbr = "Practical English"
        elif "Contemporary College English" in clean_series:
            series_abbr = "Contemporary"
        elif "An Integrated Skills Course" in clean_series:
            series_abbr = "Integrated Skills"
        elif len(clean_series) <= 16:
            series_abbr = clean_series
        else:
            series_abbr = clean_series[:14] + "…"

    # Compose series_display
    if edition_tag:
        series_display = f"{series_abbr} · {edition_tag}"
    else:
        series_display = series_abbr

    return clean_series, edition_tag, series_abbr, series_display


def inject_or_lock_frontmatter(text: str, target_series: str = None, target_book: str = None) -> str:
    """
    Smart Frontmatter Infill & Lock:
    - If frontmatter exists, locks/preserves any existing `series` or `book` fields.
      Only appends missing `series` or `book` fields if provided by upload context.
    - If frontmatter does not exist, synthesizes and prepends a clean YAML frontmatter
      block if target_series or target_book is specified.
    """
    if not target_series and not target_book:
        return text

    # Check for existing YAML frontmatter: ^---\s*\r?\n(.*?)\r?\n---\s*\r?\n
    fm_pattern = re.compile(r'^---\s*\r?\n(.*?)\r?\n---\s*\r?\n', re.DOTALL)
    m = fm_pattern.match(text)

    if m:
        fm_block = m.group(1)
        injected = []
        if target_series and not re.search(r'^\s*series:\s*', fm_block, re.MULTILINE):
            injected.append(f'series: "{target_series}"')
        if target_book and not re.search(r'^\s*book:\s*', fm_block, re.MULTILINE):
            try:
                b_num = int(target_book)
                injected.append(f'book: {b_num}')
            except ValueError:
                injected.append(f'book: "{target_book}"')
            if not re.search(r'^\s*book_name:\s*', fm_block, re.MULTILINE):
                injected.append(f'book_name: "Book {target_book}"')

        if injected:
            new_fm = fm_block.rstrip() + "\n" + "\n".join(injected) + "\n"
            return f"---\n{new_fm}---\n" + text[m.end():]
        return text
    else:
        # No frontmatter, synthesize new block
        injected = []
        if target_series:
            injected.append(f'series: "{target_series}"')
        if target_book:
            try:
                b_num = int(target_book)
                injected.append(f'book: {b_num}')
            except ValueError:
                injected.append(f'book: "{target_book}"')
            injected.append(f'book_name: "Book {target_book}"')

        if injected:
            header = "---\n" + "\n".join(injected) + "\n---\n\n"
            return header + text.lstrip()
        return text


# Global state to keep track of active background jobs and logging
class DashboardState:
    jobs = {}
    lock = threading.Lock()

    # --- Server lifecycle (auto-shutdown when the browser is closed) ---
    last_activity_ts = None        # timestamp of the last request received
    shutdown_requested_at = None   # set by /api/shutdown (browser beforeunload beacon);
                                   # any later request (touch()) cancels a pending shutdown

    @classmethod
    def touch(cls):
        """Record that the browser client is still alive (call on every request).
        Also cancels a pending shutdown: if a request arrives after the beacon,
        the client is coming back (e.g. an F5 refresh) and must not be killed."""
        import time
        with cls.lock:
            cls.last_activity_ts = time.time()
            cls.shutdown_requested_at = None

    @classmethod
    def mark_shutdown(cls):
        """Arm the shutdown (browser beforeunload beacon). The supervisor exits
        only after SHUTDOWN_GRACE elapses without any new request. The beacon
        request itself does NOT count as activity (it is the last words of a
        leaving page), so it resets last_activity_ts."""
        import time
        with cls.lock:
            cls.shutdown_requested_at = time.time()
            cls.last_activity_ts = None

    @classmethod
    def lifecycle(cls):
        """Return (shutdown_requested_at, last_activity_ts, has_running_jobs) under one lock."""
        with cls.lock:
            has_running = any(j.get("status") == "running" for j in cls.jobs.values())
            return cls.shutdown_requested_at, cls.last_activity_ts, has_running

    @classmethod
    def create_job(cls, job_type, target):
        with cls.lock:
            job_id = f"{job_type}_{uuid.uuid4().hex[:8]}"
            initial_logs = [f"🔨 Preparing build for '{target}'..."] if job_type == "compile" else [f"[{job_type.upper()}] Job created for '{target}'", f"[{job_type.upper()}] Launching background worker thread..."]
            cls.jobs[job_id] = {
                "id": job_id,
                "type": job_type,
                "target": target,
                "status": "running",
                "progress": 10,
                "stage": "starting",
                "audit_progress": 0,
                "audit_info": None,
                "logs": initial_logs,
                "saved_files": [],
                "result": None
            }
            return job_id

    @classmethod
    def update_job(cls, job_id, progress=None, status=None, log=None, saved_files=None, result=None, stage=None, audit_progress=None, audit_info=None):
        with cls.lock:
            if job_id in cls.jobs:
                if progress is not None:
                    cls.jobs[job_id]["progress"] = progress
                if status is not None:
                    cls.jobs[job_id]["status"] = status
                if log is not None:
                    cls.jobs[job_id]["logs"].append(log)
                if saved_files is not None:
                    cls.jobs[job_id]["saved_files"] = saved_files
                if result is not None:
                    cls.jobs[job_id]["result"] = result
                if stage is not None:
                    cls.jobs[job_id]["stage"] = stage
                if audit_progress is not None:
                    cls.jobs[job_id]["audit_progress"] = audit_progress
                if audit_info is not None:
                    cls.jobs[job_id]["audit_info"] = audit_info

    @classmethod
    def get_job(cls, job_id):
        with cls.lock:
            return cls.jobs.get(job_id)

    @classmethod
    def get_recent_jobs(cls):
        with cls.lock:
            running = [j for j in cls.jobs.values() if j.get("status") == "running"]
            finished = [j for j in cls.jobs.values() if j.get("status") != "running"]
            return finished[-10:] + running

def background_compile_worker(job_id, filename, categories=None):
    try:
        DashboardState.update_job(job_id, progress=25, log=f"⚙️ Reading source file: {filename}")
        
        processor = WikiProcessor()
        cats_str = f" ({', '.join(categories)})" if categories else " (All)"
        DashboardState.update_job(job_id, progress=50, log=f"🔍 Extracting information{cats_str}...")
        
        status, saved_files = processor.run_pipeline(filename, categories=categories)
        
        if "Error" in status or "failed tasks:" in status or "Pipeline failed" in status:
            DashboardState.update_job(job_id, progress=100, status="failed", log=f"❌ {status}", saved_files=saved_files)
            for path in saved_files:
                DashboardState.update_job(job_id, log=f"⚠️ Partial node saved before failure: {os.path.basename(path)}")
        else:
            DashboardState.update_job(job_id, progress=100, status="completed", log=f"✨ Finished compiling{cats_str}!", saved_files=saved_files)
            for path in saved_files:
                DashboardState.update_job(job_id, log=f"✅ Saved wiki node: {os.path.basename(path)}")
    except Exception as e:
        err_msg = traceback.format_exc()
        DashboardState.update_job(job_id, progress=100, status="failed", log=f"❌ Exception occurred: {err_msg}")

def background_quiz_worker(job_id, filename, count, template, listening_archetype=None, listening_scenario=None):
    try:
        spec_parts = []
        if listening_archetype:
            spec_parts.append(f"Archetype: {listening_archetype}")
        if listening_scenario:
            spec_parts.append(f"Scenario: {listening_scenario}")
        arch_info = f" [{', '.join(spec_parts)}]" if spec_parts else ""
        DashboardState.update_job(job_id, progress=20, stage="generating", log=f"[QUIZ] Triggering generation from unit '{filename}'{arch_info}...")
        DashboardState.update_job(job_id, progress=40, stage="generating", log=f"[QUIZ] Loading verified extraction nodes for '{filename}'...")
        DashboardState.update_job(job_id, progress=60, stage="generating", log=f"[QUIZ] Prompting LLM with template '{template}' (Target count: {count}){arch_info}...")
        
        processor = WikiProcessor()

        def on_audit_update(stage, audit_progress, message, extra=None):
            # Keep overall quiz progress at 100% since draft is ready, while audit_progress tracks the L2 audit cycle
            DashboardState.update_job(
                job_id,
                progress=100,
                stage=stage,
                audit_progress=audit_progress,
                audit_info=message,
                log=f"[AUDIT] {message}"
            )

        result = processor.generate_quiz(
            filename,
            count=count,
            template_name=template,
            audit_callback=on_audit_update,
            listening_archetype=listening_archetype,
            listening_scenario=listening_scenario,
        )
        
        if "Error" in result:
            DashboardState.update_job(job_id, progress=100, status="failed", stage="failed", log=f"[ERROR] Quiz generation failed: {result}")
        else:
            try:
                rel_path = Path(result).relative_to(Path(config.project_root))
                web_path = str(rel_path).replace("\\", "/")
            except ValueError:
                web_path = result
            DashboardState.update_job(
                job_id,
                progress=100,
                status="completed",
                stage="completed",
                audit_progress=100,
                log="[QUIZ] Vocabulary & Grammar nodes synced and Quiz generation completed successfully!",
                saved_files=[web_path]
            )
            DashboardState.update_job(job_id, log=f"[QUIZ] Handout saved to: {os.path.basename(result)}")
    except Exception as e:
        err_msg = traceback.format_exc()
        DashboardState.update_job(job_id, progress=100, status="failed", stage="failed", log=f"[ERROR] Exception occurred: {err_msg}")

def background_all_quizzes_worker(job_id, filename):
    try:
        DashboardState.update_job(job_id, progress=10, stage="generating", log=f"[ALL-QUIZZES] Starting batch quiz generation for '{filename}'...")
        processor = WikiProcessor()

        # Resolve canonical unit name and folder
        unit_stem = Path(filename).stem
        wiki_dir = Path(config.project_root) / "wiki"
        unit_dir = wiki_dir / unit_stem
        if not unit_dir.exists() and wiki_dir.exists():
            norm_target = normalize_name(unit_stem)
            for child in wiki_dir.iterdir():
                if child.is_dir() and normalize_name(child.name) == norm_target:
                    unit_dir = child
                    unit_stem = child.name
                    break
        unit_name = unit_stem

        # Prerequisite check: verified vocabulary extraction must exist
        vocab_path = unit_dir / "extractions" / f"{unit_name}_vocabulary.md"
        if not vocab_path.exists():
            DashboardState.update_job(job_id, progress=15, stage="generating", log=f"[ALL-QUIZZES] Prerequisite missing: compiling extractions for '{unit_name}' first...")
            status, saved = processor.run_pipeline(filename)
            if "Error" in status or not vocab_path.exists():
                DashboardState.update_job(job_id, progress=100, status="failed", stage="failed", log=f"[ERROR] Prerequisite extraction compilation failed: {status}")
                return

        # Check for media/video source
        media_dir = unit_dir / "sources" / "media"
        has_video = media_dir.exists() and any(media_dir.glob("*.md"))

        templates = ["vocabulary", "cloze", "reading", "translation", "listening"]
        if has_video:
            templates.append("video")

        saved_handouts = []
        total = len(templates)

        for idx, t in enumerate(templates):
            current_pct = int(20 + (idx / total) * 75)
            target_count = config.get_quiz_config(t, "count", 10 if t in ("vocabulary", "cloze") else 5)
            DashboardState.update_job(
                job_id,
                progress=current_pct,
                stage="generating",
                log=f"[{idx+1}/{total}] Generating '{t}' quiz (count: {target_count}) for '{unit_name}'..."
            )
            result = processor.generate_quiz(unit_name, count=target_count, template_name=t)
            if "Error" in result:
                DashboardState.update_job(job_id, log=f"[WARN] Quiz '{t}' generation encountered error: {result}")
            else:
                try:
                    rel_path = str(Path(result).relative_to(Path(config.project_root))).replace("\\", "/")
                except ValueError:
                    rel_path = result
                saved_handouts.append(rel_path)
                DashboardState.update_job(job_id, log=f"✅ Saved handout: {os.path.basename(result)}")

        if saved_handouts:
            DashboardState.update_job(
                job_id,
                progress=100,
                status="completed",
                stage="completed",
                audit_progress=100,
                log=f"[ALL-QUIZZES] Successfully generated {len(saved_handouts)}/{total} quizzes for '{unit_name}'!",
                saved_files=saved_handouts
            )
        else:
            DashboardState.update_job(
                job_id,
                progress=100,
                status="failed",
                stage="failed",
                log="[ALL-QUIZZES] Failed to generate any quizzes."
            )
    except Exception as e:
        err_msg = traceback.format_exc()
        DashboardState.update_job(job_id, progress=100, status="failed", stage="failed", log=f"[ERROR] Exception occurred: {err_msg}")

def background_video_import_worker(job_id, url_or_path, cookies_from_browser=None, cookies=None, subtitle=None, current_unit=None):
    try:
        DashboardState.update_job(job_id, progress=20, log=f"[VIDEO-IMPORT] Detecting source for '{url_or_path}'...")
        from .video import video_service
        
        DashboardState.update_job(job_id, progress=40, log="[VIDEO-IMPORT] Processing video transcript or running local Whisper model...")
        saved_file, has_subtitles = video_service.import_video(
            url_or_path,
            cookies_from_browser=cookies_from_browser,
            cookies=cookies,
            subtitle=subtitle,
            current_unit=current_unit
        )
        
        try:
            filename = str(Path(saved_file).relative_to(Path(config.project_root))).replace("\\", "/")
        except ValueError:
            filename = os.path.basename(saved_file)
            
        if has_subtitles:
            if current_unit:
                media_name = os.path.basename(saved_file)
                DashboardState.update_job(job_id, log=f"[VIDEO-IMPORT] Successfully associated media '{media_name}' with unit '{current_unit}'")

            DashboardState.update_job(job_id, progress=100, status="completed", log=f"[VIDEO-IMPORT] Success! Saved video transcript to {filename}", saved_files=[filename])
        else:
            DashboardState.update_job(job_id, progress=100, status="completed", log=f"[WARNING] Video metadata imported successfully, but no subtitles/transcripts could be retrieved. Placeholder created at {filename}", saved_files=[filename])
    except Exception as e:
        err_msg = traceback.format_exc()
        DashboardState.update_job(job_id, progress=100, status="failed", log=f"[ERROR] Import failed: {err_msg}")

def background_audit_batch_worker(job_id, units=None, template=None, workers=2):
    """Re-run the Level-2 expert audit over existing handouts, streaming progress to the job."""
    try:
        from .audit_batch import run_batch
        scope = f"{len(units)} unit(s)" if units else "ALL units"
        DashboardState.update_job(job_id, log=f"🔎 Starting Level-2 re-audit over {scope}…")

        def _progress(done, total, row):
            mark = "⚠" if row.get("error") else ("✅" if row.get("passed") else "❌")
            acc = row.get("blind_solve_accuracy")
            acc_s = f"{acc * 100:.0f}%" if isinstance(acc, (int, float)) else "n/a"
            DashboardState.update_job(
                job_id,
                progress=min(99, int(done / total * 100)) if total else 50,
                log=f"[{done}/{total}] {mark} {row['unit']} [{row['template']}] blind {acc_s}",
            )

        summary = run_batch(
            unit_filter=units,
            template_filter=template,
            max_workers=workers,
            progress_cb=_progress,
        )

        if summary["total"] == 0:
            DashboardState.update_job(job_id, progress=100, status="failed",
                                      log="❌ No quiz handouts found to re-audit.")
            return

        # Persist a timestamped report for traceability.
        report_name = None
        try:
            import datetime
            logs_dir = Path(config.project_root) / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            report_name = f"re_audit_report_{stamp}.json"
            with open(logs_dir / report_name, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
        except (OSError, IOError):
            report_name = None

        if summary["failed"] == 0 and summary["errors"] == 0:
            done_log = f"✅ Re-audit complete: {summary['passed']}/{summary['total']} passed, {summary['audited']} audited."
        else:
            done_log = f"🏁 Re-audit complete: {summary['passed']} passed / {summary['failed']} failed / {summary['errors']} error(s)."
        DashboardState.update_job(job_id, progress=100, status="completed", log=done_log, result=summary)
        if report_name:
            DashboardState.update_job(job_id, log=f"📄 Report saved to logs/{report_name}")
    except Exception as e:
        err_msg = traceback.format_exc()
        DashboardState.update_job(job_id, progress=100, status="failed", log=f"❌ Re-audit exception: {err_msg}")

class DashboardHTTPRequestHandler(DashboardRoutesMixin, http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Silence standard HTTP logger to keep terminal clean
        pass

    def handle_one_request(self):
        try:
            super().handle_one_request()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            # Normal when browser cancels in-flight media Range requests during seeking/single-sentence playback
            self.close_connection = True

    def do_GET(self):
        # Keep the server alive only while the browser client is active
        DashboardState.touch()

        url = urllib.parse.urlparse(self.path)
        path = url.path

        # Handle API routes
        if path.startswith("/api/"):
            if path == "/api/export-zip":
                self.serve_export_zip(url.query)
                return
            if path.startswith("/api/media/"):
                self.serve_wiki_file(path)
                return
            self.handle_api_get(path, url.query)
            return

        # Serve static wiki/quiz files if requested
        if path.startswith("/wiki/") or path.startswith("/logs/"):
            self.serve_wiki_file(path)
            return

        # Serve static assets (CSS, JS, fonts)
        if path.startswith("/static/"):
            self.serve_static_asset(path)
            return

        # Serve the Dashboard SPA (or index page)
        if path == "/" or path == "/index.html" or path == "/dashboard":
            self.serve_dashboard()
            return

        # Fallback 404
        self.send_error(404, "File Not Found")

    def do_POST(self):
        # Keep the server alive only while the browser client is active
        DashboardState.touch()

        url = urllib.parse.urlparse(self.path)
        path = url.path

        if path.startswith("/api/"):
            self.handle_api_post(path)
            return

        self.send_error(404, "Not Found")

    def serve_export_zip(self, query_str):
        query = urllib.parse.parse_qs(query_str or "")
        unit_name = query.get("unit_name", [None])[0]
        target_dir = query.get("target_dir", [None])[0]
        if not unit_name:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"success": False, "error": "unit_name query parameter is required"}).encode("utf-8"))
            return
        try:
            from .exporter import export_unit_to_zip
            res = export_unit_to_zip(unit_name, custom_target_dir=target_dir)
            if not res.get("success"):
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"success": False, "error": res.get("error", "ZIP export failed")}).encode("utf-8"))
                return
            zip_path = Path(res["zip_path"])
            if not zip_path.exists():
                self.send_response(404)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"success": False, "error": "Exported ZIP file not found"}).encode("utf-8"))
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/zip")
            self.send_header("Content-Disposition", f'attachment; filename="{zip_path.name}"')
            self.send_header("Content-Length", str(zip_path.stat().st_size))
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.end_headers()
            with open(zip_path, "rb") as f:
                shutil.copyfileobj(f, self.wfile)
        except Exception as e:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"success": False, "error": f"Error generating ZIP: {e}"}).encode("utf-8"))

    def serve_wiki_file(self, path):
        # Resolve path relative to project root
        rel_path = urllib.parse.unquote(path.lstrip("/"))
        file_path = Path(config.project_root) / rel_path

        # Handle explicit /api/media/ lookup across unit folders
        if rel_path.startswith("api/media/"):
            filename = Path(file_path).name
            wiki_dir = Path(config.project_root) / "wiki"
            found = False
            if wiki_dir.exists():
                for child in wiki_dir.iterdir():
                    if child.is_dir():
                        candidate = child / "sources" / "media" / filename
                        if candidate.exists() and candidate.is_file():
                            file_path = candidate
                            found = True
                            break
            if not found:
                self.send_error(404, "Media File Not Found")
                return


        # Security check to stay strictly inside allowed directories (wiki or logs)
        try:
            resolved_file = file_path.resolve()
            wiki_root = (Path(config.project_root) / "wiki").resolve()
            logs_root = (Path(config.project_root) / "logs").resolve()
            if rel_path.startswith("logs/"):
                allowed = resolved_file.is_relative_to(logs_root)
            else:
                allowed = resolved_file.is_relative_to(wiki_root)
            if not allowed:
                self.send_error(403, "Access Denied")
                return
        except (OSError, ValueError):
            self.send_error(404, "File Not Found")
            return

        if resolved_file.exists() and resolved_file.is_file():
            mime_type, _ = mimetypes.guess_type(str(resolved_file))
            if not mime_type:
                mime_type = "text/plain" if resolved_file.suffix == ".md" else "application/octet-stream"

            # Always serve markdown as text/plain or text/markdown
            if resolved_file.suffix == ".md":
                mime_type = "text/markdown; charset=utf-8"
            elif resolved_file.suffix == ".html":
                mime_type = "text/html; charset=utf-8"

            # Handle HTTP Range requests for video/audio seeking/scrubbing
            size = resolved_file.stat().st_size
            range_header = self.headers.get("Range")
            if range_header and range_header.startswith("bytes="):
                try:
                    ranges = range_header.replace("bytes=", "").split(",")[0].split("-")
                    start = int(ranges[0]) if ranges[0] else 0
                    end = int(ranges[1]) if (len(ranges) > 1 and ranges[1]) else size - 1
                    if start >= size or start < 0:
                        self.send_error(416, "Requested range not satisfiable")
                        return
                    if end >= size:
                        end = size - 1
                    length = end - start + 1
                except ValueError:
                    self.send_error(400, "Invalid Range header")
                    return

                try:
                    self.send_response(206)
                    self.send_header("Content-Type", mime_type)
                    self.send_header("Accept-Ranges", "bytes")
                    self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                    self.send_header("Content-Length", str(length))
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()

                    with open(resolved_file, "rb") as f:
                        f.seek(start)
                        remaining = length
                        buffer_size = 64 * 1024
                        while remaining > 0:
                            chunk_size = min(buffer_size, remaining)
                            data = f.read(chunk_size)
                            if not data:
                                break
                            self.wfile.write(data)
                            remaining -= len(data)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    self.close_connection = True
                return

            try:
                self.send_response(200)
                self.send_header("Content-Type", mime_type)
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Length", str(size))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()

                with open(resolved_file, "rb") as f:
                    buffer_size = 64 * 1024
                    while True:
                        data = f.read(buffer_size)
                        if not data:
                            break
                        self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                self.close_connection = True
        else:
            self.send_error(404, "File Not Found")

    def serve_dashboard(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self.end_headers()

        template_path = Path(__file__).parent / "templates" / "dashboard.html"

        if template_path.exists():
            html_content = template_path.read_text(encoding="utf-8")
        else:
            html_content = "<h1>Error: templates/dashboard.html not found!</h1>"

        self.wfile.write(html_content.encode("utf-8"))

    def serve_static_asset(self, path):
        # Resolve path relative to librarian/static/ directory
        rel_path = urllib.parse.unquote(path).lstrip("/")
        # Remove 'static/' prefix to locate inside librarian/static
        if rel_path.startswith("static/"):
            rel_path = rel_path[len("static/"):]

        static_root = (Path(__file__).parent / "static").resolve()
        file_path = (static_root / rel_path).resolve()

        # Security check: ensure file stays inside librarian/static
        if not file_path.is_relative_to(static_root):
            self.send_error(403, "Access Denied")
            return

        if file_path.exists() and file_path.is_file():
            mime_type, _ = mimetypes.guess_type(str(file_path))
            if not mime_type:
                if file_path.suffix == ".css":
                    mime_type = "text/css; charset=utf-8"
                elif file_path.suffix == ".js":
                    mime_type = "application/javascript; charset=utf-8"
                else:
                    mime_type = "application/octet-stream"
            elif mime_type == "text/css":
                mime_type = "text/css; charset=utf-8"
            elif mime_type in ["application/javascript", "text/javascript"]:
                mime_type = "application/javascript; charset=utf-8"

            self.send_response(200)
            self.send_header("Content-Type", mime_type)
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            with open(file_path, "rb") as f:
                self.wfile.write(f.read())
        else:
            self.send_error(404, "Static Asset Not Found")