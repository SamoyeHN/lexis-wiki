"""Asynchronous background worker functions for compile, quiz, video, and audit jobs."""

from __future__ import annotations

import json
import os
import traceback
from pathlib import Path
from typing import List, Optional

from ..config import config, normalize_name
from ..processor import WikiProcessor
from .state import DashboardState


def background_compile_worker(
    job_id: str,
    filename: str,
    categories: Optional[List[str]] = None,
) -> None:
    try:
        DashboardState.update_job(job_id, progress=25, log=f"⚙️ Reading source file: {filename}")

        processor = WikiProcessor()
        cats_str = f" ({', '.join(categories)})" if categories else " (All)"
        DashboardState.update_job(job_id, progress=50, log=f"🔍 Extracting information{cats_str}...")

        status, saved_files = processor.run_pipeline(filename, categories=categories)

        if "Error" in status or "failed tasks:" in status or "Pipeline failed" in status:
            DashboardState.update_job(
                job_id,
                progress=100,
                status="failed",
                log=f"❌ {status}",
                saved_files=saved_files,
            )
            for path in saved_files:
                DashboardState.update_job(
                    job_id,
                    log=f"⚠️ Partial node saved before failure: {os.path.basename(path)}",
                )
        else:
            DashboardState.update_job(
                job_id,
                progress=100,
                status="completed",
                log=f"✨ Finished compiling{cats_str}!",
                saved_files=saved_files,
            )
            for path in saved_files:
                DashboardState.update_job(
                    job_id,
                    log=f"✅ Saved wiki node: {os.path.basename(path)}",
                )
    except Exception:
        err_msg = traceback.format_exc()
        DashboardState.update_job(
            job_id,
            progress=100,
            status="failed",
            log=f"❌ Exception occurred: {err_msg}",
        )


def background_quiz_worker(
    job_id: str,
    filename: str,
    count: int,
    template: str,
    listening_archetype: Optional[str] = None,
    listening_scenario: Optional[str] = None,
) -> None:
    try:
        spec_parts = []
        if listening_archetype:
            spec_parts.append(f"Archetype: {listening_archetype}")
        if listening_scenario:
            spec_parts.append(f"Scenario: {listening_scenario}")
        arch_info = f" [{', '.join(spec_parts)}]" if spec_parts else ""
        DashboardState.update_job(
            job_id,
            progress=20,
            stage="generating",
            log=f"[QUIZ] Triggering generation from unit '{filename}'{arch_info}...",
        )
        DashboardState.update_job(
            job_id,
            progress=40,
            stage="generating",
            log=f"[QUIZ] Loading verified extraction nodes for '{filename}'...",
        )
        DashboardState.update_job(
            job_id,
            progress=60,
            stage="generating",
            log=f"[QUIZ] Prompting LLM with template '{template}' (Target count: {count}){arch_info}...",
        )

        processor = WikiProcessor()

        def on_audit_update(stage: str, audit_progress: int, message: str, extra: Optional[dict] = None) -> None:
            # Keep overall quiz progress at 100% since draft is ready, while audit_progress tracks the L2 audit cycle
            DashboardState.update_job(
                job_id,
                progress=100,
                stage=stage,
                audit_progress=audit_progress,
                audit_info=message,
                log=f"[AUDIT] {message}",
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
            DashboardState.update_job(
                job_id,
                progress=100,
                status="failed",
                stage="failed",
                log=f"[ERROR] Quiz generation failed: {result}",
            )
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
                saved_files=[web_path],
            )
            DashboardState.update_job(job_id, log=f"[QUIZ] Handout saved to: {os.path.basename(result)}")
    except Exception:
        err_msg = traceback.format_exc()
        DashboardState.update_job(
            job_id,
            progress=100,
            status="failed",
            stage="failed",
            log=f"[ERROR] Exception occurred: {err_msg}",
        )


def background_all_quizzes_worker(job_id: str, filename: str) -> None:
    try:
        DashboardState.update_job(
            job_id,
            progress=10,
            stage="generating",
            log=f"[ALL-QUIZZES] Starting batch quiz generation for '{filename}'...",
        )
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
            DashboardState.update_job(
                job_id,
                progress=15,
                stage="generating",
                log=f"[ALL-QUIZZES] Prerequisite missing: compiling extractions for '{unit_name}' first...",
            )
            status, _saved = processor.run_pipeline(filename)
            if "Error" in status or not vocab_path.exists():
                DashboardState.update_job(
                    job_id,
                    progress=100,
                    status="failed",
                    stage="failed",
                    log=f"[ERROR] Prerequisite extraction compilation failed: {status}",
                )
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
                log=f"[{idx+1}/{total}] Generating '{t}' quiz (count: {target_count}) for '{unit_name}'...",
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
                saved_files=saved_handouts,
            )
        else:
            DashboardState.update_job(
                job_id,
                progress=100,
                status="failed",
                stage="failed",
                log="[ALL-QUIZZES] Failed to generate any quizzes.",
            )
    except Exception:
        err_msg = traceback.format_exc()
        DashboardState.update_job(
            job_id,
            progress=100,
            status="failed",
            stage="failed",
            log=f"[ERROR] Exception occurred: {err_msg}",
        )


def background_video_import_worker(
    job_id: str,
    url_or_path: str,
    cookies_from_browser: Optional[str] = None,
    cookies: Optional[str] = None,
    subtitle: Optional[str] = None,
    current_unit: Optional[str] = None,
) -> None:
    try:
        DashboardState.update_job(job_id, progress=20, log=f"[VIDEO-IMPORT] Detecting source for '{url_or_path}'...")
        from ..video import video_service

        DashboardState.update_job(job_id, progress=40, log="[VIDEO-IMPORT] Processing video transcript or running local Whisper model...")
        saved_file, has_subtitles = video_service.import_video(
            url_or_path,
            cookies_from_browser=cookies_from_browser,
            cookies=cookies,
            subtitle=subtitle,
            current_unit=current_unit,
        )

        try:
            filename = str(Path(saved_file).relative_to(Path(config.project_root))).replace("\\", "/")
        except ValueError:
            filename = os.path.basename(saved_file)

        if has_subtitles:
            if current_unit:
                media_name = os.path.basename(saved_file)
                DashboardState.update_job(
                    job_id,
                    log=f"[VIDEO-IMPORT] Successfully associated media '{media_name}' with unit '{current_unit}'",
                )

            DashboardState.update_job(
                job_id,
                progress=100,
                status="completed",
                log=f"[VIDEO-IMPORT] Success! Saved video transcript to {filename}",
                saved_files=[filename],
            )
        else:
            DashboardState.update_job(
                job_id,
                progress=100,
                status="completed",
                log=f"[WARNING] Video metadata imported successfully, but no subtitles/transcripts could be retrieved. Placeholder created at {filename}",
                saved_files=[filename],
            )
    except Exception:
        err_msg = traceback.format_exc()
        DashboardState.update_job(job_id, progress=100, status="failed", log=f"[ERROR] Import failed: {err_msg}")


def background_audit_batch_worker(
    job_id: str,
    units: Optional[List[str]] = None,
    template: Optional[str] = None,
    workers: int = 2,
) -> None:
    """Re-run the Level-2 expert audit over existing handouts, streaming progress to the job."""
    try:
        from ..audit_batch import run_batch

        scope = f"{len(units)} unit(s)" if units else "ALL units"
        DashboardState.update_job(job_id, log=f"🔎 Starting Level-2 re-audit over {scope}…")

        def _progress(done: int, total: int, row: dict) -> None:
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
            DashboardState.update_job(
                job_id,
                progress=100,
                status="failed",
                log="❌ No quiz handouts found to re-audit.",
            )
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
    except Exception:
        err_msg = traceback.format_exc()
        DashboardState.update_job(job_id, progress=100, status="failed", log=f"❌ Re-audit exception: {err_msg}")
