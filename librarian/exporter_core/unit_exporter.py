"""
Full unit distribution packaging into standalone directories or ZIP bundles.
"""

import logging
import os
import platform
import re
import shutil
import subprocess
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

from ..config import config
from .bundle_packager import detect_video_path_from_html, make_portable_html_content
from .markdown_renderer import render_markdown_to_standalone_html
from .portal_renderer import render_index_portal_html

logger = logging.getLogger("librarian.exporter")


def normalize_export_target_dir(
    target_dir_str: Optional[Union[str, Path]], unit_name: str
) -> Path:
    """
    Resolves custom export directory with cross-platform and WSL Windows drive compatibility.
    Supports Windows paths (e.g. C:/Users/..., D:\\Exports) inside WSL.
    """
    raw = str(target_dir_str or "").strip()
    if not raw:
        cfg_export = str(config.data.get("export_dir", "") or "").strip()
        if cfg_export:
            raw = cfg_export

    clean_unit = re.sub(r"\.[^/.]+$", "", unit_name)
    default_dir = Path(config.project_root) / "wiki" / clean_unit / "exports"

    if not raw:
        return default_dir

    # Support Windows drive letter syntax in WSL (e.g., C:\..., C:/..., D:/...)
    if len(raw) >= 2 and raw[1] == ":":
        wslpath = shutil.which("wslpath")
        if wslpath:
            try:
                res = subprocess.run([wslpath, "-u", raw], capture_output=True, text=True, check=True)
                converted = res.stdout.strip()
                if converted:
                    raw = converted
            except (subprocess.SubprocessError, FileNotFoundError, OSError):
                pass
        if len(raw) >= 2 and raw[1] == ":":
            drive = raw[0].lower()
            rest = raw[2:].replace("\\", "/").lstrip("/")
            raw = f"/mnt/{drive}/{rest}"

    p = Path(raw)
    if not p.is_absolute():
        p = (Path(config.project_root) / p).resolve()

    # If the user specified a general export folder (e.g. C:/Exports), nest the unit folder inside it
    if p.name != clean_unit and not p.name.endswith("exports"):
        p = p / clean_unit

    return p.resolve()


def to_windows_display_path(p: Path) -> str:
    """
    Returns a Windows-friendly UNC or drive path if running inside WSL, or native path.
    """
    is_wsl = bool(
        os.environ.get("WSL_DISTRO_NAME")
        or (Path("/proc/version").exists() and "microsoft" in Path("/proc/version").read_text().lower())
    )
    if is_wsl:
        wslpath = shutil.which("wslpath")
        if wslpath:
            try:
                res = subprocess.run([wslpath, "-w", str(p)], capture_output=True, text=True, check=True)
                win_path = res.stdout.strip()
                if win_path:
                    return win_path
            except (subprocess.SubprocessError, FileNotFoundError, OSError):
                pass
        s = str(p)
        if s.startswith("/mnt/") and len(s) > 6 and s[6] == "/":
            drive = s[5].upper()
            rest = s[7:].replace("/", "\\")
            return f"{drive}:\\{rest}"
    return str(p)


def export_unit_to_directory(unit_name: str, custom_target_dir: Optional[Union[str, Path]] = None) -> dict:
    """
    Exports all verified extractions, companion quizzes, and an index portal
    for a specific unit into `wiki/<UnitName>/exports/` or a custom user-defined directory.
    """
    unit_name = str(unit_name).strip()
    clean_unit = re.sub(r"\.[^/.]+$", "", unit_name)
    unit_dir = Path(config.project_root) / "wiki" / clean_unit

    if not unit_dir.exists() or not unit_dir.is_dir():
        return {
            "success": False,
            "error": f"Unit directory not found: wiki/{clean_unit}",
        }

    export_dir = normalize_export_target_dir(custom_target_dir, clean_unit)
    export_extractions_dir = export_dir / "extractions"
    export_handouts_dir = export_dir / "handouts"

    export_extractions_dir.mkdir(parents=True, exist_ok=True)
    export_handouts_dir.mkdir(parents=True, exist_ok=True)

    extracted_files: List[Dict[str, str]] = []
    quiz_files: List[Dict[str, str]] = []

    # 1. Process sources/<UnitName>.md -> exports/extractions/<UnitName>_source.html
    sources_dir = unit_dir / "sources"
    primary_file = None
    display_title = clean_unit.replace("_", " ")
    cefr_level = ""

    if sources_dir.exists():
        cand = sources_dir / f"{clean_unit}.md"
        if cand.exists():
            primary_file = cand
        else:
            for f in sorted(sources_dir.iterdir()):
                if f.is_file() and f.suffix in [".md", ".txt"]:
                    primary_file = f
                    break

    if primary_file and primary_file.exists():
        try:
            raw_text = primary_file.read_text(encoding="utf-8")
            tm = re.search(
                r'^(?:title|text_title|unit_title):\s*["\']?(.*?)["\']?\s*$',
                raw_text,
                re.MULTILINE,
            )
            if tm and tm.group(1).strip():
                display_title = tm.group(1).strip()
            clean_source = re.sub(
                r"<syllabus>[\s\S]*?</syllabus>", "", raw_text, flags=re.IGNORECASE
            ).strip()
            source_html = render_markdown_to_standalone_html(
                clean_source,
                fallback_title=f"{display_title} — Source Reading",
                category="source",
            )
            out_source_html = export_extractions_dir / f"{clean_unit}_source.html"
            out_source_html.write_text(source_html, encoding="utf-8")
            extracted_files.append(
                {
                    "title": "Source Text",
                    "filename": f"{clean_unit}_source.html",
                    "rel_path": f"extractions/{clean_unit}_source.html",
                    "category": "source",
                    "description": "Original passage with clean pedagogical typography",
                }
            )
        except (OSError, IOError, ValueError) as e:
            logger.warning(f"Failed to export source text for {clean_unit}: {e}")

    # 2. Process extractions/
    extractions_dir = unit_dir / "extractions"
    if extractions_dir.exists():
        for f in sorted(extractions_dir.iterdir()):
            if not f.is_file():
                continue
            if f.suffix == ".md":
                try:
                    md_text = f.read_text(encoding="utf-8")
                    cat = "handout"
                    desc = "Pedagogical extraction node"
                    card_title = f.stem.replace(clean_unit, "").replace("_", " ").strip().title()
                    if "_vocabulary" in f.name:
                        cat = "vocabulary"
                        card_title = "Vocabulary Focus"
                        desc = "Sense-locked headwords, valency cascades & patterns"
                        cefr_m = re.search(
                            r'overall_cefr_level:\s*["\']?(.*?)["\']?\s*$', md_text, re.MULTILINE
                        )
                        if cefr_m and cefr_m.group(1).strip():
                            cefr_level = cefr_m.group(1).strip().upper()
                    elif "_grammar" in f.name:
                        cat = "grammar"
                        card_title = "Grammar Patterns"
                        desc = "Syntactic frames, constructions & SLA contrast pairs"
                    elif "_summary" in f.name:
                        cat = "summary"
                        card_title = "Discourse Summary"
                        desc = "Discourse organization, macro-propositions & outline"

                    html_doc = render_markdown_to_standalone_html(
                        md_text,
                        fallback_title=f"{display_title} — {card_title}",
                        category=cat,
                    )
                    out_html = export_extractions_dir / f"{f.stem}.html"
                    out_html.write_text(html_doc, encoding="utf-8")
                    extracted_files.append(
                        {
                            "title": card_title,
                            "filename": f"{f.stem}.html",
                            "rel_path": f"extractions/{f.stem}.html",
                            "category": cat,
                            "description": desc,
                        }
                    )
                except (OSError, IOError, ValueError) as e:
                    logger.warning(f"Failed to export extraction {f.name}: {e}")
            elif f.suffix == ".html" and "_mindmap" in f.name:
                try:
                    out_mm = export_extractions_dir / f.name
                    shutil.copy2(f, out_mm)
                    json_cand = f.with_suffix(".json")
                    if json_cand.exists():
                        shutil.copy2(json_cand, export_extractions_dir / json_cand.name)
                    extracted_files.append(
                        {
                            "title": "Knowledge Mind Map",
                            "filename": f.name,
                            "rel_path": f"extractions/{f.name}",
                            "category": "mindmap",
                            "description": "Hierarchical concept graph & structural visualization",
                        }
                    )
                except (OSError, IOError, ValueError) as e:
                    logger.warning(f"Failed to copy mindmap {f.name}: {e}")

    # 3. Process handouts/
    handouts_dir = unit_dir / "handouts"
    if handouts_dir.exists():
        for f in sorted(handouts_dir.iterdir()):
            if not f.is_file() or f.suffix != ".html" or "_quiz" not in f.name:
                continue
            try:
                html_content = f.read_text(encoding="utf-8")
                resolved_video = detect_video_path_from_html(html_content, f)
                media_filename = None
                if resolved_video and resolved_video.exists():
                    media_filename = resolved_video.name
                    shutil.copy2(resolved_video, export_handouts_dir / media_filename)

                portable_html = make_portable_html_content(html_content, media_filename)
                out_quiz = export_handouts_dir / f.name
                out_quiz.write_text(portable_html, encoding="utf-8")

                q_name = f.name
                q_title = "Assessment"
                q_desc = "Interactive assessment"
                if "vocabulary" in q_name:
                    q_title = "Vocabulary Quiz"
                    q_desc = "Single-fit context clozes with diagnostic lexicographic feedback"
                elif "cloze" in q_name:
                    q_title = "Discourse Cloze"
                    q_desc = "Pure open cloze assessment with acceptable-word scoring"
                elif "reading" in q_name:
                    q_title = "Reading Comprehension"
                    q_desc = "Discourse reasoning and fallacy-audited reading comprehension"
                elif "translation" in q_name:
                    q_title = "Translation Practice"
                    q_desc = "SLA minimal-pair error injection & structural contrast"
                elif "listening" in q_name:
                    q_title = "Listening Comprehension"
                    q_desc = "Conversation analysis multi-turn dialogues with authentic native audio"
                elif "video" in q_name:
                    q_title = "Video Comprehension"
                    q_desc = "Timestamp-anchored interactive multimodal video comprehension"

                quiz_files.append(
                    {
                        "title": q_title,
                        "filename": f.name,
                        "rel_path": f"handouts/{f.name}",
                        "description": q_desc,
                    }
                )
            except (OSError, IOError, ValueError) as e:
                logger.warning(f"Failed to export quiz {f.name}: {e}")

    # 4. Generate exports/index.html navigation portal
    index_html = render_index_portal_html(
        unit_name=clean_unit,
        display_title=display_title,
        cefr_level=cefr_level,
        extractions=extracted_files,
        quizzes=quiz_files,
    )
    (export_dir / "index.html").write_text(index_html, encoding="utf-8")

    total_count = 1 + len(extracted_files) + len(quiz_files)
    win_path = to_windows_display_path(export_dir)
    try:
        rel_path = str(export_dir.relative_to(config.project_root))
    except ValueError:
        rel_path = win_path

    return {
        "success": True,
        "unit_name": clean_unit,
        "display_title": display_title,
        "export_path": str(export_dir.resolve()),
        "windows_path": win_path,
        "relative_path": rel_path,
        "file_count": total_count,
        "extractions_count": len(extracted_files),
        "quizzes_count": len(quiz_files),
    }


def export_unit_to_zip(
    unit_name: str, custom_target_dir: Optional[Union[str, Path]] = None
) -> dict:
    """
    Exports all extractions and companion quizzes into a directory, then packages
    the complete unit bundle into a portable ZIP archive for direct browser download.
    """
    dir_res = export_unit_to_directory(unit_name, custom_target_dir)
    if not dir_res.get("success"):
        return dir_res

    export_dir = Path(dir_res["export_path"])
    clean_unit = dir_res["unit_name"]
    zip_path = (
        export_dir.parent / f"{clean_unit}_handouts.zip"
        if export_dir.name == "exports"
        else export_dir / f"{clean_unit}_handouts.zip"
    )

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in export_dir.rglob("*"):
            if file_path.is_file() and file_path != zip_path:
                arcname = file_path.relative_to(export_dir)
                if file_path.suffix.lower() in [".mp4", ".webm", ".ogg", ".mp3", ".wav", ".mkv", ".mov"]:
                    zf.write(file_path, arcname=str(arcname), compress_type=zipfile.ZIP_STORED)
                else:
                    zf.write(file_path, arcname=str(arcname), compress_type=zipfile.ZIP_DEFLATED)

    win_zip_path = to_windows_display_path(zip_path)
    return {
        "success": True,
        "unit_name": clean_unit,
        "zip_path": str(zip_path.resolve()),
        "windows_path": win_zip_path,
        "file_name": zip_path.name,
        "file_size": zip_path.stat().st_size,
    }


def open_directory_in_explorer(dir_path: Union[str, Path]) -> bool:
    """
    Opens the target directory in the host OS native file manager (cross-platform, with full WSL support).
    """
    p = Path(dir_path).resolve()
    if not p.exists():
        return False
    sys_name = platform.system()
    try:
        if sys_name == "Windows":
            os.startfile(str(p))
            return True
        elif sys_name == "Darwin":
            subprocess.Popen(["open", str(p)])
            return True
        else:  # Linux / Unix / WSL
            # 1. Detect Windows Subsystem for Linux (WSL)
            is_wsl = bool(
                os.environ.get("WSL_DISTRO_NAME")
                or os.environ.get("WSL_INTEROP")
                or (Path("/proc/version").exists() and "microsoft" in Path("/proc/version").read_text().lower())
            )
            explorer_cmd = shutil.which("explorer.exe")
            if not explorer_cmd:
                for cand in ["/mnt/c/Windows/explorer.exe", "/mnt/c/WINDOWS/explorer.exe"]:
                    if Path(cand).exists():
                        explorer_cmd = cand
                        break

            if is_wsl and explorer_cmd:
                target = str(p)
                wslpath_cmd = shutil.which("wslpath")
                if wslpath_cmd:
                    try:
                        res = subprocess.run([wslpath_cmd, "-w", str(p)], capture_output=True, text=True, check=True)
                        win_path = res.stdout.strip()
                        if win_path:
                            target = win_path
                    except (subprocess.SubprocessError, FileNotFoundError, OSError) as e:
                        logger.warning(f"wslpath failed for {p}: {e}")
                subprocess.Popen([explorer_cmd, target])
                return True

            # 2. Standard Linux desktop environment
            if shutil.which("xdg-open"):
                subprocess.Popen(["xdg-open", str(p)])
                return True
            return False
    except (subprocess.SubprocessError, FileNotFoundError, OSError) as e:
        logger.warning(f"Failed to open directory {p} in file manager: {e}")
        return False
