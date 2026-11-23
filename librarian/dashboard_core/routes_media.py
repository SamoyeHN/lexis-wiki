"""Media files, video import, and video transcription routes."""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
import threading
import urllib.parse
import uuid
from pathlib import Path
from typing import Any, Dict

from ..config import config, extract_unit_stem, is_supplement_unit, normalize_name


class RoutesMediaMixin:
    """Mixin providing media upload, video import, and transcription routes."""

    def _api_post_check_video_source(self, body: str, body_bytes: bytes, is_binary: bool) -> Dict[str, Any]:
        response_data: Dict[str, Any] = {"success": True}
        req = json.loads(body) if body else {}
        filename = req.get("filename", "")

        # Resolve core_name from filename (strip path prefixes and extensions)
        fn_parts = Path(filename).parts
        if "wiki" in fn_parts:
            core_name = fn_parts[fn_parts.index("wiki") + 1] if fn_parts.index("wiki") + 1 < len(fn_parts) else filename
        else:
            core_name = fn_parts[0] if fn_parts else filename
        for ext in [".md", ".txt"]:
            if core_name.endswith(ext):
                core_name = core_name[:-len(ext)]

        is_video = False
        project_root = Path(config.project_root)

        # Primary check: wiki/<core_name>/sources/media/ contains transcript files
        media_dir = project_root / "wiki" / core_name / "sources" / "media"
        if media_dir.exists():
            for mf in media_dir.iterdir():
                if mf.is_file() and mf.suffix in [".md", ".txt"]:
                    is_video = True
                    break

        # Secondary check: wiki source files have video frontmatter
        if not is_video:
            candidates = []
            unit_wiki = project_root / "wiki" / core_name
            if unit_wiki.exists():
                candidates += list(unit_wiki.rglob("sources/*.md")) + list(unit_wiki.rglob("sources/*.txt"))
            for candidate in candidates:
                if not candidate.exists():
                    continue
                try:
                    with open(candidate, "r", encoding="utf-8") as cf:
                        head = cf.read(800)
                    if 'category: "video_transcript"' in head or 'video_type:' in head:
                        is_video = True
                        break
                    if head.startswith("---"):
                        fm_parts = head.split("---", 2)
                        if len(fm_parts) >= 3:
                            for line in fm_parts[1].splitlines():
                                if ":" in line:
                                    k, v = line.split(":", 1)
                                    v = v.strip().strip('"\'')
                                    if k.strip().lower() in ("source_url", "video_url", "url", "source"):
                                        if v and any(d in v for d in ("youtube.com", "youtu.be")):
                                            is_video = True
                                            break
                        if is_video:
                            break
                except (OSError, IOError, UnicodeDecodeError):
                    pass

        response_data = {"success": True, "is_video": is_video}


        return response_data

    def _api_post_import_video(self, body: str, body_bytes: bytes, is_binary: bool) -> Dict[str, Any]:
        response_data: Dict[str, Any] = {"success": True}
        data = json.loads(body)
        url_or_filepath = data.get("url")
        cookies_from_browser = data.get("cookies_from_browser")
        cookies_file = data.get("cookies")
        subtitle_file = data.get("subtitle")
        current_unit = data.get("current_unit")

        if not cookies_from_browser or cookies_from_browser == "none":
            cookies_from_browser = None
        if not cookies_file:
            cookies_file = None
        if not subtitle_file:
            subtitle_file = None

        if not url_or_filepath:
            response_data = {"success": False, "error": "Video URL or filepath is required"}
        else:
            job_id = self._dashboard_state().create_job("video-import", url_or_filepath)
            with self._dashboard_state().lock:
                self._dashboard_state().jobs[job_id]["unit_name"] = current_unit
            t = threading.Thread(
                target=self._background_video_import_worker(), 
                args=(job_id, url_or_filepath, cookies_from_browser, cookies_file, subtitle_file, current_unit)
            )
            t.daemon = True
            t.start()
            response_data = {"success": True, "job_id": job_id, "message": "Video import task spawned successfully."}


        return response_data

    def _api_post_select_active_media(self, body: str, body_bytes: bytes, is_binary: bool) -> Dict[str, Any]:
        response_data: Dict[str, Any] = {"success": True}
        data = json.loads(body)
        unit_name = os.path.basename(str(data.get("unit") or ""))
        media_name = os.path.basename(str(data.get("media") or ""))

        if not unit_name or not media_name:
            response_data = {"success": False, "error": "unit and media are required"}
        else:
            wiki_dir = (Path(config.project_root) / "wiki").resolve()
            unit_dir = wiki_dir / unit_name
            
            if wiki_dir.exists():
                for child in wiki_dir.iterdir():
                    if child.is_dir() and child.name.replace(" ", "").replace("_", "").lower() == unit_name.replace(" ", "").replace("_", "").lower():
                        unit_dir = child
                        break
            
            media_file_path = (unit_dir / "sources" / "media" / media_name).resolve()
            if not media_file_path.is_relative_to(wiki_dir):
                response_data = {"success": False, "error": "Access Denied"}
            elif media_file_path.exists() and media_file_path.is_file():
                try:
                    media_file_path.touch()
                    response_data = {"success": True, "message": f"Successfully activated media {media_name} via touch."}
                except Exception as e:
                    response_data = {"success": False, "error": str(e)}
            else:
                response_data = {"success": False, "error": f"Media file {media_name} does not exist inside {unit_name} sources."}


        return response_data

    def _api_post_list_media(self, body: str, body_bytes: bytes, is_binary: bool) -> Dict[str, Any]:
        response_data: Dict[str, Any] = {"success": True}
        media_files = []
        wiki_dir = Path(config.project_root) / "wiki"
        if wiki_dir.exists():
            for f in wiki_dir.rglob("sources/media/*"):
                if f.is_file() and f.suffix.lower() in [".mp4", ".mp3", ".mkv", ".avi", ".wav", ".aac", ".mov", ".flv", ".m4a"]:
                    media_files.append({
                        "name": f.name,
                        "size": f.stat().st_size
                    })
        response_data = {"success": True, "files": media_files}


        return response_data

    def _api_post_upload_media(self, body: str, body_bytes: bytes, is_binary: bool) -> Dict[str, Any]:
        response_data: Dict[str, Any] = {"success": True}
        # Handle binary media file upload
        import urllib.parse
        raw_filename = self.headers.get('X-Filename', 'upload.mp4')
        filename = urllib.parse.unquote(raw_filename)
        filename = os.path.basename(filename)

        # Get current unit name from headers (node folder)
        raw_current_unit = self.headers.get('X-Current-Unit', '')
        current_unit = urllib.parse.unquote(raw_current_unit).strip()

        if current_unit:
            unit_name = extract_unit_stem(current_unit)
        else:
            stem = Path(filename).stem
            unit_name = stem.replace(" ", "_")

        # Find existing unit dir
        wiki_dir = Path(config.project_root) / "wiki"
        if wiki_dir.exists():
            for child in wiki_dir.iterdir():
                if child.is_dir() and normalize_name(child.name) == normalize_name(unit_name):
                    unit_name = child.name
                    break

        target_dir = wiki_dir / unit_name / "sources" / "media"
        
        # Path traversal check on target directory
        try:
            resolved_wiki = wiki_dir.resolve()
            resolved_target_dir = target_dir.resolve()
            if not resolved_target_dir.is_relative_to(resolved_wiki):
                response_data = {"success": False, "error": "Access Denied: Path traversal detected"}
                self.wfile.write(json.dumps(response_data).encode("utf-8"))
                return
        except (OSError, ValueError) as ex:
            response_data = {"success": False, "error": f"Path resolution failed: {str(ex)}"}
            self.wfile.write(json.dumps(response_data).encode("utf-8"))
            return

        target_dir.mkdir(parents=True, exist_ok=True)

        # Pre-calculate upload_stem to check if upload matches the unit name
        import re
        upload_stem = Path(filename).stem
        upload_stem = re.sub(r'[\\/*?:"<>| ]', '_', upload_stem)
        upload_stem = re.sub(r'_+', '_', upload_stem).strip('_')
        if len(upload_stem) > 40:
            upload_stem = upload_stem[:40].strip('_')

        # Determine if this is a supplement/companion file or a primary file using helper
        is_supplement = is_supplement_unit(unit_name, upload_stem, current_unit)

        if is_supplement:
            file_stem = upload_stem
        else:
            file_stem = unit_name

        # Rename uploaded files to match the appropriate stem
        target_file = target_dir / f"{file_stem}{Path(filename).suffix}"
        
        # Path traversal check on final target file
        try:
            resolved_target_file = target_file.resolve()
            if not resolved_target_file.is_relative_to(resolved_wiki):
                response_data = {"success": False, "error": "Access Denied: Path traversal detected"}
                self.wfile.write(json.dumps(response_data).encode("utf-8"))
                return
        except (OSError, ValueError) as ex:
            response_data = {"success": False, "error": f"Path resolution failed: {str(ex)}"}
            self.wfile.write(json.dumps(response_data).encode("utf-8"))
            return

        # Delete old conflicting media files with the same stem (e.g. extension or case mismatches)
        norm_stem_upload = normalize_name(file_stem)
        media_exts = {".mp4", ".mkv", ".mp3", ".m4a", ".wav", ".avi", ".flv", ".mov", ".aac"}
        sub_exts = {".srt", ".vtt"}
        
        is_upload_media = Path(filename).suffix.lower() in media_exts
        is_upload_sub = Path(filename).suffix.lower() in sub_exts
        
        if target_dir.exists():
            for f in list(target_dir.iterdir()):
                if f.is_file() and normalize_name(f.stem) == norm_stem_upload:
                    f_is_media = f.suffix.lower() in media_exts
                    f_is_sub = f.suffix.lower() in sub_exts
                    # Delete old matching stem files if same category (media vs media, or subtitle vs subtitle)
                    if (is_upload_media and f_is_media) or (is_upload_sub and f_is_sub):
                        try:
                            f.unlink()
                        except OSError:
                            pass

        with open(target_file, "wb") as f:
            f.write(body_bytes)

        # Unification: Create a transcript placeholder .md file
        # so that the unit node shows up in the library instantly while transcription runs in the background.
        if is_upload_media:
            if is_supplement:
                placeholder_file = target_dir / f"{file_stem}.md"
            else:
                placeholder_file = wiki_dir / unit_name / "sources" / f"{unit_name}.md"
            
            # Path traversal check on placeholder file
            try:
                resolved_placeholder = placeholder_file.resolve()
                if not resolved_placeholder.is_relative_to(resolved_wiki):
                    response_data = {"success": False, "error": "Access Denied: Path traversal detected"}
                    self.wfile.write(json.dumps(response_data).encode("utf-8"))
                    return
            except (OSError, ValueError) as ex:
                response_data = {"success": False, "error": f"Path resolution failed: {str(ex)}"}
                self.wfile.write(json.dumps(response_data).encode("utf-8"))
                return

            placeholder_file.parent.mkdir(parents=True, exist_ok=True)
            
            clean_title = file_stem.replace("_", " ").title()
            placeholder_content = f"""---
        title: "{clean_title}"
        video_type: "local"
        source_url: "/api/media/{file_stem}{Path(filename).suffix}"
        category: "video_transcript"
        ---
        # {clean_title}

        ⏳ Transcription is currently in progress. Please wait...
        """
            try:
                placeholder_file.write_text(placeholder_content, encoding="utf-8")
            except OSError as pe:
                print(f"Error writing placeholder transcript: {pe}")

        response_data = {
            "success": True,
            "message": f"Successfully uploaded {filename}",
            "filepath": str(target_file),
            "unit_name": unit_name
        }


        return response_data

