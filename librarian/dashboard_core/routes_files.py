"""Raw files, compiled units, graph visualization, and file management routes."""

from __future__ import annotations

import json
import os
import re
import shutil
import urllib.parse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..config import config, extract_unit_stem, is_supplement_unit, normalize_name
from ..processor import WikiProcessor


class RoutesFilesMixin:
    """Mixin providing file browsing, unit compilation state, and file export routes."""

    def _api_get_raw_files(self, query_str: str) -> Dict[str, Any]:
        response_data: Dict[str, Any] = {"success": True}
        wiki_dir = Path(config.project_root) / "wiki"
        files_list = []

        if wiki_dir.exists():
            for unit_dir in wiki_dir.iterdir():
                if unit_dir.is_dir() and not unit_dir.name.startswith("."):
                    unit_name = unit_dir.name
                    sources_dir = unit_dir / "sources"
                    
                    # Find primary source file
                    primary_file = None
                    if sources_dir.exists():
                        # 1. Match matching unit folder name exactly
                        for ext in [".md", ".txt"]:
                            candidate = sources_dir / f"{unit_name}{ext}"
                            if candidate.exists() and candidate.is_file():
                                primary_file = candidate
                                break
                        
                        # 2. Fallback to any .md or .txt file directly under sources/
                        if not primary_file:
                            for f in sources_dir.iterdir():
                                if f.is_file() and f.suffix in [".md", ".txt"]:
                                    primary_file = f
                                    break
                    
                    if primary_file:
                        stem = unit_name  # The card name/stem on the dashboard is the folder name!
                        
                        # Check compiled status: check if extractions directory contains compiled files
                        compiled = False
                        extractions_dir = unit_dir / "extractions"
                        if extractions_dir.exists():
                            compiled_files = list(extractions_dir.glob("*.md"))
                            if compiled_files:
                                compiled = True
                        
                        # Check video status & extract title
                        is_video = False
                        is_transcribing = False
                        display_title = None
                        try:
                            with open(primary_file, "r", encoding="utf-8") as f_read:
                                head = f_read.read(2000)
                            if "category: \"video_transcript\"" in head or "video_type:" in head:
                                is_video = True
                            if "Transcription is currently in progress" in head:
                                is_transcribing = True
                            
                            # Extract frontmatter title if present (e.g. title: "...")
                            title_match = re.search(r'^title:\s*["\']?(.*?)["\']?\s*$', head, re.MULTILINE)
                            if title_match and title_match.group(1).strip():
                                display_title = title_match.group(1).strip()
                            else:
                                text_title_match = re.search(r'^text_title:\s*["\']?(.*?)["\']?\s*$', head, re.MULTILINE)
                                unit_title_match = re.search(r'^unit_title:\s*["\']?(.*?)["\']?\s*$', head, re.MULTILINE)
                                if text_title_match and text_title_match.group(1).strip():
                                    display_title = text_title_match.group(1).strip()
                                elif unit_title_match and unit_title_match.group(1).strip():
                                    display_title = unit_title_match.group(1).strip()

                            # Extract series, book, and unit hierarchy metadata
                            series_val = None
                            book_val = None
                            book_name_val = None
                            unit_val = None
                            unit_name_val = None
                            section_val = None

                            m_series = re.search(r'^series:\s*["\']?(.*?)["\']?\s*$', head, re.MULTILINE)
                            if m_series and m_series.group(1).strip():
                                series_val = m_series.group(1).strip()

                            m_edition = re.search(r'^edition:\s*["\']?(.*?)["\']?\s*$', head, re.MULTILINE)
                            edition_val = m_edition.group(1).strip() if m_edition and m_edition.group(1).strip() else None

                            m_series_abbr = re.search(r'^series_abbr:\s*["\']?(.*?)["\']?\s*$', head, re.MULTILINE)
                            series_abbr_val = m_series_abbr.group(1).strip() if m_series_abbr and m_series_abbr.group(1).strip() else None

                            clean_series, edition_tag, series_abbr, series_display = self._extract_series_edition(
                                series_val, raw_edition=edition_val, raw_abbr=series_abbr_val
                            )

                            m_book = re.search(r'^book:\s*["\']?(.*?)["\']?\s*$', head, re.MULTILINE)
                            if m_book and m_book.group(1).strip():
                                raw_b = m_book.group(1).strip()
                                try:
                                    book_val = int(raw_b)
                                except ValueError:
                                    book_val = raw_b

                            m_book_name = re.search(r'^book_name:\s*["\']?(.*?)["\']?\s*$', head, re.MULTILINE)
                            if m_book_name and m_book_name.group(1).strip():
                                book_name_val = m_book_name.group(1).strip()
                            elif book_val is not None:
                                book_name_val = f"Book {book_val}"

                            m_unit = re.search(r'^unit:\s*["\']?(.*?)["\']?\s*$', head, re.MULTILINE)
                            if m_unit and m_unit.group(1).strip():
                                raw_u = m_unit.group(1).strip()
                                try:
                                    unit_val = int(raw_u)
                                except ValueError:
                                    unit_val = raw_u

                            m_unit_name = re.search(r'^unit_name:\s*["\']?(.*?)["\']?\s*$', head, re.MULTILINE)
                            if m_unit_name and m_unit_name.group(1).strip():
                                unit_name_val = m_unit_name.group(1).strip()
                            elif unit_val is not None:
                                unit_name_val = f"Unit {unit_val}"

                            m_section = re.search(r'^section:\s*["\']?(.*?)["\']?\s*$', head, re.MULTILINE)
                            if m_section and m_section.group(1).strip():
                                section_val = m_section.group(1).strip()
                        except (OSError, IOError, UnicodeDecodeError):
                            pass
                            
                        if not is_video:
                            media_dir = sources_dir / "media"
                            if media_dir.exists():
                                for mf in media_dir.iterdir():
                                    if mf.is_file() and mf.suffix in [".md", ".txt"]:
                                        is_video = True
                                        break

                        # Detect CEFR level from extractions
                        cefr_level = None
                        if extractions_dir.exists():
                            vocab_candidate = extractions_dir / f"{stem}_vocabulary.md"
                            if not vocab_candidate.exists():
                                for vf in extractions_dir.glob("*_vocabulary.md"):
                                    vocab_candidate = vf
                                    break
                            if vocab_candidate.exists():
                                try:
                                    with open(vocab_candidate, "r", encoding="utf-8") as vf:
                                        v_head = vf.read(1000)
                                        m_cefr = re.search(r'overall_cefr_level:\s*["\']?(.*?)["\']?\s*$', v_head, re.MULTILINE)
                                        if m_cefr and m_cefr.group(1).strip():
                                            cefr_level = m_cefr.group(1).strip().upper()
                                except (OSError, IOError, UnicodeDecodeError):
                                    pass

                        # Check handouts / quizzes count
                        handouts_dir = unit_dir / "handouts"
                        quizzes_list = [h.name for h in handouts_dir.glob("*_quiz.html")] if handouts_dir.exists() else []
                        quiz_count = len(quizzes_list)
                        has_quizzes = quiz_count > 0
                        
                        files_list.append({
                            "name": primary_file.name,
                            "stem": stem,
                            "title": display_title,
                            "series": series_val,
                            "series_clean": clean_series,
                            "series_abbr": series_abbr,
                            "edition": edition_tag,
                            "series_display": series_display,
                            "book": book_val,
                            "book_name": book_name_val,
                            "unit": unit_val,
                            "unit_name": unit_name_val,
                            "section": section_val,
                            "size": primary_file.stat().st_size,
                            "compiled": compiled,
                            "cefr_level": cefr_level,
                            "has_quizzes": has_quizzes,
                            "quiz_count": quiz_count,
                            "is_video": is_video,
                            "is_transcribing": is_transcribing,
                            "path": str(primary_file.resolve()),
                            "mtime": primary_file.stat().st_mtime
                        })

        files_list.sort(key=lambda x: x["mtime"], reverse=True)
        response_data = {"files": files_list}



        return response_data

    def _api_get_compiled_units(self, query_str: str) -> Dict[str, Any]:
        response_data: Dict[str, Any] = {"success": True}
        wiki_dir = Path(config.project_root) / "wiki"
        units_dict = {
            "vocabulary": [],
            "grammar": [],
            "expressions": [],
            "summaries": [],
            "concepts": [],
            "quizzes": [],
            "concepts_by_source": {},
            "supplemental": [],
            "media": []
        }
        
        if wiki_dir.exists():
            concepts_by_source = {}
            media_mtimes = {}  # {unit_name: [(mtime, filename), ...]}
            
            # Targeted scan of structured subdirectory files instead of recursive search over the whole wiki tree.
            for unit_dir in wiki_dir.iterdir():
                if not unit_dir.is_dir():
                    continue
                
                sub_files = []
                
                # 1. sources & sources/media
                sources_path = unit_dir / "sources"
                if sources_path.exists() and sources_path.is_dir():
                    for item in sources_path.iterdir():
                        if item.is_file():
                            sub_files.append(item)
                    media_path = sources_path / "media"
                    if media_path.exists() and media_path.is_dir():
                        for item in media_path.iterdir():
                            if item.is_file():
                                sub_files.append(item)
                                
                # 2. extractions
                extractions_path = unit_dir / "extractions"
                if extractions_path.exists() and extractions_path.is_dir():
                    for item in extractions_path.iterdir():
                        if item.is_file():
                            sub_files.append(item)
                            
                # 3. handouts
                handouts_path = unit_dir / "handouts"
                if handouts_path.exists() and handouts_path.is_dir():
                    for item in handouts_path.iterdir():
                        if item.is_file():
                            sub_files.append(item)
                            
                for f in sub_files:
                    if "sources" in f.parts:
                        if "media" in f.parts:
                            rel_path = str(f.relative_to(wiki_dir)).replace("\\", "/")
                            if rel_path not in units_dict["media"]:
                                units_dict["media"].append(rel_path)
                            rel_parts = rel_path.split("/")
                            if len(rel_parts) >= 4:
                                unit_name = rel_parts[0]
                                filename = rel_parts[-1]
                                media_mtimes.setdefault(unit_name, []).append((f.stat().st_mtime, filename))
                        continue
                    category = None
                    if f.suffix == ".md":
                        if "_vocabulary" in f.name:
                            category = "vocabulary"
                        elif "_grammar" in f.name:
                            category = "grammar"
                        elif "_expressions" in f.name:
                            category = "expressions"
                        elif "_summary" in f.name:
                            category = "summaries"
                        else:
                            category = "supplemental"
                    elif f.suffix == ".html" and "_quiz" in f.name:
                        category = "quizzes"
                    elif "_mindmap" in f.name:
                        # Map mindmap to quizzes list for frontend compatibility
                        category = "quizzes"
                    
                    if category:
                        if category == "quizzes":
                            try:
                                rel_path = str(f.relative_to(Path(config.project_root))).replace("\\", "/")
                                if rel_path not in units_dict[category]:
                                    units_dict[category].append(rel_path)
                            except ValueError:
                                pass
                        else:
                            rel_path = str(f.relative_to(wiki_dir)).replace("\\", "/")
                            if rel_path not in units_dict[category]:
                                units_dict[category].append(rel_path)
                                if category == "summaries":
                                    units_dict["concepts"].append(rel_path)
                                    try:
                                        content = f.read_text(encoding="utf-8")
                                        m = re.search(r'source:\s*"\[\[(.*?)\]\]"', content)
                                        if m:
                                            src = m.group(1)
                                            if src not in concepts_by_source:
                                                concepts_by_source[src] = []
                                            concepts_by_source[src].append(rel_path)
                                    except (OSError, IOError, UnicodeDecodeError):
                                        pass
            units_dict["concepts_by_source"] = concepts_by_source
            active_media = {}
            for unit_name, files in media_mtimes.items():
                if files:
                    files.sort(key=lambda x: x[0], reverse=True)
                    active_media[unit_name] = files[0][1]
            units_dict["active_media"] = active_media
            
        response_data = {"units": units_dict}


        return response_data

    def _api_get_wiki_graph(self, query_str: str) -> Dict[str, Any]:
        response_data: Dict[str, Any] = {"success": True}
        # Generate nodes and links from wiki folder
        wiki_dir = Path(config.project_root) / "wiki"
        nodes = []
        links = []
        node_ids = set()

        if wiki_dir.exists():
            # 1. Gather all compiled markdown nodes recursively
            for f in wiki_dir.rglob("*.md"):
                if f.is_file():
                    cat = None
                    if "_vocabulary" in f.name:
                        cat = "vocabulary"
                    elif "_grammar" in f.name:
                        cat = "grammar"
                    elif "_summary" in f.name:
                        cat = "summaries"
                    
                    if cat:
                        node_id = f"{cat}:{f.stem}"
                        label = f.stem
                        if cat == "summaries":
                            label = f.stem.replace("_", " ").title()
                            
                        rel_p = f.relative_to(Path(config.project_root))
                        nodes.append({
                            "id": node_id,
                            "label": label,
                            "group": cat,
                            "path": str(rel_p).replace("\\", "/")
                        })
                        node_ids.add(node_id)
                        
                        # If it's a summary, also extract individual concepts as virtual nodes pointing to this summary document!
                        if cat == "summaries":
                            try:
                                content = f.read_text(encoding="utf-8")
                                # Parse parent concepts and their nested sub-concepts
                                # Regex matches '### [[Concept Name]]' followed by content until the next '###'
                                concept_blocks = re.split(r'(?m)^###\s+\[\[(.*?)\]\]', content)
                                # concept_blocks[0] is preamble, then [name1, body1, name2, body2, ...]
                                if len(concept_blocks) >= 3:
                                    for i in range(1, len(concept_blocks), 2):
                                        cname = concept_blocks[i].strip()
                                        cbody = concept_blocks[i+1] if i+1 < len(concept_blocks) else ""
                                        c_node_id = f"concept:{cname}"
                                        if c_node_id not in node_ids:
                                            nodes.append({
                                                "id": c_node_id,
                                                "label": cname,
                                                "group": "concepts", # Keep group as concepts for styling
                                                "path": str(rel_p).replace("\\", "/") # Point directly to the unified Unit Summary document!
                                            })
                                            node_ids.add(c_node_id)
                                            links.append({
                                                "source": c_node_id,
                                                "target": node_id
                                            })
                                        # Parse sub-concepts within this concept body: - **[[SubConcept]]**
                                        sub_matches = re.findall(r'-\s+\*\*\[\[(.*?)\]\]\*\*', cbody)
                                        for sname in sub_matches:
                                            sname = sname.strip()
                                            s_node_id = f"subconcept:{sname}"
                                            if s_node_id not in node_ids:
                                                nodes.append({
                                                    "id": s_node_id,
                                                    "label": sname,
                                                    "group": "concepts",
                                                    "path": str(rel_p).replace("\\", "/")
                                                })
                                                node_ids.add(s_node_id)
                                                # Link sub-concept to parent concept
                                                links.append({
                                                    "source": s_node_id,
                                                    "target": c_node_id
                                                })
                                else:
                                    # Fallback for simple matches
                                    concepts_found = re.findall(r'###\s+\[\[(.*?)\]\]', content)
                                    for cname in concepts_found:
                                        c_node_id = f"concept:{cname}"
                                        if c_node_id not in node_ids:
                                            nodes.append({
                                                "id": c_node_id,
                                                "label": cname,
                                                "group": "concepts",
                                                "path": str(rel_p).replace("\\", "/")
                                            })
                                            node_ids.add(c_node_id)
                                            links.append({
                                                "source": c_node_id,
                                                "target": node_id
                                            })
                            except (OSError, IOError, UnicodeDecodeError):
                                pass

            # 2. Extract mother source nodes from frontmatter of each compiled file
            for node in list(nodes): # iterate over a copy as we will add source nodes dynamically
                if node["group"] in ["vocabulary", "grammar", "summaries"]:
                    file_path = Path(config.project_root) / node["path"]
                    if file_path.exists():
                        try:
                            with open(file_path, "r", encoding="utf-8") as f:
                                content = f.read()
                            
                            # Look for source: "[[Some_File.md]]"
                            src_match = re.search(r'source:\s*["\']?\[\[(.*?)\]\]["\']?', content)
                            if src_match:
                                raw_file_name = src_match.group(1).strip()
                                clean_label = raw_file_name.replace(".md", "").replace("_", " ").title() # e.g. Book 4 Unit 1
                                source_node_id = f"source:{raw_file_name}"
                                
                                if source_node_id not in node_ids:
                                    # Resolve unit directory from filename
                                    u_dir = raw_file_name.replace(".md", "").replace(".txt", "")
                                    nodes.append({
                                        "id": source_node_id,
                                        "label": clean_label,
                                        "group": "source",
                                        "path": f"wiki/{u_dir}/sources/{raw_file_name}"
                                    })
                                    node_ids.add(source_node_id)
                                
                                # Create the link connection to the mother source node
                                links.append({
                                    "source": node["id"],
                                    "target": source_node_id
                                })
                        except (OSError, IOError, UnicodeDecodeError):
                            pass

            # 3. Parse links between nodes (Obsidian-style wikilinks)
            for node in nodes:
                if "path" in node and not node["id"].startswith("source:"):
                    file_path = Path(config.project_root) / node["path"]
                    if file_path.exists():
                        try:
                            with open(file_path, "r", encoding="utf-8") as f:
                                content = f.read()
                            
                            # Look for Obsidian-style wikilinks [[Target]]
                            links_found = re.findall(r"\[\[(.*?)\]\]", content)
                            for target in links_found:
                                target_clean = target.strip()
                                # Search in our nodes
                                for potential_node in nodes:
                                    if potential_node["label"].lower() == target_clean.lower():
                                        link_exists = any(
                                            l["source"] == node["id"] and l["target"] == potential_node["id"]
                                            for l in links
                                        )
                                        if not link_exists and node["id"] != potential_node["id"]:
                                            links.append({
                                                "source": node["id"],
                                                "target": potential_node["id"]
                                            })
                                        break
                        except (OSError, IOError, UnicodeDecodeError):
                            pass

        response_data = {"nodes": nodes, "links": links}


        return response_data

    def _api_post_check_upload_collision(self, body: str, body_bytes: bytes, is_binary: bool) -> Dict[str, Any]:
        response_data: Dict[str, Any] = {"success": True}
        data = json.loads(body)
        filename = data.get("filename", "")
        is_media = data.get("is_media", False)
        current_unit = data.get("current_unit", "")

        filename = os.path.basename(filename)
        stem = Path(filename).stem
        
        # Determine unit_name
        if is_media and current_unit:
            unit_stem = extract_unit_stem(current_unit)
            
            unit_name = unit_stem
        else:
            unit_name = stem.replace(" ", "_")

        # Find existing unit directory using case-insensitive normalized matching
        wiki_dir = Path(config.project_root) / "wiki"
        existing_unit_dir = None
        if wiki_dir.exists():
            for child in wiki_dir.iterdir():
                if child.is_dir() and normalize_name(child.name) == normalize_name(unit_name):
                    existing_unit_dir = child
                    unit_name = child.name
                    break

        unit_exists = existing_unit_dir is not None and existing_unit_dir.exists()
        
        collision = False
        collision_type = "none"
        conflicting_file = ""
        unit_exists_no_source = False

        if unit_exists:
            if is_media:
                target_dir = existing_unit_dir / "sources" / "media"
            else:
                target_dir = existing_unit_dir / "sources"

            # For Scenario 3 check (raw files only)
            if not is_media:
                source_exists = False
                if target_dir.exists():
                    for f in target_dir.iterdir():
                        if f.is_file() and f.suffix in [".md", ".txt"]:
                            source_exists = True
                            break
                if not source_exists:
                    unit_exists_no_source = True

            if target_dir.exists():
                norm_stem_upload = normalize_name(stem)
                
                media_exts = {".mp4", ".mkv", ".mp3", ".m4a", ".wav", ".avi", ".flv", ".mov", ".aac"}
                sub_exts = {".srt", ".vtt"}
                
                is_upload_media = Path(filename).suffix.lower() in media_exts
                is_upload_sub = Path(filename).suffix.lower() in sub_exts

                for f in target_dir.iterdir():
                    if f.is_file():
                        f_norm_stem = normalize_name(f.stem)
                        
                        # For media, check if same category to avoid media-subtitle false positives
                        if is_media:
                            f_is_media = f.suffix.lower() in media_exts
                            f_is_sub = f.suffix.lower() in sub_exts
                            category_match = (is_upload_media and f_is_media) or (is_upload_sub and f_is_sub)
                        else:
                            category_match = True # for raw files, both md/txt are comparable
                        
                        if f_norm_stem == norm_stem_upload and category_match:
                            collision = True
                            conflicting_file = f.name
                            
                            # Determine collision type
                            if f.name == filename:
                                collision_type = "exact_match"
                                break # exact match takes highest precedence
                            elif f.name.lower() == filename.lower():
                                collision_type = "case_mismatch"
                            else:
                                collision_type = "ext_mismatch"

        response_data = {
            "success": True,
            "collision": collision,
            "collision_type": collision_type,
            "conflicting_file": conflicting_file,
            "unit_exists": unit_exists,
            "existing_unit_name": unit_name,
            "unit_exists_no_source": unit_exists_no_source
        }


        return response_data

    def _api_post_upload_file(self, body: str, body_bytes: bytes, is_binary: bool) -> Dict[str, Any]:
        response_data: Dict[str, Any] = {"success": True}
        # Handle file upload (ingest only, compile manually)
        import urllib.parse
        raw_filename = self.headers.get('X-Filename', '')
        raw_series = self.headers.get('X-Series', '')
        raw_book = self.headers.get('X-Book', '')
        target_series = urllib.parse.unquote(raw_series).strip() if raw_series else None
        target_book = urllib.parse.unquote(raw_book).strip() if raw_book else None
        if raw_filename:
            filename = urllib.parse.unquote(raw_filename)
            if is_binary:
                file_bytes = body_bytes
            else:
                file_bytes = body.encode('utf-8')
        else:
            try:
                data = json.loads(body)
                filename = data.get("filename")
                content = data.get("content")
                if not target_series and data.get("series"):
                    target_series = str(data.get("series")).strip()
                if not target_book and data.get("book"):
                    target_book = str(data.get("book")).strip()
                if isinstance(content, str):
                    file_bytes = content.encode("utf-8")
                elif isinstance(content, bytes):
                    file_bytes = content
                else:
                    file_bytes = b""
            except (json.JSONDecodeError, KeyError, ValueError, TypeError):
                filename = None
                file_bytes = b""

        if not filename or file_bytes is None:
            response_data = {"success": False, "error": "Filename and content required"}
        else:
            filename = os.path.basename(filename)
            stem = Path(filename).stem
            unit_name = stem.replace(" ", "_")
            
            # Find existing unit dir if any
            wiki_dir = Path(config.project_root) / "wiki"
            existing_unit_dir = None
            if wiki_dir.exists():
                for child in wiki_dir.iterdir():
                    if child.is_dir() and normalize_name(child.name) == normalize_name(unit_name):
                        existing_unit_dir = child
                        unit_name = child.name
                        break
            
            target_dir = wiki_dir / unit_name / "sources"
            
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

            # Check Scenario 3: Unit folder exists but source file does not
            unit_existed = (wiki_dir / unit_name).exists()
            source_existed = False
            if target_dir.exists():
                for f in target_dir.iterdir():
                    if f.is_file() and f.suffix in [".md", ".txt"]:
                        source_existed = True
                        break
            
            target_dir.mkdir(parents=True, exist_ok=True)
            target_file = target_dir / f"{unit_name}{Path(filename).suffix}"
            
            # Delete any file under target_dir that has the same normalized stem
            # to prevent side-by-side duplicates (e.g. .md vs .txt, or case mismatch)
            norm_stem_upload = normalize_name(stem)
            if target_dir.exists():
                for f in list(target_dir.iterdir()):
                    if f.is_file():
                        if normalize_name(f.stem) == norm_stem_upload:
                            try:
                                f.unlink()
                            except OSError:
                                pass
            
            if target_file.suffix.lower() in (".md", ".txt"):
                try:
                    from ..linguistics import LinguisticEngine
                    raw_text = file_bytes.decode("utf-8-sig", errors="replace")
                    cleaned_text = LinguisticEngine.sanitize_source_markdown(raw_text)
                    cleaned_text = self._inject_or_lock_frontmatter(cleaned_text, target_series=target_series, target_book=target_book)
                    target_file.write_text(cleaned_text, encoding="utf-8")
                except (OSError, ValueError):
                    with open(target_file, "wb") as f:
                        f.write(file_bytes)
            else:
                with open(target_file, "wb") as f:
                    f.write(file_bytes)
            
            if unit_existed and not source_existed:
                msg = f"Successfully added {filename} as the source material for the existing unit '{unit_name}'!"
            else:
                msg = f"Successfully uploaded and ingested {filename} to {unit_name}/sources/!"
                
            response_data = {
                "success": True, 
                "message": msg,
                "filename": target_file.name
            }


        return response_data

    def _api_post_export_unit(self, body: str, body_bytes: bytes, is_binary: bool) -> Dict[str, Any]:
        response_data: Dict[str, Any] = {"success": True}
        data = json.loads(body) if body else {}
        unit_name = data.get("unit_name")
        target_dir = data.get("target_dir")
        save_as_default = bool(data.get("save_as_default", False))
        if not unit_name:
            response_data = {"success": False, "error": "unit_name is required"}
        else:
            try:
                from ..exporter import export_unit_to_directory
                if save_as_default and target_dir:
                    config.update_config("export_dir", str(target_dir).strip())
                response_data = export_unit_to_directory(unit_name, custom_target_dir=target_dir)
            except Exception as e:
                response_data = {"success": False, "error": str(e)}


        return response_data

    def _api_post_open_folder(self, body: str, body_bytes: bytes, is_binary: bool) -> Dict[str, Any]:
        response_data: Dict[str, Any] = {"success": True}
        data = json.loads(body) if body else {}
        folder_path = data.get("path")
        if not folder_path:
            response_data = {"success": False, "error": "path is required"}
        else:
            try:
                from ..exporter import open_directory_in_explorer
                p = Path(folder_path).resolve()
                if p.exists():
                    ok = open_directory_in_explorer(p)
                    response_data = {"success": ok, "path": str(p)}
                else:
                    response_data = {"success": False, "error": f"Folder does not exist: {p}"}
            except Exception as e:
                response_data = {"success": False, "error": str(e)}


        return response_data

    def _api_post_export_exe(self, body: str, body_bytes: bytes, is_binary: bool) -> Dict[str, Any]:
        response_data: Dict[str, Any] = {"success": True}
        data = json.loads(body)
        html_path = data.get("html_path")
        if not html_path:
            response_data = {"success": False, "error": "HTML handout path is required"}
        else:
            try:
                from ..exporter import export_portable_bundle
                result_path = export_portable_bundle(html_path)
                if result_path:
                    try:
                        rel_url = Path(result_path).relative_to(config.project_root).as_posix()
                        download_url = f"/{rel_url}"
                    except ValueError:
                        download_url = f"/wiki/{Path(result_path).parent.name}/handouts/{Path(result_path).name}"
                    response_data = {
                        "success": True,
                        "download_url": download_url,
                        "message": "Successfully bundled portable ZIP (HTML + Video)!"
                    }
                else:
                    response_data = {"success": False, "error": "Portable ZIP packaging failed."}
            except Exception as e:
                response_data = {"success": False, "error": str(e)}


        return response_data

