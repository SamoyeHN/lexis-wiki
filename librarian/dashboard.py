import http.server
import socketserver
import json
import os
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
def get_raw_file_stem(filepath: Path) -> str:
    return filepath.stem


# Global state to keep track of active background jobs and logging
class DashboardState:
    jobs = {}
    lock = threading.Lock()

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
                "logs": initial_logs,
                "saved_files": []
            }
            return job_id

    @classmethod
    def update_job(cls, job_id, progress=None, status=None, log=None, saved_files=None):
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
        
        if "Error" in status and not saved_files:
            DashboardState.update_job(job_id, progress=100, status="failed", log=f"❌ Compilation failed: {status}")
        else:
            DashboardState.update_job(job_id, progress=100, status="completed", log=f"✨ Finished compiling{cats_str}!", saved_files=saved_files)
            for path in saved_files:
                DashboardState.update_job(job_id, log=f"✅ Saved wiki node: {os.path.basename(path)}")
    except Exception as e:
        err_msg = traceback.format_exc()
        DashboardState.update_job(job_id, progress=100, status="failed", log=f"❌ Exception occurred: {err_msg}")

def background_quiz_worker(job_id, filename, count, template):
    try:
        DashboardState.update_job(job_id, progress=30, log=f"[QUIZ] Triggering generation from unit '{filename}'...")
        DashboardState.update_job(job_id, progress=50, log=f"[QUIZ] Prompting LLM with template '{template}' (Target count: {count})...")
        
        processor = WikiProcessor()
        result = processor.generate_quiz(filename, count=count, template_name=template)
        
        if "Error" in result:
            DashboardState.update_job(job_id, progress=100, status="failed", log=f"[ERROR] Quiz generation failed: {result}")
        else:
            try:
                rel_path = Path(result).relative_to(Path(config.project_root))
                web_path = str(rel_path).replace("\\", "/")
            except Exception:
                web_path = result
            DashboardState.update_job(job_id, progress=100, status="completed", log="[QUIZ] Quiz generation completed successfully!", saved_files=[web_path])
            DashboardState.update_job(job_id, log=f"[QUIZ] Handout saved to: {os.path.basename(result)}")
    except Exception as e:
        err_msg = traceback.format_exc()
        DashboardState.update_job(job_id, progress=100, status="failed", log=f"[ERROR] Exception occurred: {err_msg}")

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
        except Exception:
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

class DashboardHTTPRequestHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Silence standard HTTP logger to keep terminal clean
        pass

    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        path = url.path

        # Handle API routes
        if path.startswith("/api/"):
            if path.startswith("/api/media/"):
                self.serve_wiki_file(path)
                return
            self.handle_api_get(path, url.query)
            return

        # Serve static wiki/quiz files if requested
        if path.startswith("/wiki/") or path.startswith("/logs/"):
            self.serve_wiki_file(path)
            return

        # Serve the Dashboard SPA (or index page)
        if path == "/" or path == "/index.html" or path == "/dashboard":
            self.serve_dashboard()
            return

        # Fallback 404
        self.send_error(404, "File Not Found")

    def do_POST(self):
        url = urllib.parse.urlparse(self.path)
        path = url.path

        if path.startswith("/api/"):
            self.handle_api_post(path)
            return

        self.send_error(404, "Not Found")

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


        # Security check to stay inside wiki dir or project root
        try:
            resolved_file = file_path.resolve()
            resolved_root = Path(config.project_root).resolve()
            if not str(resolved_file).startswith(str(resolved_root)):
                self.send_error(403, "Access Denied")
                return
        except Exception:
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
            range_header = self.headers.get("Range")
            if range_header and range_header.startswith("bytes="):
                try:
                    size = resolved_file.stat().st_size
                    ranges = range_header.replace("bytes=", "").split("-")
                    start = int(ranges[0]) if ranges[0] else 0
                    end = int(ranges[1]) if ranges[1] else size - 1
                    if start >= size:
                        self.send_error(416, "Requested range not satisfiable")
                        return
                    if end >= size:
                        end = size - 1
                    length = end - start + 1

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
                    return
                except Exception as range_err:
                    pass

            self.send_response(200)
            self.send_header("Content-Type", mime_type)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            with open(resolved_file, "rb") as f:
                self.wfile.write(f.read())
        else:
            self.send_error(404, "File Not Found")

    def handle_api_get(self, path, query_str):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        response_data = {}

        if path == "/api/config":
            response_data = {
                "config": config.data,
                "project_root": str(config.project_root)
            }

        elif path == "/api/models":
            try:
                models = llm.list_models()
                response_data = {
                    "models": models,
                    "active_model": config.get("model")
                }
            except Exception as e:
                response_data = {"error": str(e), "models": [], "active_model": config.get("model")}

        elif path == "/api/raw-files":
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
                            
                            # Check video status
                            is_video = False
                            is_transcribing = False
                            try:
                                with open(primary_file, "r", encoding="utf-8") as f_read:
                                    head = f_read.read(1000)
                                if "category: \"video_transcript\"" in head or "video_type:" in head:
                                    is_video = True
                                if "Transcription is currently in progress" in head:
                                    is_transcribing = True
                            except Exception:
                                pass
                                
                            if not is_video:
                                media_dir = sources_dir / "media"
                                if media_dir.exists():
                                    for mf in media_dir.iterdir():
                                        if mf.is_file() and mf.suffix in [".md", ".txt"]:
                                            is_video = True
                                            break
                            
                            files_list.append({
                                "name": primary_file.name,
                                "stem": stem,
                                "size": primary_file.stat().st_size,
                                "compiled": compiled,
                                "is_video": is_video,
                                "is_transcribing": is_transcribing,
                                "path": str(primary_file.resolve()),
                                "mtime": primary_file.stat().st_mtime
                            })

            files_list.sort(key=lambda x: x["mtime"], reverse=True)
            response_data = {"files": files_list}


        elif path == "/api/compiled-units":
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
                                except Exception:
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
                                        except Exception:
                                            pass
                units_dict["concepts_by_source"] = concepts_by_source
                active_media = {}
                for unit_name, files in media_mtimes.items():
                    if files:
                        files.sort(key=lambda x: x[0], reverse=True)
                        active_media[unit_name] = files[0][1]
                units_dict["active_media"] = active_media
                
            response_data = {"units": units_dict}

        elif path == "/api/recent-jobs":
            response_data = {"jobs": DashboardState.get_recent_jobs()}

        elif path == "/api/logs":
            logs_dir = Path(config.project_root) / "logs"
            logs_list = []
            if logs_dir.exists():
                for f in logs_dir.iterdir():
                    if f.is_file() and f.suffix in [".log", ".txt"]:
                        logs_list.append({
                            "name": f.name,
                            "size": f.stat().st_size,
                            "mtime": f.stat().st_mtime,
                            "path": f"logs/{f.name}"
                        })
                # Sort logs by modification time descending (newest first)
                logs_list.sort(key=lambda x: x["mtime"], reverse=True)
            response_data = {"logs": logs_list}

        elif path == "/api/hero-board":
            from .evaluator import LogEvaluator
            response_data = LogEvaluator.audit_all_logs()

        elif path == "/api/ask/stats":
            wiki_dir = Path(config.project_root) / "wiki"
            all_nodes = []
            if wiki_dir.exists():
                for f in wiki_dir.glob("**/*"):
                    if f.is_file() and f.suffix in [".md", ".html", ".txt"]:
                        all_nodes.append(f)
            response_data = {
                "success": True,
                "node_count": len(all_nodes)
            }

        elif path == "/api/wiki-graph":
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
                                    concepts_found = re.findall(r'###\s+\[\[(.*?)\]\]', content)
                                    for cname in concepts_found:
                                        c_node_id = f"concept:{cname}"
                                        if c_node_id not in node_ids:
                                            nodes.append({
                                                "id": c_node_id,
                                                "label": cname,
                                                "group": "concepts", # Keep group as concepts for styling
                                                "path": str(rel_p).replace("\\", "/") # Point directly to the unified Unit Summary document!
                                            })
                                            node_ids.add(c_node_id)
                                            # Also automatically link the concept node to its parent summary node!
                                            links.append({
                                                "source": c_node_id,
                                                "target": node_id
                                            })
                                except Exception:
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
                            except Exception:
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
                            except Exception:
                                pass

            response_data = {"nodes": nodes, "links": links}

        self.wfile.write(json.dumps(response_data).encode("utf-8"))

    def handle_api_post(self, path):
        # Read request body
        content_length = int(self.headers.get('Content-Length', 0))
        
        # Enforce tailored limits: 5GB for media, 20MB for text/source uploads and rest
        max_limit = 5 * 1024 * 1024 * 1024 if path == "/api/upload-media" else 20 * 1024 * 1024
        
        if content_length > max_limit:
            limit_mb = max_limit // (1024 * 1024) if max_limit < 1024 * 1024 * 1024 else max_limit // (1024 * 1024 * 1024)
            unit_str = "MB" if max_limit < 1024 * 1024 * 1024 else "GB"
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({
                "success": False, 
                "error": f"File size exceeds the maximum limit of {limit_mb}{unit_str}."
            }).encode("utf-8"))
            return

        # Check if binary upload (e.g. video/audio)
        is_binary = self.headers.get('Content-Type') == 'application/octet-stream' or path == '/api/upload-media'
        
        if is_binary:
            body_bytes = self.rfile.read(content_length) if content_length > 0 else b""
            body = ""
        else:
            body_bytes = b""
            body = self.rfile.read(content_length).decode('utf-8') if content_length > 0 else ""

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        response_data = {"success": True}

        try:
            if path == "/api/config/update":
                data = json.loads(body)
                for key, val in data.items():
                    success, msg = config.update_config(key, val)
                    if not success:
                        response_data = {"success": False, "error": msg}
                        break
                response_data["message"] = "Configuration updated successfully."

            elif path == "/api/compile":
                data = json.loads(body)
                filename = data.get("filename")
                categories = data.get("categories") # list of strings
                if not filename:
                    response_data = {"success": False, "error": "Filename is required"}
                else:
                    job_name = f"{filename} ({', '.join(categories)})" if categories else filename
                    job_id = DashboardState.create_job("compile", job_name)
                    # Start asynchronous background compiler with categories list
                    t = threading.Thread(target=background_compile_worker, args=(job_id, filename, categories))
                    t.daemon = True
                    t.start()
                    response_data = {"success": True, "job_id": job_id, "message": "Compilation task spawned successfully."}

            elif path == "/api/export-exe":
                data = json.loads(body)
                html_path = data.get("html_path")
                if not html_path:
                    response_data = {"success": False, "error": "HTML handout path is required"}
                else:
                    try:
                        from .exporter import export_standalone_exe
                        result_path = export_standalone_exe(html_path)
                        if result_path:
                            try:
                                rel_url = Path(result_path).relative_to(config.project_root).as_posix()
                                download_url = f"/{rel_url}"
                            except Exception:
                                download_url = f"/wiki/{Path(result_path).parent.name}/handouts/{Path(result_path).name}"
                            response_data = {
                                "success": True,
                                "download_url": download_url,
                                "message": "Successfully bundled standalone EXE!"
                            }
                        else:
                            response_data = {"success": False, "error": "Standalone EXE packaging failed. Check PyInstaller installation."}
                    except Exception as e:
                        response_data = {"success": False, "error": str(e)}

            elif path == "/api/generate-quiz":
                data = json.loads(body)
                filename = data.get("filename")
                count = max(1, min(int(data.get("count", 10)), 100))
                template = data.get("template", "vocabulary")

                if not filename:
                    response_data = {"success": False, "error": "Unit or vocabulary filename is required"}
                else:
                    job_id = DashboardState.create_job("quiz", f"{filename} ({template})")
                    t = threading.Thread(target=background_quiz_worker, args=(job_id, filename, count, template))
                    t.daemon = True
                    t.start()
                    response_data = {"success": True, "job_id": job_id, "message": "Quiz generation task spawned successfully."}

            elif path == "/api/check-video-source":
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
                        except Exception:
                            pass

                response_data = {"success": True, "is_video": is_video}

            elif path == "/api/test-endpoint":
                response_data = {"success": True, "test": "working"}

            elif path == "/api/import-video":
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
                    job_id = DashboardState.create_job("video-import", url_or_filepath)
                    with DashboardState.lock:
                        DashboardState.jobs[job_id]["unit_name"] = current_unit
                    t = threading.Thread(
                        target=background_video_import_worker, 
                        args=(job_id, url_or_filepath, cookies_from_browser, cookies_file, subtitle_file, current_unit)
                    )
                    t.daemon = True
                    t.start()
                    response_data = {"success": True, "job_id": job_id, "message": "Video import task spawned successfully."}

            elif path == "/api/select-active-media":
                data = json.loads(body)
                unit_name = data.get("unit")
                media_name = data.get("media")

                if not unit_name or not media_name:
                    response_data = {"success": False, "error": "unit and media are required"}
                else:
                    wiki_dir = Path(config.project_root) / "wiki"
                    unit_dir = wiki_dir / unit_name
                    
                    if wiki_dir.exists():
                        for child in wiki_dir.iterdir():
                            if child.is_dir() and child.name.replace(" ", "").replace("_", "").lower() == unit_name.replace(" ", "").replace("_", "").lower():
                                unit_dir = child
                                break
                    
                    media_file_path = unit_dir / "sources" / "media" / media_name
                    if media_file_path.exists() and media_file_path.is_file():
                        try:
                            media_file_path.touch()
                            response_data = {"success": True, "message": f"Successfully activated media {media_name} via touch."}
                        except Exception as e:
                            response_data = {"success": False, "error": str(e)}
                    else:
                        response_data = {"success": False, "error": f"Media file {media_name} does not exist inside {unit_name} sources."}

            elif path == "/api/list-media":
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

            elif path == "/api/job-status":
                data = json.loads(body)
                job_id = data.get("job_id")
                job = DashboardState.get_job(job_id)
                if not job:
                    response_data = {"success": False, "error": "Job not found"}
                else:
                    response_data = {"success": True, "job": job}

            elif path == "/api/check-upload-collision":
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

            elif path == "/api/upload-file":
                # Handle file upload (ingest only, compile manually)
                import urllib.parse
                raw_filename = self.headers.get('X-Filename', '')
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
                        if isinstance(content, str):
                            file_bytes = content.encode("utf-8")
                        elif isinstance(content, bytes):
                            file_bytes = content
                        else:
                            file_bytes = b""
                    except Exception:
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
                        if not str(resolved_target_dir).startswith(str(resolved_wiki)):
                            response_data = {"success": False, "error": "Access Denied: Path traversal detected"}
                            self.wfile.write(json.dumps(response_data).encode("utf-8"))
                            return
                    except Exception as ex:
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
                                    except Exception:
                                        pass
                    
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

            elif path == "/api/upload-media":
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
                    if not str(resolved_target_dir).startswith(str(resolved_wiki)):
                        response_data = {"success": False, "error": "Access Denied: Path traversal detected"}
                        self.wfile.write(json.dumps(response_data).encode("utf-8"))
                        return
                except Exception as ex:
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
                    if not str(resolved_target_file).startswith(str(resolved_wiki)):
                        response_data = {"success": False, "error": "Access Denied: Path traversal detected"}
                        self.wfile.write(json.dumps(response_data).encode("utf-8"))
                        return
                except Exception as ex:
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
                                except Exception:
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
                        if not str(resolved_placeholder).startswith(str(resolved_wiki)):
                            response_data = {"success": False, "error": "Access Denied: Path traversal detected"}
                            self.wfile.write(json.dumps(response_data).encode("utf-8"))
                            return
                    except Exception as ex:
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
                    except Exception as pe:
                        print(f"Error writing placeholder transcript: {pe}")

                response_data = {
                    "success": True,
                    "message": f"Successfully uploaded {filename}",
                    "filepath": str(target_file),
                    "unit_name": unit_name
                }

            elif path == "/api/ask":
                data = json.loads(body)
                question = data.get("question", "")

                try:
                    processor = WikiProcessor()
                    answer = processor.ask_wiki(question)
                    response_data = {
                        "success": True,
                        "answer": answer
                    }
                except Exception as e:
                    response_data = {
                        "success": False,
                        "error": str(e)
                    }

        except Exception as e:
            response_data = {"success": False, "error": str(e)}

        self.wfile.write(json.dumps(response_data).encode("utf-8"))

    def serve_dashboard(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()

        template_path = Path(__file__).parent / "templates" / "dashboard.html"

        if template_path.exists():
            html_content = template_path.read_text(encoding="utf-8")
        else:
            html_content = "<h1>Error: templates/dashboard.html not found!</h1>"

        self.wfile.write(html_content.encode("utf-8"))