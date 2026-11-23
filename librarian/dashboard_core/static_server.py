"""Static asset and wiki node file streaming handlers for DashboardHTTPRequestHandler."""

from __future__ import annotations

import mimetypes
import shutil
import urllib.parse
from pathlib import Path

from ..config import config


class DashboardStaticMixin:
    """Mixin handling static asset, wiki file, and export ZIP streaming."""

    def serve_export_zip(self, query_str: str) -> None:
        query = urllib.parse.parse_qs(query_str or "")
        unit_name = query.get("unit_name", [None])[0]
        target_dir = query.get("target_dir", [None])[0]
        if not unit_name:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(
                b'{"success": false, "error": "unit_name query parameter is required"}'
            )
            return
        try:
            from ..exporter import export_unit_to_zip

            res = export_unit_to_zip(unit_name, custom_target_dir=target_dir)
            if not res.get("success"):
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                import json

                self.wfile.write(
                    json.dumps(
                        {"success": False, "error": res.get("error", "ZIP export failed")}
                    ).encode("utf-8")
                )
                return
            zip_path = Path(res["zip_path"])
            if not zip_path.exists():
                self.send_response(404)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(
                    b'{"success": false, "error": "Exported ZIP file not found"}'
                )
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
            import json

            self.wfile.write(
                json.dumps({"success": False, "error": f"Error generating ZIP: {e}"}).encode(
                    "utf-8"
                )
            )

    def serve_wiki_file(self, path: str) -> None:
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
                mime_type = (
                    "text/plain"
                    if resolved_file.suffix == ".md"
                    else "application/octet-stream"
                )

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

    def serve_dashboard(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self.end_headers()

        template_path = Path(__file__).resolve().parent.parent / "templates" / "dashboard.html"

        if template_path.exists():
            html_content = template_path.read_text(encoding="utf-8")
        else:
            html_content = "<h1>Error: templates/dashboard.html not found!</h1>"

        self.wfile.write(html_content.encode("utf-8"))

    def serve_static_asset(self, path: str) -> None:
        # Resolve path relative to librarian/static/ directory
        rel_path = urllib.parse.unquote(path).lstrip("/")
        # Remove 'static/' prefix to locate inside librarian/static
        if rel_path.startswith("static/"):
            rel_path = rel_path[len("static/"):]

        static_root = (Path(__file__).resolve().parent.parent / "static").resolve()
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
