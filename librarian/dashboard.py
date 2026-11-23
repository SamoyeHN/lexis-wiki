"""HTTP Server and request handler for the Lexis Wiki Web Dashboard.

Provides interactive dashboard for inspecting wiki nodes, triggering builds,
managing vocabulary/grammar extractions, and generating SLA quizzes.
"""

from __future__ import annotations

import http.server
import socketserver
import urllib.parse

from .dashboard_core import (
    DashboardState,
    DashboardStaticMixin,
    background_all_quizzes_worker,
    background_audit_batch_worker,
    background_compile_worker,
    background_quiz_worker,
    background_video_import_worker,
    extract_series_edition,
    get_raw_file_stem,
    inject_or_lock_frontmatter,
)
from .dashboard_routes import DashboardRoutesMixin


class DashboardHTTPRequestHandler(
    DashboardRoutesMixin,
    DashboardStaticMixin,
    http.server.BaseHTTPRequestHandler,
):
    """Unified HTTP request handler combining modular domain routes and static file streaming."""

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
        if path in ("/", "/index.html", "/dashboard"):
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


class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    """Handle requests in separate threads to allow concurrent media streaming and API polling."""

    daemon_threads = True


__all__ = [
    "DashboardHTTPRequestHandler",
    "DashboardState",
    "ThreadedHTTPServer",
    "get_raw_file_stem",
    "extract_series_edition",
    "inject_or_lock_frontmatter",
    "background_compile_worker",
    "background_quiz_worker",
    "background_all_quizzes_worker",
    "background_video_import_worker",
    "background_audit_batch_worker",
]
