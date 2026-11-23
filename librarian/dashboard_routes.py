"""Modular route dispatchers for the Lexis Wiki Web Dashboard.

Facade mixin uniting focused domain route mixins under librarian/dashboard_core/
into a unified dispatch table for DashboardHTTPRequestHandler.
"""

from __future__ import annotations

import json
from typing import Dict

from .dashboard_core.routes_ask import RoutesAskMixin
from .dashboard_core.routes_base import RoutesBaseMixin
from .dashboard_core.routes_files import RoutesFilesMixin
from .dashboard_core.routes_glossary import RoutesGlossaryMixin
from .dashboard_core.routes_jobs import RoutesJobsMixin
from .dashboard_core.routes_media import RoutesMediaMixin
from .dashboard_core.routes_quizzes import RoutesQuizzesMixin
from .dashboard_core.routes_system import RoutesSystemMixin


class DashboardRoutesMixin(
    RoutesBaseMixin,
    RoutesSystemMixin,
    RoutesGlossaryMixin,
    RoutesJobsMixin,
    RoutesQuizzesMixin,
    RoutesFilesMixin,
    RoutesMediaMixin,
    RoutesAskMixin,
):
    """Unified route dispatcher mixin for DashboardHTTPRequestHandler."""

    _GET_ROUTE_DISPATCH: Dict[str, str] = {
        "/api/shutdown": "_api_get_shutdown",
        "/api/config": "_api_get_config",
        "/api/tts/voices": "_api_get_tts_voices",
        "/api/models": "_api_get_models",
        "/api/raw-files": "_api_get_raw_files",
        "/api/compiled-units": "_api_get_compiled_units",
        "/api/recent-jobs": "_api_get_recent_jobs",
        "/api/llm-health": "_api_get_llm_health",
        "/api/logs": "_api_get_logs",
        "/api/hero-board": "_api_get_hero_board",
        "/api/ask/stats": "_api_get_ask_stats",
        "/api/wiki-graph": "_api_get_wiki_graph",
        "/api/book-glossary": "_api_get_book_glossary",
    }

    _POST_ROUTE_DISPATCH: Dict[str, str] = {
        "/api/shutdown": "_api_post_shutdown",
        "/api/config/update": "_api_post_config_update",
        "/api/tts/preview": "_api_post_tts_preview",
        "/api/sync-book-glossary": "_api_post_sync_book_glossary",
        "/api/import-pdf-glossary": "_api_post_import_pdf_glossary",
        "/api/compile": "_api_post_compile",
        "/api/re-audit": "_api_post_re_audit",
        "/api/export-unit": "_api_post_export_unit",
        "/api/open-folder": "_api_post_open_folder",
        "/api/export-exe": "_api_post_export_exe",
        "/api/generate-quiz": "_api_post_generate_quiz",
        "/api/generate-all-quizzes": "_api_post_generate_all_quizzes",
        "/api/check-video-source": "_api_post_check_video_source",
        "/api/test-endpoint": "_api_post_test_endpoint",
        "/api/import-video": "_api_post_import_video",
        "/api/select-active-media": "_api_post_select_active_media",
        "/api/list-media": "_api_post_list_media",
        "/api/job-status": "_api_post_job_status",
        "/api/check-upload-collision": "_api_post_check_upload_collision",
        "/api/upload-file": "_api_post_upload_file",
        "/api/upload-media": "_api_post_upload_media",
        "/api/ask": "_api_post_ask",
    }

    def handle_api_get(self, path: str, query_str: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self.end_headers()

        handler_name = self._GET_ROUTE_DISPATCH.get(path)
        if handler_name:
            handler = getattr(self, handler_name, None)
            if handler:
                response_data = handler(query_str)
            else:
                response_data = {"success": False, "error": f"Handler {handler_name} not implemented"}
        else:
            response_data = {}

        self.wfile.write(json.dumps(response_data).encode("utf-8"))

    def handle_api_post(self, path: str) -> None:
        content_length = int(self.headers.get("Content-Length", 0))
        max_limit = 5 * 1024 * 1024 * 1024 if path == "/api/upload-media" else 20 * 1024 * 1024

        if content_length > max_limit:
            limit_mb = (
                max_limit // (1024 * 1024)
                if max_limit < 1024 * 1024 * 1024
                else max_limit // (1024 * 1024 * 1024)
            )
            unit_str = "MB" if max_limit < 1024 * 1024 * 1024 else "GB"
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(
                json.dumps(
                    {
                        "success": False,
                        "error": f"File size exceeds the maximum limit of {limit_mb}{unit_str}.",
                    }
                ).encode("utf-8")
            )
            return

        is_binary = (
            self.headers.get("Content-Type") == "application/octet-stream"
            or path == "/api/upload-media"
        )

        if is_binary:
            body_bytes = self.rfile.read(content_length) if content_length > 0 else b""
            body = ""
        else:
            body_bytes = b""
            body = (
                self.rfile.read(content_length).decode("utf-8") if content_length > 0 else ""
            )

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self.end_headers()

        response_data = {"success": True}

        try:
            handler_name = self._POST_ROUTE_DISPATCH.get(path)
            if handler_name:
                handler = getattr(self, handler_name, None)
                if handler:
                    response_data = handler(body, body_bytes, is_binary)
                else:
                    response_data = {"success": False, "error": f"Handler {handler_name} not implemented"}
            else:
                response_data = {"success": False, "error": f"Unknown POST endpoint: {path}"}
        except Exception as e:
            response_data = {"success": False, "error": str(e)}

        self.wfile.write(json.dumps(response_data).encode("utf-8"))
