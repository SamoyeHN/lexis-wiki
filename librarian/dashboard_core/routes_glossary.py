"""Cumulative Lexical Ledger and Book Glossary route handlers."""

from __future__ import annotations

import json
import urllib.parse
from pathlib import Path
from typing import Any, Dict

from ..config import config


class RoutesGlossaryMixin:
    """Mixin providing book glossary ledger and PDF import routes."""

    def _api_get_book_glossary(self, query_str: str) -> Dict[str, Any]:
        response_data: Dict[str, Any] = {"success": True}
        from ..glossary_ledger import get_or_create_ledger
        q = urllib.parse.parse_qs(query_str or "")
        series = q.get("series", ["New Practical English"])[0]
        book = q.get("book", ["1"])[0]
        ledger = get_or_create_ledger(series, book)
        response_data = {
            "success": True,
            "summary": ledger.get_summary(),
            "entries": ledger.data.get("entries", {}),
        }

        self.wfile.write(json.dumps(response_data).encode("utf-8"))

        return response_data

    def _api_post_sync_book_glossary(self, body: str, body_bytes: bytes, is_binary: bool) -> Dict[str, Any]:
        response_data: Dict[str, Any] = {"success": True}
        from ..glossary_ledger import get_or_create_ledger
        data = json.loads(body) if body else {}
        series = data.get("series", "New Practical English")
        book = data.get("book", "1")
        ledger = get_or_create_ledger(series, book)
        res = ledger.sync_bottom_up()
        response_data = {
            "success": True,
            "result": res,
            "summary": ledger.get_summary(),
        }


        return response_data

    def _api_post_import_pdf_glossary(self, body: str, body_bytes: bytes, is_binary: bool) -> Dict[str, Any]:
        response_data: Dict[str, Any] = {"success": True}
        from ..glossary_ledger import get_or_create_ledger
        import urllib.parse
        raw_filename = self.headers.get('X-Filename', '')
        raw_series = self.headers.get('X-Series', '')
        raw_book = self.headers.get('X-Book', '')

        target_series = urllib.parse.unquote(raw_series).strip() if raw_series else None
        target_book = urllib.parse.unquote(raw_book).strip() if raw_book else None

        if is_binary and body_bytes:
            series = target_series or "New Practical English"
            book = target_book or "1"
            filename = urllib.parse.unquote(raw_filename) if raw_filename else f"Book_{book}_glossary.pdf"

            # Persist uploaded PDF source to wiki/_glossary/<series>/
            glossary_dir = Path(config.project_root) / "wiki" / "_glossary" / series.replace(" ", "_")
            glossary_dir.mkdir(parents=True, exist_ok=True)
            target_pdf = glossary_dir / f"Book_{book}_appendix_source.pdf"
            target_pdf.write_bytes(body_bytes)

            ledger = get_or_create_ledger(series, book)
            res = ledger.import_pdf_glossary(str(target_pdf))
            ledger.sync_bottom_up()
            response_data = {
                "success": True,
                "result": res,
                "summary": ledger.get_summary(),
            }
        else:
            data = json.loads(body) if body else {}
            pdf_path = data.get("pdf_path")
            series = data.get("series") or target_series or "New Practical English"
            book = data.get("book") or target_book or "1"
            if not pdf_path:
                response_data = {"success": False, "error": "pdf_path or uploaded PDF file is required"}
            else:
                ledger = get_or_create_ledger(series, book)
                res = ledger.import_pdf_glossary(pdf_path)
                ledger.sync_bottom_up()
                response_data = {
                    "success": True,
                    "result": res,
                    "summary": ledger.get_summary(),
                }


        return response_data

