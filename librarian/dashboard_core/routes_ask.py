"""Discourse RAG / Lexis Ask search routes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from ..config import config
from ..llm import llm
from ..processor import WikiProcessor


class RoutesAskMixin:
    """Mixin providing Lexis Ask discourse search and stats routes."""

    def _api_get_ask_stats(self, query_str: str) -> Dict[str, Any]:
        response_data: Dict[str, Any] = {"success": True}
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


        return response_data

    def _api_post_ask(self, body: str, body_bytes: bytes, is_binary: bool) -> Dict[str, Any]:
        response_data: Dict[str, Any] = {"success": True}
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


        return response_data

