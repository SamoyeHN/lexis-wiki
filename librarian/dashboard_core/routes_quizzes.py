"""Quiz generation route handlers."""

from __future__ import annotations

import json
import threading
import uuid
from pathlib import Path
from typing import Any, Dict

from ..config import config, normalize_name


class RoutesQuizzesMixin:
    """Mixin providing individual and batch quiz generation routes."""

    def _api_post_generate_quiz(self, body: str, body_bytes: bytes, is_binary: bool) -> Dict[str, Any]:
        response_data: Dict[str, Any] = {"success": True}
        data = json.loads(body)
        filename = data.get("filename")
        count = max(1, min(int(data.get("count", 10)), 100))
        template = data.get("template", "vocabulary")
        listening_archetype = data.get("listening_archetype") or data.get("archetype")
        listening_scenario = data.get("listening_scenario") or data.get("scenario")

        if not filename:
            response_data = {"success": False, "error": "Unit or vocabulary filename is required"}
        else:
            target_label = f"{filename} ({template})"
            if template == "listening":
                spec_labels = []
                if listening_archetype:
                    spec_labels.append(listening_archetype)
                if listening_scenario:
                    spec_labels.append(listening_scenario)
                if spec_labels:
                    target_label += f" [{', '.join(spec_labels)}]"
            job_id = self._dashboard_state().create_job("quiz", target_label)
            t = threading.Thread(
                target=self._background_quiz_worker(),
                args=(job_id, filename, count, template, listening_archetype, listening_scenario)
            )
            t.daemon = True
            t.start()
            response_data = {"success": True, "job_id": job_id, "message": "Quiz generation task spawned successfully."}


        return response_data

    def _api_post_generate_all_quizzes(self, body: str, body_bytes: bytes, is_binary: bool) -> Dict[str, Any]:
        response_data: Dict[str, Any] = {"success": True}
        data = json.loads(body) if body else {}
        filename = data.get("filename")
        if not filename:
            response_data = {"success": False, "error": "Unit or vocabulary filename is required"}
        else:
            target_label = f"{filename} (All Quizzes)"
            job_id = self._dashboard_state().create_job("quiz", target_label)
            t = threading.Thread(
                target=self._background_all_quizzes_worker(),
                args=(job_id, filename)
            )
            t.daemon = True
            t.start()
            response_data = {"success": True, "job_id": job_id, "message": "Batch quiz generation task spawned successfully."}


        return response_data

