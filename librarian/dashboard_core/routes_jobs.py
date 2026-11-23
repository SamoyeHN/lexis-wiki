"""Background job execution, compilation, and re-audit route handlers."""

from __future__ import annotations

import json
import os
import threading
import uuid
from pathlib import Path
from typing import Any, Dict

from ..config import config, normalize_name
from ..llm import llm


class RoutesJobsMixin:
    """Mixin providing compile jobs, re-audit, and background job polling routes."""

    def _api_get_recent_jobs(self, query_str: str) -> Dict[str, Any]:
        response_data: Dict[str, Any] = {"success": True}
        response_data = {
            "jobs": self._dashboard_state().get_recent_jobs(),
            "llm_health": llm.check_health(force=False),
        }


        return response_data

    def _api_get_hero_board(self, query_str: str) -> Dict[str, Any]:
        response_data: Dict[str, Any] = {"success": True}
        from ..evaluator import LogEvaluator
        response_data = LogEvaluator.audit_all_logs()


        return response_data

    def _api_post_compile(self, body: str, body_bytes: bytes, is_binary: bool) -> Dict[str, Any]:
        response_data: Dict[str, Any] = {"success": True}
        data = json.loads(body)
        filename = data.get("filename")
        categories = data.get("categories") # list of strings
        if not filename:
            response_data = {"success": False, "error": "Filename is required"}
        else:
            job_name = f"{filename} ({', '.join(categories)})" if categories else filename
            job_id = self._dashboard_state().create_job("compile", job_name)
            # Start asynchronous background compiler with categories list
            t = threading.Thread(target=self._background_compile_worker(), args=(job_id, filename, categories))
            t.daemon = True
            t.start()
            response_data = {"success": True, "job_id": job_id, "message": "Compilation task spawned successfully."}


        return response_data

    def _api_post_re_audit(self, body: str, body_bytes: bytes, is_binary: bool) -> Dict[str, Any]:
        response_data: Dict[str, Any] = {"success": True}
        data = json.loads(body) if body else {}
        units = data.get("units") or None
        template = data.get("template") or None
        try:
            workers = int(data.get("workers") or 2)
        except (TypeError, ValueError):
            workers = 2
        job_name = f"Re-audit {len(units)} unit(s)" if units else "Re-audit ALL units"
        job_id = self._dashboard_state().create_job("audit-batch", job_name)
        t = threading.Thread(target=self._background_audit_batch_worker(), args=(job_id, units, template, workers))
        t.daemon = True
        t.start()
        response_data = {"success": True, "job_id": job_id, "message": "Re-audit task spawned successfully."}


        return response_data

    def _api_post_job_status(self, body: str, body_bytes: bytes, is_binary: bool) -> Dict[str, Any]:
        response_data: Dict[str, Any] = {"success": True}
        data = json.loads(body)
        job_id = data.get("job_id")
        job = self._dashboard_state().get_job(job_id)
        if not job:
            response_data = {"success": False, "error": "Job not found"}
        else:
            response_data = {"success": True, "job": job}


        return response_data

