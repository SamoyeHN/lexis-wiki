"""System, config, TTS, and health check route handlers."""

from __future__ import annotations

import json
import os
import urllib.parse
from pathlib import Path
from typing import Any, Dict

from ..config import config
from ..llm import llm


class RoutesSystemMixin:
    """Mixin providing system configuration, health check, and TTS routes."""

    def _api_get_shutdown(self, query_str: str) -> Dict[str, Any]:
        response_data: Dict[str, Any] = {"success": True}
        # Explicit shutdown signal (browser beforeunload beacon).
        self._dashboard_state().mark_shutdown()
        response_data = {"success": True, "message": "Dashboard server is shutting down."}


        return response_data

    def _api_get_config(self, query_str: str) -> Dict[str, Any]:
        response_data: Dict[str, Any] = {"success": True}
        from ..tts import tts_service
        config.load_config()
        tts_service._refresh()
        response_data = {
            "config": config.data,
            "tts_voices": tts_service.get_available_voices(),
            "project_root": str(config.project_root)
        }


        return response_data

    def _api_get_tts_voices(self, query_str: str) -> Dict[str, Any]:
        response_data: Dict[str, Any] = {"success": True}
        from ..tts import tts_service
        q = urllib.parse.parse_qs(query_str or "")
        engine = (q.get("engine", [None])[0] or None)
        force = str(q.get("refresh", ["0"])[0]).lower() in ("1", "true", "yes")
        response_data = {
            "success": True,
            "voices": tts_service.get_available_voices(engine=engine, force_refresh=force),
        }


        return response_data

    def _api_get_models(self, query_str: str) -> Dict[str, Any]:
        response_data: Dict[str, Any] = {"success": True}
        try:
            # Allow the dashboard to preview models for the engine SELECTED in the
            # dropdown (which may differ from the saved config until the user saves).
            q = urllib.parse.parse_qs(query_str or "")
            ov_api_type = (q.get("api_type", [None])[0] or None)
            ov_api_url = (q.get("api_url", [None])[0] or None)
            ov_api_key = (q.get("api_key", [None])[0] or None)
            models = llm.list_models(api_type=ov_api_type, api_url=ov_api_url, api_key=ov_api_key)
            resolved_api_type = ov_api_type or config.get("api_type")
            # "offline" = the engine endpoint could not be reached (distinct from a
            # reachable server that simply lists no models).
            offline = not getattr(llm, "last_models_ok", False)
            response_data = {
                "models": models,
                "active_model": config.get("model"),
                "api_type": resolved_api_type,
                "offline": offline,
                "error": getattr(llm, "last_models_error", None)
            }
        except Exception as e:
            response_data = {"error": str(e), "models": [], "active_model": config.get("model"),
                            "api_type": config.get("api_type"), "offline": True}


        return response_data

    def _api_get_llm_health(self, query_str: str) -> Dict[str, Any]:
        response_data: Dict[str, Any] = {"success": True}
        response_data = llm.check_health(force=True)


        return response_data

    def _api_get_logs(self, query_str: str) -> Dict[str, Any]:
        response_data: Dict[str, Any] = {"success": True}
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


        return response_data

    def _api_post_shutdown(self, body: str, body_bytes: bytes, is_binary: bool) -> Dict[str, Any]:
        response_data: Dict[str, Any] = {"success": True}
        # Explicit shutdown signal (browser beforeunload beacon via sendBeacon).
        self._dashboard_state().mark_shutdown()
        response_data = {"success": True, "message": "Dashboard server is shutting down."}


        return response_data

    def _api_post_config_update(self, body: str, body_bytes: bytes, is_binary: bool) -> Dict[str, Any]:
        response_data: Dict[str, Any] = {"success": True}
        data = json.loads(body)
        for key, val in data.items():
            success, msg = config.update_config(key, val)
            if not success:
                response_data = {"success": False, "error": msg}
                break
        response_data["message"] = "Configuration updated successfully."


        return response_data

    def _api_post_tts_preview(self, body: str, body_bytes: bytes, is_binary: bool) -> Dict[str, Any]:
        response_data: Dict[str, Any] = {"success": True}
        import base64
        from ..tts import tts_service
        data = json.loads(body) if body else {}
        engine = data.get("engine") or config.get("tts_engine") or "edge"
        voice = data.get("voice") or ""
        text = data.get("text") or None
        audio_bytes = tts_service.preview_voice(engine=engine, voice=voice, text=text)
        if audio_bytes:
            b64 = base64.b64encode(audio_bytes).decode("ascii")
            response_data = {
                "success": True,
                "audio_url": f"data:audio/mp3;base64,{b64}",
            }
        else:
            response_data = {
                "success": False,
                "error": f"Failed to synthesize preview for voice '{voice}' ({engine}).",
            }


        return response_data

    def _api_post_test_endpoint(self, body: str, body_bytes: bytes, is_binary: bool) -> Dict[str, Any]:
        response_data: Dict[str, Any] = {"success": True}
        response_data = {"success": True, "test": "working"}


        return response_data

