# -*- coding: utf-8 -*-
"""Model lifecycle, health check, and endpoint management mixin."""

import logging
import time
from typing import Any, Dict, List, Optional
import requests

from librarian.config import config
from .model_profiles import LLMError

logger = logging.getLogger("librarian.llm")


class ModelLifecycleMixin:
    """Mixin managing LLM configuration, health checks, model listing, and memory unloading."""

    def _refresh_config(self):
        """Refreshes configuration from the config object."""
        raw_type = config.get("api_type") or "ollama"
        self.api_url = (config.get("api_url") or config.get("ollama_url") or "http://localhost:11434").strip().rstrip('/')
        # Auto-heal: If configured endpoint is on Ollama's standard port (11434) and api_type was set to 'openai',
        # auto-switch to native 'ollama' to guarantee correct options (think: false, format: json) are dispatched.
        if raw_type == "openai" and ":11434" in self.api_url:
            self.api_type = "ollama"
        else:
            self.api_type = raw_type
        self.api_key = config.get("api_key") or "ollama"
        self.model = config.get("model")
        # Default read timeout: 300 seconds (5 mins) aligned with reverse-proxy (Nginx/NPM) read timeouts
        self.timeout = config.get("request_timeout", 300)

    def list_models(self, api_type=None, api_url=None, api_key=None):
        """Fetches models from Ollama (/api/tags) or OpenAI-compatible servers like llama-server (/v1/models).

        Optional overrides (api_type / api_url / api_key) allow previewing a different
        engine/endpoint than the one currently saved in config. The dashboard uses this
        when the user switches the API Engine dropdown before saving, so the model list
        reflects the selected engine instead of silently staying on the saved one.
        Falls back to the saved config when an override is not provided.
        """
        self._refresh_config()
        api_type = api_type or self.api_type
        api_url = (api_url or self.api_url).strip().rstrip('/')
        api_key = api_key if api_key else self.api_key
        # Surface reachability so the dashboard can distinguish "server offline"
        # (request failed) from "server online but lists no models".
        self.last_models_ok = False
        self.last_models_error = None
        if api_type == "ollama":
            url = f"{api_url}/api/tags"
            try:
                response = requests.get(url, timeout=5)
                response.raise_for_status()
                data = response.json()
                self.last_models_ok = True
                return [m["name"] for m in data.get("models", [])]
            except Exception as e:
                self.last_models_error = str(e)
                return []
        elif api_type == "openai":
            url = f"{api_url}/v1/models"
            headers = {}
            if api_key and api_key != "ollama":
                headers["Authorization"] = f"Bearer {api_key}"
            try:
                response = requests.get(url, headers=headers, timeout=5)
                response.raise_for_status()
                data = response.json()
                # Support OpenAI standard {"data": [{"id": "model_name"}]} and llama-server {"models": [...]}
                models = []
                if "data" in data and isinstance(data["data"], list):
                    models.extend([m.get("id") for m in data["data"] if m.get("id")])
                elif "models" in data and isinstance(data["models"], list):
                    models.extend([m.get("name") or m.get("model") for m in data["models"] if (m.get("name") or m.get("model"))])
                self.last_models_ok = True
                return models
            except Exception as e:
                self.last_models_error = str(e)
                return []
        return []

    def unload_model(self, model_name: Optional[str] = None):
        """Unloads a model from GPU memory on Ollama to prevent VRAM exhaustion."""
        target = model_name or self.model
        if not target:
            return
        self._refresh_config()
        if self.api_type == "ollama":
            url = f"{self.api_url}/api/generate"
            try:
                # keep_alive: 0 immediately evicts the model from VRAM
                requests.post(url, json={"model": target, "keep_alive": 0}, timeout=5)
                import logging
                logging.getLogger("librarian.llm").info(f"Unloaded model '{target}' from Ollama memory (keep_alive: 0).")
            except Exception as e:
                import logging
                logging.getLogger("librarian.llm").warning(f"Failed to unload model '{target}' from Ollama: {e}")

    @staticmethod
    def _normalize_model_id(name: str) -> str:
        s = str(name or "").strip().lower()
        for ch in (":", "/", "-", "_", "."):
            s = s.replace(ch, "")
        return s

    def check_health(self, force: bool = False, model_override: Optional[str] = None) -> Dict[str, Any]:
        """Probes LLM endpoint reachability and active model availability with a 3s TTL cache."""
        import time
        now = time.monotonic()
        self._refresh_config()
        if self._model_override:
            self.model = self._model_override
        active_model = str(model_override or self.model or "").strip()
        if (
            not force
            and self._health_cache is not None
            and (now - self._health_cache_ts) < 3.0
            and self._health_cache.get("api_url") == self.api_url
            and self._health_cache.get("active_model") == active_model
        ):
            return dict(self._health_cache)

        models = self.list_models()
        online = bool(getattr(self, "last_models_ok", False))
        model_ready = False
        if online and active_model:
            norm_active = self._normalize_model_id(active_model)
            for m in models:
                if m == active_model or self._normalize_model_id(m) == norm_active:
                    model_ready = True
                    break
                if ":" not in active_model and str(m).split(":")[0].lower() == active_model.lower():
                    model_ready = True
                    break

        err = getattr(self, "last_models_error", None)
        if online and not active_model:
            err = "No active LLM model configured in wiki_config.json"
        elif online and not model_ready:
            err = f"Configured model '{active_model}' is not available on {self.api_url} (available: {', '.join(models[:5]) or 'none'})"

        health = {
            "online": online,
            "model_ready": model_ready,
            "model": active_model,
            "active_model": active_model,
            "available_models": models,
            "api_type": self.api_type,
            "api_url": self.api_url,
            "error": err,
        }
        self._health_cache = health
        self._health_cache_ts = now
        return dict(health)

    def ensure_available(
        self,
        task_name: Optional[str] = None,
        prompt_summary: str = "",
        model_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Pre-flight fail-fast gate: verifies LLM is online & model is ready; otherwise writes a FAILED log and raises LLMError."""
        health = self.check_health(force=True, model_override=model_override)
        if health["online"] and health["model_ready"]:
            return health

        if not health["online"]:
            failure_category = "LLM_OFFLINE"
            err_msg = (
                f"LLM service is unreachable at {health['api_url']} ({health['api_type']}). "
                f"Please start Ollama / LLM server before running this task. Details: {health.get('error') or 'Connection refused'}"
            )
        else:
            failure_category = "MODEL_NOT_FOUND"
            err_msg = (
                f"LLM server at {health['api_url']} is online, but configured model '{health['active_model']}' "
                f"is not loaded or installed. Details: {health.get('error')}"
            )

        if task_name:
            try:
                from ..logger import log_task
                log_task(
                    task_name=task_name,
                    system_prompt="",
                    user_prompt=prompt_summary or f"[Pre-Flight Gate] Task '{task_name}' requires active LLM '{health['active_model']}' at {health['api_url']}.",
                    response_text=f"[LLM_PREFLIGHT_FAILED] {err_msg}",
                    status="FAILED",
                    failure_category=failure_category,
                    mode="PREFLIGHT_GATE",
                    duration=0.0,
                    model=health["active_model"] or "unknown",
                )
            except Exception as log_err:
                import logging
                logging.getLogger("librarian").warning(f"Failed to write preflight failure log: {log_err}")

        raise LLMError(err_msg)

