# -*- coding: utf-8 -*-
"""Unified LLMClient orchestrating lifecycle, backends, JSON repair, and chat execution."""

from typing import Any, Dict, Optional

from .backends import LLMBackendsMixin
from .chat_executor import ChatExecutorMixin
from .json_repair import JsonRepairMixin
from .model_lifecycle import ModelLifecycleMixin
from .model_profiles import LLMError


class LLMClient(ModelLifecycleMixin, JsonRepairMixin, LLMBackendsMixin, ChatExecutorMixin):
    """Unified client for local and remote LLM execution."""

    def __init__(self, model: Optional[str] = None):
        self._model_override = model
        self._health_cache: Optional[Dict[str, Any]] = None
        self._health_cache_ts: float = 0.0
        self._refresh_config()
        if model:
            self.model = model
        self.last_raw_response = None
        self.last_done_reason = None
        self.last_log_path = None

