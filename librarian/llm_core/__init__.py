# -*- coding: utf-8 -*-
"""LLM Core package: modular submodules for LLM client orchestration."""

from .client import LLMClient
from .json_repair import _is_real_string_closer, _prune_to_schema
from .model_profiles import LLMError, MODEL_CAPABILITY_PROFILES, get_model_profile

__all__ = [
    "LLMClient",
    "LLMError",
    "MODEL_CAPABILITY_PROFILES",
    "get_model_profile",
    "_prune_to_schema",
    "_is_real_string_closer",
]
