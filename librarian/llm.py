# -*- coding: utf-8 -*-
"""LLM Client Facade for local and remote LLM orchestration.

Backward-compatible facade delegating to modular submodules in librarian.llm_core.
"""

from .llm_core import (
    LLMClient,
    LLMError,
    MODEL_CAPABILITY_PROFILES,
    _is_real_string_closer,
    _prune_to_schema,
    get_model_profile,
)

# Global singleton client instance
llm = LLMClient()

__all__ = [
    "LLMClient",
    "LLMError",
    "MODEL_CAPABILITY_PROFILES",
    "get_model_profile",
    "_prune_to_schema",
    "_is_real_string_closer",
    "llm",
]
