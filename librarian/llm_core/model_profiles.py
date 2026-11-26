# -*- coding: utf-8 -*-
"""Model capability profiles and execution configuration."""

from typing import Any, Dict, List, Optional
from librarian.config import config


class LLMError(Exception):
    """Base exception for LLM related errors."""
    pass



MODEL_CAPABILITY_PROFILES = [
    {
        # Specialized Decision / Classification / Guardrail models (Ollama 'decision' family):
        # Single-pass or fast typed classification, routing, WSD arbitration, and groundedness scoring.
        # Must run with GBNF and CoT thinking disabled for sub-second classification and to prevent
        # grammar-masking crashes on encoder/classifier heads (e.g. laya/ModernBERT, kev, tev1, nimble, clef).
        "match": [
            "kev",
            "nimble",
            "strands-decider",
            "clef",
            "laya",
            "tev1",
            "guardian",
            "shieldgemma",
            "llama-guard",
        ],
        "role": "classifier",
        "enforce_gbnf": False,
        "enforce_json_format": False,
        "think": False,
    },
    {
        # Granite family (e.g. granite4.2:30b): Prompt-guided JSON mode with thinking disabled
        # Avoids GBNF grammar token-masking repetition loops and reasoning stalls while maintaining schema-compliant output
        "match": ["granite"],
        "enforce_gbnf": False,
        "think": False,
    },
    {
        # Dedicated reasoning models: Native strict schema, disable reasoning tokens during structured JSON extraction
        "match": ["deepseek-r1", "qwq", "ornith"],
        "enforce_gbnf": True,
        "think": False,
    },
    {
        # Specialized translation checkpoints (e.g. translategemma):
        # Setting API format: "json" causes translategemma to emit empty '{}'
        # Follows prompt-injected JSON schema natively without the Ollama format constraint.
        "match": ["translategemma"],
        "enforce_gbnf": False,
        "enforce_json_format": False,
        "think": False,
    },
    {
        # Standard Gemma family (e.g. gemma2, gemma4):
        # Fully compatible with prompt-guided JSON mode (format: "json")
        "match": ["gemma"],
        "enforce_gbnf": False,
        "enforce_json_format": True,
        "think": False,
    },
    {
        # Tokenizer-constrained or large-vocab families (Nemotron, Phi, Qwen, GPT-OSS): Prompt-guided JSON mode
        # Eliminates CPU-bound GBNF token-masking bottlenecks across large tokenizers (131k/200k/248k) and reasoning stalls while maintaining 100% schema fidelity
        "match": ["nemotron", "phi", "qwen", "gpt-oss"],
        "enforce_gbnf": False,
        "think": False,
    },
    {
        # Muse family (e.g., muse-glimmer): Prompt-guided JSON mode with thinking disabled
        # Avoids 300s+ hidden reasoning loops and prompt regurgitation while preserving in-schema design_audit.
        "match": ["muse"],
        "enforce_gbnf": False,
        "think": False,
    },
    {
        # Fragile / Prompt-based families (Hermes custom tags, Vicuna legacy attention):
        # Hermes uses native XML/markdown tags (<tool_call>, <thought>) which crash under rigid GBNF clamps.
        # Vicuna lacks modern attention calibration for strict character-level grammar masking.
        "match": ["hermes", "vicuna"],
        "enforce_gbnf": False,
        "think": False,
    },
    {
        # Inkling family (e.g. inkling-2-mini): Looped-layer model with default 'high' thinking
        # Requires think=False to prevent context/token exhaustion, temperature=0.7 to avoid
        # looped resonance limits, and enforce_gbnf=True to guarantee strict schema adherence.
        "match": ["inkling"],
        "enforce_gbnf": True,
        "enforce_json_format": True,
        "think": False,
        "temperature": 0.7,
    },
    {
        # Native GBNF / strict grammar standard families (Llama 3+, Mistral, Mixtral, Codestral):
        # Perfectly compatible with context-free grammar parsers (llama.cpp, vLLM, SGLang).
        "match": ["llama", "mistral", "mixtral", "codestral"],
        "enforce_gbnf": True,
        "think": False,
    },
]



def get_model_profile(model_name: str) -> dict:
    """
    Intelligently determines optimal execution profile for structured output via a 3-tier precedence hierarchy:
    - Tier 1 (Base): Declarative architectural heuristics (Decision/Classifier, Granite, Qwen, Gemma, Llama, Hermes, etc.)
    - Tier 2 (Global): Global settings in wiki_config.json (e.g. global enforce_gbnf)
    - Tier 3 (Override): Explicit per-model overrides in wiki_config.json['model_options'][model_name]

    Default Safe Baseline for unknown / newly introduced models:
    - Prompt-guided JSON mode (enforce_gbnf=False, enforce_json_format=True)
    - Thinking OFF (think=False) to prevent token exhaustion and multi-minute reasoning stalls.
    """
    m_lower = (model_name or "").lower()
    
    # Tier 1: Base architectural profile matching (Safe default baseline)
    base_gbnf = False
    base_think = False
    base_json_format = True
    base_role = "generator"
    matched_rule = {}
    for profile_rule in MODEL_CAPABILITY_PROFILES:
        if any(keyword in m_lower for keyword in profile_rule["match"]):
            matched_rule = profile_rule
            base_gbnf = profile_rule.get("enforce_gbnf", False)
            base_think = profile_rule.get("think", False)
            base_json_format = profile_rule.get("enforce_json_format", True)
            base_role = profile_rule.get("role", "generator")
            if base_gbnf is True and ("thinking" in m_lower or "reasoning" in m_lower):
                base_think = True
            break

    # Tier 2: Global config settings (if explicitly enforced, except for dedicated classifiers)
    if config.data.get("enforce_gbnf") is True and base_role != "classifier":
        base_gbnf = True

    # Tier 3: Per-model explicit overrides (Highest Precedence)
    model_opts = config.get("model_options", {}).get(model_name, {})
    resolved_gbnf = model_opts.get("enforce_gbnf", base_gbnf)
    resolved_think = model_opts.get("think", base_think)
    resolved_json_format = model_opts.get("enforce_json_format", base_json_format)
    resolved_role = model_opts.get("role", base_role)

    profile = {k: v for k, v in matched_rule.items() if k not in ("match",)}
    profile.update(model_opts)
    profile["role"] = resolved_role
    profile["enforce_gbnf"] = resolved_gbnf
    profile["enforce_json_format"] = resolved_json_format
    if resolved_think is not None:
        profile["think"] = resolved_think

    return profile

