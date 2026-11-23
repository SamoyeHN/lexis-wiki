"""Dimension 1: Schema & Structural Adherence (W_SCHEMA = 25.0)."""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

from .eval_helpers import W_SCHEMA


def _score_schema(parsed: Any, raw_response: str = "", task_type: str = "") -> Tuple[Optional[float], List[str]]:
    """Dimension 1 (0–25). Always applicable."""
    if isinstance(parsed, dict):
        if len(parsed) == 0:
            return 0.0, ["❌ Valid JSON but empty object"]

        flags = []
        deduction = 0.0
        if raw_response and isinstance(raw_response, str):
            trimmed = raw_response.strip()
            # If closing brace was missing or raw string didn't end properly
            if trimmed.count("{") > trimmed.count("}") or trimmed.count("[") > trimmed.count("]"):
                deduction += 5.0
                flags.append("⚠️ Structural repair: unclosed braces/brackets in raw output")

        # Array element schema & type enforcement
        ARRAY_SPECS = {
            "mindmap": ("branches", 3, 5, "branch_name"),
            "vocabulary": ("vocabulary", 1, 50, "word"),
            "expressions": ("expressions", 1, 50, "word"),
            "grammar": ("grammar_patterns", 1, 20, "pattern_formula"),
            "summary": ("concepts", 1, 10, "concept_name"),
            "quiz": ("questions", 1, 50, "question"),
        }
        if task_type in ARRAY_SPECS:
            arr_key, min_items, max_items, required_field = ARRAY_SPECS[task_type]
            arr_val = parsed.get(arr_key)
            if not isinstance(arr_val, list):
                deduction += 15.0
                flags.append(f"❌ [INVALID_SCHEMA] Expected list for '{arr_key}', got {type(arr_val).__name__}")
            else:
                non_dict_count = sum(1 for el in arr_val if not isinstance(el, dict))
                if non_dict_count > 0:
                    deduction += 15.0
                    flags.append(f"❌ [INVALID_ITEM_TYPE] {non_dict_count}/{len(arr_val)} elements in '{arr_key}' are not objects (e.g. unparsed JSON strings)")
                valid_dict_count = len(arr_val) - non_dict_count
                if valid_dict_count < min_items:
                    deduction += 10.0
                    flags.append(f"❌ [INSUFFICIENT_ITEMS] '{arr_key}' contains only {valid_dict_count} valid objects (minimum required: {min_items})")

        score = max(0.0, W_SCHEMA - deduction)
        return score, flags
    return 0.0, ["❌ Invalid or missing JSON output"]
