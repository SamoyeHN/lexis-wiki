"""Lexis Wiki Level 1 Deterministic Evaluator & Quality Gate Facade.

Composes:
- evaluator_core.eval_helpers: Normalization, token variants, extraction and grammar formulas.
- evaluator_core.dimension_scorers: Four-dimension deterministic code gate scorers (_score_*).
- evaluator_core.pruning: In-place hallucination pruning and quote magnetic snapping.
- evaluator_core.log_auditor: LogEvaluator for batch execution log audit and hero board metrics.
"""

from __future__ import annotations

# Re-export four-dimension scorers
from .evaluator_core.dimension_scorers import (
    _score_pedagogy,
    _score_schema,
    _score_uniqueness,
    _score_verbatim,
)

# Re-export constants and helpers
from .evaluator_core.eval_helpers import (
    ALLOWED_GRAMMAR_SLOTS,
    COBUILD_POS_TOKENS,
    COMMON_IRREGULARS,
    FATAL_QA_FLAGS,
    GRAMMAR_SLOT_IDENTITIES,
    ITEM_KEYS,
    MIN_SOURCE_WORDS_FOR_CEILING,
    STOP_SLOTS,
    VALID_EXPR_POS_SET,
    VALID_POS_SET,
    W_PEDAGOGY,
    W_SCHEMA,
    W_UNIQUENESS,
    W_VERBATIM,
    _batch_target_conflict,
    _cefr_source_ceiling,
    _clean_core,
    _detect_task_type,
    _expand_contractions,
    _extract_items,
    _extract_json,
    _extract_source_content,
    _extract_target_skeletons,
    _extract_wordlist,
    _form_in_text,
    _formula_slot_identity,
    _inflected_variants_of,
    _is_hallucinated_quote,
    _ngram_coverage,
    _normalize_text,
    _safe_str,
    _source_token_variants,
    _wordlist_matches,
    auto_remap_grammar_category,
    canonical_formula_key,
    degenerate_grammar_slots,
)

# Re-export LogEvaluator
from .evaluator_core.log_auditor import LogEvaluator

# Re-export pruning gate
from .evaluator_core.pruning import prune_hallucinated_items

__all__ = [
    "LogEvaluator",
    "auto_remap_grammar_category",
    "prune_hallucinated_items",
    "canonical_formula_key",
    "degenerate_grammar_slots",
    "FATAL_QA_FLAGS",
    "W_SCHEMA",
    "W_VERBATIM",
    "W_PEDAGOGY",
    "W_UNIQUENESS",
    "_score_schema",
    "_score_verbatim",
    "_score_pedagogy",
    "_score_uniqueness",
]
