"""Evaluator core modular sub-package for Lexis Wiki.

Decomposed from evaluator.py:
- eval_helpers: Normalization, token variants, extraction and grammar formulas.
- dimension_scorers: Four-dimension deterministic code gate scorers (_score_*).
- pruning: In-place hallucination pruning and quote magnetic snapping.
- log_auditor: LogEvaluator for batch execution log audit and hero board metrics.
"""

from .dimension_scorers import (
    _score_pedagogy,
    _score_schema,
    _score_uniqueness,
    _score_verbatim,
)
from .eval_helpers import (
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
    auto_remap_grammar_category,
    canonical_formula_key,
    degenerate_grammar_slots,
)
from .log_auditor import LogEvaluator
from .pruning import prune_hallucinated_items

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
