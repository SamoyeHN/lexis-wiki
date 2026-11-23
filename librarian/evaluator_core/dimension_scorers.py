"""Level 1 deterministic code gate scoring heuristics across 4 pedagogical quality dimensions:

1. Schema & Structural Adherence (W_SCHEMA = 25.0) -> scorer_schema.py
2. Source Faithfulness & Verbatim Verification (W_VERBATIM = 30.0) -> scorer_verbatim.py
3. Pedagogical & Slot-Filling Quality (W_PEDAGOGY = 25.0) -> scorer_pedagogy.py
4. Uniqueness & Deduplication (W_UNIQUENESS = 20.0) -> scorer_uniqueness.py
"""

from __future__ import annotations

from .scorer_pedagogy import _score_pedagogy
from .scorer_schema import _score_schema
from .scorer_uniqueness import _score_uniqueness
from .scorer_verbatim import _score_verbatim

__all__ = [
    "_score_pedagogy",
    "_score_schema",
    "_score_uniqueness",
    "_score_verbatim",
]
