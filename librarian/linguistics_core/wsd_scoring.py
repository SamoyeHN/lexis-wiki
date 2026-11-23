"""Per-sense scoring engine for neuro-symbolic WSD sense locking (Facade)."""

from __future__ import annotations

from typing import Any, Dict

from .wsd_scoring_context import WsdContext, WsdSenseInfo
from .wsd_scoring_gates import WsdScoringGatesMixin
from .wsd_scoring_patterns import WsdScoringPatternsMixin
from .wsd_scoring_dependencies import WsdScoringDependenciesMixin


class WsdScoringMixin(WsdScoringGatesMixin, WsdScoringPatternsMixin, WsdScoringDependenciesMixin):
    """Fine-grained syntactic and semantic scoring across individual sense candidates."""

    @classmethod
    def _score_single_sense(cls, s: Dict[str, Any], i: int, ctx: WsdContext) -> float:
        score, s_info = cls._score_sense_gates_and_lexical(s, i, ctx)
        score = cls._score_sense_patterns_and_units(s, i, ctx, s_info, score)
        score = cls._score_sense_dependencies_and_cues(s, i, ctx, s_info, score)
        return float(score)


__all__ = [
    "WsdContext",
    "WsdSenseInfo",
    "WsdScoringMixin",
    "WsdScoringGatesMixin",
    "WsdScoringPatternsMixin",
    "WsdScoringDependenciesMixin",
]
