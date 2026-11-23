"""Unified WSD Engine facade composing modular domain mixins."""

from __future__ import annotations

from .wsd_gates import WsdGatesMixin
from .wsd_wordnet import WsdWordNetMixin
from .wsd_scoring import WsdContext, WsdScoringMixin
from .wsd_lock import WsdLockMixin
from .wsd_definitions import WsdDefinitionsMixin
from .wsd_expressions import WsdExpressionsMixin
from .wsd_arbitration import WsdArbitrationMixin


class WsdEngineMixin(
    WsdGatesMixin,
    WsdWordNetMixin,
    WsdScoringMixin,
    WsdLockMixin,
    WsdDefinitionsMixin,
    WsdExpressionsMixin,
    WsdArbitrationMixin,
):
    """Unified Word Sense Disambiguation Facade Mixin.

    Composes specialized WSD domain mixins while preserving 100% backward-compatible class interface:
    - WsdGatesMixin: POS normalization, short codes, pattern frame matching, and slash expansion
    - WsdWordNetMixin: WordNet supersenses, synset disambiguation, and sense cue index lookups
    - WsdScoringMixin: Fine-grained syntactic/semantic scoring across individual sense candidates
    - WsdLockMixin: Two-stage neuro-symbolic WSD sense locking orchestrator and confidence evaluation
    - WsdDefinitionsMixin: Definition formatting, boilerplate detection, and authentic example extraction
    - WsdExpressionsMixin: Multi-word unit expression cascade, unit senses, and collocation definitions
    - WsdArbitrationMixin: WSD session logging and Top-K candidate 1-token local LLM arbitration
    """
    pass


__all__ = [
    "WsdEngineMixin",
    "WsdGatesMixin",
    "WsdWordNetMixin",
    "WsdContext",
    "WsdScoringMixin",
    "WsdLockMixin",
    "WsdDefinitionsMixin",
    "WsdExpressionsMixin",
    "WsdArbitrationMixin",
]
