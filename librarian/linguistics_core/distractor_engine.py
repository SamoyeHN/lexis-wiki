"""Unified Distractor Engine facade composing modular domain mixins."""

from __future__ import annotations

from .distractor_guards import DistractorGuardsMixin
from .distractor_paronyms import DistractorParonymsMixin
from .distractor_phrases import DistractorPhrasesMixin
from .distractor_inflections import DistractorInflectionsMixin
from .distractor_fallbacks import DistractorFallbacksMixin
from .distractor_vocab import DistractorVocabMixin


class DistractorEngineMixin(
    DistractorGuardsMixin,
    DistractorParonymsMixin,
    DistractorPhrasesMixin,
    DistractorInflectionsMixin,
    DistractorFallbacksMixin,
    DistractorVocabMixin,
):
    """Unified Distractor Engine Facade Mixin.

    Composes specialized distractor mixins while preserving 100% backward-compatible class interface:
    - DistractorGuardsMixin: Syntactic slot clash, double-key collision, countability contrast & light verb checks
    - DistractorParonymsMixin: Orthographic paronym generation, morpho-phonetic distance & semantic fields
    - DistractorPhrasesMixin: Multi-word phrase distractors, idiom mapping & phrase normalization
    - DistractorInflectionsMixin: Inflection screening, voice tags & target reconciliation
    - DistractorFallbacksMixin: Antonym/synonym retrieval, dynamic corpus fallback & construction contrast
    - DistractorVocabMixin: Generative vocabulary distractor orchestrator
    """
    pass


__all__ = [
    "DistractorEngineMixin",
    "DistractorGuardsMixin",
    "DistractorParonymsMixin",
    "DistractorPhrasesMixin",
    "DistractorInflectionsMixin",
    "DistractorFallbacksMixin",
    "DistractorVocabMixin",
]
