"""
Discourse Reading Prototype Engine (Neuro-Symbolic Discourse Reading Assessment)
================================================================================
Backward-compatible wrapper re-exporting production classes from quiz_reading.py.
"""

from .quiz_reading import (
    CEFRDifficultyTier,
    DiscourseGraph,
    DiscourseItemBlueprint,
    DiscourseReadingEngine,
    DiscourseRelationType,
    DistractorFallacyOperator,
    PrescribedDistractor,
    QuizReadingMixin,
    ReadingDistractorSpec,
    ReadingEvaluationResult,
    ReadingItemBlueprint,
)

__all__ = [
    "QuizReadingMixin",
    "DiscourseReadingEngine",
    "DiscourseRelationType",
    "CEFRDifficultyTier",
    "DistractorFallacyOperator",
    "ReadingDistractorSpec",
    "PrescribedDistractor",
    "ReadingItemBlueprint",
    "DiscourseItemBlueprint",
    "DiscourseGraph",
    "ReadingEvaluationResult",
]
