"""
Discourse Reading Assessment Blueprints and Psychometric Code Gates Facade.
"""

from .quiz_reading_blueprints import QuizReadingBlueprintsMixin
from .quiz_reading_evaluator import QuizReadingEvaluatorMixin
from .quiz_reading_graph import QuizReadingGraphMixin
from .quiz_reading_models import (
    CAUSAL_CONNECTIVES_REGEX,
    CEFRDifficultyTier,
    CONTRAST_CONNECTIVES_REGEX,
    DiscourseGraph,
    DiscourseItemBlueprint,
    DiscourseRelationType,
    DiscourseSection,
    DistractorFallacyOperator,
    INFERENCE_TRIGGER_REGEX,
    MODALITY_HEDGES_REGEX,
    PrescribedDistractor,
    ReadingDistractorSpec,
    ReadingEvaluationResult,
    ReadingItemBlueprint,
)


class QuizReadingMixin(
    QuizReadingGraphMixin,
    QuizReadingBlueprintsMixin,
    QuizReadingEvaluatorMixin,
):
    """
    Unified facade composing discourse graph extraction, blueprint generation,
    and Level 1 deterministic code gate mixins.
    """
    pass


class DiscourseReadingEngine(QuizReadingMixin):
    """
    Standalone class alias for direct imports.
    """
    pass


__all__ = [
    "CAUSAL_CONNECTIVES_REGEX",
    "CEFRDifficultyTier",
    "CONTRAST_CONNECTIVES_REGEX",
    "DiscourseGraph",
    "DiscourseItemBlueprint",
    "DiscourseReadingEngine",
    "DiscourseRelationType",
    "DiscourseSection",
    "DistractorFallacyOperator",
    "INFERENCE_TRIGGER_REGEX",
    "MODALITY_HEDGES_REGEX",
    "PrescribedDistractor",
    "QuizReadingBlueprintsMixin",
    "QuizReadingEvaluatorMixin",
    "QuizReadingGraphMixin",
    "QuizReadingMixin",
    "ReadingDistractorSpec",
    "ReadingEvaluationResult",
    "ReadingItemBlueprint",
]
