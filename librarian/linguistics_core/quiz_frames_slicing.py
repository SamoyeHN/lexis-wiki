"""
Cloze slot frame derivation, collocation half-slice, and clause isolation facade.
"""

from .quiz_frames_clause_isolation import QuizFramesClauseIsolationMixin
from .quiz_frames_slot_slicing import QuizFramesSlotSlicingMixin


class QuizFramesSlicingMixin(
    QuizFramesSlotSlicingMixin,
    QuizFramesClauseIsolationMixin,
):
    """
    Unified facade composing cloze slot frame derivation,
    collocation half-slicing, and clause isolation mixins.
    """
    pass


__all__ = [
    "QuizFramesClauseIsolationMixin",
    "QuizFramesSlicingMixin",
    "QuizFramesSlotSlicingMixin",
]
