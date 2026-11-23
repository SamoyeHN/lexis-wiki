"""
Verb patient ontology classification and cloze slot frame derivation facade.
"""

from .quiz_frames_patient import QuizFramesPatientMixin
from .quiz_frames_slicing import (
    QuizFramesClauseIsolationMixin,
    QuizFramesSlicingMixin,
    QuizFramesSlotSlicingMixin,
)


class QuizFramesMixin(QuizFramesPatientMixin, QuizFramesSlicingMixin):
    """
    Unified facade mixin composing verb patient ontology classification
    and cloze slot frame derivation.
    """
    pass


__all__ = [
    "QuizFramesClauseIsolationMixin",
    "QuizFramesMixin",
    "QuizFramesPatientMixin",
    "QuizFramesSlicingMixin",
    "QuizFramesSlotSlicingMixin",
]
