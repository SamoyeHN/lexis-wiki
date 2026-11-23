"""
Verb patient ontology classification and cloze slot frame derivation facade.
"""

from .quiz_frames_patient import QuizFramesPatientMixin
from .quiz_frames_slicing import QuizFramesSlicingMixin


class QuizFramesMixin(QuizFramesPatientMixin, QuizFramesSlicingMixin):
    """
    Unified facade mixin composing verb patient ontology classification
    and cloze slot frame derivation.
    """
    pass


__all__ = [
    "QuizFramesMixin",
    "QuizFramesPatientMixin",
    "QuizFramesSlicingMixin",
]
