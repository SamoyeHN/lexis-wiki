"""
Generative vocabulary quiz skeletons and authentic cloze item synthesis facade.
Composes modular domain mixins for authentic cloze and target skeleton synthesis.
"""

from .quiz_vocab_cloze import QuizVocabClozeMixin
from .quiz_skeleton_prep import QuizSkeletonPrepMixin
from .quiz_skeleton_anchors import QuizSkeletonAnchorsMixin
from .quiz_skeleton_distractors import QuizSkeletonDistractorsMixin
from .quiz_skeleton_frames import QuizSkeletonFramesMixin
from .quiz_skeleton_builder import QuizSkeletonBuilderMixin


class QuizVocabMixin(
    QuizVocabClozeMixin,
    QuizSkeletonPrepMixin,
    QuizSkeletonAnchorsMixin,
    QuizSkeletonDistractorsMixin,
    QuizSkeletonFramesMixin,
    QuizSkeletonBuilderMixin,
):
    """Facade composing all vocabulary cloze and generative skeleton mixins."""
    pass
