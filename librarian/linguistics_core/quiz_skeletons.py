"""
Atomic Pattern-Example-Anchor binding and precomputed target skeleton blueprints for downstream quizzes.
Facade mixin composing specialized modular mixins:
- QuizPatternsMixin: Five-source authentic example selection & pattern bundles
- QuizFramesMixin: Verb patient ontology & cloze slot frame derivation
- QuizVocabMixin: Generative vocabulary quiz target skeletons & authentic cloze items
- QuizTranslationMixin: Translation micro-task blueprints & lexical/syntactic traps
- QuizListeningMixin: Functional dialogue listening archetypes & blueprints
"""

from .quiz_patterns import QuizPatternsMixin
from .quiz_frames import QuizFramesMixin
from .quiz_vocab import QuizVocabMixin
from .quiz_translation import QuizTranslationMixin
from .quiz_listening import QuizListeningMixin


class QuizSkeletonsMixin(
    QuizPatternsMixin,
    QuizFramesMixin,
    QuizVocabMixin,
    QuizTranslationMixin,
    QuizListeningMixin,
):
    """
    Unified Quiz Skeletons Mixin facade.
    Composes modular domain mixins while preserving 100% backward-compatible class interface.
    """
    pass


__all__ = [
    "QuizSkeletonsMixin",
    "QuizPatternsMixin",
    "QuizFramesMixin",
    "QuizVocabMixin",
    "QuizTranslationMixin",
    "QuizListeningMixin",
]
