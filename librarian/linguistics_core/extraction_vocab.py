"""
Deterministic academic vocabulary extraction facade mixin.
"""

from .extraction_vocab_candidates import ExtractionVocabCandidatesMixin
from .extraction_vocab_examples import ExtractionVocabExamplesMixin
from .extraction_vocab_extractor import ExtractionVocabExtractorMixin


class ExtractionVocabMixin(
    ExtractionVocabCandidatesMixin,
    ExtractionVocabExamplesMixin,
    ExtractionVocabExtractorMixin,
):
    """
    Unified facade mixin composing academic vocabulary candidate mining,
    distinct authentic example selection, and extraction execution.
    """
    pass


__all__ = [
    "ExtractionVocabCandidatesMixin",
    "ExtractionVocabExamplesMixin",
    "ExtractionVocabExtractorMixin",
    "ExtractionVocabMixin",
]
