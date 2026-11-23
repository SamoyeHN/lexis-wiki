"""
Unified Extraction Pipeline facade composing source hygiene, grammar, expression, and vocabulary mixins.
"""

from .extraction_source import ExtractionSourceMixin
from .extraction_grammar import ExtractionGrammarMixin
from .extraction_expressions import ExtractionExpressionsMixin
from .extraction_vocab import (
    ExtractionVocabCandidatesMixin,
    ExtractionVocabExamplesMixin,
    ExtractionVocabExtractorMixin,
    ExtractionVocabMixin,
)

__all__ = [
    "ExtractionExpressionsMixin",
    "ExtractionGrammarMixin",
    "ExtractionPipelineMixin",
    "ExtractionSourceMixin",
    "ExtractionVocabCandidatesMixin",
    "ExtractionVocabExamplesMixin",
    "ExtractionVocabExtractorMixin",
    "ExtractionVocabMixin",
]


class ExtractionPipelineMixin(
    ExtractionSourceMixin,
    ExtractionGrammarMixin,
    ExtractionExpressionsMixin,
    ExtractionVocabMixin,
):
    """
    Unified Extraction Pipeline facade.
    Composes specialized source hygiene, grammar, multi-word expression,
    and vocabulary extraction domain mixins while preserving 100% backward compatibility.
    """
    pass
