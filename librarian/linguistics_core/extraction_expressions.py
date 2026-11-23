"""
Deterministic multi-word expression and collocation extraction facade.
"""

from .extraction_expressions_candidates import ExtractionExpressionsCandidatesMixin
from .extraction_expressions_extractor import ExtractionExpressionsExtractorMixin
from .extraction_expressions_skeletons import ExtractionExpressionsSkeletonsMixin


class ExtractionExpressionsMixin(
    ExtractionExpressionsSkeletonsMixin,
    ExtractionExpressionsCandidatesMixin,
    ExtractionExpressionsExtractorMixin,
):
    """
    Unified facade composing multi-word expression mining, candidate selection,
    and deterministic extraction mixins.
    """
    pass


__all__ = [
    "ExtractionExpressionsCandidatesMixin",
    "ExtractionExpressionsExtractorMixin",
    "ExtractionExpressionsMixin",
    "ExtractionExpressionsSkeletonsMixin",
]
