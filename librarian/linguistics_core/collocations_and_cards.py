"""
Collocation extraction, zero-collision collocational anchor mining, and zero-token vocabulary card enrichment.
Unified facade mixin composing modular domain mixins.
"""

from .collocation_extractor import CollocationExtractorMixin
from .anchor_evidence import AnchorEvidenceMixin
from .vocab_card_enrichment import VocabCardEnrichmentMixin
from .zero_collision_anchors import ZeroCollisionAnchorsMixin


class CollocationsAndCardsMixin(
    CollocationExtractorMixin,
    AnchorEvidenceMixin,
    VocabCardEnrichmentMixin,
    ZeroCollisionAnchorsMixin,
):
    """Unified facade mixin for collocation extraction, anchor evidence, and vocabulary card enrichment."""

    # Backward-compatible exposure of class-level constants
    _COLLOCATION_GAP_WORDS = CollocationExtractorMixin._COLLOCATION_GAP_WORDS
    _COLLOCATION_DROPABLE_GAP = CollocationExtractorMixin._COLLOCATION_DROPABLE_GAP
    _EXAMPLE_ANCHOR_TYPES = ZeroCollisionAnchorsMixin._EXAMPLE_ANCHOR_TYPES
    _DELEXICAL_VERBS = ZeroCollisionAnchorsMixin._DELEXICAL_VERBS
    _DELEXICAL_QUANTIFIER_NOUNS = ZeroCollisionAnchorsMixin._DELEXICAL_QUANTIFIER_NOUNS
