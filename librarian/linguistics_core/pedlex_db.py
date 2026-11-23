"""
PedLex SQLite connection management, entry & derived-form resolution, typed slot patterns, and phrase index lookup.
Unified facade mixin composing modular domain mixins.
"""

from .pedlex_core import PedlexCoreMixin
from .pedlex_patterns import PedlexPatternsMixin
from .pedlex_phrases import PedlexPhrasesMixin
from .pedlex_thesaurus import PedlexThesaurusMixin
from .pedlex_cxn_traps import PedlexCxnTrapsMixin


class PedlexDbMixin(
    PedlexCoreMixin,
    PedlexPatternsMixin,
    PedlexPhrasesMixin,
    PedlexThesaurusMixin,
    PedlexCxnTrapsMixin,
):
    """Unified facade mixin for PedLex database operations and lexicographical lookups."""

    # Backward-compatible exposure of class-level constants
    _HEADWORD_FORM_EXCEPTIONS = PedlexCoreMixin._HEADWORD_FORM_EXCEPTIONS
    _COLLOCATION_KEY_ALIASES = PedlexCoreMixin._COLLOCATION_KEY_ALIASES
    _DERIVED_BORROW_KEYS = PedlexCoreMixin._DERIVED_BORROW_KEYS
    _NATIVE_LL_STEMS = PedlexCoreMixin._NATIVE_LL_STEMS
    _XREF_SPELLING_RE = PedlexCoreMixin._XREF_SPELLING_RE
    _PHRASAL_PARTICLES = PedlexPatternsMixin._PHRASAL_PARTICLES
    _PHRASE_SLOT_WORDS = PedlexPatternsMixin._PHRASE_SLOT_WORDS
    _PHRASE_LEAD_WORDS = PedlexPhrasesMixin._PHRASE_LEAD_WORDS
    _PHRASE_PARTICLE_WORDS = PedlexPhrasesMixin._PHRASE_PARTICLE_WORDS
    _PHRASE_TIERS = PedlexPhrasesMixin._PHRASE_TIERS
    _PHRASE_STRONG_TIERS = PedlexPhrasesMixin._PHRASE_STRONG_TIERS
    _SLASH_PLACEHOLDERS = PedlexPhrasesMixin._SLASH_PLACEHOLDERS
    _HYPHEN_SLASH_RE = PedlexPhrasesMixin._HYPHEN_SLASH_RE
    _ANY_SLASH_GROUP_RE = PedlexPhrasesMixin._ANY_SLASH_GROUP_RE
