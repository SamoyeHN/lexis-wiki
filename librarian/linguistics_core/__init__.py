"""Internal modular mixins for LinguisticEngine."""

from .distractor_engine import (
    DistractorEngineMixin,
    DistractorFallbacksMixin,
    DistractorGuardsMixin,
    DistractorInflectionsMixin,
    DistractorParonymsMixin,
    DistractorPhrasesMixin,
    DistractorVocabMixin,
)
from .slot_frame import SlotFrame
from .wsd_arbitration import WsdArbitrationMixin
from .wsd_definitions import WsdDefinitionsMixin
from .wsd_engine import WsdEngineMixin
from .wsd_expressions import WsdExpressionsMixin
from .wsd_gates import WsdGatesMixin
from .wsd_lock import WsdLockMixin
from .wsd_scoring import WsdContext, WsdScoringMixin
from .wsd_wordnet import WsdWordNetMixin

__all__ = [
    "DistractorEngineMixin",
    "DistractorFallbacksMixin",
    "DistractorGuardsMixin",
    "DistractorInflectionsMixin",
    "DistractorParonymsMixin",
    "DistractorPhrasesMixin",
    "DistractorVocabMixin",
    "SlotFrame",
    "WsdArbitrationMixin",
    "WsdContext",
    "WsdDefinitionsMixin",
    "WsdEngineMixin",
    "WsdExpressionsMixin",
    "WsdGatesMixin",
    "WsdLockMixin",
    "WsdScoringMixin",
    "WsdWordNetMixin",
]
