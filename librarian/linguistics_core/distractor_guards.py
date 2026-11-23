"""Syntactic slot clash, double-key collision, countability contrast, and valency guards (Facade)."""

from __future__ import annotations

import logging

from .distractor_guards_collision import DistractorGuardsCollisionMixin
from .distractor_guards_countability import DistractorGuardsCountabilityMixin
from .distractor_guards_inversion import DistractorGuardsInversionMixin
from .distractor_guards_valency import DistractorGuardsValencyMixin

logger = logging.getLogger("librarian.linguistics.guards")


class DistractorGuardsMixin(
    DistractorGuardsInversionMixin,
    DistractorGuardsCountabilityMixin,
    DistractorGuardsValencyMixin,
    DistractorGuardsCollisionMixin,
):
    """Unified facade exposing all syntactic slot clash, double-key collision, countability, and valency guards."""
    pass


__all__ = [
    "DistractorGuardsMixin",
    "DistractorGuardsInversionMixin",
    "DistractorGuardsCountabilityMixin",
    "DistractorGuardsValencyMixin",
    "DistractorGuardsCollisionMixin",
]
