"""Modular route mixins for Lexis Wiki Web Dashboard."""

from .routes_ask import RoutesAskMixin
from .routes_base import RoutesBaseMixin
from .routes_files import RoutesFilesMixin
from .routes_glossary import RoutesGlossaryMixin
from .routes_jobs import RoutesJobsMixin
from .routes_media import RoutesMediaMixin
from .routes_quizzes import RoutesQuizzesMixin
from .routes_system import RoutesSystemMixin

__all__ = [
    "RoutesBaseMixin",
    "RoutesSystemMixin",
    "RoutesGlossaryMixin",
    "RoutesJobsMixin",
    "RoutesQuizzesMixin",
    "RoutesFilesMixin",
    "RoutesMediaMixin",
    "RoutesAskMixin",
]
