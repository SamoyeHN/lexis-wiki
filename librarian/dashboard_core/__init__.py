"""Modular route mixins and core services for Lexis Wiki Web Dashboard."""

from .frontmatter_utils import extract_series_edition, get_raw_file_stem, inject_or_lock_frontmatter
from .routes_ask import RoutesAskMixin
from .routes_base import RoutesBaseMixin
from .routes_files import RoutesFilesMixin
from .routes_glossary import RoutesGlossaryMixin
from .routes_jobs import RoutesJobsMixin
from .routes_media import RoutesMediaMixin
from .routes_quizzes import RoutesQuizzesMixin
from .routes_system import RoutesSystemMixin
from .state import DashboardState
from .static_server import DashboardStaticMixin
from .workers import (
    background_all_quizzes_worker,
    background_audit_batch_worker,
    background_compile_worker,
    background_quiz_worker,
    background_video_import_worker,
)

__all__ = [
    "DashboardState",
    "DashboardStaticMixin",
    "get_raw_file_stem",
    "extract_series_edition",
    "inject_or_lock_frontmatter",
    "background_compile_worker",
    "background_quiz_worker",
    "background_all_quizzes_worker",
    "background_video_import_worker",
    "background_audit_batch_worker",
    "RoutesBaseMixin",
    "RoutesSystemMixin",
    "RoutesGlossaryMixin",
    "RoutesJobsMixin",
    "RoutesQuizzesMixin",
    "RoutesFilesMixin",
    "RoutesMediaMixin",
    "RoutesAskMixin",
]
