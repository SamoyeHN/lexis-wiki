"""Base helper methods for dashboard route dispatchers."""

from __future__ import annotations

from .frontmatter_utils import extract_series_edition, inject_or_lock_frontmatter
from .state import DashboardState
from .workers import (
    background_all_quizzes_worker,
    background_audit_batch_worker,
    background_compile_worker,
    background_quiz_worker,
    background_video_import_worker,
)


class RoutesBaseMixin:
    """Base helpers for background workers, frontmatter locks, and dashboard state."""

    @staticmethod
    def _dashboard_state():
        return DashboardState

    @staticmethod
    def _extract_series_edition(*args, **kwargs):
        return extract_series_edition(*args, **kwargs)

    @staticmethod
    def _inject_or_lock_frontmatter(*args, **kwargs):
        return inject_or_lock_frontmatter(*args, **kwargs)

    @staticmethod
    def _background_compile_worker():
        return background_compile_worker

    @staticmethod
    def _background_audit_batch_worker():
        return background_audit_batch_worker

    @staticmethod
    def _background_quiz_worker():
        return background_quiz_worker

    @staticmethod
    def _background_all_quizzes_worker():
        return background_all_quizzes_worker

    @staticmethod
    def _background_video_import_worker():
        return background_video_import_worker
