"""Base helper methods for dashboard route dispatchers."""

from __future__ import annotations

from typing import Any, Callable


class RoutesBaseMixin:
    """Base lazy-import helpers for background workers and dashboard state."""

    @staticmethod
    def _dashboard_state():
        from ..dashboard import DashboardState
        return DashboardState

    @staticmethod
    def _extract_series_edition(*args, **kwargs):
        from ..dashboard import extract_series_edition
        return extract_series_edition(*args, **kwargs)

    @staticmethod
    def _inject_or_lock_frontmatter(*args, **kwargs):
        from ..dashboard import inject_or_lock_frontmatter
        return inject_or_lock_frontmatter(*args, **kwargs)

    @staticmethod
    def _background_compile_worker():
        from ..dashboard import background_compile_worker
        return background_compile_worker

    @staticmethod
    def _background_audit_batch_worker():
        from ..dashboard import background_audit_batch_worker
        return background_audit_batch_worker

    @staticmethod
    def _background_quiz_worker():
        from ..dashboard import background_quiz_worker
        return background_quiz_worker

    @staticmethod
    def _background_all_quizzes_worker():
        from ..dashboard import background_all_quizzes_worker
        return background_all_quizzes_worker

    @staticmethod
    def _background_video_import_worker():
        from ..dashboard import background_video_import_worker
        return background_video_import_worker


