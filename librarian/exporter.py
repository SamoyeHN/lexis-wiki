"""
Lexis Wiki Exporter & Portable Handout Generator.
Unified facade exposing export packaging, markdown rendering, portal generation, and explorer helpers.
"""

import logging

from .config import config
from .exporter_core import (
    _BI_ICONS_SVG,
    detect_video_path_from_html,
    export_portable_bundle,
    export_standalone_exe,
    export_unit_to_directory,
    export_unit_to_zip,
    format_inline_lexis_markdown,
    make_portable_html_content,
    normalize_export_target_dir,
    open_directory_in_explorer,
    render_index_portal_html,
    render_markdown_to_standalone_html,
    to_windows_display_path,
)

logger = logging.getLogger("librarian.exporter")

__all__ = [
    "config",
    "logger",
    "detect_video_path_from_html",
    "make_portable_html_content",
    "export_portable_bundle",
    "export_standalone_exe",
    "format_inline_lexis_markdown",
    "render_markdown_to_standalone_html",
    "_BI_ICONS_SVG",
    "render_index_portal_html",
    "normalize_export_target_dir",
    "to_windows_display_path",
    "export_unit_to_directory",
    "export_unit_to_zip",
    "open_directory_in_explorer",
]
