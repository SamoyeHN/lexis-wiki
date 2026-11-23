"""
Lightweight template and static asset loader with in-memory caching.

Provides zero-dependency template and CSS/JS asset retrieval and inlining
for unit portals, handouts, and standalone exports.
"""

from pathlib import Path

_ASSET_CACHE: dict[str, str] = {}
_TEMPLATE_CACHE: dict[str, str] = {}

_TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
_ASSETS_DIR = _TEMPLATES_DIR / "assets"


def get_template(template_name: str) -> str:
    """
    Retrieves the raw HTML template content by filename (e.g., 'portal.html').
    Results are cached in memory for sub-millisecond repeated renders.
    """
    if template_name in _TEMPLATE_CACHE:
        return _TEMPLATE_CACHE[template_name]

    template_file = _TEMPLATES_DIR / template_name
    if template_file.exists():
        content = template_file.read_text(encoding="utf-8")
        _TEMPLATE_CACHE[template_name] = content
        return content
    return ""


def get_asset(asset_name: str) -> str:
    """
    Retrieves the raw static asset content (e.g., 'portal.css', 'handout.css').
    Results are cached in memory.
    """
    if asset_name in _ASSET_CACHE:
        return _ASSET_CACHE[asset_name]

    asset_file = _ASSETS_DIR / asset_name
    if asset_file.exists():
        content = asset_file.read_text(encoding="utf-8")
        _ASSET_CACHE[asset_name] = content
        return content
    return ""


def inline_asset_css(
    html_content: str,
    injection_marker: str,
    link_tag: str,
    css_name: str,
) -> str:
    """
    Replaces an injection marker and stylesheet link tag with self-contained <style> blocks.
    Enforces the 100% offline self-contained delivery invariant.
    """
    css_content = get_asset(css_name)
    if not css_content:
        return html_content

    if injection_marker in html_content:
        html_content = html_content.replace(
            injection_marker,
            f"<style>\n{css_content}\n    </style>",
        )
    if link_tag in html_content:
        html_content = html_content.replace(link_tag, "")

    return html_content


def clear_template_cache() -> None:
    """Clears both template and asset caches (used primarily in test suites)."""
    _TEMPLATE_CACHE.clear()
    _ASSET_CACHE.clear()
