"""Regression tests for Dashboard frontend single-source modules & auto cache-busting."""

import re
import time
from pathlib import Path

from librarian.dashboard_core.static_server import (
    _compute_static_version,
    _inject_dashboard_asset_versions,
    _inject_es_module_versions,
)

ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = ROOT / "librarian" / "static"
MODULES_DIR = STATIC_DIR / "js" / "modules"
DASHBOARD_JS = STATIC_DIR / "js" / "dashboard.js"
DASHBOARD_HTML = ROOT / "librarian" / "templates" / "dashboard.html"


def _collect_module_window_exports() -> dict[str, list[str]]:
    exports: dict[str, list[str]] = {}
    for p in sorted(MODULES_DIR.glob("*.js")):
        text = p.read_text(encoding="utf-8")
        wins = set(re.findall(r"window\.([A-Za-z0-9_$]+)\s*=", text))
        for m in re.finditer(r"Object\.assign\(\s*window\s*,\s*\{([^}]+)\}", text, re.S):
            wins.update(re.findall(r"\b([A-Za-z0-9_$]+)\b", m.group(1)))
        for s in wins:
            exports.setdefault(s, []).append(p.name)
    return exports


def test_no_shadowed_globals_in_dashboard_js():
    """Ensure dashboard.js never re-declares functions/window bindings owned by modules/*.js."""
    mod_exports = _collect_module_window_exports()
    dash_text = DASHBOARD_JS.read_text(encoding="utf-8")

    dash_defs: set[str] = set()
    for line in dash_text.splitlines():
        m_fn = re.match(r"^\s*(?:async\s+)?function\s+([A-Za-z0-9_$]+)\s*\(", line)
        if m_fn:
            dash_defs.add(m_fn.group(1))
        m_win = re.match(r"^\s*window\.([A-Za-z0-9_$]+)\s*=", line)
        if m_win:
            dash_defs.add(m_win.group(1))

    overlaps = sorted(set(mod_exports) & dash_defs)
    assert overlaps == [], f"Shadowed symbols in dashboard.js: {overlaps}"


def test_dashboard_html_has_no_hardcoded_cache_versions():
    """Ensure dashboard.html and modules/index.js do not hardcode ?v=... query strings."""
    html = DASHBOARD_HTML.read_text(encoding="utf-8")
    assert "/static/js/dashboard.js?v=" not in html
    assert "/static/js/modules/index.js?v=" not in html

    index_js = (MODULES_DIR / "index.js").read_text(encoding="utf-8")
    assert "?v=" not in index_js


def test_inject_dashboard_asset_versions(tmp_path: Path):
    """Verify HTML asset URLs get numeric mtime versions and index.js tracks max module mtime."""
    css = tmp_path / "css" / "dashboard.css"
    js_main = tmp_path / "js" / "dashboard.js"
    js_idx = tmp_path / "js" / "modules" / "index.js"
    js_sub = tmp_path / "js" / "modules" / "hero_board.js"

    for f in (css, js_main, js_idx, js_sub):
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("// stub", encoding="utf-8")

    now = int(time.time())
    import os

    os.utime(css, (now - 100, now - 100))
    os.utime(js_main, (now - 50, now - 50))
    os.utime(js_idx, (now - 200, now - 200))
    os.utime(js_sub, (now - 10, now - 10))

    assert _compute_static_version(tmp_path, "js/modules/index.js") == str(now - 10)

    sample_html = (
        '<link rel="stylesheet" href="/static/css/dashboard.css">\n'
        '<script src="/static/js/dashboard.js"></script>\n'
        '<script type="module" src="/static/js/modules/index.js"></script>\n'
    )
    rendered = _inject_dashboard_asset_versions(sample_html, tmp_path)
    assert f"/static/css/dashboard.css?v={now - 100}" in rendered
    assert f"/static/js/dashboard.js?v={now - 50}" in rendered
    assert f"/static/js/modules/index.js?v={now - 10}" in rendered


def test_inject_es_module_versions(tmp_path: Path):
    """Verify relative ES module imports inside js/modules/ receive ?v=<target_mtime>."""
    idx = tmp_path / "index.js"
    hb = tmp_path / "hero_board.js"
    idx.write_text("import * as hero_board from './hero_board.js';\n", encoding="utf-8")
    hb.write_text("export function ok() {}\n", encoding="utf-8")

    expected_mtime = int(hb.stat().st_mtime)
    rewritten = _inject_es_module_versions(idx.read_text(encoding="utf-8"), idx)
    assert f"from './hero_board.js?v={expected_mtime}'" in rewritten
