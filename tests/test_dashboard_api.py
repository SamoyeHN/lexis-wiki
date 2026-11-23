"""
Test suite for Dashboard API endpoints and background workers.
"""

from unittest.mock import MagicMock, patch
from pathlib import Path
from librarian.processor import WikiProcessor
from librarian.dashboard import DashboardState, background_all_quizzes_worker


def test_wiki_processor_normalize_name():
    assert WikiProcessor.normalize_name("Book 1 Unit 1 Passage A") == "book1unit1passagea"
    p = WikiProcessor()
    assert p.normalize_name("Book_1_Unit_1") == "book1unit1"


def test_background_all_quizzes_worker():
    job_id = DashboardState.create_job("quiz", "Test_Unit (All Quizzes)")

    with patch.object(Path, "exists", return_value=True), patch.object(
        WikiProcessor, "generate_quiz", return_value="/path/to/Book_1_Unit_1_Passage_A_quiz.html"
    ) as mock_gen:
        background_all_quizzes_worker(job_id, "Book_1_Unit_1_Passage_A")

    job = DashboardState.get_job(job_id)
    assert job is not None
    assert job["status"] == "completed"
    assert job["progress"] == 100
    assert len(job["saved_files"]) >= 5
    called_templates = [call.kwargs.get("template_name") for call in mock_gen.call_args_list]
    assert "cloze" in called_templates
    assert "vocabulary" in called_templates


def test_cloze_quiz_config_default():
    from librarian.config import config

    assert config.get_quiz_config("cloze", "count", 10) == 10


def test_inject_or_lock_frontmatter_no_existing_frontmatter():
    from librarian.dashboard import inject_or_lock_frontmatter

    raw_text = "# Passage A\n\nThis is a sample text without frontmatter."
    res = inject_or_lock_frontmatter(raw_text, target_series="New Horizon College English", target_book="2")

    assert res.startswith("---\n")
    assert 'series: "New Horizon College English"' in res
    assert "book: 2" in res
    assert 'book_name: "Book 2"' in res
    assert "# Passage A" in res


def test_inject_or_lock_frontmatter_existing_frontmatter_missing_series():
    from librarian.dashboard import inject_or_lock_frontmatter

    existing = "---\nunit: 3\nunit_title: \"Life and learning\"\n---\n\n## Content"
    res = inject_or_lock_frontmatter(existing, target_series="New Practical English", target_book="1")

    assert 'unit: 3' in res
    assert 'unit_title: "Life and learning"' in res
    assert 'series: "New Practical English"' in res
    assert 'book: 1' in res
    assert 'book_name: "Book 1"' in res
    assert "## Content" in res


def test_inject_or_lock_frontmatter_locks_existing_series_and_book():
    from librarian.dashboard import inject_or_lock_frontmatter

    existing = "---\nseries: \"Original Series\"\nbook: 3\n---\n\n## Content"
    res = inject_or_lock_frontmatter(existing, target_series="Overriding Series", target_book="4")

    # Must preserve existing frontmatter values (Locked!)
    assert 'series: "Original Series"' in res
    assert "book: 3" in res
    assert "Overriding Series" not in res
    assert "book: 4" not in res


def test_inject_or_lock_frontmatter_no_targets():
    from librarian.dashboard import inject_or_lock_frontmatter

    raw_text = "Just plain text"
    res = inject_or_lock_frontmatter(raw_text, target_series=None, target_book=None)
    assert res == raw_text


def test_api_raw_files_parses_series_and_book(tmp_path, monkeypatch):
    from unittest.mock import MagicMock
    from librarian.dashboard import DashboardHTTPRequestHandler
    from librarian.config import config
    import json
    import io

    # Create hermetic dummy wiki directory with frontmatter
    test_wiki = tmp_path / "wiki"
    unit_dir = test_wiki / "Test_Unit_1"
    sources_dir = unit_dir / "sources"
    sources_dir.mkdir(parents=True)
    source_file = sources_dir / "Test_Unit_1.md"
    source_file.write_text(
        "---\nseries: \"Test Series\"\nbook: 1\nunit: 1\nsection: \"A\"\n---\n# Test Unit\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "project_root", str(tmp_path))

    # Mock handler and request
    handler = DashboardHTTPRequestHandler.__new__(DashboardHTTPRequestHandler)
    handler.wfile = io.BytesIO()
    handler.send_response = MagicMock()
    handler.send_header = MagicMock()
    handler.end_headers = MagicMock()

    # Call handle_api_get for /api/raw-files
    handler.handle_api_get("/api/raw-files", "")
    
    output = handler.wfile.getvalue().decode("utf-8")
    data = json.loads(output)
    assert "files" in data
    assert len(data["files"]) > 0

    # Verify at least some units have series and book populated
    units_with_series = [f for f in data["files"] if f.get("series")]
    assert len(units_with_series) > 0

    first = units_with_series[0]
    assert "series" in first
    assert "series_clean" in first
    assert "series_display" in first
    assert "book" in first
    assert "unit" in first
    assert "section" in first


def test_extract_series_edition():
    from librarian.dashboard import extract_series_edition

    # Test parens edition
    clean, ed, abbr, disp = extract_series_edition("New Horizon College English (Fourth Edition)")
    assert clean == "New Horizon College English"
    assert ed == "4e"
    assert abbr == "New Horizon"
    assert disp == "New Horizon · 4e"

    # Test no edition
    clean, ed, abbr, disp = extract_series_edition("New Practical English")
    assert clean == "New Practical English"
    assert ed is None
    assert abbr == "Practical English"
    assert disp == "Practical English"

    # Test 2nd Edition unparenthesized
    clean, ed, abbr, disp = extract_series_edition("Contemporary College English 2nd Ed.")
    assert clean == "Contemporary College English"
    assert ed == "2e"
    assert abbr == "Contemporary"
    assert disp == "Contemporary · 2e"

    # Test Revised Edition
    clean, ed, abbr, disp = extract_series_edition("Advanced English (Revised Edition)")
    assert clean == "Advanced English"
    assert ed == "Rev"
    assert disp == "Advanced English · Rev"

    # Test explicit edition override
    clean, ed, abbr, disp = extract_series_edition("General English", raw_edition="3rd", raw_abbr="GenEng")
    assert clean == "General English"
    assert ed == "3e"
    assert abbr == "GenEng"
    assert disp == "GenEng · 3e"

    # Test None
    assert extract_series_edition(None) == (None, None, None, None)


def test_book_glossary_api_and_ledger():
    from librarian.glossary_ledger import get_or_create_ledger
    ledger = get_or_create_ledger("New Practical English", 1)
    summary = ledger.summary()
    assert summary["series"] == "New Practical English"
    assert summary["book"] == "1"
    assert "total_entries" in summary
    assert "cefr_breakdown" in summary


def test_api_import_pdf_glossary_binary_upload():
    from unittest.mock import MagicMock, patch
    from librarian.dashboard import DashboardHTTPRequestHandler
    import json
    import io
    import urllib.parse

    handler = DashboardHTTPRequestHandler.__new__(DashboardHTTPRequestHandler)
    pdf_bytes = b"%PDF-1.4 test mock stream"
    handler.rfile = io.BytesIO(pdf_bytes)
    handler.wfile = io.BytesIO()
    handler.path = "/api/import-pdf-glossary"
    handler.headers = {
        "Content-Type": "application/octet-stream",
        "Content-Length": str(len(pdf_bytes)),
        "X-Filename": urllib.parse.quote("【Glossary】Sample.pdf"),
        "X-Series": urllib.parse.quote("New Practical English"),
        "X-Book": "1",
    }
    handler.send_response = MagicMock()
    handler.send_header = MagicMock()
    handler.end_headers = MagicMock()

    with patch("librarian.glossary_ledger.GlossaryLedger.import_pdf_glossary") as mock_import:
        mock_import.return_value = {
            "entries_added": 12,
            "entries_updated": 3,
            "total_entries": 15,
        }
        handler.do_POST()

    output = handler.wfile.getvalue().decode("utf-8")
    data = json.loads(output)
    assert data["success"] is True
    assert data["result"]["entries_added"] == 12
    assert data["result"]["total_entries"] == 15
    mock_import.assert_called_once()


def test_routes_mixin_direct_dispatch():
    import io
    import json
    from unittest.mock import MagicMock
    from librarian.dashboard import DashboardHTTPRequestHandler

    handler = DashboardHTTPRequestHandler.__new__(DashboardHTTPRequestHandler)
    handler.wfile = io.BytesIO()
    handler.send_response = MagicMock()
    handler.send_header = MagicMock()
    handler.end_headers = MagicMock()

    # Test GET /api/recent-jobs
    handler.wfile.seek(0)
    handler.wfile.truncate(0)
    handler.handle_api_get('/api/recent-jobs', '')
    res_get = json.loads(handler.wfile.getvalue().decode('utf-8'))
    assert 'jobs' in res_get

    # Test POST /api/test-endpoint
    handler.wfile.seek(0)
    handler.wfile.truncate(0)
    handler.headers = {'Content-Length': '0'}
    handler.rfile = io.BytesIO(b'')
    handler.handle_api_post('/api/test-endpoint')
    res_post = json.loads(handler.wfile.getvalue().decode('utf-8'))
    assert res_post.get('test') == 'working'

def test_static_asset_serving_es_modules():
    import io
    from unittest.mock import MagicMock
    from librarian.dashboard import DashboardHTTPRequestHandler

    handler = DashboardHTTPRequestHandler.__new__(DashboardHTTPRequestHandler)
    handler.send_response = MagicMock()
    handler.send_header = MagicMock()
    handler.end_headers = MagicMock()
    handler.wfile = io.BytesIO()

    for asset in [
        "/static/js/dashboard.js",
        "/static/js/modules/index.js",
        "/static/js/modules/api.js",
        "/static/js/modules/state.js",
        "/static/js/modules/utils.js",
        "/static/js/modules/glossary.js"
    ]:
        handler.wfile.seek(0)
        handler.wfile.truncate(0)
        handler.serve_static_asset(asset)
        content = handler.wfile.getvalue()
        assert len(content) > 0, f"Failed to serve static asset {asset}"
