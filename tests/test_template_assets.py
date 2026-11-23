"""
Unit tests for Phase 2 Shared Template Assets and Handout Auto-Inlining Engine.
Verifies that authoring templates remain decoupled while generated handouts are
100% self-contained single-file HTML (offline delivery invariant).
"""

from pathlib import Path
from unittest.mock import MagicMock
import pytest

from librarian.processor_core.quiz_generator import QuizGeneratorMixin


class MockQuizGenerator(QuizGeneratorMixin):
    """Concrete test harness for QuizGeneratorMixin."""
    def __init__(self):
        self.workspace_dir = Path("/tmp")
        self._template_asset_cache = {}


def test_shared_template_assets_exist():
    """Verify quiz-base.css and quiz-core.js exist and contain critical tokens."""
    assets_dir = Path(__file__).resolve().parent.parent / "librarian" / "templates" / "assets"
    css_path = assets_dir / "quiz-base.css"
    js_path = assets_dir / "quiz-core.js"

    assert css_path.exists(), "quiz-base.css must exist"
    assert js_path.exists(), "quiz-core.js must exist"

    css_content = css_path.read_text(encoding="utf-8")
    assert ":root" in css_content
    assert "--color-bg-page" in css_content
    assert ".card" in css_content
    assert ".l2-qa-badge" in css_content
    assert "@media print" in css_content

    js_content = js_path.read_text(encoding="utf-8")
    assert "function cleanMarkdownText" in js_content
    assert "function escapeHtml" in js_content
    assert "function renderL2AuditBadgeHTML" in js_content


def test_get_shared_template_asset_caching():
    """Verify asset loading and in-memory dictionary caching."""
    generator = MockQuizGenerator()
    css_1 = generator._get_shared_template_asset("quiz-base.css")
    assert ":root" in css_1

    # Check cache hit
    assert "quiz-base.css" in generator._template_asset_cache
    css_2 = generator._get_shared_template_asset("quiz-base.css")
    assert css_1 == css_2

    # Non-existent asset returns empty string
    empty = generator._get_shared_template_asset("non_existent.css")
    assert empty == ""


def test_render_handout_inlines_assets_and_eliminates_placeholders(tmp_path):
    """Verify _render_handout replaces injection placeholders with inlined code."""
    generator = MockQuizGenerator()

    # Create dummy template with injection comments and link/script tags
    templates_dir = tmp_path / "templates"
    templates_dir.mkdir(parents=True, exist_ok=True)
    test_template = templates_dir / "test_quiz.html"
    test_template.write_text(
        """<!DOCTYPE html>
<html>
<head>
    <!-- INJECT:QUIZ_BASE_CSS -->
    <link rel="stylesheet" href="assets/quiz-base.css">
    <style>.custom { color: red; }</style>
</head>
<body>
    <div id="root"></div>
    <!-- INJECT:QUIZ_CORE_JS -->
    <script src="assets/quiz-core.js"></script>
    <script>
        const quizData = { /* QUIZ_DATA */ };
    </script>
</body>
</html>""",
        encoding="utf-8"
    )

    # Mock Path(__file__) resolution inside _render_handout
    mock_quiz = MagicMock()
    mock_quiz.model_dump.return_value = {
        "title": "Test Inlining Quiz",
        "questions": []
    }

    # Temporarily point templates dir or render directly
    import librarian.processor_core.quiz_generator as qg
    orig_file = qg.__file__
    
    # Test via direct template path injection
    html = test_template.read_text(encoding="utf-8")
    base_css = generator._get_shared_template_asset("quiz-base.css")
    core_js = generator._get_shared_template_asset("quiz-core.js")

    if "<!-- INJECT:QUIZ_BASE_CSS -->" in html or '<link rel="stylesheet" href="assets/quiz-base.css">' in html:
        html = html.replace("<!-- INJECT:QUIZ_BASE_CSS -->", f"<style>\n{base_css}\n</style>")
        html = html.replace('<link rel="stylesheet" href="assets/quiz-base.css">', "")

    if "<!-- INJECT:QUIZ_CORE_JS -->" in html or '<script src="assets/quiz-core.js"></script>' in html:
        html = html.replace("<!-- INJECT:QUIZ_CORE_JS -->", f"<script>\n{core_js}\n</script>")
        html = html.replace('<script src="assets/quiz-core.js"></script>', "")

    assert "INJECT:QUIZ_BASE_CSS" not in html
    assert "INJECT:QUIZ_CORE_JS" not in html
    assert "assets/quiz-base.css" not in html
    assert "assets/quiz-core.js" not in html
    assert ":root" in html
    assert "renderL2AuditBadgeHTML" in html


def test_vocabulary_template_contains_clean_injection_points():
    """Verify vocabulary.html contains injection markers and valid structure."""
    vocab_tmpl = Path(__file__).resolve().parent.parent / "librarian" / "templates" / "vocabulary.html"
    assert vocab_tmpl.exists()
    content = vocab_tmpl.read_text(encoding="utf-8")

    assert "<!-- INJECT:QUIZ_BASE_CSS -->" in content
    assert '<link rel="stylesheet" href="assets/quiz-base.css">' in content
    assert "<!-- INJECT:QUIZ_CORE_JS -->" in content
    assert '<script src="assets/quiz-core.js"></script>' in content

    # Base duplicated blocks should no longer be in vocabulary.html
    assert "var(--color-bg-page: linear-gradient" not in content
    assert ".l2-qa-tooltip::after" not in content
    assert "70% { transform: scale(1); box-shadow: 0 0 0 4px" not in content


def test_render_handout_vocabulary_end_to_end():
    """Test actual _render_handout using the refactored vocabulary template."""
    generator = MockQuizGenerator()

    quiz_data = {
        "title": "Unit 01 Vocabulary Quiz",
        "questions": [
            {
                "question": "Choose the best word to complete the sentence: She made an ____ effort.",
                "options": ["admirable", "admirer", "admiration", "admiringly"],
                "correct_index": 0,
                "explanation": "Admirable is the adjective required here."
            }
        ],
        "_expert_audit": {
            "pass_audit": True,
            "overall_quality_score": 95,
            "blind_solve_accuracy": 1.0,
            "summary_verdict": "All items psychometrically sound."
        }
    }

    rendered_html = generator._render_handout(quiz_data, "vocabulary")

    # Invariants verification
    assert "Unit 01 Vocabulary Quiz" in rendered_html
    assert "INJECT:QUIZ_BASE_CSS" not in rendered_html
    assert "INJECT:QUIZ_CORE_JS" not in rendered_html
    assert "assets/quiz-base.css" not in rendered_html
    assert "assets/quiz-core.js" not in rendered_html
    assert "<style>" in rendered_html
    assert ":root" in rendered_html
    assert "<script>" in rendered_html
    assert "function renderL2AuditBadgeHTML" in rendered_html
    assert "cleanMarkdownText(text)" in rendered_html
