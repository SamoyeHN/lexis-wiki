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


@pytest.mark.parametrize(
    "template_name",
    ["vocabulary", "listening", "reading", "translation", "video", "cloze"]
)
def test_all_templates_contain_clean_injection_points(template_name):
    """Verify each template contains shared asset injection markers and link/script tags."""
    tmpl_path = Path(__file__).resolve().parent.parent / "librarian" / "templates" / f"{template_name}.html"
    assert tmpl_path.exists(), f"Template {template_name}.html must exist"
    content = tmpl_path.read_text(encoding="utf-8")

    assert "<!-- INJECT:QUIZ_BASE_CSS -->" in content, f"{template_name}.html missing CSS injection tag"
    assert '<link rel="stylesheet" href="assets/quiz-base.css">' in content
    assert "<!-- INJECT:QUIZ_CORE_JS -->" in content, f"{template_name}.html missing JS injection tag"
    assert '<script src="assets/quiz-core.js"></script>' in content


@pytest.mark.parametrize(
    "template_name",
    ["vocabulary", "listening", "reading", "translation", "video", "cloze"]
)
def test_render_handout_all_modalities_end_to_end(template_name):
    """
    Test end-to-end handout rendering across all 6 quiz modalities.
    Verifies that all external links/scripts are eliminated and inlined into
    a 100% self-contained offline HTML delivery document.
    """
    generator = MockQuizGenerator()

    quiz_data = {
        "title": f"Unit 01 {template_name.capitalize()} Handout",
        "questions": [
            {
                "question": "Sample assessment item?",
                "options": ["Option A", "Option B", "Option C", "Option D"],
                "correct_index": 0,
                "correct_answer_index": 0,
                "explanation": "Valid test explanation."
            }
        ],
        "blanks": [
            {
                "blank_index": 1,
                "target_word": "test",
                "accepted_answers": ["test"],
                "hint": "a trial",
                "category": "P2"
            }
        ],
        "passage": "This is a test passage with [1] _____.",
        "masked_passage": "This is a test passage with [1] _____.",
        "_expert_audit": {
            "pass_audit": True,
            "overall_quality_score": 98,
            "blind_solve_accuracy": 1.0,
            "summary_verdict": "Psychometrically robust item."
        }
    }

    rendered_html = generator._render_handout(quiz_data, template_name)

    # Invariants verification
    assert f"Unit 01 {template_name.capitalize()} Handout" in rendered_html
    assert "<!-- INJECT:QUIZ_BASE_CSS -->" not in rendered_html
    assert "<!-- INJECT:QUIZ_CORE_JS -->" not in rendered_html
    assert 'href="assets/quiz-base.css"' not in rendered_html
    assert 'src="assets/quiz-core.js"' not in rendered_html
    assert "<style>" in rendered_html
    assert ":root" in rendered_html
    assert "<script>" in rendered_html
    assert "function renderL2AuditBadgeHTML" in rendered_html
    assert "cleanMarkdownText(text)" in rendered_html
