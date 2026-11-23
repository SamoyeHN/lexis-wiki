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
    assert ".btn" in css_content
    assert ".l2-qa-badge" in css_content
    assert "@media print" in css_content

    # Invariant: quiz-base.css must NOT contain structural layout rules (.container, .card)
    # to prevent cross-modality CSS collisions (e.g. cloze 3-column blowout or translation vertical voids).
    assert ".container {" not in css_content, "quiz-base.css must not enforce layout container rules"

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


def test_portal_and_handout_assets_exist():
    """Verify portal.css and handout.css exist and contain design tokens."""
    assets_dir = Path(__file__).resolve().parent.parent / "librarian" / "templates" / "assets"
    portal_css = assets_dir / "portal.css"
    handout_css = assets_dir / "handout.css"

    assert portal_css.exists(), "portal.css must exist"
    assert handout_css.exists(), "handout.css must exist"

    portal_content = portal_css.read_text(encoding="utf-8")
    assert "--accent-node" in portal_content
    assert "--accent-quiz" in portal_content
    assert ".portal-card" in portal_content

    handout_content = handout_css.read_text(encoding="utf-8")
    assert "--accent-navy" in handout_content
    assert ".lex-entry-card" in handout_content
    assert ".lex-pill" in handout_content


def test_portal_and_handout_templates_contain_injection_markers():
    """Verify portal.html and handout.html exist and contain clean injection markers."""
    tmpl_dir = Path(__file__).resolve().parent.parent / "librarian" / "templates"
    portal_tmpl = tmpl_dir / "portal.html"
    handout_tmpl = tmpl_dir / "handout.html"

    assert portal_tmpl.exists(), "portal.html must exist"
    assert handout_tmpl.exists(), "handout.html must exist"

    portal_content = portal_tmpl.read_text(encoding="utf-8")
    assert "<!-- INJECT:PORTAL_CSS -->" in portal_content
    assert '<link rel="stylesheet" href="assets/portal.css">' in portal_content

    handout_content = handout_tmpl.read_text(encoding="utf-8")
    assert "<!-- INJECT:HANDOUT_CSS -->" in handout_content
    assert '<link rel="stylesheet" href="../assets/handout.css">' in handout_content


def test_template_loader_caching_and_asset_retrieval():
    """Verify template loader caches results and handles missing files gracefully."""
    from librarian.exporter import clear_template_cache, get_asset, get_template

    clear_template_cache()

    # Load template
    tmpl_1 = get_template("portal.html")
    assert "{{ UNIT_TITLE }}" in tmpl_1
    tmpl_2 = get_template("portal.html")
    assert tmpl_1 == tmpl_2

    # Load asset
    css_1 = get_asset("portal.css")
    assert "--accent-node" in css_1
    css_2 = get_asset("portal.css")
    assert css_1 == css_2

    # Missing returns empty string
    assert get_template("missing_template_xyz.html") == ""
    assert get_asset("missing_asset_xyz.css") == ""


def test_render_index_portal_html_inlining_end_to_end():
    """Verify render_index_portal_html inlines portal.css for self-contained offline browsing."""
    from librarian.exporter import render_index_portal_html

    extractions = [
        {"category": "source", "title": "Source Passage", "rel_path": "extractions/Demo_source.html"}
    ]
    quizzes = [
        {"filename": "Demo_vocabulary_quiz.html", "title": "Vocabulary Quiz", "rel_path": "handouts/Demo_vocabulary_quiz.html"}
    ]

    html_out = render_index_portal_html(
        unit_name="Demo_Unit",
        display_title="Demo Unit Title",
        cefr_level="B2",
        extractions=extractions,
        quizzes=quizzes,
    )

    assert "<!DOCTYPE html>" in html_out
    assert "Demo Unit Title — Study Suite" in html_out
    assert "CEFR B2" in html_out
    assert "<!-- INJECT:PORTAL_CSS -->" not in html_out
    assert 'href="assets/portal.css"' not in html_out
    assert "<style>" in html_out
    assert "--accent-node" in html_out
    assert "--accent-quiz" in html_out


def test_render_markdown_to_standalone_html_inlining_end_to_end():
    """Verify render_markdown_to_standalone_html inlines handout.css for self-contained offline reading."""
    from librarian.exporter import render_markdown_to_standalone_html

    sample_md = """---
title: Sample Vocabulary
overall_cefr_level: C1
item_count: 5
---

# Sample Vocabulary

## [[serendipity]]
- **Definition**: finding valuable things by chance
- [formal] `[uncountable]`
"""

    html_out = render_markdown_to_standalone_html(sample_md, fallback_title="Fallback Vocab", category="vocabulary")

    assert "<!DOCTYPE html>" in html_out
    assert "Sample Vocabulary" in html_out
    assert "CEFR C1" in html_out
    assert "5 Items" in html_out
    assert "<!-- INJECT:HANDOUT_CSS -->" not in html_out
    assert 'href="../assets/handout.css"' not in html_out
    assert "<style>" in html_out
    assert "--accent-navy" in html_out
    assert "serendipity" in html_out

