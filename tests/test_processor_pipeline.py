"""End-to-end and boundary contract tests for WikiProcessor extraction and pipeline runner."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from librarian.config import config
from librarian.processor import WikiProcessor
from librarian.schemas import MindMapExtraction, SummaryExtraction


def test_processor_normalize_grammar_formula_brackets():
    wp = WikiProcessor()
    # Literal functional words wrapped in brackets should be unwrapped
    formula = "[S] + [has to] + [V] + [that] + [clause]"
    normalized = wp.normalize_grammar_formula(formula)
    assert "[has to]" not in normalized
    assert "has to" in normalized
    assert "[that]" not in normalized
    assert "that" in normalized
    assert "[S]" in normalized
    assert "[V]" in normalized


def test_processor_unwrap_literal_brackets():
    wp = WikiProcessor()
    # Test functional words unwrapping
    res = wp.unwrap_literal_brackets("[so that] + [S] + [can] + [V]")
    assert "so that" in res
    assert "[so that]" not in res


def test_processor_run_pipeline_file_not_found(tmp_path: Path):
    wp = WikiProcessor()
    with patch.object(config, "project_root", tmp_path):
        status, saved = wp.run_pipeline("NonExistent_File_123.md")
        assert "Error" in status or "File not found" in status
        assert saved == []


def test_processor_run_pipeline_end_to_end_mocked(tmp_path: Path):
    wp = WikiProcessor()

    # Create hermetic source unit
    unit_name = "Mock_Unit_Alpha"
    sources_dir = tmp_path / "wiki" / unit_name / "sources"
    sources_dir.mkdir(parents=True)
    source_file = sources_dir / f"{unit_name}.md"
    source_file.write_text(
        "---\ntitle: Mock Unit\nseries: Demo Series\nbook: 1\n---\n\n"
        "Alice decided to learn a new language. She worked hard every day.\n\n"
        "## Vocabulary\n"
        "- decide /dɪˈsaɪd/ v. to make a choice\n",
        encoding="utf-8",
    )

    mock_summary = SummaryExtraction(
        title="Mock Unit Summary",
        text_summary_or_plot="Alice learns a new language with persistence.",
        estimated_reading_time="3 mins",
        essential_questions=["How does persistence affect language learning?"],
        lesson_hook="Can you learn a new language in a year?",
        concepts=["persistence", "linguistic practice"],
    )
    mock_mindmap = MindMapExtraction(
        title="Mock Mind Map",
        root_name="Language Learning",
        branches=[],
    )

    with patch.object(config, "project_root", tmp_path), \
         patch("librarian.processor_core.extraction_runner.llm.ensure_available"), \
         patch("librarian.processor_core.extraction_runner.llm.chat") as mock_chat:
        # Mock LLM calls matched by task_name
        def side_effect_chat(messages, schema=None, task_name=None):
            if task_name and "summary" in task_name:
                return mock_summary
            if task_name and "mindmap" in task_name:
                return mock_mindmap
            return None

        mock_chat.side_effect = side_effect_chat

        status, saved = wp.run_pipeline(f"{unit_name}.md", categories=["summary", "mindmap"])
        assert "Pipeline completed" in status
        assert len(saved) >= 2

        extractions_dir = tmp_path / "wiki" / unit_name / "extractions"
        assert (extractions_dir / f"{unit_name}_summary.md").exists()
        assert (extractions_dir / f"{unit_name}_mindmap.json").exists()
        assert (extractions_dir / f"{unit_name}_mindmap.html").exists()
