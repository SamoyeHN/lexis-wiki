"""Smoke and contract tests for librarian.exporter."""

import zipfile
from pathlib import Path
from unittest.mock import patch

from librarian import exporter


def test_format_inline_lexis_markdown():
    rendered = exporter.format_inline_lexis_markdown(
        "Use **bold** and [[target_word]] in `code`."
    )
    assert "<strong>bold</strong>" in rendered
    assert "target_word" in rendered
    assert '<code class="lex-inline-code">code</code>' in rendered


def test_render_markdown_to_standalone_html():
    md = "# Demo Reading\n\nParagraph text.\n"
    html = exporter.render_markdown_to_standalone_html(
        md, fallback_title="Fallback", category="source"
    )
    assert "<html" in html.lower()
    assert "Demo Reading" in html
    assert "Paragraph text." in html


def test_export_unit_to_directory_and_zip(tmp_path):
    unit_name = "Demo_Unit"
    unit_dir = tmp_path / "wiki" / unit_name
    (unit_dir / "sources").mkdir(parents=True)
    (unit_dir / "extractions").mkdir(parents=True)
    (unit_dir / "handouts").mkdir(parents=True)

    (unit_dir / "sources" / f"{unit_name}.md").write_text(
        "---\ntitle: Demo Unit Title\n---\n\nThis is a sample passage.\n",
        encoding="utf-8",
    )
    (unit_dir / "extractions" / f"{unit_name}_vocabulary.md").write_text(
        "# Vocabulary\n\n## [[sample]]\n- **Definition**: an example\n",
        encoding="utf-8",
    )
    (unit_dir / "handouts" / f"{unit_name}_cloze_quiz.html").write_text(
        "<!DOCTYPE html><html><body><p>Cloze Quiz</p></body></html>",
        encoding="utf-8",
    )

    out_dir = tmp_path / "out_export"

    with patch.object(exporter.config, "project_root", tmp_path):
        res_dir = exporter.export_unit_to_directory(unit_name, custom_target_dir=out_dir)
        assert res_dir["success"] is True
        export_path = Path(res_dir["export_path"])
        assert (export_path / "index.html").exists()
        assert (export_path / "extractions" / f"{unit_name}_source.html").exists()
        assert (export_path / "extractions" / f"{unit_name}_vocabulary.html").exists()
        assert (export_path / "handouts" / f"{unit_name}_cloze_quiz.html").exists()

        res_zip = exporter.export_unit_to_zip(unit_name, custom_target_dir=out_dir)
        assert res_zip["success"] is True
        zip_path = Path(res_zip["zip_path"])
        assert zip_path.exists()
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = set(zf.namelist())
        assert "index.html" in names
        assert f"handouts/{unit_name}_cloze_quiz.html" in names


def test_export_missing_unit_returns_error(tmp_path):
    with patch.object(exporter.config, "project_root", tmp_path):
        res = exporter.export_unit_to_directory("NonExistent_Unit_XYZ")
    assert res["success"] is False
    assert "not found" in res["error"].lower()
