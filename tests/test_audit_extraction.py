"""Unit tests for the Extraction Semantic Re-Audit Suite."""

from __future__ import annotations

import json
from pathlib import Path

from librarian.audit_extraction import (
    audit_grammar_markdown,
    audit_mindmap_json,
    audit_summary_markdown,
    audit_unit_extractions,
    audit_vocabulary_markdown,
)


def test_audit_vocabulary_clean_and_defects():
    source = "The quick brown fox jumps over the lazy dog in the sunny morning."

    clean_vocab = """
# Vocabulary
## [[fox]] noun B1
- **Definition**: a wild animal like a dog with red-brown fur and a bushy tail
- **Quoted Sentence**: The quick brown fox jumps over the lazy dog in the sunny morning.
"""
    defects, stats = audit_vocabulary_markdown(clean_vocab, source.lower())
    assert defects == []
    assert stats["item_count"] == 1
    assert "fox" in stats["lemmas"]

    hallucinated_vocab = """
# Vocabulary
## [[wolf]] noun B1
- **Definition**: a wild carnivorous mammal
- **Quoted Sentence**: A dangerous wolf roamed freely in the dark northern forest.
"""
    defects, stats = audit_vocabulary_markdown(hallucinated_vocab, source.lower())
    assert len(defects) == 1
    assert "quote not found in source text" in defects[0]


def test_audit_grammar_clean_and_defects():
    source = "With the coming of the telephone, people could not only read words, but also hear each other."

    clean_grammar = """
# Grammar
## [[Correlative Coordination]]
- **Quote**: With the coming of the telephone, people could not only read words, but also hear each other.
- **Pattern Formula**: [S] + not only + [VP], but also + [VP]
- **Pedagogical Function**: Balances two equal ideas or actions together.
"""
    defects, stats = audit_grammar_markdown(clean_grammar, source.lower())
    assert defects == []
    assert stats["item_count"] == 1

    bad_formula_grammar = """
# Grammar
## [[Correlative Coordination]]
- **Quote**: With the coming of the telephone, people could not only read words, but also hear each other.
- **Pattern Formula**: not only but also
- **Pedagogical Function**: Balances two equal ideas or actions together.
"""
    defects, stats = audit_grammar_markdown(bad_formula_grammar, source.lower())
    assert any("formula lacks Quirk/Biber syntactic markers" in d for d in defects)


def test_audit_summary_clean_and_defects():
    source = (
        "The passage explores modern communication technology from mail to internet software. "
        "Young people seek balance between social connection, family love, personal autonomy, and privacy. "
        "New tools provide freedom and smarter choices when all generations communicate."
    )

    clean_summary = """
# Summary
## Narrative Overview
The passage discusses modern communication technology moving from traditional mail to internet software.
Young people navigate personal freedom, family connection, and privacy.

## Lesson Hook
How has digital communication transformed your personal freedom?

## Essential Questions
- How do communication tools evolve over time?
"""
    defects, stats = audit_summary_markdown(clean_summary, source.lower())
    assert defects == []
    assert stats["has_narrative"] is True
    assert stats["has_hook"] is True
    assert stats["has_questions"] is True

    missing_sections_summary = """
# Summary
## Quick Notes
Some random notes without narrative overview or essential questions.
"""
    defects, _ = audit_summary_markdown(missing_sections_summary, source.lower())
    assert any("Missing 'Narrative Overview'" in d for d in defects)
    assert any("Missing 'Lesson Hook'" in d for d in defects)
    assert any("Missing 'Essential Questions'" in d for d in defects)


def test_audit_mindmap_json_mece(tmp_path: Path):
    clean_mm = {
        "title": "Clean Mindmap",
        "root_name": "Root Concept",
        "branches": [
            {
                "branch_name": "Historical Background",
                "sub_branches": [
                    {"sub_branch_name": "Early Period", "leaves": ["Item 1", "Item 2"]},
                    {"sub_branch_name": "Modern Period", "leaves": ["Item 3"]},
                ],
            },
            {
                "branch_name": "Sociological Impact",
                "sub_branches": [
                    {"sub_branch_name": "Family Relations", "leaves": ["Item 4"]},
                    {"sub_branch_name": "Peer Relations", "leaves": ["Item 5"]},
                ],
            },
        ],
    }
    mm_file = tmp_path / "clean_mindmap.json"
    mm_file.write_text(json.dumps(clean_mm), encoding="utf-8")

    defects, stats = audit_mindmap_json(mm_file)
    assert defects == []
    assert stats["branch_count"] == 2
    assert stats["max_depth"] >= 2

    # Duplicate branch MECE violation
    dup_mm = {
        "title": "Dup Mindmap",
        "root_name": "Root Concept",
        "branches": [
            {"branch_name": "History", "sub_branches": []},
            {"branch_name": "History", "sub_branches": []},
        ],
    }
    dup_file = tmp_path / "dup_mindmap.json"
    dup_file.write_text(json.dumps(dup_mm), encoding="utf-8")

    defects, _ = audit_mindmap_json(dup_file)
    assert any("MECE violation: Duplicate branch name" in d for d in defects)


def test_audit_unit_extractions_golden_unit():
    report = audit_unit_extractions("Book_1_Unit_1_Passage_A")
    assert report.passed is True
    assert report.score == 100.0
    assert report.defects == []
    assert "vocabulary" in report.stats
    assert "grammar" in report.stats
    assert "summary" in report.stats
    assert "mindmap" in report.stats
