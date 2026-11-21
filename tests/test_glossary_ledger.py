"""
Unit tests for Cumulative Lexical Ledger (GlossaryLedger).
Tests bottom-up harvest from wiki units, top-down official PDF ingestion,
and collision prevention (prior targets and reused examples).
"""

import json
from pathlib import Path
import pytest

from librarian.glossary_ledger import (
    GlossaryLedger,
    normalize_series_slug,
    normalize_pos,
    hash_example_text,
    get_or_create_ledger,
)


def test_normalize_series_slug():
    assert normalize_series_slug("New Practical English") == "New_Practical_English"
    assert normalize_series_slug("New Horizon (4th Edition)") == "New_Horizon_4th_Edition"
    assert normalize_series_slug("  Leading Edge / ESP  ") == "Leading_Edge_ESP"
    assert normalize_series_slug("!!!") == "General_Series"


def test_normalize_pos():
    assert normalize_pos("n.") == "noun"
    assert normalize_pos("noun") == "noun"
    assert normalize_pos("v.") == "verb"
    assert normalize_pos("vt.") == "verb"
    assert normalize_pos("vi.") == "verb"
    assert normalize_pos("verb") == "verb"
    assert normalize_pos("a.") == "adj"
    assert normalize_pos("adj.") == "adj"
    assert normalize_pos("adjective") == "adj"
    assert normalize_pos("ad.") == "adv"
    assert normalize_pos("adv.") == "adv"
    assert normalize_pos("adverb") == "adv"
    assert normalize_pos("prep.") == "prep"
    assert normalize_pos("conj.") == "conj"


def test_hash_example_text():
    ex1 = "Send a stamped, self-addressed envelope."
    ex2 = "send a stamped self addressed envelope"
    assert hash_example_text(ex1) == hash_example_text(ex2)
    assert len(hash_example_text(ex1)) == 12


def test_ledger_initialization_and_persistence(tmp_path):
    ledger = GlossaryLedger("Test Series", 1, project_root=tmp_path)
    assert ledger.series == "Test Series"
    assert ledger.book == "1"
    assert ledger.slug == "Test_Series"

    saved_path = ledger.save()
    assert saved_path.exists()

    # Reload from disk
    ledger2 = GlossaryLedger("Test Series", 1, project_root=tmp_path)
    assert ledger2.data["series"] == "Test Series"
    assert ledger2.data["book"] == "1"


def test_sync_bottom_up(tmp_path):
    # Setup mock wiki structure
    wiki_dir = tmp_path / "wiki"
    u1_dir = wiki_dir / "Book_1_Unit_1_Passage_A"
    u1_src = u1_dir / "sources" / "Book_1_Unit_1_Passage_A.md"
    u1_voc = u1_dir / "extractions" / "Book_1_Unit_1_Passage_A_vocabulary.md"

    u1_src.parent.mkdir(parents=True, exist_ok=True)
    u1_voc.parent.mkdir(parents=True, exist_ok=True)

    u1_src.write_text(
        '---\nseries: "Test Series"\nbook: 1\nunit: 1\n---\n# Content',
        encoding="utf-8",
    )
    u1_voc.write_text(
        """# Vocabulary: Book 1 Unit 1 Passage A

## [[stamp]] noun B1 `[countable]`
- **Definition**: a small piece of paper that you buy and stick onto an envelope
- **Example Usage**: Send a stamped, self-addressed envelope.

## [[control]] noun A2 `[uncountable]`
- **Definition**: the power to make someone or something do what you want
- **Example Usage**: The disease robs you of muscle control.
""",
        encoding="utf-8",
    )

    ledger = GlossaryLedger("Test Series", 1, project_root=tmp_path)
    res = ledger.sync_bottom_up()

    assert res["status"] == "success"
    assert res["matching_units_scanned"] == 1
    assert res["harvested_count"] == 2

    # Check entries
    entries = ledger.data["entries"]
    assert "stamp#noun" in entries
    assert entries["stamp#noun"]["first_unit"] == "Book_1_Unit_1_Passage_A"
    assert entries["stamp#noun"]["cefr"] == "B1"

    # Check summary
    summary = ledger.summary()
    assert summary["total_entries"] == 2
    assert summary["cefr_breakdown"]["B1"] == 1
    assert summary["cefr_breakdown"]["A2"] == 1


def test_collision_guards(tmp_path):
    ledger = GlossaryLedger("Test Series", 1, project_root=tmp_path)
    ledger.data["entries"] = {
        "control#noun": {
            "lemma": "control",
            "pos": "noun",
            "first_unit": "Book_1_Unit_1_Passage_A",
        }
    }
    ledger.data["used_example_hashes"] = {
        hash_example_text("The disease robs you of muscle control."): {
            "unit": "Book_1_Unit_1_Passage_A",
            "lemma": "control",
        }
    }

    # Test headword prior target check
    is_prior, first_u = ledger.is_headword_prior_target("control", "Book_1_Unit_2_Passage_A")
    assert is_prior is True
    assert first_u == "Book_1_Unit_1_Passage_A"

    is_prior_same, _ = ledger.is_headword_prior_target("control", "Book_1_Unit_1_Passage_A")
    assert is_prior_same is False

    is_prior_unknown, _ = ledger.is_headword_prior_target("unknown", "Book_1_Unit_2_Passage_A")
    assert is_prior_unknown is False

    # Test example collision check
    used = ledger.is_example_used("The disease robs you of muscle control.")
    assert used is True

    unused = ledger.is_example_used("He had no control over his feelings.")
    assert unused is False

    # Test recording quiz usage
    ledger.record_quiz_usage("feeling", "Book_1_Unit_2_Passage_A", "He had no control over his feelings.")
    assert ledger.is_example_used("He had no control over his feelings.") is True


def test_pdf_import_real_sample():
    sample_pdf = Path("/mnt/d/Huanan/教案/新编实用英语综合教程/Book 1/【Glossary】新编实用英语（第五版）（修订版）综合教程1.pdf")
    if not sample_pdf.exists():
        pytest.skip("Local sample PDF not available in this environment")

    ledger = GlossaryLedger("New Practical English", 1)
    res = ledger.import_pdf_glossary(sample_pdf)

    assert res["status"] == "success"
    assert res["total_entries"] >= 430
    assert "control#noun" in ledger.data["entries"]

    entry = ledger.data["entries"]["control#noun"]
    assert entry["lemma"] == "control"
    assert entry["display_word"] == "control"
    assert entry["pos"] == "noun"
    assert "ipa" not in entry
    assert "chinese" not in entry
    assert entry["first_unit"] == "Book_1_Unit_1_Passage_A"
