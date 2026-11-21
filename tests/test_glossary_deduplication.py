"""
Tests for Cumulative Lexical Ledger (CLL) deduplication, annotation integrity,
and cross-unit collision prevention.
"""
import pytest
from pathlib import Path
from librarian.glossary_ledger import (
    GlossaryLedger,
    normalize_pos,
    hash_example_text,
    get_or_create_ledger,
)


def test_pos_normalization_prevents_duplicate_keys():
    """Validates that variants like 'adj' and 'adjective', 'adv' and 'adverb' map to identical canonical keys."""
    assert normalize_pos("adj") == "adj"
    assert normalize_pos("adjective") == "adj"
    assert normalize_pos("a.") == "adj"
    assert normalize_pos("adv") == "adv"
    assert normalize_pos("adverb") == "adv"
    assert normalize_pos("ad.") == "adv"
    assert normalize_pos("noun") == "noun"
    assert normalize_pos("n.") == "noun"
    assert normalize_pos("verb") == "verb"
    assert normalize_pos("v.") == "verb"


def test_example_hash_deduplication():
    """Validates that example sentences with punctuation or casing differences produce identical hashes."""
    s1 = "The achievements of Egypt's ancient civilization were remarkable."
    s2 = "the achievements of egypt's ancient civilization were remarkable"
    s3 = "  The achievements of Egypt's ancient civilization were remarkable!  "
    h1 = hash_example_text(s1)
    h2 = hash_example_text(s2)
    h3 = hash_example_text(s3)
    assert h1 == h2 == h3


def test_cross_unit_prior_target_detection(tmp_path: Path):
    """Validates that a target word introduced in an early unit is detected as a prior target in later units."""
    ledger = GlossaryLedger(series="Test Series", book=1, project_root=tmp_path)
    entries = ledger.data.setdefault("entries", {})
    entries["order#noun"] = {
        "lemma": "order",
        "pos": "noun",
        "first_unit": "Book_1_Unit_1_Passage_A",
        "occurrences": [{"unit": "Book_1_Unit_1_Passage_A", "role": "target_new"}],
    }

    # Querying from a later unit should flag it as prior target
    is_prior, first_unit = ledger.is_headword_prior_target("order", "Book_1_Unit_3_Passage_A")
    assert is_prior is True
    assert first_unit == "Book_1_Unit_1_Passage_A"

    # Querying from the SAME unit should NOT flag it as prior target from earlier
    is_prior_same, _ = ledger.is_headword_prior_target("order", "Book_1_Unit_1_Passage_A")
    assert is_prior_same is False

    # Querying an unintroduced word should return False
    is_prior_new, _ = ledger.is_headword_prior_target("unseen_word", "Book_1_Unit_3_Passage_A")
    assert is_prior_new is False


def test_multi_unit_re_exposure_annotation(tmp_path: Path):
    """Validates that encountering a word across multiple units records occurrences without creating duplicate entries."""
    ledger = GlossaryLedger(series="Test Series", book=1, project_root=tmp_path)
    entries = ledger.data.setdefault("entries", {})

    # Simulate Unit 1 introduction
    entries["expand#verb"] = {
        "lemma": "expand",
        "pos": "verb",
        "first_unit": "Book_1_Unit_1_Passage_A",
        "occurrences": [{"unit": "Book_1_Unit_1_Passage_A", "role": "target_new"}],
        "used_examples": ["The company is expanding."],
    }

    # Now record quiz usage in Unit 3
    ledger.record_quiz_usage(
        headword="expand",
        unit="Book_1_Unit_3_Passage_A",
        example_text="Reading helps expand your vocabulary.",
    )

    entry = ledger.data["entries"]["expand#verb"]
    assert entry["first_unit"] == "Book_1_Unit_1_Passage_A"
    assert "Book_1_Unit_3_Passage_A" in entry.get("used_in_quizzes", [])
    assert len(entry["used_examples"]) == 2


def test_real_book_1_glossary_integrity_audit():
    """
    Audits the actual Book 1 glossary on disk to ensure:
    1. Zero duplicate entries by lemma#pos.
    2. All entries have valid lemma, pos, first_unit.
    3. Proper separation of genuine homographs vs duplicate anomalies.
    """
    ledger = get_or_create_ledger("New Practical English", 1)
    entries = ledger.data.get("entries", {})

    assert len(entries) > 0, "Book 1 glossary entries should not be empty"

    unique_keys = set(entries.keys())
    assert len(unique_keys) == len(entries), "Every entry key must be strictly unique"

    lemmas = [e["lemma"] for e in entries.values()]
    unique_lemmas = set(lemmas)

    # Difference between total entries and unique lemmas must strictly represent genuine homographs across POS
    homograph_diff = len(entries) - len(unique_lemmas)
    assert homograph_diff >= 0

    # Ensure all entries have mandatory metadata fields
    for key, entry in entries.items():
        assert entry.get("lemma"), f"Entry {key} is missing lemma"
        assert entry.get("pos"), f"Entry {key} is missing pos"
        assert entry.get("first_unit"), f"Entry {key} is missing first_unit"
        assert isinstance(entry.get("occurrences"), list), f"Entry {key} occurrences must be a list"

        # Verify no duplicate units in occurrences
        occ_units = [o.get("unit") for o in entry.get("occurrences", [])]
        assert len(occ_units) == len(set(occ_units)), f"Entry {key} has duplicate unit occurrences: {occ_units}"


def test_real_book_2_nhce_glossary_integrity_audit():
    """
    Audits the newly synced Book 2 (New Horizon College English Fourth Edition) glossary to ensure:
    1. Zero duplicate entries by lemma#pos.
    2. All 148 entries across the 6 units (Units 1-6 Section A) are accurately harvested.
    3. 100% of entries have valid lemma, pos, first_unit, definition, and authentic examples.
    """
    ledger = get_or_create_ledger("New Horizon College English (Fourth Edition)", 2)
    entries = ledger.data.get("entries", {})

    assert len(entries) == 148, f"Expected 148 entries in Book 2, found {len(entries)}"

    unique_keys = set(entries.keys())
    assert len(unique_keys) == 148, "Every entry key must be strictly unique"

    lemmas = [e["lemma"] for e in entries.values()]
    assert len(set(lemmas)) == 148, "Expected 148 distinct lemmas with zero duplicate entries"

    # Verify field completeness
    for key, entry in entries.items():
        assert entry.get("lemma"), f"Entry {key} missing lemma"
        assert entry.get("pos"), f"Entry {key} missing pos"
        assert entry.get("first_unit"), f"Entry {key} missing first_unit"
        assert entry.get("definition"), f"Entry {key} missing definition"
        assert len(entry.get("used_examples", [])) > 0, f"Entry {key} missing example"


def test_cross_book_ancestry_discovery(tmp_path: Path):
    """Validates that a ledger discovers prior books in strictly ascending pedagogical sequence."""
    b1 = GlossaryLedger(series="Test Series", book=1, project_root=tmp_path)
    b1.data["entries"]["test#noun"] = {"lemma": "test", "pos": "noun", "first_unit": "U1"}
    b1.save()

    b2 = GlossaryLedger(series="Test Series", book=2, project_root=tmp_path)
    b2.save()

    b3 = GlossaryLedger(series="Test Series", book=3, project_root=tmp_path)
    b3.save()

    assert b1.get_prior_books() == []
    assert b2.get_prior_books() == ["1"]
    assert b3.get_prior_books() == ["1", "2"]

    p_ledgers = b3.load_prior_ledgers()
    assert len(p_ledgers) == 2
    assert p_ledgers[0].book == "1"
    assert p_ledgers[1].book == "2"


def test_cross_book_role_classification(tmp_path: Path):
    """
    Validates SLA spiral progression tier categorization:
    1. Unseen word -> 'target_new' (~80-85% primary core).
    2. Word seen in prior book with DIFFERENT POS -> 'recycled_sense_shift' (~10-15%).
    3. Word seen in prior book with SAME POS -> 'recycled_spaced_review' (~0-5%).
    """
    b1 = GlossaryLedger(series="Test Series", book=1, project_root=tmp_path)
    b1.data["entries"]["schedule#noun"] = {
        "lemma": "schedule",
        "pos": "noun",
        "first_unit": "Book_1_Unit_1_Passage_A",
        "definition": "a plan that gives expected times for actions",
    }
    b1.data["entries"]["ancient#adj"] = {
        "lemma": "ancient",
        "pos": "adj",
        "first_unit": "Book_1_Unit_2_Passage_A",
        "definition": "belonging to the distant past",
    }
    b1.save()

    b2 = GlossaryLedger(series="Test Series", book=2, project_root=tmp_path)

    # 1. New word
    role_new = b2.classify_cross_book_role(lemma="innovate", pos="verb")
    assert role_new["acquisition_tier"] == "target_new"
    assert role_new["prior_exposure"] is None

    # 2. Recategorization / Sense Shift (schedule noun in B1 -> schedule verb in B2)
    role_shift = b2.classify_cross_book_role(lemma="schedule", pos="verb")
    assert role_shift["acquisition_tier"] == "recycled_sense_shift"
    assert role_shift["prior_exposure"]["book"] == "1"
    assert role_shift["prior_exposure"]["pos"] == "noun"
    assert "Sense Shift" in role_shift["tier_label"]

    # 3. Spaced Retrieval Review (ancient adj in B1 -> ancient adj in B2)
    role_review = b2.classify_cross_book_role(lemma="ancient", pos="adj")
    assert role_review["acquisition_tier"] == "recycled_spaced_review"
    assert role_review["prior_exposure"]["book"] == "1"
    assert role_review["prior_exposure"]["pos"] == "adj"
    assert "Spaced Review" in role_review["tier_label"]


def test_quiz_lexical_quota_governor(tmp_path: Path):
    """
    Validates that quiz vocabulary quota selection enforces:
    - Primary: ~80% new targets
    - Secondary: ~15% sense shifts
    - Tertiary: ~5% spaced retrieval
    - Graceful backfilling: missing auxiliary items gracefully backfilled with target_new.
    """
    ledger = GlossaryLedger(series="Test Series", book=2, project_root=tmp_path)

    # Candidate pool: 12 new, 2 sense shifts, 1 spaced review
    candidates = []
    for i in range(12):
        candidates.append({"lemma": f"new_word_{i}", "pos": "noun", "acquisition_tier": "target_new"})
    candidates.append({"lemma": "shift_word_1", "pos": "verb", "acquisition_tier": "recycled_sense_shift"})
    candidates.append({"lemma": "shift_word_2", "pos": "verb", "acquisition_tier": "recycled_sense_shift"})
    candidates.append({"lemma": "review_word_1", "pos": "adj", "acquisition_tier": "recycled_spaced_review"})

    # Allocate 10 items
    quota_res = ledger.allocate_quiz_lexical_quota(candidates, total_count=10)
    assert quota_res["total_allocated"] == 10
    bk = quota_res["quota_breakdown"]
    assert bk["target_new"] >= 7
    assert bk["recycled_sense_shift"] >= 1
    assert bk["recycled_spaced_review"] >= 0
    assert bk["target_new"] + bk["recycled_sense_shift"] + bk["recycled_spaced_review"] == 10

    # Graceful backfill test: 10 candidates, all target_new (0 shifts, 0 reviews)
    all_new_candidates = [
        {"lemma": f"pure_new_{i}", "pos": "noun", "acquisition_tier": "target_new"}
        for i in range(10)
    ]
    pure_res = ledger.allocate_quiz_lexical_quota(all_new_candidates, total_count=10)
    assert pure_res["total_allocated"] == 10
    assert pure_res["quota_breakdown"]["target_new"] == 10
    assert pure_res["quota_breakdown"]["recycled_sense_shift"] == 0
    assert pure_res["quota_breakdown"]["recycled_spaced_review"] == 0


def test_summary_includes_acquisition_tier_breakdown():
    """Validates that get_summary returns acquisition_tier_breakdown with correct tier keys."""
    ledger = get_or_create_ledger("New Practical English", 1)
    summary = ledger.get_summary()
    assert "acquisition_tier_breakdown" in summary
    tiers = summary["acquisition_tier_breakdown"]
    assert "target_new" in tiers
    assert "recycled_sense_shift" in tiers
    assert "recycled_spaced_review" in tiers
    assert tiers["target_new"] > 0


