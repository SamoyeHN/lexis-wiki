"""End-to-End Regression Suite across the 5 Canonical Golden Set Units (A1-C1)."""

from __future__ import annotations

import pytest

from librarian.golden_set import (
    GOLDEN_SET_REGISTRY,
    get_all_golden_units,
    run_golden_set_benchmarks,
    validate_golden_unit_files,
)
from librarian.linguistics import LinguisticEngine


def test_golden_set_registry_integrity():
    """Verify registry structure, CEFR progression, and spec contracts."""
    specs = get_all_golden_units()
    assert len(specs) == 5

    expected_levels = ["A1", "A2", "B1", "B2", "C1"]
    actual_levels = [s.cefr_level for s in specs]
    assert actual_levels == expected_levels

    for s in specs:
        assert s.unit_name.startswith("Book_")
        assert len(s.expected_extractions) == 5
        assert len(s.expected_handouts) == 5
        assert s.word_count >= 200


@pytest.mark.parametrize("unit_slug", list(GOLDEN_SET_REGISTRY.keys()))
def test_golden_unit_end_to_end(unit_slug: str):
    """End-to-end regression test for each canonical Golden Set unit.

    Verifies:
      1. Physical asset completeness (source, 5 extractions, 5 handouts).
      2. Deterministic CEFR calibration against ground-truth CEFR level.
      3. Level 1 extraction code gates (zero defects, 100 score).
      4. Level 1 handout code gates across all 5 modalities (<5ms, zero defects).
    """
    spec = GOLDEN_SET_REGISTRY[unit_slug]

    # 1. File verification
    file_res = validate_golden_unit_files(spec)
    assert file_res["is_complete"] is True, f"Missing files in {unit_slug}: {file_res}"

    # 2. Benchmarking execution
    summary = run_golden_set_benchmarks(units=[unit_slug])
    assert summary["total_units"] == 1
    assert summary["passed_units"] == 1
    assert summary["overall_status"] == "passed"

    unit_res = summary["units"][0]
    assert unit_res["passed"] is True
    assert unit_res["defects"] == []
    assert unit_res["cefr_match"] is True
    assert unit_res["calibrated_cefr"] == spec.cefr_level
    assert unit_res["extraction_score"] == 100.0
    assert unit_res["handouts_audited"] == 5


def test_golden_set_full_benchmark_suite():
    """Verify the entire Golden Set pipeline passes with zero defects across all 25 handouts."""
    summary = run_golden_set_benchmarks()
    assert summary["total_units"] == 5
    assert summary["passed_units"] == 5
    assert summary["failed_units"] == 0
    assert summary["total_handouts"] == 25
    assert summary["handout_defects"] == 0
    assert summary["extraction_defects"] == 0
    assert summary["overall_status"] == "passed"
    # L1 code gate invariant: <5ms execution speed per item
    assert summary["elapsed_ms"] < 10000.0
