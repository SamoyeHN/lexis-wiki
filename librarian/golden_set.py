"""Golden Set End-to-End Regression Suite.

Curates 5 canonical units spanning CEFR levels A1-C1 with diverse discourse genres
and complete multi-modality artifacts (sources, 5 extractions, 5 quiz handouts).
Provides zero-token L1 deterministic code gates and regression benchmarking
across the entire Lexis Wiki neuro-symbolic pipeline.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .audit_batch import load_quiz_data, resolve_source_text
from .audit_extraction import audit_unit_extractions
from .config import config
from .linguistics import LinguisticEngine
from .processor import WikiProcessor

logger = logging.getLogger("librarian.golden_set")


@dataclass(frozen=True)
class GoldenUnitSpec:
    """Specification of a canonical Golden Set regression unit."""

    unit_name: str
    cefr_level: str
    genre: str
    word_count: int
    min_vocab: int
    min_grammar: int
    min_expressions: int
    expected_extractions: List[str] = field(
        default_factory=lambda: [
            "vocabulary.md",
            "grammar.md",
            "summary.md",
            "mindmap.json",
            "mindmap.html",
        ]
    )
    expected_handouts: List[str] = field(
        default_factory=lambda: [
            "vocabulary_quiz.html",
            "cloze_quiz.html",
            "reading_quiz.html",
            "translation_quiz.html",
            "listening_quiz.html",
        ]
    )
    description: str = ""


GOLDEN_SET_REGISTRY: Dict[str, GoldenUnitSpec] = {
    "Book_1_Unit_1_Passage_A": GoldenUnitSpec(
        unit_name="Book_1_Unit_1_Passage_A",
        cefr_level="A1",
        genre="Dialogue & Interpersonal Narrative",
        word_count=272,
        min_vocab=20,
        min_grammar=4,
        min_expressions=3,
        description="A1 foundational dialogue and interpersonal communication narrative.",
    ),
    "Book_1_Unit_2_Passage_A": GoldenUnitSpec(
        unit_name="Book_1_Unit_2_Passage_A",
        cefr_level="A2",
        genre="Childhood & Family Narrative",
        word_count=450,
        min_vocab=20,
        min_grammar=4,
        min_expressions=3,
        description="A2 autobiographical childhood memory and family interaction narrative.",
    ),
    "Book_1_Unit_1_Section_A": GoldenUnitSpec(
        unit_name="Book_1_Unit_1_Section_A",
        cefr_level="B1",
        genre="Informative / Explanatory Essay",
        word_count=915,
        min_vocab=20,
        min_grammar=4,
        min_expressions=3,
        description="B1 intermediate informative and explanatory essay on education and social habits.",
    ),
    "Book_3_Unit_6_Section_A": GoldenUnitSpec(
        unit_name="Book_3_Unit_6_Section_A",
        cefr_level="B2",
        genre="Business, Economics & Financial Analysis",
        word_count=1024,
        min_vocab=20,
        min_grammar=4,
        min_expressions=3,
        description="B2 upper-intermediate socio-economic and business market analysis.",
    ),
    "Book_4_Unit_1_Section_A": GoldenUnitSpec(
        unit_name="Book_4_Unit_1_Section_A",
        cefr_level="C1",
        genre="Advanced Academic / Cultural Discourse",
        word_count=1050,
        min_vocab=20,
        min_grammar=4,
        min_expressions=3,
        description="C1 advanced academic, philosophical, and cultural criticism discourse.",
    ),
}


def get_golden_unit(name: str) -> Optional[GoldenUnitSpec]:
    """Retrieve a golden unit specification by its directory slug."""
    return GOLDEN_SET_REGISTRY.get(name)


def get_all_golden_units() -> List[GoldenUnitSpec]:
    """Retrieve all golden unit specifications in ascending order."""
    return list(GOLDEN_SET_REGISTRY.values())


def _project_root(project_root: Optional[str]) -> Path:
    return Path(project_root) if project_root else Path(config.project_root)


def validate_golden_unit_files(
    unit_spec: GoldenUnitSpec, project_root: Optional[str] = None
) -> Dict[str, Any]:
    """Verify that all physical source, extraction, and handout assets exist on disk."""
    root = _project_root(project_root)
    unit_dir = root / "wiki" / unit_spec.unit_name

    source_path = unit_dir / "sources" / f"{unit_spec.unit_name}.md"
    source_exists = source_path.is_file()

    extractions_dir = unit_dir / "extractions"
    missing_extractions: List[str] = []
    for ext_suffix in unit_spec.expected_extractions:
        target = extractions_dir / f"{unit_spec.unit_name}_{ext_suffix}"
        if not target.is_file():
            missing_extractions.append(target.name)

    handouts_dir = unit_dir / "handouts"
    missing_handouts: List[str] = []
    for handout_suffix in unit_spec.expected_handouts:
        target = handouts_dir / f"{unit_spec.unit_name}_{handout_suffix}"
        if not target.is_file():
            missing_handouts.append(target.name)

    is_complete = source_exists and not missing_extractions and not missing_handouts

    return {
        "unit": unit_spec.unit_name,
        "is_complete": is_complete,
        "source_exists": source_exists,
        "missing_extractions": missing_extractions,
        "missing_handouts": missing_handouts,
    }


def audit_handout_code_gates(
    unit_name: str, source_text: str, project_root: Optional[str] = None
) -> Tuple[int, List[str]]:
    """Run Level 1 zero-token deterministic code gates across all 5 quiz handouts for a unit."""
    root = _project_root(project_root)
    handouts_dir = root / "wiki" / unit_name / "handouts"
    defects: List[str] = []
    audited_count = 0

    # 1. Vocabulary
    vocab_p = handouts_dir / f"{unit_name}_vocabulary_quiz.html"
    if vocab_p.is_file():
        audited_count += 1
        q_data = load_quiz_data(str(vocab_p))
        if q_data:
            _, msgs = WikiProcessor.audit_quiz_integrity(q_data)
            defects.extend([f"[{unit_name}:vocab] {m}" for m in msgs])
        else:
            defects.append(f"[{unit_name}:vocab] Failed to load quizData from {vocab_p.name}")

    # 2. Cloze
    cloze_p = handouts_dir / f"{unit_name}_cloze_quiz.html"
    if cloze_p.is_file():
        audited_count += 1
        q_data = load_quiz_data(str(cloze_p))
        if q_data:
            _, msgs = WikiProcessor.audit_cloze_integrity(q_data)
            defects.extend([f"[{unit_name}:cloze] {m}" for m in msgs])
        else:
            defects.append(f"[{unit_name}:cloze] Failed to load quizData from {cloze_p.name}")

    # 3. Reading
    reading_p = handouts_dir / f"{unit_name}_reading_quiz.html"
    if reading_p.is_file():
        audited_count += 1
        q_data = load_quiz_data(str(reading_p))
        if q_data:
            _, msgs = WikiProcessor.audit_reading_integrity(q_data, source_text)
            defects.extend([f"[{unit_name}:reading] {m}" for m in msgs])
        else:
            defects.append(f"[{unit_name}:reading] Failed to load quizData from {reading_p.name}")

    # 4. Translation
    trans_p = handouts_dir / f"{unit_name}_translation_quiz.html"
    if trans_p.is_file():
        audited_count += 1
        q_data = load_quiz_data(str(trans_p))
        if q_data:
            _, msgs = WikiProcessor.audit_translation_integrity(q_data)
            defects.extend([f"[{unit_name}:translation] {m}" for m in msgs])
        else:
            defects.append(f"[{unit_name}:translation] Failed to load quizData from {trans_p.name}")

    # 5. Listening
    list_p = handouts_dir / f"{unit_name}_listening_quiz.html"
    if list_p.is_file():
        audited_count += 1
        q_data = load_quiz_data(str(list_p))
        if q_data:
            _, msgs = WikiProcessor.audit_listening_integrity(q_data, source_text)
            defects.extend([f"[{unit_name}:listening] {m}" for m in msgs])
        else:
            defects.append(f"[{unit_name}:listening] Failed to load quizData from {list_p.name}")

    return audited_count, defects


def run_golden_set_benchmarks(
    units: Optional[List[str]] = None,
    project_root: Optional[str] = None,
    fast: bool = False,
    enable_llm: bool = False,
) -> Dict[str, Any]:
    """Execute end-to-end regression validation and benchmarking across Golden Set units."""
    start_time = time.perf_counter()

    if units:
        target_specs = [
            GOLDEN_SET_REGISTRY[u]
            for u in units
            if u in GOLDEN_SET_REGISTRY
        ]
    else:
        target_specs = list(GOLDEN_SET_REGISTRY.values())

    results: List[Dict[str, Any]] = []
    total_handouts = 0
    total_handout_defects = 0
    total_extraction_defects = 0

    for spec in target_specs:
        unit_start = time.perf_counter()
        unit_defects: List[str] = []

        # Step 1: File integrity validation
        file_check = validate_golden_unit_files(spec, project_root)
        if not file_check["is_complete"]:
            if not file_check["source_exists"]:
                unit_defects.append("Missing primary source file.")
            if file_check["missing_extractions"]:
                unit_defects.append(f"Missing extractions: {', '.join(file_check['missing_extractions'])}")
            if file_check["missing_handouts"]:
                unit_defects.append(f"Missing handouts: {', '.join(file_check['missing_handouts'])}")

        # Step 2: Source text & CEFR calibration verification
        source_text = resolve_source_text(spec.unit_name, project_root)
        actual_cefr = LinguisticEngine.calculate_text_cefr(source_text) if source_text else "UNKNOWN"
        cefr_match = actual_cefr == spec.cefr_level
        if not cefr_match:
            unit_defects.append(f"CEFR mismatch: expected {spec.cefr_level}, calibrated {actual_cefr}")

        # Step 3: Extraction Level 1 Code Gate
        extraction_report = audit_unit_extractions(
            spec.unit_name, project_root, enable_llm=enable_llm
        )
        if not extraction_report.passed:
            unit_defects.extend(extraction_report.defects)
        total_extraction_defects += len(extraction_report.defects)

        # Step 4: Handout Level 1 Code Gate (all 5 modalities)
        handouts_audited, handout_defects = audit_handout_code_gates(
            spec.unit_name, source_text, project_root
        )
        total_handouts += handouts_audited
        total_handout_defects += len(handout_defects)
        unit_defects.extend(handout_defects)

        unit_elapsed_ms = (time.perf_counter() - unit_start) * 1000.0
        unit_passed = len(unit_defects) == 0

        results.append(
            {
                "unit": spec.unit_name,
                "cefr_level": spec.cefr_level,
                "genre": spec.genre,
                "calibrated_cefr": actual_cefr,
                "cefr_match": cefr_match,
                "passed": unit_passed,
                "defects": unit_defects,
                "elapsed_ms": round(unit_elapsed_ms, 1),
                "extraction_score": extraction_report.score,
                "handouts_audited": handouts_audited,
            }
        )

    elapsed_total_ms = (time.perf_counter() - start_time) * 1000.0
    passed_units = sum(1 for r in results if r["passed"])
    total_units = len(results)

    return {
        "total_units": total_units,
        "passed_units": passed_units,
        "failed_units": total_units - passed_units,
        "total_handouts": total_handouts,
        "handout_defects": total_handout_defects,
        "extraction_defects": total_extraction_defects,
        "elapsed_ms": round(elapsed_total_ms, 1),
        "overall_status": "passed" if (total_units > 0 and passed_units == total_units) else "failed",
        "units": results,
    }


def format_golden_set_summary(summary: Dict[str, Any]) -> str:
    """Format golden set regression results into an educational audit summary."""
    lines: List[str] = []
    lines.append("\n=======================================================")
    lines.append("     GOLDEN SET END-TO-END REGRESSION BENCHMARK        ")
    lines.append("=======================================================")
    status = "🟢 ALL PASSED" if summary.get("overall_status") == "passed" else "🔴 REGRESSION DETECTED"
    lines.append(f"Status:             {status}")
    lines.append(f"Units Benchmarked:  {summary.get('passed_units', 0)} / {summary.get('total_units', 0)} passed")
    lines.append(f"Handouts Audited:   {summary.get('total_handouts', 0)} (Defects: {summary.get('handout_defects', 0)})")
    lines.append(f"Extraction Defects: {summary.get('extraction_defects', 0)}")
    lines.append(f"Total Execution:    {summary.get('elapsed_ms', 0):.1f} ms (<5ms per item code gate)")
    lines.append("-------------------------------------------------------")

    for u in summary.get("units", []):
        mark = "✓ PASS" if u.get("passed") else "✗ FAIL"
        unit_name = u.get("unit", "")
        cefr = u.get("cefr_level", "")
        genre = u.get("genre", "")
        t_ms = u.get("elapsed_ms", 0.0)
        lines.append(f"[{mark}] {unit_name:<28} [{cefr}] {genre:<32} ({t_ms:>5.1f} ms)")
        defects = u.get("defects", [])
        if defects:
            for d in defects[:4]:
                lines.append(f"    - {d}")
            if len(defects) > 4:
                lines.append(f"    ... and {len(defects) - 4} more defect(s)")

    lines.append("=======================================================\n")
    return "\n".join(lines)
