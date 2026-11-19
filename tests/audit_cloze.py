#!/usr/bin/env python3
"""
Zero-Token Full-Corpus Auditor for Pure Open Cloze (`quiz_cloze.py`).

Scans all units under `wiki/*/sources/*.md`, generates a Pure Open Cloze
assessment for each unit, and audits:
  1. Target word uniqueness per unit (zero duplicates)
  2. Exact sentence-offset masking (reconstructing original sentence verbatim)
  3. Pillar balance across P1 (Discourse Cohesion), P2 (MWU/Preposition), P3 (Lexico-Grammar)
  4. Level-1 code gate (`audit_cloze_integrity`) zero-defect compliance
  5. Blank count and accepted_answers distribution
"""
import glob
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from librarian.linguistics import LinguisticEngine
from librarian.processor_core.quiz_integrity_gates import QuizIntegrityGatesMixin


def _strip_frontmatter(text: str) -> str:
    return re.sub(r"^---.*?---\s*", "", text, flags=re.S).strip()


def run_audit(target_count: int = 10) -> int:
    source_files = sorted(glob.glob("wiki/*/sources/*.md"))
    total_units = 0
    total_blanks = 0
    dup_units = 0
    dup_blanks = 0
    offset_mismatches = 0
    l1_defect_units = 0
    under_target_units = 0
    multi_accept_blanks = 0

    pillar_counts = Counter()
    category_counts = Counter()
    word_counts = Counter()
    all_defects = []

    for path in source_files:
        unit_name = os.path.basename(os.path.dirname(os.path.dirname(path)))
        with open(path, "r", encoding="utf-8") as fh:
            passage = _strip_frontmatter(fh.read())
        if len(passage.split()) < 80:
            continue

        total_units += 1
        quiz = LinguisticEngine.build_discourse_cloze_assessment(passage, target_count=target_count)
        blanks = quiz.get("blanks", [])
        total_blanks += len(blanks)
        if len(blanks) < target_count:
            under_target_units += 1

        # 1. Duplicate target check
        targets = [b["target_word"].lower() for b in blanks]
        unit_dups = len(targets) - len(set(targets))
        if unit_dups > 0:
            dup_units += 1
            dup_blanks += unit_dups

        # 2. Exact offset reconstruction check
        span_text = quiz.get("passage", "")
        for b in blanks:
            pillar_counts[b["pillar"]] += 1
            category_counts[b["category"]] += 1
            word_counts[b["target_word"].lower()] += 1
            if len(b.get("accepted_answers", [])) > 1:
                multi_accept_blanks += 1

            marker = f"[{b['blank_index']}] ____"
            restored = b["question"].replace(marker, b["raw_word"], 1)
            if restored not in span_text:
                offset_mismatches += 1
                all_defects.append(f"[{unit_name}] Blank #{b['blank_index']} offset mismatch: {b['question']!r}")

        # 3. L1 code gate
        flagged, defects = QuizIntegrityGatesMixin.audit_cloze_integrity(quiz, passage_text=span_text)
        if flagged or defects:
            l1_defect_units += 1
            for d in defects:
                all_defects.append(f"[{unit_name}] {d}")

    print(f"=== Pure Open Cloze Full-Corpus Audit ({total_units} units, {total_blanks} blanks) ===")
    print(f"Units < {target_count} blanks      : {under_target_units}/{total_units}")
    print(f"Units with duplicate targets : {dup_units}/{total_units} ({dup_blanks} duplicate blanks)")
    print(f"Exact offset mismatches      : {offset_mismatches}")
    print(f"L1 code gate defect units    : {l1_defect_units}/{total_units}")
    print(f"Blanks with >1 accepted ans  : {multi_accept_blanks}/{total_blanks}")
    if total_blanks > 0:
        p1 = pillar_counts[1]
        p2 = pillar_counts[2]
        p3 = pillar_counts[3]
        print(
            f"Pillar distribution          : "
            f"P1={p1} ({p1/total_blanks*100:.1f}%), "
            f"P2={p2} ({p2/total_blanks*100:.1f}%), "
            f"P3={p3} ({p3/total_blanks*100:.1f}%)"
        )
    print(f"Categories                   : {category_counts.most_common()}")
    print(f"Top 15 blanked words         : {word_counts.most_common(15)}")

    if all_defects:
        print("\nDefects detected:")
        for d in all_defects[:20]:
            print(f"  - {d}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(run_audit())
