"""
Zero-token offline Blueprint Pre-Flight Auditor across all wiki/ vocabulary extractions.

Usage:
    python3 tests/audit_blueprints.py
    python3 tests/audit_blueprints.py --limit 10
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from librarian.linguistics import LinguisticEngine as LE

_BARE_COPULA_END_RE = re.compile(r"\b(?:is|was|are|were|be|has|have|had)\s*$", re.IGNORECASE)
_UNRESOLVED_META_RE = re.compile(r"\[(?:sb|sth|sb/sth|sth/sb|doing sth)\]|/", re.IGNORECASE)
_STOP_FRAME_TOKENS = {
    "...", "____", "the", "a", "an", "to", "in", "on", "at", "of", "for", "with",
    "by", "from", "as", "is", "was", "are", "were", "be", "been", "being",
    "something", "someone", "somebody", "that", "this", "it", "they",
}


def audit_single_blueprint(skel: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Returns a list of (severity, code_and_detail) findings for a single blueprint item."""
    findings: List[Tuple[str, str]] = []
    target = str(skel.get("target_word") or "")
    base_hw = str(skel.get("base_headword") or target).strip().lower()
    pos = str(skel.get("part_of_speech") or "").strip().lower()
    frame = skel.get("cloze_slot_frame")
    micro = str(skel.get("micro_task") or "")
    mechs = skel.get("distractor_mechanisms") or {}
    opts = skel.get("prescribed_options") or []
    c_idx = skel.get("correct_answer_index")
    v_tag = skel.get("verb_form_tag")
    downgrade = skel.get("anchor_downgrade")
    defn = str(skel.get("definition") or "").strip().lower()

    # 1. Option cardinality & uniqueness
    if len(opts) != 4 or len({str(o).strip().lower() for o in opts}) != 4:
        findings.append(("ERROR", f"OPTION_CARDINALITY: options={opts}"))
    if not isinstance(c_idx, int) or not (0 <= c_idx < len(opts)):
        findings.append(("ERROR", f"INVALID_ANSWER_IDX: {c_idx}"))

    # 2. Dangling bare copula/aux in frame
    if frame and _BARE_COPULA_END_RE.search(frame.strip()):
        findings.append(("ERROR", f"DANGLING_COPULA_FRAME: frame={frame!r}"))

    # 3. Unresolved metasyntax or slash alternatives
    if frame and _UNRESOLVED_META_RE.search(frame):
        # Allow intentional semantic ontology hints like [process/activity, e.g. ...] and number-neutral is/are, was/were
        cleaned_allowed = re.sub(
            r"\[(?!\s*(?:sb|sth|sb/sth|sth/sb|doing sth)\s*\])[^\]]+\]",
            "",
            frame,
            flags=re.IGNORECASE,
        )
        cleaned_allowed = re.sub(r"\b(?:is/are|was/were)\b", "", cleaned_allowed, flags=re.IGNORECASE)
        if _UNRESOLVED_META_RE.search(cleaned_allowed):
            findings.append(("WARN", f"UNRESOLVED_META_OR_SLASH: frame={frame!r}"))

    # 4. Weak / low-information frame (no content word outside stopwords)
    if frame:
        toks = [t.lower() for t in re.findall(r"[a-zA-Z]+", frame)]
        content_toks = [t for t in toks if t not in _STOP_FRAME_TOKENS]
        if not content_toks and pos != "function_word" and not re.match(r"^The ____ (?:is|was) that \.\.\.$", frame.strip()):
            findings.append(("WARN", f"WEAK_LOW_INFO_FRAME: frame={frame!r}"))

    # 5. Anchor downgrade to sense_recognition
    if downgrade == "sense_recognition":
        findings.append(("WARN", "MISSING_FRAME_SENSE_RECOGNITION: cloze_slot_frame=None"))

    # 6. Fallback distractor density
    fb_count = sum(1 for m in mechs.values() if str(m or "").endswith("fallback"))
    if mechs and fb_count == len(mechs):
        findings.append(("ERROR", f"ALL_FALLBACK_DISTRACTORS: {dict(mechs)}"))
    elif fb_count >= 2:
        findings.append(("WARN", f"MAJORITY_FALLBACK_DISTRACTORS ({fb_count}/{len(mechs)}): {dict(mechs)}"))

    # 7. Grammar Box inflection clash vs slot requirement
    if v_tag == "VB" and "Grammar Box:" in micro and "Base-form verb slot" in micro:
        gb_part = micro.split("Grammar Box:", 1)[1]
        # Remove the target base form and check if inflected forms of base_hw appear in Grammar Box example
        for tag in ("VBZ", "VBD", "VBG", "VBN"):
            infl = (LE._inflect_verb(base_hw, tag) or "").lower()
            if infl and infl != base_hw and re.search(rf"\b{re.escape(infl)}\b", gb_part.lower()):
                findings.append(("WARN", f"GRAMMAR_BOX_INFLECTION_CLASH: slot=VB ('{base_hw}'), Grammar Box uses '{infl}'"))
                break

    # 8. Non-primary polysemous sense without disambiguating cue
    if pos in ("noun", "verb", "adj", "adjective") and " " not in base_hw and defn and "Sense Guard:" not in micro:
        ent = LE.get_pedlex_entry(base_hw)
        senses = (ent or {}).get("senses") or []
        if len(senses) > 1:
            s0_def = str(senses[0].get("definition") or "").strip().lower()
            if s0_def and defn != s0_def and s0_def not in defn and defn not in s0_def:
                # Check if locked sense is sense #2+ and frame lacks strong content cue
                frame_content = [t for t in re.findall(r"[a-zA-Z]+", frame or "") if t.lower() not in _STOP_FRAME_TOKENS]
                if len(frame_content) <= 1:
                    findings.append(("INFO", f"NON_PRIMARY_SENSE_LOW_CUE: locked!='{s0_def[:40]}...', frame={frame!r}"))

    # 9. Countability contrast frame mismatch
    if "Countability contrast active" in micro and frame:
        if re.search(r"\b(?:the|a|an)\s+____\b", frame, re.IGNORECASE) and not re.search(r"\b(?:several|many|two|three|few|fewer|these|those|multiple)\s+____", frame, re.IGNORECASE):
            findings.append(("WARN", f"COUNTABILITY_FRAME_NEEDS_PLURAL_CUE: frame={frame!r}"))

    return findings


def run_corpus_blueprint_audit(limit: int = 0) -> None:
    vocab_files = sorted((ROOT / "wiki").glob("*/extractions/*_vocabulary.md"))
    if limit > 0:
        vocab_files = vocab_files[:limit]

    total_items = 0
    severity_counts: Counter = Counter()
    code_counts: Counter = Counter()
    examples_by_code: Dict[str, List[str]] = defaultdict(list)

    for vf in vocab_files:
        unit_name = vf.parent.parent.name
        md_text = vf.read_text(encoding="utf-8")
        try:
            skels = LE.build_precomputed_target_skeletons(md_text, target_count=10)
        except Exception as e:
            print(f"[CRASH] {unit_name}: {e}")
            continue

        for skel in skels:
            total_items += 1
            tw = skel.get("target_word")
            findings = audit_single_blueprint(skel)
            for sev, msg in findings:
                code = msg.split(":", 1)[0].split(" ", 1)[0]
                severity_counts[sev] += 1
                code_counts[f"[{sev}] {code}"] += 1
                if len(examples_by_code[code]) < 8:
                    examples_by_code[code].append(f"{unit_name} :: '{tw}' -> {msg}")

    print("=" * 72)
    print(f"BLUEPRINT PRE-FLIGHT AUDIT REPORT ({len(vocab_files)} units, {total_items} items)")
    print("=" * 72)
    print("Severity Summary:", dict(severity_counts))
    print("\nDefect Breakdown:")
    for k, cnt in code_counts.most_common():
        print(f"  {cnt:3d}x  {k}")

    print("\nRepresentative Samples (up to 8 per category):")
    for k, _ in code_counts.most_common():
        code = k.split("] ", 1)[1]
        print(f"\n--- {k} ---")
        for ex in examples_by_code[code]:
            print(f"  * {ex}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Limit number of units to scan")
    args = parser.parse_args()
    run_corpus_blueprint_audit(limit=args.limit)
