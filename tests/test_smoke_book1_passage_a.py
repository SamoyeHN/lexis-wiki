"""
End-to-End Smoke Test Suite for wiki/Book_1_Unit_x_Passage_A (Units 1-6).
Verifies deterministic extractions, all 4 SLA assessment modality blueprints,
and Level 1 Deterministic Code Gate invariants across existing handouts.
"""

import json
import re
from pathlib import Path
import pytest

from librarian.linguistics import LinguisticEngine as LE
from librarian.processor import WikiProcessor as WP


@pytest.mark.parametrize("unit_num", range(1, 7))
def test_smoke_book1_passage_a_unit(unit_num: int):
    unit_name = f"Book_1_Unit_{unit_num}_Passage_A"
    u_dir = Path("wiki") / unit_name
    src_file = u_dir / "sources" / f"{unit_name}.md"
    assert src_file.exists(), f"Source file does not exist: {src_file}"
    text = src_file.read_text(encoding="utf-8")
    assert len(text) > 200, f"Source text is unexpectedly short in {unit_name}"

    # 1. Deterministic Extractions
    vocab = LE.extract_deterministic_vocabulary(text, target_count=10)
    assert len(vocab) >= 8, f"Extracted vocabulary count {len(vocab)} < 8 in {unit_name}"

    exprs = LE.mine_expression_skeletons(text, target_count=5)
    assert len(exprs) >= 3, f"Extracted expressions count {len(exprs)} < 3 in {unit_name}"

    grams = LE.extract_deterministic_grammar(text, target_count=5)
    assert len(grams) >= 3, f"Extracted grammar count {len(grams)} < 3 in {unit_name}"

    # 2. Modality 1: Vocab Skeletons (Proficiency MCQ) & Authentic Cloze
    skeletons = LE.build_precomputed_target_skeletons(vocab, target_count=10)
    assert len(skeletons) >= 8, f"Precomputed target skeletons count {len(skeletons)} < 8 in {unit_name}"
    for idx, sk in enumerate(skeletons):
        opts = sk.get("prescribed_options") or []
        assert len(opts) == 4, f"[{unit_name}] Skeleton {idx} options != 4: {opts}"
        assert len(set(opts)) == 4, f"[{unit_name}] Skeleton {idx} duplicate options: {opts}"
        c_idx = sk.get("correct_answer_index", 0)
        assert 0 <= c_idx < 4, f"[{unit_name}] Skeleton {idx} invalid correct index {c_idx}"

    cloze_items = LE.build_authentic_cloze_items(vocab, target_count=10)
    assert len(cloze_items) >= 8, f"Authentic cloze count {len(cloze_items)} < 8 in {unit_name}"
    for idx, cl in enumerate(cloze_items):
        opts = cl.get("options") or []
        assert len(opts) == 4, f"[{unit_name}] Cloze {idx} options != 4: {opts}"
        assert len(set(opts)) == 4, f"[{unit_name}] Cloze {idx} duplicate options: {opts}"
        q_stem = cl.get("question") or ""
        assert len(re.findall(r"_{2,}", q_stem)) == 1, f"[{unit_name}] Cloze {idx} blank count != 1: {q_stem}"

    # 3. Modality 2: Reading Discourse Blueprints
    reading_bps = LE.build_discourse_blueprints(text, count=5, cefr_level="A2")
    assert len(reading_bps) >= 3, f"[{unit_name}] Reading blueprints count {len(reading_bps)} < 3"
    for rb in reading_bps:
        assert rb.question_stem_prompt, f"[{unit_name}] Missing reading question stem"
        assert len(rb.distractors) == 3, f"[{unit_name}] Reading distractors count != 3"

    # 4. Modality 3 & 4: Translation Minimal Pairs & Listening Blueprints
    trans_bps = LE.build_translation_micro_task_blueprints(
        vocab_content=vocab[:5],
        grammar_content=grams[:3],
        target_count=5,
        target_language="Chinese",
        cefr_level="A2",
    )
    assert len(trans_bps) >= 3, f"[{unit_name}] Translation blueprints count {len(trans_bps)} < 3"

    listening_bp = LE.build_listening_micro_task_blueprints(
        vocab_content=vocab,
        thematic_topic=unit_name,
        target_count=5,
        cefr_level="A2",
    )
    assert listening_bp and "questions" in listening_bp, f"[{unit_name}] Listening blueprint missing questions"
    assert len(listening_bp["questions"]) >= 3, f"[{unit_name}] Listening questions count < 3"

    # 5. Handouts Code Gate Audits
    handouts = sorted(list((u_dir / "handouts").glob("*.html")))
    assert len(handouts) >= 2, f"[{unit_name}] Expected at least 2 handouts, found {len(handouts)}"
    for h in handouts:
        h_text = h.read_text(encoding="utf-8")
        m = re.search(r"const\s+quizData\s*=\s*(\{.*?\});\s*(?:var|let|const|\n|</script>)", h_text, re.DOTALL)
        if not m:
            continue
        q_data = json.loads(m.group(1))
        h_name = h.name.lower()
        if "translation" in h_name:
            flagged, defects = WP.audit_translation_integrity(q_data, target_language="Chinese", passage_text=text)
        elif "reading" in h_name:
            flagged, defects = WP.audit_reading_integrity(q_data, text)
        elif "listening" in h_name:
            flagged, defects = WP.audit_listening_integrity(q_data, cefr_level="A2")
        elif "video" in h_name:
            v_md_name = h.stem.replace("_video_quiz", "") + ".md"
            v_md_file = u_dir / "sources" / "media" / v_md_name
            v_text = v_md_file.read_text(encoding="utf-8") if v_md_file.exists() else text
            flagged, defects = WP.audit_video_integrity(q_data, transcript_text=v_text)
        else:
            flagged, defects = WP.audit_quiz_integrity(q_data, allow_authentic_cloze=True)

        fatal = [d for d in defects if "exceeding CEFR" not in d and "minor" not in d.lower()]
        assert len(fatal) == 0, f"[{unit_name}] Handout {h.name} has fatal defects: {fatal}"
