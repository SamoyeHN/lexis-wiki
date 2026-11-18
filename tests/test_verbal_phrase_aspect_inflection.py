"""
Neuro-symbolic tests for verbal phrase inflection reconciliation and cloze frame slicing.

Guarantees:
1. Verbal phrases in active perfect aspect ('has/have/had shut down') or passive voice
   ('was shut down') are recognized as VBN, inflecting all prescribed options to past participle
   (e.g., 'brought down', 'shut down', 'stood down', 'set down').
2. Cloze slot frames correctly align with the verb inflection tag:
   - When options are VBN: frame preserves the auxiliary (e.g., '... a shop has ____').
   - When options are base form (VB): frame never introduces an auxiliary requiring VBN ('has/have/had/is/was'),
     preventing auxiliary agreement clashes (助动词语法冲突).
3. Evaluator's auxiliary agreement gate passes with zero false rejections.
"""

from __future__ import annotations

import re
import pytest
from librarian.linguistics import LinguisticEngine as LE


def test_verbal_phrase_target_tag_detection():
    cases = [
        ("Our local hardware shop has shut down.", "shut down", "VBN"),
        ("The shop was shut down last year.", "shut down", "VBN"),
        ("The lid was firmly nailed down.", "nail down", "VBN"),
        ("The company shut down its factory.", "shut down", "VBD"),
        ("They nailed down the agreement.", "nail down", "VBD"),
        ("They are shutting down the system.", "shut down", "VBG"),
        ("The server shuts down at midnight.", "shut down", "VBZ"),
        ("We need to shut down the plant.", "shut down", "VB"),
        ("They will shut down tomorrow.", "shut down", "VB"),
        ("He had shut down all computers.", "shut down", "VBN"),
        ("The factory got shut down.", "shut down", "VBN"),
    ]
    for ex, phrase, expected_tag in cases:
        tag = LE._verbal_phrase_target_tag(ex, phrase)
        assert tag == expected_tag, f"Failed for '{phrase}' in '{ex}': got {tag}, expected {expected_tag}"


def test_verbal_phrase_reconcile_inflection_shut_down():
    base_options = ["bring down", "shut down", "stand down", "set down"]
    
    # Active perfect aspect example
    desc, opts, target, tag = LE._reconcile_prescribed_inflection(
        base_options=base_options,
        target_index=1,
        target_word="shut down",
        canonical_pos="phrase",
        inflection_desc="base form",
        authentic_example="Our local hardware shop has shut down.",
    )
    assert tag == "VBN"
    assert opts == ["brought down", "shut down", "stood down", "set down"]
    assert target == "shut down"
    assert "past participle" in desc.lower() or "vbn" in desc.lower()

    # Base form example
    desc_base, opts_base, target_base, tag_base = LE._reconcile_prescribed_inflection(
        base_options=base_options,
        target_index=1,
        target_word="shut down",
        canonical_pos="phrase",
        inflection_desc="base form",
        authentic_example="We need to shut down the system immediately.",
    )
    assert tag_base in (None, "VB")
    assert opts_base == base_options


def test_verbal_phrase_cloze_frame_slicing_concordance():
    example = "Our local hardware shop has shut down."

    # When verb_form_tag is VBN, frame includes 'has' and agrees with VBN options
    frame_vbn = LE.derive_cloze_slot_frame_from_example(
        example_text=example,
        final_target="shut down",
        base_word="shut down",
        canonical_pos="phrase",
        verb_form_tag="VBN",
    )
    assert frame_vbn is not None
    assert "has ____" in frame_vbn

    # When verb_form_tag is base form / None, frame must NEVER force 'has ____'
    frame_base = LE.derive_cloze_slot_frame_from_example(
        example_text=example,
        final_target="shut down",
        base_word="shut down",
        canonical_pos="phrase",
        verb_form_tag=None,
    )
    assert frame_base is not None
    assert not re.search(r"\b(?:has|have|had|is|are|was|were|been)\s+____", frame_base, re.IGNORECASE), (
        f"Base form verbal phrase frame must not contain auxiliary requiring participle: {frame_base}"
    )


def test_evaluator_auxiliary_agreement_gate():
    # Simulate Evaluator L1 check logic
    q_text = "Our local hardware shop has ____."
    options = ["brought down", "shut down", "stood down", "set down"]

    m_left_aux = re.search(r"\b(has|have|had)(?:\s+been)?\s+____", q_text, re.IGNORECASE)
    assert m_left_aux is not None

    vb_clashing = []
    for o in options:
        o_head = str(o).strip().split()[0].lower()
        o_base = LE.lemma_of(o_head) or o_head
        if o_head == o_base and LE._is_base_verb_lemma(o_base):
            vbn_form = (LE._inflect_verb(o_base, "VBN") or "").lower()
            if vbn_form and vbn_form != o_base and o_head != vbn_form:
                vb_clashing.append(o)

    assert vb_clashing == [], f"VBN inflected verbal phrase options must not clash: {vb_clashing}"
