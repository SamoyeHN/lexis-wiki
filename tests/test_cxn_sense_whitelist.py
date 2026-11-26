"""
Regression tests for declarative PedLex Construction Sense Whitelist and physical sense gates.
Ensures polysemous headwords only trigger syntactic construction subcategories (e.g. mandative subjunctive,
gerund-only complements, tough-movement) when locked to their attested senses.
"""

import pytest
from librarian.linguistics import LinguisticEngine as LE


def test_is_cxn_subcat_allowed_for_sense():
    # suggest: Sense 0 is proposal/advice, Sense 1 is epistemic ("evidence suggests")
    assert LE.is_cxn_subcat_allowed_for_sense("suggest", "V_MANDATIVE_SUBJ", 0) is True
    assert LE.is_cxn_subcat_allowed_for_sense("suggest", "V_MANDATIVE_SUBJ", 1) is False
    assert LE.is_cxn_subcat_allowed_for_sense("suggest", "V_GERUND_ONLY", 0) is True
    assert LE.is_cxn_subcat_allowed_for_sense("suggest", "V_GERUND_ONLY", 1) is False

    # insist: Sense 0 is factual assertion, Sense 1 is demand
    assert LE.is_cxn_subcat_allowed_for_sense("insist", "V_MANDATIVE_SUBJ", 0) is False
    assert LE.is_cxn_subcat_allowed_for_sense("insist", "V_MANDATIVE_SUBJ", 1) is True

    # resist: Sense 0 is temptation/urge (can't resist doing), Sense 2 is physical/force (resist arrest)
    assert LE.is_cxn_subcat_allowed_for_sense("resist", "V_GERUND_ONLY", 0) is True
    assert LE.is_cxn_subcat_allowed_for_sense("resist", "V_GERUND_ONLY", 2) is False

    # hard: Sense 0 is physical firmness, Sense 1 is difficulty (tough-movement)
    assert LE.is_cxn_subcat_allowed_for_sense("hard", "ADJ_TOUGH_BOTH", 0) is False
    assert LE.is_cxn_subcat_allowed_for_sense("hard", "ADJ_TOUGH_BOTH", 1) is True


def test_get_word_cxn_specs_with_sense_idx():
    specs_s0 = LE.get_word_cxn_specs("suggest", pos="verb", sense_idx=0)
    subcats_s0 = {s["subcat_group"] for s in specs_s0}
    assert "V_MANDATIVE_SUBJ" in subcats_s0
    assert "V_GERUND_ONLY" in subcats_s0

    specs_s1 = LE.get_word_cxn_specs("suggest", pos="verb", sense_idx=1)
    subcats_s1 = {s["subcat_group"] for s in specs_s1}
    assert "V_MANDATIVE_SUBJ" not in subcats_s1
    assert "V_GERUND_ONLY" not in subcats_s1


def test_build_precomputed_target_skeletons_sense_isolation():
    # suggest Sense 1: should generate an epistemic frame, NOT a subjunctive frame
    items = [{"word": "suggest", "sense_idx": 1, "part_of_speech": "verb"}]
    skeletons = LE.build_precomputed_target_skeletons(items, target_count=1)
    assert len(skeletons) == 1
    skel = skeletons[0]
    frame = skel.get("cloze_slot_frame") or ""
    # Should not be forced into "that sb (should) do sth"
    assert "that sb (should)" not in frame
    assert "committee review" not in frame

    # resist Sense 2: should not force gerund frame
    items_resist = [{"word": "resist", "sense_idx": 2, "part_of_speech": "verb"}]
    skeletons_resist = LE.build_precomputed_target_skeletons(items_resist, target_count=1)
    assert len(skeletons_resist) == 1
    skel_r = skeletons_resist[0]
    frame_r = skel_r.get("cloze_slot_frame") or ""
    assert "V-ing" not in frame_r
    assert "working" not in frame_r
