"""Unit tests for the SlotFrame domain dataclass and surface frame rendering."""

import pytest
from librarian.linguistics import SlotFrame
from librarian.processor_core.quiz_hydrators import QuizHydratorMixin


def test_slot_frame_initialization():
    frame = SlotFrame(
        left_anchor="their",
        right_anchor="achievement",
        prep="",
        ontology_hint="",
        constraints=[],
    )
    assert frame.left_anchor == "their"
    assert frame.right_anchor == "achievement"
    assert frame.to_symbolic_str() == "... their ____ achievement"


def test_slot_frame_equality_and_str_compatibility():
    raw_str = "... their ____ achievement"
    frame = SlotFrame.from_symbolic_str(raw_str)

    # String conversion compatibility
    assert str(frame) == raw_str

    # Equality against str
    assert frame == raw_str
    assert raw_str == frame
    assert frame != "... some other ____"

    # Equality against another SlotFrame
    frame2 = SlotFrame.from_symbolic_str(raw_str)
    assert frame == frame2

    # Set and hashability
    s = {frame, frame2}
    assert len(s) == 1
    assert frame in s


def test_slot_frame_from_symbolic_str_parsing():
    # 1. Trailing preposition
    f1 = SlotFrame.from_symbolic_str("... ____ into the market")
    assert f1.prep == "into"
    assert f1.right_anchor == "the market"

    # 2. Auxiliary lock
    f2 = SlotFrame.from_symbolic_str("... [is] ____ to")
    assert f2.aux_lock == "is"
    assert f2.prep == "to"

    # 3. Ontology hint
    f3 = SlotFrame.from_symbolic_str("... ____ [process/activity, e.g. production / distribution]")
    assert "[process/activity, e.g. production / distribution]" in f3.ontology_hint

    # 4. Constraints
    f4 = SlotFrame.from_symbolic_str("... ____ to do [sth]")
    assert "to-infinitive" in f4.constraints

    f5 = SlotFrame.from_symbolic_str("... that [sb] (should) [V-base]")
    assert "mandative-subjunctive" in f5.constraints


def test_slot_frame_to_llm_surface_ontology():
    frame = SlotFrame.from_symbolic_str("... ____ [concrete object, e.g. hammer / saw]")
    surface, notes = frame.to_llm_surface()
    assert "hammer" in surface
    assert any("concrete object" in n for n in notes)


def test_slot_frame_to_llm_surface_infinitive():
    frame = SlotFrame.from_symbolic_str("... ____ to do [sth]")
    # With example sentence containing an infinitive
    surface, notes = frame.to_llm_surface(example_txt="She wanted to explore new opportunities.")
    assert "to explore" in surface

    # Without example sentence
    surface2, notes2 = frame.to_llm_surface(example_txt="")
    assert "to" in surface2
    assert any("Follow 'to' with a specific action verb" in n for n in notes2)

    # When example sentence's infinitive verb IS the target word itself, it must NOT leak into the frame
    skel_offer = {
        "target_word": "offer",
        "base_headword": "offer",
        "prescribed_options": ["consider", "practice", "offer", "enjoy"],
    }
    surface3, notes3 = frame.to_llm_surface(
        example_txt="I don't need any help, but it was nice of you to offer.",
        skeleton=skel_offer,
    )
    assert surface3 == "... ____ to"
    assert "offer" not in surface3
    assert any("Follow 'to' with a specific action verb" in n for n in notes3)


def test_slot_frame_to_llm_surface_mandative_subjunctive():
    frame = SlotFrame.from_symbolic_str("... that [sb] (should) [V-base]")
    surface, notes = frame.to_llm_surface()
    assert "that the committee review" in surface
    assert any("subjunctive" in n.lower() for n in notes)


def test_slot_frame_to_llm_surface_passive_vbn():
    frame = SlotFrame.from_symbolic_str("... ____ by the team")
    skeleton = {
        "verb_form_tag": "VBN",
        "target_word": "discovered",
        "example_sentence": "The anomaly was discovered by researchers last year.",
    }
    surface, notes = frame.to_llm_surface(
        example_txt=skeleton["example_sentence"],
        skeleton=skeleton,
    )
    assert "was ____" in surface
    assert any("passive auxiliary" in n.lower() for n in notes)


def test_quiz_hydrator_render_llm_surface_frame_delegation():
    # Verify QuizHydratorMixin._render_llm_surface_frame works with both str and SlotFrame
    raw = "... their ____ achievement"
    skel = {"example_sentence": "This was their greatest achievement."}

    surf_from_str, notes_str = QuizHydratorMixin._render_llm_surface_frame(raw, skel)
    frame_obj = SlotFrame.from_symbolic_str(raw)
    surf_from_obj, notes_obj = QuizHydratorMixin._render_llm_surface_frame(frame_obj, skel)

    assert surf_from_str == surf_from_obj
    assert notes_str == notes_obj


def test_slot_frame_to_llm_surface_body_part_noun():
    frame = SlotFrame.from_symbolic_str("... ____ [body part, e.g. shoulders]")
    surface, notes = frame.to_llm_surface()
    assert surface == "... ____ shoulders"
    assert "the shoulders" not in surface
    assert any("subject-matching possessive" in n and "shoulders" in n for n in notes)


def test_slot_frame_to_llm_surface_open_person_object():
    # 1. With trailing preposition and generic someone
    f1 = SlotFrame.from_symbolic_str("... ____ [person/group, e.g. someone] as")
    surf1, notes1 = f1.to_llm_surface()
    assert surf1 == "... ____ ... as"
    assert "someone" not in surf1
    assert any("concrete person or group noun phrase" in n for n in notes1)

    # 2. Bare [person/group] without exemplars
    f2 = SlotFrame.from_symbolic_str("... ____ [person/group] into")
    surf2, notes2 = f2.to_llm_surface()
    assert surf2 == "... ____ ... into"
    assert "[person/group]" not in surf2
    assert any("concrete person or group noun phrase" in n for n in notes2)

    # 3. Trailing [person/group] at clause end
    f3 = SlotFrame.from_symbolic_str("... ____ [person/group]")
    surf3, notes3 = f3.to_llm_surface()
    assert surf3 == "... ____ ..."
    assert "someone" not in surf3
    assert any("concrete person or group noun phrase" in n for n in notes3)


def test_slot_frame_to_llm_surface_aux_agreement():
    f_have = SlotFrame.from_symbolic_str("... have ____ over time")
    surf, notes = f_have.to_llm_surface()
    assert surf == "... have ____ over time"
    assert any("plural subject" in n for n in notes)


def test_wordnet_abstract_slot_classification():
    from librarian.linguistics import LinguisticEngine
    assert LinguisticEngine._classify_abstract_slot_category(["confidence"]) == ("feeling/state", "feeling_state")
    assert LinguisticEngine._classify_abstract_slot_category(["problem"]) == ("problem/issue", "problem_issue")
    assert LinguisticEngine._classify_abstract_slot_category(["funds"]) == ("funds/resources", "funds_resources")
    assert LinguisticEngine._classify_abstract_slot_category(["negotiation"]) == ("process/activity", "process_activity")


def test_patient_ontology_no_forced_the_on_poss_uncountable_or_plural():
    from librarian.linguistics import LinguisticEngine
    slot_conf = LinguisticEngine._extract_verb_patient_ontology(
        word="corrode",
        canonical_pos="verb",
        target_entry=None,
        locked_sense=None,
        definition="to gradually make something weaker",
        authentic_example="Corruption has corroded our confidence in the police force.",
    )
    assert slot_conf is not None
    assert slot_conf["exemplars"][0] == "confidence"

    slot_plural = LinguisticEngine._extract_verb_patient_ontology(
        word="ignore",
        canonical_pos="verb",
        target_entry=None,
        locked_sense=None,
        definition="to pay no attention to something",
        authentic_example="They ignored his warnings.",
    )
    assert slot_plural is not None
    assert slot_plural["exemplars"][0] == "warnings"

    slot_sg = LinguisticEngine._extract_verb_patient_ontology(
        word="express",
        canonical_pos="verb",
        target_entry=None,
        locked_sense=None,
        definition="to state what you think",
        authentic_example="She expressed her opinion.",
    )
    assert slot_sg is not None
    assert slot_sg["exemplars"][0] == "the opinion"


def test_slot_frame_to_llm_surface_possessive_placeholder_and_exert_collision():
    from librarian.linguistics import LinguisticEngine
    f_poss = SlotFrame.from_symbolic_str("... ____ someone's displeasure")
    surf, notes = f_poss.to_llm_surface()
    assert surf == "... ____ displeasure"
    assert "someone's" not in surf
    assert any("never write literal 'someone's'" in n for n in notes)

    cands, _ = LinguisticEngine.get_construction_contrast_distractors(
        "exert", pos="verb", context_anchor="pressure", anchor_type="lv_triad"
    )
    assert "exercise" not in cands
    assert "impose" not in cands
    is_dk, _ = LinguisticEngine.double_key_collision("exert", "exercise", pos="verb")
    assert is_dk is True


def test_adj_quote_unattested_noun_falls_back_to_five_source():
    from pathlib import Path
    from librarian.linguistics import LinguisticEngine
    p = Path("wiki/Book_4_Unit_6_Section_A/extractions/Book_4_Unit_6_Section_A_vocabulary.md")
    if not p.exists():
        pytest.skip("Book_4_Unit_6_Section_A_vocabulary.md not present")
    skels = LinguisticEngine.build_precomputed_target_skeletons(p.read_text(), target_count=20, passage_level="B2")
    c_skel = next((s for s in skels if s.get("target_word", "").lower() == "controversial"), None)
    assert c_skel is not None
    assert "uranium" not in str(c_skel.get("cloze_slot_frame") or "").lower()

    m_skel = next((s for s in skels if s.get("target_word", "").lower() == "monopoly"), None)
    if m_skel is not None:
        frame_str = str(m_skel.get("cloze_slot_frame") or "").lower()
        assert "each ____" not in frame_str
        assert "have a ____ on" in frame_str
        m_opts = {o.lower() for o in m_skel.get("prescribed_options") or []}
        assert "ascendant" not in m_opts
        assert "monotony" in m_opts
        assert "monotone" in m_opts
        # All options must start with consonant to prevent 'a' vs 'an' leakage
        for opt in m_opts:
            assert opt[0] not in "aeiou", f"Vowel initial option '{opt}' causes article leakage with 'have a ____ on'"

    l_skel = next((s for s in skels if s.get("target_word", "").lower() == "lump"), None)
    assert l_skel is not None
    assert "____ of" in str(l_skel.get("cloze_slot_frame") or "").lower()
    l_opts = {o.lower() for o in l_skel.get("prescribed_options") or []}
    assert "banana" not in l_opts
    assert "massif" not in l_opts
