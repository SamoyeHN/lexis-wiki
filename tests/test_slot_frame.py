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
