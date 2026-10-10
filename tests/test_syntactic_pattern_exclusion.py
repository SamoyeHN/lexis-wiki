"""Tests for Syntactic Pattern Exclusion for Single-Fit Distractors.

Validates that:
1. Target verb governing a specific non-finite/clausal complement (to_inf, gerund, that_clause)
   deterministically rejects near-synonyms sharing the identical pattern (Double-Key Collision).
2. Distractors with incompatible syntactic complement patterns are prioritized as single-fit discriminated.
"""

from librarian.linguistics import LinguisticEngine


def test_get_syntactic_pattern_slots_to_inf():
    slots_refuse = LinguisticEngine.get_syntactic_pattern_slots("refuse")
    assert "to_inf" in slots_refuse
    assert "gerund" not in slots_refuse

    slots_afford = LinguisticEngine.get_syntactic_pattern_slots("afford")
    assert "to_inf" in slots_afford


def test_get_syntactic_pattern_slots_gerund():
    slots_avoid = LinguisticEngine.get_syntactic_pattern_slots("avoid")
    assert "gerund" in slots_avoid
    assert "to_inf" not in slots_avoid

    slots_suggest = LinguisticEngine.get_syntactic_pattern_slots("suggest")
    assert "gerund" in slots_suggest
    assert "to_inf" not in slots_suggest


def test_double_key_pattern_collision():
    # refuse (to_inf) vs decline (to_inf near-synonym) -> Double-Key Collision
    is_dk, reason = LinguisticEngine.double_key_collision(
        "refuse", "decline",
        anchor_type="to_inf",
        pos="verb"
    )
    assert is_dk is True
    assert reason == "double_key_pattern"

    # hope (to_inf) vs wish (to_inf near-synonym) -> Double-Key Collision
    is_dk_hope, reason_hope = LinguisticEngine.double_key_collision(
        "hope", "wish",
        anchor_type="to_inf",
        pos="verb"
    )
    assert is_dk_hope is True
    assert reason_hope == "double_key_pattern"


def test_single_fit_pattern_discrimination():
    # refuse (to_inf) vs suggest (gerund/that_clause) -> Discriminated by pattern
    is_dk, reason = LinguisticEngine.double_key_collision(
        "refuse", "suggest",
        anchor_type="to_inf",
        pos="verb"
    )
    assert is_dk is False
    assert reason == "none"

    # enjoy (gerund) vs decide (to_inf) -> Discriminated by pattern
    is_dk_enjoy, reason_enjoy = LinguisticEngine.double_key_collision(
        "enjoy", "decide",
        anchor_type="gerund",
        pos="verb"
    )
    assert is_dk_enjoy is False
    assert reason_enjoy == "none"


def test_generate_vocab_distractors_with_pattern_exclusion():
    # Target: refuse with to_inf slot
    distractors = LinguisticEngine.generate_vocab_distractors(
        target_word="refuse",
        pos="verb",
        anchor_type="to_inf",
        target_count=3
    )
    assert len(distractors) == 3
    # decline should be excluded by double_key_pattern
    assert "decline" not in distractors
