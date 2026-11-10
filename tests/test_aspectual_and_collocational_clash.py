"""
Tests for Step 1, Step 2 & Step 3: Aspectual Duration and Bound Collocational Modifier Gating,
Near-Synonym / Thesaurus Distractor Admission, and End-to-End Zero Double-Key Regression Suite.
"""

import pytest
from librarian.linguistics import LinguisticEngine as LE


def test_aspectual_duration_clash_quick_glance():
    """Verify that instantaneous aspect slots (e.g. quick/brief) syntactically bar durative gaze/stare,
    exempting them from double-key collisions while correctly identifying peek as a double-key risk."""
    quote = "She cast a quick glance at the clock."

    # 'stare' and 'gaze' specify durative perception ('for a long time') and must clash with 'quick'
    for cand in ("stare", "gaze", "glare"):
        clash, reason = LE.evaluate_syntactic_slot_clash(
            "glance", cand, anchor="quick", quote=quote, pos="noun"
        )
        assert clash is True, f"Expected {cand} to clash with instantaneous aspect 'quick'"
        assert reason == "durative_candidate_in_instantaneous_aspect_slot"

        # Under double_key_collision, the slot clash guarantees single-fit safety ('frame_only')
        is_dk, dk_kind = LE.double_key_collision(
            "glance", cand, anchor="quick", quote=quote, pos="noun"
        )
        assert is_dk is False, f"Expected {cand} to be exempt from double-key collision due to aspectual clash"
        assert dk_kind == "frame_only"

    # 'peek' is also brief/quick, so it DOES NOT have an aspectual duration clash
    clash_peek, _ = LE.evaluate_syntactic_slot_clash(
        "glance", "peek", anchor="quick", quote=quote, pos="noun"
    )
    assert clash_peek is False


def test_durative_aspect_slot_bars_instantaneous_candidates():
    """Verify that durative aspect slots ('steady', 'prolonged') syntactically bar
    instantaneous perception words ('glance', 'glimpse')."""
    quote = "He directed a steady gaze at the portrait."

    for cand in ("glance", "glimpse"):
        clash, reason = LE.evaluate_syntactic_slot_clash(
            "gaze", cand, anchor="steady", quote=quote, pos="noun"
        )
        assert clash is True, f"Expected {cand} to clash with durative aspect 'steady'"
        assert reason == "instantaneous_candidate_in_durative_aspect_slot"

        is_dk, dk_kind = LE.double_key_collision(
            "gaze", cand, anchor="steady", quote=quote, pos="noun"
        )
        assert is_dk is False, f"Expected {cand} to be exempt from double-key collision"
        assert dk_kind == "frame_only"


def test_bound_degree_intensifier_clash_sound_asleep():
    """Verify that bound degree/intensifier collocations ('sound asleep') syntactically bar
    semantically adjacent words ('sleepy', 'sleeping', 'awake') that do not form attested units with 'sound'."""
    quote = "The baby was sound asleep in his crib."

    for cand in ("sleepy", "sleeping", "awake"):
        clash, reason = LE.evaluate_syntactic_slot_clash(
            "asleep", cand, anchor="sound", quote=quote, pos="adjective"
        )
        assert clash is True, f"Expected {cand} to clash with bound intensifier 'sound'"
        assert reason == "unattested_bound_degree_collocation_clash"

        is_dk, dk_kind = LE.double_key_collision(
            "asleep", cand, anchor="sound", quote=quote, pos="adjective"
        )
        assert is_dk is False, f"Expected {cand} to be exempt from double-key collision"
        assert dk_kind == "frame_only"


def test_bound_degree_intensifier_clash_fast_asleep():
    """Verify 'fast asleep' bound collocation bars 'drowsy', 'sleepy', and 'awake'."""
    quote = "He was fast asleep on the sofa."

    for cand in ("drowsy", "sleepy", "awake"):
        clash, reason = LE.evaluate_syntactic_slot_clash(
            "asleep", cand, anchor="fast", quote=quote, pos="adjective"
        )
        assert clash is True
        assert reason == "unattested_bound_degree_collocation_clash"

        is_dk, dk_kind = LE.double_key_collision(
            "asleep", cand, anchor="fast", quote=quote, pos="adjective"
        )
        assert is_dk is False
        assert dk_kind == "frame_only"


def test_bound_degree_intensifier_clash_wide_awake():
    """Verify 'wide awake' bound collocation gates 'asleep' and 'sleepy'."""
    quote = "She was wide awake despite the late hour."

    for cand in ("asleep", "sleepy"):
        clash, reason = LE.evaluate_syntactic_slot_clash(
            "awake", cand, anchor="wide", quote=quote, pos="adjective"
        )
        assert clash is True
        assert reason == "unattested_bound_degree_collocation_clash"

        is_dk, dk_kind = LE.double_key_collision(
            "awake", cand, anchor="wide", quote=quote, pos="adjective"
        )
        assert is_dk is False
        assert dk_kind == "frame_only"


def test_bound_degree_intensifier_clash_bitterly_disappointed():
    """Verify 'bitterly disappointed' bound collocation gates 'pleased', 'happy', 'glad'."""
    quote = "They were bitterly disappointed with the verdict."

    for cand in ("pleased", "happy", "glad"):
        clash, reason = LE.evaluate_syntactic_slot_clash(
            "disappointed", cand, anchor="bitterly", quote=quote, pos="adjective"
        )
        assert clash is True
        assert reason == "unattested_bound_degree_collocation_clash"

        is_dk, dk_kind = LE.double_key_collision(
            "disappointed", cand, anchor="bitterly", quote=quote, pos="adjective"
        )
        assert is_dk is False
        assert dk_kind == "frame_only"


def test_end_to_end_glance_near_synonym_thesaurus_distractors():
    """Step 2 End-to-End: Verify that glance in a quick glance context automatically draws
    high-discrimination visual perception near-synonyms (gaze, glare/stare) from LDOCE Thesaurus."""
    distractors, meta = LE.generate_vocab_distractors(
        "glance",
        pos="noun",
        context_anchor="quick",
        quote="She cast a quick glance at the clock.",
        return_metadata=True
    )
    assert len(distractors) == 3
    # At least one (or two) distractors must be from ldoce_thesaurus (gaze / glare / stare)
    thes_dists = [d for d in distractors if meta.get(d) == "ldoce_thesaurus"]
    assert len(thes_dists) >= 1
    assert any(d in ("gaze", "stare", "glare") for d in thes_dists)
    # Peek must NOT be present as it would double-key under 'quick'
    assert "peek" not in distractors


def test_end_to_end_gaze_draws_instantaneous_near_synonym():
    """Step 2 End-to-End: Verify that gaze in a steady gaze context draws glance from LDOCE Thesaurus."""
    distractors, meta = LE.generate_vocab_distractors(
        "gaze",
        pos="noun",
        context_anchor="steady",
        quote="He directed a steady gaze at the portrait.",
        return_metadata=True
    )
    assert len(distractors) == 3
    assert "glance" in distractors
    assert meta.get("glance") == "ldoce_thesaurus"
    # Verify zero double-key collision
    is_dk, _ = LE.double_key_collision("gaze", "glance", anchor="steady", quote="He directed a steady gaze at the portrait.", pos="noun")
    assert is_dk is False


def test_end_to_end_asleep_near_synonym_and_antonym_distractors():
    """Step 2 End-to-End: Verify that asleep in sound asleep context draws valid sleep-domain
    candidates (awake, sleepy) with zero double-key collisions."""
    distractors, meta = LE.generate_vocab_distractors(
        "asleep",
        pos="adjective",
        definition="sleeping",
        context_anchor="sound",
        quote="The baby was sound asleep in his crib.",
        return_metadata=True
    )
    assert len(distractors) == 3
    assert "awake" in distractors
    assert "sleepy" in distractors or "sleepless" in distractors
    # Verify zero double-key collisions against the quote
    for d in distractors:
        is_dk, _ = LE.double_key_collision(
            "asleep", d, anchor="sound", quote="The baby was sound asleep in his crib.", pos="adjective"
        )
        assert is_dk is False, f"Distractor {d} collided as double key!"


def test_step3_comprehensive_multi_target_zero_double_key_verification():
    """Step 3 Comprehensive Verification: Verify that across multiple syllabus vocabulary items
    spanning different parts of speech, distractors always satisfy:
    1. Exactly 3 distinct distractors.
    2. Zero target collision (target not in distractors).
    3. 100% Zero Double-Key Collision against the contextual anchor."""
    corpus_cases = [
        ("glance", "noun", "quick", "She cast a quick glance at the clock."),
        ("asleep", "adjective", "sound", "The baby was sound asleep in his crib."),
        ("awake", "adjective", "wide", "She was wide awake despite the late hour."),
        ("disappointed", "adjective", "bitterly", "They were bitterly disappointed with the verdict."),
        ("gaze", "noun", "steady", "He directed a steady gaze at the portrait."),
        ("hesitate", "verb", "to", "Do not hesitate to contact us."),
        ("reluctant", "adjective", "to", "He was reluctant to admit his mistake."),
        ("anxious", "adjective", "about", "Parents are anxious about their children safety."),
        ("wander", "verb", "about", "We wandered about the streets of the old town."),
        ("scarcely", "adverb", "when", "Scarcely had she entered when the phone rang.")
    ]

    for word, pos, anchor, quote in corpus_cases:
        dists, meta = LE.generate_vocab_distractors(
            word, pos=pos, context_anchor=anchor, quote=quote, return_metadata=True
        )
        assert len(dists) == 3, f"Target {word} generated {len(dists)} distractors instead of 3"
        assert word not in dists, f"Target {word} leaked into distractors"
        assert len(set(dists)) == 3, f"Duplicate distractors generated for {word}: {dists}"

        for d in dists:
            is_dk, reason = LE.double_key_collision(word, d, anchor=anchor, quote=quote, pos=pos)
            assert not is_dk, f"Double-key collision for target '{word}' vs distractor '{d}': {reason}"
