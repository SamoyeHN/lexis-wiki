import pytest
from librarian.linguistics import LinguisticEngine


class TestNegativeInversionArena:
    """Psychometric SLA validation suite for Family 7 Fronted Negative Inversion Trigger Distractor Arena."""

    def test_fronted_inversion_context_detection(self):
        """Verify fronted inversion context detector correctly distinguishes inverted clauses from standard SVO."""
        # Positive inverted clauses
        assert LinguisticEngine.is_fronted_inversion_context(
            quote="Hardly had the meeting begun when the lights went out.", target="hardly"
        )
        assert LinguisticEngine.is_fronted_inversion_context(
            quote="____ had the meeting begun when the lights went out.", target="hardly"
        )
        assert LinguisticEngine.is_fronted_inversion_context(
            quote="Seldom have I seen such devotion to duty.", target="seldom"
        )
        assert LinguisticEngine.is_fronted_inversion_context(
            quote="\"Seldom have I seen such devotion to duty.\"", target="seldom"
        )
        assert LinguisticEngine.is_fronted_inversion_context(
            quote="No sooner had the match started than it began to rain.", target="no sooner"
        )
        assert LinguisticEngine.is_fronted_inversion_context(
            quote="Little did he realize what was about to happen.", target="little"
        )
        assert LinguisticEngine.is_fronted_inversion_context(
            quote="Rarely does one encounter such honesty.", target="rarely"
        )
        assert LinguisticEngine.is_fronted_inversion_context(
            quote="Under no circumstances should passengers leave their luggage unattended.",
            target="under no circumstances",
        )
        assert LinguisticEngine.is_fronted_inversion_context(anchor_type="fronted_inv")

        # Negative non-inverted clauses
        assert not LinguisticEngine.is_fronted_inversion_context(
            quote="I could hardly move in the crowded room.", target="hardly"
        )
        assert not LinguisticEngine.is_fronted_inversion_context(
            quote="We seldom go to the cinema these days.", target="seldom"
        )
        assert not LinguisticEngine.is_fronted_inversion_context(
            quote="There is little doubt that the plan will succeed.", target="little"
        )
        assert not LinguisticEngine.is_fronted_inversion_context(
            quote="She rarely speaks in public.", target="rarely"
        )

    def test_evaluate_syntactic_slot_clash_inversion(self):
        """Verify non-inverting adverbs are ruled out as ungrammatical giveaways, and correlative pairs clash accurately."""
        q_when = "Hardly had the meeting begun when the lights went out."
        q_than = "No sooner had the match started than it began to rain."
        q_seldom = "Seldom have I seen such devotion to duty."

        # Non-inverting adverbs clash in fronted inversion slot
        for non_inv in ("often", "completely", "steadily", "fully", "totally", "always"):
            clash, reason = LinguisticEngine.evaluate_syntactic_slot_clash(
                "hardly", non_inv, quote=q_when
            )
            assert clash, f"Expected {non_inv} to clash in fronted inversion slot"
            assert reason == "non_inverting_candidate_in_fronted_inversion_slot"

            clash_s, reason_s = LinguisticEngine.evaluate_syntactic_slot_clash(
                "seldom", non_inv, quote=q_seldom
            )
            assert clash_s, f"Expected {non_inv} to clash in seldom inversion slot"
            assert reason_s == "non_inverting_candidate_in_fronted_inversion_slot"

        # Correlative 'than' vs 'when'/'before' clash
        # 'no sooner' requires 'than' -> clashes in 'when' sentence
        clash_ns, reason_ns = LinguisticEngine.evaluate_syntactic_slot_clash(
            "hardly", "no sooner", quote=q_when
        )
        assert clash_ns
        assert reason_ns == "correlative_when_mismatch_in_inversion_slot"

        # 'hardly', 'scarcely', 'barely' require 'when'/'before' -> clash in 'than' sentence
        for inv in ("hardly", "scarcely", "barely"):
            clash_t, reason_t = LinguisticEngine.evaluate_syntactic_slot_clash(
                "no sooner", inv, quote=q_than
            )
            assert clash_t
            assert reason_t == "correlative_than_mismatch_in_inversion_slot"

        # Aspectual mismatch: frequency adverbs ('seldom', 'rarely') clash with point-in-time 'when'
        for freq in ("seldom", "rarely", "little"):
            clash_f, reason_f = LinguisticEngine.evaluate_syntactic_slot_clash(
                "hardly", freq, quote=q_when
            )
            assert clash_f
            assert reason_f == "aspectual_when_mismatch_in_inversion_slot"

        # Normal non-inverted sentence allows standard adverbs without clash
        clash_norm, _ = LinguisticEngine.evaluate_syntactic_slot_clash(
            "hardly", "often", quote="I could hardly move in the crowded room."
        )
        assert not clash_norm

    def test_double_key_collision_inversion(self):
        """Verify synonymous peers in the same subcategorization frame are detected as double keys."""
        q_when = "Hardly had the meeting begun when the lights went out."
        q_seldom = "Seldom have I seen such devotion to duty."

        # In 'Hardly had ... when ...': 'scarcely' and 'barely' are synonymous double keys
        is_dk1, reason1 = LinguisticEngine.double_key_collision("hardly", "scarcely", quote=q_when)
        assert is_dk1
        assert reason1 == "double_key_pattern"

        is_dk2, reason2 = LinguisticEngine.double_key_collision("hardly", "barely", quote=q_when)
        assert is_dk2
        assert reason2 == "double_key_pattern"

        # In 'Seldom have I seen...': 'rarely' is a synonymous double key
        is_dk3, reason3 = LinguisticEngine.double_key_collision("seldom", "rarely", quote=q_seldom)
        assert is_dk3
        assert reason3 == "double_key_pattern"

        # Reverse direction: 'rarely' vs 'seldom'
        is_dk4, reason4 = LinguisticEngine.double_key_collision("rarely", "seldom", quote=q_seldom)
        assert is_dk4
        assert reason4 == "double_key_pattern"

        # Contrasting foil 'no sooner' in 'when' sentence is NOT a double key (it has correlative clash)
        is_dk5, reason5 = LinguisticEngine.double_key_collision("hardly", "no sooner", quote=q_when)
        assert not is_dk5

    def test_generate_vocab_distractors_inversion_arena(self):
        """Verify end-to-end distractor generation for fronted inversion targets."""
        # Target 1: hardly with 'when'
        q_hardly = "Hardly had the meeting begun when the lights went out."
        distractors_h, meta_h = LinguisticEngine.generate_vocab_distractors(
            target_word="hardly",
            pos="adv",
            quote=q_hardly,
            return_metadata=True,
        )
        assert len(distractors_h) == 3
        # Must be drawn from the negative inversion triggers
        assert set(distractors_h).issubset(LinguisticEngine._ALL_NEGATIVE_INVERSION_TRIGGERS)
        # Must NOT contain synonymous double keys
        assert "scarcely" not in distractors_h
        assert "barely" not in distractors_h
        # Must NOT contain non-inverting adverbs
        assert set(distractors_h).isdisjoint({"often", "completely", "steadily", "fully", "totally"})

        # Target 2: seldom
        q_seldom = "Seldom have I seen such devotion to duty."
        distractors_s, meta_s = LinguisticEngine.generate_vocab_distractors(
            target_word="seldom",
            pos="adv",
            quote=q_seldom,
            return_metadata=True,
        )
        assert len(distractors_s) == 3
        assert set(distractors_s).issubset(LinguisticEngine._ALL_NEGATIVE_INVERSION_TRIGGERS)
        assert "rarely" not in distractors_s
        assert set(distractors_s).isdisjoint({"often", "regularly", "steadily", "frequently"})

        # Target 3: no sooner with 'than'
        q_ns = "No sooner had the match started than it began to rain."
        distractors_ns, meta_ns = LinguisticEngine.generate_vocab_distractors(
            target_word="no sooner",
            pos="phrase",
            quote=q_ns,
            return_metadata=True,
        )
        assert len(distractors_ns) == 3
        assert set(distractors_ns).issubset(LinguisticEngine._ALL_NEGATIVE_INVERSION_TRIGGERS)

    def test_cloze_stem_with_blank_trigger(self):
        """Verify distractor engine operates seamlessly on cloze stems containing '____'."""
        q_blank = "____ had the meeting begun when the lights went out."
        distractors, meta = LinguisticEngine.generate_vocab_distractors(
            target_word="hardly",
            pos="adv",
            quote=q_blank,
            return_metadata=True,
        )
        assert len(distractors) == 3
        assert set(distractors).issubset(LinguisticEngine._ALL_NEGATIVE_INVERSION_TRIGGERS)
        assert "scarcely" not in distractors
        assert "barely" not in distractors
