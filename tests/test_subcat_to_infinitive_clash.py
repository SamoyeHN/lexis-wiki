"""Tests for Subcategorization Frame Invariant: to + infinitive vs to + gerund/preposition.

Verifies zero-token deterministic syntactic slot clash detection between:
  1. Words governing infinitive complements (to + do), e.g. reluctance, refusal, hesitant
  2. Words governing prepositional/gerund complements (to + noun/gerund), e.g. aversion, objection, committed
  3. Dual-taking words, e.g. willingness, tendency
"""

import pytest
from librarian.linguistics import LinguisticEngine


class TestSubcatToInfinitiveClash:
    """Subcategorization frame clash test suite."""

    def test_word_to_complement_profile(self):
        """Verify deterministic extraction of complement profile from PedLex/LDOCE/OCD."""
        # Infinitive-only nouns and adjectives
        inf_words = ["reluctance", "refusal", "decision", "hesitant"]
        for w in inf_words:
            c_inf, c_prep = LinguisticEngine.get_word_to_complement_profile(w)
            assert c_inf is True, f"{w} should accept to-infinitive complement"
            assert c_prep is False, f"{w} should NOT accept to-preposition/gerund complement"

        # Preposition/gerund-only nouns and adjectives
        prep_words = ["aversion", "objection", "resistance", "committed", "allergic"]
        for w in prep_words:
            c_inf, c_prep = LinguisticEngine.get_word_to_complement_profile(w)
            assert c_prep is True, f"{w} should accept to-preposition/gerund complement"
            assert c_inf is False, f"{w} should NOT accept to-infinitive complement"

        # Dual-taking nouns
        dual_words = ["willingness", "tendency"]
        for w in dual_words:
            c_inf, c_prep = LinguisticEngine.get_word_to_complement_profile(w)
            assert c_inf is True, f"{w} should accept to-infinitive complement"
            assert c_prep is True, f"{w} should accept to-preposition complement"

    def test_noun_infinitive_slot_bars_gerund_nouns(self):
        """In an infinitive slot ('reluctance to speak'), aversion/objection/resistance are barred."""
        quote = "Despite his initial reluctance to speak in public, he delivered an inspiring speech."
        for cand in ["aversion", "objection", "resistance"]:
            has_clash, reason = LinguisticEngine.evaluate_syntactic_slot_clash(
                "reluctance", cand, quote=quote, pos="noun"
            )
            assert has_clash is True
            assert reason == "incompatible_to_infinitive_subcat_clash"

    def test_noun_gerund_slot_bars_infinitive_nouns(self):
        """In a gerund/preposition slot ('objections to killing animals'), refusal/decision are barred."""
        quote = "He had strong objections to killing animals for food."
        for cand in ["refusal", "decision"]:
            has_clash, reason = LinguisticEngine.evaluate_syntactic_slot_clash(
                "objection", cand, quote=quote, pos="noun"
            )
            assert has_clash is True
            assert reason == "incompatible_to_gerund_subcat_clash"

    def test_adj_infinitive_slot_bars_gerund_adjectives(self):
        """In an infinitive adjective slot ('hesitant to speak'), committed/allergic are barred."""
        quote = "He was hesitant to speak before the large audience."
        for cand in ["committed", "allergic"]:
            has_clash, reason = LinguisticEngine.evaluate_syntactic_slot_clash(
                "hesitant", cand, quote=quote, pos="adj"
            )
            assert has_clash is True
            assert reason == "incompatible_to_infinitive_subcat_clash"

    def test_adj_gerund_slot_bars_infinitive_adjectives(self):
        """In a gerund adjective slot ('committed to helping children'), hesitant/reluctant are barred."""
        quote = "She is deeply committed to helping underprivileged children."
        for cand in ["hesitant", "reluctant"]:
            has_clash, reason = LinguisticEngine.evaluate_syntactic_slot_clash(
                "committed", cand, quote=quote, pos="adj"
            )
            assert has_clash is True
            assert reason == "incompatible_to_gerund_subcat_clash"

    def test_dual_taking_words_no_false_positive(self):
        """Dual-taking words like willingness and tendency do not trigger false-positive clashes."""
        q_inf = "Despite his initial reluctance to speak in public, he delivered an inspiring speech."
        has_clash, _ = LinguisticEngine.evaluate_syntactic_slot_clash(
            "reluctance", "willingness", quote=q_inf, pos="noun"
        )
        assert has_clash is False

        q_gerund = "He noted a strong objection to killing animals for food."
        has_clash, _ = LinguisticEngine.evaluate_syntactic_slot_clash(
            "objection", "tendency", quote=q_gerund, pos="noun"
        )
        assert has_clash is False

    def test_cloze_blank_stem_syntax_parsing(self):
        """Cloze blank stems with '____' correctly trigger subcat clash."""
        blank_quote = "Despite his initial ____ to speak in public, he delivered an inspiring speech."
        has_clash, reason = LinguisticEngine.evaluate_syntactic_slot_clash(
            "reluctance", "aversion", quote=blank_quote, pos="noun"
        )
        assert has_clash is True
        assert reason == "incompatible_to_infinitive_subcat_clash"

    def test_anchor_only_detection(self):
        """Complement slot is identified directly from anchor without quote."""
        has_clash, reason = LinguisticEngine.evaluate_syntactic_slot_clash(
            "reluctance", "aversion", anchor="to speak", pos="noun"
        )
        assert has_clash is True
        assert reason == "incompatible_to_infinitive_subcat_clash"

        has_clash, reason = LinguisticEngine.evaluate_syntactic_slot_clash(
            "objection", "refusal", anchor="to killing", pos="noun"
        )
        assert has_clash is True
        assert reason == "incompatible_to_gerund_subcat_clash"

    def test_double_key_exemption_on_subcat_clash(self):
        """When subcat clash occurs, double_key_collision exempts the candidate as frame_only."""
        quote = "Despite his initial reluctance to speak in public, he delivered an inspiring speech."
        is_collision, kind = LinguisticEngine.double_key_collision(
            "reluctance", "aversion", quote=quote, pos="noun"
        )
        assert is_collision is False
        assert kind == "frame_only"
