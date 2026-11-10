"""Unit tests for Prepositional Government Hard Constraint Gate.

Verifies deterministic, 0-token, 0ms enforcement of bound prepositional
government clashes for nouns and adjectives in `evaluate_syntactic_slot_clash`
and `double_key_collision`.
"""

import pytest
from librarian.linguistics import LinguisticEngine


class TestPrepositionalGovernmentClash:
    """Tests prepositional valency and government clashes for nouns and adjectives."""

    def test_noun_prep_government_clash_with_anchor_prep(self):
        """Insight takes 'into'; near-synonyms taking 'of' (overview, description) must clash."""
        for distractor in ["overview", "description", "foresight", "perception", "summary"]:
            has_clash, reason = LinguisticEngine.evaluate_syntactic_slot_clash(
                target="insight",
                candidate=distractor,
                anchor="into",
                anchor_type="prep",
                pos="noun"
            )
            assert has_clash is True, f"Expected {distractor} to clash with 'into' slot"
            assert reason == "unattested_prepositional_government_clash"

    def test_noun_prep_government_clash_with_sentence_quote(self):
        """Extracts governed preposition directly from sentence quote via spaCy."""
        quote = "The documentary provides a fascinating insight into the daily lives of ancient Roman citizens."
        for distractor in ["overview", "description", "perception", "summary", "account"]:
            has_clash, reason = LinguisticEngine.evaluate_syntactic_slot_clash(
                target="insight",
                candidate=distractor,
                quote=quote,
                pos="noun"
            )
            assert has_clash is True, f"Expected {distractor} to clash with 'into' extracted from quote"
            assert reason == "unattested_prepositional_government_clash"

        # 'foresight' clashes even earlier due to [U] countability constraint in 'a fascinating ____' slot
        has_clash, reason = LinguisticEngine.evaluate_syntactic_slot_clash(
            target="insight",
            candidate="foresight",
            quote=quote,
            pos="noun"
        )
        assert has_clash is True
        assert reason == "uncountable_noun_in_countable_slot"

    def test_noun_prep_compatible_candidate_does_not_clash(self):
        """Candidates sharing the governed preposition must not trigger government clash."""
        # Both 'solution' and 'access' govern 'to'
        has_clash, _ = LinguisticEngine.evaluate_syntactic_slot_clash(
            target="solution",
            candidate="access",
            anchor="to",
            anchor_type="prep",
            pos="noun"
        )
        assert has_clash is False

    def test_adj_prep_government_clash_with_anchor_prep(self):
        """Adjectives bound to opposing prepositions (allergic to vs aware of / compatible with) must clash."""
        # 'allergic' takes 'to', 'aware' takes 'of', 'compatible' takes 'with'
        has_clash, reason = LinguisticEngine.evaluate_syntactic_slot_clash(
            target="allergic",
            candidate="aware",
            anchor="to",
            anchor_type="prep",
            pos="adj"
        )
        assert has_clash is True
        assert reason == "unattested_prepositional_government_clash"

        has_clash, reason = LinguisticEngine.evaluate_syntactic_slot_clash(
            target="allergic",
            candidate="compatible",
            anchor="to",
            anchor_type="prep",
            pos="adj"
        )
        assert has_clash is True
        assert reason == "unattested_prepositional_government_clash"

        # Reverse: target is 'aware' (governing 'of')
        has_clash, reason = LinguisticEngine.evaluate_syntactic_slot_clash(
            target="aware",
            candidate="allergic",
            anchor="of",
            anchor_type="prep",
            pos="adj"
        )
        assert has_clash is True
        assert reason == "unattested_prepositional_government_clash"

    def test_adj_prep_government_clash_with_sentence_quote(self):
        """Extracts governed preposition for adjectives from sentence quote."""
        quote_allergic = "He is severely allergic to peanuts."
        has_clash, reason = LinguisticEngine.evaluate_syntactic_slot_clash(
            target="allergic",
            candidate="aware",
            quote=quote_allergic,
            pos="adj"
        )
        assert has_clash is True
        assert reason == "unattested_prepositional_government_clash"

        quote_aware = "She was acutely aware of the imminent danger."
        has_clash, reason = LinguisticEngine.evaluate_syntactic_slot_clash(
            target="aware",
            candidate="allergic",
            quote=quote_aware,
            pos="adj"
        )
        assert has_clash is True
        assert reason == "unattested_prepositional_government_clash"

    def test_adj_prep_compatible_candidate_does_not_clash(self):
        """Adjectives sharing the required preposition do not trigger preposition clash."""
        # Both 'allergic' and 'vulnerable' govern 'to'
        has_clash, _ = LinguisticEngine.evaluate_syntactic_slot_clash(
            target="allergic",
            candidate="vulnerable",
            anchor="to",
            anchor_type="prep",
            pos="adj"
        )
        assert has_clash is False

    def test_double_key_collision_exempts_prepositional_clashing_candidates(self):
        """Double key detector marks candidates with government clashes as single-key safe ('frame_only')."""
        is_dk, kind = LinguisticEngine.double_key_collision(
            target="insight",
            distractor="overview",
            anchor="into",
            anchor_type="prep",
            pos="noun"
        )
        assert is_dk is False
        assert kind == "frame_only"

        is_dk, kind = LinguisticEngine.double_key_collision(
            target="allergic",
            distractor="aware",
            anchor="to",
            anchor_type="prep",
            pos="adj"
        )
        assert is_dk is False
        assert kind == "frame_only"

    def test_get_governed_prepositions_accuracy(self):
        """Tests deterministic extraction of governed prepositions across nouns and adjectives."""
        insight_preps = LinguisticEngine.get_governed_prepositions("insight")
        assert "into" in insight_preps

        overview_preps = LinguisticEngine.get_governed_prepositions("overview")
        assert "of" in overview_preps
        assert "into" not in overview_preps

        allergic_preps = LinguisticEngine.get_governed_prepositions("allergic")
        assert "to" in allergic_preps
        assert "of" not in allergic_preps

        aware_preps = LinguisticEngine.get_governed_prepositions("aware")
        assert "of" in aware_preps
        assert "to" not in aware_preps
