"""
Unit tests for Noun Countability Symmetry in distractor generation.
Verifies that:
1. For strictly uncountable noun targets:
   - At least 2 distractors are uncountable peers.
   - At most 1 distractor is a countable foil (tagged 'countability_contrast_countable').
2. Symmetrically, for strictly countable noun targets:
   - At least 2 distractors are countable peers.
   - At most 1 distractor is an uncountable foil (tagged 'countability_contrast_uncountable').
3. Bidirectional contrast mapping correctly links countable <-> uncountable pairs.
"""

import pytest
from librarian.linguistics import LinguisticEngine


class TestNounCountabilitySymmetry:

    def test_countability_classification_helpers(self):
        """Verify deterministic classification of primarily uncountable vs countable nouns."""
        unc_words = ["advice", "equipment", "information", "furniture", "luggage", "progress", "homework", "traffic"]
        for w in unc_words:
            assert LinguisticEngine._is_primarily_uncountable_noun(w), f"{w} should be classified as primarily uncountable"
            assert not LinguisticEngine._is_primarily_countable_noun(w), f"{w} should NOT be primarily countable"

        c_words = ["tip", "suggestion", "device", "chair", "tool", "task", "vehicle", "breakthrough"]
        for w in c_words:
            assert LinguisticEngine._is_primarily_countable_noun(w), f"{w} should be classified as primarily countable"
            assert not LinguisticEngine._is_primarily_uncountable_noun(w), f"{w} should NOT be primarily uncountable"

    def test_bidirectional_contrast_map(self):
        """Verify reverse mapping contains expected countable foils for core uncountable nouns."""
        forward_map = LinguisticEngine.get_noun_countability_contrast_map(reverse=False)
        reverse_map = LinguisticEngine.get_noun_countability_contrast_map(reverse=True)

        assert "device" in forward_map
        assert "equipment" in forward_map["device"]

        assert "advice" in reverse_map
        assert "suggestion" in reverse_map["advice"]
        assert "tip" in reverse_map["advice"]

        assert "equipment" in reverse_map
        assert "tool" in reverse_map["equipment"] or "device" in reverse_map["equipment"]

        assert "homework" in reverse_map
        assert "task" in reverse_map["homework"]

        assert "traffic" in reverse_map
        assert "vehicle" in reverse_map["traffic"]

    def test_uncountable_target_advice_symmetry(self):
        """Verify 'advice' receives at least 2 uncountable peers and at most 1 countable foil."""
        quote = "Can you give me some advice on what to buy?"
        res, meta = LinguisticEngine.generate_vocab_distractors(
            "advice", pos="noun", quote=quote, context_anchor="give", return_metadata=True
        )
        assert len(res) == 3
        c_count = sum(1 for d in res if LinguisticEngine._is_primarily_countable_noun(d))
        unc_count = sum(1 for d in res if LinguisticEngine._is_primarily_uncountable_noun(d))

        assert c_count <= 1, f"Expected at most 1 countable foil for 'advice', got {c_count}: {res}"
        assert unc_count >= 2, f"Expected at least 2 uncountable peers for 'advice', got {unc_count}: {res}"
        for d in res:
            if LinguisticEngine._is_primarily_countable_noun(d):
                assert meta.get(d) == "countability_contrast_countable"

    def test_uncountable_target_equipment_symmetry(self):
        """Verify 'equipment' receives at least 2 uncountable peers and at most 1 countable foil."""
        quote = "The hospital needs modern medical equipment."
        res, meta = LinguisticEngine.generate_vocab_distractors(
            "equipment", pos="noun", quote=quote, context_anchor="medical", return_metadata=True
        )
        assert len(res) == 3
        c_count = sum(1 for d in res if LinguisticEngine._is_primarily_countable_noun(d))
        unc_count = sum(1 for d in res if LinguisticEngine._is_primarily_uncountable_noun(d))

        assert c_count <= 1, f"Expected at most 1 countable foil for 'equipment', got {c_count}: {res}"
        assert unc_count >= 2, f"Expected at least 2 uncountable peers for 'equipment', got {unc_count}: {res}"
        for d in res:
            if LinguisticEngine._is_primarily_countable_noun(d):
                assert meta.get(d) == "countability_contrast_countable"

    def test_uncountable_target_homework_symmetry(self):
        """Verify 'homework' receives 'task' as countable contrast foil with remaining uncountable/mass peers."""
        quote = "The teacher gave us a lot of homework for the weekend."
        res, meta = LinguisticEngine.generate_vocab_distractors(
            "homework", pos="noun", quote=quote, context_anchor="lot", return_metadata=True
        )
        assert len(res) == 3
        c_count = sum(1 for d in res if LinguisticEngine._is_primarily_countable_noun(d))
        mass_compat_count = sum(1 for d in res if not LinguisticEngine._is_primarily_countable_noun(d))

        assert c_count <= 1
        assert mass_compat_count >= 2
        assert "task" in res
        assert meta.get("task") == "countability_contrast_countable"

    def test_uncountable_target_traffic_symmetry(self):
        """Verify 'traffic' receives at most 1 countable foil ('vehicle') and at least 2 uncountable/mass peers."""
        quote = "There was heavy traffic on the motorway this morning."
        res, meta = LinguisticEngine.generate_vocab_distractors(
            "traffic", pos="noun", quote=quote, context_anchor="heavy", return_metadata=True
        )
        assert len(res) == 3
        c_count = sum(1 for d in res if LinguisticEngine._is_primarily_countable_noun(d))
        mass_compat_count = sum(1 for d in res if not LinguisticEngine._is_primarily_countable_noun(d))

        assert c_count <= 1
        assert mass_compat_count >= 2
        assert "vehicle" in res
        assert meta.get("vehicle") == "countability_contrast_countable"

    def test_uncountable_target_furniture_symmetry(self):
        """Verify 'furniture' maintains countability symmetry quota."""
        quote = "They bought some antique furniture for their living room."
        res, meta = LinguisticEngine.generate_vocab_distractors(
            "furniture", pos="noun", quote=quote, context_anchor="antique", return_metadata=True
        )
        assert len(res) == 3
        c_count = sum(1 for d in res if LinguisticEngine._is_primarily_countable_noun(d))
        unc_count = sum(1 for d in res if LinguisticEngine._is_primarily_uncountable_noun(d))

        assert c_count <= 1
        assert unc_count >= 2

    def test_countable_target_device_symmetry(self):
        """Verify countable 'device' receives at least 2 countable peers and at most 1 uncountable foil."""
        quote = "He invented a new device for measuring temperature."
        res, meta = LinguisticEngine.generate_vocab_distractors(
            "device", pos="noun", quote=quote, context_anchor="new", return_metadata=True
        )
        assert len(res) == 3
        c_count = sum(1 for d in res if LinguisticEngine._is_primarily_countable_noun(d))
        unc_count = sum(1 for d in res if LinguisticEngine._is_primarily_uncountable_noun(d))

        assert unc_count <= 1, f"Expected at most 1 uncountable foil for 'device', got {unc_count}: {res}"
        assert c_count >= 2, f"Expected at least 2 countable peers for 'device', got {c_count}: {res}"
        has_unc_foil = any(meta.get(d) == "countability_contrast_uncountable" for d in res)
        assert has_unc_foil

    def test_countable_target_suggestion_symmetry(self):
        """Verify countable 'suggestion' receives at least 2 countable peers and at most 1 uncountable foil."""
        quote = "She made a helpful suggestion during the meeting."
        res, meta = LinguisticEngine.generate_vocab_distractors(
            "suggestion", pos="noun", quote=quote, context_anchor="helpful", return_metadata=True
        )
        assert len(res) == 3
        unc_count = sum(1 for d in res if LinguisticEngine._is_primarily_uncountable_noun(d))
        assert unc_count <= 1
        has_unc_foil = any(meta.get(d) == "countability_contrast_uncountable" for d in res)
        assert has_unc_foil

    def test_countable_target_tool_symmetry(self):
        """Verify countable 'tool' receives at least 2 countable peers and at most 1 uncountable foil ('equipment')."""
        quote = "A hammer is a useful tool for building."
        res, meta = LinguisticEngine.generate_vocab_distractors(
            "tool", pos="noun", quote=quote, context_anchor="useful", return_metadata=True
        )
        assert len(res) == 3
        c_count = sum(1 for d in res if LinguisticEngine._is_primarily_countable_noun(d))
        unc_count = sum(1 for d in res if LinguisticEngine._is_primarily_uncountable_noun(d))

        assert unc_count <= 1
        assert c_count >= 2
        assert "equipment" in res
        assert meta.get("equipment") == "countability_contrast_uncountable"
