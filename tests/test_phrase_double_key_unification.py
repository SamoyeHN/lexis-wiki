import pytest
from librarian.linguistics import LinguisticEngine


class TestPhraseDoubleKeyUnification:
    """
    Tests for Step 2: Multi-word phrase double-key collision unification and front-loading.
    Verifies that phrase collisions across PedLex slash groups, WordNet synsets,
    preposition synonym frames, and 1-token-difference clusters are strictly intercepted.
    """

    @pytest.mark.parametrize(
        "phrase1,phrase2",
        [
            ("in light of", "in view of"),
            ("for the sake of", "for the good of"),
            ("bear in mind", "keep in mind"),
            ("show up", "turn up"),
            ("care about", "worry about"),
            ("keep in touch with", "stay in touch with"),
            ("point of view", "standpoint"),
            # Inflected forms
            ("showed up", "turned up"),
            ("cared about", "worried about"),
            ("keeps in mind", "bears in mind"),
            # Possessive variants
            ("come to one's aid", "come to her aid"),
        ],
    )
    def test_phrase_double_key_collisions_positive(self, phrase1: str, phrase2: str):
        is_col, kind = LinguisticEngine.double_key_collision(phrase1, phrase2, pos="phrase")
        assert is_col, f"Expected collision between '{phrase1}' and '{phrase2}', got False ({kind})"

    @pytest.mark.parametrize(
        "phrase1,phrase2",
        [
            ("pros and cons", "bits and pieces"),
            ("peace of mind", "state of mind"),
            ("worry about", "wonder about"),
            ("worry about", "complain about"),
            ("show up", "show off"),
            ("point of view", "field of view"),
        ],
    )
    def test_phrase_double_key_collisions_negative(self, phrase1: str, phrase2: str):
        is_col, kind = LinguisticEngine.double_key_collision(phrase1, phrase2, pos="phrase")
        assert not is_col, f"Unexpected collision between '{phrase1}' and '{phrase2}' ({kind})"

    @pytest.mark.parametrize(
        "phrase,banned_candidates",
        [
            ("worry about", ["care about", "carry out", "take over"]),
            ("show up", ["turn up", "come out"]),
            ("keep in touch with", ["stay in touch with", "be in touch with"]),
        ],
    )
    def test_generate_phrase_distractors_avoids_collisions_and_hardcoded_fallbacks(
        self, phrase: str, banned_candidates: list[str]
    ):
        distractors = LinguisticEngine.generate_phrase_distractors(phrase, count=3)
        assert len(distractors) == 3, f"Expected 3 distractors for {phrase}, got {distractors}"
        for cand in distractors:
            assert cand not in banned_candidates, f"Banned candidate '{cand}' appeared in distractors for '{phrase}'"
            is_col, kind = LinguisticEngine.double_key_collision(phrase, cand, pos="phrase")
            assert not is_col, f"Collision detected for candidate '{cand}' in distractors for '{phrase}' ({kind})"
