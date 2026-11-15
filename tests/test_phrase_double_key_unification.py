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
            ("high praise", ["a34 a40", "cant abide", "humble abode"]),
        ],
    )
    def test_generate_phrase_distractors_avoids_collisions_and_hardcoded_fallbacks(
        self, phrase: str, banned_candidates: list[str]
    ):
        distractors = LinguisticEngine.generate_phrase_distractors(phrase, count=3)
        assert len(distractors) == 3, f"Expected 3 distractors for {phrase}, got {distractors}"
        for cand in distractors:
            assert cand not in banned_candidates, f"Banned candidate '{cand}' appeared in distractors for '{phrase}'"
            assert not any(ch.isdigit() for ch in cand), f"Candidate '{cand}' contains digits!"
            is_col, kind = LinguisticEngine.double_key_collision(phrase, cand, pos="phrase")
            assert not is_col, f"Collision detected for candidate '{cand}' in distractors for '{phrase}' ({kind})"

    def test_nine_mwu_categories_half_slice(self):
        """
        Verifies SLA Lexical Approach: 9 Multi-Word Unit (MWU) assessment families
        produce precise single-word cloze slots or gracefully delegate to whole-unit assessment.
        """
        le = LinguisticEngine

        # 1. Nominal Genitive Idioms (peace of mind)
        hs_peace = le.collocation_half_slice("peace of mind", "Meditation gave her a deep peace of mind during difficult times.")
        assert hs_peace is not None
        assert hs_peace["target_slot"] == "peace"
        assert hs_peace["cloze_slot_frame"] == "... ____ of mind"
        assert all(d != "peace" for d in hs_peace["distractors"])
        assert "tranquillity" in hs_peace["distractors"] or "calm" in hs_peace["distractors"]

        # 2. Prepositional Zero-Article Adverbials (by accident)
        hs_acc = le.collocation_half_slice("by accident", "I found the old letter quite by accident in the drawer.")
        assert hs_acc is not None
        assert hs_acc["target_slot"] == "accident"
        assert hs_acc["cloze_slot_frame"] == "... by ____"
        assert "mishap" in hs_acc["distractors"] or "misfortune" in hs_acc["distractors"]

        # 3. Complex Prepositions (in terms of)
        hs_terms = le.collocation_half_slice("in terms of", "The strategy is effective in terms of long-term growth.")
        assert hs_terms is not None
        assert hs_terms["target_slot"] == "terms"
        assert hs_terms["cloze_slot_frame"] == "... in ____ of"

        # 4. Bare Verbal Idioms (give rise to)
        hs_rise = le.collocation_half_slice("give rise to", "The new policy may give rise to widespread protests.")
        assert hs_rise is not None
        assert hs_rise["target_slot"] == "rise"
        assert hs_rise["cloze_slot_frame"] == "... give ____ to"
        assert "birth" in hs_rise["distractors"] or "way" in hs_rise["distractors"]

        # 5. Phrasal Verbs (shut down)
        hs_shut = le.collocation_half_slice("shut down", "The factory had to shut down after the storm.")
        assert hs_shut is not None
        assert hs_shut["target_slot"] == "down"
        assert hs_shut["cloze_slot_frame"] == "... shut ____"

        # 6. Tripartite Phrasal-Prepositional Verbs (keep in touch with)
        hs_touch = le.collocation_half_slice("keep in touch with", "They promised to keep in touch with each other after graduation.")
        assert hs_touch is not None
        assert hs_touch["target_slot"] == "touch"
        assert hs_touch["cloze_slot_frame"] == "... keep in ____ with"

        # 7. Compound Nouns without attested modifiers: gracefully delegates to whole phrase
        hs_plane = le.collocation_half_slice("passenger plane", "The company purchased a new passenger plane for international routes.")
        assert hs_plane is None
        dists_plane = le.generate_phrase_distractors("passenger plane", count=3)
        assert len(dists_plane) == 3
        assert "passenger ship" in dists_plane or "fighter plane" in dists_plane

        # 8. Verbless / Evaluative Idioms: gracefully delegates to whole phrase
        hs_hats = le.collocation_half_slice("hats off to", "Hats off to the rescue team for their courageous efforts.")
        assert hs_hats is None
        dists_hats = le.generate_phrase_distractors("hats off to", count=3)
        assert len(dists_hats) == 3
        assert "good luck to" in dists_hats or "good riddance to" in dists_hats

        # 9. Binomials: gracefully delegates to whole phrase
        hs_pros = le.collocation_half_slice("pros and cons", "We weighed the pros and cons before deciding.")
        assert hs_pros is None
        dists_pros = le.generate_phrase_distractors("pros and cons", count=3)
        assert len(dists_pros) == 3
        assert "bits and pieces" in dists_pros or "back and forth" in dists_pros

    def test_mwu_phrase_distractors_robustness(self):
        """
        Verifies whole-phrase distractor robustness:
        - 'peace of mind' matches authentic '... of mind' idioms without word family filter false rejection.
        - 'get ahead' excludes passive participles like 'get burned'.
        - 'shut it down' preserves pronominal object symmetry.
        """
        le = LinguisticEngine

        # peace of mind tail match
        dists_peace = le.generate_phrase_distractors("peace of mind", count=3)
        assert len(dists_peace) == 3
        assert any("of mind" in d for d in dists_peace)
        assert "act of god" not in dists_peace
        assert "acres of room" not in dists_peace

        # get ahead particle constraint
        dists_get = le.generate_phrase_distractors("get ahead", count=3)
        assert len(dists_get) == 3
        assert "get burned" not in dists_get
        assert "get burnt" not in dists_get

        # shut it down pronominal mirroring
        dists_shut_it = le.generate_phrase_distractors("shut it down", count=3)
        assert len(dists_shut_it) == 3
        assert all(len(d.split()) == 3 and d.split()[1] == "it" for d in dists_shut_it)

