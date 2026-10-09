"""
Tests for frame-aware Longman pattern matching in the sense lock.

A Longman pattern line ('voice of', 'accustomed to something') states a *frame*: the
words sit next to each other in a real sentence. Scoring them as an unordered bag of
words let a headword and a distant preposition earn +40 for a sentence that never used
the frame, which is how 'hear each other's voices' was pinned to the 'representative'
sense of voice in Book_1_Unit_1_Passage_A.
"""

import pytest

from librarian.linguistics import LinguisticEngine as L


VOICE_QUOTE = ("With the coming of the telephone, people could not only read words, "
               "but also hear each other\u2019s voices.")


def _tokens(sentence: str):
    import re
    return re.findall(r"\b[a-z]+\b", sentence.lower())


class TestPatternFrameWindow:

    def test_words_inside_the_window_form_a_frame(self):
        assert L._quote_has_frame(_tokens("she is accustomed to cold mornings"),
                                  ["accustomed", "to"])

    def test_words_far_apart_are_not_a_frame(self):
        # 'accustomed' and 'to' sit 8 tokens apart; the second 'to' belongs to another
        # construction, so the frame is not evidenced.
        seq = _tokens("she is accustomed because she had been walking to the station")
        assert L._quote_has_frame(seq, ["accustomed", "to"], window=12)
        assert not L._quote_has_frame(seq, ["accustomed", "to"])

    def test_distant_words_rejected(self):
        seq = _tokens("the voice of the senator was heard")
        assert L._quote_has_frame(seq, ["voice", "of"])  # adjacent
        far = _tokens("voices echoed in the hall and the sound of rain followed")
        assert not L._quote_has_frame(far, ["voice", "of"])

    def test_pattern_order_is_preserved(self):
        # 'voice heard' describes 'make their voice heard', not 'hear ... voices'.
        assert L._quote_has_frame(_tokens("they made their voice heard"), ["voice", "heard"])
        assert not L._quote_has_frame(_tokens("they hear each other's voices"),
                                      ["voice", "heard"])

    def test_inflected_surface_form_satisfies_the_pattern_word(self):
        assert L._quote_has_frame(_tokens("he attaches himself to the group"),
                                  ["attach", "to"])

    def test_headword_only_pattern_line_grants_no_pattern_evidence(self):
        # A pattern line that is only the headword ('smart') states no frame, so it must
        # not earn the +15 a real single-word frame like 'smart move' earns.
        entry = L.get_ldoce_entry("smart")
        assert entry
        senses = entry.get("senses") or []
        clothes_idx = next(i for i, s in enumerate(senses)
                           if "tidy appearance" in (s.get("definition") or ""))
        bare = L.sense_confidence(entry, quote="a smart choice", target_pos="adjective")
        solo = {"word": "smart", "senses": [dict(senses[clothes_idx])]}
        isolated = L.sense_confidence(solo, quote="a smart choice", target_pos=None)
        assert isolated["score"] < 15, isolated


class TestSenseLockFrames:

    def test_voice_quote_locks_the_sound_sense(self):
        entry = L.get_ldoce_entry("voice")
        assert entry
        lock = L.sense_confidence(entry, quote=VOICE_QUOTE, target_pos="noun")
        senses = entry.get("senses") or []
        definition = senses[lock["index"]].get("definition", "").lower()
        assert "sounds that you make" in definition, definition

    def test_voice_of_frame_still_selects_the_representative_sense(self):
        entry = L.get_ldoce_entry("voice")
        quote = "The senator is the voice of the religious right."
        lock = L.sense_confidence(entry, quote=quote, target_pos="noun")
        definition = (entry["senses"][lock["index"]].get("definition") or "").lower()
        assert "expresses the opinions" in definition, definition

    def test_valency_pattern_needs_the_headword_in_the_frame(self):
        # 'the voice of' must not degrade into 'the ... of', which any sentence has.
        entry = L.get_ldoce_entry("voice")
        quote = "The cost of the trip was higher than the price of the ticket."
        lock = L.sense_confidence(entry, quote=quote, target_pos="noun")
        definition = (entry["senses"][lock["index"]].get("definition") or "").lower()
        assert "expresses the opinions" not in definition, definition

    def test_placeholder_verb_phrase_is_not_part_of_the_frame(self):
        # 'chance to do something' is a template; the frame it states is 'chance to'.
        assert L._pattern_words("chance to do something") == ["chance", "to"]
        entry = L.get_ldoce_entry("chance")
        quote = ("However, the new ways of communication could give them freedom "
                 "and chance to make smart choices.")
        lock = L.sense_confidence(entry, quote=quote, target_pos="noun")
        definition = (entry["senses"][lock["index"]].get("definition") or "").lower()
        assert "use to do something that you want" in definition, definition

    def test_generic_quote_words_do_not_score_example_overlap(self):
        # 'new' in 'smart new offices' is not evidence that 'make smart choices' is about clothes.
        entry = L.get_ldoce_entry("smart")
        quote = ("However, the new ways of communication could give them freedom "
                 "and chance to make smart choices.")
        lock = L.sense_confidence(entry, quote=quote, target_pos="adjective")
        definition = (entry["senses"][lock["index"]].get("definition") or "").lower()
        assert definition.startswith("intelligent"), definition

    def test_signpost_that_is_a_generic_adverb_is_not_evidence(self):
        # The 'only' signpost of the emphasis sense of simple matched 'only need a simple Hello'.
        entry = L.get_ldoce_entry("simple")
        quote = ("Often, they only need a simple \u201cHello\u201d to change yesterday\u2019s "
                 "strangers into today\u2019s friends.")
        lock = L.sense_confidence(entry, quote=quote, target_pos="adjective")
        definition = (entry["senses"][lock["index"]].get("definition") or "").lower()
        assert "not difficult or complicated" in definition, definition


class TestBareParticlePattern:

    def test_bare_particle_pattern_line_is_a_frame_not_a_bare_word(self):
        # Longman prints 'along' as a one-word pattern line under the 'way of doing
        # something' sense of 'line'. It states 'line along', so a sentence that has
        # 'along' with the headword on the far side of it says nothing about the frame
        # - yet the bare word used to be worth 15 points to every sense that prints it.
        entry = L.get_ldoce_entry("line")
        assert entry
        way_idx = next(i for i, s in enumerate(entry["senses"])
                       if (s.get("definition") or "").startswith("a particular way of doing"))
        solo = {"word": "line", "senses": [dict(entry["senses"][way_idx], patterns=["along"])]}
        quote = "The queue stretched along the line outside the bank."

        with_particles = L.sense_confidence(solo, quote=quote, target_pos=None)["score"]
        saved = L._PATTERN_PARTICLES
        L._PATTERN_PARTICLES = ()
        try:
            bare_word_scoring = L.sense_confidence(solo, quote=quote, target_pos=None)["score"]
        finally:
            L._PATTERN_PARTICLES = saved
        assert with_particles < bare_word_scoring - 10, (with_particles, bare_word_scoring)

    def test_particle_frame_scores_when_the_headword_sits_in_it(self):
        # The rule is not a ban on particles: 'line along' earns its points when the
        # headword is in the frame, exactly as 'serious about' does.
        entry = L.get_ldoce_entry("line")
        way_idx = next(i for i, s in enumerate(entry["senses"])
                       if (s.get("definition") or "").startswith("a particular way of doing"))
        solo = {"word": "line", "senses": [dict(entry["senses"][way_idx])]}
        framed = L.sense_confidence(solo, quote="they lined up along the ridge",
                                    target_pos=None)["score"]
        bare = L.sense_confidence(solo, quote="they lined up on the ridge",
                                  target_pos=None)["score"]
        assert framed > bare, (framed, bare)
