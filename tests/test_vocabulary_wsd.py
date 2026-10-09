import pytest
from librarian.linguistics import LinguisticEngine


class TestVocabularyWSD:
    """
    Test suite for Syntax-Pattern Guided WSD (Word Sense Disambiguation)
    validating the 9 key misalignments identified in scripts/检查方式.md.
    """

    def test_potluck_disambiguation(self):
        # Quote: "I like to hold potlucks at the office and invite volunteers to make them feel like part of the team."
        quote = "I like to hold potlucks at the office and invite volunteers to make them feel like part of the team."
        entry = LinguisticEngine.get_ldoce_entry("potluck")
        assert entry is not None
        idx = LinguisticEngine._lock_sense(entry, quote=quote, target_pos="noun")
        # Sense 2: a meal in which everyone who is invited brings something to eat
        assert idx is not None
        s = entry["senses"][idx]
        assert "meal" in s.get("definition", "").lower()

    def test_tip_greeting_disambiguation(self):
        # Quote: "The staff wore hats all week and tipped them toward the volunteers as we saw them."
        quote = "The staff wore hats all week and tipped them toward the volunteers as we saw them."
        entry = LinguisticEngine.get_ldoce_entry("tip")
        assert entry is not None
        idx = LinguisticEngine._lock_sense(entry, quote=quote, target_pos="verb")
        # Sense 18: to touch or raise your hat as a greeting to someone
        assert idx is not None
        s = entry["senses"][idx]
        assert any(k in s.get("definition", "").lower() for k in ("hat", "greeting", "raise", "touch"))

    def test_team_work_disambiguation(self):
        # Quote: "I like to hold potlucks at the office and invite volunteers to make them feel like part of the team."
        quote = "I like to hold potlucks at the office and invite volunteers to make them feel like part of the team."
        entry = LinguisticEngine.get_ldoce_entry("team")
        assert entry is not None
        idx = LinguisticEngine._lock_sense(entry, quote=quote, target_pos="noun")
        # Sense 1: a group of people who have been chosen to work together to do a particular job
        assert idx == 1
        s = entry["senses"][idx]
        assert "work together" in s.get("definition", "").lower()

    def test_recognition_appreciation_disambiguation(self):
        # Quote: "We had our recognition awards program before the movie started, followed by a 15-minute coffee break."
        quote = "We had our recognition awards program before the movie started, followed by a 15-minute coffee break."
        entry = LinguisticEngine.get_ldoce_entry("recognition")
        assert entry is not None
        idx = LinguisticEngine._lock_sense(entry, quote=quote, target_pos="noun")
        # Sense 1: public respect and thanks for someone's work or achievements
        assert idx == 1
        s = entry["senses"][idx]
        assert any(k in s.get("definition", "").lower() for k in ("respect", "thanks", "achievements", "work"))

    def test_post_internet_disambiguation(self):
        # Quote 1: "Here are some posts online to thank those who have volunteered to offer all kinds of help." (Noun)
        quote_noun = "Here are some posts online to thank those who have volunteered to offer all kinds of help."
        entry = LinguisticEngine.get_ldoce_entry("post")
        assert entry is not None
        idx_noun = LinguisticEngine._lock_sense(entry, quote=quote_noun, target_pos="noun")
        assert idx_noun is not None
        s_noun = entry["senses"][idx_noun]
        assert any(k in s_noun.get("definition", "").lower() for k in ("internet", "message", "online", "discussion"))

        # Quote 2: "(Posted on 29 December, 2022 by Nazia Anderson)" (Verb)
        quote_verb = "(Posted on 29 December, 2022 by Nazia Anderson)"
        idx_verb = LinguisticEngine._lock_sense(entry, quote=quote_verb, target_pos="verb")
        assert idx_verb is not None
        s_verb = entry["senses"][idx_verb]
        assert any(k in s_verb.get("definition", "").lower() for k in ("internet", "message", "online", "send"))

    def test_work_effective_disambiguation(self):
        # Quote: "Here’s a very inexpensive idea that should work for any type of organization."
        quote = "Here’s a very inexpensive idea that should work for any type of organization."
        entry = LinguisticEngine.get_ldoce_entry("work")
        assert entry is not None
        idx = LinguisticEngine._lock_sense(entry, quote=quote, target_pos="verb")
        # Sense 7: to be effective or successful (unpacked from sense 6a/6b)
        assert idx is not None
        s = entry["senses"][idx]
        assert "effective" in s.get("definition", "").lower()

    def test_cut_out_phrasal_verb_disambiguation(self):
        # Quote: "The teens at our community help me cut them out."
        quote = "The teens at our community help me cut them out."
        # Testing expression definition and example
        defn, ex, src = LinguisticEngine.expression_definition_evidence(
            phrase="cut out",
            expr_type="phrasal verb",
            context_sentence=quote
        )
        assert src in ("phrasal_verb_block", "ldoce_entry")
        assert "shape" in defn.lower() or "cutting" in defn.lower()
        # Must not be the noun cutout definition
        assert not defn.startswith("the shape of a person, object etc that has been cut out of wood")
        # Example must be an authentic dictionary example, not verbatim quote
        assert ex and ex != quote

    def test_program_activity_disambiguation(self):
        # Quote: "We had our recognition awards program before the movie started, followed by a 15-minute coffee break."
        quote = "We had our recognition awards program before the movie started, followed by a 15-minute coffee break."
        defn, ex = LinguisticEngine.get_ldoce_definition_and_example(
            "program",
            target_pos="noun",
            context_sentence=quote
        )
        assert defn
        # Should not be computer software
        assert "computer" not in defn.lower()
        # Should align with event/plan/activities
        assert any(k in defn.lower() for k in ("action", "plan", "activity", "event", "series", "programme"))

    def test_stranger_wsd_passage_a(self):
        # Passage A sentence: "Often, they only need a simple 'Hello' to change yesterday’s strangers into today’s friends."
        quote = "Often, they only need a simple “Hello” to change yesterday’s strangers into today’s friends."
        entry = LinguisticEngine.get_ldoce_entry("stranger")
        assert entry is not None
        idx = LinguisticEngine._lock_sense(entry, quote=quote, target_pos="noun")
        # Sense 0: someone that you do not know (MUST NOT be "hello, stranger!" greeting sense)
        assert idx == 0
        s = entry["senses"][idx]
        assert "not know" in s.get("definition", "").lower()
        assert "greet" not in s.get("definition", "").lower()

    def test_make_possible_expression_definition(self):
        # Quote: "The new ways of communication make this possible."
        quote = "The new ways of communication make this possible."
        defn, ex, src = LinguisticEngine.expression_definition_evidence(
            phrase="make possible",
            expr_type="collocation",
            context_sentence=quote
        )
        assert defn
        # Collocation/causative meaning: to cause something to happen, or cause a particular state or condition
        assert "cause" in defn.lower() or "happen" in defn.lower()
        # Must not be the static adjective definition "if something is possible, it can be done or achieved"
        assert not defn.startswith("if something is possible")

    def test_cleft_and_correlative_grammar_warnings(self):
        # Test Cleft sentence warning does not get hijacked by "that"
        cleft_text = "In the past, it was a small stamp that helped family members and friends to keep in touch with each other."
        g_cleft = LinguisticEngine.extract_deterministic_grammar(cleft_text)
        assert len(g_cleft) >= 1
        cleft_item = next(g for g in g_cleft if "Cleft" in g.get("syntax_topic", ""))
        assert "Cleft Sentences" in cleft_item["common_mistakes"]
        assert "Don't use another negative word" not in cleft_item["common_mistakes"]

        # Test Correlative coordination warning does not get hijacked by "not"
        correlative_text = "With the coming of the telephone, people could not only read words, but also hear each other’s voices."
        g_corr = LinguisticEngine.extract_deterministic_grammar(correlative_text)
        assert len(g_corr) >= 1
        corr_item = next(g for g in g_corr if "Correlative" in g.get("syntax_topic", ""))
        assert "Correlative Coordinators" in corr_item["common_mistakes"]
        assert "Don't use another negative word" not in corr_item["common_mistakes"]

    def test_lookup_ldoce_phrase_dual_track(self):
        # Track 1 & 2 exact and contraction matching
        res1 = LinguisticEngine.lookup_ldoce_phrase("hello, stranger!")
        assert res1 is not None
        assert res1[0] == "stranger"
        assert res1[1] == 3

        res2 = LinguisticEngine.lookup_ldoce_phrase("be no stranger to")
        assert res2 is not None
        assert res2[0] == "stranger"
        assert res2[1] == 1

        res3 = LinguisticEngine.lookup_ldoce_phrase("don't be a stranger!")
        assert res3 is not None
        assert res3[0] == "stranger"
        assert res3[1] == 4

        # Track 3: lemmatized verb fallback
        res4 = LinguisticEngine.lookup_ldoce_phrase("banked on")
        assert res4 is not None
        assert res4[0] == "bank"

    def test_format_ldoce_definition(self):
        sense_with_labels = {
            "definition": "used to greet someone",
            "register": "spoken",
            "variety": "American English"
        }
        formatted = LinguisticEngine.format_ldoce_definition(sense_with_labels)
        assert formatted == "[American English, spoken] used to greet someone"

        # Safe fallback without labels
        sense_plain = {"definition": "a plain definition"}
        assert LinguisticEngine.format_ldoce_definition(sense_plain) == "a plain definition"

    def test_stranger_sense_lock_not_hijacked_by_quoted_hello(self):
        entry = LinguisticEngine.get_ldoce_entry("stranger")
        quote = "Often, they only need a simple 'Hello' to change yesterday's strangers into today's friends."
        locked_idx, score, margin, conf = LinguisticEngine._lock_sense(entry, quote=quote, target_pos="noun", return_confidence=True)
        # Sense 0 is "someone that you do not know", NOT Sense 3 ("used to greet someone who you have not seen for a long time")
        assert locked_idx == 0

        # In contrast, an actual formulaic greeting locution MUST lock to Sense 3
        greeting_quote = "Hello, stranger! It has been years since we met."
        locked_greeting_idx, _, _, _ = LinguisticEngine._lock_sense(entry, quote=greeting_quote, target_pos="noun", return_confidence=True)
        assert locked_greeting_idx == 3

    def test_generate_phrase_distractors_from_index(self):
        """Verifies that multi-word distractors are strictly grounded in LDOCE's ldoce_phrase_index."""
        # 1. Tail-sharing 3-word noun phrases ('peace of mind' -> sister '... of mind' phrases)
        dists_mind = LinguisticEngine.generate_phrase_distractors("peace of mind", count=3)
        assert len(dists_mind) == 3
        for d in dists_mind:
            assert len(d.split()) == 3
            assert d.endswith(" of mind")
            assert d != "peace of mind"
        assert "frame of mind" in dists_mind or "state of mind" in dists_mind or "cast of mind" in dists_mind

        # 2. Tail-sharing multi-word noun phrase ('quality of life' -> 'sign of life', 'expectation of life', 'time of life')
        dists_life = LinguisticEngine.generate_phrase_distractors("quality of life", count=3)
        assert len(dists_life) == 3
        for d in dists_life:
            assert d.endswith(" of life")
            assert d != "quality of life"
        assert "sign of life" in dists_life or "expectation of life" in dists_life or "time of life" in dists_life

        # 3. Binomial coordinate symmetrical phrases ('pros and cons')
        dists_pros = LinguisticEngine.generate_phrase_distractors("pros and cons", count=3)
        assert len(dists_pros) == 3
        for d in dists_pros:
            assert len(d.split()) == 3
            assert d.split()[1] in ("and", "or")
            assert d != "pros and cons"
        assert "bits and pieces" in dists_pros or "back and forth" in dists_pros or "give and take" in dists_pros

        # 4. Light-verb / construct prefix match ('have a try')
        dists_try = LinguisticEngine.generate_phrase_distractors("have a try", count=3)
        assert len(dists_try) == 3
        for d in dists_try:
            assert len(d.split()) == 3
            assert d != "have a try"

        # 5. 2-word phrasal verbs ('send out' -> particle contrast & verb contrast)
        dists_send = LinguisticEngine.generate_phrase_distractors("send out", count=3)
        assert len(dists_send) == 3
        for d in dists_send:
            assert len(d.split()) == 2
            assert d != "send out"

    def test_line_queue_sense_beats_the_railway_sense(self):
        # Quote: "The queue stretched along the line outside the bank for over an hour."
        # 'along' describes the shape of the queue; the railway sense of 'line' shares
        # nothing with this sentence but the particle.
        quote = "The queue stretched along the line outside the bank for over an hour."
        entry = LinguisticEngine.get_ldoce_entry("line")
        assert entry is not None
        idx = LinguisticEngine._lock_sense(entry, quote=quote, target_pos="noun")
        assert idx is not None
        assert "waiting one behind the other" in entry["senses"][idx]["definition"].lower(), (
            entry["senses"][idx]["definition"])

        # The railway sense must still win when the sentence talks about a railway.
        rail = "We were delayed because of a problem further along the railway line."
        idx_rail = LinguisticEngine._lock_sense(entry, quote=rail, target_pos="noun")
        assert "track that a train" in entry["senses"][idx_rail]["definition"].lower(), (
            entry["senses"][idx_rail]["definition"])


