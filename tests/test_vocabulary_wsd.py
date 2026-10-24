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

    def test_text_contains_form_rejects_hyphenated_sub_tokens(self):
        """Step 2A: text_contains_form(standalone=True) must never match a severed root inside a hyphenated compound."""
        sent_waste_prone = (
            "The shift to an industrial, centralized food system has led to a system that is "
            "increasingly complex, opaque, and — when it comes to food loss — proves to be a waste-prone system."
        )
        assert not LinguisticEngine.text_contains_form(sent_waste_prone, "prone")
        assert LinguisticEngine.text_contains_form(sent_waste_prone, "waste-prone")

        sent_self_indulgent = "How can we learn to be less self-indulgent and put others first?"
        assert not LinguisticEngine.text_contains_form(sent_self_indulgent, "indulgent")
        assert LinguisticEngine.text_contains_form(sent_self_indulgent, "self-indulgent")

    def test_verbatim_compound_promotion_and_lexicon_gate(self):
        """Step 1 & Step 2B: Bare syllabus root promotes to verbatim hyphenated compound in passage,
        shipping only when the compound passes Gate 1 & Gate 2 in PedLex (e.g., 'self-indulgent' ships,
        while 'waste-prone' is dropped by Gate 2)."""
        passage_waste_prone = (
            "The shift to an industrial, centralized food system has led to a system that is "
            "increasingly complex, opaque, and — when it comes to food loss — proves to be a waste-prone system. "
            "Therefore, researchers must design an optimal strategy for sustainable agriculture."
        )
        vocab_wp = LinguisticEngine.extract_deterministic_vocabulary(
            passage_waste_prone,
            target_count=5,
            syllabus_vocab=["prone", "optimal"],
            enable_llm_disambiguation=False,
        )
        extracted_words_wp = [item["word"] for item in vocab_wp]
        assert "prone" not in extracted_words_wp
        assert "waste-prone" not in extracted_words_wp
        assert "optimal" in extracted_words_wp

        passage_indulgent = (
            "Many young consumers struggle with impulsive spending habits in modern society. "
            "How can we learn to be less self-indulgent and cultivate lasting financial discipline?"
        )
        vocab_ind = LinguisticEngine.extract_deterministic_vocabulary(
            passage_indulgent,
            target_count=5,
            syllabus_vocab=["indulgent"],
            enable_llm_disambiguation=False,
        )
        extracted_words_ind = [item["word"] for item in vocab_ind]
        assert "indulgent" not in extracted_words_ind
        assert "self-indulgent" in extracted_words_ind

    def test_mine_vocabulary_skeletons_never_splits_hyphenated_compounds(self):
        """Step 3: Free-mining via spaCy must never extract severed sub-tokens adjacent to a hyphen."""
        text = (
            "The shift to an industrial food system has led to a system that proves to be a waste-prone system "
            "with muscle-bound logistics and self-indulgent consumption."
        )
        skeletons = LinguisticEngine.mine_vocabulary_skeletons(text, target_count=10)
        mined_words = [s["word"] for s in skeletons]
        assert "prone" not in mined_words
        assert "indulgent" not in mined_words
        assert "muscle" not in mined_words
        assert "bound" not in mined_words

    def test_quiz_blueprint_governed_prep_anchor_source(self):
        """Step 4: build_precomputed_target_skeletons must only set anchor_source='quote' for a preposition
        anchor when that preposition is syntactically governed by the target word in the quote."""
        # In this sentence, 'to' appears in 'led to' and 'proves to be', NOT governed by 'prone'.
        sent = (
            "The shift to an industrial food system has led to a system that proves to be a waste-prone system."
        )
        vocab_md = (
            "## [[prone]]\n"
            "- **Part of Speech**: adjective\n"
            "- **Definition**: likely to do something or suffer from something, especially something bad or harmful\n"
            f"- **Quoted Sentence**: {sent}\n"
        )
        skels = LinguisticEngine.build_precomputed_target_skeletons(
            vocab_md,
            target_count=1,
        )
        assert len(skels) == 1
        assert skels[0].get("anchor_source") != "quote"

    def test_symbolic_hard_pruning_in_candidate_pool(self):
        """Symbolic Hard Pruning eliminates grammatically incompatible senses from Top-K candidate pools,
        converting unambiguous polysemous words directly to the 0-token single-candidate fast path
        and resolving near-tie traps (e.g. a/an vs [uncountable], predicative vs [only before noun],
        direct object vs [intransitive], and [always + adverb/preposition] without locative complement)."""
        # 1. Indefinite article 'a/an' vs [uncountable]: 'sentiment' and 'innovation'
        cands_sentiment = LinguisticEngine.build_top_k_vocab_candidates(
            "sentiment",
            pos="noun",
            quote="It's a sentiment shared by many college students.",
            k=4,
        )
        assert len(cands_sentiment) == 1
        assert "[uncountable]" not in cands_sentiment[0]["definition"].lower()

        cands_innovation = LinguisticEngine.build_top_k_vocab_candidates(
            "innovation",
            pos="noun",
            quote="Wait and Shake, an innovation not from its famous survival training course, has become popular.",
            k=4,
        )
        assert len(cands_innovation) == 1
        assert "[uncountable]" not in cands_innovation[0]["definition"].lower()

        # 2. Singular noun with 'a' vs [plural]: 'ethic'
        cands_ethic = LinguisticEngine.build_top_k_vocab_candidates(
            "ethic",
            pos="noun",
            quote="They are known as loyal employees with a strong work ethic.",
            k=4,
        )
        assert len(cands_ethic) == 1
        assert "[plural]" not in (cands_ethic[0]["raw_sense"].get("grammar") or "").lower()

        # 3. Predicative slot vs [only before noun]: 'accustomed' & 'conventional'
        cands_accustomed = LinguisticEngine.build_top_k_vocab_candidates(
            "accustomed",
            pos="adjective",
            quote="Today's young people are accustomed to using smartphones.",
            k=4,
        )
        assert len(cands_accustomed) == 1

        cands_conventional = LinguisticEngine.build_top_k_vocab_candidates(
            "conventional",
            pos="adjective",
            quote="Their work style is not conventional either.",
            k=4,
        )
        assert len(cands_conventional) == 1
        assert "[only before noun]" not in (cands_conventional[0]["raw_sense"].get("grammar") or "").lower()

        # 4. Transitive direct object vs [intransitive] & missing phrasal particle: 'endure' & 'shrug'
        cands_endure = LinguisticEngine.build_top_k_vocab_candidates(
            "endure",
            pos="verb",
            quote="They are willing to endure hardship for the future.",
            k=4,
        )
        assert len(cands_endure) == 1
        assert "[intransitive]" not in (cands_endure[0]["raw_sense"].get("grammar") or "").lower()

        cands_shrug = LinguisticEngine.build_top_k_vocab_candidates(
            "shrug",
            pos="verb",
            quote="You may shrug your shoulders and ignore the comment.",
            k=4,
        )
        assert len(cands_shrug) == 1
        assert "shoulders" in cands_shrug[0]["definition"].lower()

        # 5. Mandatory locative/prepositional complement '[always + adverb/preposition]': 'conduct'
        cands_conduct = LinguisticEngine.build_top_k_vocab_candidates(
            "conduct",
            pos="verb",
            quote="They are more likely to use instant messaging than email to conduct their daily activities.",
            k=4,
        )
        assert len(cands_conduct) >= 1
        assert "carry out a particular activity" in cands_conduct[0]["definition"].lower()
        for c in cands_conduct:
            assert "always + adverb/preposition" not in (c["raw_sense"].get("grammar") or "").lower()

    def test_disambiguate_via_llm_zero_fallback_and_session_logging(self, monkeypatch, tmp_path):
        """Step 3: disambiguate_via_llm must never silently fall back to index 0 (Candidate A)
        when LLM times out or returns unparseable text, and wsd_log_session must flush a FAILED log."""
        from librarian.llm import llm, LLMError
        from librarian.evaluator import LogEvaluator
        import librarian.logger as lib_logger

        monkeypatch.setattr(lib_logger.config, "project_root", tmp_path)

        candidates = [
            {"definition": "Candidate A sense", "example_usage": "Example A."},
            {"definition": "Candidate B sense", "example_usage": "Example B."},
        ]

        # Case 1: LLM raises connection timeout -> raises LLMError (never returns 0) and writes FAILED WSD log
        def _mock_timeout(*args, **kwargs):
            raise TimeoutError("Read timed out after 30s")

        monkeypatch.setattr(llm, "chat", _mock_timeout)
        with pytest.raises(LLMError, match="WSD LLM arbitration failed"):
            with LinguisticEngine.wsd_log_session("Unit_Test_Timeout"):
                LinguisticEngine.disambiguate_via_llm(
                    item_name="compromise",
                    sentence="We reached a compromise.",
                    candidates=candidates,
                )

        log_files = list((tmp_path / "logs").glob("*extract_wsd_Unit_Test_Timeout*.log"))
        assert len(log_files) == 1
        parsed = LogEvaluator.parse_log_file(log_files[0])
        assert parsed["status"] == "FAILED"
        assert parsed["failure_category"] == "CONNECTION_TIMEOUT"
        eval_res = LogEvaluator.evaluate_log(parsed)
        assert eval_res["status"] == "FAILED"
        assert eval_res["composite_score"] == 0.0
        assert any("LLM_EXECUTION_FAILED" in f for f in eval_res["flags"])

        # Case 2: LLM returns unparseable response -> raises LLMError (never returns 0)
        monkeypatch.setattr(llm, "chat", lambda *a, **kw: "I am not sure which sense fits.")
        with pytest.raises(LLMError, match="unparseable"):
            LinguisticEngine.disambiguate_via_llm(
                item_name="compromise",
                sentence="We reached a compromise.",
                candidates=candidates,
            )

    def test_llm_preflight_ensure_available_circuit_breaker(self, monkeypatch, tmp_path):
        """Step 3: llm.ensure_available fails fast and logs LLM_OFFLINE / MODEL_NOT_FOUND when service or model is unavailable."""
        from librarian.llm import llm, LLMError
        from librarian.evaluator import LogEvaluator
        import librarian.logger as lib_logger

        monkeypatch.setattr(lib_logger.config, "project_root", tmp_path)

        monkeypatch.setattr(
            llm,
            "check_health",
            lambda force=False, model_override=None: {
                "online": False,
                "model_ready": False,
                "model": "qwen2.5:7b",
                "active_model": "qwen2.5:7b",
                "available_models": [],
                "api_type": "ollama",
                "api_url": "http://localhost:11434",
                "error": "Connection refused",
            },
        )
        with pytest.raises(LLMError, match="LLM service is unreachable"):
            llm.ensure_available(task_name="pipeline_Unit_1", prompt_summary="Pre-flight check")

        log_files = list((tmp_path / "logs").glob("*pipeline_Unit_1*.log"))
        assert len(log_files) == 1
        parsed = LogEvaluator.parse_log_file(log_files[0])
        assert parsed["status"] == "FAILED"
        assert parsed["failure_category"] == "LLM_OFFLINE"
        eval_res = LogEvaluator.evaluate_log(parsed)
        assert eval_res["status"] == "FAILED"
        assert eval_res["composite_score"] == 0.0




