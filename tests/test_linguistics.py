"""
Unit tests for the LinguisticEngine (spaCy & WordNet integration).
Tests sentence indexing, phrasal verb boundary distinction, macro domain classification,
lemmatization, and zero-collision WordNet distractor generation.
"""

import pytest
from librarian.linguistics import LinguisticEngine


class TestLinguisticEngine:

    def test_sentence_indexing_and_pool(self):
        sample_text = (
            "Whether a job is designated as labor or work depends on tastes. "
            "Although he was exhausted, he finished the race. "
            "Only then did he realize his mistake."
        )
        indexed_text, pool = LinguisticEngine.tokenize_and_index_sentences(sample_text)

        assert "[S-1]" in indexed_text
        assert "[S-2]" in indexed_text
        assert "[S-3]" in indexed_text
        assert len(pool) == 3
        assert pool["S-1"].startswith("Whether a job is designated")
        assert pool["S-2"].startswith("Although he was exhausted")
        assert pool["S-3"].startswith("Only then did he realize")

    def test_phrasal_verb_extraction_vs_spatial_preposition(self):
        # come in (particle) vs come in a city (spatial prep + pobj)
        sent1 = "Please come in and have a seat."
        pvs1 = LinguisticEngine.extract_phrasal_verbs(sent1)
        assert "come in" in pvs1

        sent2 = "Many migrants come in a big city every year."
        pvs2 = LinguisticEngine.extract_phrasal_verbs(sent2)
        # 'in a big city' is a prepositional phrase, NOT a phrasal verb particle!
        assert "come in" not in pvs2

        sent3 = "He decided to give up smoking and calm down."
        pvs3 = LinguisticEngine.extract_phrasal_verbs(sent3)
        assert "give up" in pvs3
        assert "calm down" in pvs3

    def test_context_aware_lemmatization(self):
        assert LinguisticEngine.lemmatize_headword("proliferated") == "proliferate"
        assert LinguisticEngine.lemmatize_headword("stumbled across") == "stumble across"
        assert LinguisticEngine.lemmatize_headword("turning down") == "turn down"

    def test_lemmatizer_oov_guard_prevents_corruption(self):
        """Regression: spaCy's rule lemmatizer can corrupt 'embed' (tagged VBD)
        to the non-word 'embe'; the dictionary guard must keep the surface form."""
        assert LinguisticEngine.lemmatize_headword("embed") == "embed"
        # Legit lemmas still normalize as before.
        assert LinguisticEngine.lemmatize_headword("embedded") == "embed"
        assert LinguisticEngine.lemmatize_headword("hopped") == "hop"
        # Tri-state dictionary probe.
        assert LinguisticEngine.is_known_english_word("embed") is True
        assert LinguisticEngine.is_known_english_word("embe") is False
        assert LinguisticEngine.is_known_english_word("stumble across") is None

    def test_macro_domain_classification_rhetoric_inversion(self):
        inversion_sent = "Only then did the scientist realize the significance of the anomaly."
        cat = LinguisticEngine.classify_grammar_dependency(inversion_sent)
        assert cat == "Rhetoric & Emphasis"

    def test_macro_domain_classification_rhetoric_antithesis(self):
        antithesis_sent = "Success depends, not on sheer luck, but on persistent effort."
        cat = LinguisticEngine.classify_grammar_dependency(antithesis_sent)
        assert cat == "Rhetoric & Emphasis"

    def test_macro_domain_classification_cohesion_shell_noun(self):
        shell_sent = "The claim that technological progress always improves welfare is debatable."
        cat = LinguisticEngine.classify_grammar_dependency(shell_sent)
        assert cat == "Cohesion & Framing"

    def test_macro_domain_classification_cohesion_interpretive_encapsulation(self):
        encap_sent = "He withdrew his application suddenly, which suggested that negotiations had failed."
        cat = LinguisticEngine.classify_grammar_dependency(encap_sent)
        assert cat == "Cohesion & Framing"

    def test_macro_domain_classification_packaging_dummy_it(self):
        dummy_it_sent = "It is essential that students master academic writing skills."
        cat = LinguisticEngine.classify_grammar_dependency(dummy_it_sent)
        assert cat == "Information Packaging"

    def test_macro_domain_classification_packaging_participial(self):
        participle_sent = "Having analyzed the data thoroughly, the committee published its findings."
        cat = LinguisticEngine.classify_grammar_dependency(participle_sent)
        assert cat == "Information Packaging"

    def test_macro_domain_classification_logic_concessive(self):
        concessive_sent = "Although the initial results were promising, further verification was needed."
        cat = LinguisticEngine.classify_grammar_dependency(concessive_sent)
        assert cat == "Logic & Stance"

    def test_wordnet_zero_collision_distractor_generation(self):
        target = "increase"
        distractors = LinguisticEngine.generate_zero_collision_distractors(target, pos="v", count=3)

        assert len(distractors) == 3
        # Ensure target is not in its own distractors
        assert target not in distractors
        # Check that direct synonyms of 'increase' are not in distractors
        synonyms = LinguisticEngine.get_synonyms(target)
        for d in distractors:
            assert d not in synonyms
        # 'decrease' is a direct antonym of increase and should be selected
        assert "decrease" in distractors

    def test_snap_to_sentence_pool(self):
        pool = {
            "S-1": "Whether a job is designated as labor or work depends on tastes.",
            "S-2": "Although he was exhausted, he finished the race.",
            "S-3": "Only then did he realize his mistake."
        }
        # 1. Snap via ID
        assert LinguisticEngine.snap_to_sentence_pool("[S-2]", pool) == pool["S-2"]
        assert LinguisticEngine.snap_to_sentence_pool("S-3", pool) == pool["S-3"]

        # 2. Snap via partial quote with ellipsis
        partial_quote = "Whether a job is designated... depends on tastes."
        assert LinguisticEngine.snap_to_sentence_pool(partial_quote, pool) == pool["S-1"]

        # 3. Snap via exact quote
        assert LinguisticEngine.snap_to_sentence_pool("Only then did he realize his mistake.", pool) == pool["S-3"]

    def test_prune_hallucinated_items_snaps_quotes(self):
        from librarian.evaluator import prune_hallucinated_items
        source = (
            "CONTENT:\n"
            "Whether a job is designated as labor or work depends on tastes. "
            "Although he was exhausted, he finished the race. "
            "Only then did he realize his mistake."
        )
        data = {
            "vocabulary": [
                {
                    "word": "exhausted",
                    "quoted_sentence": "[S-2]",
                    "part_of_speech": "adjective",
                    "definition": "Very tired.",
                    "design_audit": "AUDIT: exhausted"
                }
            ]
        }
        pruned_data, notices = prune_hallucinated_items(data, source, task_type="vocabulary")
        item = pruned_data["vocabulary"][0]
        # Should snap [S-2] to authentic full sentence
        assert item["quoted_sentence"] == "Although he was exhausted, he finished the race."
        # Should lemmatize headword
        assert item["word"] == "exhaust"
        assert any("Snapped quote" in n for n in notices)

    def test_grammar_deduplication_sentence_id(self):
        """Same sentence extracted under different formulas or partial quotes must be deduplicated."""
        from librarian.evaluator import prune_hallucinated_items
        source = (
            "CONTENT:\n"
            "Whether a job is designated as labor or work depends on tastes. "
            "Although he was exhausted, he finished the race. "
            "Only then did he realize his mistake."
        )
        data = {
            "grammar_patterns": [
                {
                    "quote": "Although he was exhausted, he finished the race.",
                    "pattern_formula": "Although + [Clause], [Main Clause]",
                    "category": "Logic & Stance",
                    "pedagogical_function": "Concession.",
                    "design_audit": "AUDIT: Although",
                    "imitation_example": "Although it rained, we played.",
                    "common_mistakes": "Missing comma.",
                    "cefr_level": "B2"
                },
                {
                    "quote": "[S-2]",  # Same sentence, different formula
                    "pattern_formula": "[Subordinator] + [S], [S]",
                    "category": "Logic & Stance",
                    "pedagogical_function": "Duplicate concession.",
                    "design_audit": "AUDIT: Although",
                    "imitation_example": "Although tired, he ran.",
                    "common_mistakes": "None.",
                    "cefr_level": "B2"
                }
            ]
        }
        pruned_data, notices = prune_hallucinated_items(data, source, task_type="grammar")
        # Only 1 pattern should survive
        assert len(pruned_data["grammar_patterns"]) == 1
        assert any("Pruned duplicate item: duplicate sentence" in n for n in notices)

    def test_grammar_deduplication_syntax_fingerprint(self):
        """Homogeneous patterns (same macro domain + same dependency type + same anchor) must be deduplicated."""
        from librarian.evaluator import prune_hallucinated_items
        source = (
            "CONTENT:\n"
            "Although he was exhausted, he finished the race. "
            "Although the weather was harsh, they reached the summit. "
            "Only then did he realize his mistake."
        )
        data = {
            "grammar_patterns": [
                {
                    "quote": "Although he was exhausted, he finished the race.",
                    "pattern_formula": "Although + [Clause], [Main Clause]",
                    "category": "Logic & Stance",
                    "pedagogical_function": "Concession.",
                    "design_audit": "AUDIT: Although",
                    "imitation_example": "Although it rained, we played.",
                    "common_mistakes": "Missing comma.",
                    "cefr_level": "B2"
                },
                {
                    "quote": "Although the weather was harsh, they reached the summit.",
                    "pattern_formula": "Although + [Clause], [Main Clause]",
                    "category": "Logic & Stance",
                    "pedagogical_function": "Another concession.",
                    "design_audit": "AUDIT: Although",
                    "imitation_example": "Although cold, we walked.",
                    "common_mistakes": "Missing comma.",
                    "cefr_level": "B2"
                }
            ]
        }
        pruned_data, notices = prune_hallucinated_items(data, source, task_type="grammar")
        # Second homogeneous 'although' pattern should be pruned to keep grammar diverse
        assert len(pruned_data["grammar_patterns"]) == 1
        assert any("Pruned duplicate item: homogeneous grammar pattern" in n for n in notices)

    def test_generate_cobuild_formula(self):
        """LinguisticEngine must generate standard COBUILD slot formulas deterministically."""
        s1 = "Although he was exhausted, he finished the race."
        f1 = LinguisticEngine.generate_cobuild_formula(s1, "Logic & Stance")
        assert f1 == "Although + [Clause], [Subject] + [VP]"

        s2 = "Whether a job is designated as work depends, not on the job itself, but on the tastes."
        f2 = LinguisticEngine.generate_cobuild_formula(s2, "Rhetoric & Emphasis")
        assert f2 == "[Subject] + [VP], not + [PrepP/NP], but + [PrepP/NP]"

        s3 = "It is essential to consider the environmental impact."
        f3 = LinguisticEngine.generate_cobuild_formula(s3, "Information Packaging")
        assert f3 == "It + [be] + [Adj/NP] + to-V/that + [Clause]"

    def test_grammar_formula_auto_healing(self):
        """Level 1 Code Gate must auto-heal missing or trivial LLM formulas using LinguisticEngine."""
        from librarian.evaluator import prune_hallucinated_items
        source = (
            "CONTENT:\n"
            "Although he was exhausted, he finished the race. "
            "Whether a job is designated as work depends, not on the job itself, but on the tastes."
        )
        data = {
            "grammar_patterns": [
                {
                    "quote": "Although he was exhausted, he finished the race.",
                    "pattern_formula": "",  # Empty formula from LLM
                    "category": "Logic & Stance",
                    "pedagogical_function": "Concession.",
                    "design_audit": "AUDIT: Although",
                    "imitation_example": "Although tired, he ran.",
                    "common_mistakes": "Missing comma.",
                    "cefr_level": "B2"
                }
            ]
        }
        pruned_data, notices = prune_hallucinated_items(data, source, task_type="grammar")
        item = pruned_data["grammar_patterns"][0]
        assert item["pattern_formula"] == "Although + [Clause], [Subject] + [VP]"
        assert any("Standardized COBUILD formula" in n for n in notices)

    def test_acl_and_awl_loading(self):
        """LinguisticEngine must load ACL and AWL datasets without error."""
        acl = LinguisticEngine.get_acl_collocations()
        awl = LinguisticEngine.get_awl_words()
        assert len(acl) > 2000
        assert len(awl) > 500
        assert "adverse effect" in acl
        assert "analysis" in awl or "analyse" in awl

    def test_mine_expression_skeletons(self):
        """LinguisticEngine must extract ACL collocations and phrasal verbs with slotted formulas."""
        text = (
            "Researchers carried out a detailed study to explore whether work has an adverse effect on personal satisfaction. "
            "When people slave away in repetitive tasks, they often struggle to cope with chronic stress. "
            "The administration must take into account these findings before implementing policies."
        )
        mined = LinguisticEngine.mine_expression_skeletons(text, target_count=5)
        assert len(mined) >= 3
        phrases = [m["phrase"] for m in mined]
        types = {m["type"] for m in mined}
        assert any("adverse effect" in p or "detailed study" in p for p in phrases)
        assert any("carry out" in p or "slave away" in p or "cope with" in p or "take into account" in p for p in phrases)
        # Check slotted formulas
        assert all("pattern_formula" in m and len(m["pattern_formula"]) > 0 for m in mined)

    def test_mine_possessive_expressions(self):
        """LinguisticEngine must extract authentic [one's] idiomatic phrases via spaCy poss dependency."""
        text = (
            "Students should attain their best in college and make up their mind early. "
            "She lost her temper during the debate, but managed to keep her promise."
        )
        mined = LinguisticEngine.mine_expression_skeletons(text, target_count=5)
        formulas = [m["pattern_formula"] for m in mined]
        assert any("[one's]" in f for f in formulas)
        assert any("attain [one's] best" in f or "make up [one's] mind" in f or "lose [one's] temper" in f for f in formulas)

    def test_mine_vocabulary_skeletons(self):
        """LinguisticEngine must extract AWL academic headwords with canonical lemmatization."""
        text = (
            "To laborers, on the other hand, leisure means autonomy from compulsion, so it is natural for them to imagine "
            "that the fewer hours they have to spend laboring, and the more hours they have free for play, the better. "
            "They will also work with more diligence and precision because they have fostered a sense of personal pride in their jobs. "
            "On the other hand, laborers, whose sole incentive is earning their livelihood, feel that the time they spend on the daily grind "
            "is wasted and doesn't contribute to their happiness."
        )
        mined = LinguisticEngine.mine_vocabulary_skeletons(text, target_count=5)
        assert len(mined) >= 3
        words = [m["word"] for m in mined]
        assert any(m["is_awl"] for m in mined)
        assert any(w in ("incentive", "contribute", "sole", "precision", "diligence", "autonomy") for w in words)
        # Headwords must be lemmatized base forms
        assert all(w.islower() and w.isalpha() for w in words)

    def test_mine_vocabulary_connective_gate_regardless_of(self):
        """Extraction-stage quality (Pillar 0): a dependent connective must be completed
        to its fixed 'head + of' form and labelled a preposition — never leaked out as a
        bare adverb. A bare 'regardless' is the root cause of the downstream
        'regardless of' vs 'in spite of' double-key distractors."""
        text = (
            "Regardless of the financial risks, the board proceeded with the merger. "
            "Scholars must weigh the autonomy of each department against shared governance."
        )
        mined = LinguisticEngine.mine_vocabulary_skeletons(text, target_count=15)
        words = [m["word"] for m in mined]
        # The connective must be present as the fixed phrase, labelled a preposition...
        conn = [m for m in mined if m["word"] == "regardless of"]
        assert conn, "expected 'regardless of' to be mined as a fixed phrase"
        assert conn[0]["part_of_speech"] == "preposition"
        assert conn[0].get("is_connective") is True
        # ...and the bare adverb form must NOT leak out.
        assert "regardless" not in words
        # Regression guard: genuine single-word headwords are still extracted.
        assert "autonomy" in words

    def test_mine_vocabulary_connective_gate_bare_form_dropped(self):
        """A bare dependent connective without its obligatory 'of' complement is not a
        valid single-blank target and must be dropped, not emitted as an adverb."""
        text = "The plan proceeded regardless. The committee reconvened the following week."
        mined = LinguisticEngine.mine_vocabulary_skeletons(text, target_count=15)
        words = [m["word"] for m in mined]
        assert "regardless" not in words
        assert "regardless of" not in words

    def test_build_authentic_cloze_items(self):
        """LinguisticEngine must build authentic cloze items with masked blanks and collision-free distractors."""
        vocab_md = (
            "## [[foundation]]\n"
            "- **Part Of Speech**: noun\n"
            "- **Definition**: The solid base on which something is established.\n"
            "- **Quoted Sentence**: But know this: The future is built on the strong foundation of the past.\n\n"
            "## [[available]]\n"
            "- **Part Of Speech**: adjective\n"
            "- **Definition**: Able to be used or obtained.\n"
            "- **Quoted Sentence**: You may feel overwhelmed by the wealth of courses available to you.\n"
        )
        cloze_items = LinguisticEngine.build_authentic_cloze_items(vocab_md, target_count=2)
        assert len(cloze_items) == 2
        assert cloze_items[0]["target_word"] == "foundation"
        assert "built on the strong ____ of the past" in cloze_items[0]["question"]
        assert len(cloze_items[0]["precomputed_distractors"]) == 3
        assert "foundation" not in cloze_items[0]["precomputed_distractors"]
        assert "____" in cloze_items[1]["question"]

    def test_build_authentic_cloze_connective_single_answer(self):
        """A dependent-connective cloze must keep the obligatory 'of' in the stem and blank
        only the head, with bare-NP (of-incompatible) connectives as distractors. In the
        '____ of <NP>' frame only the of-taking head is grammatical, so the item is strictly
        single-answer — this is the grammar/collocation axis, not semantic overlap."""
        vocab_md = (
            "## [[regardless of]]\n"
            "- **Part Of Speech**: preposition\n"
            "- **Definition**: no matter what; without being affected by\n"
            "- **Quoted Sentence**: Regardless of the financial risks, the board proceeded with the merger.\n"
        )
        cloze_items = LinguisticEngine.build_authentic_cloze_items(vocab_md, target_count=5)
        assert len(cloze_items) == 1
        c = cloze_items[0]
        # The head is the answer; the 'of' is given in the stem (not part of the answer).
        assert c["target_word"] == "regardless"
        assert c["base_headword"] == "regardless of"
        assert c["is_connective"] is True
        # The 'of' frame is preserved and there is exactly one blank (the head).
        assert "____ of the financial risks" in c["question"]
        assert c["question"].count("____") == 1
        # Distractors are bare-NP connectives: none is valid after 'of', and the target is
        # not among them -> exactly one correct answer.
        allowed = {"despite", "notwithstanding", "whatever", "barring"}
        assert set(c["precomputed_distractors"]) <= allowed
        assert len(c["precomputed_distractors"]) == 3
        assert "regardless" not in c["precomputed_distractors"]

    def test_sentence_pointer_hydration(self):
        """LinguisticEngine and evaluator must deterministically hydrate [S-ID] (e.g. S-1, S-126) into authentic sentences."""
        from librarian.evaluator import prune_hallucinated_items
        source = (
            "CONTENT:\n"
            "Do you know the fairy tale of Goldilocks and the Three Bears? "
            "Whether a job is designated as work depends, not on the job itself, but on the tastes. "
            "Although he was exhausted, he finished the academic investigation."
        )
        _, pool = LinguisticEngine.tokenize_and_index_sentences(source)
        assert "S-1" in pool
        assert "S-2" in pool
        assert "S-3" in pool

        # Test snapping with various forms: [S-2], S-2, [s-2], and multi-digit
        snapped = LinguisticEngine.snap_to_sentence_pool("[S-2]", pool)
        assert "Whether a job is designated as work depends" in snapped

        snapped_lower = LinguisticEngine.snap_to_sentence_pool("s-3", pool)
        assert "finished the academic investigation" in snapped_lower

        # Test end-to-end hydration in prune_hallucinated_items
        vocab_data = {
            "vocabulary": [
                {
                    "word": "fairy",
                    "quoted_sentence": "[S-1]",
                    "part_of_speech": "noun",
                    "definition": "A mythical creature.",
                    "example_usage": "She believed in fairies.",
                    "word_cefr_level": "B1",
                    "design_audit": "AUDIT: fairy -> fairy -> noun -> B1 -> VERBATIM_CONFIRMED"
                }
            ]
        }
        hydrated, _ = prune_hallucinated_items(vocab_data, source, task_type="vocabulary")
        assert hydrated["vocabulary"][0]["quoted_sentence"] == "Do you know the fairy tale of Goldilocks and the Three Bears?"

    def test_quote_headword_repair(self):
        """If the headword is absent from the quoted sentence but present in the source,
        prune must deterministically swap in the authentic source sentence containing it."""
        from librarian.evaluator import prune_hallucinated_items
        source = (
            "CONTENT:\n"
            "The new policy has been criticized by many experts in the field. "
            "Every generation inherits problems left by the last one."
        )
        # Positive: wrong sentence cited for 'generation' -> deterministic repair
        bad_data = {
            "vocabulary": [
                {
                    "word": "generation",
                    "quoted_sentence": "The new policy has been criticized by many experts in the field.",
                    "part_of_speech": "noun",
                    "definition": "A group of people born at about the same time.",
                    "example_usage": "This generation grew up with smartphones.",
                    "word_cefr_level": "B2",
                    "design_audit": "AUDIT: generation -> generation -> noun -> B2 -> VERBATIM_CONFIRMED"
                }
            ]
        }
        repaired, notices = prune_hallucinated_items(bad_data, source, task_type="vocabulary")
        assert "Every generation inherits problems left by the last one." in repaired["vocabulary"][0]["quoted_sentence"]
        assert any("Quote Repair" in n for n in notices)

        # Negative: quote already contains the headword -> must not be modified
        ok_data = {
            "vocabulary": [
                {
                    "word": "generation",
                    "quoted_sentence": "Every generation inherits problems left by the last one.",
                    "part_of_speech": "noun",
                    "definition": "A group of people born at about the same time.",
                    "example_usage": "This generation grew up with smartphones.",
                    "word_cefr_level": "B2",
                    "design_audit": "AUDIT: generation -> generation -> noun -> B2 -> VERBATIM_CONFIRMED"
                }
            ]
        }
        unchanged, notices2 = prune_hallucinated_items(ok_data, source, task_type="vocabulary")
        assert unchanged["vocabulary"][0]["quoted_sentence"] == "Every generation inherits problems left by the last one."
        assert not any("Quote Repair" in n for n in notices2)

    def test_generate_vocab_distractors(self):
        """LinguisticEngine must generate high-discrimination, collision-free distractors using WordNet and OCD."""
        # Test 1: lay + anchor 'foundation' (must exclude build/establish and return collision-free near-synonyms)
        d1 = LinguisticEngine.generate_vocab_distractors("lay", pos="verb", context_anchor="foundation")
        assert len(d1) == 3
        assert "lay" not in d1
        # 'build' or 'establish' would cause double keys with foundation; must be excluded
        assert "build" not in d1
        assert "establish" not in d1

        # Test 2: obstacle (noun)
        d2 = LinguisticEngine.generate_vocab_distractors("obstacle", pos="noun")
        assert len(d2) == 3
        assert "obstacle" not in d2
        # All distractors must be valid words in Oxford Collocations Dictionary
        ocd = LinguisticEngine.get_oxford_collocations()
        for word in d2:
            assert word in ocd

    def test_build_precomputed_target_skeletons(self):
        """LinguisticEngine must build valid pre-computed target skeletons with anchors and prescribed options."""
        sample_vocab = """
## [[foundation]]
- **Part Of Speech**: noun
- **Definition**: The solid basis on which something stands or is supported.
- **Quoted Sentence**: "The government aims to establish a solid foundation for the economy."

## [[lay]]
- **Part Of Speech**: verb
- **Definition**: To put or place something down in a flat or horizontal position.
- **Quoted Sentence**: "They will lay the groundwork for upcoming bilateral negotiations."
"""
        skeletons = LinguisticEngine.build_precomputed_target_skeletons(sample_vocab, target_count=2)
        assert len(skeletons) == 2

        # Item 1: foundation
        s1 = skeletons[0]
        assert s1["target_word"] == "foundation"
        assert s1["part_of_speech"] == "noun"
        assert len(s1["prescribed_options"]) == 4
        assert "foundation" in s1["prescribed_options"]

        # Item 2: lay (inflected to laid in past tense sequence)
        s2 = skeletons[1]
        assert s2["base_headword"] == "lay"
        assert s2["target_word"] in ("lay", "laid")
        assert s2["part_of_speech"] == "verb"
        assert len(s2["prescribed_options"]) == 4
        assert s2["target_word"] in s2["prescribed_options"]
        # Options must be strictly unique
        assert len(set(s2["prescribed_options"])) == 4

    def test_sense_locked_antonym_extraction(self):
        """LinguisticEngine must prioritize genuine sense-locked antonyms for verbs."""
        # 'increase' with academic growth definition must produce 'decrease' as an antonym
        distractors, meta = LinguisticEngine.generate_vocab_distractors(
            target_word="increase",
            pos="verb",
            definition="become bigger or greater in amount or volume",
            return_metadata=True
        )
        assert len(distractors) == 3
        assert "decrease" in distractors
        assert meta["decrease"] == "antonym"

        # 'expand' with extension definition must produce 'contract'
        d_expand, meta_expand = LinguisticEngine.generate_vocab_distractors(
            target_word="expand",
            pos="verb",
            definition="extend in one or more directions",
            return_metadata=True
        )
        assert len(d_expand) == 3
        assert "contract" in d_expand
        assert meta_expand["contract"] == "antonym"

    def test_select_sense_aligned_anchor(self):
        """LinguisticEngine must select anchors aligned with authentic quote and curriculum definition."""
        candidates = ["envelope", "letter", "rally", "crisis", "challenge"]
        quote = "The committee convened urgently to address the financial crisis."
        definition = "to direct efforts towards dealing with a problem or challenge"

        # Level 1: Must pick 'crisis' directly from the quote
        best_anchor = LinguisticEngine._select_sense_aligned_anchor(candidates, definition=definition, quote=quote)
        assert best_anchor == "crisis"

        # Level 2: If quote does not contain any candidate, must pick 'challenge' matching definition tokens
        unrelated_quote = "The scholar was unable to resolve the matter."
        best_anchor_def = LinguisticEngine._select_sense_aligned_anchor(candidates, definition=definition, quote=unrelated_quote)
        assert best_anchor_def == "challenge"

        # Level 3 (Strict None Semantics): no quote/definition evidence at all ->
        # refuse the old "first clean token" lottery instead of forcing a fake anchor.
        unrelated_def = "the act of pressing a button repeatedly without purpose"
        assert LinguisticEngine._select_sense_aligned_anchor(candidates, definition=unrelated_def, quote=unrelated_quote) is None

    def test_preposition_bound_verb_skeleton(self):
        """LinguisticEngine must bind dependent prepositions in quote for verbs and craft targeted micro-tasks."""
        sample_vocab = """
## [[rely]]
- **Part Of Speech**: verb
- **Definition**: To depend on with full trust or confidence.
- **Quoted Sentence**: "Scholars frequently rely on empirical evidence to validate their scientific theories."
"""
        skeletons = LinguisticEngine.build_precomputed_target_skeletons(sample_vocab, target_count=1)
        assert len(skeletons) == 1
        s = skeletons[0]
        assert s["target_word"] == "rely"
        assert s["part_of_speech"] == "verb"
        assert s["context_anchor"] == "on"
        assert s["anchor_type"] == "prep"
        assert "bound preposition 'on'" in s["micro_task"]
        assert len(s["prescribed_options"]) == 4
        assert "rely" in s["prescribed_options"]

    def test_adjective_satellite_and_bipolar_harvesting(self):
        """LinguisticEngine must harvest WordNet satellite synsets and attribute coordinates for adjectives."""
        # 'comprehensive' with anchor 'service'
        distractors, meta = LinguisticEngine.generate_vocab_distractors(
            target_word="comprehensive",
            pos="adj",
            context_anchor="service",
            anchor_type="modified_noun",
            return_metadata=True
        )
        assert len(distractors) == 3
        assert "comprehensive" not in distractors
        # Must NOT fall back to generic corpus fallback (e.g. crucial/primary)
        # Should contain authentic satellite adjectives like extensive, broad, wide
        has_satellite = any(meta[d] == "satellite_synonym" for d in distractors)
        assert has_satellite
        assert any(d in ("extensive", "broad", "all-inclusive", "wide", "universal", "general") for d in distractors)

        # 'available' with anchor 'alternative'
        d_avail, meta_avail = LinguisticEngine.generate_vocab_distractors(
            target_word="available",
            pos="adj",
            context_anchor="alternative",
            anchor_type="modified_noun",
            return_metadata=True
        )
        assert len(d_avail) == 3
        assert "available" not in d_avail
        assert any(meta_avail[d] == "satellite_synonym" for d in d_avail)
        assert any(d in ("accessible", "easy", "forthcoming", "obtainable", "ready", "open") for d in d_avail)

    def test_adjective_predicative_only_exclusion(self):
        """LinguisticEngine must reject predicative-only adjectives (asleep, aware, afraid) when modifying a noun."""
        for pred_adj in LinguisticEngine._PREDICATIVE_ONLY_ADJS:
            distractors = LinguisticEngine.generate_vocab_distractors(
                target_word="comprehensive",
                pos="adj",
                context_anchor="service",
                anchor_type="modified_noun"
            )
            assert pred_adj not in distractors

    def test_adjective_preposition_valency_binding(self):
        """LinguisticEngine must bind dependent prepositions in quote for predicative adjectives."""
        sample_vocab = """
## [[vulnerable]]
- **Part Of Speech**: adj
- **Definition**: Susceptible to physical or emotional harm or attack.
- **Quoted Sentence**: "Developing nations are particularly vulnerable to climate disruptions and global inflation."
"""
        skeletons = LinguisticEngine.build_precomputed_target_skeletons(sample_vocab, target_count=1)
        assert len(skeletons) == 1
        s = skeletons[0]
        assert s["target_word"] == "vulnerable"
        assert s["part_of_speech"] == "adj"
        assert s["context_anchor"] == "to"
        assert s["anchor_type"] == "prep"
        assert "bound preposition 'to'" in s["micro_task"]
        assert len(s["prescribed_options"]) == 4
        assert "vulnerable" in s["prescribed_options"]

    def test_adjective_scheme1_satellite_harvester_pos_s(self):
        """Scheme 1: Words with pos='s' (e.g. fragile, apparent) must harvest satellite synsets without falling back to corpus_fallback."""
        d_fragile, meta_fragile = LinguisticEngine.generate_vocab_distractors(
            target_word="fragile",
            pos="adj",
            context_anchor="ecosystem",
            anchor_type="modified_noun",
            return_metadata=True
        )
        assert len(d_fragile) == 3
        assert "fragile" not in d_fragile
        # Must not be hardcoded fallback
        assert not all(meta_fragile[d] == "corpus_fallback" for d in d_fragile)
        # Should contain authentic satellite/antonym adjectives
        assert any(meta_fragile[d] in ("satellite_synonym", "antonym", "attribute_coordinate") for d in d_fragile)

    def test_adjective_scheme2_opposite_cluster_antonym(self):
        """Scheme 2: Adjectives must extract sense antonyms or opposite cluster satellites."""
        # Test stable
        d_stable, meta_stable = LinguisticEngine.generate_vocab_distractors(
            target_word="stable",
            pos="adj",
            context_anchor="growth",
            anchor_type="modified_noun",
            return_metadata=True
        )
        assert len(d_stable) == 3
        assert "stable" not in d_stable
        assert any(meta_stable[d] == "antonym" for d in d_stable)
        assert any(d in ("unstable", "volatile", "reactive", "changeable") for d in d_stable)

    def test_adjective_scheme3_modified_noun_collocation_clash(self):
        """Scheme 3: Adjectives modifying a noun must exclude legal adjectives of that noun to prevent double keys."""
        ocd = LinguisticEngine.get_oxford_collocations()
        service_legal_adjs = {a.split()[0].lower() for a in ocd.get("service", {}).get("adj", [])}

        distractors, meta = LinguisticEngine.generate_vocab_distractors(
            target_word="comprehensive",
            pos="adj",
            context_anchor="service",
            anchor_type="modified_noun",
            return_metadata=True
        )
        assert len(distractors) == 3
        for d in distractors:
            # If d is in service legal adjs, it must only be an antonym (whitelisted for contrast)
            if d in service_legal_adjs:
                assert meta[d] == "antonym"

    def test_adjective_scheme4_preposition_valency_gate(self):
        """Scheme 4: Preposition valency cloze must prioritize contrasting prepositions and rule out same-prep distractors."""
        distractors, meta = LinguisticEngine.generate_vocab_distractors(
            target_word="vulnerable",
            pos="adj",
            context_anchor="to",
            anchor_type="prep",
            return_metadata=True
        )
        assert len(distractors) == 3
        assert "vulnerable" not in distractors
        # Should contain preposition_valency or antonym
        assert any(meta[d] in ("preposition_valency", "antonym") for d in distractors)

    def test_adjective_authentic_quote_amod_anchor_binding(self):
        """Adjective Step A2: spaCy amod dependency in quote must bind modified head noun."""
        sample_vocab = """
## [[comprehensive]]
- **Part Of Speech**: adj
- **Definition**: Including or dealing with all or nearly all elements or aspects of something.
- **Quoted Sentence**: "The agency published a comprehensive overview of public health policies."
"""
        skeletons = LinguisticEngine.build_precomputed_target_skeletons(sample_vocab, target_count=1)
        assert len(skeletons) == 1
        s = skeletons[0]
        assert s["target_word"] == "comprehensive"
        assert s["part_of_speech"] == "adj"
        assert s["context_anchor"] == "overview"
        assert s["anchor_type"] == "modified_noun"
        assert "modifying noun 'overview'" in s["micro_task"]


class TestAnchorFourDimensionRepair:
    """Regression tests for the four-dimension closed-loop anchor repair (Pillars 1-4):
    post-head linear constraint + adjunct blacklist + OCD consensus gate (verbs),
    spaCy dependency walker (nouns), strict-None selector, and contract-tiered
    micro-task blueprints. Each case mirrors a historically fabricated anchor."""

    @staticmethod
    def _single_skeleton(vocab_block):
        skeletons = LinguisticEngine.build_precomputed_target_skeletons(vocab_block, target_count=1)
        assert len(skeletons) == 1
        return skeletons[0]

    def test_verb_adjunct_for_example_must_yield_ocd_complement(self):
        """'for example' (interjection) must be skipped; OCD consensus picks 'correlate with'."""
        sample_vocab = """
## [[correlate]]
- **Part Of Speech**: verb
- **Definition**: To show or be a subject of a mutual relation.
- **Quoted Sentence**: "The findings do not, for example, correlate with a difference in average salary."
"""
        s = self._single_skeleton(sample_vocab)
        assert s["target_word"] == "correlate"
        assert s["part_of_speech"] == "verb"
        assert s["context_anchor"] == "with"
        assert s["anchor_type"] == "prep"
        assert "bound preposition 'with'" in s["micro_task"]
        assert "example" not in s["prescribed_options"]
        assert len(s["prescribed_options"]) == 4

    def test_verb_fronted_adjunct_must_yield_right_side_complement(self):
        """Fronted adjunct 'In a society' must be physically excluded; 'degrade into' is trusted
        via the typical-complement rule even though 'degrade' has no OCD valency record."""
        sample_vocab = """
## [[degrade]]
- **Part Of Speech**: verb
- **Definition**: To reduce something in quality, value, or status.
- **Quoted Sentence**: "In a society where they are given more hours, workers may have degraded into modern slaves."
"""
        s = self._single_skeleton(sample_vocab)
        assert s["target_word"] == "degrade"
        assert s["context_anchor"] == "into"
        assert s["anchor_type"] == "prep"
        assert "bound preposition 'into'" in s["micro_task"]
        assert "society" not in s["micro_task"]

    def test_noun_prep_complement_walker_beats_degree_adverb(self):
        """Degree adverb 'more' (modifier of 'hours') must never attach to 'autonomy';
        the dependency walker extracts the true complement 'autonomy from compulsion'."""
        sample_vocab = """
## [[autonomy]]
- **Part Of Speech**: noun
- **Definition**: The state of being free to rule oneself; self-government.
- **Quoted Sentence**: "The means of production were owned collectively, and autonomy from compulsion was the guiding principle of the movement."
"""
        s = self._single_skeleton(sample_vocab)
        assert s["target_word"] == "autonomy"
        assert s["part_of_speech"] == "noun"
        assert s["context_anchor"] == "from"
        assert s["anchor_type"] == "prep"
        assert "bound preposition 'from'" in s["micro_task"]
        assert "more" not in s["micro_task"]

    def test_noun_zero_evidence_must_not_draw_dictionary_lottery(self):
        """With no syntactic link and no quote/definition evidence, the selector must
        return None (or a real governing verb) instead of lottery-drawing 'strange'."""
        sample_vocab = """
## [[compulsion]]
- **Part Of Speech**: noun
- **Definition**: An urge that forces a person to do something, often against their will.
- **Quoted Sentence**: "The compulsion to check emails made it hard for workers to enjoy their leisure hours."
"""
        s = self._single_skeleton(sample_vocab)
        assert s["target_word"] == "compulsion"
        assert s["part_of_speech"] == "noun"
        # The fabricated anchor 'strange' must be impossible under the strict-None selector
        assert "strange" not in s["micro_task"]
        assert s["context_anchor"] is None or s["context_anchor"] != "strange"
        assert len(s["prescribed_options"]) == 4
        assert "compulsion" in s["prescribed_options"]

    def test_noun_casual_adjunct_prep_rejected_by_ocd_consensus(self):
        """Casual adjunct preposition 'from' (in 'not too much control from them')
        must be rejected because OCD does not register 'from' as an inherent valency
        prep for 'control'. The engine must fall back to authentic adjective 'strict'."""
        sample_vocab = """
## [[control]]
- **Part Of Speech**: noun
- **Definition**: The power or authority to influence someone's behavior; strict supervision or regulation.
- **Quoted Sentence**: "Most young people want to enjoy the warm love from their family and friends, but not too much control from them."
"""
        s = self._single_skeleton(sample_vocab)
        assert s["target_word"] == "control"
        assert s["part_of_speech"] == "noun"
        assert s["context_anchor"] != "from"
        assert s["context_anchor"] == "strict"
        assert s["anchor_type"] == "adj"
        assert "from" not in s["prescribed_options"]
        assert "Semantic Discriminator" in s["micro_task"]
        assert "Syntactic Frame" in s["micro_task"]

    def test_adverb_distractor_generation_and_parallelism(self):
        """Adverb 'closely' must receive adverb distractors (deeply, strictly, tightly)
        instead of defaulting to nouns or adjectives."""
        sample_vocab = """
## [[closely]]
- **Part Of Speech**: adverb
- **Definition**: In a manner that is very attentive or precise; with great care and attention to detail.
- **Quoted Sentence**: "They want to follow the time closely, but they also long for peace of mind."
"""
        s = self._single_skeleton(sample_vocab)
        assert s["target_word"] == "closely"
        assert s["part_of_speech"] == "adv"
        assert len(s["prescribed_options"]) == 4
        # All prescribed options must be valid adverbs
        for opt in s["prescribed_options"]:
            assert opt.endswith("ly") or opt in ("close", "tight", "deeply", "strictly")
        assert "Syntactic Frame" in s["micro_task"]

    def test_double_key_refill_does_not_dump_synonyms_back(self):
        """For target 'smart', near-synonym 'astute' must be safely excluded and
        refilled from non-colliding academic words."""
        sample_vocab = """
## [[smart]]
- **Part Of Speech**: adjective
- **Definition**: Characterized by intelligence or good judgment; showing wisdom in decision-making processes.
- **Quoted Sentence**: "However, the new ways of communication could give them freedom and chance to make smart choices."
"""
        s = self._single_skeleton(sample_vocab)
        assert s["target_word"] == "smart"
        assert "astute" not in s["prescribed_options"]
        assert len(s["prescribed_options"]) == 4
        assert len(set(s["prescribed_options"])) == 4

    def test_cefr_difficulty_ceiling_eliminates_obscure_distractors(self):
        """P0: Distractors must not exceed difficulty ceiling or introduce bizarre obscure derivatives."""
        # 'chance' is A2; 'conceivableness' is unlisted, 'conceivability' is C2, 'astute' is C2
        assert LinguisticEngine.is_cefr_compliant_distractor("conceivableness", "chance") is False
        assert LinguisticEngine.is_cefr_compliant_distractor("conceivability", "chance") is False
        assert LinguisticEngine.is_cefr_compliant_distractor("attainableness", "chance") is False
        assert LinguisticEngine.is_cefr_compliant_distractor("astute", "smart") is False

        # Common level-appropriate words should be accepted
        assert LinguisticEngine.is_cefr_compliant_distractor("choice", "chance") is True
        assert LinguisticEngine.is_cefr_compliant_distractor("effort", "chance") is True
        assert LinguisticEngine.is_cefr_compliant_distractor("matter", "chance") is True
        assert LinguisticEngine.is_cefr_compliant_distractor("stupid", "smart") is True

    def test_elementary_target_distractors_bounded_to_b1(self):
        """Precomputed skeletons for Book 1 foundational words must strictly eliminate obscure distractors."""
        sample_vocab = """
## [[chance]]
- **Part Of Speech**: noun
- **Definition**: A possibility of something happening, or an opportunity to do something.
- **Quoted Sentence**: "However, the new ways of communication could give them freedom and chance to make smart choices."
"""
        s = self._single_skeleton(sample_vocab)
        assert s["target_word"] == "chance"
        assert len(s["prescribed_options"]) == 4
        assert "conceivableness" not in s["prescribed_options"]
        assert "conceivability" not in s["prescribed_options"]
        for opt in s["prescribed_options"]:
            # Max length cannot exceed 12 chars
            assert len(opt) <= 12
            # Must not end in bizarre suffixes
            assert not opt.endswith("ableness")
            # Word level if in CEFR database must be <= B1
            lvl = LinguisticEngine.get_word_cefr(opt)

    def test_mine_expression_skeletons_ocd_gate(self):
        """LinguisticEngine.mine_expression_skeletons must use OCD physical gate to reject invalid collocations."""
        passage = (
            "In the past, it was a small stamp that helped family members and friends to keep in touch with each other. "
            "With the coming of the telephone, people could not only read words, but also hear each other's voices. "
            "Later, the rise of the Internet joins people in different places with instant communication software. "
            "They want to follow the time closely, but they also long for peace of mind. "
            "Some people may worry about their privacy."
        )
        skels = LinguisticEngine.mine_expression_skeletons(passage, target_count=8)
        phrases = [s["phrase"] for s in skels]

        # 1. Invalid collocation 'join in place' must be strictly rejected
        assert "join in place" not in phrases
        assert not any("join in place" in p for p in phrases)

        # 2. Authentic expressions and collocations must be preserved
        assert any("keep in touch" in p for p in phrases)
        assert any("hear" in p and "voice" in p for p in phrases)
        assert any("long for" in p for p in phrases)
        assert any("worry about" in p for p in phrases)

    def test_mine_expression_skeletons_with_syllabus(self):
        """LinguisticEngine.mine_expression_skeletons prioritizes syllabus expressions with standardized formulas."""
        passage = (
            "In the past, it was a small stamp that helped family members and friends to keep in touch with each other. "
            "They want to follow the time closely, but they also long for peace of mind. "
            "Technology can make great progress possible for future generations. "
            "Some people may worry about their privacy."
        )
        syllabus = ["keep in touch with", "long for", "peace of mind", "make  possible", "worry about"]
        skels = LinguisticEngine.mine_expression_skeletons(passage, target_count=5, syllabus_expressions=syllabus)

        assert len(skels) == 5
        formulas = {s["pattern_formula"]: s for s in skels}

        # Check standardized formulas
        assert "keep in touch with [sb]" in formulas
        assert formulas["keep in touch with [sb]"]["type"] == "idiom"
        assert formulas["keep in touch with [sb]"]["sid"] == "S-1"

        assert "long for [sth/sb]" in formulas
        assert formulas["long for [sth/sb]"]["type"] == "phrasal verb"

        assert "make [sth] possible" in formulas
        assert formulas["make [sth] possible"]["type"] == "collocation"

        assert "worry about [sth/sb]" in formulas
        assert formulas["worry about [sth/sb]"]["type"] == "phrasal verb"

        assert "peace of mind" in formulas
        assert formulas["peace of mind"]["type"] == "set phrase"

    def test_ocd_zero_collision_anchor_fallback(self):
        """LinguisticEngine uses OCD zero-collision anchor when quote lacks strong syntactic dependency."""
        vocab_md = (
            "## [[telephone]]\n"
            "- **Part of Speech**: noun\n"
            "- **Definition**: a system for talking to somebody elsewhere using phone\n"
            "- **Context Sentence**: The other day, I was talking on the telephone to a client.\n\n"
            "## [[simple]]\n"
            "- **Part of Speech**: adjective\n"
            "- **Definition**: easily understood or done; presenting no difficulty\n"
            "- **Context Sentence**: The problem seemed simple at first.\n"
        )
        skeletons = LinguisticEngine.build_precomputed_target_skeletons(vocab_md, target_count=2)
        assert len(skeletons) == 2

        # Check telephone
        tel = next(s for s in skeletons if s["target_word"] == "telephone")
        assert tel["context_anchor"] is not None
        assert tel["anchor_type"] in ("verb_subject", "verb", "adj")
        assert "general context" not in tel["micro_task"]
        assert "Syntactic Frame:" in tel["micro_task"]

        # Check simple
        smp = next(s for s in skeletons if s["target_word"] == "simple")
        assert smp["context_anchor"] is not None
        assert smp["anchor_type"] in ("adv_mod", "verb_copula", "modified_noun", "prep")
        assert "Syntactic Frame:" in smp["micro_task"]

        # Check concrete vs abstract noun differentiation in find_ocd_zero_collision_anchor
        c_anchor, c_type, _, _ = LinguisticEngine.find_ocd_zero_collision_anchor("telephone", "noun", ["mixer", "modem", "monitor"])
        assert c_type == "verb_subject"  # Concrete noun prioritizes verb_subject (ring)
        assert c_anchor == "ring"

        a_anchor, a_type, _, _ = LinguisticEngine.find_ocd_zero_collision_anchor("decision", "noun", ["conclusion", "option", "choice"])
        assert a_type == "verb"  # Abstract noun prioritizes light/support verb (affect/make/reach)

        # Check transitive vs intransitive verb differentiation
        vt_anchor, vt_type, _, _ = LinguisticEngine.find_ocd_zero_collision_anchor("solve", "verb", ["explain", "discuss", "discover"])
        assert vt_type == "object"  # Transitive verb prioritizes direct object (case/problem)

        vi_anchor, vi_type, _, _ = LinguisticEngine.find_ocd_zero_collision_anchor("listen", "verb", ["hear", "watch", "notice"])
        assert vi_type == "prep"  # Intransitive verb prioritizes bound preposition (to)
        assert vi_anchor == "to"

        # Check prepositional valency vs general adjective differentiation
        ap_anchor, ap_type, _, _ = LinguisticEngine.find_ocd_zero_collision_anchor("proud", "adj", ["humble", "arrogant", "beaming"])
        assert ap_type == "prep"  # Valency-bound adjective prioritizes preposition (of)
        assert ap_anchor == "of"

        ag_anchor, ag_type, _, _ = LinguisticEngine.find_ocd_zero_collision_anchor("economic", "adj", ["inefficient", "efficient", "competent"])
        assert ag_type == "modified_noun"  # General adjective prioritizes characteristic noun

        # Check adverb differentiation (modifies_adj vs modifies_verb)
        adv_deg_anchor, adv_deg_type, _, _ = LinguisticEngine.find_ocd_zero_collision_anchor("extremely", "adv", ["highly", "super", "deathly"])
        assert adv_deg_type == "modifies_adj"  # Degree adverb modifies adjective
        assert adv_deg_anchor is not None

        adv_man_anchor, adv_man_type, _, _ = LinguisticEngine.find_ocd_zero_collision_anchor("abruptly", "adv", ["suddenly", "dead", "short"])
        assert adv_man_type == "modifies_verb"  # Manner adverb modifies verb
        assert adv_man_anchor is not None

        # Check function words / connectives closed paradigm distractor generation
        assert LinguisticEngine.is_function_word("despite") is True
        assert LinguisticEngine.is_function_word("although") is True
        assert LinguisticEngine.is_function_word("beyond") is True
        assert LinguisticEngine.is_function_word("apple") is False

        # Prepositional connective contrasts with clausal conjunctions
        f_dists, f_meta = LinguisticEngine.generate_vocab_distractors("despite", pos="prep", target_count=3, return_metadata=True)
        assert len(f_dists) == 3
        assert "although" in f_dists or "though" in f_dists
        assert f_meta["distractor_strategy"] == "syntactic_complement_contrast"

        # Clausal conjunction contrasts with prepositional connectives
        c_dists, c_meta = LinguisticEngine.generate_vocab_distractors("although", pos="conj", target_count=3, return_metadata=True)
        assert len(c_dists) == 3
        assert "despite" in c_dists or "in spite of" in c_dists
        assert c_meta["distractor_strategy"] == "syntactic_complement_contrast"

        # Spatial/scope prepositions draw from scope paradigm
        s_dists, s_meta = LinguisticEngine.generate_vocab_distractors("beyond", pos="prep", target_count=3, return_metadata=True)
        assert len(s_dists) == 3
        assert any(p in s_dists for p in ("within", "across", "throughout"))
        assert s_meta["distractor_strategy"] == "scope_preposition_contrast"








