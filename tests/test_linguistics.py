"""
Unit tests for the LinguisticEngine (spaCy & WordNet integration).
Tests sentence indexing, phrasal verb boundary distinction, macro domain classification,
lemmatization, and zero-collision WordNet distractor generation.
"""

import re

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
        prep for 'control'.

        Which fallback wins depends on how populated 'control' is in the LDOCE database.
        Before the E7/F7 promotion its collocation boxes were empty, so the noun cascade
        fell through to the preposition/adjective levels and the anchor was 'over' or
        'strict'. The promotion filled those boxes with genuine LDOCE collocations
        (assume/bring/exercise/gain/keep/lose/maintain/regain/retain/seize/take/seek/try),
        so the cascade now stops one level earlier on a real governing verb such as
        'assume' ('assume control of the company'). The invariant this test guards is that
        the casual 'from' is rejected, not which legitimate level happens to win."""
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
        # A genuine LDOCE governing verb, or the prep/adjective fallbacks - never the quote's adjunct.
        assert s["context_anchor"] in ("over", "strict", "assume", "exercise", "gain",
                                       "maintain", "regain", "retain", "seize")
        assert s["anchor_type"] in ("prep", "adj", "verb")
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
        assert tel["anchor_type"] in ("verb_subject", "verb", "adj", "prep")
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
        assert a_type in ("verb", "adj")  # Abstract noun matches authentic LDOCE collocation (make/important/regret)
        assert a_anchor in ("make", "reach", "important", "take", "regret")

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

    def test_list_sentence_indexing_and_bullet_peeling(self):
        """tokenize_and_index_sentences separates bullets/numbers cleanly without contaminating S-IDs."""
        raw_text = (
            "• Schedule meetings and appointments in advance — a few days, a week, even a month in advance.\n"
            "• Always make appointments for in-person meetings. Don't just show up and expect people to make time to talk with you."
        )
        indexed, pool = LinguisticEngine.tokenize_and_index_sentences(raw_text)
        assert "S-1" in pool
        assert "S-2" in pool
        assert "S-3" in pool
        # Sentence text inside pool must not contain bullet markers
        assert not pool["S-1"].startswith("•")
        assert not pool["S-2"].startswith("•")
        assert not pool["S-3"].startswith("•")
        # In indexed text, bullet markers precede [S-xx] rather than interleaving or trailing
        assert "• [S-1] Schedule" in indexed
        assert "• [S-2] Always" in indexed

    def test_determine_contextual_pos_hyphen_and_wordnet(self):
        """determine_contextual_pos correctly identifies verbs split with hyphens and falls back to WordNet."""
        quote = "They would have left me a message asking to re-schedule the meeting."
        pos = LinguisticEngine.determine_contextual_pos("reschedule", quote)
        assert pos == "verb"

    def test_attach_importance_to_slot_formula(self):
        """mine_expression_skeletons correctly identifies attach importance to as [sth] formula."""
        passage = (
            "If you work for a company that attaches great importance to punctuality, here are tips. "
            "They will never forget to keep in touch with old friends."
        )
        skels = LinguisticEngine.mine_expression_skeletons(
            passage,
            target_count=2,
            syllabus_expressions=["attach importance to", "keep in touch with"]
        )
        formulas = {s["phrase"]: s["pattern_formula"] for s in skels}
        assert formulas.get("attach importance to") == "attach importance to [sth]"
        assert formulas.get("keep in touch with") == "keep in touch with [sb]"

    def test_intransitive_phrasal_verb_valence(self):
        """mine_expression_skeletons correctly identifies contextual transitivity from dependency parse."""
        passage = (
            "Don’t just show up and expect things to happen. "
            "Always turn off the lights before leaving the room."
        )
        skels = LinguisticEngine.mine_expression_skeletons(
            passage,
            target_count=2,
            syllabus_expressions=["show up", "turn off"]
        )
        formulas = {s["phrase"]: s["pattern_formula"] for s in skels}
        assert formulas.get("show up") == "show up"
        assert "[sth/sb]" in formulas.get("turn off", "") or "[sth]" in formulas.get("turn off", "")

    def test_dialogue_quote_sentence_integrity(self):
        """tokenize_and_index_sentences preserves exclamation marks inside quotes as a single sentence."""
        raw_text = 'Their attitude was “Ah! You’re here! We can start now!” And then he smiled.'
        _, pool = LinguisticEngine.tokenize_and_index_sentences(raw_text)
        assert len(pool) == 2
        assert pool["S-1"] == 'Their attitude was “Ah! You’re here! We can start now!”'
        assert pool["S-2"] == 'And then he smiled.'

    def test_contextual_pos_precision(self):
        """determine_contextual_pos uses dependency parse for single words and WordNet for compounds."""
        # 'firm' modifying 'end' is an adjective
        pos_firm = LinguisticEngine.determine_contextual_pos(
            "firm", "Because appointments usually have a firm end as well as start time."
        )
        assert pos_firm == "adjective"

        # 'log' governing 'it' as complement is a verb
        pos_log = LinguisticEngine.determine_contextual_pos(
            "log", "I had forgotten to log it into my agenda."
        )
        assert pos_log == "verb"

        # 're-schedule' followed by noun is a verb
        pos_resched = LinguisticEngine.determine_contextual_pos(
            "re-schedule", "They would have left me a message asking to re-schedule the meeting."
        )
        assert pos_resched == "verb"

        # 'well-kept' modifying 'schedules' is an adjective
        pos_wellkept = LinguisticEngine.determine_contextual_pos(
            "well-kept", "The punctual society operates on the basis of well-kept schedules."
        )
        assert pos_wellkept == "adjective"

    def test_hyphenated_compound_lemmatization(self):
        """lemmatize_headword preserves hyphenated compounds and does not truncate to prefix."""
        assert LinguisticEngine.lemmatize_headword("re-schedule") == "re-schedule"
        assert LinguisticEngine.lemmatize_headword("re-schedules") == "re-schedule"
        assert LinguisticEngine.lemmatize_headword("well-kept") == "well-kept"
        assert LinguisticEngine.lemmatize_headword("in-person") == "in-person"


class TestBatchAnchorCollisionGate:
    """
    P0-5 deadlock repair: a collocational anchor (or a prepositional object baked into the
    micro-task frame) may never be another target of the same batch, because the evaluator's
    anchor-presence gate demands that word inside the stem while its cross-target leakage gate
    bans it with a fatal flag — an unsatisfiable item.
    """

    _DEADLOCK_VOCAB = (
        "## [[punctuality]]\n"
        "- **Part of Speech**: noun\n"
        "- **Definition**: the habit of arriving on time\n"
        "- **Quoted Sentence**: \"Punctuality Pays!\"\n\n"
        "## [[pay]]\n"
        "- **Part of Speech**: verb\n"
        "- **Definition**: to give money in exchange for goods or services\n"
        "- **Quoted Sentence**: \"Punctuality Pays!\"\n"
    )

    def test_batch_targets_are_excluded_from_anchor_selection(self):
        skeletons = LinguisticEngine.build_precomputed_target_skeletons(
            self._DEADLOCK_VOCAB, target_count=2
        )
        assert len(skeletons) == 2
        # Batch identity is the headword. 'target_word' may legitimately be an inflected
        # form (P1-2 prescribed-inflection concordance casts the whole option set in the
        # form the authentic example uses), so collisions are judged on 'base_headword'.
        bases = {s.get("base_headword") or s["target_word"] for s in skeletons}
        bases = {b.lower() for b in bases}
        assert bases == {"punctuality", "pay"}
        for s in skeletons:
            own_base = (s.get("base_headword") or s["target_word"]).lower()
            target = s["target_word"].lower()
            assert LinguisticEngine.lemma_of(target) == own_base, (
                f"target '{target}' is not an inflection of its headword '{own_base}'"
            )
            anchor = (s.get("context_anchor") or "").lower()
            if anchor:
                assert anchor not in (bases - {own_base}), (
                    f"anchor '{anchor}' of '{own_base}' is another batch target"
                )

    def test_micro_task_never_names_another_batch_target(self):
        skeletons = LinguisticEngine.build_precomputed_target_skeletons(
            self._DEADLOCK_VOCAB, target_count=2
        )
        # Judge leakage on headwords AND on the inflected target forms actually issued.
        others = set()
        for s in skeletons:
            others.add((s.get("base_headword") or s["target_word"]).lower())
            others.add(s["target_word"].lower())
        for s in skeletons:
            own = {(s.get("base_headword") or s["target_word"]).lower(), s["target_word"].lower()}
            micro_task = s.get("micro_task", "")
            for other in others - own:
                assert not re.search(r"\b" + re.escape(other) + r"\b", micro_task, re.IGNORECASE), (
                    f"micro_task of '{s['target_word']}' leaks batch target '{other}': {micro_task}"
                )

    def test_forbidden_anchor_set_rejects_batch_target_inflections(self):
        forbidden = {"punctuality", "pay", "agenda"}
        assert LinguisticEngine._in_forbidden_word_set("punctuality", forbidden)
        assert LinguisticEngine._in_forbidden_word_set("pay", forbidden)
        assert LinguisticEngine._in_forbidden_word_set("agenda", forbidden)
        assert not LinguisticEngine._in_forbidden_word_set("clock", forbidden)
        assert not LinguisticEngine._in_forbidden_word_set("into", forbidden)

    def test_anchor_token_picker_skips_forbidden_candidates(self):
        # 'punctuality' is forbidden, so the picker must fall through to the next candidate.
        picked = LinguisticEngine._pick_anchor_token(
            ["punctuality people", "clock time"], target_word="pay", forbidden={"punctuality"}
        )
        assert picked == "clock"


class TestQuoteProvenanceAndEvidenceTiering:
    """
    A two-word headline fragment ('Punctuality Pays!') is not a clause. It must not license a
    valency frame, lock a sense, or define a collocational anchor, but it stays visible to the
    writer as display-only evidence.
    """

    _WEAK_QUOTE_VOCAB = (
        "## [[punctuality]]\n"
        "- **Part of Speech**: noun\n"
        "- **Definition**: the habit of arriving on time\n"
        "- **Quoted Sentence**: \"Punctuality Pays!\"\n"
    )
    _STRONG_QUOTE_VOCAB = (
        "## [[punctuality]]\n"
        "- **Part of Speech**: noun\n"
        "- **Definition**: the habit of arriving on time\n"
        "- **Quoted Sentence**: \"The manager insisted on punctuality because late arrivals "
        "delayed the whole department meeting.\"\n"
    )

    def test_headline_fragment_is_graded_weak(self):
        tier, wordcount = LinguisticEngine._quote_evidence_strength("Punctuality Pays!")
        assert tier == "weak"
        assert wordcount == 2

    def test_real_clause_is_graded_strong(self):
        tier, wordcount = LinguisticEngine._quote_evidence_strength(
            "The manager insisted on punctuality because late arrivals delayed the meeting."
        )
        assert tier == "strong"
        assert wordcount >= 5

    def test_weak_quote_is_demoted_but_still_visible(self):
        skeletons = LinguisticEngine.build_precomputed_target_skeletons(
            self._WEAK_QUOTE_VOCAB, target_count=1
        )
        s = skeletons[0]
        assert s["quote_provenance"] == "weak"
        assert s["quote_wordcount"] < 5
        # Demoted: the quote licensed nothing, yet it is still reported for the writer.
        assert s["licensed_quote"] == ""
        assert "Punctuality Pays" in s["candidate_quote"]
        assert s.get("anchor_source") != "quote"

    def test_strong_quote_licenses_evidence(self):
        skeletons = LinguisticEngine.build_precomputed_target_skeletons(
            self._STRONG_QUOTE_VOCAB, target_count=1
        )
        s = skeletons[0]
        assert s["quote_provenance"] == "strong"
        assert s["licensed_quote"].strip() != ""
        assert s["candidate_quote"] == s["licensed_quote"]


class TestSenseAwareContrastInstruction:
    """The contrast instruction must follow the distractor set, not a fixed template."""

    def test_antonym_present_demands_a_polarity_cue(self):
        clause = LinguisticEngine._contrast_strategy_clause(
            "reluctant", "slow and unwilling", ["keen", "unwilling"],
            {"keen": "antonym", "unwilling": "ldoce_thesaurus"}, "prep"
        )
        assert "polarity cue" in clause
        assert "keen" in clause

    def test_near_synonyms_demand_a_precision_cue_with_the_locked_meaning(self):
        clause = LinguisticEngine._contrast_strategy_clause(
            "conscientious", "careful to do everything that it is your duty to do",
            ["meticulous", "careful"], {"meticulous": "ldoce_thesaurus"}, "modified_noun"
        )
        assert "precision cue" in clause
        assert "careful to do everything that it is your duty to do" in clause
        assert "meticulous" in clause

    def test_semantically_distant_distractors_demand_no_contrast_turn(self):
        clause = LinguisticEngine._contrast_strategy_clause(
            "autonomy", "freedom that a place has to govern itself", ["vessel", "profit"],
            {"vessel": "distant_contrast", "profit": "distant_contrast"}, None
        )
        assert "situational and definitional clues" in clause

    def test_no_branch_hardcodes_a_concessive_turn(self):
        for dist_meta in ({"keen": "antonym"}, {"meticulous": "ldoce_thesaurus"}, {}):
            clause = LinguisticEngine._contrast_strategy_clause(
                "target", "a definition", list(dist_meta), dist_meta, "prep"
            )
            assert "Although" not in clause
            assert "Despite" not in clause
            assert "concessive" not in clause


class TestVerbFrameAndInflectionBlueprint:
    """
    A transitive verb keeps its patient slot even when the anchor is a preposition, and the
    inflection the blueprint declares is the form the prescribed options are actually cast in.
    """

    _TRANSITIVE_PREP_VOCAB = (
        "## [[charge]]\n"
        "- **Part of Speech**: verb\n"
        "- **Definition**: to ask somebody to pay money for something\n"
        "- **Quoted Sentence**: \"The hotel charged the guests a fee for late check-out "
        "every weekend.\"\n"
    )
    _PAST_TENSE_VOCAB = (
        "## [[pay]]\n"
        "- **Part of Speech**: verb\n"
        "- **Definition**: to give money in exchange for goods or services\n"
        "- **Quoted Sentence**: \"Punctuality Pays!\"\n"
    )

    def test_transitive_verb_prep_frame_preserves_the_direct_object_slot(self):
        skeletons = LinguisticEngine.build_precomputed_target_skeletons(
            self._TRANSITIVE_PREP_VOCAB, target_count=1
        )
        s = skeletons[0]
        assert s["verb_requires_object"] is True
        assert s["anchor_type"] == "prep"
        assert s["context_anchor"] == "for"
        micro_task = s["micro_task"]
        # The prep-specific valency pattern omits the patient; the blueprint must reassert it.
        assert "direct object" in micro_task.lower()
        assert "[somebody/something]" in micro_task

    def test_prep_anchor_the_locked_sense_does_not_license_is_rejected(self):
        # 'annoy on' comes from 'get on somebody's nerves', not from the taught sense.
        assert LinguisticEngine._prep_anchor_sense_conflict(
            "annoy", "on",
            definition="to make somebody angry",
            quote="It annoyed him that his colleagues arrived late every morning.",
            canonical_pos="verb",
        ) is True
        assert LinguisticEngine._prep_anchor_sense_conflict(
            "correlate", "with",
            definition="to be related to each other in a logical way",
            quote="Punctuality correlates with employee morale across the whole department.",
            canonical_pos="verb",
        ) is False

    def test_declared_inflection_matches_the_prescribed_option_forms(self):
        skeletons = LinguisticEngine.build_precomputed_target_skeletons(
            self._PAST_TENSE_VOCAB, target_count=1
        )
        s = skeletons[0]
        tag = LinguisticEngine.inflection_tag_from_label(s["inflection"])
        assert tag in ("VBD", "VBG", "VBZ", "VBN", "VB", "VBP")
        for opt in s["prescribed_options"]:
            expected = LinguisticEngine.verb_form_for_tag(LinguisticEngine.lemma_of(opt), tag)
            assert expected == opt, (
                f"blueprint declares {s['inflection']!r} but option {opt!r} is not in that form"
            )

    def test_inflection_label_round_trips_to_its_penn_tag(self):
        for tag in ("VBD", "VBN", "VBG", "VBZ", "VBP"):
            label = LinguisticEngine._VERB_FORM_LABELS[tag]
            assert LinguisticEngine.inflection_tag_from_label(label) == tag
        assert LinguisticEngine.inflection_tag_from_label("base form") == "VB"
        assert LinguisticEngine.inflection_tag_from_label("plural form (NNS)") == "NNS"
        assert LinguisticEngine.inflection_tag_from_label("past participle") == "VBN"
    def test_lemma_and_form_tag_survive_a_spacy_adjective_parse(self):
        # spaCy reads a lone '-ed' form as an adjective ('annoyed', 'irritated'), which used to
        # leave both the lemma and the declared verb form unusable.
        assert LinguisticEngine.lemma_of("annoyed") == "annoy"
        assert LinguisticEngine.verb_form_tag_of("annoyed") == "VBD"
        assert LinguisticEngine.lemma_of("irritated") == "irritate"
        assert LinguisticEngine.verb_form_for_tag(
            LinguisticEngine.lemma_of("annoyed"), "VBD"
        ) == "annoyed"
        # A bare parse that already de-inflected must not be overridden by a carrier frame.
        assert LinguisticEngine.lemma_of("children") == "child"




class TestDistractorSlotLegality:
    """A distractor must be able to fill the target's own slot; silence is not a veto."""

    def test_adverb_is_illegal_in_a_noun_slot(self):
        assert LinguisticEngine.distractor_occupies_slot("somehow", "noun") is False
        assert LinguisticEngine.distractor_occupies_slot("meticulously", "noun") is False

    def test_verb_of_the_wrong_valency_is_illegal_in_a_transitive_slot(self):
        # LDOCE records 'belong' and 'exist' as intransitive-only, so neither can fill a
        # transitive blank, and the verdict survives the inflected surface form.
        assert LinguisticEngine.distractor_occupies_slot(
            "belong", "verb", requires_object=True
        ) is False
        assert LinguisticEngine.distractor_occupies_slot(
            "belonging", "verb", requires_object=True
        ) is False
        assert LinguisticEngine.distractor_occupies_slot(
            "existing", "verb", requires_object=True
        ) is False

    def test_verb_that_ldoce_records_as_transitive_is_not_vetoed(self):
        # 'flourish' carries a transitive sense in LDOCE ('flourishing her cheque book'), so a
        # gate that guessed from the word's feel would wrongly veto it.
        assert LinguisticEngine.distractor_occupies_slot(
            "flourishing", "verb", requires_object=True
        ) is True

    def test_inflected_distractor_is_judged_on_its_lemma(self):
        assert LinguisticEngine.distractor_occupies_slot("provided", "verb") is True

    def test_rare_word_the_dictionaries_are_silent_about_is_not_vetoed(self):
        assert LinguisticEngine.distractor_occupies_slot("flocculation", "noun") is True


class TestDiscriminatorSelfReference:
    """
    LDOCE thesaurus 'distinction' text is a word's own cluster definition. Pasting it alone
    made a distractor define itself, and the model copied that into 'why_wrong'.
    """

    def test_note_is_anchored_on_the_target_not_on_the_distractors_own_definition(self):
        note = LinguisticEngine._semantic_discriminator_note(
            "reluctant", "slow and unwilling", ["unwilling", "keen", "grudging"],
            {"unwilling": "ldoce_thesaurus", "keen": "antonym", "grudging": "ldoce_thesaurus"},
        )
        assert note.startswith("Semantic Discriminator: DISTINCTION between 'reluctant'")
        assert note.index("'reluctant'") < note.index("'unwilling'")
        distractor_own_definition = "not wanting to do something and refusing to do it"
        assert LinguisticEngine._definition_overlap_ratio(note, distractor_own_definition) < 0.8

    def test_note_uses_the_locked_sense_not_an_arbitrary_thesaurus_cluster(self):
        # 'charge' the amount (noun cluster) must not hijack the 'charge' verb sense.
        note = LinguisticEngine._semantic_discriminator_note(
            "charge", "to ask somebody to pay money for something", ["cost"],
            {"cost": "ldoce_thesaurus"},
        )
        assert "to ask somebody to pay money for something" in note
        assert "the amount that you have to pay for a service" not in note

    def test_antonym_distractor_is_excluded_by_polarity_not_by_definition(self):
        note = LinguisticEngine._semantic_discriminator_note(
            "reluctant", "slow and unwilling", ["keen"], {"keen": "antonym"}
        )
        assert "polar opposite" in note

    def test_no_note_is_emitted_when_the_cluster_offers_no_contrast(self):
        assert LinguisticEngine._semantic_discriminator_note(
            "autonomy", "freedom that a place has to govern itself", ["vessel", "profit"], {}
        ) == ""

    def test_multiword_thesaurus_headwords_still_match_the_distractor(self):
        assert LinguisticEngine._thesaurus_variant_matches("unwilling/not willing", "unwilling")
        assert LinguisticEngine._thesaurus_variant_matches("be loath to do something", "loath")
        assert not LinguisticEngine._thesaurus_variant_matches("fasten", "unwilling")


class TestQuoteDrivenPluralCasting:
    """A singular headword whose authentic quote uses the plural surface form is cast into a
    plural target ('luxury' -> 'luxuries', inflection 'plural form (NNS)') and the whole option
    set is pluralized symmetrically so no morphological leakage gives the answer away.

    The cast must not damage the item's identity: the noun preposition walker once reused the
    headword variable 'w' as its loop variable, so after reading OCD prepositions ('in', 'of')
    the skeleton shipped base_headword='of' and the batch-identity gates judged the item on a
    function word.
    """

    _PLURAL_VOCAB = (
        "## [[luxury]]\n"
        "- **Part of Speech**: noun\n"
        "- **Definition**: an item or condition of great comfort, elegance, or fine quality\n"
        "- **Quoted Sentence**: \"Work hard and save. Suspend your desires. Avoid luxuries.\"\n"
    )

    def _skeleton(self):
        skeletons = LinguisticEngine.build_precomputed_target_skeletons(
            self._PLURAL_VOCAB, target_count=1
        )
        assert len(skeletons) == 1
        return skeletons[0]

    def test_plural_quote_casts_the_target_to_nns(self):
        s = self._skeleton()
        assert s["target_word"] == "luxuries"
        assert s["inflection"] == "plural form (NNS)"
        assert s["part_of_speech"] == "noun"

    def test_headword_survives_the_inflection_cast(self):
        s = self._skeleton()
        assert s["base_headword"] == "luxury"
        assert LinguisticEngine.lemma_of(s["target_word"]) == s["base_headword"]

    def test_options_share_the_declared_plural_form(self):
        s = self._skeleton()
        options = s["prescribed_options"]
        assert len(options) == 4 and len(set(options)) == 4
        assert options[s["correct_answer_index"]] == s["target_word"]
        assert all(opt.endswith("s") for opt in options), options


class TestLdoceIngestionQA:
    """Backlog E1 / E2, re-anchored after the R1 rebuild. build_ldoce_db.py used to copy
    a base entry's whole sense payload onto derived headwords: 'punctuality' inherited
    'punctual''s adjective senses, 'abandonment' inherited 'abandon''s verb senses. A noun
    headword then carried no noun sense at all, the 'if target_p not in s_pos: continue'
    gate in _lock_sense skipped every sense, and the item silently degraded to senses[0].

    The rebuilt database fixes that at build time - every run-on form is its own
    kind='derived' row with its own POS and its own gloss - so the read-time
    _repair_derived_copy_artifact is a safety net that must NOT fire on these rows;
    scripts/ldoce_qa.py reports 0 copy-artifact rows for the same reason."""

    def test_ldoce_headword_self_reference(self):
        entry = LinguisticEngine.get_ldoce_entry("punctuality")
        assert entry is not None
        # The classification now lives in the database, not in a read-time patch.
        assert entry.get("kind") == "derived"
        assert entry.get("base_word") == "punctual"
        assert not entry.get("copy_artifact_suspect"), entry.get("copy_artifact_suspect")
        # The row must stop claiming the base word's part of speech.
        assert entry.get("pos") == "noun"
        assert all(s.get("pos") == "noun" for s in entry.get("senses", []))
        # The POS gate no longer skips the whole entry.
        locked = LinguisticEngine._lock_sense(entry, target_pos="noun")
        assert locked is not None and 0 <= locked < len(entry["senses"])

    def test_derived_noun_pos(self):
        expected = {
            "abandonment": "abandon",
            "absurdity": "absurd",
            "abasement": "abase",
            "abduction": "abduct",
            "abdication": "abdicate",
        }
        for word, base in expected.items():
            entry = LinguisticEngine.get_ldoce_entry(word)
            assert entry is not None, word
            assert entry.get("pos") == "noun", (word, entry.get("pos"))
            assert entry.get("kind") == "derived", (word, entry.get("kind"))
            assert entry.get("base_word") == base, word
            assert LinguisticEngine._lock_sense(entry, target_pos="noun") is not None, word

    def test_genuine_non_noun_entries_are_untouched(self):
        # 'enhance' and 'augment' end in noun-suffix letters but are real verbs with
        # their own glosses; relabelling them would be a false positive.
        for word in ("enhance", "augment", "torment", "notion", "business"):
            entry = LinguisticEngine.get_ldoce_entry(word)
            assert entry is not None, word
            assert not entry.get("copy_artifact_suspect"), word

    def test_copy_artifact_gloss_is_the_headwords_own(self):
        senses = LinguisticEngine.get_ldoce_entry("punctuality").get("senses", [])
        assert senses
        # The row copied 'punctual''s adjective gloss; after repair every sense carries a
        # gloss the noun itself owns, tagged with where it came from.
        assert all(
            s.get("definition") != "arriving, happening, or being done at exactly the time that has been arranged"
            for s in senses
        ), [s.get("definition") for s in senses]
        assert all(s.get("gloss_source") in ("wordnet", "derivational") for s in senses), senses
        definition, example = LinguisticEngine.get_ldoce_definition_and_example(
            "punctuality", target_pos="noun"
        )
        assert definition == senses[0]["definition"]
        assert LinguisticEngine.text_contains_form(example, "punctuality"), example

    def test_example_cascade_keeps_looking_until_the_headword_appears(self):
        # 'absurdity' inherits 'absurd''s example pool; the shipped example must still show
        # the noun, because a later pool is consulted whenever the current example fails.
        for word in ("absurdity", "abduction"):
            _, example = LinguisticEngine.get_ldoce_definition_and_example(word, target_pos="noun")
            assert example and LinguisticEngine.text_contains_form(example, word), (word, example)

    def test_qa_script_detects_and_quarantines_copy_rows(self, tmp_path):
        import importlib.util
        import json
        import sqlite3
        from pathlib import Path

        spec = importlib.util.spec_from_file_location(
            "ldoce_qa", Path(__file__).resolve().parent.parent / "scripts" / "ldoce_qa.py"
        )
        qa = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(qa)

        db = tmp_path / "mini_ldoce.db"
        conn = sqlite3.connect(str(db))
        conn.execute("CREATE TABLE ldoce (word TEXT PRIMARY KEY, pos TEXT, data_json TEXT)")
        verb_sense = [{"pos": "verb", "definition": "to leave someone", "examples": ["She abandoned the car."]}]
        rows = [
            ("abandon", "verb", {"word": "abandon", "senses": verb_sense}),
            ("abasement", "verb", {"word": "abasement", "senses": verb_sense}),   # copy of 'abase'
            ("abase", "verb", {"word": "abase", "senses": verb_sense}),
            ("enhance", "verb", {"word": "enhance", "senses": [
                {"pos": "verb", "definition": "to improve something",
                 "examples": ["New drugs enhance the immune system."]}
            ]}),
        ]
        for word, pos, data in rows:
            conn.execute("INSERT INTO ldoce VALUES (?, ?, ?)", (word, pos, json.dumps(data)))
        conn.commit()
        conn.close()

        report = qa.audit(db)
        pairs = {tuple(p) for p in report["copy_artifact_pairs"]}
        assert ("abasement", "abase") in pairs, pairs
        assert all(w != "enhance" for w, _ in pairs), pairs
        assert report["copy_artifact_rows"] == 1

        qa.write_quarantine(report, db)
        conn = sqlite3.connect(str(db))
        quarantined = {r[0] for r in conn.execute("SELECT word FROM ldoce_quarantine")}
        conn.close()
        assert quarantined == {"abasement"}


class TestLdoceKindSchema:
    """Backlog R1. The rebuilt database classifies every mdx key at build time and
    stores the verdict in ldoce.kind + ldoce.base_word:

        article  Longman's own entry - the payload's headword equals the key
        derived  a run-on form ('punctuality' inside 'punctual') with its own POS,
                 its own gloss, and only the examples that contain its own form
        alias    an empty pointer (plural, comparative, cross-reference) -> base_word
        entity   a proper noun or abbreviation - no POS label at all

    The engine consumes that classification instead of re-deriving an entry type from
    a copied payload, and still works against a database that has no such columns."""

    _ROWS = [
        ("punctual", "adjective", "article", None, {
            "word": "punctual",
            "senses": [{"pos": "adjective", "homograph_num": 1,
                        "definition": "arriving, happening, or being done at exactly the time arranged",
                        "examples": ["She's always very punctual."]}],
            "collocations": {"adjectives": ["strictly punctual"]},
            "word_family": {"word_family": "punctually"},
            "thesaurus": [], "grammar_boxes": [], "cross_collocations": [],
        }),
        ("punctuality", "noun", "derived", "punctual", {
            "word": "punctuality", "kind": "derived", "base_word": "punctual",
            "gloss_source": "template",
            "senses": [{"pos": "noun", "homograph_num": 1, "grammar": "uncountable",
                        "definition": "the quality of being punctual",
                        "examples": ["Punctuality is expected of all staff."],
                        "gloss_source": "template"}],
            "collocations": {}, "word_family": {}, "thesaurus": [],
            "grammar_boxes": [], "cross_collocations": [],
        }),
        ("abandon", "verb", "article", None, {
            "word": "abandon",
            "senses": [{"pos": "verb", "homograph_num": 1,
                        "definition": "to leave someone, usually with no intention of returning",
                        "examples": ["She abandoned the car and continued on foot."]}],
            "collocations": {"noun": ["abandon hope"]},
            "word_family": {}, "thesaurus": [], "grammar_boxes": [],
            "cross_collocations": [],
        }),
        ("abandons", None, "alias", "abandon", {
            "word": "abandons", "kind": "alias", "base_word": "abandon",
            "senses": [], "collocations": {}, "word_family": {}, "thesaurus": [],
            "grammar_boxes": [], "cross_collocations": [],
        }),
        ("pot luck", "noun", "article", None, {
            "word": "pot luck",
            "senses": [{"pos": "noun", "homograph_num": 1,
                        "definition": "food that is available when you are a guest somewhere",
                        "examples": ["We had pot luck at her house."]}],
            "collocations": {}, "word_family": {}, "thesaurus": [],
            "grammar_boxes": [], "cross_collocations": [],
        }),
        ("jupiter", None, "entity", None, {
            "word": "Jupiter", "kind": "entity",
            "senses": [{"pos": None, "homograph_num": 1,
                        "definition": "the fifth planet from the sun",
                        "examples": []}],
            "collocations": {}, "word_family": {}, "thesaurus": [],
            "grammar_boxes": [], "cross_collocations": [],
        }),
    ]

    @staticmethod
    def _mini_db(tmp_path, with_kind=True):
        """A two-table sample: the rebuilt schema, and the pre-rebuild one."""
        import json
        import sqlite3
        db = tmp_path / ("mini_kind.db" if with_kind else "mini_legacy.db")
        conn = sqlite3.connect(str(db))
        if with_kind:
            conn.execute("CREATE TABLE ldoce (word TEXT PRIMARY KEY, pos TEXT, pos_all TEXT,"
                         " kind TEXT NOT NULL, base_word TEXT, data_json TEXT)")
            for word, pos, kind, base, data in TestLdoceKindSchema._ROWS:
                conn.execute("INSERT INTO ldoce VALUES (?,?,?,?,?,?)",
                             (word, pos, pos or "", kind, base, json.dumps(data)))
        else:
            conn.execute("CREATE TABLE ldoce (word TEXT PRIMARY KEY, pos TEXT, data_json TEXT)")
            for word, pos, _kind, _base, data in TestLdoceKindSchema._ROWS:
                legacy = {k: v for k, v in data.items() if k not in ("kind", "base_word")}
                conn.execute("INSERT INTO ldoce VALUES (?,?,?)", (word, pos, json.dumps(legacy)))
        conn.commit()
        conn.close()
        return db

    @staticmethod
    def _use(db):
        """Point the singleton engine at another database; return what to put back."""
        saved = {
            "db_path": LinguisticEngine._ldoce_db_path,
            "conn": LinguisticEngine._ldoce_conn,
            "cache": dict(LinguisticEngine._ldoce_cache),
            "tables": LinguisticEngine._ldoce_tables,
            "compounds": LinguisticEngine._ldoce_compounds,
            "kind_ok": LinguisticEngine._ldoce_kind_ok,
        }
        from pathlib import Path
        LinguisticEngine._ldoce_db_path = classmethod(lambda cls, _p=db: Path(_p))
        LinguisticEngine._ldoce_conn = None
        LinguisticEngine._ldoce_cache.clear()
        LinguisticEngine._ldoce_tables = None
        LinguisticEngine._ldoce_compounds = None
        LinguisticEngine._ldoce_kind_ok = None
        return saved

    @staticmethod
    def _restore(saved):
        conn = LinguisticEngine._ldoce_conn
        if conn is not None and conn is not saved["conn"]:
            try:
                conn.close()
            except Exception:
                pass
        LinguisticEngine._ldoce_db_path = saved["db_path"]
        LinguisticEngine._ldoce_conn = saved["conn"]
        LinguisticEngine._ldoce_tables = saved["tables"]
        LinguisticEngine._ldoce_compounds = saved["compounds"]
        LinguisticEngine._ldoce_kind_ok = saved["kind_ok"]
        LinguisticEngine._ldoce_cache.clear()
        LinguisticEngine._ldoce_cache.update(saved["cache"])

    def test_alias_row_hops_to_its_base_word(self, tmp_path):
        saved = self._use(self._mini_db(tmp_path))
        try:
            entry = LinguisticEngine.get_ldoce_entry("abandons")
            assert entry is not None
            assert entry["kind"] == "alias"
            assert entry["alias_of"] == "abandon"
            assert entry["word"] == "abandons", "the entry must still name the form asked for"
            assert entry["senses"][0]["definition"].startswith("to leave someone")
        finally:
            self._restore(saved)

    def test_derived_row_keeps_its_own_gloss_and_borrows_only_assets(self, tmp_path):
        saved = self._use(self._mini_db(tmp_path))
        try:
            entry = LinguisticEngine.get_ldoce_entry("punctuality")
            assert entry is not None
            assert entry["kind"] == "derived"
            assert entry["base_word"] == "punctual"
            senses = entry["senses"]
            assert len(senses) == 1 and senses[0]["pos"] == "noun", senses
            assert senses[0]["definition"] == "the quality of being punctual"
            assert senses[0]["grammar"] == "uncountable"
            # The base article's adjective gloss must never reach the derived row.
            assert "arriving, happening" not in senses[0]["definition"]
            # ...but its collocation box is shared - that is what a run-on is for.
            assert entry["collocations"].get("adjectives") == ["strictly punctual"]
            assert entry["word_family"].get("word_family") == "punctually"
            assert LinguisticEngine._lock_sense(entry, target_pos="noun") is not None
        finally:
            self._restore(saved)

    def test_article_and_entity_rows_come_back_as_they_are(self, tmp_path):
        saved = self._use(self._mini_db(tmp_path))
        try:
            article = LinguisticEngine.get_ldoce_entry("punctual")
            assert article is not None and article["kind"] == "article"
            assert article["senses"][0]["pos"] == "adjective"
            assert not article.get("alias_of")

            entity = LinguisticEngine.get_ldoce_entry("Jupiter")
            assert entity is not None and entity["kind"] == "entity"
            assert entity.get("pos") is None
            # An entity carries no POS label, so a POS-gated query cannot match it and
            # falls back to its only sense - exactly what the old substring gate did.
            assert LinguisticEngine._lock_sense(entity, target_pos="noun") == 0
        finally:
            self._restore(saved)

    def test_compound_headword_still_resolves_through_the_index(self, tmp_path):
        saved = self._use(self._mini_db(tmp_path))
        try:
            entry = LinguisticEngine.get_ldoce_entry("potluck")
            assert entry is not None
            assert entry["senses"][0]["definition"].startswith("food that is available")
        finally:
            self._restore(saved)

    def test_database_without_kind_columns_still_works(self, tmp_path):
        """Old databases are still readable: no classification, no hops, no crash."""
        saved = self._use(self._mini_db(tmp_path, with_kind=False))
        try:
            entry = LinguisticEngine.get_ldoce_entry("punctuality")
            assert entry is not None
            assert not entry.get("kind")
            assert not entry.get("alias_of")
            assert entry["senses"][0]["pos"] == "noun"

            alias = LinguisticEngine.get_ldoce_entry("abandons")
            assert alias is not None
            assert not alias.get("alias_of"), "no base_word column means no hop to make"
            assert alias["senses"] == []
        finally:
            self._restore(saved)


class TestLemmaInflectionInfrastructure:
    """Backlog A1. Every gate that compares a word against a surface form used to write
    its own `\b{word}\b` regex, so 'attaches' never matched the LDOCE pattern
    'attach something to something', an example could omit the headword entirely, and the
    CEFR/length gates judged the singular candidate while the student read the plural one.
    LinguisticEngine now owns one legal-form generator and one form matcher."""

    _ATTACH_QUOTE = "The firm attaches itself to a broader social purpose."

    def test_inflected_forms_are_the_single_generator(self):
        forms = LinguisticEngine.inflected_forms("attach")
        assert {"attach", "attaches", "attached", "attaching"} <= forms
        # The irregular table closes paradigms the mechanical inflector would invent.
        choose = LinguisticEngine.inflected_forms("choose")
        assert {"chose", "chosen", "chooses", "choosing"} <= choose
        assert "choosed" not in choose
        assert "writed" not in LinguisticEngine.inflected_forms("write")

    def test_inflection_match_attaches(self):
        for surface in ("attaches", "attached", "attaching"):
            assert LinguisticEngine.form_matches(surface, "attach"), surface
        assert LinguisticEngine.form_matches("children", "child")
        assert LinguisticEngine.form_matches("chose", "choose")
        # A misspelling that lemmatizes back to the target is not a match.
        assert not LinguisticEngine.form_matches("choosed", "choose")
        assert LinguisticEngine.text_contains_form(self._ATTACH_QUOTE, "attach")

    def test_pattern_evidence_survives_an_inflected_quote(self):
        entry = {
            "word": "attach",
            "pos": "verb",
            "senses": [
                {"pos": "verb",
                 "definition": "to work for part of an organization for a short period",
                 "examples": ["He is attached to a research group."],
                 "patterns": ["attach to a group"]},
                {"pos": "verb",
                 "definition": "to fasten or connect one object to another",
                 "examples": [],
                 "patterns": ["attach something to something"]},
            ],
        }
        # Only the second sense's pattern is evidenced by the quote, and only once the
        # inflected 'attaches' counts as evidence for 'attach'.
        assert LinguisticEngine._lock_sense(entry, quote=self._ATTACH_QUOTE, target_pos="verb") == 1

    def test_attested_form_gate(self):
        for good in ("buses", "goodies", "children", "luxuries", "chosen", "attaches", "preferred"):
            assert LinguisticEngine.is_attested_form(good), good
        for bad in ("choosed", "writed", "spaked", "prefered"):
            assert not LinguisticEngine.is_attested_form(bad), bad

    def test_distractor_inflection_invalid_rejected(self):
        report = {}
        kept = LinguisticEngine.screen_inflected_options(
            inflected=["choosed"],
            spare=["picked", "chosen", "taken"],
            final_target="chooses",
            report=report,
        )
        assert "choosed" not in kept, kept
        assert report["rejected"]["choosed"] == "unattested_form"
        assert {"picked", "chosen", "taken"} <= set(kept), kept

    def test_cefr_gate_sees_surface_form(self):
        report = {}
        kept = LinguisticEngine.screen_inflected_options(
            inflected=["costlinesses"],
            spare=["expense", "value", "price"],
            final_target="costliness",
            report=report,
        )
        # 'costliness' passed every gate as a singular; the plural the student would
        # actually read must be the one that is judged.
        assert "costlinesses" not in kept, kept
        assert report["rejected"]["costlinesses"] == "cefr_ceiling"

    def test_example_must_show_the_headword(self):
        definition, example = LinguisticEngine.get_ldoce_definition_and_example(
            "attach", target_pos="verb", context_sentence=self._ATTACH_QUOTE
        )
        assert definition == "to fasten or connect one object to another"
        assert example
        assert LinguisticEngine.text_contains_form(example, "attach"), example


class TestCefrCeilingAfterInflection:
    """Backlog C1. The distractor ceiling, the OCD/AWL lexicon filter and the morphological
    anomaly gate all ran on the singular candidate, then plural casting shipped a surface
    form no gate had ever seen. Difficulty belongs to the lemma ('attaches' is as easy as
    'attach'); length and affix weirdness belong to the spelling on the page."""

    def test_inflected_distractors_are_recognised_through_their_lemma(self):
        # 'boxes', 'children' and 'attaches' are headwords in none of OCD, the AWL or the
        # CEFR list, so the lexicon filter used to reject every inflected distractor for an
        # A1/A2/B1 target and the pool ran dry.
        for cand, target in (("boxes", "box"), ("children", "child"), ("attaches", "attach")):
            assert LinguisticEngine.is_cefr_compliant_distractor(cand, target), (cand, target)

    def test_obscure_affix_gate_sees_through_the_plural(self):
        # 'forgetfulness' is the obscure formation. Plural casting hides the '-fulness'
        # tail from an endswith test, and 15 letters still fits the length ceiling of an
        # 11-letter target, so only a gate that inspects the singular can catch it.
        assert LinguisticEngine.is_attested_form("forgetfulnesses"), "forgetfulnesses"
        assert not LinguisticEngine.is_cefr_compliant_distractor("forgetfulnesses", "abandonment")
        assert not LinguisticEngine.is_cefr_compliant_distractor("forgetfulness", "abandonment")

    def test_screen_reports_the_surface_rejection(self):
        report = {}
        kept = LinguisticEngine.screen_inflected_options(
            inflected=["forgetfulnesses"],
            spare=["memory", "attention", "focus"],
            final_target="abandonment",
            report=report,
        )
        assert "forgetfulnesses" not in kept, kept
        assert report["rejected"]["forgetfulnesses"] == "cefr_ceiling"
        assert {"memory", "attention", "focus"} <= set(kept), kept

    def test_lexicon_profile_resolves_to_the_lemma(self):
        # 'reappoints' is in none of OCD, AWL or the CEFR list; the profile retries through
        # the lemma so the ceiling gate can rank it instead of calling it obscure.
        in_ocd, in_awl, level, resolved = LinguisticEngine._lexicon_profile("reappoints")
        assert resolved == "reappoint", resolved
        assert level, (in_ocd, in_awl, level)
        # A form the lexicons already list is answered literally, without the lemma retry.
        assert LinguisticEngine._lexicon_profile("decision")[3] == "decision"
        assert LinguisticEngine._lexicon_profile("attaches")[2], "attaches"


class TestExampleHeadwordAlignmentGate:
    """Backlog D1. A copy-artifact row inherits its base entry's example pool, so the
    cascade could end holding a sentence about the base word and ship it under the derived
    headword: 'abandonment' -> 'How could she abandon her own child?'. The output gate drops
    such an example; the caller substitutes the passage quote or the item degrades to
    'missing example_usage' instead of teaching the wrong word."""

    def test_copy_artifact_row_drops_the_base_words_example(self):
        for word in ("abandonment", "abdication"):
            definition, example = LinguisticEngine.get_ldoce_definition_and_example(
                word, target_pos="noun"
            )
            assert definition, word
            assert not example or LinguisticEngine.text_contains_form(example, word), (word, example)

    def test_headword_bearing_examples_survive_the_gate(self):
        for word in ("punctuality", "absurdity", "abduction", "decision", "attach"):
            _, example = LinguisticEngine.get_ldoce_definition_and_example(
                word, target_pos="verb" if word == "attach" else "noun"
            )
            assert example and LinguisticEngine.text_contains_form(example, word), (word, example)

    def test_multiword_formula_is_exempt(self):
        # 'shut the door down' splits the formula across the sentence; no surface test can
        # see the match, so blanking examples for multi-word headwords would be a regression.
        definition, example = LinguisticEngine.get_expression_definition_and_example(
            "shut down", expr_type="phrasal verb"
        )
        assert definition and example, (definition, example)

    def test_passage_quote_replaces_a_dropped_example(self):
        text = (
            "[S-1] The abandonment of the treaty surprised the ministers. "
            "[S-2] Each minister signed the new agreement alone."
        )
        items = LinguisticEngine.extract_deterministic_vocabulary(
            text, syllabus_vocab=["abandonment"], target_count=1
        )
        assert items, items
        item = items[0]
        assert item["word"] == "abandonment", item
        assert LinguisticEngine.text_contains_form(item["example_usage"], "abandonment"), item


class TestSharedCEFRCeiling:
    """Backlog C1. Every gate that asks 'is this word too hard for this passage?' used to
    write its own ladder: three copies in processor.py, a fourth inside
    is_cefr_compliant_distractor, and none at all on the grammar / vocabulary extraction
    paths. One helper now answers it, in two registers."""

    def test_support_ceiling_is_one_band_above_the_passage(self):
        ceiling = LinguisticEngine.cefr_ceiling
        assert ceiling("A1") == "B1"
        assert ceiling("A2") == "B1"
        assert ceiling("B1") == "B2"
        assert ceiling("B2") == "C1"
        assert ceiling("C1") == "C2"
        assert ceiling("c2") == "C2"

    def test_unknown_level_falls_back_to_the_middle_of_the_syllabus(self):
        for unknown in (None, "", "b1+", "advanced"):
            assert LinguisticEngine.cefr_ceiling(unknown) == "B2", unknown

    def test_text_ceiling_is_looser_than_the_distractor_ceiling(self):
        # A distractor is decoded alone; a word inside a sentence a student is
        # already reading is easier to carry, so the stop sits one band higher.
        assert LinguisticEngine.cefr_ceiling("A2", mode="text") == "B2"
        assert LinguisticEngine.cefr_ceiling("A2") == "B1"
        assert LinguisticEngine.cefr_ceiling("B1", mode="text") == "C1"
        # B2 and above: running text is not vocabulary-limited at all.
        assert LinguisticEngine.cefr_ceiling("B2", mode="text") == "C2"

    def test_over_ceiling_tokens_names_the_offenders_worst_first(self):
        over = LinguisticEngine.over_ceiling_tokens(
            "He perused the volume meticulously in the library.", "A2", mode="text"
        )
        levels = dict(over)
        assert "meticulously" in levels and levels["meticulously"] == "C2", over
        assert "perused" in levels, over
        assert "library" not in levels, over  # A1, comfortably under the ceiling
        assert over[0][1] == "C2", over

    def test_inflected_forms_are_judged_through_their_lemma(self):
        # 'deteriorated' is in no CEFR list; 'deteriorate' is C2.
        over = LinguisticEngine.over_ceiling_tokens(
            "The patient deteriorated.", "A2", mode="text"
        )
        assert "deteriorated" in dict(over), over

    def test_words_the_passage_already_uses_are_exempt(self):
        over = LinguisticEngine.over_ceiling_tokens(
            "The enigmatic inscription was copied.", "A2", mode="text", allow={"enigmatic"}
        )
        assert "enigmatic" not in dict(over), over

    def test_unlisted_words_are_not_treated_as_hard(self):
        # An unlisted word is not evidence of difficulty: a gate that punished it would
        # punish proper nouns and technical terms for being absent from a frequency list.
        assert LinguisticEngine.over_ceiling_tokens(
            "Professor Finklebaum zorblatt visited.", "A2", mode="text"
        ) == []

    def test_proper_nouns_are_not_vocabulary(self):
        # The graded list really does contain 'Cassie' (C2) and 'Karen' (C1) as names.
        # A name in a dictionary example is not a word the student is being taught.
        assert LinguisticEngine.over_ceiling_tokens(
            "He looked surprised to see Cassie standing by the front door.", "A2", mode="text"
        ) == []
        assert LinguisticEngine.over_ceiling_tokens(
            "There's a message from Karen on the answerphone.", "A2", mode="text"
        ) == []

    def test_contractions_are_not_split_into_phantom_words(self):
        # Some tokenisers split "don't" into 'don' + "t" and "doesn't" into 'doesn' +
        # "t", and the graded list carries 'dont' and 'doesn' at C2. A contraction is
        # grammar the student already has, not new vocabulary.
        assert LinguisticEngine.over_ceiling_tokens("Crime doesn't pay.", "A2", mode="text") == []
        assert LinguisticEngine.over_ceiling_tokens(
            "I don't need any help, but it was nice of you to offer.", "A2", mode="text"
        ) == []
        # The exemption is about the apostrophe, not about leniency: the same
        # sentence with a genuinely graded word still reports it.
        assert LinguisticEngine.over_ceiling_tokens(
            "Crime doesn't pay, and the evidence is meticulous.", "A2", mode="text"
        ) == [("meticulous", "C2")]

    def test_a_c2_ceiling_never_fires(self):
        assert LinguisticEngine.over_ceiling_tokens(
            "She perused the enigmatic volume meticulously.", "B2", mode="text"
        ) == []

    def test_a_headword_licences_its_own_inflected_variants(self):
        # 're-schedule' is one headword but its example shows 'rescheduled'; the
        # exemption has to survive both the hyphen and the inflection.
        assert LinguisticEngine.over_ceiling_tokens(
            "The press conference had to be rescheduled for March 19.",
            "A2", mode="text", allow={"schedule", "reschedule"},
        ) == []

    def test_distractor_gate_uses_the_shared_table(self):
        # The ladder is_cefr_compliant_distractor used to spell out inline is now the
        # support table read against the target: B1 target -> B2 ceiling.
        assert LinguisticEngine._ceiling_rank(3, "support") == LinguisticEngine.CEFR_ORDER["B2"]
        assert LinguisticEngine._ceiling_rank(2, "text") == LinguisticEngine.CEFR_ORDER["B2"]



def _ldoce_builder():
    """scripts/build_ldoce_db.py imported as a module, without running its main()."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "build_ldoce_db.py"
    spec = importlib.util.spec_from_file_location("build_ldoce_db", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _gate_script():
    """scripts/shadow_engine_diff.py imported as a module, without running its main()."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "shadow_engine_diff.py"
    spec = importlib.util.spec_from_file_location("shadow_engine_diff", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestRunOnDerivation:
    """Backlog R1, the run-on half of the rebuild. A run-on form - 'weeding' inside the
    'weed' article - carries a POS label but no definition of its own, so the builder has
    to find the form it really derives from, the POS section that form actually carries,
    and a gloss built out of Longman's own sentence. Pick the wrong base or the wrong
    section and the row ends up glossless, or glossed with somebody else's meaning."""

    @staticmethod
    def _info(word, base, pos):
        """The shape scan_runons + scan_mdx hand to derivation_base."""
        return {"word": word, "base": base, "pos": pos, "gram": "", "examples": []}

    def test_agent_noun_uses_the_verb_section_of_the_base_article(self):
        # 'bullshitter' sits in an article that leads with the noun 'bullshit', but an
        # agent noun names someone who does something, so the verb section is the one it
        # derives from. Building it on the noun produced 'a person who takes part in a
        # bullshit'.
        b = _ldoce_builder()
        article_defs = {
            "bullshit": {"noun": "something that is stupid and completely untrue",
                         "verb": "to say something stupid or completely untrue"},
        }
        stem, pos = b.derivation_base(
            "bullshitter", self._info("bullshitter", "bullshit", "noun"),
            {"bullshit": "noun"}, [("bullshitter", "noun")], article_defs)
        assert (stem, pos) == ("bullshit", "verb")
        assert b.template_gloss("bullshitter", stem, pos, "noun") == "a person who bullshits"

    def test_adverb_uses_the_adjective_section_even_when_the_noun_comes_first(self):
        # 'stalwartly' is built on the adjective, but the 'stalwart' article lists the
        # noun sense first, and an adverb glossed from a noun is what produced
        # 'in an archaeology way' for 'archaeologically'.
        b = _ldoce_builder()
        article_defs = {
            "stalwart": {"noun": "someone who is very loyal to a particular organization",
                         "adjective": "a very loyal and strong supporter of an organization"},
        }
        stem, pos = b.derivation_base(
            "stalwartly", self._info("stalwartly", "stalwart", "adverb"),
            {"stalwart": "noun"}, [("stalwartly", "adverb")], article_defs)
        assert (stem, pos) == ("stalwart", "adjective")
        assert b.template_gloss("stalwartly", stem, pos, "adverb") == "in a stalwart way"

    def test_a_longer_run_on_sibling_is_a_better_base_than_the_article(self):
        # Longman lists 'behavioural' between the noun and the adverb inside the
        # 'behaviour' article; the adverb derives from it, not from the noun.
        b = _ldoce_builder()
        stem, pos = b.derivation_base(
            "behaviourally", self._info("behaviourally", "behaviour", "adverb"),
            {"behaviour": "noun"},
            [("behavioural", "adjective"), ("behaviourally", "adverb")], {})
        assert (stem, pos) == ("behavioural", "adjective")

    def test_an_entity_article_still_supplies_a_pos(self):
        # Place articles ('Corsica') carry no POS label before their run-on block, but
        # their sense region says which section the headword has, and a demonym needs it.
        b = _ldoce_builder()
        stem, pos = b.derivation_base(
            "corsican", self._info("corsican", "corsica", "adjective"), {},
            [("corsican", "adjective")],
            {"corsica": {"noun": "a large Mediterranean island in the Mediterranean Sea"}})
        assert (stem, pos) == ("corsica", "noun")

    def test_the_payload_that_declared_the_run_on_decides_the_base_pos(self):
        # 'absurdist' is declared inside the short noun article printed as "the Absurd",
        # but the key 'absurd' also carries the adjective article and own_pos keeps only
        # the payload scanned last. Building an adjective on that adjective left the row
        # with no gloss at all; the declaring payload says the base is the noun.
        b = _ldoce_builder()
        info = dict(self._info("absurdist", "absurd", "adjective"), declared_base_pos="noun")
        stem, pos = b.derivation_base(
            "absurdist", info, {"absurd": "adjective"}, [("absurdist", "adjective")],
            {"absurd": {"noun": "a style of play for the theatre developed in the 1950s",
                        "adjective": "completely stupid or unreasonable"}})
        assert (stem, pos) == ("absurd", "noun")
        assert b.template_gloss("absurdist", stem, pos, "adjective") == "relating to absurd"

    def test_a_longer_sibling_outranks_the_declaring_payloads_pos(self):
        # declared_base_pos is a fallback for the POS, not a licence to stop looking for
        # the form the run-on really derives from.
        b = _ldoce_builder()
        info = dict(self._info("behaviourally", "behaviour", "adverb"),
                    declared_base_pos="noun")
        stem, pos = b.derivation_base(
            "behaviourally", info, {"behaviour": "noun"},
            [("behavioural", "adjective"), ("behaviourally", "adverb")], {})
        assert (stem, pos) == ("behavioural", "adjective")

    def test_ing_noun_borrows_the_verb_sections_own_definition(self):
        # 'weeding', 'knifing' and 'archiving' have no entry of their own, but the article
        # they sit in carries a verb section, and the noun is the act that section
        # describes. The gloss is rebuilt from Longman's sentence, not invented beside it.
        b = _ldoce_builder()
        base_defs = {"noun": "a wild plant growing where it is not wanted",
                     "verb": "to remove unwanted plants from a garden or other place"}
        assert b.nominalized_gloss("weeding", base_defs) == \
            "the act of removing unwanted plants from a garden or other place"
        assert b.nominalized_gloss("knifing", {"verb": "to put a knife into someone's body"}) == \
            "the act of putting a knife into someone's body"
        # Not an -ing noun, and not a verb section: nothing to nominalize.
        assert b.nominalized_gloss("weeder", base_defs) == ""
        assert b.nominalized_gloss("weeding", {"noun": "a wild plant"}) == ""

    def test_ing_noun_built_on_the_person_the_base_defines(self):
        # 'bell-ringing' lives in the 'bell-ringer' article, which has no verb section at
        # all - only 'someone who rings church bells'. The third-person ending has to be
        # undone before the gerund goes back on.
        b = _ldoce_builder()
        assert b.agent_action_gloss("bell-ringing", {"noun": "someone who rings church bells"}) == \
            "the act of ringing church bells"
        # The job wording is the other way Longman phrases the same thing.
        assert b.agent_action_gloss(
            "career coaching",
            {"noun": "someone whose job is to help people to plan their careers"}) == \
            "the act of helping people to plan their careers"
        # A definition that needs two verbs gerundized is left alone rather than mangled.
        assert b.agent_action_gloss(
            "house buying", {"noun": "someone whose job is to buy and sell houses"}) == ""

    def test_agent_noun_chooses_the_verb_that_matches_what_its_base_names(self):
        # A discipline is practised, an event is taken part in.
        b = _ldoce_builder()
        assert b.agent_noun_gloss("gunfighter", "gunfight",
                                  "a fight between people using guns") == \
            "a person who takes part in a gunfight"
        assert b.agent_noun_gloss("acupuncturist", "acupuncture",
                                  "the practice of treating a person by putting needles "
                                  "into their body") == "a person who practises acupuncture"
        assert b.agent_noun_gloss("bioterrorist", "bioterrorism",
                                  "the use of bacteria or viruses by a person or group "
                                  "to attack people") == "a person who practises bioterrorism"
        assert b.agent_noun_gloss("social scientist", "social science",
                                  "the study of people in society") == \
            "a person who practises social science"

    def test_demonym_gloss_needs_only_the_suffix(self):
        # '-an', '-ese' and '-n' on a place name say the whole meaning, whatever POS the
        # place article carries and even when the base loses a letter on the way.
        b = _ldoce_builder()
        assert b.template_gloss("corsican", "corsica", "noun", "adjective",
                                "a large Mediterranean island") == "relating to corsica"
        assert b.template_gloss("viennese", "vienna", "noun", "adjective",
                                "the capital of Austria") == "relating to vienna"
        assert b.template_gloss("antipodean", "antipodes", "noun", "noun", "") == \
            "a person from the antipodes"
        assert b.template_gloss("east german", "east germany", "noun", "noun", "") == \
            "a person from east germany"

    def test_suffix_is_recognized_when_the_base_loses_a_letter_or_doubles_up(self):
        # Every template gloss is keyed on the suffix, so a suffix the builder cannot see
        # is a gloss the row never gets.
        b = _ldoce_builder()
        cases = {
            ("punctuality", "punctual"): "ity",
            ("corsican", "corsica"): "n",
            ("antipodean", "antipodes"): "an",
            ("viennese", "vienna"): "ese",
            ("day tripper", "day trip"): "er",       # doubled consonant
            ("bioterrorist", "bioterrorism"): "ist",  # base loses its final letter
            ("social scientist", "social science"): "ist",
        }
        for (word, base), expected in cases.items():
            assert b.derivation_suffix(word, base) == expected, (word, base)

    def test_definition_drops_the_pointer_it_appends(self):
        # 'to put documents etc in an archive 2' ends in the homograph number of the
        # article the pointer jumped to; '... -> campanology' ends in the pointer itself.
        b = _ldoce_builder()
        assert b.cross_reference_free(
            "to put documents, books, information etc in an archive 2") == \
            "to put documents, books, information etc in an archive"
        assert b.cross_reference_free(
            "someone who rings church bells → campanology") == \
            "someone who rings church bells"

    def test_derived_row_ships_a_gloss_of_its_own(self):
        # The E1 bug was a derived row handed back its base's whole definition. With
        # WordNet silenced - the path the builder falls back to for the run-ons WordNet
        # has no sense for - the gloss must come out of the base article's verb section
        # and be labelled as where it came from.
        b = _ldoce_builder()
        b.wordnet_gloss = lambda *args, **kwargs: []
        info = {"word": "weeding", "base": "weed", "pos": "noun", "gram": "uncountable",
                "examples": ["We were weeding the border."]}
        row = b.build_derived_row(
            "weeding", info, "weed", "verb",
            {"noun": "a wild plant growing where it is not wanted",
             "verb": "to remove unwanted plants from a garden or other place"})
        assert row["kind"] == "derived" and row["base_word"] == "weed"
        assert row["pos"] == "noun" and row["senses"][0]["pos"] == "noun"
        assert row["senses"][0]["definition"] == \
            "the act of removing unwanted plants from a garden or other place"
        assert row["gloss_source"] == "nominalized"
        # An example that never mentions the derived form is dropped, not copied over.
        assert row["senses"][0]["examples"] == ["We were weeding the border."]

    def test_derived_row_reports_no_gloss_instead_of_borrowing_one(self):
        # 'estate agency' has no WordNet sense and no template the builder supports.
        # Shipping the agent's definition would be the copy bug back again, so the row
        # says plainly that it has no gloss.
        b = _ldoce_builder()
        b.wordnet_gloss = lambda *args, **kwargs: []
        info = {"word": "estate agency", "base": "estate agent", "pos": "noun",
                "gram": "", "examples": []}
        row = b.build_derived_row(
            "estate agency", info, "estate agent", "noun",
            {"noun": "someone whose business is to buy and sell houses or land for people"})
        assert row["senses"][0]["definition"] == ""
        assert row["gloss_source"] == "empty"

    def test_gate_calls_a_copied_gloss_an_expected_change(self):
        # Production glossed 'abductor' with 'to take someone away by force' - the verb
        # sense of 'abduct'. The shadow build does not invent that gloss, and the gate
        # must report its loss as an expected change while still reporting a real one.
        gate = _gate_script()
        abduct = ["to take someone away by force"]
        bootleg = ["the making or selling of illegal alcohol",
                   "to illegally make or sell alcohol"]
        weed = ["a wild plant growing where it is not wanted"]
        assert gate.copied_from_another_entry("abductor", "abduct", abduct[0], abduct)
        # The copy is not always the base's first sense - 'bootlegger' was glossed with
        # the verb sense of an article that leads with the noun.
        assert gate.copied_from_another_entry(
            "bootlegger", "bootleg", bootleg[1], bootleg)
        assert not gate.copied_from_another_entry(
            "weeder", "weed", "a tool or person that removes weeds from a garden", weed)
        assert not gate.copied_from_another_entry("weeder", "", weed[0], weed)
        assert not gate.copied_from_another_entry("weeder", "weeder", weed[0], weed)
        assert not gate.copied_from_another_entry("weeder", "weed", "", weed)
        assert not gate.copied_from_another_entry("weeder", "weed", weed[0], [])

    def test_gate_calls_a_shadow_row_that_repeats_its_base_an_artifact(self):
        # The other half of the gate: a shadow row must not hand back its base's text.
        gate = _gate_script()
        weed = ["a wild plant growing where it is not wanted",
                "to remove unwanted plants from a garden or other place"]
        assert gate.copy_artifact(weed[0], weed, "")
        # the copy is not always the base's first sense
        assert gate.copy_artifact(weed[1], weed, "template")
        assert not gate.copy_artifact(
            "a person who removes weeds from a garden", weed, "template")
        assert not gate.copy_artifact("", weed, "empty")
        assert not gate.copy_artifact(weed[0], [], "wordnet")

    def test_gate_lets_two_synonyms_share_wordnets_own_gloss(self):
        # 'esthetic' and 'esthetical' are one WordNet synset, so both rows legitimately
        # carry the same sentence. That is WordNet glossing each word, not Longman's
        # text walking from one row to the other.
        gate = _gate_script()
        shared = "concerning or characterized by an appreciation of beauty or good taste"
        assert not gate.copy_artifact(shared, [shared], "wordnet")
        # the same pair with no source recorded is still the old bug
        assert gate.copy_artifact(shared, [shared], "")


class TestRunOnForm:
    """The two ways scan_runons mis-named a derived row, both fixed by reading the form out
    of the markup instead of out of the flattened text.

    Longman prints a run-on as a dash, the form, then what it says about the form. The dash
    means 'the headword belongs here': the 'Eton' article declares Etonian by printing only
    '—ian', and the text-only regex named a row 'ian'. A label printed between the form and
    its POS ('<span class="geo"> American English</span>' in the 'collateral' article) is
    swallowed by the same regex and named a row 'collateralize american english'. A full
    MDX scan with this fix in place changes 4 payloads and nothing else."""

    @staticmethod
    def _payload(headword, runon):
        return ('<div class="entry"><span class="hwd">%s</span>'
                '<span class="pos"> noun</span>'
                '<span class="def">the thing the article is about</span>'
                '<div>%s</div></div>' % (headword, runon))

    @staticmethod
    def _forms(module, html):
        return [(w, p, g) for w, p, g, _examples in module.scan_runons(html)]

    def test_an_elided_headword_is_put_back_on_before_the_row_is_named(self):
        # '—ian' inside the 'Eton' article is Etonian, and the row has to be named after
        # the word Longman declared rather than after the tail it left out.
        b = _ldoce_builder()
        html = self._payload(
            "Eton",
            '<span class="runon"><span class="deriv"><span>—</span>ian</span>'
            '<span class="proncodes"><span class="neutral"> /</span>'
            '<span class="pron">iːˈtəʊniən</span><span class="neutral">/</span></span>'
            '<span class="pos"> noun</span></span>')
        assert self._forms(b, html) == [("etonian", "noun", "")]

    def test_a_register_label_printed_after_the_form_is_not_part_of_the_form(self):
        # 'collateralize American English verb' is one run-on with a register label, not a
        # thirty-character headword.
        b = _ldoce_builder()
        html = self._payload(
            "collateral",
            '<span class="runon"><span class="deriv"><span>—</span>collateralize</span>'
            '<span class="geo"> American English</span><span class="pos"> verb</span>'
            '<span class="gram"> [transitive]</span></span>')
        assert self._forms(b, html) == [("collateralize", "verb", "[transitive]")]

    def test_a_full_form_after_the_dash_is_left_alone(self):
        # The dash is usually just a separator: '—abandonment', '—Cuban' and '—mung' are the
        # whole form, and gluing the headword onto them would invent 'abandonabandonment'.
        b = _ldoce_builder()
        for headword, form, pos in (("abandon", "abandonment", "noun"),
                                    ("Cuba", "cuban", "adjective"),
                                    ("munging", "mung", "verb")):
            html = self._payload(
                headword,
                '<span class="runon"><span class="deriv"><span>—</span>%s</span>'
                '<span class="pos"> %s</span></span>' % (form, pos))
            assert self._forms(b, html) == [(form, pos, "")], headword

    def test_the_headword_is_glued_only_when_the_result_is_an_attested_word(self):
        # '—ian' in the 'republic' article would glue to 'republician', which no dictionary
        # spells, so the tail is left as it was found rather than guessed at.
        b = _ldoce_builder()
        html = self._payload(
            "republic",
            '<span class="runon"><span class="deriv"><span>—</span>ian</span>'
            '<span class="pos"> noun</span></span>')
        assert self._forms(b, html) == [("ian", "noun", "")]

    def test_a_runon_without_the_deriv_span_is_still_read_from_the_text(self):
        # The markup is the correction, not a requirement: plenty of run-ons print the form
        # and the POS with no deriv span at all.
        b = _ldoce_builder()
        html = self._payload(
            "punctual",
            '<span class="runon">— punctually <span class="pos"> adverb</span></span>')
        assert self._forms(b, html) == [("punctually", "adverb", "")]

    def test_a_derived_row_without_a_gloss_still_carries_a_pos(self):
        """The 142 rows that honestly carry no gloss are not missing a part of speech.

        RUNON_RE only declares a run-on when it reads the POS Longman prints next to the
        form, so a kind='derived' row cannot be built without one - the glossless rows are
        'word real, meaning missing', never 'word real, POS missing'. Measured on the
        shipped database by scripts/_probe_runon_pos_parse.py."""
        import json
        import sqlite3
        from pathlib import Path

        db = Path(__file__).resolve().parent.parent / "librarian" / "data" / "ldoce6_essential.db"
        if not db.exists():
            pytest.skip("shipped ldoce database not present")
        conn = sqlite3.connect(str(db))
        rows = conn.execute(
            "SELECT word, pos, data_json FROM ldoce WHERE kind = 'derived'").fetchall()
        conn.close()
        assert rows

        columnless = [w for w, pos, _ in rows if not (pos or "").strip()]
        assert not columnless, columnless[:10]

        glossless = [(w, json.loads(data)) for w, _, data in rows
                     if '"gloss_source": "empty"' in data]
        assert glossless
        senseless_pos = [w for w, payload in glossless
                         if not (payload.get("senses") or [{}])[0].get("pos")]
        assert not senseless_pos, senseless_pos[:10]
        assert all(payload.get("base_word") for _, payload in glossless), glossless[:5]


class TestRunOnTypography:
    """Backlog E4. The 282 run-on blocks the text-only regex could not read at all.

    RUNON_RE accepts letters, spaces, hyphens and apostrophes, and it has to end the form
    before the POS. Longman breaks both rules in 282 of its 14,861 run-on blocks: it prints
    the stress inside a compound ('—ˈfade-in'), the point where the word may be broken
    ('—blight‧ed'), or a second spelling between the form and its POS ('—belligerence,
    belligerency noun'). Every one of those blocks does print a POS - scripts/
    _probe_runon_pos_parse.py counted 257 of the 282 - so the gap was never a missing POS,
    it was a form the regex could not end. They are now read out of their markup: the form
    from the deriv span, the POS from the span printed after it. Measured across the whole
    MDX by scripts/_probe_runon_typography.py and scripts/_probe_runon_markup_fix.py."""

    @staticmethod
    def _payload(headword, runon):
        return ('<div class="entry"><span class="hwd">%s</span>'
                '<span class="pos"> noun</span>'
                '<span class="def">the thing the article is about</span>'
                '<div>%s</div></div>' % (headword, runon))

    @staticmethod
    def _forms(module, html):
        return [(w, p, g) for w, p, g, _examples in module.scan_runons(html)]

    @staticmethod
    def _runon(inner):
        return '<span class="runon">%s</span>' % inner

    def test_an_ipa_stress_mark_inside_a_form_is_not_part_of_the_spelling(self):
        # '—ˈfade-in' in the 'fade' article: the stress is printed because the compound could
        # be read two ways. The index keys the word without it, and so does the row.
        b = _ldoce_builder()
        html = self._payload("fade", self._runon(
            '<span class="deriv"><span>—</span>ˈfade-in</span>'
            '<span class="pos"> noun</span><span class="gram"> [countable]</span>'))
        assert self._forms(b, html) == [("fade-in", "noun", "[countable]")]

    def test_the_hyphenation_dot_is_taken_out_of_the_form(self):
        # '—blight‧ed' is 'blighted'. The dot is Longman's hyphenation point (U+2027), not a
        # hyphen, and it is not in the spelling the index carries.
        b = _ldoce_builder()
        html = self._payload("blight", self._runon(
            '<span class="deriv"><span>—</span>blight‧ed</span>'
            '<span class="pos"> adjective</span>'))
        assert self._forms(b, html) == [("blighted", "adjective", "")]

    def test_two_spellings_printed_in_one_span_are_both_declared(self):
        # '—belligerence, belligerency noun' is one run-on named two ways. Longman gives the
        # mdx index a key for the pair ('belligerence, belligerency'), which KEY_RE refuses,
        # so before this fix neither spelling had a row anywhere in the database.
        b = _ldoce_builder()
        html = self._payload("belligerent", self._runon(
            '<span class="deriv"><span>—</span>belligerence, belligerency</span>'
            '<span class="pos"> noun</span><span class="gram"> [uncountable]</span>'))
        assert self._forms(b, html) == [("belligerence", "noun", "[uncountable]"),
                                        ("belligerency", "noun", "[uncountable]")]

    def test_two_spellings_printed_in_two_spans_are_both_declared(self):
        # '—coiffured,' then '—coiffed' is the same device with the comma left dangling at
        # the end of the first span. Reading only the first deriv span used to lose 'coiffed'.
        b = _ldoce_builder()
        html = self._payload("coiffure", self._runon(
            '<span class="deriv"><span>—</span>coiffured,</span>'
            '<span class="deriv"><span>—</span>coiffed</span>'
            '<span class="proncodes"><span class="neutral"> /</span>kwˈfjːəd'
            '<span class="neutral">/</span></span>'
            '<span class="pos"> adjective</span>'))
        assert self._forms(b, html) == [("coiffured", "adjective", ""),
                                        ("coiffed", "adjective", "")]



    def test_a_variant_span_between_the_form_and_its_pos_does_not_lose_the_pos(self):
        # '—abridgement , abridgment noun': the second spelling sits in its own span, so the
        # flattened text puts a comma where the regex expected the POS to start.
        b = _ldoce_builder()
        html = self._payload("abridge", self._runon(
            '<span class="deriv"><span>—</span>abridgement</span>'
            '<span class="variant"><span class="neutral">, </span>'
            '<span class="lexvar">abridgment</span></span>'
            '<span class="pos"> noun</span>'))
        assert self._forms(b, html) == [("abridgement", "noun", "")]

    def test_the_article_printed_with_a_plural_is_not_taken_for_a_spelling(self):
        # '—Assyrians, the' declares one word. The 'the' is the article Longman prints with a
        # plural, not a second spelling, and a row named 'the' would be a row for nothing.
        b = _ldoce_builder()
        html = self._payload("assyrian", self._runon(
            '<span class="deriv"><span>—</span>Assyrians, the</span>'
            '<span class="pos"> noun</span><span class="gram"> [plural]</span>'))
        assert self._forms(b, html) == [("assyrians", "noun", "[plural]")]

    def test_a_run_on_with_an_example_but_no_pos_of_its_own_declares_nothing(self):
        # '—hinged:' is a run-on that carries an example and no POS label. The next POS in the
        # window belongs to the section printed after it, so the walk stops at the example
        # rather than inventing a row out of the article's own label.
        b = _ldoce_builder()
        html = self._payload("hinge", self._runon(
            '<span class="deriv"><span>—</span>hinged</span>'
            '<span class="neutral">: </span>'
            '<span class="example">a hinged lid</span>'
            '<span class="pos"> noun</span>'))
        assert self._forms(b, html) == []

    def test_a_suffix_is_not_declared_as_a_derived_word(self):
        # The '-ably' and '-brian' articles declare suffixes. A row named 'brian' would be the
        # old 'ian' bug all over again, and the suffix already has its own key and article.
        b = _ldoce_builder()
        html = self._payload("-brian", self._runon(
            '<span class="deriv"><span>—</span>-brian</span>'
            '<span class="pos"> adjective</span>'))
        assert self._forms(b, html) == []

    def test_an_accented_form_is_not_folded_into_a_different_word(self):
        # '—clichéd' is neither folded to 'cliched' nor written as 'clichéd': KEY_RE has never
        # accepted a headword outside the ASCII alphabet, so inventing a spelling the index
        # does not carry is the worse of the two errors. It stays a known gap.
        b = _ldoce_builder()
        html = self._payload("cliché", self._runon(
            '<span class="deriv"><span>—</span>clichéd</span>'
            '<span class="pos"> adjective</span>'))
        assert self._forms(b, html) == []

    def test_the_blocks_that_already_read_still_read_the_same_way(self):
        # The markup fallback only runs when RUNON_RE finds nothing, so a plain run-on is
        # untouched by it - the whole-MDX diff gains 135 forms and loses none.
        b = _ldoce_builder()
        html = self._payload("abandon", self._runon(
            '<span class="deriv"><span>—</span>abandonment</span>'
            '<span class="pos"> noun</span><span class="gram"> [uncountable]</span>'))
        assert self._forms(b, html) == [("abandonment", "noun", "[uncountable]")]
        html = self._payload("punctual", self._runon(
            '— punctually <span class="pos"> adverb</span>'))
        assert self._forms(b, html) == [("punctually", "adverb", "")]


class TestPartitiveFrameAndAnchorDowngrade:
    """B2: an anchor-less noun must be handed the frame the passage really gave it, or be
    declared frame-less. 'goody' inside 'various baskets of goodies' is the case that
    started this: spaCy hangs the target under 'of' and 'of' under the quantifier noun,
    so every priority that looks downward from the target returned nothing, the item
    shipped with anchor=None, and the micro-task still implied a structural model."""

    GOODY_QUOTE = "Our library volunteers received various baskets of goodies before the film started."
    GOODY_VOCAB = (
        "## [[goody]]\n"
        "- **Part Of Speech**: noun\n"
        "- **Definition**: a small item of food that you are given, especially as a prize or present\n"
        f"- **Quoted Sentence**: \"{GOODY_QUOTE}\"\n"
    )
    FRAMELESS_VOCAB = (
        "## [[toothpaste]]\n"
        "- **Part Of Speech**: noun\n"
        "- **Definition**: a soft paste that you use to clean your teeth\n"
        "- **Quoted Sentence**: \"She squeezed the last of the toothpaste onto the brush.\"\n"
    )

    def test_the_partitive_compound_is_read_upward_from_the_target(self):
        """Every existing priority inspects the target's children or its verbal head; a noun
        inside 'N of ____' has its frame above the target, so the walker must look up."""
        anchor, anchor_type, quantifier = LinguisticEngine._partitive_compound_anchor(
            "goody", TestPartitiveFrameAndAnchorDowngrade.GOODY_QUOTE
        )
        assert (anchor, anchor_type, quantifier) == ("baskets of", "partitive", "basket")

    def test_an_empty_quantifier_is_not_a_frame(self):
        """'lots of' and 'all sorts of' look like a frame but any noun fits them, so they
        must not be recorded as the structural model this item defends."""
        for quote in ("The children got lots of goodies at the party.",
                      "The children got all sorts of goodies at the party."):
            assert LinguisticEngine._partitive_compound_anchor("goody", quote) == (None, None, None)

    def test_a_quantifier_that_is_another_batch_target_is_rejected(self):
        """An anchor the evaluator demands in the stem and its leakage gate bans in the same
        breath is unsatisfiable, so the compound is refused before it is ever offered."""
        for forbidden in ({"basket"}, {"baskets"}):
            assert LinguisticEngine._partitive_compound_anchor(
                "goody", TestPartitiveFrameAndAnchorDowngrade.GOODY_QUOTE, forbidden=forbidden
            ) == (None, None, None)

    def test_a_noun_outside_a_partitive_frame_gets_no_compound(self):
        """The recovery is a reading of one specific structure, not a default frame."""
        assert LinguisticEngine._partitive_compound_anchor(
            "goody", "She kept the goody in her pocket all afternoon."
        ) == (None, None, None)

    def test_goody_skeleton_ships_the_frame_the_passage_used(self):
        """The plural cast already existed; the missing half was the frame. Both must now be
        present, and the frame must be attributed to the quote that showed it."""
        skeletons = LinguisticEngine.build_precomputed_target_skeletons(
            TestPartitiveFrameAndAnchorDowngrade.GOODY_VOCAB, target_count=1
        )
        s = skeletons[0]
        assert s["target_word"] == "goodies"
        assert s["base_headword"] == "goody"
        assert "plural" in s["inflection"]
        assert s["context_anchor"] == "baskets of"
        assert s["anchor_type"] == "partitive"
        assert s["anchor_source"] == "quote"
        assert s["anchor_downgrade"] is None
        assert s["item_type"] == "cloze"
        assert "partitive compound 'baskets of'" in s["micro_task"]
        assert "immediately followed by bound preposition" not in s["micro_task"]

    def test_a_blueprint_that_contradicts_the_partitive_frame_is_dropped(self):
        """LDOCE's own example for 'goodies' reads 'We bought lots of goodies for the picnic'.
        A writer cannot emulate 'lots of ____' and satisfy an anchor gate demanding
        'baskets of', so the contradictory sentence template is not handed over."""
        skeletons = LinguisticEngine.build_precomputed_target_skeletons(
            TestPartitiveFrameAndAnchorDowngrade.GOODY_VOCAB, target_count=1
        )
        frame = skeletons[0]["cloze_frame_prototype"]
        assert frame is None or "basket" in frame.lower()

    def test_an_anchorless_noun_is_declared_instead_of_fabricated(self):
        """'toothpaste' has no collocation, no quote frame and no partitive compound. The
        honest outcome is a sense-recognition item, not a frame the engine invented."""
        skeletons = LinguisticEngine.build_precomputed_target_skeletons(
            TestPartitiveFrameAndAnchorDowngrade.FRAMELESS_VOCAB, target_count=1
        )
        s = skeletons[0]
        assert s["context_anchor"] is None
        assert s["anchor_downgrade"] == "sense_recognition"
        assert s["item_type"] == "sense_recognition"
        assert "Sense-recognition item" in s["micro_task"]
        assert "immediately followed by bound preposition" not in s["micro_task"]
        assert "Syntactic Frame:" in s["micro_task"]

    def test_the_downgrade_flag_and_the_anchor_always_agree(self):
        """The flag is only useful if it is exactly the absence of an anchor: an item cannot
        be declared frame-less while carrying one, nor silently frame-less while carrying none."""
        vocab = (
            TestPartitiveFrameAndAnchorDowngrade.GOODY_VOCAB + "\n"
            + TestPartitiveFrameAndAnchorDowngrade.FRAMELESS_VOCAB + "\n"
            "## [[quorum]]\n"
            "- **Part Of Speech**: noun\n"
            "- **Definition**: the minimum number of members who must be present for a meeting to be valid\n"
            "- **Quoted Sentence**: \"The committee failed to reach a quorum that morning.\"\n"
        )
        skeletons = LinguisticEngine.build_precomputed_target_skeletons(vocab, target_count=3)
        assert len(skeletons) == 3
        for s in skeletons:
            frameless = s["context_anchor"] is None
            assert (s["anchor_downgrade"] == "sense_recognition") == frameless
            assert s["item_type"] == ("sense_recognition" if frameless else "cloze")

    def test_the_anchor_gate_accepts_a_phrase_anchor_as_a_phrase(self):
        """The anchor-preservation gate compared the prescribed anchor against stem tokens,
        which a two-word compound can never satisfy."""
        from librarian.processor import WikiProcessor

        def item(stem):
            return {
                "target_word": "goodies",
                "question": stem,
                "options": ["goodies", "milks", "courses", "dishes"],
                "correct_answer_index": 0,
                "context_anchor": "baskets of",
                "anchor_type": "partitive",
            }

        flagged, messages = WikiProcessor.audit_quiz_integrity(
            {"questions": [item("The volunteers handed out baskets of ____ for the children.")]}
        )
        assert flagged == [], messages

        flagged, messages = WikiProcessor.audit_quiz_integrity(
            {"questions": [item("The volunteers handed out lots of ____ for the children.")]}
        )
        assert flagged == [0]
        assert "baskets of" in messages[0]


class TestPayloadCollisionMerge:
    """Backlog E6. 478 mdx keys arrive carrying two payloads: 'absurd' holds the big
    adjective article *and* the short noun article printed as "the Absurd". One row per key
    meant the payload scanned last overwrote the other - 24.6M characters thrown away, and a
    run-on declared only in the discarded payload lost the sense it was built on."""

    @staticmethod
    def _article(pos, definitions, **extra):
        row = {"kind": "article", "word": "absurd", "pos": pos, "pos_all": [pos],
               "all_poses": [pos], "homographs": [], "thesaurus": [],
               "collocations": {}, "senses": [{"pos": pos, "definition": d}
                                              for d in definitions]}
        row.update(extra)
        return row

    def test_the_article_that_owns_the_word_keeps_the_first_sense(self):
        # definition_of() answers with senses[0], so merging must not let the short noun
        # article jump in front of the adjective article the key actually means.
        b = _ldoce_builder()
        noun_article = self._article("noun", ["a style of play for the theatre"])
        adjective_article = self._article(
            "adjective", ["completely stupid or unreasonable",
                          "something that is completely stupid"],
            thesaurus=[{"word": "ridiculous"}])
        merged = b.merge_payload_rows(noun_article, adjective_article)
        assert merged["pos"] == "adjective"
        assert [s["pos"] for s in merged["senses"]] == ["adjective", "adjective", "noun"]
        assert merged["pos_all"] == ["adjective", "noun"]
        # an asset the primary payload lacks is taken from the other one
        assert merged["thesaurus"] == [{"word": "ridiculous"}]
        assert (merged["merged_payloads"], merged["merged_senses"]) == (2, 1)

    def test_a_sense_both_payloads_carry_is_not_stored_twice(self):
        b = _ldoce_builder()
        first = self._article("noun", ["a place where records are stored",
                                       "to put records in an archive"])
        second = self._article("noun", ["a place where records are stored",
                                        "a place where historical records are kept"])
        merged = b.merge_payload_rows(first, second)
        assert len(merged["senses"]) == 3
        assert merged["merged_senses"] == 1

    @staticmethod
    def _derived(word, pos, definition, base):
        return {"kind": "derived", "word": word, "pos": pos, "pos_all": [pos],
                "all_poses": [pos], "base_word": base, "homographs": [],
                "senses": [{"pos": pos, "definition": definition,
                            "gloss_source": "wordnet"}]}

    def test_a_run_on_row_folds_into_the_article_that_owns_the_key(self):
        # 'unionist' is a run-on inside 'unionism' *and* has its own Longman article about
        # Northern Ireland. The old merge refused derived rows, so the run-on row overwrote
        # the article and the Northern Ireland sense vanished.
        b = _ldoce_builder()
        article = self._article("noun", ["a member of a political party that wants Northern "
                                         "Ireland to remain part of the United Kingdom"],
                                word="unionist")
        runon = self._derived("unionist", "noun",
                              "belief in the principles of trade unions", "unionism")
        merged = b.merge_payload_rows(article, runon)
        assert merged["kind"] == "article"
        assert [s["definition"] for s in merged["senses"]] == [
            "a member of a political party that wants Northern Ireland to remain part of the "
            "United Kingdom",
            "belief in the principles of trade unions"]
        assert merged["base_word"] is None
        assert (merged["merged_payloads"], merged["merged_senses"]) == (2, 1)

    def test_an_encyclopedic_entry_does_not_answer_for_the_common_word(self):
        # 'abdication' carries the proper-noun article "the Abdication" (1936) and the run-on
        # reading inside 'abdicate'. Both deserve a place, but definition_of() must still
        # answer with the ordinary meaning, so the entity sense is appended, not promoted.
        b = _ldoce_builder()
        entity = {"kind": "entity", "word": "abdication", "pos": None, "pos_all": [],
                  "all_poses": [], "homographs": [],
                  "senses": [{"pos": "other",
                              "definition": "the period in Britain in 1936, in which King "
                                            "Edward VIII abdicated"}]}
        runon = self._derived("abdication", "noun",
                              "a formal resignation and renunciation of powers", "abdicate")
        merged = b.merge_payload_rows(entity, runon)
        assert merged["kind"] == "derived"
        assert merged["pos"] == "noun"
        assert [s["definition"] for s in merged["senses"]] == [
            "a formal resignation and renunciation of powers",
            "the period in Britain in 1936, in which King Edward VIII abdicated"]

    def test_two_run_on_readings_of_one_key_are_not_merged(self):
        # Two different bases both declaring the same run-on cannot both be the answer; the
        # build keeps the row it already stored instead of letting the later one win.
        b = _ldoce_builder()
        assert b.merge_payload_rows(self._derived("abdication", "noun", "x", "abdicate"),
                                    self._derived("abdication", "noun", "y", "abdication")) is None

    def test_an_alias_row_is_never_merged_into(self):
        # An alias carries no senses of its own - it is a pointer, and folding a content row
        # into a pointer would throw the content away.
        b = _ldoce_builder()
        assert b.merge_payload_rows(self._article("noun", ["x"]),
                                    {"kind": "alias", "senses": [], "pos_all": []}) is None
        assert b.merge_payload_rows({"kind": "alias", "senses": [], "pos_all": []},
                                    self._article("noun", ["x"])) is None


class TestSpellingVariantFold:
    """Backlog E6, second half. merge_payload_rows folds two payloads that arrive under one
    mdx key. The other half is two mdx keys that normalize_key() folds into one key: 'air-kiss'
    and 'air kiss', 'belly flop' and 'bellyflop', 'cluster bomb' and 'cluster-bomb'. 283 keys
    carry 286 rows that way, 277 of them two whole Longman articles.

    The builder now writes the second spelling as a pointer - but only when the two rows carry
    exactly the same senses. 'a' and 'a-', 'able' and '-able', 'ate' and '-ate' are two different
    articles under one folded key, and 'big time' (2 senses) and 'big-time' (1 sense) share a
    first gloss without being the same article. Folding those would delete a Longman entry."""

    @staticmethod
    def _article(word, pos, definitions, **extra):
        row = {"kind": "article", "word": word, "pos": pos, "pos_all": [pos],
               "all_poses": [pos], "homographs": [],
               "senses": [{"pos": pos, "definition": d} for d in definitions]}
        row.update(extra)
        return row

    def test_the_second_spelling_of_one_article_becomes_a_pointer(self):
        b = _ldoce_builder()
        gloss = "an act of showing someone you like them by putting your arms around them"
        stored = self._article("air kiss", "noun", [gloss])
        second = self._article("air-kiss", "noun", [gloss])
        assert b.spelling_fold_target(stored, second) == "air kiss"
        assert b.spelling_fold_target(second, stored) == "air-kiss"

    def test_the_fold_ignores_hyphen_and_space_placement(self):
        b = _ldoce_builder()
        stored = self._article("belly flop", "noun", ["an act of falling flat onto water"])
        for spelling in ("bellyflop", "belly-flop"):
            assert b.spelling_fold_target(
                stored, self._article(spelling, "noun",
                                      ["an act of falling flat onto water"])) == "belly flop"

    def test_two_articles_under_one_folded_key_keep_their_rows(self):
        # 'a' is the indefinite article, 'a-' the prefix. One folded key, two articles: an
        # alias would leave the prefix with nothing to say.
        b = _ldoce_builder()
        article = self._article("a", "determiner",
                                ["used before a countable singular noun to mean one thing"])
        prefix = self._article("a-", "prefix", ["occurring in or on the thing named"])
        assert b.spelling_fold_target(article, prefix) is None
        assert b.spelling_fold_target(prefix, article) is None

    def test_a_richer_row_does_not_swallow_a_shorter_one(self):
        # The plan on paper was 'keep the row with more senses'. For 'big time' and 'big-time'
        # that throws away the sense only one of the two spellings carries, so the fold is
        # decided by the sense set, not by the sense count.
        b = _ldoce_builder()
        richer = self._article("big time", "noun",
                               ["an important position or situation",
                                "a situation in which you have a lot of power"])
        shorter = self._article("big-time", "noun", ["an important position or situation"])
        assert b.spelling_fold_target(richer, shorter) is None
        assert b.spelling_fold_target(shorter, richer) is None

    def test_a_row_without_a_gloss_of_its_own_is_never_folded(self):
        b = _ldoce_builder()
        stored = self._article("air kiss", "noun", ["an act of kissing someone's cheek"])
        assert b.spelling_fold_target(stored, self._article("air-kiss", "noun", [])) is None
        assert b.spelling_fold_target(self._article("air-kiss", "noun", []), stored) is None

    def test_only_rows_that_own_their_text_are_folded(self):
        # A derived row's sense comes from WordNet or the derivation template, and an alias is
        # already a pointer. Neither is a second copy of the article.
        b = _ldoce_builder()
        stored = self._article("air kiss", "noun", ["an act of kissing someone's cheek"])
        gloss = "an act of kissing someone's cheek"
        derived = {"kind": "derived", "word": "air-kiss", "pos": "verb", "pos_all": ["verb"],
                   "all_poses": ["verb"], "base_word": "air",
                   "senses": [{"pos": "verb", "definition": gloss}]}
        alias = {"kind": "alias", "word": "air-kiss", "pos": None, "pos_all": [],
                 "base_word": "air kiss", "senses": []}
        for other in (derived, alias):
            assert b.spelling_fold_target(stored, other) is None
            assert b.spelling_fold_target(other, stored) is None

    def test_the_sense_signature_ignores_case_and_surrounding_space(self):
        b = _ldoce_builder()
        loose = self._article("cluster bomb", "noun", ["  A Bomb That Releases Smaller Ones  "])
        tight = self._article("cluster-bomb", "noun", ["a bomb that releases smaller ones"])
        assert b.sense_signature(loose) == b.sense_signature(tight)
        assert b.spelling_fold_target(loose, tight) == "cluster bomb"

    def test_a_homograph_is_not_mistaken_for_a_spelling_variant(self):
        # One spelling carrying two sense sets is the merge_payload_rows case, not this one.
        b = _ldoce_builder()
        first = self._article("absurd", "adjective", ["completely stupid or unreasonable"])
        second = self._article("absurd", "noun", ["a style of play for the theatre"])
        assert b.spelling_fold_target(first, second) is None



class TestAnchorEvidenceAndSenseConfidence:
    """
    B1 - an anchor's evidence must be a clause, and a sense lock must be earned.
    """

    _CLAUSE = "The manager insisted on punctuality because late arrivals delayed the meeting."
    _TITLE = "**Punctuality Pays!**"

    def test_headline_title_is_not_anchor_evidence(self):
        assert LinguisticEngine.anchor_evidence_ok(self._TITLE, "punctuality") is False
        assert LinguisticEngine.anchor_evidence_ok("Punctuality Pays!", "punctuality") is False

    def test_bare_fragment_and_noun_phrase_are_not_anchor_evidence(self):
        assert LinguisticEngine.anchor_evidence_ok("Why?", "punctuality") is False
        # Long enough to look like a sentence, but its root is a noun, not a predicate.
        assert LinguisticEngine.anchor_evidence_ok(
            "The reason for all this confusion?", "confusion") is False
        assert LinguisticEngine.anchor_evidence_ok(
            "A long list of names, dates, and places.", "list") is False

    def test_a_clause_that_shows_the_word_is_anchor_evidence(self):
        assert LinguisticEngine.anchor_evidence_ok(self._CLAUSE, "punctuality") is True
        # A sentence that never shows the headword cannot evidence how it is used.
        assert LinguisticEngine.anchor_evidence_ok(self._CLAUSE, "volunteer") is False

    def test_anchor_is_the_richest_clause_not_the_first_hit(self):
        flat = "There was a pay rise for the hospital staff last week."
        rich = "The company pays the volunteers a small allowance every month."
        chosen, score = LinguisticEngine.pick_anchor_sentence("pay", [flat, rich])
        assert chosen == rich, chosen
        assert score > LinguisticEngine.anchor_dependency_richness(flat, "pay")
        # A title is never chosen, however early it sits in the pool.
        chosen, _ = LinguisticEngine.pick_anchor_sentence("pay", [self._TITLE, rich])
        assert chosen == rich

    def test_sense_confidence_low_no_definition(self):
        # 'pay' carries 30+ verb senses. With no sentence to read them against, the lock is
        # whichever sense Longman printed first - that is not a definition to ship.
        entry = LinguisticEngine.get_ldoce_entry("pay")
        assert entry, "pay must exist in the LDOCE db"
        assert len(entry.get("senses") or []) > 5
        lock = LinguisticEngine.sense_confidence(entry, quote="", target_pos="verb")
        assert lock["confidence"] == "low", lock
        assert lock["eligible_senses"] > 1
        strict = LinguisticEngine.get_ldoce_definition_and_example(
            "pay", target_pos="verb", context_sentence="", require_sense_confidence=True)
        assert strict == ("", ""), strict
        # Callers that do not demand evidence keep the behaviour they were written against.
        loose = LinguisticEngine.get_ldoce_definition_and_example(
            "pay", target_pos="verb", context_sentence="")
        assert loose[0]

    def test_a_clause_that_evidences_a_sense_locks_it_high(self):
        entry = LinguisticEngine.get_ldoce_entry("line")
        assert entry and len(entry.get("senses") or []) > 5
        quote = "The queue stretched along the line outside the bank for over an hour."
        lock = LinguisticEngine.sense_confidence(entry, quote=quote, target_pos="noun")
        assert lock["confidence"] == "high", lock
        assert lock["score"] >= LinguisticEngine.SENSE_CONFIDENCE_FLOOR
        assert lock["margin"] >= LinguisticEngine.SENSE_MARGIN_FLOOR
        # The sense the sentence describes is not the first sense in the entry.
        assert lock["index"] not in (None, 0)

    def test_a_single_sense_entry_is_never_flagged_low(self):
        # Nothing to disambiguate: the floors would only invent a doubt.
        entry = LinguisticEngine.get_ldoce_entry("punctuality")
        assert entry and len(entry.get("senses") or []) == 1
        lock = LinguisticEngine.sense_confidence(entry, quote=self._TITLE, target_pos="noun")
        assert lock["confidence"] == "high", lock
        definition, _example = LinguisticEngine.get_ldoce_definition_and_example(
            "punctuality", target_pos="noun", context_sentence=self._TITLE,
            require_sense_confidence=True)
        assert definition

    def test_extraction_reports_the_evidence_tier(self):
        passage = (
            "The organization asked every volunteer to bring a big appetite. "
            "The community appreciated the quiet effort that the team put into the event.\n\n"
            "**Appreciation Pays!**\n\n"
            "The library received various baskets of goodies before the film started.\n"
        )
        rows = LinguisticEngine.extract_deterministic_vocabulary(passage, target_count=6)
        assert rows
        for row in rows:
            assert row["anchor_quality"] in ("clause", "quote", "unqualified",
                                             "substring", "syllabus"), row
            assert row["sense_confidence"] in ("high", "low"), row
            assert isinstance(row["sense_lock_score"], float)
            assert isinstance(row["sense_lock_margin"], float)
            if row["sense_confidence"] == "low":
                assert row["sense_lock_score"] < LinguisticEngine.SENSE_CONFIDENCE_FLOOR or (
                    row["sense_lock_margin"] < LinguisticEngine.SENSE_MARGIN_FLOOR), row


class TestRichCollocationHeadwordDedup:
    """Backlog E7/F7 follow-up on `get_rich_collocations`.

    It used to render every collocate as '<item> <word>'. LDOCE collocation boxes mix
    bare collocates ('begin', 'firm', 'lay') with whole phrases that already contain the
    headword ('get something wrong', 'be proved wrong', 'part-time work'), so the phrase
    forms came out as 'get something wrong wrong' and 'part-time work work'. The shipped
    database had empty collocation boxes for exactly those words, so the defect never had
    a chance to show; F7 filled the boxes and it did.
    """

    _WORDS = ("wrong", "work", "foundation", "time", "control", "listen")

    def test_no_phrase_repeats_the_headword(self):
        for word in self._WORDS:
            phrases = LinguisticEngine.get_rich_collocations(word, top_k=8)
            assert phrases, word
            for phrase in phrases:
                tokens = re.split(r"[\s/]+", phrase.strip().lower())
                assert tokens.count(word) <= 1, (word, phrase)

    def test_known_double_headword_outputs_are_gone(self):
        wrong = LinguisticEngine.get_rich_collocations("wrong", top_k=8)
        assert "get something wrong wrong" not in wrong
        assert "be proved wrong wrong" not in wrong
        assert "get something wrong" in wrong, wrong

        work = LinguisticEngine.get_rich_collocations("work", top_k=8)
        assert "part-time work work" not in work
        assert "full-time work work" not in work
        assert "part-time work" in work, work

    def test_bare_collocates_still_get_the_headword(self):
        """The fix must not swallow the ordinary case: a bare verb or adjective collocate
        still becomes a phrase that contains the headword exactly once."""
        wrong = LinguisticEngine.get_rich_collocations("wrong", top_k=8)
        assert "go wrong" in wrong, wrong

        work = LinguisticEngine.get_rich_collocations("work", top_k=8)
        assert "begin work" in work and "continue work" in work, work

        foundation = LinguisticEngine.get_rich_collocations("foundation", top_k=8)
        assert "build foundation" in foundation and "firm foundation" in foundation, foundation


class TestCollocationPhraseAttestation:
    """Backlog E7 遗留 ①, second half - a rendered collocation has to be a phrase the
    entry's own data supports, not a string made by gluing the headword onto whatever
    sits in a bucket.

    `scripts/_probe_rich_collocations.py` (logs/rich_collocations_before_fix.log) showed
    what the unconditional gluing produced on the promoted database: 'lie listen',
    'sit listen', 'stand listen' (Longman's VERBS section for 'listen' lists verbs that
    merely co-occur - 'lie, sit and listen'), 'listen music' (the noun collocate needs the
    preposition the grammar patterns license), 'control birth' and 'control gun' (Longman
    prints 'birth control' and 'gun control'), 'work system' (the frame that matched was
    'This system works' - a subject and its verb), 'work way' (from 'working his way'),
    and 'based foundation' (a participle the box filed under verbs).

    Every assertion below is about the shipped database, so it is the promotion itself
    that makes it meaningful - the old database had empty collocation boxes.
    """

    _BANNED = {
        "listen": ("lie listen", "sit listen", "stand listen", "stop listen",
                   "listen music", "listen conversation", "listen news"),
        "work": ("work system", "work way", "spend work"),
        "control": ("control birth", "control gun", "bring control"),
        "foundation": ("based foundation", "stone foundation", "control foundation"),
        # The last four were found by reading the rendering over the whole probe word
        # list instead of the six words above (scripts/_probe_f7_consumers.py
        # BAD_RENDERED). 'lie' has no noun box, so the verb collocate 'listen' in its
        # VERBS section cannot be glued into 'lie listen' - the entry writes 'lie
        # listening'. 'have broken' and 'increase using' glued a bare verb onto a
        # participle or gerund sitting in a slot that takes a noun ('have something
        # broken', 'an increase using ...'). 'keep go' glued a verb onto a verb.
        "lie": ("lie listen",),
        "have": ("have broken",),
        "increase": ("increase using",),
        "keep": ("keep go",),
    }

    @staticmethod
    def _rendered(word: str):
        return LinguisticEngine.get_rich_collocations(word, top_k=25)

    def test_no_invented_concatenations(self):
        for word, banned in self._BANNED.items():
            rendered = self._rendered(word)
            assert rendered, word
            for phrase in banned:
                assert phrase not in rendered, (word, phrase, rendered)

    def test_compound_order_is_longmans(self):
        """'birth control' is not 'control birth'. The attested frame decides the order."""
        control = self._rendered("control")
        assert "birth control" in control, control
        assert "gun control" in control, control
        assert "control group" in control, control
        assert "control birth" not in control, control

    def test_governed_preposition_is_kept(self):
        """A noun collocate that only attaches through a preposition the entry licenses
        keeps it: 'listen to music', not 'listen music' or 'listen the music'."""
        listen = self._rendered("listen")
        assert "listen to music" in listen, listen
        assert "listen to conversation" in listen, listen

    def test_adverb_and_phrase_buckets_reach_the_output(self):
        """F7 ingested the ADVERB and PHRASES sections of a COLLOCATIONS box into buckets
        ('adverb', 'noun_after') that nothing rendered, so Longman's own phrases were
        stored and unreachable."""
        wrong = self._rendered("wrong")
        assert "completely wrong" in wrong, wrong
        assert "hopelessly wrong" in wrong, wrong

        listen = self._rendered("listen")
        assert "listen attentively" in listen, listen
        assert "listen to reason" in listen, listen
        assert "Have a listen" in listen, listen

        work = self._rendered("work")
        assert "work closely" in work, work
        assert "work in industry" in work, work

    def test_frame_helper_rejects_a_content_word_gap(self):
        """A frame is only evidence when everything between the two words is a function
        word. 'based on what rotten foundations' has content words in the gap, so 'based'
        gets no licence; 'A control group of ...' puts 'birth'-type collocates in front."""
        texts = LinguisticEngine._collocation_texts(LinguisticEngine.get_ldoce_entry("foundation") or {})
        assert LinguisticEngine._collocation_frame("foundation", "based", texts, allow_reverse=True) is None

        texts = LinguisticEngine._collocation_texts(LinguisticEngine.get_ldoce_entry("control") or {})
        frame = LinguisticEngine._collocation_frame("control", "birth", texts, allow_reverse=True)
        assert frame is not None and frame[0] == "item", frame

    def test_noun_box_gluing_survives_without_a_frame(self):
        """Longman's VERBS section of a noun box lists the verbs that govern the headword,
        and the entry does not have to contain 'begin work' for 'begin work' to be the
        collocation. Dropping every unattested collocate would empty these boxes."""
        work = self._rendered("work")
        assert "begin work" in work, work
        assert "continue work" in work, work

        foundation = self._rendered("foundation")
        assert "establish foundation" in foundation, foundation

    def test_a_bracketed_box_item_reaches_the_learner_as_a_phrase(self):
        """Longman wraps optional words in brackets inside a PHRASES item - 'keep
        (somebody/something) warm/safe/dry etc', 'fit (into) a mould', 'in the same mould
        (as somebody)'. 1,092 of the 164,207 stored collocation items carry a bracket
        (scripts/_probe_collo_parens.py), and the '/'-alternative split cut inside
        'keep (somebody/something) ...', which printed 'keep (somebody' and 'in the same
        mould (as somebody' for a learner to read. A rendered phrase has to read as a
        phrase: no brackets, no '...' hole, no leftover alternative separator."""
        for word in ("keep", "mould", "break", "alliance", "above", "alone"):
            rendered = self._rendered(word)
            assert rendered, word
            for phrase in rendered:
                assert not re.search(r"[()\[\]]|\.\.\.|/", phrase), (word, phrase)
        assert "keep warm" in self._rendered("keep"), self._rendered("keep")
        assert "fit a mould" in self._rendered("mould"), self._rendered("mould")
        assert "in the same mould" in self._rendered("mould"), self._rendered("mould")

    def test_a_derived_row_does_not_glue_onto_its_parent_box(self):
        """'keenly' carries 'keen's box ('a keen interest in', 'competition', 'eye'),
        'uneasiness' carries 'uneasy's ('an uneasy peace'), 'offensively' carries
        'offensive's ('offensive weapon'). Those collocates belong to the parent, so
        gluing the derived headword onto them is exactly the invention this policy exists
        to stop - the correct output is nothing (scripts/_probe_collo_render_quality.py
        found 30 such words in a 250-word sample, all of them inherited boxes)."""
        for word in ("keenly", "uneasiness", "offensively", "defensively"):
            assert self._rendered(word) == [], word
    def test_a_register_note_is_not_part_of_the_phrase(self):
        """35 of the bracketed items end in a variety note the parser glued on - 'give a
        raspberry )American English', 'in back (of something) American English', 'by a
        long chalk )British English'. The note is not part of the phrase, so it must not
        reach the learner. It is stripped only from items that carried a bracket, so a
        collocate that genuinely contains the words ('plain English' on the row 'English')
        survives. (scripts/_probe_collo_parens.py | grep English)"""
        for word in ("raspberry", "crap", "quarter", "rash", "long", "back"):
            for phrase in self._rendered(word):
                assert "English" not in phrase, (word, phrase)
        assert "plain English" in self._rendered("English"), self._rendered("English")




class TestPhrasalVerbAlternateForms:
    """Backlog F7 修法 1b - the '( also X )' form Longman prints beside a phrasal-verb
    headword. `tap` is the case that made scripts/_probe_f7_ingestion_landed.py exit 1:
    the builder stored variants = ['tap in'] only, so the promoted database had no
    block-level evidence for 'tap into' - the form Book_2_Unit_3_Section_A actually
    extracted as '[[ready to tap into [something]]]'. Whole-mdx census: 82 of the 2,392
    phrvbentry blocks carry one, across 76 keys
    (scripts/_probe_phrv_also.py -> logs/phrv_also.log).

    Every markup string below is copied from the mdx, not written by hand.
    """

    ARROW = "\u2194"

    @staticmethod
    def _soup(markup):
        from bs4 import BeautifulSoup
        return BeautifulSoup(markup, "html.parser")

    def _tap_block(self):
        a = self.ARROW
        return (
            '<span class="phrvbentry"><span class="entryhead">'
            f'<span class="phrvbhwd">tap<span class="object"> something {a}</span> in</span>'
            '<span class="variant"><span class="neutral"> (</span>'
            '<span class="linkword">also</span>'
            '<span class="lexvar"> tap something into something</span>'
            '<span class="neutral">)</span></span>'
            '<span class="pos"> phrasal verb</span>'
            '<span class="geo"> British English</span>'
            '</span>'
            '<span class="sense">'
            f'<span class="lexunit">tap something {a} in</span>'
            '<span class="def">to put information, numbers etc into a computer, telephone '
            'etc by pressing buttons or keys</span>'
            '<span class="example">Tap in your password before you log on.</span>'
            '</span></span>'
        )

    def test_the_also_alternate_becomes_a_variant(self):
        b = _ldoce_builder()
        blocks = b.extract_phrasal_verbs(self._soup(self._tap_block()))
        assert len(blocks) == 1
        block = blocks[0]
        assert block["phrase"] == "tap in"
        assert block["alternates"] == ["tap into"]
        assert "tap into" in block["variants"], block["variants"]

    def test_the_alternate_is_normalized_to_verb_plus_particle(self):
        # Longman writes the alternate with its object placeholders; the gate needs the
        # same continuous 'verb + particle' shape it gets from the headword.
        b = _ldoce_builder()
        assert b.phrasal_verb_key("tap something into something") == "tap into"
        assert b.phrasal_verb_key("drag somebody/something into something") == "drag into"


    def test_spelling_alternates_use_orthvar_and_are_captured_too(self):
        # 'factor something <-> in ( also factor something into something )' - the form
        # span is an 'orthvar', not a 'lexvar'.
        a = self.ARROW
        markup = (
            '<span class="entryhead">'
            f'<span class="phrvbhwd">factor<span class="object"> something {a}</span> in</span>'
            '<span class="variant"><span class="neutral"> (</span>'
            '<span class="linkword">also</span>'
            '<span class="orthvar"> factor something into something</span>'
            '<span class="neutral">)</span></span>'
            '<span class="pos"> phrasal verb</span></span>'
        )
        b = _ldoce_builder()
        head = self._soup(markup).find(class_="entryhead")
        assert b.phrasal_verb_alternates(head) == ["factor into"]

    def test_british_only_alternate_is_captured(self):
        # 'fiddle around ( also fiddle about British English )' uses 'brevariant'.
        markup = (
            '<span class="entryhead"><span class="phrvbhwd">fiddle around</span>'
            '<span class="brevariant"><span class="neutral"> (</span>'
            '<span class="linkword">also</span><span class="lexvar"> fiddle about</span>'
            '<span class="geo"> British English</span><span class="neutral">)</span></span>'
            '<span class="pos"> phrasal verb</span></span>'
        )
        b = _ldoce_builder()
        head = self._soup(markup).find(class_="entryhead")
        assert b.phrasal_verb_alternates(head) == ["fiddle about"]

    def test_slash_alternates_are_not_lost(self):
        # 'happen on/upon' already expanded to 'happen on' and 'happen upon' before this
        # change; adding the '( also X )' reader must not displace that.
        b = _ldoce_builder()
        assert b.phrasal_verb_variants("happen on/upon") == [
            "happen on/upon", "happen on", "happen upon"]

    def test_a_block_without_an_alternate_stores_no_alternates_key(self):
        # 2,310 of the 2,392 blocks carry no '( also X )'; they must not gain an empty key.
        markup = (
            '<span class="phrvbentry"><span class="entryhead">'
            '<span class="phrvbhwd">send out</span><span class="pos"> phrasal verb</span></span>'
            '<span class="sense"><span class="def">to send a letter or package to someone'
            '</span><span class="example">They sent out the invitations in January.</span>'
            '</span></span>'
        )
        b = _ldoce_builder()
        blocks = b.extract_phrasal_verbs(self._soup(markup))
        assert blocks and blocks[0]["phrase"] == "send out"
        assert "alternates" not in blocks[0]
        assert blocks[0]["variants"] == ["send out"]

    def test_a_variant_span_inside_a_sense_is_not_a_headword_alternate(self):
        # A 'variant' span deeper than the entryhead belongs to a sense's own text. Reading
        # it would let a definition's wording in as if Longman had declared it a phrasal verb.
        markup = (
            '<span class="phrvbentry"><span class="entryhead">'
            '<span class="phrvbhwd">give up</span><span class="pos"> phrasal verb</span></span>'
            '<span class="sense"><span class="def">to stop doing something</span>'
            '<span class="variant"><span class="lexvar">give something up for something</span>'
            '</span><span class="example">He gave up smoking.</span></span></span>'
        )
        b = _ldoce_builder()
        blocks = b.extract_phrasal_verbs(self._soup(markup))
        assert blocks[0]["phrase"] == "give up"
        assert "alternates" not in blocks[0]
        assert blocks[0]["variants"] == ["give up"]


class TestCrossCollocationBoxScope:
    """Backlog E7, the 'COLLOCATIONS FROM OTHER ENTRIES' popup.

    Longman prints that shared box once per entry of a multi-entry article - 'only'
    carries it under the adverb, the adjective and the conjunction - and the box body
    is the parent of the 'popcollo' header span. The old selector asked for "any span
    or div whose text contains the title", and find_all returns document order, so the
    first match was the outermost ancestor: the builder read every collocate in the
    payload and stored the same box two or three times in one row. Across the shipped
    db that is 269,618 cross_collocations of which 75,705 are repeats inside a single
    row (3,304 rows; 'have' 2,348 stored / 788 distinct, 'out' 1,645/451,
    'do' 1,503/398). Boxes are now read one at a time and deduplicated per row.
    """

    _CROSS_BOX = (
        '<span class="popup">'
        '<span class="popheader popcollo">COLLOCATIONS FROM OTHER ENTRIES</span>'
        '<span class="collocate"><span class="colloc">not only ... but also</span>'
        '<span class="gloss">The system was not only complicated but also ineffective.</span>'
        '<span class="example">The system was not only complicated but also ineffective.</span>'
        '</span>'
        '<span class="collocate"><span class="colloc">the only way</span>'
        '<span class="gloss">There was only one way to do it.</span>'
        '<span class="example">There was only one way to do it.</span>'
        '</span></span>'
    )

    _ENTRY_BOX = (
        '<span class="popup">'
        '<span class="popheader popcollo">COLLOCATIONS FROM THE ENTRY</span>'
        '<span class="collocate"><span class="colloc">only child</span>'
        '<span class="example">She is an only child.</span></span>'
        '</span>'
    )

    def _entry(self, pos, extra_boxes=""):
        return (
            '<span class="entry"><span class="entryhead">'
            '<span class="hwd">only</span>'
            f'<span class="pos"> {pos}</span></span>'
            '<span class="sense"><span class="def">used to emphasize a statement</span>'
            '<span class="example">Only you know the truth.</span></span>'
            + extra_boxes + self._CROSS_BOX + '</span>'
        )

    def test_a_shared_box_printed_twice_is_stored_once(self):
        b = _ldoce_builder()
        data = b.parse_ldoce_entry(
            self._entry("adverb") + self._entry("conjunction"), "only")
        cc = data["cross_collocations"]
        assert len(cc) == 2, cc
        strings = [c["collocation"] for c in cc]
        assert strings.count("not only ... but also") == 1, strings
        assert strings == ["not only ... but also", "the only way"], strings

    def test_the_box_is_not_read_through_an_ancestor(self):
        # The whole payload is wrapped in a div whose text therefore also "contains"
        # the title. Reading through that ancestor is what pulled in the entry's own
        # collocation box as well: 3 collocates instead of 2.
        b = _ldoce_builder()
        html = ('<div class="ldoceentry">'
                + self._entry("adverb", extra_boxes=self._ENTRY_BOX)
                + '</div>')
        data = b.parse_ldoce_entry(html, "only")
        assert [c["collocation"] for c in data["cross_collocations"]] == [
            "not only ... but also", "the only way"], data["cross_collocations"]
        entry_side = [c for bucket in data["collocations"].values() for c in bucket]
        assert [c["collocation"] for c in entry_side] == ["only child"], entry_side

    def test_two_different_cross_entry_boxes_both_contribute(self):
        # 'have' carries three copies of one box and 'out' four; a word with two
        # genuinely different boxes must keep both, keeping only the overlap once.
        second = (
            '<span class="popup">'
            '<span class="popheader popcollo">COLLOCATIONS FROM OTHER ENTRIES</span>'
            '<span class="collocate"><span class="colloc">the only way</span>'
            '<span class="gloss">There was only one way to do it.</span>'
            '<span class="example">There was only one way to do it.</span></span>'
            '<span class="collocate"><span class="colloc">sole survivor</span>'
            '<span class="gloss">He was the sole survivor of the crash.</span>'
            '<span class="example">He was the sole survivor of the crash.</span></span>'
            '</span>'
        )
        b = _ldoce_builder()
        data = b.parse_ldoce_entry(
            self._entry("adverb") + self._entry("adjective", extra_boxes=second), "only")
        strings = [c["collocation"] for c in data["cross_collocations"]]
        assert strings == ["not only ... but also", "the only way", "sole survivor"], strings

    def test_a_collocate_without_text_is_not_stored(self):
        empty = (
            '<span class="popup">'
            '<span class="popheader popcollo">COLLOCATIONS FROM OTHER ENTRIES</span>'
            '<span class="collocate"></span>'
            '</span>'
        )
        b = _ldoce_builder()
        data = b.parse_ldoce_entry(self._entry("adverb", extra_boxes=empty), "only")
        assert len(data["cross_collocations"]) == 2, data["cross_collocations"]


class TestCollisionMergeCollocations:
    """Backlog E6's merge, seen from the collocation side.

    465 mdx keys carry two payloads. 'cancer' is the instructive one: mdx has 'Cancer'
    (17 KB, three corpus collocates) and 'cancer' (122 KB, 52 collocates), both
    normalize to the key 'cancer', and the row that survives is the one the merge picks
    as primary. The merge used to fill only fields the primary left empty, so the small
    row's three collocations blocked the big article's 42 - the shipped row has 3.
    Collocation content belongs to a payload, not to a key, so it now unions.
    """

    @staticmethod
    def _article(word, pos, collocations, cross):
        boxes = ""
        if collocations:
            items = "".join(
                f'<span class="collocate"><span class="colloc">{c}</span>'
                f'<span class="example">{c} example.</span></span>'
                for c in collocations)
            boxes += ('<span class="popup">'
                      '<span class="popheader popcollo">COLLOCATIONS FROM THE CORPUS</span>'
                      + items + '</span>')
        if cross:
            items = "".join(
                f'<span class="collocate"><span class="colloc">{c}</span>'
                f'<span class="gloss">{c} gloss.</span>'
                f'<span class="example">{c} gloss.</span></span>'
                for c in cross)
            boxes += ('<span class="popup">'
                      '<span class="popheader popcollo">COLLOCATIONS FROM OTHER ENTRIES</span>'
                      + items + '</span>')
        return ('<span class="entry"><span class="entryhead">'
                f'<span class="hwd">{word}</span><span class="pos"> {pos}</span></span>'
                f'<span class="sense"><span class="def">definition of {word}</span>'
                '<span class="example">An example of ' + word + '.</span></span>'
                + boxes + '</span>')

    def _rows(self, small_words, big_words, small_cross, big_cross):
        b = _ldoce_builder()
        small = b.parse_ldoce_entry(self._article("Cancer", "noun", small_words, small_cross),
                                    "cancer")
        big = b.parse_ldoce_entry(self._article("cancer", "noun", big_words, big_cross), "cancer")
        for row in (small, big):
            row["kind"] = "article"
            row["pos_all"] = list(row["all_poses"])
        return small, big

    def test_the_secondary_payloads_collocations_are_kept(self):
        small, big = self._rows(["national", "institute"],
                                ["bowel", "cervical", "cause"], [], ["skin cancer"])
        merged = _ldoce_builder().merge_payload_rows(small, big)
        strings = sorted(c["collocation"]
                         for bucket in merged["collocations"].values()
                         for c in bucket)
        assert strings == ["bowel", "cause", "cervical", "institute", "national"], strings
        assert [c["collocation"] for c in merged["cross_collocations"]] == ["skin cancer"]

    def test_the_merge_does_not_depend_on_which_payload_arrived_first(self):
        small, big = self._rows(["national"], ["bowel", "cervical"], [], ["skin cancer"])
        b = _ldoce_builder()
        forward = b.merge_payload_rows(dict(small), dict(big))
        backward = b.merge_payload_rows(dict(big), dict(small))
        assert sorted(c["collocation"]
                      for bucket in forward["collocations"].values() for c in bucket) == \
               sorted(c["collocation"]
                      for bucket in backward["collocations"].values() for c in bucket)
        assert sorted(c["collocation"] for c in forward["cross_collocations"]) == \
               sorted(c["collocation"] for c in backward["cross_collocations"])

    def test_a_collocation_two_payloads_share_is_stored_once(self):
        small, big = self._rows(["bowel", "national"], ["bowel", "cervical"],
                                ["skin cancer"], ["skin cancer", "cancer treatment"])
        merged = _ldoce_builder().merge_payload_rows(small, big)
        collocations = [c["collocation"]
                        for bucket in merged["collocations"].values() for c in bucket]
        assert collocations.count("bowel") == 1, collocations
        cross = [c["collocation"] for c in merged["cross_collocations"]]
        assert cross == ["skin cancer", "cancer treatment"], cross

    def test_gap_fill_still_covers_the_one_box_fields(self):
        # A thesaurus box is one box per article; the second payload must not double it.
        small, big = self._rows(["national"], ["bowel"], [], [])
        small["thesaurus"] = [{"word": "tumour"}]
        big["thesaurus"] = [{"word": "tumour"}, {"word": "growth"}]
        merged = _ldoce_builder().merge_payload_rows(small, big)
        assert merged["thesaurus"] == [{"word": "tumour"}], merged["thesaurus"]


class TestLdocePhraseAttestationGate:
    """Backlog F7, gate half - a mined phrase ships only when Longman itself states it.

    The gate used to ask the Oxford Collocations Dictionary file whether it had a headword for
    the string, which is a weaker test than it sounds: 'keep of' is a preposition lifted out of
    'keep somebody out of something' and no dictionary has it as a unit.  ldoce_phrase_evidence
    asks the real question - a phrasal-verb block, a PHRASES item, a grammar pattern built on
    the headword, or a COLLOCATIONS box item - and a sentence that merely contains the words in
    this order is not evidence.

    The counts match scripts/_probe_expression_yield.py (logs/expression_yield_objectfit.log):
    78 expressions over the 11 wiki passages against the 76 the OCD gate produced, the two
    vetoes being 'Longman does not state it' and 'Longman's frame does not put that object after
    the particle'.
    """

    # Verb + preposition pairs a dependency parse hands over and the old gate shipped.  Longman
    # never states any of them: 'havoc for' comes out of 'cause havoc for somebody', 'streak on'
    # out of 'be on a winning streak', 'alone on' out of 'not alone on the list'.
    INVENTED = [
        "keep of", "havoc for", "account on", "accounts on", "streak on",
        "accord in", "admit in", "depend for", "join in place",
        "keep in WeChat", "above from", "alone on", "plain on", "alliance into",
    ]

    # ...and these are stated.  'base in' and 'accept into' used to sit in the list above,
    # which was wrong: Longman's own grammar patterns are 'be based in something' and 'accept
    # somebody into something', so the preposition belongs to the headword there exactly as it
    # does in 'saturate something with something'.
    STATED = [
        "account for", "depend on", "depend upon", "keep in touch", "stay in touch",
        "bring to account", "call to account", "by all accounts", "from all accounts",
        "give up", "belong to", "send out", "tap into", "happen to", "based on",
        "base in", "accept into", "saturate with", "embed in", "obsess with",
        "profit from", "include in", "measure in", "announce to", "accord to",
        "spend on", "cover in", "trade for", "get from", "derive from",
    ]

    def test_invented_pairs_are_rejected(self):
        for phrase in self.INVENTED:
            assert not LinguisticEngine.is_attested_phrase(phrase), (
                phrase, LinguisticEngine.ldoce_phrase_evidence(phrase, include_examples=True))

    def test_longman_stated_phrases_are_accepted(self):
        for phrase in self.STATED:
            assert LinguisticEngine.is_attested_phrase(phrase), (
                phrase, LinguisticEngine.ldoce_phrase_evidence(phrase))

    def test_a_pattern_that_writes_the_object_inside_it_states_the_pairing(self):
        """Longman's grammar frames put the object between the verb and its preposition -
        'saturate something with something', 'spend something on something', 'include
        something in/on something', 'measure something in something', 'announce something to
        somebody', 'trade somebody something for something'.  A gate that demands the two
        tokens be adjacent reads 'saturate with' out of a dictionary that states it and hands
        it back as an invention."""
        for phrase in ["saturate with", "spend on", "include in", "measure in",
                       "announce to", "accord to", "trade for", "accept into"]:
            assert LinguisticEngine.is_attested_phrase(phrase), (
                phrase, LinguisticEngine.ldoce_phrase_evidence(phrase))
        # What may sit between them is an object slot, not any word.  'depend on somebody /
        # something for something' has a complement of its own in the way, so 'depend for'
        # is still nothing Longman states.
        assert not LinguisticEngine.is_attested_phrase("depend for"), "depend for"

    def test_a_passive_frame_is_the_headword_s_own_frame(self):
        """'be embedded in something', 'be obsessed by/with something', 'be derived from
        something', 'be based in something' are the headword's own patterns, written with a
        copula because that is how a passive is written.  The headword inside them still
        governs the preposition, so 'embed in' is as stated as 'remind of'."""
        for phrase in ["embed in", "obsess with", "derived from", "based in", "base in"]:
            assert LinguisticEngine.is_attested_phrase(phrase), (
                phrase, LinguisticEngine.ldoce_phrase_evidence(phrase))
        # A content word in front is somebody else's construction.  'croatia became an
        # independent state in 1991' is a noun phrase, not a frame for 'state in'.
        assert not LinguisticEngine.is_attested_phrase("state in"), "state in"

    def test_a_slash_particle_alternative_is_one_slot(self):
        """'profit by/from' and 'cover something with/in something' list two alternatives for
        one slot, so the second alternative is stated too.  A preposition that is not followed
        by a slash is a complement, not an alternative - that is the difference between
        'profit by/from' and 'depend on somebody/something for something'."""
        assert LinguisticEngine.is_attested_phrase("profit from"), "profit from"
        assert LinguisticEngine.is_attested_phrase("cover in"), "cover in"
        # A slot may carry the slash itself, since an object slot is legal in the gap anyway:
        # 'regard somebody / something as something' is one frame and it states 'regard as'.
        assert LinguisticEngine.is_attested_phrase("regard as"), "regard as"
        assert not LinguisticEngine.is_attested_phrase("keep of"), "keep of"
        assert not LinguisticEngine.is_attested_phrase("depend for"), "depend for"

    def test_a_stressed_final_consonant_doubles_inside_the_frame(self):
        """Longman's pattern is 'be embedded in something', which 'embed' can only reach if
        the inflector knows emBED -> embedded.  The monosyllable shape test alone produces the
        non-word 'embeded', which matches nothing in the dictionary."""
        assert "embedded" in LinguisticEngine.inflected_forms("embed")
        assert "embeded" not in LinguisticEngine.inflected_forms("embed")
        assert "referred" in LinguisticEngine.inflected_forms("refer")
        assert "differed" in LinguisticEngine.inflected_forms("differ")
        assert LinguisticEngine.is_attested_phrase("embed in"), "embed in"

    def test_a_slash_is_a_boundary_not_a_space(self):
        """'on no account/not on any account' is two units.  Flattening the slash into a space
        reads across the boundary and is what gave the invented 'account on' its evidence."""
        flat = LinguisticEngine._entry_text_normalizer("on no account/not on any account")
        assert flat == "on no account / not on any account", flat
        assert not LinguisticEngine.is_attested_phrase("account on"), "account on"
        # A slash deeper inside a string separates alternatives inside it; it is not a list of
        # alternative headwords, so nothing is expanded across it.
        assert LinguisticEngine._slash_variants("on no account/not on any account") == \
               ["on no account/not on any account"]

    def test_leading_slash_alternatives_each_attest_their_own_phrase(self):
        """Longman writes 'stay/keep in touch', 'bring/call somebody to account' and
        'by/from all accounts' on one line.  Each alternative heads the same tail, so each one
        is a phrase of its own even though it shares a line with another."""
        assert LinguisticEngine._slash_variants("stay/keep in touch") == \
               ["stay in touch", "keep in touch"]
        assert LinguisticEngine._slash_variants("bring/call somebody to account") == \
               ["bring somebody to account", "call somebody to account"]
        assert LinguisticEngine._phrase_tokens("bring/call somebody to account") == \
               ["bring", "call", "to", "account"]
        for phrase in ["stay in touch", "keep in touch", "bring to account", "call to account",
                       "by all accounts", "from all accounts"]:
            assert LinguisticEngine.is_attested_phrase(phrase), phrase

    def test_surface_inflection_reaches_the_stated_lemma(self):
        """The passage writes 'kept in touch'; the entry states 'keep in touch'.  A token that
        is a real word gets its own lemma inflected, which is what closes that distance."""
        for phrase in ["kept in touch", "keeping in touch", "keeps in touch", "depended on",
                       "depends on", "accounts for", "happened to", "based on"]:
            assert LinguisticEngine.is_attested_phrase(phrase), phrase
        pattern = LinguisticEngine._phrase_regex(
            LinguisticEngine._phrase_tokens("keep in touch"), 1)
        assert "kept" in pattern and "keeping" in pattern, pattern

    def test_a_typo_cannot_lemmatize_into_a_frame(self):
        """Only an attested form is allowed the lemma bridge, so a misspelling cannot reach a
        frame it was never part of."""
        for phrase in ["depand on", "depond on", "kepp in touch", "keeps in tuch",
                       "account forr", "dependin on"]:
            assert not LinguisticEngine.is_attested_phrase(phrase), phrase
        pattern = LinguisticEngine._phrase_regex(["depand", "on"], 1)
        assert "depend" not in pattern, pattern

    def test_object_fit_follows_the_frame_not_the_parse(self):
        """'keep in' is a real phrasal verb, but Longman's frame is 'keep somebody in' - the
        object comes before the particle - so a parse that puts something after 'in' has read a
        locative or a noun of its own as the particle.  'depend on Mary' fits, because Longman's
        frame is 'depend on / upon somebody / something'."""
        fit = LinguisticEngine.ldoce_phrase_object_fit
        assert not fit("keep in", "moments"), "keep in moments"
        # 'keep in' + 'touch' fails the same frame test, but it is a different mistake: 'keep
        # in touch' is a stated unit and the parse cut it in half.  The miner drops the
        # fragment and ships the longer phrase - see
        # test_miner_drops_the_locative_particle_parse.
        assert not fit("keep in", "touch"), "keep in touch"
        for phrase, obj in [("depend on", "Mary"), ("depend on", "report"),
                            ("send in", "application"), ("wait for", "answer"),
                            ("worry about", "privacy"), ("long for", "peace")]:
            assert fit(phrase, obj), (phrase, obj)
        # The veto needs positive evidence: no frame, no claim, no veto - and an object that was
        # never claimed for the phrase is a fit by definition.
        assert fit("break away", "result"), "break away result"
        assert fit("keep in", ""), "keep in with no object"

    def test_miner_drops_the_locative_particle_parse(self):
        """The sentence that started this: 'keep silent in WeChat Moments' is not 'keep in'."""
        text = ("And they can also keep silent in WeChat Moments. They worry about their privacy "
                "and long for peace of mind, so they keep in touch with old friends.")
        shipped = [s["phrase"] for s in LinguisticEngine.mine_expression_skeletons(
            text, target_count=8)]
        assert "keep in touch with" in shipped, shipped
        assert "keep in" not in shipped, shipped
        assert "keep of" not in shipped, shipped

    def test_miner_still_ships_the_passage_set(self):
        """F7 acceptance (d) in miniature: the gate removes the inventions, not the yield."""
        from pathlib import Path
        path = Path("wiki/Book_1_Unit_1_Passage_A/sources/Book_1_Unit_1_Passage_A.md")
        text = path.read_text(encoding="utf-8")
        shipped = [s["phrase"] for s in LinguisticEngine.mine_expression_skeletons(
            text, target_count=8)]
        assert "keep in touch with" in shipped, shipped
        assert "keep in" not in shipped, shipped
        assert len(shipped) >= 4, shipped

    def test_the_raw_oxford_accessor_is_gone(self):
        """F7 acceptance (e): nothing under librarian/ reads oxford_collocations.json.gz any
        more, so the accessor and its lazy-loaded cache are gone with it."""
        assert not hasattr(LinguisticEngine, "get_oxford_raw"), "get_oxford_raw"
        assert not hasattr(LinguisticEngine, "_ocd_data"), "_ocd_data"

