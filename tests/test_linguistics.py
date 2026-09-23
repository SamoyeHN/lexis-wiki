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

