"""
Tests for Discourse Reading Prototype Engine (Neuro-Symbolic Discourse Reading Assessment)
"""

from pathlib import Path
import pytest

from librarian.linguistics_core.discourse_reading_prototype import (
    CEFRDifficultyTier,
    DiscourseReadingEngine,
    DiscourseRelationType,
    DistractorFallacyOperator,
    ReadingItemBlueprint,
    ReadingDistractorSpec,
)


@pytest.fixture
def unit6_passage() -> str:
    path = Path("wiki/Book_4_Unit_6_Section_A/sources/Book_4_Unit_6_Section_A.md")
    if not path.exists():
        pytest.skip("Unit 6 source file not available")
    return path.read_text(encoding="utf-8")


@pytest.fixture
def book1_unit2_passage() -> str:
    path = Path("wiki/Book_1_Unit_2_Passage_A/sources/Book_1_Unit_2_Passage_A.md")
    if not path.exists():
        pytest.skip("Book 1 Unit 2 source file not available")
    return path.read_text(encoding="utf-8")


class TestDiscourseReadingPrototype:
    def test_extract_discourse_graph(self, unit6_passage: str):
        graph = DiscourseReadingEngine.extract_discourse_graph(unit6_passage)
        assert len(graph.paragraphs) >= 8, f"Expected at least 8 paragraphs, got {len(graph.paragraphs)}"
        assert len(graph.sentence_pool) >= 25, "Expected at least 25 indexed sentences"
        assert len(graph.causal_chains) >= 2, "Expected causal chains in Unit 6"
        assert len(graph.contrast_spans) >= 3, "Expected contrast spans in Unit 6"
        assert len(graph.modality_spans) >= 2, "Expected modality spans in Unit 6"
        assert graph.conclusion_unit is not None, "Conclusion unit must be identified"
        assert graph.introduction_unit is not None, "Introduction unit must be identified"

    def test_build_discourse_blueprints_diversity_and_anchors(self, unit6_passage: str):
        blueprints = DiscourseReadingEngine.build_discourse_blueprints(unit6_passage, count=5)
        assert len(blueprints) == 5

        # Check skill types diversity
        skills = [bp.skill_type for bp in blueprints]
        assert DiscourseRelationType.MACRO_GIST in skills
        assert DiscourseRelationType.CAUSAL_CHAIN in skills
        assert DiscourseRelationType.RHETORICAL_CONTRAST in skills
        assert DiscourseRelationType.FACTUAL_MECHANISM in skills
        assert DiscourseRelationType.MODALITY_HEDGING in skills

        # Check unique anchor sentence IDs
        anchor_ids = [bp.anchor_sentence_id for bp in blueprints]
        assert len(set(anchor_ids)) == 5, f"All 5 items must use distinct anchor sentences: {anchor_ids}"

        # Verify each blueprint has exactly 3 distractor specifications
        for bp in blueprints:
            assert len(bp.distractors) == 3
            ops = [d.operator for d in bp.distractors]
            assert DistractorFallacyOperator.TRUE_BUT_IRRELEVANT in ops
            assert (
                DistractorFallacyOperator.CAUSAL_REVERSAL in ops
                or DistractorFallacyOperator.POLARITY_INVERSION in ops
            )
            assert DistractorFallacyOperator.MODALITY_ESCALATION in ops

    def test_true_but_irrelevant_cross_paragraph_distance(self, unit6_passage: str):
        blueprints = DiscourseReadingEngine.build_discourse_blueprints(unit6_passage, count=5)
        for bp in blueprints:
            irrel_distractor = next(
                d for d in bp.distractors if d.operator == DistractorFallacyOperator.TRUE_BUT_IRRELEVANT
            )
            assert irrel_distractor.source_reference is not None
            # Must not cite the same anchor as the item's target anchor
            assert irrel_distractor.source_reference != bp.anchor_sentence_id

    def test_lexical_leakage_gate_catches_verbatim_copying(self, unit6_passage: str):
        blueprints = DiscourseReadingEngine.build_discourse_blueprints(unit6_passage, count=5)
        bp = blueprints[0]  # Macro Gist

        # Case 1: Bad item where correct answer is an exact copy-paste of the anchor sentence
        leaky_item = {
            "options": [
                "Alternative distractor one with balanced length for test.",
                bp.anchor_sentence_text,  # Verbatim anchor leakage!
                "Alternative distractor three with balanced length for test.",
                "Alternative distractor four with balanced length for test.",
            ],
            "correct_answer_index": 1,
        }
        res = DiscourseReadingEngine.evaluate_reading_item(leaky_item, bp)
        assert not res.is_valid
        assert any("Lexical Leakage Gate Violation" in f for f in res.flags)
        assert res.lexical_overlap_ratio > 0.35

        # Case 2: Good item where correct answer is a clean C1 paraphrase
        paraphrased_item = {
            "options": [
                "A reliance on single dominant technologies is economically superior.",
                "Global energy security depends on synthesizing a diverse range of renewable solutions.",
                "Immediate international bans on hydrocarbons should be implemented unilaterally.",
                "Petroleum products will remain plentiful and continue powering the next century.",
            ],
            "correct_answer_index": 1,
        }
        res_good = DiscourseReadingEngine.evaluate_reading_item(paraphrased_item, bp)
        assert res_good.is_valid
        assert res_good.lexical_overlap_ratio == 0.0
        assert len(res_good.flags) == 0

    def test_length_parity_gate_catches_giveaway_options(self, unit6_passage: str):
        blueprints = DiscourseReadingEngine.build_discourse_blueprints(unit6_passage, count=5)
        bp = blueprints[1]

        # Case 1: Unbalanced options (one very long correct answer, 3 short distractors)
        unbalanced_item = {
            "options": [
                "Solar works well.",
                (
                    "The eventual depletion of fossil fuels coupled with escalating global demand necessitates "
                    "a strategic transition toward renewable synthesis and technological innovation across all sectors."
                ),
                "Wind is clean.",
                "Dams need maintenance.",
            ],
            "correct_answer_index": 1,
        }
        res_unbalanced = DiscourseReadingEngine.evaluate_reading_item(unbalanced_item, bp)
        assert any("Length Parity Gate Warning" in f for f in res_unbalanced.flags)
        assert res_unbalanced.length_std_dev_ratio > 0.28

        # Case 2: Balanced options (all ~12-14 words)
        balanced_item = {
            "options": [
                "Reservoirs require continuous sediment excavation to maintain long-term hydroelectric operational capacity.",
                "Rapid population expansion accelerates the physical exhaustion of conventional fossil fuel deposits.",
                "Nuclear power generation completely eliminates hazardous atmospheric pollutants without incurring any economic costs.",
                "Solar apparatus manufacturing relies exclusively on obsolete combustion machinery and mechanical motors.",
            ],
            "correct_answer_index": 1,
        }
        res_balanced = DiscourseReadingEngine.evaluate_reading_item(balanced_item, bp)
        assert not any("Length Parity Gate Warning" in f for f in res_balanced.flags)
        assert res_balanced.length_std_dev_ratio <= 0.28

    def test_blueprint_micro_task_serialization(self, unit6_passage: str):
        blueprints = DiscourseReadingEngine.build_discourse_blueprints(unit6_passage, count=5)
        for bp in blueprints:
            mt = bp.to_micro_task()
            assert "item_index" in mt
            assert "skill_type" in mt
            assert "anchor_id" in mt
            assert "anchor_text" in mt
            assert "prescribed_distractor_operators" in mt
            assert len(mt["prescribed_distractor_operators"]) == 3

    def test_cefr_difficulty_tier_mapping(self):
        assert CEFRDifficultyTier.from_cefr("A1") == CEFRDifficultyTier.FOUNDATIONAL
        assert CEFRDifficultyTier.from_cefr("A2") == CEFRDifficultyTier.FOUNDATIONAL
        assert CEFRDifficultyTier.from_cefr("B1") == CEFRDifficultyTier.INTERMEDIATE
        assert CEFRDifficultyTier.from_cefr("B2") == CEFRDifficultyTier.INTERMEDIATE
        assert CEFRDifficultyTier.from_cefr("C1") == CEFRDifficultyTier.ADVANCED
        assert CEFRDifficultyTier.from_cefr("C2") == CEFRDifficultyTier.ADVANCED

    def test_foundational_tier_blueprints_and_concrete_stems(self, book1_unit2_passage: str):
        blueprints = DiscourseReadingEngine.build_discourse_blueprints(
            book1_unit2_passage, count=5, cefr_level="A2"
        )
        assert len(blueprints) == 5
        for bp in blueprints:
            assert bp.tier == CEFRDifficultyTier.FOUNDATIONAL
            mt = bp.to_micro_task()
            assert mt["tier"] == "foundational"
            assert "5 to 10 words" in mt["option_length_guidance"]
            assert "A1-B1" in mt["lexical_ceiling_instruction"]

            # Operators for Foundational must include EXTREME_WORD, OPPOSITE_FACT, or TRUE_BUT_IRRELEVANT
            ops = [d.operator for d in bp.distractors]
            assert (
                DistractorFallacyOperator.EXTREME_WORD in ops
                or DistractorFallacyOperator.OPPOSITE_FACT in ops
            )

        # Check concrete stem wording
        stems = [bp.question_stem_prompt for bp in blueprints]
        # Item 1 is clean main idea
        assert "What is the main idea of this passage?" in stems[0]
        # No meta-linguistic jargon in foundational stems
        for stem in stems:
            assert "rhetorical contrast or tension" not in stem
            assert "causal mechanism" not in stem
            assert "authorial stance" not in stem

    def test_lexical_ceiling_gate_for_foundational_tier(self, book1_unit2_passage: str):
        blueprints = DiscourseReadingEngine.build_discourse_blueprints(
            book1_unit2_passage, count=5, cefr_level="A2"
        )
        bp = blueprints[0]

        # Case 1: All words are A1-B1 or in passage (volunteer, cards, thanks, party)
        clean_foundational_item = {
            "options": [
                "Organizations should show gratitude to people who help.",
                "Volunteers must receive expensive gifts after every event.",
                "Staff members never send paper cards to volunteers.",
                "People should stop doing community service completely.",
            ],
            "correct_answer_index": 0,
        }
        res_clean = DiscourseReadingEngine.evaluate_reading_item(clean_foundational_item, bp)
        assert res_clean.is_valid
        assert not any("Lexical Ceiling Warning" in f for f in res_clean.flags)
        assert res_clean.score == 100.0

        # Case 2: Option introduces out-of-passage B2/C1 academic vocabulary
        violating_item = {
            "options": [
                "Organizations should show gratitude to people who help.",
                "A mandatory requirement is established for all volunteers.",  # 'mandatory' is B2
                "Staff members encapsulate the institutional dichotomy.",  # 'encapsulate' is C2, 'dichotomy' is C2
                "People should stop doing community service completely.",
            ],
            "correct_answer_index": 0,
        }
        res_violating = DiscourseReadingEngine.evaluate_reading_item(violating_item, bp)
        assert res_violating.is_valid  # Still physically valid JSON, but penalizes score
        assert any("Lexical Ceiling Warning" in f for f in res_violating.flags)
        assert res_violating.score < 100.0

    def test_intermediate_and_advanced_tier_differentiation(self, unit6_passage: str):
        bp_inter = DiscourseReadingEngine.build_discourse_blueprints(
            unit6_passage, count=5, cefr_level="B2"
        )
        bp_adv = DiscourseReadingEngine.build_discourse_blueprints(
            unit6_passage, count=5, cefr_level="C1"
        )

        assert bp_inter[0].tier == CEFRDifficultyTier.INTERMEDIATE
        assert bp_adv[0].tier == CEFRDifficultyTier.ADVANCED

        assert bp_inter[0].question_stem_prompt == "Which of the following best describes the main idea of this passage?"
        assert bp_adv[0].question_stem_prompt == "Which statement best encapsulates the author's primary thesis in this passage?"

        assert "8 to 14 words" in bp_inter[0].to_micro_task()["option_length_guidance"]
        assert "10 to 16 words" in bp_adv[0].to_micro_task()["option_length_guidance"]

    def test_pragmatic_inference_blueprint_and_operators(self, book1_unit2_passage: str, unit6_passage: str):
        # 1. Test 6-item generation on Book 1 Unit 2 (Foundational)
        bp_found = DiscourseReadingEngine.build_discourse_blueprints(
            book1_unit2_passage, count=6, cefr_level="A2"
        )
        assert len(bp_found) == 6
        inf_bp_found = bp_found[5]
        assert inf_bp_found.skill_type == DiscourseRelationType.PRAGMATIC_INFERENCE
        assert "suggested" in inf_bp_found.question_stem_prompt or "infer" in inf_bp_found.question_stem_prompt

        ops_found = [d.operator for d in inf_bp_found.distractors]
        assert DistractorFallacyOperator.EXPLICIT_RESTATEMENT in ops_found
        assert DistractorFallacyOperator.OVER_EXTENSION in ops_found

        # 2. Test 6-item generation on Book 4 Unit 6 (Advanced)
        bp_adv = DiscourseReadingEngine.build_discourse_blueprints(
            unit6_passage, count=6, cefr_level="C1"
        )
        assert len(bp_adv) == 6
        inf_bp_adv = bp_adv[5]
        assert inf_bp_adv.skill_type == DiscourseRelationType.PRAGMATIC_INFERENCE
        assert "presupposes" in inf_bp_adv.question_stem_prompt or "suggest" in inf_bp_adv.question_stem_prompt

        ops_adv = [d.operator for d in inf_bp_adv.distractors]
        assert DistractorFallacyOperator.EXPLICIT_RESTATEMENT in ops_adv
        assert DistractorFallacyOperator.OVER_EXTENSION in ops_adv
        assert DistractorFallacyOperator.REVERSE_ENTAILMENT in ops_adv

    def test_section_bijection_book1_unit2(self, book1_unit2_passage: str):
        """Asserts 1-to-1 bijection between sections and quiz items for multi-section passage."""
        blueprints = DiscourseReadingEngine.build_discourse_blueprints(
            book1_unit2_passage, count=6, cefr_level="A2"
        )
        assert len(blueprints) == 6

        section_titles = [bp.section_title for bp in blueprints]
        expected_sections = [
            "Paragraph 1",
            "Potlucks for Staff and Volunteers",
            "Giving Cards to Volunteers",
            "Hats Off to Volunteers",
            "Afternoon at the Movies",
            "Pat on the Back",
        ]
        assert section_titles == expected_sections, f"Expected exact 1-to-1 section mapping: {section_titles}"

        # Stems must reference section titles for custom sections
        for i in range(1, 6):
            sec = expected_sections[i]
            assert sec in blueprints[i].question_stem_prompt, (
                f"Item {i+1} stem '{blueprints[i].question_stem_prompt}' must embed section '{sec}'"
            )

    def test_section_clustering_gate_and_hydration(self, book1_unit2_passage: str):
        from librarian.processor_core.quiz_integrity_gates import QuizIntegrityGatesMixin
        from librarian.processor_core.quiz_generator import QuizGeneratorMixin

        blueprints = DiscourseReadingEngine.build_discourse_blueprints(
            book1_unit2_passage, count=6, cefr_level="A2"
        )
        bp_specs = [bp.to_micro_task() for bp in blueprints]

        # Case 1: Clustering defect where Hats Off to Volunteers is tested twice (Item 4 and Item 5)
        clustered_quiz = {
            "questions": [
                {"question": "What is the main idea?", "options": ["A", "B", "C", "D"], "correct_answer_index": 0, "category": "Main Idea"},
                {"question": "In the 'Potlucks for Staff and Volunteers' section, why potluck?", "options": ["A", "B", "C", "D"], "correct_answer_index": 0, "category": "Detail/Recall"},
                {"question": "In the 'Giving Cards to Volunteers' section, why give cards?", "options": ["A", "B", "C", "D"], "correct_answer_index": 0, "category": "Detail/Recall"},
                {"question": "In the 'Hats Off to Volunteers' section, what happens?", "options": ["A", "B", "C", "D"], "correct_answer_index": 0, "category": "Detail/Recall"},
                {"question": "In the 'Hats Off to Volunteers' section, why hats?", "options": ["A", "B", "C", "D"], "correct_answer_index": 0, "category": "Author's Tone/Purpose"},
                {"question": "In the 'Pat on the Back' section, what idea is shared?", "options": ["A", "B", "C", "D"], "correct_answer_index": 0, "category": "Inference"},
            ]
        }
        flagged, defects = QuizIntegrityGatesMixin.audit_reading_integrity(
            clustered_quiz, passage_text=book1_unit2_passage, cefr_level="A2", blueprint_specs=bp_specs
        )
        assert 4 in flagged, f"Item #5 (idx 4) must be flagged for duplicate section: {defects}"
        assert any("Section clustering defect" in d for d in defects)

        # Case 2: Hydration automatically heals the clustered stem to the blueprint's prescribed stem
        healed_quiz = QuizGeneratorMixin.hydrate_reading_blueprint_items(clustered_quiz, bp_specs)
        item5_stem = healed_quiz["questions"][4]["question"]
        assert "Afternoon at the Movies" in item5_stem, f"Item #5 must be healed to Afternoon at the Movies: {item5_stem}"

        # Re-audit on healed quiz should have 0 section clustering defects
        flagged_after, defects_after = QuizIntegrityGatesMixin.audit_reading_integrity(
            healed_quiz, passage_text=book1_unit2_passage, cefr_level="A2", blueprint_specs=bp_specs
        )
        assert not any("Section clustering defect" in d for d in defects_after)
        assert not any("Section mismatch" in d for d in defects_after)

    def test_chronological_paragraph_ordering_and_distinct_distant_refs(self, unit6_passage: str):
        blueprints = DiscourseReadingEngine.build_discourse_blueprints(unit6_passage, count=5)
        assert len(blueprints) == 5

        # Item 1 should be MACRO_GIST
        assert blueprints[0].skill_type == DiscourseRelationType.MACRO_GIST

        # Items 2..N should be monotonically non-decreasing by target_paragraph_index
        para_indices = [bp.target_paragraph_index for bp in blueprints[1:]]
        assert para_indices == sorted(para_indices), f"Paragraph indices must progress chronologically: {para_indices}"

        # Distractor source references across items must be distinct (zero recycling)
        all_refs = []
        for bp in blueprints:
            for d in bp.distractors:
                if d.source_reference:
                    all_refs.append(d.source_reference)
        assert len(all_refs) == len(set(all_refs)), f"Distractor references must be distinct, got duplicates: {all_refs}"

    def test_macro_gist_dual_pole_and_title_scope(self, unit6_passage: str):
        blueprints = DiscourseReadingEngine.build_discourse_blueprints(unit6_passage, count=5)
        gist_bp = blueprints[0]
        assert gist_bp.skill_type == DiscourseRelationType.MACRO_GIST

        # Check that Title Scope is embedded in paraphrase target instruction
        assert "The coming energy crisis" in gist_bp.paraphrase_target_instruction

        # Check that Dual-Pole (both intro and conclusion anchors) are referenced
        assert "initial problem/proposition" in gist_bp.paraphrase_target_instruction or "introductory proposition" in gist_bp.paraphrase_target_instruction
        assert "final resolution" in gist_bp.paraphrase_target_instruction or "concluding resolution" in gist_bp.paraphrase_target_instruction
