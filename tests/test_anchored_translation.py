import unittest
import dataclasses
from librarian.processor import WikiProcessor
from librarian.expert_auditor import ExpertAuditor
from librarian.schemas import TranslationQuestion, TranslationQuiz


class TestAnchoredTranslation(unittest.TestCase):
    def setUp(self):
        pass

    def test_schema_instantiation(self):
        """Verify TranslationQuestion supports english_skeleton and target_keyword."""
        q = TranslationQuestion(
            translated_sentence="尽管面临重重阻力，他们依然鼓起勇气违抗不公的指令。",
            english_skeleton="Despite facing immense pressure, they [ ____ ] unjust orders.",
            target_keyword="defy",
            correct_english_answer="Despite facing immense pressure, they summoned the courage to defy unjust orders.",
            options=[
                "summoned the courage to defy",
                "summoned the courage for defying",
                "summoned the courage defy",
                "summoned courage to defying"
            ],
            correct_answer_index=0,
            hint="锁定词: defy; 注意不定式搭配",
            explanation="Option A 正确。summon the courage to do 是固定搭配，to 后接动词原形 defy。",
            design_audit="AUDIT: [中文 -> Target Keyword: defy + 不定式搭配] -> [English Skeleton Slot] -> [Trap 1: ...] [Trap 2: ...] [Trap 3: ...] -> [Why Distractors Fail]"
        )
        self.assertEqual(q.target_keyword, "defy")
        self.assertIn("[ ____ ]", q.english_skeleton)
        self.assertEqual(len(q.options), 4)

    def test_l1_audit_anchored_skeleton_success(self):
        """Verify anchored skeleton quiz passes Level 1 integrity gate without defects."""
        questions = [
            {
                "translated_sentence": "尽管面临重重阻力，他们依然鼓起勇气违抗不公的指令。",
                "english_skeleton": "Despite facing immense pressure, they [ ____ ] unjust orders.",
                "target_keyword": "defy",
                "correct_english_answer": "Despite facing immense pressure, they summoned the courage to defy unjust orders.",
                "options": [
                    "summoned the courage to defy",
                    "summoned courage for defying",
                    "summoned courage defy",
                    "summoned courage to defying"
                ],
                "correct_answer_index": 0,
                "hint": "锁定词: defy",
                "explanation": "Option A is correct.",
                "design_audit": "AUDIT: [违抗 -> Target Keyword: defy] -> [...]"
            }
        ]
        flagged, defects = WikiProcessor.audit_translation_integrity(
            {"questions": questions},
            unit_headwords=["defy", "pressure"],
            target_language="Chinese"
        )
        self.assertEqual(flagged, [])
        self.assertEqual(defects, [])

    def test_l1_audit_missing_skeleton_slot(self):
        """Verify Level 1 gate catches skeleton missing the [ ____ ] blank."""
        questions = [
            {
                "translated_sentence": "尽管面临重重阻力，他们依然鼓起勇气违抗不公的指令。",
                "english_skeleton": "Despite facing immense pressure, they defied unjust orders.",  # Missing slot!
                "target_keyword": "defy",
                "correct_english_answer": "Despite facing immense pressure, they defied unjust orders.",
                "options": ["defied", "defying", "to defy", "defies"],
                "correct_answer_index": 0,
                "explanation": "A is correct."
            }
        ]
        flagged, defects = WikiProcessor.audit_translation_integrity(
            {"questions": questions},
            unit_headwords=["defy"],
            target_language="Chinese"
        )
        self.assertIn(0, flagged)
        self.assertTrue(any("must contain exactly one '[ ____ ]' slot" in d for d in defects))

    def test_l1_audit_missing_keyword(self):
        """Verify Level 1 gate catches declared keyword missing in correct translation/options."""
        questions = [
            {
                "translated_sentence": "尽管面临重重阻力，他们依然鼓起勇气违抗不公的指令。",
                "english_skeleton": "Despite facing immense pressure, they [ ____ ] unjust orders.",
                "target_keyword": "hinder",  # Keyword declared but not present!
                "correct_english_answer": "Despite facing immense pressure, they summoned the courage to defy unjust orders.",
                "options": [
                    "summoned the courage to defy",
                    "summoned courage for defying",
                    "summoned courage defy",
                    "summoned courage to defying"
                ],
                "correct_answer_index": 0,
                "explanation": "A is correct."
            }
        ]
        flagged, defects = WikiProcessor.audit_translation_integrity(
            {"questions": questions},
            unit_headwords=["hinder", "defy"],
            target_language="Chinese"
        )
        self.assertIn(0, flagged)
        self.assertTrue(any("missing in correct translation/option" in d for d in defects))

    def test_l1_audit_legacy_full_sentence_backward_compatibility(self):
        """Verify legacy full-sentence translation quizzes without skeleton still pass smoothly."""
        questions = [
            {
                "translated_sentence": "随着人工智能技术的飞速发展，教育范式正在经历前所未有的深刻变革。",
                "correct_english_answer": "With the rapid advancement of artificial intelligence technology, educational paradigms are undergoing an unprecedented profound transformation.",
                "options": [
                    "With the rapid advancement of artificial intelligence technology, educational paradigms are undergoing an unprecedented profound transformation.",
                    "Along with rapid advancement for artificial intelligence technology, educational paradigms are undergoing an unprecedented profound transformation.",
                    "With the rapid advancement of artificial intelligence technology, educational paradigms undergo an unprecedented profound transformation.",
                    "Followed by rapid advancement of artificial intelligence technology, educational paradigms are undergoing an unprecedented profound transformation."
                ],
                "correct_answer_index": 0,
                "hint": "advancement",
                "explanation": "Option A is pristine.",
                "design_audit": "AUDIT: [人工智能飞速发展 -> Target Vocab: advancement + Prepositional Modifier] -> [...]"
            }
        ]
        flagged, defects = WikiProcessor.audit_translation_integrity(
            {"questions": questions},
            unit_headwords=["advancement", "transformation"],
            target_language="Chinese"
        )
        self.assertEqual(flagged, [])
        self.assertEqual(defects, [])

    def test_format_quiz_for_blind_audit_includes_skeleton_and_keyword(self):
        """Verify format_quiz_for_blind_audit outputs Target Keyword and English Skeleton anchors."""
        questions = [
            {
                "translated_sentence": "尽管面临重重阻力，他们依然鼓起勇气违抗不公的指令。",
                "english_skeleton": "Despite facing immense pressure, they [ ____ ] unjust orders.",
                "target_keyword": "defy",
                "options": [
                    "summoned the courage to defy",
                    "summoned courage for defying",
                    "summoned courage defy",
                    "summoned courage to defying"
                ],
                "correct_answer_index": 0
            }
        ]
        formatted = ExpertAuditor.format_quiz_for_blind_audit(
            source_text="Test source text",
            questions=questions,
            is_translation=True
        )
        self.assertIn("Target Keyword: defy", formatted)
        self.assertIn("English Skeleton: Despite facing immense pressure, they [ ____ ] unjust orders.", formatted)
        self.assertIn("[A] summoned the courage to defy  <-- [DECLARED TARGET TRANSLATION]", formatted)

    def test_auto_heal_index_desync_via_skeleton_and_explanation(self):
        """Verify sanitize_and_balance_quiz_keys auto-heals an LLM index desync (e.g. overbooking vs overbooks)."""
        quiz_data = {
            "title": "Translation Assessment",
            "questions": [
                {
                    "translated_sentence": "为了吸引更多客户，航空公司过度预订客票导致旅客排长队。",
                    "english_skeleton": "The airline [ ____ ] flights to meet demand, even though it caused inconvenience to passengers.",
                    "target_keyword": "overbook",
                    "correct_english_answer": "The airline overbooks flights to meet demand, even though it caused inconvenience to passengers.",
                    "options": [
                        "overbook",
                        "overbooking",
                        "overbooked",
                        "overbooks"
                    ],
                    "correct_answer_index": 1,  # Erroneous index pointing at 'overbooking'
                    "explanation": "Option D ‘overbooks’ is the only form that matches the present-simple tense required by the singular subject ‘airline’. Option A lacks the s-ending... Option B uses a noun form... Option C presents a past-tense..."
                }
            ]
        }
        healed_quiz = WikiProcessor.shuffle_quiz_options(quiz_data)
        q0 = healed_quiz["questions"][0]
        # In the healed quiz, the correct answer index must point to 'overbooks' (which matches the skeleton and explanation)
        correct_option_text = q0["options"][q0["correct_answer_index"]]
        self.assertEqual(correct_option_text, "overbooks")
        self.assertNotEqual(correct_option_text, "overbooking")

    def test_evaluator_translation_distractor_recycling_false_positive_prevention(self):
        """Verify evaluator does not flag morphological distractors of target keyword as in-list recycling."""
        from librarian.evaluator import _score_pedagogy
        items = [
            {
                "translated_sentence": "作为对传统文化的保护，学者们呼吁各方共同努力。",
                "english_skeleton": "As is argued in the study, \"The community must [ ____ ] the cultural heritage to prevent its loss.\",",
                "target_keyword": "preserve",
                "correct_english_answer": "As is argued in the study, \"The community must preserve the cultural heritage to prevent its loss.\",",
                "hint": "锁定词: preserve；注意动词原形与引语搭配。",
                "options": [
                    "preserve",
                    "to preserve",
                    "preserving the cultural heritage",
                    "preserves"
                ],
                "correct_answer_index": 0,
                "explanation": "Option A correctly supplies the base verb 'preserve' after modal 'must'."
            }
        ]
        # Unit study list contains 'preserve'
        user_prompt = "Vocabulary Study List: preserve, overbook, delegate, prune, raid"
        score, flags = _score_pedagogy(items, task_type="quiz", user_prompt=user_prompt)
        # Should not flag in-list distractor recycling
        recycling_flags = [f for f in flags if "in-list distractor recycling" in f]
        self.assertEqual(recycling_flags, [])
        self.assertGreaterEqual(score, 20.0)

    def test_evaluator_translation_target_keyword_in_skeleton(self):
        """Verify that target_keyword located in skeleton/answer (not options) passes pedagogy check."""
        from librarian.evaluator import _score_pedagogy
        items = [
            {
                "translated_sentence": "尽管我们想保留所有的记录，但为了简化流程，必须消除不必要的文件。",
                "english_skeleton": "We must eliminate unnecessary documents, even though [ ____ ].",
                "target_keyword": "eliminate",
                "correct_english_answer": "We must eliminate unnecessary documents, even though the staff prefers to keep them.",
                "hint": "锁定词: eliminate (消除); 记得在‘even though’后使用主语谓语一致的结构。",
                "options": [
                    "the staff prefer keep them",
                    "the staff prefers keeping them",
                    "the staff prefers to keep them",
                    "the staff prefer to keep them"
                ],
                "correct_answer_index": 2,
                "explanation": "Option C correctly uses singular verb with collective noun staff."
            }
        ]
        score, flags = _score_pedagogy(items, task_type="quiz")
        # Ensure it does not report target 'eliminate' not matching options[2]
        target_mismatch_flags = [f for f in flags if "target 'eliminate' not matching" in f]
        self.assertEqual(target_mismatch_flags, [])
        self.assertGreaterEqual(score, 25.0)


    def test_scheme_b_comparative_translation_appraisal(self):
        """Verify that Scheme B (idiomatic vs flawed translation appraisal) passes L1 Gate and Evaluator."""
        from librarian.evaluator import _score_pedagogy
        from librarian.schemas import TranslationQuestion, TranslationQuiz

        q = TranslationQuestion(
            translated_sentence="尽管面临重重阻力，他们依然鼓起勇气违抗不公的指令。",
            target_keyword="defy",
            target_grammar="Concessive clause",
            idiomatic_translation="Despite facing immense pressure, they summoned the courage to defy the unjust directives.",
            flawed_translation="Although facing big pressure, but they took courage to defy the unjust orders.",
            flaw_type="Chinglish syntax & redundant connective (Although... but)",
            diagnostic_critique="Version B commits an L1 negative-transfer error by pairing 'Although' with 'but'. Furthermore, 'took courage' is a weak translation compared to 'summoned the courage'."
        )
        self.assertEqual(q.target_keyword, "defy")
        self.assertEqual(q.idiomatic_translation, "Despite facing immense pressure, they summoned the courage to defy the unjust directives.")

        # Test L1 Gate auto-synthesis and validation
        quiz_data = {"questions": [dataclasses.asdict(q)]}
        flagged, defects = WikiProcessor.audit_translation_integrity(
            quiz_data,
            unit_headwords=["defy", "pressure"],
            target_language="Chinese"
        )
        self.assertEqual(flagged, [])
        self.assertEqual(defects, [])
        # Verify options were synthesized
        self.assertEqual(len(quiz_data["questions"][0]["options"]), 2)
        self.assertEqual(quiz_data["questions"][0]["correct_answer_index"], 0)

        # Test Evaluator pedagogy scoring
        score, flags = _score_pedagogy(quiz_data["questions"], task_type="quiz")
        self.assertEqual([f for f in flags if "failed pedagogy check" in f], [])
        self.assertGreaterEqual(score, 25.0)

    def test_render_handout_comparative_appraisal(self):
        """Verify _render_handout correctly produces an interactive HTML handout for Scheme B."""
        from librarian.schemas import TranslationQuestion, TranslationQuiz
        processor = WikiProcessor()
        q = TranslationQuestion(
            translated_sentence="尽管面临重重阻力，他们依然鼓起勇气违抗不公的指令。",
            target_keyword="defy",
            target_grammar="Concessive clause",
            idiomatic_translation="Despite facing immense pressure, they summoned the courage to defy the unjust directives.",
            flawed_translation="Although facing big pressure, but they took courage to defy the unjust orders.",
            flaw_type="Chinglish redundant connective (Although... but)",
            diagnostic_critique="Version B commits an L1 negative-transfer error by pairing Although with but.",
            options=[
                "Despite facing immense pressure, they summoned the courage to defy the unjust directives.",
                "Although facing big pressure, but they took courage to defy the unjust orders."
            ],
            correct_answer_index=0
        )
        quiz = TranslationQuiz(
            title="Unit Test Translation Handout",
            questions=[q]
        )
        html = processor._render_handout(quiz, "translation", language="Chinese")
        self.assertIn("Unit Test Translation Handout", html)
        self.assertIn("defy", html)
        self.assertIn("appraisal-target-bar", html)
        self.assertIn("idiomatic_translation", html)
        self.assertIn("flawed_translation", html)
        self.assertIn("diagnostic_critique", html)

    def test_step1_and_step2_translation_micro_task_blueprint_generation(self):
        """
        Verify Step 1 (Register-Aligned Lexical & Syntactic Pairing) and
        Step 2 (Single-Point Minimal-Pair Corruption Operator) in build_translation_micro_task_blueprints.
        """
        from librarian.linguistics import LinguisticEngine

        vocab_items = [
            {
                "word": "pack up",
                "part_of_speech": "phrasal verb",
                "definition": "[spoken, informal] to stop working or operating properly",
                "quote": "The old printer packed up right before the deadline.",
                "example_usage": "My car engine packed up on the motorway.",
            },
            {
                "word": "originate",
                "part_of_speech": "verb",
                "definition": "[formal] to come from a particular place or start in a particular situation",
                "quote": "Many academic traditions originated in medieval universities.",
                "example_usage": "Many Christmas traditions originated in Germany.",
            },
        ]
        grammar_items = [
            {
                "name": "Negative Adverbial Inversion",
                "formula": "Never / Seldom / Rarely + Auxiliary + Subject + Main Verb",
                "pedagogical_function": "Creates formal academic emphasis at the clause front.",
                "common_mistakes": "Grammar Warning (NEGATIVE INVERSION): Never keep normal declarative Subject + Verb order after fronted 'Seldom' or 'Never'.",
            },
            {
                "name": "Concessive Clause",
                "formula": "Although / Even though + Subject + Verb, Main Clause",
                "pedagogical_function": "Frames an everyday contrast between an obstacle and an outcome.",
                "common_mistakes": "Grammar Warning (CONCESSIVE CONNECTIVES): Never pair 'Although / Even though' with 'but' in the same sentence.",
            },
        ]

        blueprints = LinguisticEngine.build_translation_micro_task_blueprints(
            vocab_content=vocab_items,
            grammar_content=grammar_items,
            target_count=2,
            target_language="Simplified Chinese",
            cefr_level="B2",
        )
        self.assertEqual(len(blueprints), 2)

        bp_pack = next(bp for bp in blueprints if bp["target_keyword"] == "pack up")
        bp_orig = next(bp for bp in blueprints if bp["target_keyword"] == "originate")

        self.assertEqual(bp_pack["vocab_register"], "informal")
        self.assertEqual(bp_pack["target_grammar_name"], "Concessive Clause")
        self.assertEqual(bp_orig["vocab_register"], "formal")
        self.assertEqual(bp_orig["target_grammar_name"], "Negative Adverbial Inversion")

        for bp in blueprints:
            self.assertIn(bp["correct_answer_index"], (0, 1))
            self.assertIn("### Item", bp["micro_task_block"])
            self.assertIn("🎯 Micro-Task for LLM:", bp["micro_task_block"])
            self.assertIn("Minimal Pair", bp["micro_task_block"])

    def test_step3_zero_redundancy_schema_and_hydration(self):
        """
        Verify Step 3:
        1. translation_quiz.json schema requires only the 5 pruned LLM keys.
        2. hydrate_translation_blueprint_items deterministically populates target_keyword,
           target_grammar, flaw_type, CRC32-ordered options, correct_answer_index, and explanation.
        3. audit_translation_integrity enforces the >=85% Minimal-Pair overlap contract.
        """
        import json
        from pathlib import Path
        from librarian.linguistics import LinguisticEngine

        schema_path = Path("librarian/prompts/translation_quiz.json")
        schema_data = json.loads(schema_path.read_text(encoding="utf-8"))
        item_schema = schema_data["properties"]["questions"]["items"]
        expected_keys = {
            "design_audit",
            "translated_sentence",
            "idiomatic_translation",
            "flawed_translation",
            "diagnostic_critique",
        }
        self.assertEqual(set(item_schema["required"]), expected_keys)
        self.assertEqual(set(item_schema["properties"].keys()), expected_keys)

        blueprints = [
            {
                "item_index": 1,
                "target_keyword": "originate",
                "target_grammar": "Concessive Clause: Although / Even though + Subject + Verb, Main Clause",
                "flaw_type": "L1 Conjunction Redundancy (`Although ... but ...` double conjunction)",
                "correct_answer_index": 1,
            }
        ]
        raw_llm_output = {
            "title": "Translation Appraisal Quiz",
            "questions": [
                {
                    "design_audit": "AUDIT: [originate] + [Although + Clause] -> L1 Conjunction Redundancy",
                    "translated_sentence": "虽然许多圣诞传统起源于德国，但它们如今已传播到世界各地。",
                    "idiomatic_translation": "Although many Christmas traditions originated in Germany, they have now spread around the world.",
                    "flawed_translation": "Although many Christmas traditions originated in Germany, but they have now spread around the world.",
                    "diagnostic_critique": "The flawed translation commits an L1 conjunction redundancy error by adding 'but' after an 'Although' clause, whereas the idiomatic version uses 'originated in' and a clean main clause.",
                }
            ],
        }

        hydrated = WikiProcessor.hydrate_translation_blueprint_items(raw_llm_output, blueprints)
        q0 = hydrated["questions"][0]
        self.assertEqual(q0["target_keyword"], "originate")
        self.assertEqual(
            q0["target_grammar"],
            "Concessive Clause: Although / Even though + Subject + Verb, Main Clause",
        )
        self.assertEqual(
            q0["flaw_type"],
            "L1 Conjunction Redundancy (`Although ... but ...` double conjunction)",
        )
        self.assertEqual(q0["correct_answer_index"], 1)
        self.assertEqual(q0["options"][1], q0["idiomatic_translation"])
        self.assertEqual(q0["options"][0], q0["flawed_translation"])
        self.assertEqual(q0["explanation"], q0["diagnostic_critique"])

        flagged, defects = WikiProcessor.audit_translation_integrity(
            hydrated,
            unit_headwords=["originate"],
            target_language="Chinese",
            cefr_level="B2",
            enforce_minimal_pair=True,
        )
        self.assertEqual(flagged, [])
        self.assertEqual(defects, [])

    def test_pruned_schema_evaluator_and_blueprint_isolation(self):
        """
        Regression test for:
        1. _score_pedagogy on pruned 5-field LLM output (before hydrate_translation_blueprint_items).
        2. POS & sense_idx isolation in get_word_error_traps ('interior' noun vs adj; 'cram' sense 0 vs sense 3).
        3. Syntactic trap resolution for passive/unaccusative (ignoring incidental 'never' in common_mistakes)
           and causative 'let sb. do' (avoiding truncation at 'sb.').
        """
        from librarian.evaluator import _score_pedagogy
        from librarian.linguistics import LinguisticEngine

        raw_pruned_items = [
            {
                "design_audit": "AUDIT: [interior] + [S VP, V-ing Phrase] -> Lexical Valency Trap",
                "translated_sentence": "参观完大厅后，我们注意到教堂的内部很昏暗。",
                "idiomatic_translation": "Having visited the hall, we noticed that the interior of the church was dark.",
                "flawed_translation": "Having visited the hall, we noticed that the interior from the church was dark.",
                "diagnostic_critique": "The flawed version uses 'interior from' instead of 'the interior of [sth]'.",
            },
            {
                "design_audit": "AUDIT: [cram] + [S let O Bare Inf] -> Causative Infinitive Trap (let sb to do)",
                "translated_sentence": "她允许我们挤进那辆小汽车，尽管空间很紧张。",
                "idiomatic_translation": "She let us cram into the small car, even though space was tight.",
                "flawed_translation": "She let us to cram into the small car, even though space was tight.",
                "diagnostic_critique": "The flawed version adds 'to' after causative 'let'.",
            },
        ]
        user_prompt = (
            "### Item 1 Specification ###\n"
            "- Target Keyword: interior (noun) — \"[written] the inner part or inside of something\"\n"
            "### Item 2 Specification ###\n"
            "- Target Keyword: cram (verb) — \"if a lot of people cram into a place or vehicle\"\n"
        )
        score, flags = _score_pedagogy(raw_pruned_items, task_type="quiz", user_prompt=user_prompt)
        self.assertEqual(flags, [])
        self.assertEqual(score, 25.0)

        interior_res = LinguisticEngine._resolve_translation_lexical_trap(
            word="interior",
            pos="noun",
            definition="[written] the inner part or inside of something",
            example_usage="The interior of the church was dark.",
        )
        self.assertEqual(interior_res["required_frame"], "the interior of [sth]")
        if interior_res.get("trap"):
            self.assertNotEqual(interior_res["trap"].get("trap_type"), "GRAM_ATTRIBUTIVE_ONLY")

        cram_res = LinguisticEngine._resolve_translation_lexical_trap(
            word="cram",
            pos="verb",
            definition="if a lot of people cram into a place or vehicle, they go into it so it is then full",
            example_usage="We all crammed in and Pete started the car.",
        )
        self.assertNotIn("week", cram_res["required_frame"].lower())
        self.assertIn("cram", cram_res["required_frame"].lower())

        passive_trap = LinguisticEngine._resolve_translation_syntactic_trap({
            "name": "[S] + [be/modal be] + [V3]",
            "formula": "[S] + [be/modal be] + [V3]",
            "common_mistakes": (
                "Grammar Warning (Unaccusative Verbs in Passive Voice): Learners frequently passivize "
                "intransitive change-of-state verbs. Intransitive verbs like 'take place' never form a passive voice."
            ),
        })
        self.assertIn("Passive Voice", passive_trap["flaw_label"])
        self.assertNotIn("Inversion", passive_trap["flaw_label"])

        causative_trap = LinguisticEngine._resolve_translation_syntactic_trap({
            "name": "[S] + let + [NP] + [Bare Infinitive]",
            "formula": "[S] + let + [NP] + [Bare Infinitive]",
            "common_mistakes": (
                "Grammar Warning (Causative 'Let' vs. 'Allow'): Learners frequently blend 'let sb. do' with 'allow sb. to do'."
            ),
        })
        self.assertIn("Causative Infinitive Trap", causative_trap["flaw_label"])
        self.assertNotIn("blend 'let sb)", causative_trap["flaw_label"])


if __name__ == "__main__":
    import dataclasses
    unittest.main()



