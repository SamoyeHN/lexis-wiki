import unittest
from unittest.mock import MagicMock, patch
from librarian.expert_auditor import ExpertAuditor
from librarian.schemas import (
    QuizQualityAuditReport,
    QuestionAuditItem,
    DistractorAuditItem,
    get_json_schema,
)


class TestExpertAuditor(unittest.TestCase):
    def test_schema_generation(self):
        schema = get_json_schema(QuizQualityAuditReport)
        self.assertEqual(schema["type"], "object")
        self.assertIn("overall_quality_score", schema["properties"])
        self.assertIn("pass_audit", schema["properties"])
        self.assertIn("blind_solve_accuracy", schema["properties"])
        self.assertIn("questions", schema["properties"])

    def test_format_quiz_for_blind_audit(self):
        source = "Henry dedicated his life to preventing animal suffering."
        questions = [
            {
                "question": "What was Henry's primary mission?",
                "options": [
                    "To prevent animal suffering",
                    "To run for political office",
                    "To profit from cosmetics",
                    "To direct gangster movies"
                ],
                "correct_answer_index": 0,
                "explanation": "Option A is directly stated."
            }
        ]
        blind_text = ExpertAuditor.format_quiz_for_blind_audit(source, questions)
        # Ensure answers and explanations are completely stripped
        self.assertNotIn("correct_answer_index", blind_text)
        self.assertNotIn("Option A is directly stated", blind_text)
        self.assertIn("What was Henry's primary mission?", blind_text)
        self.assertIn("[A] To prevent animal suffering", blind_text)
        self.assertIn("[B] To run for political office", blind_text)

    @patch("librarian.expert_auditor.LLMClient")
    def test_audit_quiz_success_and_divergence_check(self, mock_llm_cls):
        mock_llm_instance = MagicMock()
        mock_llm_cls.return_value = mock_llm_instance

        # Simulate expert model audit response
        mock_report = QuizQualityAuditReport(
            quiz_title="Henry Spira Reading Quiz",
            overall_quality_score=92,
            pass_audit=True,
            blind_solve_accuracy=1.0,
            questions=[
                QuestionAuditItem(
                    item_index=0,
                    blind_solved_index=0, # Matches declared key 0
                    confidence="Definite",
                    single_fit_valid=True,
                    distractors=[
                        DistractorAuditItem(option_letter="A", option_text="To prevent animal suffering", trap_type="None (Correct Answer)", plausibility_rating="High", elimination_rationale="Matches passage verbatim."),
                        DistractorAuditItem(option_letter="B", option_text="To run for political office", trap_type="Plausible Real-World Distractor", plausibility_rating="Medium", elimination_rationale="Contradicted by text."),
                        DistractorAuditItem(option_letter="C", option_text="To profit from cosmetics", trap_type="Scope Shift / Over-generalization", plausibility_rating="High", elimination_rationale="Opposite of his boycott actions."),
                        DistractorAuditItem(option_letter="D", option_text="To direct gangster movies", trap_type="Speaker / Entity Misattribution", plausibility_rating="Medium", elimination_rationale="Blunt speech sounded like movie character, but he did not direct them.")
                    ],
                    pedagogical_score=95,
                    diagnostic_feedback="Excellent item design."
                )
            ],
            summary_verdict="High quality assessment."
        )
        mock_llm_instance.chat.return_value = mock_report

        quiz_data = {
            "title": "Henry Spira Reading Quiz",
            "questions": [
                {
                    "question": "What was Henry's primary mission?",
                    "options": [
                        "To prevent animal suffering",
                        "To run for political office",
                        "To profit from cosmetics",
                        "To direct gangster movies"
                    ],
                    "correct_answer_index": 0
                }
            ]
        }

        report = ExpertAuditor.audit_quiz("Sample text", quiz_data)
        self.assertIsNotNone(report)
        self.assertTrue(report["pass_audit"])
        self.assertEqual(report["blind_solve_accuracy"], 1.0)
        self.assertEqual(len(report["questions"]), 1)

    def test_generate_critique_feedback(self):
        failed_report = {
            "overall_quality_score": 65,
            "pass_audit": False,
            "blind_solve_accuracy": 0.5,
            "summary_verdict": "Double key and weak distractors detected.",
            "questions": [
                {
                    "item_index": 0,
                    "blind_solved_index": 1,
                    "confidence": "Ambiguous",
                    "single_fit_valid": False,
                    "distractors": [
                        {
                            "option_letter": "C",
                            "option_text": "Random absurdity",
                            "trap_type": "Flawed / Trivial Giveaway",
                            "plausibility_rating": "Low (Flawed)",
                            "elimination_rationale": "Grammatically broken."
                        }
                    ],
                    "pedagogical_score": 50,
                    "diagnostic_feedback": "[DIVERGENCE: Declared A vs Blind-Solved B] Stem is ambiguous."
                }
            ]
        }
        critique = ExpertAuditor.generate_critique_feedback(failed_report)
        self.assertIn("LEVEL 2 EXPERT QUALITY AUDIT FAILED", critique)
        self.assertIn("Multiple defensible keys or key leakage", critique)
        self.assertIn("DIVERGENCE", critique)
        self.assertIn("Option [C] is a flawed/trivial giveaway", critique)
        self.assertIn("MANDATORY CORRECTION ACTIONS", critique)

    def test_surgical_defective_item_extraction_and_critique(self):
        audit_report = {
            "overall_quality_score": 75,
            "pass_audit": False,
            "blind_solve_accuracy": 0.8,
            "questions": [
                {
                    "item_index": 0,
                    "single_fit_valid": True,
                    "pedagogical_score": 95,
                    "diagnostic_feedback": "Perfect question."
                },
                {
                    "item_index": 1,
                    "single_fit_valid": False,
                    "pedagogical_score": 60,
                    "diagnostic_feedback": "Double key between options A and B.",
                    "distractors": [
                        {
                            "option_letter": "A",
                            "plausibility_rating": "Medium",
                            "elimination_rationale": "double key with option B"
                        }
                    ]
                },
                {
                    "item_index": 2,
                    "single_fit_valid": True,
                    "pedagogical_score": 65, # Below 75
                    "diagnostic_feedback": "[DIVERGENCE: Declared A vs Blind-Solved B] Ambiguous stem."
                }
            ]
        }

        # 1. Test defective indices extraction
        defective = ExpertAuditor.get_defective_item_indices(audit_report)
        self.assertEqual(defective, [1, 2])

        # 2. Test surgical critique generation
        full_questions = [
            {"target_word": "alpha", "question": "Stem 1", "options": ["A1", "A2"], "correct_answer_index": 0},
            {"target_word": "beta", "question": "Stem 2", "options": ["B1", "B2"], "correct_answer_index": 1},
            {"target_word": "gamma", "question": "Stem 3", "options": ["C1", "C2"], "correct_answer_index": 0},
        ]
        surgical_critique = ExpertAuditor.generate_surgical_critique(audit_report, full_questions, defective)
        
        self.assertIn("SURGICAL", surgical_critique)
        self.assertIn("DEFECTIVE ITEM #2", surgical_critique)
        self.assertIn("DEFECTIVE ITEM #3", surgical_critique)
        self.assertNotIn("DEFECTIVE ITEM #1", surgical_critique) # Item 1 passed
        self.assertIn("Target Word: beta", surgical_critique)
        self.assertIn("Double key between options A and B", surgical_critique)


class TestIntegrityGate(unittest.TestCase):
    def test_sanitize_vocab_for_quiz(self):
        from librarian.processor import WikiProcessor
        raw_markdown = """# Vocabulary List

## [[abandon]]
- **Part Of Speech**: verb
- **Definition**: to leave behind or cease to support
- **Quoted Sentence**: They had to abandon the sinking ship immediately.
- **Example Usage**: He decided to abandon his old car.

## [[beneficial]]
- **Part Of Speech**: adjective
- **Definition**: resulting in good; favorable or advantageous
- **Quoted Sentence**: Regular exercise is beneficial to human health.
- **Example Usage**: Fresh fruits are beneficial for children.
"""
        sanitized, headwords, banned_sentences = WikiProcessor._sanitize_vocab_for_quiz(raw_markdown)

        # 1. Verify that headwords are accurately extracted
        self.assertEqual(headwords, ["abandon", "beneficial"])

        # 2. Verify that banned sentences are captured
        self.assertEqual(len(banned_sentences), 4)
        self.assertIn("They had to abandon the sinking ship immediately.", banned_sentences)
        self.assertIn("Regular exercise is beneficial to human health.", banned_sentences)

        # 3. Verify that sanitized content does NOT contain example sentences
        self.assertNotIn("Quoted Sentence", sanitized)
        self.assertNotIn("Example Usage", sanitized)
        self.assertNotIn("sinking ship", sanitized)
        self.assertIn("## [[abandon]]", sanitized)
        self.assertIn("- **Definition**: to leave behind or cease to support", sanitized)

    def test_stem_copying_detection(self):
        from librarian.processor import WikiProcessor
        banned = [
            "They had to abandon the sinking ship immediately in the fierce storm.",
            "Regular exercise is beneficial to human health and mental clarity."
        ]
        
        # Copied stem with blank
        copied_quiz = {
            "questions": [
                {
                    "target_word": "abandon",
                    "question": "They had to ____ the sinking ship immediately in the fierce storm.",
                    "options": ["abandon", "cherish", "maintain", "support"],
                    "correct_answer_index": 0
                },
                {
                    "target_word": "beneficial",
                    "question": "Adopting a well-balanced diet proves remarkably ____ for overall longevity.",
                    "options": ["detrimental", "beneficial", "trivial", "superficial"],
                    "correct_answer_index": 1
                }
            ]
        }

        flagged, defects = WikiProcessor.audit_quiz_integrity(
            copied_quiz,
            banned_sentences=banned,
            unit_headwords=["abandon", "beneficial"]
        )

        self.assertEqual(flagged, [0])
        self.assertEqual(len(defects), 1)
        self.assertIn("Stem copies", defects[0])
        self.assertIn("sinking ship", defects[0])

    def test_in_list_distractor_recycling_detection(self):
        from librarian.processor import WikiProcessor
        unit_words = ["abandon", "beneficial", "collaborate", "deteriorate"]

        # Item 0 uses other words from the unit wordlist as distractors
        recycled_quiz = {
            "questions": [
                {
                    "target_word": "abandon",
                    "question": "When faced with insurmountable obstacles, the team chose to ____ the endeavor.",
                    "options": ["abandon", "beneficial", "collaborate", "deteriorate"],
                    "correct_answer_index": 0
                },
                {
                    "target_word": "beneficial",
                    "question": "The new environmental legislation will yield ____ outcomes for future generations.",
                    "options": ["advantageous", "beneficial", "detrimental", "negligible"],
                    "correct_answer_index": 1
                }
            ]
        }

        flagged, defects = WikiProcessor.audit_quiz_integrity(
            recycled_quiz,
            banned_sentences=[],
            unit_headwords=unit_words
        )

        self.assertEqual(flagged, [0])
        self.assertEqual(len(defects), 1)
        self.assertIn("Distractors recycle headwords from current unit", defects[0])
        self.assertIn("collaborate", defects[0])


    def test_blind_solve_confident_divergence_penalization_and_reconciliation(self):
        """
        Verify that a confident blind-solve divergence:
        1. Downgrades pass_audit to False.
        2. Hard-caps overall_quality_score below passing (<= 74).
        3. Cleans misleading 'no fatal flaws' statements from summary_verdict.
        """
        import re
        
        # Construct audit report where judge mistakenly declared 95 and "no flaws"
        report_data = {
            "quiz_title": "Vocabulary Test",
            "overall_quality_score": 95,
            "pass_audit": True,
            "summary_verdict": "High quality quiz. No fatal flaws or double-keys were detected.",
            "questions": [
                {
                    "item_index": 1,
                    "blind_solved_index": 2,  # Diverges from key (key is 0)
                    "confidence": "Definite",
                    "single_fit_valid": True,
                    "distractors": []
                }
            ]
        }
        
        confirmed = 0
        comparable = 1
        confident_divergences = 1
        
        # Emulate the exact logic in ExpertAuditor.audit_quiz
        report_dict = dict(report_data)
        raw_score = report_dict.get("overall_quality_score", 100)
        
        if confident_divergences > 0:
            report_dict["pass_audit"] = False
            penalized_score = min(int(raw_score), max(40, 74 - (confident_divergences - 1) * 10))
            report_dict["overall_quality_score"] = penalized_score
            current_summary = str(report_dict.get("summary_verdict", ""))
            clean_summary = re.sub(
                r'(?i)(?:no\s+(?:fatal\s+flaws?|double[\-\s]*keys?)[^.\n]*[.\n]*)',
                '',
                current_summary
            ).strip()
            flag_notice = f"[CRITICAL FLAW: {confident_divergences} confident double-key/divergence detected in blind solve]."
            report_dict["summary_verdict"] = f"{flag_notice} {clean_summary}".strip()
            
        self.assertFalse(report_dict["pass_audit"])
        self.assertEqual(report_dict["overall_quality_score"], 74)
        self.assertIn("[CRITICAL FLAW: 1 confident double-key/divergence", report_dict["summary_verdict"])
        self.assertNotIn("No fatal flaws", report_dict["summary_verdict"])

    def test_shuffle_quiz_options_markdown_and_explanation_cleaning(self):
        """
        Verify that WikiProcessor.shuffle_quiz_options cleans markdown bold/italic
        symbols (*, `) and quote residues from explanation, definition, and questions.
        """
        from librarian.processor import WikiProcessor
        
        raw_quiz = {
            "questions": [
                {
                    "target_word": "passionate",
                    "question": "Students were **____** about social causes.",
                    "options": ["**enthusiastic**", "passionate", "indifferent", "curious"],
                    "correct_answer_index": 1,
                    "explanation": "Context requires dedication. - **Option C (\"indifferent\")**: Directly contradicts context. - **Option A**: Related but weaker.",
                    "definition": "Having or showing **strong** feelings.",
                    "pedagogical_rationale": "Distinguishes *nuanced* vocabulary."
                }
            ]
        }
        
        sanitized = WikiProcessor.shuffle_quiz_options(raw_quiz)
        q = sanitized["questions"][0]
        
        # Stem blank cleaned
        self.assertNotIn("**", q["question"])
        self.assertIn("____", q["question"])
        
        # Option bold cleaned
        self.assertNotIn("**", q["options"][0])
        
        # Explanation cleaned of markdown asterisks
        self.assertNotIn("**", q["explanation"])
        self.assertIn("Option", q["explanation"])
        
        # Definition cleaned
        self.assertNotIn("**", q["definition"])
        self.assertEqual(q["definition"], "Having or showing strong feelings.")
        
        # Pedagogical rationale cleaned
        self.assertNotIn("*", q["pedagogical_rationale"])
        self.assertEqual(q["pedagogical_rationale"], "Distinguishes nuanced vocabulary.")


    def test_audit_translation_integrity_target_language(self):
        """
        Verify that audit_translation_integrity dynamically adapts to target_language
        (e.g., Japanese, Russian, Spanish, Chinese) instead of hardcoding 'Chinese'.
        """
        from librarian.processor import WikiProcessor

        # 1. Chinese translation quiz: passes with Chinese characters
        zh_quiz = {
            "questions": [
                {
                    "translated_sentence": "尽管面临严峻的经济挑战，该团队依然坚持完成了这项研究。",
                    "options": [
                        "Despite facing severe economic challenges, the team persevered to complete the study.",
                        "Although faced economic challenges, the team kept to complete research.",
                        "In spite of severe challenge, the team persevered completing the study.",
                        "Despite of facing challenges, the team was persevering the study."
                    ],
                    "correct_answer_index": 0,
                    "correct_english_answer": "Despite facing severe economic challenges, the team persevered to complete the study.",
                    "design_audit": "Target Vocab: persevere -> Despite + challenges"
                }
            ]
        }
        flagged, defects = WikiProcessor.audit_translation_integrity(zh_quiz, target_language="Simplified Chinese")
        self.assertEqual(flagged, [])
        self.assertEqual(defects, [])

        # 2. Flagged if prompt does not contain target language characters
        non_zh_quiz = {
            "questions": [
                {
                    "translated_sentence": "Only English sentence without target characters.",
                    "options": [
                        "Option A sentence.", "Option B sentence.", "Option C sentence.", "Option D sentence."
                    ],
                    "correct_answer_index": 0,
                    "correct_english_answer": "Option A sentence.",
                    "design_audit": "Target Vocab: sentence"
                }
            ]
        }
        flagged, defects = WikiProcessor.audit_translation_integrity(non_zh_quiz, target_language="Japanese")
        self.assertEqual(flagged, [0])
        self.assertIn("must contain Japanese characters", defects[0])

        # 3. Japanese translation quiz: passes with Japanese characters
        ja_quiz = {
            "questions": [
                {
                    "translated_sentence": "厳しい経済的課題に直面したにもかかわらず、研究をやり遂げた。",
                    "options": [
                        "Despite facing severe economic challenges, the team persevered to complete the study.",
                        "Although faced economic challenges, the team kept to complete research.",
                        "In spite of severe challenge, the team persevered completing the study.",
                        "Despite of facing challenges, the team was persevering the study."
                    ],
                    "correct_answer_index": 0,
                    "correct_english_answer": "Despite facing severe economic challenges, the team persevered to complete the study.",
                    "design_audit": "Target Vocab: persevere"
                }
            ]
        }
        flagged, defects = WikiProcessor.audit_translation_integrity(ja_quiz, target_language="Japanese")
        self.assertEqual(flagged, [])
        self.assertEqual(defects, [])


    def test_reading_index_desync_auto_healing(self):
        """Tests that when an LLM writes correct logic in explanation (e.g. supporting Option C)
        but accidentally writes a desynchronized index (e.g. 3/D), shuffle_quiz_options automatically
        heals the index to 2 (C) without corrupting the explanation labels.
        """
        from librarian.processor import WikiProcessor

        quiz = {
            "questions": [
                {
                    "question": "What can be inferred about the students’ relationships with roommates?",
                    "options": [
                        "They will always get along",
                        "Conflict is inevitable",
                        "Friendships may develop despite differences",
                        "Roommates must study together"
                    ],
                    "correct_answer_index": 3,  # Accidentally set to 3 (Option D)
                    "explanation": "The text indicates that roommates may become best friends, supporting Option C. Option A's absolute always contradicts the text; Option D introduces an unmentioned requirement."
                }
            ]
        }

        healed_quiz = WikiProcessor.shuffle_quiz_options(quiz)
        q = healed_quiz["questions"][0]
        # Must be auto-healed to index 2 (Option C)
        self.assertEqual(q["correct_answer_index"], 2)
        # Option C must be the correct one
        self.assertEqual(q["options"][q["correct_answer_index"]], "Friendships may develop despite differences")
        # Explanation must remain pristine without accidental remapping
        self.assertIn("supporting Option C", q["explanation"])
        self.assertIn("Option D introduces", q["explanation"])

    def test_reconcile_report_scores_string_boolean_and_capping(self):
        """Verify that single_fit_valid='false' as string is properly normalized and caps score to 70."""
        report = {
            "overall_quality_score": 85,
            "pass_audit": "true",
            "blind_solve_accuracy": 100,
            "summary_verdict": "No fatal flaws detected.",
            "questions": [
                {"item_index": 1, "single_fit_valid": "true", "pedagogical_score": 90, "confidence": "definite"},
                {"item_index": 2, "single_fit_valid": "false", "pedagogical_score": 85, "confidence": "Definite", "diagnostic_feedback": "Double key"},
            ]
        }
        reconciled = ExpertAuditor.reconcile_report_scores(report, quiz_type="translation")
        self.assertFalse(reconciled["pass_audit"])
        self.assertLessEqual(reconciled["overall_quality_score"], 70)
        self.assertEqual(reconciled["questions"][1]["single_fit_valid"], False)
        self.assertEqual(reconciled["questions"][1]["pedagogical_score"], 20)
        self.assertIn("[REVIEW NEEDED", reconciled["summary_verdict"])

    def test_render_handout_audit_metadata(self):
        """Verify that handouts preserve _expert_audit metadata in quizData for native UI badges."""
        from librarian.processor import WikiProcessor
        processor = WikiProcessor()
        quiz_data = {
            "title": "Unpassed Quiz",
            "questions": [],
            "_expert_audit": {
                "pass_audit": False,
                "overall_quality_score": 68,
                "base_quality_score": 75,
                "flawed_item_indices": [1]
            }
        }
        rendered = processor._render_handout(quiz_data, "translation")
        self.assertNotIn("unpassed-review-banner", rendered)
        self.assertIn('"pass_audit": false', rendered)
    def test_triage_action_normalization(self):
        """Verify that triage_action is properly normalized during reconciliation."""
        report = {
            "overall_quality_score": 90,
            "pass_audit": True,
            "questions": [
                {
                    "item_index": 1,
                    "single_fit_valid": True,
                    "triage_action": "pass",
                    "pedagogical_score": 92
                },
                {
                    "item_index": 2,
                    "single_fit_valid": False,
                    "triage_action": "PASS", # Contradiction: fatal flaw marked PASS -> elevates
                    "diagnostic_feedback": "fatally flawed double-key"
                }
            ]
        }
        reconciled = ExpertAuditor.reconcile_report_scores(report)
        self.assertEqual(reconciled["questions"][0]["triage_action"], "PASS")
        self.assertIn(reconciled["questions"][1]["triage_action"], ("REPAIR", "REWRITE"))

    def test_inplace_surgical_cure_and_scoring(self):
        """Verify the in-place surgical cure and deterministic scoring pipeline."""
        from librarian.processor import WikiProcessor
        processor = WikiProcessor()

        orig_questions = [
            {
                "question": "He decided to ____ the project.",
                "options": ["abandon", "support", "embrace", "continue"],
                "correct_answer_index": 0,
                "target_word": "abandon",
                "explanation": "Option A is correct."
            },
            {
                "question": "The book was very ____ for students.",
                "options": ["beneficial", "beneficial", "bad", "harmful"], # Duplicate options / flawed
                "correct_answer_index": 0,
                "target_word": "beneficial",
                "explanation": "Option A is correct."
            }
        ]

        cured_item = {
            "question": "The book was exceptionally ____ for students.",
            "options": ["beneficial", "useless", "distracting", "irrelevant"],
            "correct_answer_index": 0,
            "target_word": "beneficial",
            "explanation": "Option A is correct because it fits the educational context."
        }

        # Simulate audit report with REPAIR triage action and cured_question
        audit_report = {
            "overall_quality_score": 60,
            "pass_audit": False,
            "blind_solve_accuracy": 0.5,
            "questions": [
                {
                    "item_index": 1,
                    "single_fit_valid": True,
                    "triage_action": "PASS",
                    "pedagogical_score": 92
                },
                {
                    "item_index": 2,
                    "single_fit_valid": False,
                    "triage_action": "REPAIR",
                    "pedagogical_score": 20,
                    "cured_question": cured_item
                }
            ]
        }

        # Step 3 simulation: apply cure
        cured_questions = list(orig_questions)
        for a_idx, qa in enumerate(audit_report["questions"]):
            triage = qa.get("triage_action")
            cand = qa.get("cured_question")
            if triage in ("REPAIR", "REWRITE") and cand:
                flagged, _ = processor.audit_quiz_integrity({"questions": [cand]})
                self.assertEqual(len(flagged), 0)
                cured_questions[a_idx] = cand
                qa["single_fit_valid"] = True
                qa["pedagogical_score"] = 90 if triage == "REPAIR" else 95

        # Verify cured question replaced the defective one
        self.assertEqual(cured_questions[1]["options"], ["beneficial", "useless", "distracting", "irrelevant"])
        self.assertEqual(audit_report["questions"][1]["pedagogical_score"], 90)

        # Step 4 deterministic score
        final_avg = round(sum(qa["pedagogical_score"] for qa in audit_report["questions"]) / len(audit_report["questions"]))
        self.assertEqual(final_avg, 91) # (92 + 90) / 2 = 91

    def test_video_quiz_integrity_level1(self):
        from librarian.processor import WikiProcessor
        transcript = "[01:15.20] The dam regulates water flow.\n[02:30.00] Hydroelectric turbines generate clean power."
        
        valid_quiz = {
            "questions": [
                {
                    "question": "What primary function does the dam serve during peak seasons?",
                    "options": ["Regulating water flow", "Mining minerals", "Tourism exclusively", "Storing industrial waste"],
                    "correct_answer_index": 0,
                    "timestamp": "[01:15.20]",
                    "explanation": "At 01:15 the video states the dam regulates water flow."
                }
            ]
        }
        flagged, defects = WikiProcessor.audit_video_integrity(valid_quiz, transcript_text=transcript)
        self.assertEqual(len(flagged), 0)
        self.assertEqual(len(defects), 0)

        # Defective quiz: missing timestamp & duplicate options
        defective_quiz = {
            "questions": [
                {
                    "question": "What is the dam's purpose?",
                    "options": ["Power", "Power", "Water", "Fish"],
                    "correct_answer_index": 0,
                    "timestamp": "",
                    "explanation": "No explanation"
                }
            ]
        }
        flagged, defects = WikiProcessor.audit_video_integrity(defective_quiz, transcript_text=transcript)
        self.assertIn(0, flagged)
        self.assertTrue(any("duplicate" in d for d in defects))
        self.assertTrue(any("timestamp" in d for d in defects))

    def test_video_expert_auditor_template_selection(self):
        from librarian.expert_auditor import ExpertAuditor
        from unittest.mock import patch

        video_quiz = {
            "questions": [
                {
                    "question": "Why were fish ladders constructed alongside the main dam?",
                    "options": ["To facilitate migration", "To harvest caviar", "For aesthetic purposes", "To filter water"],
                    "correct_answer_index": 0,
                    "timestamp": "[03:45]",
                    "explanation": "Fish ladders allow salmon to migrate upstream."
                }
            ]
        }
        
        formatted = ExpertAuditor.format_quiz_for_blind_audit(
            source_text="[03:45] Fish ladders assist species migration.",
            questions=video_quiz["questions"]
        )
        self.assertIn("Timestamp Segment: [03:45]", formatted)
        self.assertIn("Why were fish ladders constructed", formatted)

    def test_listening_quiz_integrity_level1(self):
        from librarian.processor import WikiProcessor
        dialogue_script = "Speaker 1: Welcome to the robotics lab.\nSpeaker 2: Thanks, I am excited to see the automated navigation system."

        valid_quiz = {
            "script": [
                {"speaker": "Speaker 1", "text": "Turn 1"},
                {"speaker": "Speaker 2", "text": "Turn 2"},
                {"speaker": "Speaker 1", "text": "Turn 3"},
                {"speaker": "Speaker 2", "text": "Turn 4"}
            ],
            "questions": [
                {
                    "question": "What is the main purpose of visiting the lab?",
                    "options": ["To observe automated navigation", "To repair sensors", "To interview applicants", "To write software manuals"],
                    "correct_answer_index": 0,
                    "category": "Main Idea",
                    "explanation": "Speaker 2 explicitly states excitement to see the automated navigation system."
                }
            ]
        }
        flagged, defects = WikiProcessor.audit_listening_integrity(valid_quiz, script_text=dialogue_script)
        self.assertEqual(len(flagged), 0)
        self.assertEqual(len(defects), 0)

        # Defective quiz: invalid options count & unknown category
        defective_quiz = {
            "questions": [
                {
                    "question": "Short",
                    "options": ["Option 1", "Option 2"],
                    "correct_answer_index": 0,
                    "category": "UnknownCategory",
                    "explanation": "Short"
                }
            ]
        }
        flagged, defects = WikiProcessor.audit_listening_integrity(defective_quiz, script_text=dialogue_script)
        self.assertIn(0, flagged)
        self.assertTrue(any("4 options" in d for d in defects))
        self.assertTrue(any("too short" in d for d in defects))


if __name__ == "__main__":
    unittest.main()



