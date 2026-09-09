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



if __name__ == "__main__":
    unittest.main()

