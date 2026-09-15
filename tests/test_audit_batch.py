import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from librarian.audit_batch import (
    discover_quiz_units,
    load_quiz_data,
    resolve_source_text,
    run_batch,
    format_summary,
)
from librarian.expert_auditor import ExpertAuditor


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# Deliberately includes a curly brace inside a string to prove the JSON extraction
# survives nested braces (a naive regex would break here).
QUIZ_DATA = {
    "title": "Test Reading Quiz",
    "questions": [
        {
            "question": "What is the capital of France?",
            "options": ["Berlin", "Paris", "Madrid", "Rome"],
            "correct_answer_index": 1,
            "explanation": "Paris is the capital. Option A {curly brace test} is wrong.",
        },
        {
            "question": "2 + 2 =",
            "options": ["3", "4", "5", "6"],
            "correct_answer_index": 1,
        },
    ],
    # Verdict embedded when the handout was generated (pre-fix, biased: FAIL).
    "_expert_audit": {"pass_audit": False, "blind_solve_accuracy": 0.5},
}


class TestAuditBatch(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.unit = "Test_Unit_1"
        # Unit source text
        _write(self.root / "wiki" / self.unit / "sources" / f"{self.unit}.md",
               "# Source\nParis is in France.\n")
        # Rendered handout with embedded quizData
        payload = json.dumps(QUIZ_DATA, ensure_ascii=False)
        html = f"<html><body><script>const quizData = {payload};\nvar app = 1;</script></body></html>"
        _write(self.root / "wiki" / self.unit / "handouts" / f"{self.unit}_reading_quiz.html", html)

    def tearDown(self):
        self._tmp.cleanup()

    def test_discover_finds_reading_handout(self):
        refs = discover_quiz_units(project_root=str(self.root))
        self.assertEqual(len(refs), 1)
        self.assertEqual(refs[0]["unit"], self.unit)
        self.assertEqual(refs[0]["template"], "reading")

    def test_load_quiz_data_handles_nested_braces(self):
        refs = discover_quiz_units(project_root=str(self.root))
        data = load_quiz_data(refs[0]["path"])
        self.assertIsNotNone(data)
        self.assertEqual(len(data["questions"]), 2)
        # The curly brace inside the string survived the parse.
        self.assertIn("{curly brace test}", data["questions"][0]["explanation"])
        self.assertIn("_expert_audit", data)

    def test_load_quiz_data_missing_returns_none(self):
        self.assertIsNone(load_quiz_data(str(self.root / "does_not_exist.html")))
        # File with no quizData injection point.
        _write(self.root / "wiki" / "X" / "handouts" / "X_vocabulary_quiz.html", "<html>no data</html>")
        self.assertIsNone(load_quiz_data(str(self.root / "wiki" / "X" / "handouts" / "X_vocabulary_quiz.html")))

    def test_resolve_source_text(self):
        text = resolve_source_text(self.unit, project_root=str(self.root))
        self.assertIn("Paris is in France.", text)

    def test_run_batch_pass_with_flip(self):
        fake_report = {
            "pass_audit": True,
            "blind_solve_accuracy": 1.0,
            "overall_quality_score": 95,
            "blind_solve_confident_divergences": 0,
            "summary_verdict": "All items single-fit.",
        }
        with mock.patch.object(ExpertAuditor, "audit_quiz", return_value=fake_report) as m:
            summary = run_batch(project_root=str(self.root), max_workers=2)
            self.assertEqual(m.call_count, 1)

        self.assertEqual(summary["total"], 1)
        self.assertEqual(summary["audited"], 1)
        self.assertEqual(summary["passed"], 1)
        self.assertEqual(summary["failed"], 0)
        self.assertEqual(summary["errors"], 0)
        self.assertEqual(summary["avg_blind_solve_accuracy"], 1.0)

        row = summary["results"][0]
        self.assertTrue(row["passed"])
        self.assertFalse(row["prev_passed"])   # embedded verdict was FAIL
        self.assertTrue(row["changed"])          # FAIL -> PASS flip after de-bias
        self.assertEqual(row["item_count"], 2)

    def test_run_batch_error_row_when_no_questions(self):
        unit = "Broken_Unit"
        _write(self.root / "wiki" / unit / "handouts" / f"{unit}_vocabulary_quiz.html",
               "<html><script>const quizData = {};\nvar app = 1;</script></html>")
        with mock.patch.object(ExpertAuditor, "audit_quiz") as m:
            summary = run_batch(unit_filter=[unit], project_root=str(self.root))
            m.assert_not_called()   # no questions -> judge is never invoked

        self.assertEqual(summary["total"], 1)
        self.assertEqual(summary["errors"], 1)
        self.assertEqual(summary["audited"], 0)
        self.assertIn("No quizData/questions", summary["results"][0]["error"])

    def test_template_filter(self):
        with mock.patch.object(ExpertAuditor, "audit_quiz", return_value={"pass_audit": True}):
            only_voc = run_batch(template_filter="vocabulary", project_root=str(self.root))
            all_of = run_batch(project_root=str(self.root))
        self.assertEqual(only_voc["total"], 0)   # only a reading handout exists
        self.assertEqual(all_of["total"], 1)

    def test_empty_summary_when_nothing_to_do(self):
        summary = run_batch(project_root=str(self.root / "no_such_root"))
        self.assertEqual(summary["total"], 0)
        self.assertEqual(summary["results"], [])

    def test_progress_callback_invoked(self):
        events = []
        fake_report = {"pass_audit": True, "blind_solve_accuracy": 0.8, "overall_quality_score": 88}
        with mock.patch.object(ExpertAuditor, "audit_quiz", return_value=fake_report):
            run_batch(project_root=str(self.root),
                      progress_cb=lambda done, total, row: events.append((done, total)))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0], (1, 1))

    def test_format_summary_renders(self):
        fake_report = {"pass_audit": True, "blind_solve_accuracy": 0.8, "overall_quality_score": 88}
        with mock.patch.object(ExpertAuditor, "audit_quiz", return_value=fake_report):
            summary = run_batch(project_root=str(self.root))
        text = format_summary(summary)
        self.assertIn("BATCH RE-AUDIT SUMMARY", text)
        self.assertIn(self.unit, text)
        self.assertIn("PASS", text)
        self.assertIn("80%", text)
        self.assertIn("was FAIL -> now PASS", text)



class TestReadingIntegrityGate(unittest.TestCase):
    def setUp(self):
        self.passage = (
            "Artificial intelligence has witnessed unprecedented transformation over the past decade. "
            "Deep neural networks have facilitated breakthrough advancements in automated natural language translation. "
            "However, empirical validation remains indispensable to prevent subtle hallucinations and factual discrepancies. "
            "Educators must balance computational agility with rigorous pedagogical design."
        )

    def test_valid_reading_quiz_passes_cleanly(self):
        from librarian.processor import processor
        valid_quiz = {
            "questions": [
                {
                    "question": "What is the primary focus of the passage?",
                    "category": "Main Idea",
                    "options": ["AI and educational balance", "Hardware history", "Quantum physics", "Social networks"],
                    "correct_answer_index": 0
                },
                {
                    "question": "What has facilitated translation breakthroughs?",
                    "category": "Detail/Recall",
                    "options": ["Deep neural networks", "Manual books", "Random rules", "Simple math"],
                    "correct_answer_index": 0
                },
                {
                    "question": "What can be inferred regarding factual discrepancies?",
                    "category": "Inference",
                    "options": ["They require empirical validation", "They are impossible", "They improve accuracy", "They are harmless"],
                    "correct_answer_index": 0
                },
                {
                    "question": "What is the author's tone toward AI integration?",
                    "category": "Author's Tone/Purpose",
                    "options": ["Pragmatic and cautious", "Ecstatic", "Hostile", "Indifferent"],
                    "correct_answer_index": 0
                }
            ],
            "vocabulary": [
                {
                    "word": "indispensable",
                    "context_sentence": "However, empirical validation remains indispensable to prevent subtle hallucinations."
                }
            ]
        }
        flagged, defects = processor.audit_reading_integrity(valid_quiz, passage_text=self.passage)
        self.assertEqual(flagged, [])
        self.assertEqual(defects, [])

    def test_duplicate_options_and_invalid_key_caught(self):
        from librarian.processor import processor
        flawed_quiz = {
            "questions": [
                {
                    "question": "What is the central theme of the text?",
                    "category": "Main Idea",
                    "options": ["Duplicate", "Duplicate", "Choice 3", "Choice 4"],
                    "correct_answer_index": 9  # Invalid index
                }
            ],
            "vocabulary": []
        }
        flagged, defects = processor.audit_reading_integrity(flawed_quiz, passage_text=self.passage)
        self.assertIn(0, flagged)
        self.assertTrue(any("duplicate" in d.lower() for d in defects))
        self.assertTrue(any("invalid correct_answer_index" in d.lower() for d in defects))

    def test_monolithic_category_caught(self):
        from librarian.processor import processor
        monolithic_quiz = {
            "questions": [
                {"question": "Detail question 1 from passage?", "category": "Detail/Recall", "options": ["A", "B", "C", "D"], "correct_answer_index": 0},
                {"question": "Detail question 2 from passage?", "category": "Detail/Recall", "options": ["A", "B", "C", "D"], "correct_answer_index": 1},
                {"question": "Detail question 3 from passage?", "category": "Detail/Recall", "options": ["A", "B", "C", "D"], "correct_answer_index": 2},
                {"question": "Detail question 4 from passage?", "category": "Detail/Recall", "options": ["A", "B", "C", "D"], "correct_answer_index": 3},
            ],
            "vocabulary": []
        }
        flagged, defects = processor.audit_reading_integrity(monolithic_quiz, passage_text=self.passage)
        self.assertTrue(any("skill diversity" in d.lower() for d in defects))

    def test_hallucinated_vocab_sentence_caught(self):
        from librarian.processor import processor
        hallucinated_quiz = {
            "questions": [
                {"question": "What is the main idea?", "category": "Main Idea", "options": ["A", "B", "C", "D"], "correct_answer_index": 0}
            ],
            "vocabulary": [
                {"word": "fabricated", "context_sentence": "This completely imaginary sentence is absent from text."}
            ]
        }
        flagged, defects = processor.audit_reading_integrity(hallucinated_quiz, passage_text=self.passage)
        self.assertTrue(any("does not match verbatim text" in d.lower() for d in defects))


    def test_blind_solve_zero_based_vs_one_based_alignment(self):
        # Questions with declared keys: Item 0 -> Key 1 (B), Item 1 -> Key 2 (C)
        questions = [
            {"question": "Q1", "options": ["A", "B", "C", "D"], "correct_answer_index": 1},
            {"question": "Q2", "options": ["A", "B", "C", "D"], "correct_answer_index": 2},
        ]

        # Case 1: Auditor returns 0-based indexing (item_index 0, 1)
        audit_0_based = {
            "pass_audit": True,
            "overall_quality_score": 90,
            "questions": [
                {"item_index": 0, "blind_solved_index": 1, "confidence": "Definite", "single_fit_valid": True, "pedagogical_score": 90},
                {"item_index": 1, "blind_solved_index": 2, "confidence": "Definite", "single_fit_valid": True, "pedagogical_score": 90},
            ]
        }
        with mock.patch("librarian.expert_auditor.LLMClient.chat", return_value=audit_0_based):
            res = ExpertAuditor.audit_quiz(quiz_data={"questions": questions}, source_text="dummy")
            self.assertEqual(res["blind_solve_accuracy"], 1.0)
            self.assertEqual(res["blind_solve_confident_divergences"], 0)
            self.assertTrue(res["pass_audit"])

        # Case 2: Auditor returns 1-based indexing (item_index 1, 2)
        audit_1_based = {
            "pass_audit": True,
            "overall_quality_score": 90,
            "questions": [
                {"item_index": 1, "blind_solved_index": 1, "confidence": "Definite", "single_fit_valid": True, "pedagogical_score": 90},
                {"item_index": 2, "blind_solved_index": 2, "confidence": "Definite", "single_fit_valid": True, "pedagogical_score": 90},
            ]
        }
        with mock.patch("librarian.expert_auditor.LLMClient.chat", return_value=audit_1_based):
            res = ExpertAuditor.audit_quiz(quiz_data={"questions": questions}, source_text="dummy")
            self.assertEqual(res["blind_solve_accuracy"], 1.0)
            self.assertEqual(res["blind_solve_confident_divergences"], 0)
            self.assertTrue(res["pass_audit"])


if __name__ == "__main__":
    unittest.main()