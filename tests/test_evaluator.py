import unittest
import tempfile
from pathlib import Path
from typing import get_args

from librarian.evaluator import (
    LogEvaluator,
    _normalize_text,
    _clean_core,
    _ngram_coverage,
    _is_hallucinated_quote,
    _score_schema,
    _score_verbatim,
    _score_pedagogy,
    _score_uniqueness,
    VALID_POS_SET,
    W_SCHEMA,
    W_VERBATIM,
    W_PEDAGOGY,
    W_UNIQUENESS,
)
from librarian.schemas import PARTS_OF_SPEECH, EXPRESSION_TYPES


class TestNormalizerAndCleanCore(unittest.TestCase):
    def test_markdown_and_punctuation_stripping(self):
        raw = '"his solemn, **secular** oath..."'
        cleaned = _clean_core(raw)
        self.assertEqual(cleaned, "his solemn secular oath")

    def test_ellipses_and_brackets(self):
        raw = "...working on the side [of] the weak…"
        cleaned = _clean_core(raw)
        self.assertEqual(cleaned, "working on the side of the weak")

    def test_smart_quotes(self):
        raw = "“He said, «hello» to everyone’s surprise”"
        cleaned = _clean_core(raw)
        self.assertEqual(cleaned, "he said hello to everyones surprise")


class TestVerbatimEvaluation(unittest.TestCase):
    def setUp(self):
        self.source = """CONTENT:
The prime minister took his solemn, secular oath before parliament.
He dedicated his entire life to public service."""

    def test_verbatim_with_markdown_in_quote(self):
        items = [{
            "word": "secular",
            "quoted_sentence": "The prime minister took his solemn, **secular** oath before parliament."
        }]
        score, flags = _score_verbatim(items, "vocabulary", self.source)
        self.assertEqual(score, W_VERBATIM)
        self.assertEqual(flags, [])

    def test_word_not_in_quote_flagged(self):
        items = [{
            "word": "oppressed",
            "quoted_sentence": "He dedicated his entire life to public service."
        }]
        score, flags = _score_verbatim(items, "vocabulary", self.source)
        self.assertEqual(score, 0.0)
        self.assertTrue(any("does not appear in quoted sentence" in f for f in flags))

    def test_hallucinated_quote_flagged(self):
        items = [{
            "word": "succinctly",
            "quoted_sentence": "*Not present in text, but inferred from the speech context.*"
        }]
        score, flags = _score_verbatim(items, "vocabulary", self.source)
        self.assertEqual(score, 0.0)
        self.assertTrue(any("Hallucinated quote" in f for f in flags))

    def test_non_extraction_task_returns_none(self):
        items = [{"question": "What happened?"}]
        score, flags = _score_verbatim(items, "quiz", self.source)
        self.assertIsNone(score)
        self.assertEqual(flags, [])

    def test_slot_expressions_and_lemmatized_word_in_quote(self):
        source = "CONTENT:\nExcessive exploitation of natural resources and greenhouse gas emissions pose a grave threat to the earth's essential ecology.\nWith the degradation of ecosystems, life will decline."
        items = [
            {
                "word": "pose [something] grave threat to [entity]",
                "quoted_sentence": "Excessive exploitation of natural resources and greenhouse gas emissions pose a grave threat to the earth's essential ecology."
            },
            {
                "word": "degrade",
                "quoted_sentence": "With the degradation of ecosystems, life will decline."
            }
        ]
        score, flags = _score_verbatim(items, "expressions", source)
        self.assertEqual(score, W_VERBATIM)
        self.assertEqual(flags, [])



class TestPedagogyEvaluation(unittest.TestCase):
    def test_all_schema_pos_are_valid(self):
        expected_pos = set(get_args(PARTS_OF_SPEECH))
        self.assertEqual(VALID_POS_SET, expected_pos)
        self.assertIn("noun", VALID_POS_SET)
        self.assertIn("verb", VALID_POS_SET)
        self.assertIn("adjective", VALID_POS_SET)

    def test_phrasal_verb_and_slot_headword_accepted(self):
        items = [{
            "word": "put [something] out",
            "part_of_speech": "phrasal verb",
            "definition": "to extinguish something such as a fire or cigarette",
            "quoted_sentence": "Firefighters managed to put out the blaze.",
            "example_usage": "Please put your cigarette out before entering.",
        }]
        score, flags = _score_pedagogy(items, "expressions")
        self.assertEqual(score, W_PEDAGOGY)
        self.assertEqual(flags, [])

    def test_invalid_pos_flagged(self):
        items = [{
            "word": "alerting",
            "part_of_speech": "verb (present participle)",
            "definition": "warning someone",
            "quoted_sentence": "He was alerting the citizens.",
            "example_usage": "The guard was alerting everyone about the danger.",
        }]
        score, flags = _score_pedagogy(items, "vocabulary")
        self.assertEqual(score, 0.0)
        self.assertTrue(any("invalid PoS" in f for f in flags))

    def test_non_original_example_flagged(self):
        items = [{
            "word": "solemn",
            "part_of_speech": "adjective",
            "definition": "formal and dignified",
            "quoted_sentence": "He took a solemn oath.",
            "example_usage": "He took a solemn oath.",
        }]
        score, flags = _score_pedagogy(items, "vocabulary")
        self.assertEqual(score, 0.0)
        self.assertTrue(any("non-original example usage" in f for f in flags))

    def test_quiz_pedagogy_validation(self):
        valid_quiz = [{
            "question": "Choose the correct word: _____",
            "options": ["cat", "dog", "bird", "fish"],
            "correct_answer_index": 0,
            "explanation": "Cat fits the context.",
        }]
        score, flags = _score_pedagogy(valid_quiz, "quiz")
        self.assertEqual(score, W_PEDAGOGY)
        self.assertEqual(flags, [])


class TestNormalizedScoringAndLogAudit(unittest.TestCase):
    def test_composite_score_normalized_for_quiz(self):
        log_data = {
            "log_name": "test_quiz.log",
            "task": "quiz",
            "model": "test_model",
            "status": "SUCCESS",
            "user_prompt": "Generate a quiz",
            "parsed_json": {
                "questions": [{
                    "question": "What is the answer?",
                    "options": ["A", "B", "C", "D"],
                    "correct_answer_index": 1,
                    "explanation": "B is correct because of reasons.",
                }]
            },
            "raw_response": '{"questions": [...]}'
        }
        eval_result = LogEvaluator.evaluate_log(log_data)
        self.assertIsNone(eval_result["scores"]["verbatim_faithfulness"])
        self.assertEqual(eval_result["composite_score"], 100.0)

    def test_failed_status_excluded_from_quality_average(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            success_log = tmp_path / "20260901_success.log"
            success_log.write_text("""=== TASK: quiz ===
=== MODEL: test_model ===
=== TIMESTAMP: Wed Sep  2 20:00:00 2026 ===
=== STATUS: SUCCESS ===

--- USER PROMPT ---
Generate quiz

--- RAW RESPONSE ---
{
  "questions": [
    {
      "question": "Q1?",
      "options": ["A", "B", "C", "D"],
      "correct_answer_index": 0,
      "explanation": "Exp"
    }
  ]
}
""", encoding="utf-8")

            failed_log = tmp_path / "20260901_failed.log"
            failed_log.write_text("""=== TASK: quiz ===
=== MODEL: test_model ===
=== TIMESTAMP: Wed Sep  2 20:01:00 2026 ===
=== STATUS: FAILED ===
=== FAILURE_CATEGORY: TIMEOUT ===

--- USER PROMPT ---
Generate quiz

--- RAW RESPONSE ---
""", encoding="utf-8")

            result = LogEvaluator.audit_all_logs(tmp_path, use_cache=False)
            hero = result["hero_board"]
            self.assertEqual(len(hero), 1)
            model_stat = hero[0]
            self.assertEqual(model_stat["model"], "test_model")
            self.assertEqual(model_stat["runs"], 2)
            self.assertEqual(model_stat["success_runs"], 1)
            self.assertEqual(model_stat["failed_runs"], 1)
            self.assertEqual(model_stat["composite_score"], 100.0)


if __name__ == "__main__":
    unittest.main()
