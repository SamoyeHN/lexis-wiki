import unittest
import tempfile
import json
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

    def test_quiz_reading_no_target_word_returns_none(self):
        # A reading/translation style quiz has no target_word -> nothing to verify -> N/A
        items = [{"question": "What is the main idea?", "options": ["A", "B", "C", "D"],
                  "correct_answer_index": 0, "explanation": "x"}]
        score, flags = _score_verbatim(items, "quiz", "Generate a reading quiz")
        self.assertIsNone(score)

    def test_quiz_target_word_in_wordlist_passes(self):
        source = "CONTENT:\n## [[tough]]\n## [[marathon]]\n## [[annual]]\n"
        items = [
            {"target_word": "tough", "options": ["a", "b", "tough", "c"], "correct_answer_index": 2},
            {"target_word": "marathon", "options": ["a", "marathon", "b", "c"], "correct_answer_index": 1},
        ]
        score, flags = _score_verbatim(items, "quiz", source)
        self.assertEqual(score, W_VERBATIM)
        self.assertEqual(flags, [])

    def test_quiz_hallucinated_target_word_flagged(self):
        source = "CONTENT:\n## [[tough]]\n## [[marathon]]\n"
        items = [
            {"target_word": "tough", "options": ["a", "b", "tough", "c"], "correct_answer_index": 2},
            {"target_word": "glaive", "options": ["glaive", "a", "b", "c"], "correct_answer_index": 0},
        ]
        score, flags = _score_verbatim(items, "quiz", source)
        self.assertEqual(score, round(W_VERBATIM / 2, 1))
        self.assertTrue(any("not found in supplied word list" in f for f in flags))

    def test_quiz_target_word_inflected_still_matches(self):
        # 'tougher' should match the headword 'tough' via shared-stem matching
        source = "CONTENT:\n## [[tough]]\n"
        items = [{"target_word": "tougher", "options": ["tougher", "a", "b", "c"], "correct_answer_index": 0}]
        score, flags = _score_verbatim(items, "quiz", source)
        self.assertEqual(score, W_VERBATIM)
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
        self.assertTrue(any("unoriginal duplicate of quoted_sentence" in f for f in flags))

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

    def test_quiz_with_duplicate_options_penalized(self):
        dup_quiz = [{
            "target_word": "tough",
            "question": "The run was ____.",
            "options": ["tough", "easy", "tough", "innovative"],
            "correct_answer_index": 0,
            "explanation": "Tough fits.",
        }]
        score, flags = _score_pedagogy(dup_quiz, "quiz")
        self.assertEqual(score, 0.0)
        self.assertTrue(any("duplicate options detected" in f for f in flags))

    def test_quiz_target_not_matching_options_slot_penalized(self):
        mismatch_quiz = [{
            "target_word": "tough",
            "question": "The run was ____.",
            "options": ["hard", "easy", "efficient", "innovative"],
            "correct_answer_index": 0,
            "explanation": "Hard fits.",
        }]
        score, flags = _score_pedagogy(mismatch_quiz, "quiz")
        self.assertEqual(score, 0.0)
        self.assertTrue(any("target 'tough' not matching options[0]" in f for f in flags))

    def test_quiz_target_inflected_in_correct_option_passes(self):
        inflected_quiz = [{
            "target_word": "assert",
            "question": "During the heated debate, she firmly ____ her position against the opposing council.",
            "options": ["asserted", "denied", "suggested", "questioned"],
            "correct_answer_index": 0,
            "explanation": "Asserted fits.",
        }]
        score, flags = _score_pedagogy(inflected_quiz, "quiz")
        self.assertEqual(score, W_PEDAGOGY)
        self.assertEqual(flags, [])

    def test_quiz_in_list_recycling_penalized(self):
        user_prompt = "## [[activism]]\n## [[obituary]]\n## [[specimen]]\n## [[aesthetic]]"
        recycled_quiz = [{
            "target_word": "activism",
            "question": "The people joined the ____ to promote social change.",
            "options": ["activism", "obituary", "specimen", "aesthetic"],
            "correct_answer_index": 0,
            "explanation": "Activism fits.",
        }]
        score, flags = _score_pedagogy(recycled_quiz, "quiz", user_prompt=user_prompt)
        self.assertEqual(score, 0.0)
        self.assertTrue(any("in-list distractor recycling" in f for f in flags))

    def test_quiz_multiple_blanks_flagged_and_penalized(self):
        double_blank_quiz = [{
            "target_word": "challenged",
            "question": "The defense attorney presented a novel ____ case that left the prosecution deeply ____ with unresolved ethical dilemmas.",
            "options": ["challenged", "tempted", "ordered", "disputed"],
            "correct_answer_index": 0,
            "explanation": "Challenged fits.",
        }]
        score, flags = _score_pedagogy(double_blank_quiz, "quiz")
        self.assertEqual(score, 0.0)
        self.assertTrue(any("Multiple blanks (2) detected" in f for f in flags))
        # Ensure evaluate_log flags fatal and caps score
        log_data = {
            "log_name": "test_double_blank.log",
            "task": "quiz",
            "model": "test_model",
            "status": "SUCCESS",
            "parsed_json": {"questions": double_blank_quiz},
            "raw_response": "{}"
        }
        res = LogEvaluator.evaluate_log(log_data)
        self.assertLessEqual(res["composite_score"], 59.0)
        self.assertEqual(res["status"], "REVIEW_NEEDED")

    def test_quiz_stem_leakage_flagged_and_penalized(self):
        leak_quiz = [{
            "target_word": "challenged",
            "question": "The legal ________ was challenged by the opposing party during the hearing.",
            "options": ["challenged", "tempted", "ordered", "disputed"],
            "correct_answer_index": 0,
            "explanation": "Challenged fits.",
        }]
        score, flags = _score_pedagogy(leak_quiz, "quiz")
        self.assertEqual(score, 0.0)
        self.assertTrue(any("Target word leaks verbatim into question stem" in f for f in flags))
        # Ensure evaluate_log flags fatal and caps score
        log_data = {
            "log_name": "test_leak.log",
            "task": "quiz",
            "model": "test_model",
            "status": "SUCCESS",
            "parsed_json": {"questions": leak_quiz},
            "raw_response": "{}"
        }
        res = LogEvaluator.evaluate_log(log_data)
        self.assertLessEqual(res["composite_score"], 59.0)
        self.assertEqual(res["status"], "REVIEW_NEEDED")


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


class TestEvaluatorImprovements(unittest.TestCase):
    def test_safe_str_handles_none_and_types(self):
        from librarian.evaluator import _safe_str
        self.assertEqual(_safe_str(None), "")
        self.assertEqual(_safe_str(None, default="N/A"), "N/A")
        self.assertEqual(_safe_str("  word  "), "word")
        self.assertEqual(_safe_str(123), "123")

    def test_extract_items_skips_empty_preferred_list(self):
        from librarian.evaluator import _extract_items
        # If model outputs empty vocabulary list but valid expressions list
        parsed = {
            "vocabulary": [],
            "expressions": [{"word": "turn up", "part_of_speech": "phrasal verb"}]
        }
        items = _extract_items(parsed, "vocabulary")
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["word"], "turn up")

    def test_extract_source_content_avoids_instruction_fallback(self):
        from librarian.evaluator import _extract_source_content
        # Prompt without CONTENT: or headings should return empty string
        prompt_without_content = "You are an assistant. Extract words."
        self.assertEqual(_extract_source_content(prompt_without_content), "")

        # Prompt with CONTENT: returns content correctly
        prompt_with_content = "You are an assistant.\n\nCONTENT:\nReal source sentence."
        self.assertEqual(_extract_source_content(prompt_with_content), "Real source sentence.")

class TestSyllabusParsing(unittest.TestCase):
    def test_parse_syllabus_sections_with_multiword_expressions(self):
        from librarian.processor import WikiProcessor
        sample_markdown = """# Text A
Good afternoon! Welcome to the university.
You will learn to get by on very little sleep and make the most of this unique experience.

## Syllabus Vocabulary
triumph
pledge
rewarding
remind sb. of sb. / sth.
get by (on / with)
make the most of sth.
in advance
all at once
early bird
prosperous
open the door to sth.

## Syllabus Grammar
- Inversion with negative adverbs
- Subjunctive mood in that-clauses
"""
        clean_body, vocab, grammar, expressions = WikiProcessor.parse_syllabus_sections(sample_markdown)
        
        self.assertNotIn("## Syllabus Vocabulary", clean_body)
        self.assertNotIn("## Syllabus Grammar", clean_body)
        self.assertIn("Good afternoon! Welcome to the university.", clean_body)
        
        self.assertIn("triumph", vocab)
        self.assertIn("pledge", vocab)
        self.assertNotIn("make the most of sth", vocab)
        
        self.assertEqual(len(grammar), 2)
        self.assertIn("Inversion with negative adverbs", grammar)
        
        self.assertIn("remind sb of sb / sth", expressions)
        self.assertIn("get by (on / with)", expressions)
        self.assertIn("make the most of sth", expressions)
        self.assertIn("in advance", expressions)
        self.assertIn("all at once", expressions)
        self.assertIn("early bird", expressions)
        self.assertIn("open the door to sth", expressions)
        
        self.assertNotIn("triumph", expressions)
        self.assertNotIn("pledge", expressions)
        self.assertNotIn("prosperous", expressions)

    def test_strict_single_blank_gate(self):
        from librarian.processor import WikiProcessor
        # Quiz with multiple blanks in stem
        quiz_multi_blank = {
            "questions": [
                {
                    "target_word": "remind",
                    "question": "He ____ me of ____ yesterday.",
                    "options": ["reminded", "warned", "convinced", "deprived"],
                    "correct_answer_index": 0
                },
                {
                    "target_word": "triumph",
                    "question": "The final victory was a personal ____ for the athlete.",
                    "options": ["triumph", "hazard", "obstacle", "setback"],
                    "correct_answer_index": 0
                }
            ]
        }
        flagged, msgs = WikiProcessor.audit_quiz_integrity(quiz_multi_blank)
        self.assertIn(0, flagged)
        self.assertNotIn(1, flagged)
        self.assertTrue(any("Multiple blanks" in msg for msg in msgs))


    def test_target_coverage_penalty_and_fatal_flag(self):
        user_prompt = (
            "### DETERMINISTIC TARGET PATTERNS (PRE-EXTRACTED BY COMPUTATIONAL LINGUISTICS) ###\n"
            "The following 5 academic structural patterns have been pre-identified:\n"
            "1. [S-8] (Logic & Stance) Formula: `while [Clause], [Main Clause]`\n"
            "2. [S-21] (Information Packaging) Formula: `[Subject] + [VP], [V-ing Phrase]`\n"
            "3. [S-23] (Cohesion & Framing) Formula: `, which suggests that [Clause]`\n"
            "4. [S-28] (Rhetoric & Emphasis) Formula: `not only [VP], but [VP]`\n"
            "5. [S-31] (Logic & Stance) Formula: `if [Clause], [Main Clause]`\n"
            "\n"
            "### PASSAGE ###\n"
            "[S-8] While the results were preliminary, they offered promise.\n"
            "[S-21] The team worked tirelessly, conducting several tests.\n"
            "[S-23] The rate increased, which suggests that demand grew.\n"
            "[S-28] They not only investigated the cause, but solved the issue.\n"
            "[S-31] If conditions deteriorate, action will be needed.\n"
        )
        # Model only returned 1 out of 5 patterns
        partial_json = {
            "grammar_patterns": [
                {
                    "design_audit": "AUDIT: [S-8] -> Logic & Stance -> while [Clause], [Main Clause]",
                    "pattern_formula": "while [Clause], [Main Clause]",
                    "quote": "While the results were preliminary, they offered promise.",
                    "category": "Logic & Stance",
                    "pedagogical_function": "Concessive contrast",
                    "imitation_example": "While this method is costly, it yields accurate outcomes.",
                    "common_mistakes": "Do not confuse while with although in temporal contexts.",
                    "cefr_level": "B2"
                }
            ]
        }
        log_data = {
            "task": "grammar",
            "model": "test-model",
            "user_prompt": user_prompt,
            "raw_response": json.dumps(partial_json),
            "parsed_json": partial_json,
        }
        res = LogEvaluator.evaluate_log(log_data)
        # 1/5 coverage must scale composite score to ~20%, not 100%!
        self.assertLessEqual(res["composite_score"], 25.0)
        self.assertTrue(any("Incomplete target coverage" in f for f in res["flags"]))
        self.assertTrue(any("1/5 targets" in f for f in res["flags"]))

    def test_ungrounded_formula_anchor_flag(self):
        user_prompt = (
            "### PASSAGE ###\n"
            "[S-31] If conditions deteriorate, action will be needed.\n"
        )
        # Model claims formula has [V-ing Phrase] but quote has no participle
        mismatched_json = {
            "grammar_patterns": [
                {
                    "design_audit": "AUDIT: [S-31] -> Information Packaging -> [Subject] + [VP], [V-ing Phrase]",
                    "pattern_formula": "[Subject] + [VP], [V-ing Phrase]",
                    "quote": "If conditions deteriorate, action will be needed.",
                    "category": "Information Packaging",
                    "pedagogical_function": "Participial adjunct",
                    "imitation_example": "The scientist reviewed the data, noting errors.",
                    "common_mistakes": "Avoid dangling participles.",
                    "cefr_level": "B2"
                }
            ]
        }
        log_data = {
            "task": "grammar",
            "model": "test-model",
            "user_prompt": user_prompt,
            "raw_response": json.dumps(mismatched_json),
            "parsed_json": mismatched_json,
        }
        res = LogEvaluator.evaluate_log(log_data)
        self.assertTrue(any("ungrounded in quote" in f for f in res["flags"]))


    def test_transparent_delivery_failed_blocking_frontmatter(self):
        from librarian.processor import WikiProcessor
        wp = WikiProcessor()
        # Simulated extraction data that failed QA with score 25 and fatal flag
        failed_data = {
            "title": "Test Grammar Failed",
            "overall_cefr_level": "B2",
            "grammar_patterns": [
                {
                    "name": "Defective Pattern",
                    "pattern_formula": "while [Clause], [Main Clause]",
                    "quote": "Unrelated sentence.",
                    "category": "Logic & Stance"
                }
            ],
            "_qa_audit": {
                "composite_score": 25.0,
                "flags": ["❌ [INCOMPLETE_COVERAGE] Incomplete target coverage: delivered only 1/5 targets"]
            }
        }
        md = wp._format_as_markdown(failed_data, "grammar", "TestUnit.md")
        self.assertIn("qa_status: \"failed\"", md)
        self.assertIn("qa_score: 25", md)
        self.assertIn("> [!CAUTION]", md)
        self.assertIn("Extraction Quality Gate Failed (Score: 25/100)", md)
        # Content item must NOT be rendered in body to prevent contamination
        self.assertNotIn("Defective Pattern", md)
        self.assertNotIn("Unrelated sentence", md)

    def test_transparent_delivery_passed_renders_body(self):
        from librarian.processor import WikiProcessor
        wp = WikiProcessor()
        passed_data = {
            "title": "Test Grammar Passed",
            "overall_cefr_level": "B2",
            "grammar_patterns": [
                {
                    "name": "Valid Pattern",
                    "pattern_formula": "while [Clause], [Main Clause]",
                    "quote": "While the results were preliminary, they offered promise.",
                    "category": "Logic & Stance",
                    "imitation_example": "Ex.",
                    "common_mistakes": "None."
                }
            ],
            "_qa_audit": {
                "composite_score": 92.0,
                "flags": []
            }
        }
        md = wp._format_as_markdown(passed_data, "grammar", "TestUnit.md")
        self.assertIn("qa_status: \"passed\"", md)
        self.assertIn("qa_score: 92", md)
        self.assertNotIn("> [!CAUTION]", md)
        self.assertIn("Valid Pattern", md)


if __name__ == "__main__":
    unittest.main()



