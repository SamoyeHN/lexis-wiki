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
    _form_in_text,
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

    def test_multiword_headword_is_retyped_to_its_expression_type(self):
        """A vocabulary row labelled with the part of speech of one of its tokens ('tap
        into' -> 'verb') is re-typed to its Longman expression type before it is scored,
        so the label that reaches the page is 'phrasal verb'."""
        items = [{
            "word": "tap into [something]",
            "part_of_speech": "verb",
            "definition": "to make use of a supply of something such as energy or knowledge",
            "quoted_sentence": "The company tapped into a network of former students.",
            "example_usage": "Her research tapped into archives nobody had read before.",
        }]
        score, flags = _score_pedagogy(items, "vocabulary")
        self.assertEqual(items[0]["part_of_speech"], "phrasal verb")
        self.assertEqual(score, W_PEDAGOGY)
        self.assertEqual(flags, [])

    def test_frame_headword_is_retyped_to_a_set_phrase(self):
        """'as a whole' is a fixed frame, not a conjunction to be parsed."""
        items = [{
            "word": "as a whole",
            "part_of_speech": "conjunction",
            "definition": "considered all together",
            "quoted_sentence": "The industry as a whole reported a small gain.",
            "example_usage": "Viewed as a whole, the plan was sound.",
        }]
        score, flags = _score_pedagogy(items, "vocabulary")
        self.assertEqual(items[0]["part_of_speech"], "set phrase")
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


    def test_verbatim_reading_vocab_context_sentence(self):
        source = "CONTENT:\nThe company aims to accelerate its digital transformation across all branches."
        items = [{
            "word": "accelerate",
            "context_sentence": "The company aims to accelerate its digital transformation across all branches."
        }]
        score, flags = _score_verbatim(items, "vocabulary", source)
        self.assertEqual(score, W_VERBATIM)
        self.assertEqual(flags, [])

    def test_comprehension_quiz_pedagogy_passes_without_blanks(self):
        items = [{
            "question": "What is the primary objective of the company's new digital strategy?",
            "options": [
                "To accelerate overall transformation",
                "To reduce operating branches",
                "To delay technological shifts",
                "To replace human resources"
            ],
            "correct_answer_index": 0,
            "explanation": "The passage directly confirms the aim is to accelerate transformation."
        }]
        score, flags = _score_pedagogy(items, "quiz")
        self.assertEqual(score, W_PEDAGOGY)
        self.assertEqual(flags, [])


    def test_mindmap_invalid_string_items_penalized(self):
        # Double-encoded string item inside branches array
        bad_mindmap = {
            "title": "Test Map",
            "root_name": "Root",
            "branches": [
                {"branch_name": "Branch 1", "sub_branches": []},
                '{"branch_name": "Branch 2", "sub_branches": []}',
                {"branch_name": "Branch 3", "sub_branches": []}
            ]
        }
        score, flags = _score_schema(bad_mindmap, task_type="mindmap")
        self.assertLess(score, W_SCHEMA)
        self.assertTrue(any("INVALID_ITEM_TYPE" in f for f in flags))

    def test_mindmap_insufficient_items_penalized(self):
        # Only 2 branches when schema requires at least 3
        short_mindmap = {
            "title": "Test Map",
            "root_name": "Root",
            "branches": [
                {"branch_name": "Branch 1", "sub_branches": []},
                {"branch_name": "Branch 2", "sub_branches": []}
            ]
        }
        score, flags = _score_schema(short_mindmap, task_type="mindmap")
        self.assertLess(score, W_SCHEMA)
        self.assertTrue(any("INSUFFICIENT_ITEMS" in f for f in flags))

    def test_example_usage_orthography_repair(self):
        import re
        data = {
            "title": "Vocab",
            "vocabulary": [
                {
                    "word": "well-kept",
                    "example_usage": "The laboratory's **wellkept** records proved essential.",
                    "quoted_sentence": "Authentic quote."
                },
                {
                    "word": "in-person",
                    "example_usage": "The workshop will be held **inperson** next week.",
                    "quoted_sentence": "Authentic quote."
                }
            ]
        }
        # Simulate LLMClient Level 1 Code Gate for orthographic hygiene
        for item in data["vocabulary"]:
            w = str(item.get("word", "")).strip()
            clean_target_w = re.sub(r'\[.*?\]|\(.*?\)', '', w).strip()
            if "-" in clean_target_w and item.get("example_usage"):
                unhyphen = clean_target_w.replace("-", "")
                if len(unhyphen) >= 4:
                    pat = r'\b' + re.escape(unhyphen) + r'\b'
                    item["example_usage"] = re.sub(pat, clean_target_w, item["example_usage"], flags=re.IGNORECASE)

        vocab = data["vocabulary"]
        self.assertIn("well-kept", vocab[0]["example_usage"])
        self.assertNotIn("wellkept", vocab[0]["example_usage"])
        self.assertIn("in-person", vocab[1]["example_usage"])
        self.assertNotIn("inperson", vocab[1]["example_usage"])


class TestQuizGateRegressions(unittest.TestCase):
    """Regression guards for the vocabulary quiz gates in _score_pedagogy."""

    def test_translation_and_comprehension_items_do_not_crash(self):
        # These item shapes never enter the lexical fill-in-the-blank branch, so every
        # blank-specific gate must still be bound or the shared pass computation crashes.
        mixed_quiz = [
            {
                "target_keyword": "in advance",
                "source_sentence": "Please book the hotel in advance.",
                "translated_sentence": "请提前预订酒店。",
                "english_skeleton": "Please book the hotel ____ .",
                "options": ["in advance", "all at once"],
                "correct_answer_index": 0,
                "explanation": "'in advance' means beforehand.",
            },
            {
                "question": "What is the main idea of the passage?",
                "options": ["A", "B", "C", "D"],
                "correct_answer_index": 1,
                "explanation": "The passage contrasts two time cultures.",
            },
        ]
        score, flags = _score_pedagogy(mixed_quiz, "quiz")
        self.assertIsNotNone(score)
        self.assertGreater(score, 0.0)

    def test_anchor_gate_waived_when_anchor_is_another_batch_target(self):
        # Blueprint self-conflict: the anchor 'punctuality' is itself a batch target, so no stem
        # could ever satisfy the anchor requirement and the cross-target leakage ban at once.
        # Level 1 reports the contradiction in the blueprint instead of asking the model.
        user_prompt = (
            "## [[pay]]\n## [[punctuality]]\n"
            "### Item 1 ###\n"
            "- Target Word: pay\n"
            "- Part of Speech: verb\n"
            "- Inflectional Form: base form\n"
            "- Collocational Anchor: punctuality (verb_subject)\n"
        )
        quiz = [{
            "target_word": "pay",
            "question": "Even a small delay can ____ respect for your colleagues during a meeting.",
            "options": ["pay", "spend", "waste", "lose"],
            "correct_answer_index": 0,
            "explanation": "'pay' is the only verb that collocates with 'respect'.",
        }]
        score, flags = _score_pedagogy(quiz, "quiz", user_prompt=user_prompt)
        self.assertFalse(any("Anchor missing in question stem" in f for f in flags), flags)
        self.assertTrue(any("blueprint self-conflict" in f for f in flags), flags)
        self.assertFalse(any(f.startswith("❌") for f in flags), flags)
        self.assertGreater(score, 0.0)

    def test_anchor_presence_is_no_longer_a_level_one_gate(self):
        # F10 rule 1.1. 'sector' declares the anchor 'manufacture', the model writes
        # 'the manufacturing ____', a literal token search finds no 'manufacture', and that one
        # false error ate two of five repair slots and drove a 44-second regeneration round.
        # Level 1 no longer searches for a word inside a sentence; the blueprint checks its own
        # anchor against its own model sentence (LinguisticEngine blueprint_warnings) instead.
        user_prompt = (
            "## [[log]]\n## [[agenda]]\n"
            "### Item 1 ###\n"
            "- Target Word: log\n"
            "- Part of Speech: verb\n"
            "- Inflectional Form: base form\n"
            "- Collocational Anchor: into (prep)\n"
        )
        quiz = [{
            "target_word": "log",
            "question": "She forgot to ____ the meeting details before the business trip started.",
            "options": ["log", "record", "note", "enter"],
            "correct_answer_index": 0,
            "explanation": "'log' collocates with 'into' the system.",
        }]
        score, flags = _score_pedagogy(quiz, "quiz", user_prompt=user_prompt)
        self.assertFalse(any("Anchor missing in question stem" in f for f in flags), flags)
        self.assertFalse(any("Explanation anchor not grounded" in f for f in flags), flags)
        self.assertFalse(any("blank slot POS mismatch" in f for f in flags), flags)
        self.assertGreater(score, 0.0)

    def test_cross_target_leakage_gate_still_fatal(self):
        user_prompt = "## [[log]]\n## [[agenda]]\n"
        quiz = [{
            "target_word": "log",
            "question": "She forgot to ____ the agenda before the business trip started.",
            "options": ["log", "record", "note", "enter"],
            "correct_answer_index": 0,
            "explanation": "'log' fits the context.",
        }]
        score, flags = _score_pedagogy(quiz, "quiz", user_prompt=user_prompt)
        self.assertTrue(any("cross-target leakage" in f for f in flags))
        self.assertEqual(score, 0.0)


class TestBlueprintConcordanceGates(unittest.TestCase):
    """Evaluator-side gates: declared inflection and distractor slot legality (P1-1, P1-2)."""

    def _prompt(self, headword, pos, inflection, anchor="general context"):
        return (
            f"## [[{headword}]]\n"
            "### Item 1 ###\n"
            f"- Target Word: {headword}\n"
            f"- Part of Speech: {pos}\n"
            f"- Inflectional Form: {inflection}\n"
            f"- Collocational Anchor: {anchor}\n"
        )

    def test_base_form_option_against_past_tense_blueprint_is_flagged(self):
        quiz = [{
            "target_word": "annoy",
            "question": "What ____ his colleagues most was that the report arrived a day late.",
            "options": ["annoy", "bother", "disturb", "irritate"],
            "correct_answer_index": 0,
            "explanation": "'annoy' fits the context of a repeated irritation.",
        }]
        score, flags = _score_pedagogy(
            quiz, "quiz", user_prompt=self._prompt("annoy", "verb", "past tense (VBD)")
        )
        self.assertTrue(any("Inflection discordance" in f for f in flags))

    def test_stem_that_contradicts_the_declared_form_is_no_longer_guessed(self):
        # F10 rule 1.4: the auxiliary table that read '... will ____' and inferred which form
        # the slot demanded is deleted — a detector built that way covers 'is currently ____'
        # and misses 'has been ____'. What Level 1 keeps is the pure comparison of the declared
        # tag against the form the answer option actually carries, and here 'annoyed' matches
        # 'past tense (VBD)', so the item is clean.
        quiz = [{
            "target_word": "annoyed",
            "question": "What will ____ his colleagues most when the report arrives a day late?",
            "options": ["annoyed", "bothered", "disturbed", "irritated"],
            "correct_answer_index": 0,
            "explanation": "'annoyed' fits the context of a repeated irritation.",
        }]
        score, flags = _score_pedagogy(
            quiz, "quiz", user_prompt=self._prompt("annoyed", "verb", "past tense (VBD)")
        )
        self.assertFalse(any("requires VB/VBP" in f for f in flags), flags)
        self.assertFalse(any("Inflection discordance" in f for f in flags), flags)
        self.assertEqual(score, W_PEDAGOGY)

    def test_concordant_verb_item_raises_no_inflection_flag(self):
        quiz = [{
            "target_word": "annoyed",
            "question": "What annoyed his colleagues most was that the report arrived a day late.",
            "options": ["annoyed", "bothered", "disturbed", "irritated"],
            "correct_answer_index": 0,
            "explanation": "'annoyed' fits the context of a repeated irritation.",
        }]
        # The stem leaks the target, so drop the leak gate by re-writing the stem blank.
        quiz[0]["question"] = "What ____ his colleagues most was that the report arrived a day late."
        score, flags = _score_pedagogy(
            quiz, "quiz", user_prompt=self._prompt("annoyed", "verb", "past tense (VBD)")
        )
        self.assertFalse(any("Inflection discordance" in f for f in flags))

    def test_adverb_distractor_in_a_noun_slot_is_reported_not_fatal(self):
        # F10 改哪几处 2: the verdict is a pure dictionary comparison, so it stays reported — but
        # the options are blueprint-owned and the writer is forbidden from changing them, so a
        # ❌ here would trigger a regeneration that may not fix the thing it was called for.
        quiz = [{
            "target_word": "agenda",
            "question": "The committee circulated the written ____ before the quarterly meeting began.",
            "options": ["agenda", "somehow", "therefore", "immediately"],
            "correct_answer_index": 0,
            "explanation": "'agenda' is the only noun that names a document.",
        }]
        score, flags = _score_pedagogy(
            quiz, "quiz", user_prompt=self._prompt("agenda", "noun", "base form")
        )
        warnings = [f for f in flags if "Distractor slot illegality" in f]
        self.assertTrue(warnings, flags)
        self.assertTrue(all(f.startswith("⚠️") for f in warnings), warnings)
        self.assertGreater(score, 0.0)

    def test_legal_noun_distractors_are_not_flagged(self):
        quiz = [{
            "target_word": "agenda",
            "question": "The committee circulated the written ____ before the quarterly meeting began.",
            "options": ["agenda", "minutes", "timetable", "proposal"],
            "correct_answer_index": 0,
            "explanation": "'agenda' is the document circulated before a meeting.",
        }]
        score, flags = _score_pedagogy(
            quiz, "quiz", user_prompt=self._prompt("agenda", "noun", "base form")
        )
        self.assertFalse(any("Distractor slot illegality" in f for f in flags))
        self.assertGreater(score, 0.0)

    def test_comprehension_items_without_blanks_stay_evaluable(self):
        quiz = [
            {
                "question": "Why did the committee postpone the vote?",
                "options": ["delay", "procedure", "quorum", "pressure"],
                "correct_answer_index": 0,
                "explanation": "The passage attributes the postponement to a delay.",
            }
        ]
        score, flags = _score_pedagogy(quiz, "quiz")
        self.assertIsNotNone(score)
        self.assertFalse(any("Inflection discordance" in f for f in flags))

    def test_the_new_quiz_gates_are_registered_as_fatal(self):
        """A ❌ quiz gate must actually gate the pipeline, not just print a warning."""
        from librarian.evaluator import FATAL_QA_FLAGS
        self.assertIn("Inflection discordance", FATAL_QA_FLAGS)
        self.assertIn("Stem verbatim from dictionary example", FATAL_QA_FLAGS)
        self.assertIn("Stem verbatim from curriculum quote", FATAL_QA_FLAGS)
        self.assertIn("cross-target leakage", FATAL_QA_FLAGS)

    def test_the_four_sentence_gates_are_not_fatal(self):
        """F10 rule 0: nothing that needs a sentence read may be a Level-1 fatal flag."""
        from librarian.evaluator import FATAL_QA_FLAGS
        for retired in (
            "blank slot POS mismatch",
            "placed the blank in a NOUN slot",
            "placed the blank in a finite VERB slot",
            "Anchor missing in question stem",
            "Explanation anchor not grounded",
            "Distractor slot illegality",
        ):
            self.assertNotIn(retired, FATAL_QA_FLAGS)

    def test_vocabulary_quiz_prompt_restores_the_copying_guard(self):
        prompt_path = Path(__file__).resolve().parent.parent / "librarian" / "prompts" / "vocabulary_quiz.md"
        text = prompt_path.read_text(encoding="utf-8")
        self.assertIn("NO COPYING INPUT EXAMPLES", text)
        self.assertIn("NO DISTRACTOR SELF-DEFINITION", text)
        self.assertIn("Declared Inflection Is Binding", text)

class TestQuoteProvenanceGate(unittest.TestCase):
    """
    Quote provenance on the evaluator side: a licensed quote may be emulated but never copied,
    and a display-only quote is too short to license anything at all.
    """

    def _prompt(self, headword, pos, extra_lines):
        return (
            f"## [[{headword}]]\n"
            "### Item 1 ###\n"
            f"- Target Word: {headword}\n"
            f"- Part of Speech: {pos}\n"
            "- Inflectional Form: base form\n"
            "- Collocational Anchor: general context\n"
            f"{extra_lines}"
        )

    def test_stem_that_copies_the_licensed_quote_is_flagged(self):
        prompt = self._prompt(
            "punctuality", "noun",
            "- Authentic Corpus Blueprint: 'The panel reviewed the written proposal before voting.'\n"
            "- Curriculum Quote (licensed): 'The board praised her punctuality at the quarterly review meeting.'\n",
        )
        quiz = [{
            "target_word": "punctuality",
            "question": "The board praised her ____ at the quarterly review meeting.",
            "options": ["punctuality", "accuracy", "diligence", "reliability"],
            "correct_answer_index": 0,
            "explanation": "'punctuality' names the quality the board praised.",
        }]
        score, flags = _score_pedagogy(quiz, "quiz", user_prompt=prompt)
        self.assertTrue(any("Stem verbatim from curriculum quote" in f for f in flags))
        self.assertEqual(score, 0.0)

    def test_stem_that_copies_a_display_only_quote_is_flagged(self):
        prompt = self._prompt(
            "diligence", "noun",
            "- Curriculum Quote (display-only, 2 words — too short to license the anchor or the "
            "sense, do not model the stem on it): 'Punctuality Pays'\n",
        )
        quiz = [{
            "target_word": "diligence",
            "question": "The firm's noticeboard still reads 'Punctuality Pays', which is the idea "
                         "the annual ____ award honours.",
            "options": ["diligence", "attendance", "reliability", "commitment"],
            "correct_answer_index": 0,
            "explanation": "'diligence' names the steady care the award recognises.",
        }]
        score, flags = _score_pedagogy(quiz, "quiz", user_prompt=prompt)
        self.assertTrue(any("Stem verbatim from curriculum quote" in f for f in flags))
        self.assertEqual(score, 0.0)

    def test_stem_that_copies_the_dictionary_example_is_flagged_even_beside_a_quote(self):
        # Regression: the blueprint example line and the licensed quote line coexist, and the
        # example must still be parsed as the dictionary sentence, not as the quote label.
        prompt = self._prompt(
            "serious", "adj",
            "- Collocational Anchor: about (prep)\n"
            "- Authentic Corpus Blueprint: 'Is she serious about giving up her job?'\n"
            "- Curriculum Quote (licensed): 'If you come from a culture that has a more relaxed "
            "view of time, you are likely to arrive late.'\n",
        )
        quiz = [{
            "target_word": "serious",
            "question": "Is she ____ about giving up her job to start her own small business "
                        "next month?",
            "options": ["earnest", "real", "thoughtful", "serious"],
            "correct_answer_index": 3,
            "explanation": "'serious about' is the bound-preposition frame.",
        }]
        score, flags = _score_pedagogy(quiz, "quiz", user_prompt=prompt)
        self.assertTrue(any("Stem verbatim from dictionary example" in f for f in flags))
        self.assertEqual(score, 0.0)

    def test_a_stem_that_only_alludes_to_the_weak_quote_is_not_flagged(self):
        prompt = self._prompt(
            "diligence", "noun",
            "- Curriculum Quote (display-only, 2 words — too short to license the anchor or the "
            "sense, do not model the stem on it): 'Punctuality Pays'\n",
        )
        quiz = [{
            "target_word": "diligence",
            "question": "The noticeboard notice about punctuality was pinned beside the payroll "
                         "list, and the annual ____ award honoured her for it.",
            "options": ["diligence", "attendance", "reliability", "commitment"],
            "correct_answer_index": 0,
            "explanation": "'diligence' names the steady care the award recognises.",
        }]
        score, flags = _score_pedagogy(quiz, "quiz", user_prompt=prompt)
        self.assertFalse(any("Stem verbatim" in f for f in flags))
        self.assertGreater(score, 0.0)
    def test_a_short_dictionary_example_masked_into_the_stem_is_flagged(self):
        # 'City employees cannot contribute to political campaigns.' is only 7 words, so every
        # 5-gram of it contains 'contribute'. Blanking the target used to erase the evidence.
        prompt = self._prompt(
            "contribute", "verb",
            "- Authentic Corpus Blueprint: 'City employees cannot contribute to political campaigns.'\n",
        )
        quiz = [{
            "target_word": "contribute",
            "question": "City employees cannot ____ to political campaigns during working hours.",
            "options": ["contribute", "donate", "belong", "react"],
            "correct_answer_index": 0,
            "explanation": "'contribute to' is the bound-preposition frame.",
        }]
        score, flags = _score_pedagogy(quiz, "quiz", user_prompt=prompt)
        self.assertTrue(any("Stem verbatim from dictionary example" in f for f in flags))
        self.assertEqual(score, 0.0)

    def test_a_short_licensed_quote_masked_into_the_stem_is_flagged(self):
        prompt = self._prompt(
            "invite", "verb",
            "- Authentic Corpus Blueprint: 'We invited some neighbours over for dinner.'\n"
            "- Curriculum Quote (licensed): 'Who should we invite to the party?'\n",
        )
        quiz = [{
            "target_word": "invite",
            "question": "Who should we ____ to the party now that the whole office has been asked?",
            "options": ["invite", "escort", "summon", "introduce"],
            "correct_answer_index": 0,
            "explanation": "'invite someone to an event' is the frame.",
        }]
        score, flags = _score_pedagogy(quiz, "quiz", user_prompt=prompt)
        self.assertTrue(any("Stem verbatim from curriculum quote" in f for f in flags))

    def test_an_original_stem_built_on_a_short_example_is_not_flagged(self):
        # Same blueprint, but the stem borrows only the syntactic frame and reuses no wording.
        prompt = self._prompt(
            "contribute", "verb",
            "- Authentic Corpus Blueprint: 'City employees cannot contribute to political campaigns.'\n",
        )
        quiz = [{
            "target_word": "contribute",
            "question": "Although the new recruits were eager to help, none of them were allowed to "
                        "____ money to the mayoral race while still on the payroll.",
            "options": ["contribute", "donate", "lend", "commit"],
            "correct_answer_index": 0,
            "explanation": "'contribute money to' is the bound-preposition frame.",
        }]
        score, flags = _score_pedagogy(quiz, "quiz", user_prompt=prompt)
        self.assertFalse(any("Stem verbatim from" in f for f in flags))
        self.assertGreater(score, 0.0)




class TestSharedFormMatcher(unittest.TestCase):
    """Backlog A1: the evaluator kept three private copies of the same substring
    heuristic for 'does the headword appear in this quote', so an inflected occurrence
    ('attaches' for the headword 'attach') was not counted as evidence. All three now
    delegate to one matcher built on LinguisticEngine's legal-form generator."""

    def test_inflected_occurrence_counts_as_evidence(self):
        self.assertTrue(_form_in_text("attach", "the firm attaches itself to a purpose"))
        self.assertTrue(_form_in_text("take", "he took the lead in the negotiation"))
        self.assertTrue(_form_in_text("child", "the children were waiting outside"))

    def test_derivational_stem_heuristic_is_preserved(self):
        # No inflection table covers 'degrade' -> 'degradation'; the historical
        # stem-prefix heuristic must keep accepting it.
        self.assertTrue(_form_in_text("degrade", "years of slow degradation"))

    def test_absent_word_is_still_absent(self):
        self.assertFalse(_form_in_text("attach", "the firm keeps its distance"))
        self.assertFalse(_form_in_text("", "an empty token"))
        self.assertFalse(_form_in_text("attach", ""))


class TestExampleUsageCEFRCeiling(unittest.TestCase):
    """Backlog C1: the passage the extraction ran on sets one ceiling, and the
    'example_usage' an item ships has to respect it. Grammar had the same rule
    (test_imitation_example_ceiling); vocabulary and expressions had none."""

    SOURCE = "### SOURCE TEXT ###\n" + (
        "The company aims to accelerate its digital transformation across all branches. "
        "Managers said the change would take two years and that staff would be trained "
        "before the new system went live in every office in the country. The plan also "
        "includes a support desk, and the board will review progress every quarter."
    )

    def _item(self, example: str) -> dict:
        return {
            "word": "accelerate",
            "part_of_speech": "verb",
            "definition": "to make something happen sooner or more quickly",
            "quoted_sentence": "The company aims to accelerate its digital transformation across all branches.",
            "example_usage": example,
        }

    def test_example_within_the_ceiling_passes(self):
        score, flags = _score_pedagogy(
            [self._item("We need to accelerate the hiring of new staff this year.")],
            "vocabulary", self.SOURCE,
        )
        self.assertEqual(score, W_PEDAGOGY)
        self.assertFalse(any("ceiling" in f for f in flags))

    def test_example_above_the_ceiling_is_flagged(self):
        # The passage rates A2, so the shared ceiling for a sentence the student
        # reads is B2 - a C2 pair in the example is out of reach.
        score, flags = _score_pedagogy(
            [self._item("The board meticulously perused the report before the meeting.")],
            "vocabulary", self.SOURCE,
        )
        self.assertLess(score, W_PEDAGOGY)
        self.assertTrue(
            any("example_usage exceeds the CEFR B2 ceiling of the CEFR A2 source passage" in f
                and "perused" in f for f in flags),
            flags,
        )

    def test_the_headword_itself_is_exempt(self):
        # 'accelerate' is C1 and the passage is A2, but an example is required to
        # contain the headword it teaches, however hard that headword is.
        score, flags = _score_pedagogy(
            [self._item("The government wants to accelerate its spending on rural roads.")],
            "vocabulary", self.SOURCE,
        )
        self.assertEqual(score, W_PEDAGOGY)
        self.assertFalse(any("ceiling" in f for f in flags), flags)

    def test_prompts_without_a_passage_are_not_gated(self):
        # Instruction text is not source difficulty: no passage, no ceiling.
        score, flags = _score_pedagogy(
            [self._item("The board meticulously perused the report before the meeting.")],
            "vocabulary",
        )
        self.assertEqual(score, W_PEDAGOGY)
        self.assertFalse(any("ceiling" in f for f in flags))


class TestPrescribedOptionsAlignmentGate(unittest.TestCase):
    """Backlog D1: the blueprint owns the options and the answer index.

    `prescribed_options` and `correct_answer_index` are computed deterministically
    (zero-collision distractors, CRC32 option positions). The writer is allowed to
    author the stem, the explanation and the audit — nothing else. A shuffled or
    substituted option set silently invalidates every distractor guarantee the
    lexicon just made, so it is fatal, not a warning.
    """

    def _prompt(self, options="prune, trim, cut, carve", index=0):
        return (
            "## [[prune]]\n"
            "### Item 1 ###\n"
            "- Target Word: prune\n"
            "- Part of Speech: verb\n"
            "- Inflectional Form: base form\n"
            "- Collocational Anchor: back (prep)\n"
            f"- Prescribed Options: [{options}]\n"
            f"- Correct Answer Index: {index}\n"
            "- Contextual Definition: to cut away the unwanted parts of a plant.\n"
        )

    def _quiz(self, options, index=0):
        return [{
            "target_word": "prune",
            "question": "The gardener had to ____ back the overgrown roses before spring.",
            "options": options,
            "correct_answer_index": index,
            "explanation": "'prune back' is the bound frame; the alternatives do not take 'back'.",
        }]

    def test_verbatim_options_pass(self):
        score, flags = _score_pedagogy(
            self._quiz(["prune", "trim", "cut", "carve"]), user_prompt=self._prompt(), task_type="quiz"
        )
        self.assertEqual(score, 25.0)
        self.assertFalse(any("Prescribed options altered" in f for f in flags), flags)
        self.assertFalse(any("Prescribed answer index altered" in f for f in flags), flags)

    def test_substituted_option_is_flagged(self):
        score, flags = _score_pedagogy(
            self._quiz(["prune", "trim", "carve", "slice"]), user_prompt=self._prompt(), task_type="quiz"
        )
        self.assertEqual(score, 0.0)
        self.assertTrue(any("Prescribed options altered" in f for f in flags), flags)
        self.assertTrue(any("introduced: ['slice']" in f for f in flags), flags)

    def test_shuffled_options_are_flagged(self):
        """Same four words, different order: the answer index no longer points at the target."""
        score, flags = _score_pedagogy(
            self._quiz(["prune", "trim", "carve", "cut"]), user_prompt=self._prompt(), task_type="quiz"
        )
        self.assertEqual(score, 0.0)
        self.assertTrue(any("Prescribed options altered" in f for f in flags), flags)

    def test_moved_answer_index_is_flagged(self):
        score, flags = _score_pedagogy(
            self._quiz(["prune", "trim", "cut", "carve"], index=1),
            user_prompt=self._prompt(), task_type="quiz",
        )
        self.assertEqual(score, 0.0)
        self.assertTrue(any("Prescribed answer index altered" in f for f in flags), flags)

    def test_altered_options_cannot_ship_as_passing(self):
        log_data = {
            "task": "quiz_vocabulary_Book_1_Unit_1_Passage_A",
            "model": "test-model",
            "user_prompt": self._prompt(),
            "raw_response": json.dumps({"questions": self._quiz(["prune", "trim", "carve", "slice"])}),
            "parsed_json": {"questions": self._quiz(["prune", "trim", "carve", "slice"])},
        }
        res = LogEvaluator.evaluate_log(log_data)
        self.assertLess(res["composite_score"], 60.0)
        self.assertEqual(res["status"], "REVIEW_NEEDED")
        self.assertTrue(any("Prescribed options altered" in f for f in res["flags"]), res["flags"])

    def test_alignment_gates_are_registered_as_fatal(self):
        from librarian.evaluator import FATAL_QA_FLAGS
        self.assertIn("Prescribed options altered", FATAL_QA_FLAGS)
        self.assertIn("Prescribed answer index altered", FATAL_QA_FLAGS)


if __name__ == "__main__":
    unittest.main()



