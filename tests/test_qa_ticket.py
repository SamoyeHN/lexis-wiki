"""Tests for the F10 repair-ticket builder (librarian/qa_ticket.py).

The contract these tests hold: a ticket names a field, never a sentence; a ticket never carries
source text; a flag that cannot be attributed to a field never becomes an instruction.
"""

import re
import unittest

from librarian.qa_ticket import (
    ARRAY_LEVEL,
    _FIELD_RULES,
    build_qa_ticket,
    build_qa_tickets,
    detect_array_key,
    format_qa_ticket_block,
    item_name_from_flag,
    locate_item,
    mandate_for_tickets,
    rule_for_flag,
    score_ticket,
    ticket_is_repairable,
    tickets_for_flags,
)

SOURCE_SENTENCE = "Wearing rubber boots keeps the mud out of the classroom."

QUIZ_PARSED = {
    "questions": [
        {
            "target_word": "sector",
            "question": "The manufacturing ____ grew fastest after the contract was signed.",
            "options": ["sector", "section", "industry", "portion"],
            "correct_answer_index": 0,
            "explanation": "'sector' names a branch of industry.",
        },
        {
            "target_word": "grant",
            "question": "The board agreed to ____ the funds for the follow-up study.",
            "options": ["grant", "donate", "award", "present"],
            "correct_answer_index": 1,
            "explanation": "'grant' is the verb the frame declares.",
        },
    ]
}

VOCAB_PARSED = {
    "vocabulary": [
        {
            "word": "boot",
            "definition": "a strong shoe that covers the foot and ankle",
            "quoted_sentence": "Wearing rubber boots keeps the mud out of the classroom.",
        },
        {
            "word": "mud",
            "definition": "a soft mixture of earth and water",
            "quoted_sentence": "The children carried their own buckets through the mud.",
        },
    ]
}

FIELD_PATH_RE = re.compile(r"^- \[FIELD\]: `[A-Za-z_]+(\[\d+\])?(\.[A-Za-z_]+)?`", re.MULTILINE)


class TestFlagTable(unittest.TestCase):
    def test_every_rule_names_a_field_without_prose(self):
        for needles, field_name, action in _FIELD_RULES:
            self.assertTrue(needles, needles)
            # A one-element needle group must keep its trailing comma; ("x") is the string "x",
            # and iterating a string yields single letters that match every flag ever written.
            self.assertIsInstance(needles, tuple)
            for needle in needles:
                self.assertGreaterEqual(len(needle), 4, needle)
            self.assertTrue(field_name, needles)
            self.assertTrue(action, needles)
            if field_name != ARRAY_LEVEL:
                self.assertRegex(field_name, r"^[a-z_]+$")

    def test_no_instruction_supplies_content(self):
        # An instruction may name a field and a declared value, but it must not hand the model
        # wording to copy — that is the failure mode Backlog B records.
        for needles, _field_name, action in _FIELD_RULES:
            self.assertNotIn("for example", action.lower(), needles)
            self.assertNotIn("e.g.", action.lower(), needles)

    def test_unmapped_flag_falls_back_to_a_field_not_to_prose(self):
        field_name, action = rule_for_flag("❌ Quiz item 'sector': Some brand new evaluator gate")
        self.assertEqual(field_name, "question")
        self.assertIn("declared", action)


class TestItemLookup(unittest.TestCase):
    def test_headword_names_the_item(self):
        self.assertEqual(locate_item("❌ Quiz item 'grant': whatever", QUIZ_PARSED["questions"]), 1)
        self.assertEqual(locate_item("❌ Quiz item 'sector': whatever", QUIZ_PARSED["questions"]), 0)

    def test_inflected_surface_form_resolves_to_the_lemma_item(self):
        self.assertEqual(
            locate_item("❌ Quiz item 'granted': Inflection discordance", QUIZ_PARSED["questions"]), 1
        )

    def test_quoted_content_locates_the_item_without_reaching_the_ticket(self):
        flag = (
            "❌ Target word 'mud' does not appear in quoted sentence: "
            "'The children carried their own buckets through the mud.'"
        )
        self.assertEqual(locate_item(flag, VOCAB_PARSED["vocabulary"]), 1)

    def test_unlocatable_item_returns_none(self):
        self.assertIsNone(locate_item("❌ Quiz item 'whichever': whatever", QUIZ_PARSED["questions"]))

    def test_item_name_extraction(self):
        self.assertEqual(item_name_from_flag("❌ Quiz item 'sector': Prescribed options altered"), "sector")
        self.assertIsNone(item_name_from_flag("❌ [INCOMPLETE_COVERAGE] Incomplete target coverage"))

    def test_array_key_detection(self):
        self.assertEqual(detect_array_key(QUIZ_PARSED), "questions")
        self.assertEqual(detect_array_key(VOCAB_PARSED), "vocabulary")
        self.assertEqual(detect_array_key(None), "items")


class TestTicketShape(unittest.TestCase):
    def test_ticket_names_a_field_path_not_an_item(self):
        ticket = build_qa_ticket(
            "❌ Quiz item 'sector': Prescribed options altered — blueprint prescribed "
            "[sector, section, branch, segment], item shipped [sector, section, industry, portion]",
            array_key="questions",
            item_index=0,
        )
        self.assertRegex(ticket, FIELD_PATH_RE)
        self.assertIn("`questions[0].options`", ticket)
        self.assertTrue(ticket_is_repairable(ticket))

    def test_ticket_never_carries_source_text(self):
        flag = (
            "❌ Target word 'boot' does not appear in quoted sentence: "
            "'Wearing rubber boots keeps the mud out of the classroom.'"
        )
        block = tickets_for_flags([flag], task_name="vocabulary_extraction", parsed=VOCAB_PARSED)
        self.assertIn("`vocabulary[0].quoted_sentence`", block)
        self.assertNotIn(SOURCE_SENTENCE, block)
        self.assertNotIn("Wearing rubber boots", block)
        self.assertNotIn("keeps the mud out", block)

    def test_every_ticket_in_a_block_is_repairable_and_field_bounded(self):
        flags = [
            "❌ Quiz item 'sector': Prescribed options altered — blueprint prescribed [a, b, c, d]",
            "❌ Quiz item 'grant': Inflection discordance: blueprint declares 'past tense (VBD)'",
            "❌ Quiz item 'sector': cross-target leakage: ['grant']",
        ]
        tickets = build_qa_tickets(flags, task_name="quiz_vocabulary", parsed=QUIZ_PARSED)
        self.assertEqual(len(tickets), 3)
        for ticket in tickets:
            self.assertTrue(ticket_is_repairable(ticket), ticket)
            self.assertRegex(ticket, FIELD_PATH_RE)

    def test_array_coverage_flag_tickets_the_array(self):
        flag = "❌ [INCOMPLETE_COVERAGE] Incomplete target coverage: delivered only 3/5 targets"
        tickets = build_qa_tickets([flag], task_name="vocabulary_extraction", parsed=VOCAB_PARSED)
        self.assertEqual(len(tickets), 1)
        self.assertIn("`vocabulary`", tickets[0])
        self.assertIn("missing", tickets[0].lower())
        self.assertTrue(ticket_is_repairable(tickets[0]))

    def test_low_score_ticket_rechecks_instead_of_rewriting(self):
        ticket = score_ticket(62.0, "pedagogical_quality", array_key="questions")
        self.assertIn("`questions`", ticket)
        self.assertIn("Do not rewrite items", ticket)
        self.assertTrue(ticket_is_repairable(ticket))


class TestTicketFiltering(unittest.TestCase):
    def test_warning_flags_never_become_tickets(self):
        flags = [
            "⚠️ Quiz item 'sector': Distractor slot illegality: ['section'] cannot occupy the noun slot",
            "⚠️ Quiz item 'grant': mixed-form option set",
        ]
        self.assertEqual(build_qa_tickets(flags, task_name="quiz_vocabulary", parsed=QUIZ_PARSED), [])

    def test_blueprint_owned_defects_are_never_handed_to_the_model(self):
        # Both are defects in the blueprint's own declarations; a regeneration would only invent
        # a different defect, so they stay notes for the human backlog.
        flags = [
            "❌ Quiz item 'sector': Distractor slot illegality: ['section'] cannot occupy the noun slot",
            "❌ Quiz item 'grant': blueprint self-conflict — declared anchor 'granting' is itself "
            "another batch target",
        ]
        self.assertEqual(build_qa_tickets(flags, task_name="quiz_vocabulary", parsed=QUIZ_PARSED), [])

    def test_unlocatable_flag_is_dropped_rather_than_made_vague(self):
        flags = ["❌ Quiz item 'whichever': cross-target leakage: ['sector']"]
        self.assertEqual(build_qa_tickets(flags, task_name="quiz_vocabulary", parsed=QUIZ_PARSED), [])

    def test_ticket_cap(self):
        flags = [
            f"❌ Quiz item 'sector': cross-target leakage: ['{w}']" for w in ("a", "b", "c", "d", "e", "f")
        ]
        tickets = build_qa_tickets(flags, task_name="quiz_vocabulary", parsed=QUIZ_PARSED, max_tickets=3)
        self.assertEqual(len(tickets), 3)

    def test_duplicate_flags_collapse(self):
        flag = "❌ Quiz item 'sector': cross-target leakage: ['grant']"
        tickets = build_qa_tickets([flag, flag], task_name="quiz_vocabulary", parsed=QUIZ_PARSED)
        self.assertEqual(len(tickets), 1)


    def test_quiz_lookup_points_at_the_blueprint_not_the_passage(self):
        tickets = build_qa_tickets(
            ["❌ Quiz item 'sector': Prescribed options altered — blueprint prescribed [a, b]"],
            task_name="quiz_vocabulary",
            parsed=QUIZ_PARSED,
            source_header="### PASSAGE (WITH NUMBERED SENTENCES) ###",
        )
        self.assertIn("blueprint declaration", tickets[0])
        self.assertNotIn("PASSAGE", tickets[0])
        self.assertTrue(ticket_is_repairable(tickets[0]))

    def test_extraction_lookup_points_at_the_source_header(self):
        block = tickets_for_flags(
            [
                "❌ Target word 'boot' does not appear in quoted sentence: 'Wearing rubber boots "
                "keeps the mud out of the classroom.'"
            ],
            task_name="vocabulary_extraction",
            parsed=VOCAB_PARSED,
            source_header="### PASSAGE (WITH NUMBERED SENTENCES) ###",
        )
        self.assertIn("### PASSAGE (WITH NUMBERED SENTENCES) ###", block)
        self.assertNotIn("Wearing rubber boots", block)


class TestMandate(unittest.TestCase):
    def test_mandate_names_only_the_ticketed_fields(self):
        flags = ["❌ Quiz item 'sector': Prescribed options altered — blueprint prescribed [a, b]"]
        tickets = build_qa_tickets(flags, task_name="quiz_vocabulary", parsed=QUIZ_PARSED)
        mandate = mandate_for_tickets(tickets)
        self.assertIn("`questions[0].options`", mandate)
        self.assertNotIn("`questions[1]", mandate)
        self.assertIn("re-check", mandate)

    def test_mandate_is_not_a_task_type_lecture(self):
        # F10 rule 4: a declaration that was not violated must not be mentioned at all.
        flags = ["❌ Quiz item 'grant': Inflection discordance: blueprint declares 'past tense (VBD)'"]
        mandate = mandate_for_tickets(build_qa_tickets(flags, task_name="quiz_vocabulary", parsed=QUIZ_PARSED))
        for phrase in ("ONE continuous 4-underscore blank", "never repeat options", "verbatim evidence"):
            self.assertNotIn(phrase, mandate)

    def test_block_orders_tickets_then_mandate(self):
        block = tickets_for_flags(
            ["❌ Quiz item 'sector': Prescribed options altered — blueprint prescribed [a, b]"],
            task_name="quiz_vocabulary",
            parsed=QUIZ_PARSED,
        )
        self.assertIn("### 🚨 QUALITY AUDIT DEFECT TICKET", block)
        self.assertLess(block.index("[FIELD]"), block.index("MANDATE"))

    def test_empty_flags_produce_no_block(self):
        self.assertEqual(format_qa_ticket_block([]), "")
class TestEvaluatorFlagCoverage(unittest.TestCase):
    """Every flag the evaluator actually emits must hit a table rule, not the fallback.

    The strings below are copied from librarian/evaluator.py. If a gate is renamed there and the
    table is not updated, this test fails instead of the repair prompt silently degrading into
    generic advice.
    """

    REAL_FLAGS = [
        "❌ Valid JSON but empty object",
        "❌ Invalid or missing JSON output",
        "❌ [INVALID_SCHEMA] Expected list for 'questions', got str",
        "❌ [INVALID_ITEM_TYPE] 2/5 elements in 'questions' are not objects",
        "❌ [INSUFFICIENT_ITEMS] 'questions' contains only 1 valid objects",
        "❌ Target word 'boot' not found in supplied word list (possible hallucination)",
        "❌ Missing quote/quoted_sentence for item: 'boot'",
        "❌ Hallucinated quote (explicitly inferred/absent): 'Wearing rubber boots k...'",
        "❌ Prompt instruction leakage in quote: 'Provide a sentence that...'",
        "❌ Target word 'boot' does not appear in quoted sentence: 'The children carried...'",
        "❌ Headword 'bootw' is not a recognized English word and does not appear in the source",
        "❌ Non-verbatim quote detected: 'Wearing rubber boots k...'",
        "❌ Grammar pattern 'be going to': imitation_example duplicates the source quote verbatim",
        "❌ Grammar pattern 'be going to': degenerate formula slot 1 of 3 slots",
        "❌ Grammar pattern 'be going to' deviates from the pre-extracted skeleton",
        "❌ Quiz item 'sector': Multiple blanks (2) detected in question stem",
        "❌ Quiz item 'sector': Target word leaks verbatim into question stem outside blank: {'sector'}",
        "❌ Quiz item 'sector': Question stem has indefinite article ('a/an') immediately preceding blank",
        "❌ Quiz item 'sector': Question stem leaks other batch target word(s) (cross-target leakage)!",
        "❌ Quiz item 'sector': Prescribed options altered — blueprint prescribed [a, b]",
        "❌ Quiz item 'sector': Prescribed answer index altered — blueprint index 2",
        "❌ Quiz item 'sector': Stem verbatim from dictionary example",
        "❌ Quiz item 'sector': Stem verbatim from curriculum quote",
        "❌ Quiz item 'grant': Inflection discordance: blueprint declares 'past tense (VBD)'",
        "❌ Quiz item 'grant': Inflection discordance with blueprint declaration",
        "❌ Quiz item 'grant' in-list distractor recycling: {'grant'}",
        "❌ Quiz item 'grant': missing fill-in-the-blank slot (____)",
        "❌ Quiz item 'grant': indefinite article leakage before blank",
        "❌ Found 2 duplicate or copy-pasted definition(s) across different terms",
        "❌ Found 1 duplicate item(s)",
        "❌ Morphological word-family collisions detected: boot/boots",
        "❌ Distractors recycled repeatedly across quiz: section x3",
        "❌ [INCOMPLETE_COVERAGE] Incomplete target coverage: delivered only 3/5 targets",
    ]

    def test_real_flags_resolve_to_a_table_rule(self):
        fallback = rule_for_flag("❌ Something the evaluator has never emitted")
        for flag in self.REAL_FLAGS:
            self.assertNotEqual(rule_for_flag(flag), fallback, flag)

    def test_duplicate_definition_flag_does_not_steal_the_grammar_quote_flag(self):
        # 'duplicate' used to be a bare key, so 'imitation_example duplicates the source quote'
        # matched the definition rule and got a definition-field ticket.
        grammar = rule_for_flag(
            "❌ Grammar pattern 'x': imitation_example duplicates the source quote verbatim"
        )
        self.assertEqual(grammar[0], "imitation_example")

    def test_hallucinated_quote_tickets_the_quote_not_the_headword(self):
        self.assertEqual(
            rule_for_flag("❌ Hallucinated quote (explicitly inferred/absent): 'some quote here'")[0],
            "quoted_sentence",
        )


        self.assertEqual(tickets_for_flags([], task_name="quiz_vocabulary", parsed=QUIZ_PARSED), "")


if __name__ == "__main__":
    unittest.main()

