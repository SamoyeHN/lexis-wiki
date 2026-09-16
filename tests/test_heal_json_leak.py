# -*- coding: utf-8 -*-
"""Regression tests: _heal_json must repair out-of-band `(Note: ...)` comments
leaked by small models outside of JSON strings (see logs/20260916_130221_029_*.log).
"""
import json
import unittest

from librarian.llm import LLMClient


class TestOutOfBandCommentHealing(unittest.TestCase):
    def heal(self, s: str) -> dict:
        healed = LLMClient._heal_json(None, s)
        return json.loads(healed)  # raises if healing failed

    def test_log_failure_shape_quote_inside_comment(self):
        s = (
            '{"title":"t","overall_cefr_level":"B2","vocabulary":[{"design_audit":"parallel -> parallel (adjective) -> adjective",'
            '"word":"parallel","part_of_speech":"adjective","definition":"d",'
            '"quoted_sentence":"Paralleling their passion, talents, and work style...", (Note: \'Paralleling\' is gerund form of \'parallel\'. Actually text says "Paralleling their passion..." which is a verb form. We\'ll put \'adjective\'.),'
            '"example_usage":"The two roads run parallel."}]}'
        )
        data = self.heal(s)
        item = data["vocabulary"][0]
        self.assertEqual(item["word"], "parallel")
        self.assertEqual(item["example_usage"], "The two roads run parallel.")

    def test_double_comma_collapse(self):
        data = self.heal('{"a":"v", (Note: x, y), "b":2}')
        self.assertEqual(data, {"a": "v", "b": 2})

    def test_nested_parens_in_comment(self):
        data = self.heal('{"a":"v", (Note: (gerund) vs (adjective) call), "b":2}')
        self.assertEqual(data, {"a": "v", "b": 2})

    def test_missing_comma_after_comment(self):
        data = self.heal('{"a":"v", (Note: stray)\n"b":2}')
        self.assertEqual(data, {"a": "v", "b": 2})

    def test_valid_json_with_parens_in_strings_untouched(self):
        s = '{"def":"large enough (e.g. big)","note":"a (b) c"}'
        self.assertEqual(self.heal(s), json.loads(s))


class TestInnerQuoteHealing(unittest.TestCase):
    """Unescaped double quotes INSIDE a string value — the model crams several
    quoted source phrases into one field (see logs/20260916_175748_341_*.log).
    """
    def heal(self, s: str):
        return json.loads(LLMClient._heal_json(None, s))

    def test_two_phrases_joined_by_and(self):
        s = ('{"vocabulary":[{"quoted_sentence":"their sailing ships burned" and '
             '"watched their vessels go up in flames","example_usage":"x"}]}')
        data = self.heal(s)
        self.assertEqual(
            data["vocabulary"][0]["quoted_sentence"],
            'their sailing ships burned" and "watched their vessels go up in flames')
        self.assertEqual(data["vocabulary"][0]["example_usage"], "x")

    def test_comma_and_join(self):
        s = ('{"vocabulary":[{"quoted_sentence":"a fascinating investigation...", and '
             '"an experiment that investigated decision-making","example_usage":"x"}]}')
        data = self.heal(s)
        item = data["vocabulary"][0]
        self.assertIn("decision-making", item["quoted_sentence"])
        self.assertEqual(item["example_usage"], "x")

    def test_said_quote_inline(self):
        data = self.heal('{"def":"He said "hi" to her."}')
        self.assertEqual(data["def"], 'He said "hi" to her.')

    def test_key_following_healed_value_still_parsed(self):
        # the healed value must not swallow the following key/value pair
        s = ('{"vocabulary":[{"quoted_sentence":"a" and "b",'
             '"example_usage":"The two roads run parallel."}]}')
        data = self.heal(s)
        item = data["vocabulary"][0]
        self.assertEqual(item["quoted_sentence"], 'a" and "b')
        self.assertEqual(item["example_usage"], "The two roads run parallel.")

    def test_valid_json_with_escaped_quotes_untouched(self):
        s = '{"def":"value with \\"escaped\\" quotes","n":1}'
        self.assertEqual(self.heal(s), {"def": 'value with "escaped" quotes', "n": 1})


class TestRealLogHealing(unittest.TestCase):
    """End-to-end: the RAW RESPONSE from the real failing log must parse after healing."""
    LOG = r"e:\teacher-wiki\logs\20260916_175748_341_extract_vocabulary_Book_2_Unit_6_Section_A_JSON_MODE.log"

    def test_1757_book2_unit6_log_heals(self):
        import os
        if not os.path.exists(self.LOG):
            self.skipTest("log file not present")
        with open(self.LOG, encoding="utf-8") as f:
            content = f.read()
        marker = "--- RAW RESPONSE ---"
        self.assertIn(marker, content)
        raw = content.split(marker, 1)[1].strip()
        with self.assertRaises(json.JSONDecodeError):
            json.loads(raw)  # proves the raw response really was broken
        healed = LLMClient._heal_json(None, raw)
        data = json.loads(healed)  # raises if healing failed
        vocab = data["vocabulary"]
        self.assertGreaterEqual(len(vocab), 20)
        by_word = {v["word"]: v for v in vocab}
        self.assertIn('their sailing ships burned" and "watched their vessels go up in flames',
                      by_word["vessel"]["quoted_sentence"])
        self.assertIn("decision-making", by_word["investigate"]["quoted_sentence"])


if __name__ == "__main__":
    unittest.main()
