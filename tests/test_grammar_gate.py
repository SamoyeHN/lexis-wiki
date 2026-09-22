import unittest
import json
from librarian.evaluator import (
    LogEvaluator,
    _score_schema,
    _score_verbatim,
    _score_pedagogy,
    _score_uniqueness,
    prune_hallucinated_items,
)

SAMPLE_SOURCE = """CONTENT:
Considered one of the toughest marathon events in the world, the 875-kilometer annual Australian race, a route from Sydney to Melbourne, was a harsh test of endurance for the world's top athletes, regardless of their age.
Although he was still far behind the world-class athletes, he kept at it.
It was said that Cliff Young had never kept a single prize.
As the famous saying goes, "Where there's a will, there's a way!"
"""

class TestGrammarDeterministicCodeGate(unittest.TestCase):
    def test_empty_json_scored_zero(self):
        """Empty json {} must receive 0 score and fatal schema flag."""
        score, flags = _score_schema({})
        self.assertEqual(score, 0.0)
        self.assertTrue(any("empty object" in f for f in flags))

        # Test simulated evaluation of empty object
        log_entry = {
            "task": "extract_grammar_test",
            "parsed_json": {},
            "raw_response": "{}",
            "user_prompt": SAMPLE_SOURCE,
        }
        res = LogEvaluator.evaluate_log(log_entry)
        self.assertEqual(res.get("composite_score"), 0.0)

    def test_prune_hallucinated_grammar_item(self):
        """Items with quotes not found in source text must be pruned deterministically."""
        data = {
            "grammar_patterns": [
                {
                    "category": "Concessive clauses",
                    "pattern_formula": "Although + [Clause], [Main Clause]",
                    "quote": "Although he was still far behind the world-class athletes, he kept at it.",
                    "design_audit": "AUDIT: Although he was still far behind...",
                    "imitation_example": "Although empirical anomalies initially surfaced, researchers confirmed theoretical consistency.",
                    "common_mistakes": "Do not use 'but' in the main clause when using 'although'."
                },
                {
                    "category": "Participial clauses",
                    "pattern_formula": "[Past Participle] + [Noun Phrase], [Subject] + [Verb]",
                    "quote": "Having seen the news, Cliff continued running through the night without sleeping.",
                    "design_audit": "AUDIT: Having seen the news...",
                    "imitation_example": "Having synthesized the compound, the lab commenced longitudinal trials.",
                    "common_mistakes": "Dangling participle."
                }
            ]
        }
        pruned_data, notices = prune_hallucinated_items(data, SAMPLE_SOURCE, task_type="grammar")
        surviving = pruned_data["grammar_patterns"]
        self.assertEqual(len(surviving), 1)
        self.assertEqual(surviving[0]["category"], "Concessive clauses")
        self.assertTrue(any("Having seen the news" in n or "not found in source" in n for n in notices))

    def test_prune_mismatched_grammar_anchor(self):
        """If a formula has literal anchors missing from the quote, it must be pruned."""
        data = {
            "grammar_patterns": [
                {
                    "category": "Cleft sentences",
                    "pattern_formula": "It + [be] + [NP] + that + [S]",
                    "quote": "Considered one of the toughest marathon events in the world, the 875-kilometer annual Australian race was a harsh test.",
                    "design_audit": "AUDIT: misclassified cleft",
                    "imitation_example": "It is peer review that preserves integrity.",
                    "common_mistakes": "Omitting 'that'."
                }
            ]
        }
        pruned_data, notices = prune_hallucinated_items(data, SAMPLE_SOURCE, task_type="grammar")
        self.assertEqual(len(pruned_data["grammar_patterns"]), 0)
        self.assertTrue(any("missing from quote" in n for n in notices))

    def test_pedagogy_slot_whitelist_and_cobuild_notation(self):
        """Standard slot whitelist and COBUILD unbracketed bare POS notation must pass pedagogy."""
        items = [
            {
                "category": "Concessive clauses",
                "pattern_formula": "Although + [Clause], [Main Clause]",
                "quote": "Although he was still far behind the world-class athletes, he kept at it.",
                "design_audit": "AUDIT: Although + Clause",
                "imitation_example": "Although empirical anomalies surfaced, findings remained valid.",
                "common_mistakes": "Incorrect coordination.",
            },
            {
                "category": "Hedging devices",
                "pattern_formula": "It + [be] + said + that + [Clause]",
                "quote": "It was said that Cliff Young had never kept a single prize.",
                "design_audit": "AUDIT: It was said that...",
                "imitation_example": "It is estimated that global temperatures will rise.",
                "common_mistakes": "Misplaced agent.",
            },
            {
                "category": "Concessive clauses",
                # Unbracketed COBUILD notation
                "pattern_formula": "although NP V, NP V",
                "quote": "Although he was still far behind the world-class athletes, he kept at it.",
                "design_audit": "AUDIT: although NP V, NP V",
                "imitation_example": "Although researchers tested the hypothesis, conclusive proof remained elusive.",
                "common_mistakes": "Punctuation comma splice.",
            }
        ]
        score, flags = _score_pedagogy(items, "grammar", SAMPLE_SOURCE)
        self.assertEqual(score, 25.0)
        self.assertEqual(flags, [])

    def test_pedagogy_any_macro_domain_tolerance(self):
        """Boundary patterns (chosen category fails the strict check but the quote fits
        ANOTHER macro functional domain) must be tolerated; only quotes matching NO domain
        are rejected (keeping the original fatal reason)."""
        # Boundary: labelled Information Packaging but structurally Rhetoric (not only ... but).
        # The strict IP check fails, but the any-macro-domain rule tolerates it -> PASS.
        boundary_item = {
            "category": "Information Packaging",
            "pattern_formula": "[S] + [VP] + [NP]",
            "quote": "You will not only work hard in class, but also keep a balanced life outside it.",
            "design_audit": "AUDIT: boundary pattern",
            "imitation_example": "Students not only memorize, but also apply.",
            "common_mistakes": "n/a",
        }
        score_b, flags_b = _score_pedagogy([boundary_item], "grammar", SAMPLE_SOURCE)
        # Auto-remaps to Rhetoric & Emphasis and receives soft penalty deduction of 2.0 (23.0 / 25.0)
        self.assertEqual(score_b, 23.0)
        self.assertEqual(boundary_item["category"], "Rhetoric & Emphasis")
        self.assertFalse(any("failed pedagogy check" in f for f in flags_b))
        self.assertTrue(any("Deterministic Auto-Remap" in f for f in flags_b))

        # Marker-less: matches NO macro domain -> must still be rejected (fatal reason kept).
        weak_item = {
            "category": "Information Packaging",
            "pattern_formula": "[S] + [VP] + [NP]",
            "quote": "The clock woke me up every morning in college.",
            "design_audit": "AUDIT: simple sentence",
            "imitation_example": "It rang early.",
            "common_mistakes": "n/a",
        }
        score_w, flags_w = _score_pedagogy([weak_item], "grammar", SAMPLE_SOURCE)
        self.assertEqual(score_w, 0.0)
        self.assertTrue(any("failed pedagogy check" in f and "lacking" in f for f in flags_w))

    def test_pedagogy_anti_triviality_and_clean_formulas(self):
        """Conversational fillers (e.g. But [S]) must be flagged; flexible syntactic formulas pass."""
        # 1. Flexible natural formula passes without artificial slot rejections
        items_valid_slot = [{
            "category": "Concessive clauses",
            "pattern_formula": "Although [Clause], [Main Clause]",
            "quote": "Although he was still far behind the world-class athletes, he kept at it.",
            "design_audit": "AUDIT: valid flexible slot",
            "imitation_example": "Ex.",
            "common_mistakes": "Mistake.",
        }]
        score, flags1 = _score_pedagogy(items_valid_slot, "grammar", SAMPLE_SOURCE)
        self.assertEqual(score, 25.0)
        self.assertEqual(flags1, [])

        # 2. Trivial conversational filler formula is flagged
        items_trivial = [{
            "category": "Concessive clauses",
            "pattern_formula": "But [S]",
            "quote": "Although he was still far behind the world-class athletes, he kept at it.",
            "design_audit": "AUDIT: trivial",
            "imitation_example": "Ex.",
            "common_mistakes": "Mistake.",
        }]
        _, flags2 = _score_pedagogy(items_trivial, "grammar", SAMPLE_SOURCE)
        self.assertTrue(any("trivial formula" in f for f in flags2))

    def test_deduplication_in_pruning(self):
        """Duplicate (formula, quote) items must be pruned deterministically."""
        data = {
            "grammar_patterns": [
                {
                    "category": "Concessive clauses",
                    "pattern_formula": "Although + [Clause], [Main Clause]",
                    "quote": "Although he was still far behind the world-class athletes, he kept at it.",
                    "design_audit": "AUDIT: duplicate 1",
                    "imitation_example": "Although tests were hard, he passed.",
                    "common_mistakes": "Mistake.",
                },
                {
                    "category": "Concessive clauses",
                    "pattern_formula": "Although + [Clause], [Main Clause]",
                    "quote": "Although he was still far behind the world-class athletes, he kept at it.",
                    "design_audit": "AUDIT: duplicate 2",
                    "imitation_example": "Although tests were hard, he passed.",
                    "common_mistakes": "Mistake.",
                }
            ]
        }
        pruned_data, notices = prune_hallucinated_items(data, SAMPLE_SOURCE, task_type="grammar")
        self.assertEqual(len(pruned_data["grammar_patterns"]), 1)
        self.assertTrue(any("Pruned duplicate item" in n for n in notices))

    def test_normalize_grammar_formula(self):
        """Test normalization of legacy verbose grammar slots to standard COBUILD tokens."""
        from librarian.processor import WikiProcessor

        cases = [
            ("Although + [Clause], [Main Clause]", "Although + [S], [S]"),
            ("It + [Copula] + [Noun Phrase] + that + [Clause]", "It + [be] + [NP] + that + [S]"),
            ("[Past Participle] + [Noun Phrase], [Subject] + [Predicate]", "[V3] + [NP], [S] + [S]"),
            ("Not only + [Auxiliary] + [Subject] + [Base Verb]", "Not only + [aux] + [S] + [V]"),
            ("It + [be] + [Evaluative Adjective] + [Infinitive]", "It + [be] + [adj] + [to-V]"),
            ("The + [adj/adv] + [S], the + [adj/adv] + [S]", "The + [adj/adv] + [S], the + [adj/adv] + [S]"),
            ("The + [adjective/adverb] + [Clause]", "The + [adj/adv] + [S]"),
        ]
        for raw, expected in cases:
            norm = WikiProcessor.normalize_grammar_formula(raw)
            self.assertEqual(norm, expected)

    def test_compound_slash_slot_pedagogy_passes(self):
        """Compound slash-separated slots (e.g. [adj/adv]) must pass pedagogy validation."""
        items = [{
            "category": "Abstract frames",
            "pattern_formula": "The + [adj/adv] + [S], the + [adj/adv] + [S]",
            "quote": "As the famous saying goes, \"Where there's a will, there's a way!\"",
            "design_audit": "AUDIT: The + adj/adv",
            "imitation_example": "The higher the pressure, the faster the reaction proceeds.",
            "common_mistakes": "Incomplete comparative clause.",
        }]
        score, flags = _score_pedagogy(items, "grammar", SAMPLE_SOURCE)
        self.assertEqual(score, 25.0)
        self.assertEqual(flags, [])


    def test_completeness_missing_quote_and_pedagogy_fails_gate(self):
        """P0-2 fix: items missing quote or pedagogy must NOT get free 30 score and must be pruned."""
        # 1. Stripped output like Qwen (only design_audit and pattern_formula, no quote, no pedagogy)
        stripped_items = [
            {
                "design_audit": "AUDIT: Given that...",
                "pattern_formula": "Given that + [S], [S]"
            },
            {
                "design_audit": "AUDIT: The more...",
                "pattern_formula": "The + [adj/adv] + [S], the + [adj/adv] + [S]"
            }
        ]
        
        # _score_verbatim must NOT return full marks (30) when items lack quote; it must return 0.0
        v_score, v_flags = _score_verbatim(stripped_items, "grammar", SAMPLE_SOURCE)
        self.assertEqual(v_score, 0.0)
        self.assertTrue(any("Missing quote" in f for f in v_flags))

        # _score_pedagogy must flag missing imitation_example and common_mistakes
        p_score, p_flags = _score_pedagogy(stripped_items, "grammar", SAMPLE_SOURCE)
        self.assertEqual(p_score, 0.0)
        self.assertTrue(any("missing imitation_example" in f for f in p_flags))

        # prune_hallucinated_items must prune incomplete items missing required fields
        data = {"grammar_patterns": stripped_items}
        pruned_data, notices = prune_hallucinated_items(data, SAMPLE_SOURCE, task_type="grammar")
        self.assertEqual(len(pruned_data["grammar_patterns"]), 0)
        self.assertTrue(any("missing required" in n for n in notices))

        # Composite score must be low and fail the 80 threshold
        log_entry = {
            "task": "extract_grammar_test",
            "parsed_json": {"grammar_patterns": stripped_items},
            "raw_response": '{"grammar_patterns": [{"design_audit": "AUDIT: Given that...", "pattern_formula": "Given that + [S], [S]"}]}',
            "user_prompt": SAMPLE_SOURCE,
        }
        res = LogEvaluator.evaluate_log(log_entry)
        self.assertLess(res.get("composite_score", 100), 50.0)


    def test_contraction_aware_anchor_matching(self):
        """P0-1 fix: formulas with anchors like 'It' or 'there' must match contracted quotes like 'It\'s' or 'there\'s'."""
        source = "CONTENT:\nIt's high time that educators and families worked together to tackle the crisis.\nThere's no doubt that perseverance leads to success."
        item_it = {
            "category": "Abstract frames",
            "pattern_formula": "It + [be] + high time that + [S]",
            "quote": "It's high time that educators and families worked together to tackle the crisis.",
            "design_audit": "AUDIT: It is high time that...",
            "imitation_example": "It is high time that policy makers implemented systemic safeguards.",
            "common_mistakes": "Using present tense in the that-clause.",
        }
        item_there = {
            "category": "Abstract frames",
            "pattern_formula": "There + [be] + no doubt that + [S]",
            "quote": "There's no doubt that perseverance leads to success.",
            "design_audit": "AUDIT: There is no doubt that...",
            "imitation_example": "There is no doubt that rigorous methodology ensures validity.",
            "common_mistakes": "Omitting 'that'.",
        }
        items = [item_it, item_there]

        # 1. _score_verbatim must pass both items without "Grammar formula anchor does not appear" flag
        v_score, v_flags = _score_verbatim(items, "grammar", source)
        self.assertEqual(v_score, 30.0)
        self.assertFalse(any("anchor" in f and "does not appear" in f for f in v_flags))

        # 2. _score_pedagogy must pass both items with full 25 score
        p_score, p_flags = _score_pedagogy(items, "grammar", source)
        self.assertEqual(p_score, 25.0)
        self.assertEqual(p_flags, [])

        # 3. prune_hallucinated_items must NOT prune valid items due to contracted anchors
        data = {"grammar_patterns": items}
        pruned_data, notices = prune_hallucinated_items(data, source, task_type="grammar")
        self.assertEqual(len(pruned_data["grammar_patterns"]), 2)
        self.assertFalse(any("missing from quote" in n for n in notices))


    def test_truthful_quarantine_status_and_fatal_flags(self):
        """P0 fix: verify unified FATAL_QA_FLAGS and truthful QUARANTINED status recording."""
        from librarian.evaluator import FATAL_QA_FLAGS
        self.assertIn("does not appear in quoted sentence", FATAL_QA_FLAGS)
        self.assertIn("Grammar formula anchor", FATAL_QA_FLAGS)
        self.assertIn("duplicate", FATAL_QA_FLAGS)
        self.assertIn("pure fabrication", FATAL_QA_FLAGS)

        # Test transparent delivery: extractions ship directly to extractions/ with review_needed status
        from librarian.processor import WikiProcessor
        import tempfile, shutil
        from pathlib import Path

        tmp_dir = Path(tempfile.mkdtemp())
        try:
            processor = WikiProcessor()
            processor.config.project_root = tmp_dir
            processor.config.data["wiki_dir"] = "wiki"
            processor.config.ensure_dirs()

            unit_name = "Test_Unit_Transparent_Delivery"
            review_needed_data = {
                "_qa_audit": {
                    "composite_score": 65.0,
                    "flags": ["❌ Grammar formula anchor 'although' does not appear in quoted sentence: 'Test...'"],
                    "status": "SUCCESS",
                },
                "grammar_patterns": []
            }
            res = processor._save_extraction_results(review_needed_data, f"{unit_name}.md", category_override="grammar")
            md_path = tmp_dir / "wiki" / unit_name / "extractions" / f"{unit_name}_grammar.md"
            self.assertTrue(md_path.exists())
            with open(md_path, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertIn('qa_status: "review_needed"', content)
            self.assertIn("qa_score: 65", content)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_non_verbatim_quote_triggers_fatal_flag(self):
        """P1 fix: verify Non-verbatim quote triggers fatal flag and drops item."""
        from librarian.evaluator import _score_verbatim, FATAL_QA_FLAGS
        self.assertIn("Non-verbatim quote detected", FATAL_QA_FLAGS)

        source = "CONTENT:\nCliff Young won the ultra-marathon through sheer determination and unorthodox pacing."
        invented_item = {
            "pattern_formula": "It + [be] + essential that + [S]",
            "quote": "It is essential that we address the global climate crisis immediately.",
        }
        score, flags = _score_verbatim([invented_item], "grammar", source)
        self.assertEqual(score, 0.0)
        self.assertTrue(any("Non-verbatim quote detected" in f for f in flags))
        # Ensure that any flag in flags matches FATAL_QA_FLAGS
        has_fatal = any(any(fatal in f for fatal in FATAL_QA_FLAGS) for f in flags)
        self.assertTrue(has_fatal)


    def test_trivial_anchor_fails_pedagogy_check(self):
        """Verify that trivial formulas anchored only by 'But', 'And', 'Of course', etc. fail pedagogy check."""
        source = "CONTENT:\nBut he didn't give up. Of course, this was unexpected. And everyone was amazed."
        trivial_items = [
            {
                "category": "Concessive clauses",
                "pattern_formula": "But [S]",
                "quote": "But he didn't give up.",
                "design_audit": "AUDIT: But he didn't give up...",
                "imitation_example": "But the results remained inconclusive.",
                "common_mistakes": "Starting sentence with but.",
            },
            {
                "category": "Concessive clauses",
                "pattern_formula": "Of course, [S]",
                "quote": "Of course, this was unexpected.",
                "design_audit": "AUDIT: Of course, this was unexpected...",
                "imitation_example": "Of course, further research is needed.",
                "common_mistakes": "Overusing conversational fillers.",
            }
        ]
        score, flags = _score_pedagogy(trivial_items, "grammar", source)
        self.assertEqual(score, 0.0)
        self.assertTrue(any("trivial formula" in f for f in flags))


    def test_information_packaging_lacking_markers_fails_pedagogy_check(self):
        """Verify that assigning Information Packaging to a quote lacking non-finite, dummy-it, or nominalization fails check."""
        source = "CONTENT:\nHe walks to school every day."
        item = {
            "category": "Information Packaging",
            "pattern_formula": "[Subject] + walks + to + [NP]",
            "quote": "He walks to school every day.",
            "design_audit": "AUDIT: He walks to school...",
            "imitation_example": "She drives to work every morning.",
            "common_mistakes": "Simple present tense errors.",
        }
        score, flags = _score_pedagogy([item], "grammar", source)
        self.assertEqual(score, 0.0)
        self.assertTrue(any("assigned to quote lacking non-finite clauses" in f for f in flags))


    def test_auto_remap_antithesis_and_past_tense_interpretive_verbs(self):
        """Verify that explicit physical mismatches are auto-remapped with past tense verbs and soft deduction."""
        source = (
            "CONTENT:\n"
            "Whether a job is to be designated as labor or work depends, not on the job itself, but on the tastes of the individual who undertakes it. "
            "He reported the anomaly immediately, which suggested that the system was compromised."
        )
        items = [
            # 1. Antithesis mislabeled as Logic & Stance -> auto-remaps to Rhetoric & Emphasis
            {
                "category": "Logic & Stance",
                "pattern_formula": "[Subject] + depends, not on + [NP], but on + [NP]",
                "quote": "Whether a job is to be designated as labor or work depends, not on the job itself, but on the tastes of the individual who undertakes it.",
                "design_audit": "AUDIT: Antithesis not... but...",
                "imitation_example": "Success depends, not on luck, but on persistent effort.",
                "common_mistakes": "Missing comma.",
            },
            # 2. Past tense interpretive verb (, which suggested that) mislabeled as Information Packaging -> auto-remaps to Cohesion & Framing
            {
                "category": "Information Packaging",
                "pattern_formula": "[Clause], which + suggested + that + [Proposition]",
                "quote": "He reported the anomaly immediately, which suggested that the system was compromised.",
                "design_audit": "AUDIT: which suggested that...",
                "imitation_example": "The meter spiked, which showed that pressure had escalated.",
                "common_mistakes": "Incorrect relative pronoun.",
            }
        ]
        score, flags = _score_pedagogy(items, "grammar", source)
        # Both items remapped: 25.0 - (2 * 2.0) = 21.0
        self.assertEqual(score, 21.0)
        self.assertEqual(items[0]["category"], "Rhetoric & Emphasis")
        self.assertEqual(items[1]["category"], "Cohesion & Framing")
        self.assertTrue(any("Antithesis" in f or "not... but..." in f for f in flags))
        self.assertTrue(any("which + [interpretive verb] + that" in f for f in flags))


if __name__ == "__main__":
    unittest.main()



