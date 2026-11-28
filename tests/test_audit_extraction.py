"""Unit tests for the Extraction Semantic Re-Audit Suite."""

from __future__ import annotations

import json
from pathlib import Path

from librarian.audit_extraction import (
    audit_grammar_markdown,
    audit_mindmap_json,
    audit_summary_markdown,
    audit_unit_extractions,
    audit_vocabulary_markdown,
    audit_vocabulary_wsd_semantic,
    cure_vocabulary_item,
    match_definition_to_sense,
    save_extraction_audit_log,
)


def test_audit_vocabulary_clean_and_defects():
    source = "The quick brown fox jumps over the lazy dog in the sunny morning."

    clean_vocab = """
# Vocabulary
## [[fox]] noun B1
- **Definition**: a wild animal like a dog with red-brown fur and a bushy tail
- **Quoted Sentence**: The quick brown fox jumps over the lazy dog in the sunny morning.
"""
    defects, stats = audit_vocabulary_markdown(clean_vocab, source.lower())
    assert defects == []
    assert stats["item_count"] == 1
    assert "fox" in stats["lemmas"]

    hallucinated_vocab = """
# Vocabulary
## [[wolf]] noun B1
- **Definition**: a wild carnivorous mammal
- **Quoted Sentence**: A dangerous wolf roamed freely in the dark northern forest.
"""
    defects, stats = audit_vocabulary_markdown(hallucinated_vocab, source.lower())
    assert len(defects) == 1
    assert "quote not found in source text" in defects[0]


def test_audit_grammar_clean_and_defects():
    source = "With the coming of the telephone, people could not only read words, but also hear each other."

    clean_grammar = """
# Grammar
## [[Correlative Coordination]]
- **Quote**: With the coming of the telephone, people could not only read words, but also hear each other.
- **Pattern Formula**: [S] + not only + [VP], but also + [VP]
- **Pedagogical Function**: Balances two equal ideas or actions together.
"""
    defects, stats = audit_grammar_markdown(clean_grammar, source.lower())
    assert defects == []
    assert stats["item_count"] == 1

    bad_formula_grammar = """
# Grammar
## [[Correlative Coordination]]
- **Quote**: With the coming of the telephone, people could not only read words, but also hear each other.
- **Pattern Formula**: not only but also
- **Pedagogical Function**: Balances two equal ideas or actions together.
"""
    defects, stats = audit_grammar_markdown(bad_formula_grammar, source.lower())
    assert any("formula lacks Quirk/Biber syntactic markers" in d for d in defects)


def test_audit_summary_clean_and_defects():
    source = (
        "The passage explores modern communication technology from mail to internet software. "
        "Young people seek balance between social connection, family love, personal autonomy, and privacy. "
        "New tools provide freedom and smarter choices when all generations communicate."
    )

    clean_summary = """
# Summary
## Narrative Overview
The passage discusses modern communication technology moving from traditional mail to internet software.
Young people navigate personal freedom, family connection, and privacy.

## Lesson Hook
How has digital communication transformed your personal freedom?

## Essential Questions
- How do communication tools evolve over time?
"""
    defects, stats = audit_summary_markdown(clean_summary, source.lower())
    assert defects == []
    assert stats["has_narrative"] is True
    assert stats["has_hook"] is True
    assert stats["has_questions"] is True

    missing_sections_summary = """
# Summary
## Quick Notes
Some random notes without narrative overview or essential questions.
"""
    defects, _ = audit_summary_markdown(missing_sections_summary, source.lower())
    assert any("Missing 'Narrative Overview'" in d for d in defects)
    assert any("Missing 'Lesson Hook'" in d for d in defects)
    assert any("Missing 'Essential Questions'" in d for d in defects)


def test_audit_mindmap_json_mece(tmp_path: Path):
    clean_mm = {
        "title": "Clean Mindmap",
        "root_name": "Root Concept",
        "branches": [
            {
                "branch_name": "Historical Background",
                "sub_branches": [
                    {"sub_branch_name": "Early Period", "leaves": ["Item 1", "Item 2"]},
                    {"sub_branch_name": "Modern Period", "leaves": ["Item 3"]},
                ],
            },
            {
                "branch_name": "Sociological Impact",
                "sub_branches": [
                    {"sub_branch_name": "Family Relations", "leaves": ["Item 4"]},
                    {"sub_branch_name": "Peer Relations", "leaves": ["Item 5"]},
                ],
            },
        ],
    }
    mm_file = tmp_path / "clean_mindmap.json"
    mm_file.write_text(json.dumps(clean_mm), encoding="utf-8")

    defects, stats = audit_mindmap_json(mm_file)
    assert defects == []
    assert stats["branch_count"] == 2
    assert stats["max_depth"] >= 2

    # Duplicate branch MECE violation
    dup_mm = {
        "title": "Dup Mindmap",
        "root_name": "Root Concept",
        "branches": [
            {"branch_name": "History", "sub_branches": []},
            {"branch_name": "History", "sub_branches": []},
        ],
    }
    dup_file = tmp_path / "dup_mindmap.json"
    dup_file.write_text(json.dumps(dup_mm), encoding="utf-8")

    defects, _ = audit_mindmap_json(dup_file)
    assert any("MECE violation: Duplicate branch name" in d for d in defects)


def test_audit_unit_extractions_golden_unit():
    report = audit_unit_extractions("Book_1_Unit_1_Passage_A")
    assert report.passed is True
    assert report.score == 100.0
    assert report.defects == []
    assert "vocabulary" in report.stats
    assert "grammar" in report.stats
    assert "summary" in report.stats
    assert "mindmap" in report.stats


def test_match_definition_to_sense():
    candidates = [
        {"definition": "very carefully", "examples": ["watched closely"]},
        {"definition": "in a way that is close to other things in time or space", "examples": ["packed closely"]},
    ]
    assert match_definition_to_sense("very carefully", candidates) == 0
    assert match_definition_to_sense("in a way that is close to other things", candidates) == 1
    # Fallback to 0 if no match
    assert match_definition_to_sense("something entirely different", candidates) == 0


def test_cure_vocabulary_item():
    raw_md = """
# Vocabulary: Demo
## [[closely]] adverb B1
- **Definition**: in a way that is close to other things in time or space
- **Quoted Sentence**: They want to follow the time closely, but they also long for peace of mind.
- **Example Usage**: We were so closely packed in the elevator I could hardly move.
- **Pattern**: *followed closely*

## [[simple]] adjective A2
- **Definition**: not difficult
"""
    cured = cure_vocabulary_item(
        raw_md,
        "closely",
        "very carefully",
        "The detective watched him closely, waiting for a reply.",
    )
    assert "- **Definition**: very carefully" in cured
    assert "- **Example Usage**: The detective watched him closely, waiting for a reply." in cured
    assert "## [[simple]] adjective A2" in cured
    assert "- **Definition**: not difficult" in cured


def test_audit_vocabulary_wsd_semantic_detection(monkeypatch):
    from librarian.linguistics import LinguisticEngine

    fake_candidates = [
        {"definition": "very carefully", "examples": ["watched closely"]},
        {"definition": "in a way that is close to other things in time or space", "examples": ["packed closely"]},
    ]

    monkeypatch.setattr(
        LinguisticEngine,
        "build_top_k_vocab_candidates",
        lambda *args, **kwargs: fake_candidates,
    )
    # LLM chooses index 0 ("very carefully")
    monkeypatch.setattr(
        LinguisticEngine,
        "disambiguate_via_llm",
        lambda *args, **kwargs: 0,
    )

    items = [
        {
            "lemma": "closely",
            "pos": "adverb",
            "quote": "They want to follow the time closely.",
            "definition": "in a way that is close to other things in time or space",
        }
    ]

    defects, cures, overruled = audit_vocabulary_wsd_semantic(items, "They want to follow the time closely.")
    assert len(defects) == 1
    assert "closely" in defects[0]
    assert "definition misaligned" in defects[0]
    assert len(cures) == 1
    assert cures[0]["lemma"] == "closely"
    assert cures[0]["new_definition"] == "very carefully"
    assert cures[0]["new_example"] == "watched closely"
    assert len(overruled) == 0


def test_audit_vocabulary_wsd_semantic_overruled_by_gate(monkeypatch):
    from librarian.linguistics import LinguisticEngine

    fake_candidates = [
        {
            "definition": "the right to do what you want",
            "score": 15.0,
            "raw_sense": {"grammar": "[countable, uncountable]"},
            "examples": ["defending our freedoms"],
        },
        {
            "definition": "the state of not being a prisoner or slave",
            "score": 10.0,
            "raw_sense": {"grammar": "[uncountable]"},
            "examples": ["struggling for freedom"],
        },
    ]

    monkeypatch.setattr(
        LinguisticEngine,
        "build_top_k_vocab_candidates",
        lambda *args, **kwargs: fake_candidates,
    )
    # LLM chooses index 1 ([uncountable]) for a sentence containing plural "freedoms"
    monkeypatch.setattr(
        LinguisticEngine,
        "disambiguate_via_llm",
        lambda *args, **kwargs: 1,
    )

    items = [
        {
            "lemma": "freedom",
            "pos": "noun",
            "quote": "They fought for essential human freedoms across the globe.",
            "definition": "the right to do what you want",
        }
    ]

    defects, cures, overruled = audit_vocabulary_wsd_semantic(items, "They fought for essential human freedoms across the globe.")
    # Must NOT produce defect (0 pt penalty)
    assert len(defects) == 0
    assert len(cures) == 0
    # Must be recorded in overruled
    assert len(overruled) == 1
    assert overruled[0]["lemma"] == "freedom"
    assert "Gate 4" in overruled[0]["gate"]
    assert "uncountable" in overruled[0]["reason"].lower()


def test_save_extraction_audit_log(tmp_path):
    summary = {
        "target": "extraction",
        "total": 1,
        "passed": 0,
        "failed": 1,
        "avg_score": 70.0,
        "enable_llm": True,
        "fix": False,
        "reports": [
            {
                "unit": "Test_Unit_WSD",
                "passed": False,
                "score": 70.0,
                "defects": [
                    "[vocabulary-wsd] 'offer' definition misaligned: current='to say that you are willing...', context requires Sense #1 'to provide something that people need...'",
                    "[grammar-semantic] Pattern '[[Negative Inversion]]' instantiation flaw: The quote does not contain a negative inversion structure.",
                ],
                "stats": {
                    "vocabulary": {
                        "item_count": 1,
                        "wsd_defect_details": [
                            {
                                "lemma": "offer",
                                "quote": "They offer assistance.",
                                "current_def": "to say that you are willing...",
                                "chosen_idx": 0,
                                "chosen_def": "to provide something that people need...",
                                "chosen_example": "They offered help.",
                            }
                        ]
                    },
                    "grammar": {
                        "item_count": 1,
                        "grammar_defect_details": [
                            {
                                "pattern": "Negative Inversion",
                                "quote": "They offer assistance.",
                                "formula": "Not only + [Aux] + [S] + [VP]",
                                "category": "grammar-instantiation",
                                "issue": "The quote does not contain a negative inversion structure.",
                                "action": "Replace quote with authentic negative inversion sentence.",
                            }
                        ]
                    }
                },
                "error": None,
            }
        ],
    }

    paths = save_extraction_audit_log(summary, project_root=str(tmp_path))
    assert paths["log_path"].is_file()
    assert "Test_Unit_WSD_reaudit_extraction.log" in paths["log_path"].name
    assert "json_path" not in paths
    assert "latest_path" not in paths
    # Ensure no .json log file was written
    assert not list((tmp_path / "logs").glob("*.json"))

    content = paths["log_path"].read_text(encoding="utf-8")
    assert "[UNIT DEFECT TICKET] Test_Unit_WSD" in content
    assert "Category:  vocabulary-wsd" in content
    assert "Lemma:     offer" in content
    assert "Context:   \"They offer assistance.\"" in content
    assert "Suggested: Sense #1: to provide something that people need..." in content
    assert "Action:   Run `python -m librarian.cli" not in content

    # Check grammar defect ticket
    assert "Category:  grammar-semantic" in content
    assert "Pattern:   [[Negative Inversion]]" in content
    assert "Quote:     \"They offer assistance.\"" in content
    assert "Formula:   Not only + [Aux] + [S] + [VP]" in content
    assert "Issue:     The quote does not contain a negative inversion structure." in content
    assert "- [[Negative Inversion]] [FAIL]" in content


def test_save_extraction_audit_log_clean_unit(tmp_path):
    """Verify that passed units with zero defects explicitly log grammar and component inventory."""
    clean_summary = {
        "target": "extraction",
        "total": 1,
        "passed": 1,
        "failed": 0,
        "avg_score": 100.0,
        "enable_llm": True,
        "fix": False,
        "reports": [
            {
                "unit": "Book_1_Unit_4_Passage_A",
                "passed": True,
                "score": 100.0,
                "defects": [],
                "overruled": [
                    {
                        "lemma": "freedom",
                        "quote": "They fought for essential human freedoms across the globe.",
                        "current_idx": 0,
                        "current_def": "the right to do what you want",
                        "current_score": 26.0,
                        "chosen_idx": 2,
                        "chosen_def": "the state of not being a prisoner or slave",
                        "chosen_score": 8.5,
                        "gate": "Gate 4 (Noun Plurality)",
                        "reason": "Surface form 'freedoms' is plural; LLM suggested [uncountable]-only sense.",
                    }
                ],
                "stats": {
                    "vocabulary": {
                        "item_count": 20,
                    },
                    "grammar": {
                        "item_count": 3,
                        "patterns": [
                            "Restrictive Relative Clauses (who / which / that)",
                            "Gerundial Subject Nominalization",
                            "Conditional Adverbial Clauses (If-Conditionals)",
                        ],
                    },
                    "summary": {
                        "summary_words": 150,
                    },
                    "mindmap": {
                        "branch_count": 4,
                        "total_leaves": 12,
                        "max_depth": 3,
                    },
                },
                "error": None,
            }
        ],
    }

    paths = save_extraction_audit_log(clean_summary, project_root=str(tmp_path))
    content = paths["log_path"].read_text(encoding="utf-8")

    assert "[UNIT AUDIT INVENTORY] Book_1_Unit_4_Passage_A (Status: PASS, Score: 100.0 / 100)" in content
    assert "• Vocabulary: 20 items audited" in content
    assert "• Grammar:    3 patterns audited" in content
    assert "- [[Restrictive Relative Clauses (who / which / that)]] [PASS]" in content
    assert "- [[Gerundial Subject Nominalization]] [PASS]" in content
    assert "- [[Conditional Adverbial Clauses (If-Conditionals)]] [PASS]" in content
    assert "• Summary:    Verified (150 words" in content
    assert "• Mindmap:    Verified (4 branches, 12 leaves, depth 3)" in content
    assert "✓ Defect Status: Zero defects detected in this unit." in content
    assert "🎉 All audited units passed with zero defects!" in content
    assert "🛡️ [OVERRULED LLM SUGGESTIONS (0 SCORE IMPACT)] (1):" in content
    assert "[OVERRULED] [[freedom]]" in content
    assert "Veto Gate:          Gate 4 (Noun Plurality)" in content
    assert "Veto Reason:        Surface form 'freedoms' is plural; LLM suggested [uncountable]-only sense." in content
    assert "Current Extraction: Sense #1 (\"the right to do what you want\") [Score: 26.0]" in content
    assert "LLM Suggestion:     Sense #3 (\"the state of not being a prisoner or slave\") [Score: 8.5]" in content

    # Check console format output as well
    from librarian.audit_extraction import format_extraction_summary
    console_out = format_extraction_summary(clean_summary)
    assert "[Overruled LLM Suggestions (0 score impact)]:" in console_out
    assert "• [[freedom]] (LLM: Sense #3 ➔ OVERRULED)" in console_out
    assert "Reason: Gate 4 (Noun Plurality)" in console_out


def test_audit_grammar_semantic_mocked(monkeypatch):
    """Verify Level-2 grammar semantic audit with LLM judge evaluation."""
    from librarian.audit_extraction import audit_grammar_semantic
    from librarian.llm import LLMClient

    items = [
        {
            "pattern": "Negative Inversion",
            "quote": "One of the things that tend to annoy punctual people...",
            "formula": "Not only + [Aux] + [S] + [VP]",
            "pedagogy": "Fronts negative adverbial.",
            "mistakes": "✗ ... ✓ ...",
            "category": "Rhetoric & Emphasis",
        }
    ]

    mock_resp = {
        "evaluations": [
            {
                "pattern": "Negative Inversion",
                "passed": False,
                "defect_category": "instantiation",
                "issue": "The quote does not instantiate negative inversion.",
                "action": "Use an authentic negative inversion sentence.",
            }
        ]
    }

    monkeypatch.setattr(LLMClient, "chat", lambda self, *args, **kwargs: mock_resp)

    defects, details = audit_grammar_semantic(items, "One of the things that tend to annoy punctual people...")
    assert len(defects) == 1
    assert "[grammar-semantic] Pattern '[[Negative Inversion]]' instantiation flaw" in defects[0]
    assert len(details) == 1
    assert details[0]["pattern"] == "Negative Inversion"
    assert details[0]["category"] == "grammar-instantiation"
    assert "instantiate negative inversion" in details[0]["issue"]


def test_candidate_generation_advance_and_do_part():
    """Verify that contiguous idiom units like 'in advance' and light-verb collocations like 'do [one's] part' retain correct candidate senses."""
    from librarian.linguistics import LinguisticEngine

    # 1. advance in "Schedule meetings and appointments in advance..."
    quote_adv = "Schedule meetings and appointments in advance, but be prepared for changes."
    cands_adv = LinguisticEngine.build_top_k_vocab_candidates("advance", quote_adv, pos="noun")
    assert any(c.get("sense_idx") == 0 and "before something happens" in c.get("definition", "") for c in cands_adv)

    # 2. do [one's] part in "If we try, most of us can do our part to reduce..."
    quote_part = "If we try, most of us can do our part to reduce our carbon footprint."
    cands_part = LinguisticEngine.build_top_k_expression_candidates("do [one's] part", quote_part, expr_type="collocation")
    defs = [c.get("definition", "") for c in cands_part]
    assert any("plays a part" in d or "involved in it" in d or "responsible for something" in d for d in defs)


def test_audit_vocabulary_wsd_semantic_overruled_by_pos_gate(monkeypatch):
    """Verify that cross-POS hallucinations (e.g. noun context -> verb definition) are caught by Gate 0 and recorded as overruled without deducting score."""
    from librarian.audit_extraction import audit_vocabulary_wsd_semantic
    from librarian.linguistics import LinguisticEngine

    items = [
        {
            "lemma": "promise",
            "pos": "noun",
            "quote": "Standing for the promise that Beijing has made to the world.",
            "definition": "a statement that you will definitely do or provide something",
        }
    ]

    mock_cands = [
        {
            "definition": "a statement that you will definitely do or provide something",
            "pos": "noun",
            "score": 60.0,
            "margin": 2.0,
            "confidence": "low",
            "raw_sense": {"grammar": "[countable]"},
        },
        {
            "definition": "to tell someone that you will definitely do or provide something",
            "pos": "verb",
            "score": 62.0,
            "margin": 2.0,
            "confidence": "low",
            "raw_sense": {"grammar": "[transitive]"},
        },
    ]

    monkeypatch.setattr(LinguisticEngine, "build_top_k_vocab_candidates", lambda *args, **kwargs: mock_cands)
    # LLM hallucinates and chooses index 1 (the verb definition)
    monkeypatch.setattr(LinguisticEngine, "disambiguate_via_llm", lambda *args, **kwargs: 1)

    defects, cures, overruled = audit_vocabulary_wsd_semantic(items, "Standing for the promise that Beijing has made to the world.")
    assert len(defects) == 0, "Gate 0 must intercept cross-POS suggestion and not register as a defect"
    assert len(cures) == 0
    assert len(overruled) == 1
    assert overruled[0]["lemma"] == "promise"
    assert overruled[0]["gate"] == "Gate 0 (Part-of-Speech Invariant)"
    assert "noun" in overruled[0]["reason"].lower()
    assert "verb" in overruled[0]["reason"].lower()


def test_audit_summary_semantic_mocked(monkeypatch):
    """Verify Level-2 summary semantic audit pass and defect paths."""
    from librarian.audit_extraction import audit_summary_semantic
    from librarian.llm import LLMClient

    monkeypatch.setattr(LLMClient, "chat", lambda self, *args, **kwargs: {"passed": True, "score": 94, "defects": []})
    d_ok, det_ok = audit_summary_semantic("Accurate summary.", "Source passage text.")
    assert d_ok == [] and det_ok == []

    monkeypatch.setattr(
        LLMClient,
        "chat",
        lambda self, *args, **kwargs: {
            "passed": False,
            "score": 55,
            "defects": [
                {
                    "category": "factual_faithfulness",
                    "issue": "Claims 1995 instead of 1992.",
                    "action": "Fix year to 1992.",
                }
            ],
        },
    )
    d_bad, det_bad = audit_summary_semantic("In 1995...", "In 1992...")
    assert len(d_bad) == 1
    assert "[summary-semantic] (factual_faithfulness)" in d_bad[0]
    assert det_bad[0]["category"] == "summary-factual_faithfulness"


def test_audit_mindmap_semantic_mocked(monkeypatch):
    """Verify Level-2 mindmap semantic audit pass and defect paths."""
    from librarian.audit_extraction import audit_mindmap_semantic
    from librarian.llm import LLMClient

    tree = {"root": "Topic", "children": [{"name": "Branch A"}, {"name": "Branch B"}]}

    monkeypatch.setattr(LLMClient, "chat", lambda self, *args, **kwargs: {"passed": True, "score": 92, "defects": []})
    d_ok, det_ok = audit_mindmap_semantic(tree, "Source passage text.")
    assert d_ok == [] and det_ok == []

    monkeypatch.setattr(
        LLMClient,
        "chat",
        lambda self, *args, **kwargs: {
            "passed": False,
            "score": 60,
            "defects": [
                {
                    "category": "mece_orthogonality",
                    "node": "Branch B",
                    "issue": "Duplicates Branch A.",
                    "action": "Merge or differentiate.",
                }
            ],
        },
    )
    d_bad, det_bad = audit_mindmap_semantic(tree, "Source passage text.")
    assert len(d_bad) == 1
    assert "[mindmap-semantic] (mece_orthogonality) [Branch B]" in d_bad[0]
    assert det_bad[0]["category"] == "mindmap-mece_orthogonality"





