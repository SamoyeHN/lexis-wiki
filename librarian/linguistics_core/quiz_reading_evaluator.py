"""
Level 1 deterministic code gate and psychometric evaluation for reading comprehension items.
"""

import logging
import math
import re
from typing import Any, Dict, List, Set, Tuple

from .quiz_reading_models import (
    CEFRDifficultyTier,
    ReadingEvaluationResult,
    ReadingItemBlueprint,
)

logger = logging.getLogger("librarian.linguistics")


class QuizReadingEvaluatorMixin:
    """
    Mixin providing Level 1 deterministic code gate and lexical leakage evaluation.
    """

    @classmethod
    def evaluate_reading_item(
        cls,
        item: Dict[str, Any],
        blueprint: ReadingItemBlueprint,
    ) -> ReadingEvaluationResult:
        """
        Level 1 Deterministic Code Gate for Reading Items:
        1. Cardinality: Exactly 4 options.
        2. No Empty Options or Duplicates.
        3. Lexical Leakage Gate: Checks verbatim 3-gram overlap between correct option and anchor text (< 35%).
        4. Option Length Parity Gate: Relative Std Dev <= 25%.
        5. Lexical Ceiling Gate: In Foundational tier (A1/A2), options must not introduce out-of-passage B2+ words.
        """
        options = item.get("options", [])
        correct_idx = item.get("correct_answer_index", 0)
        flags: List[str] = []
        score = 100.0

        # 1. Cardinality
        if len(options) != 4:
            flags.append(f"Cardinality Gate Violation: Expected 4 options, found {len(options)}")
            score -= 40.0

        # 2. Distinctness
        clean_opts = [str(o).strip() for o in options]
        if len(set(clean_opts)) != len(clean_opts):
            flags.append("Option Distinctness Gate Violation: Duplicate choices detected.")
            score -= 30.0

        # 3. Lexical Leakage (Verbatim 3-gram check)
        leakage_ratio = 0.0
        if 0 <= correct_idx < len(clean_opts):
            key_text = clean_opts[correct_idx]
            anchor_text = blueprint.anchor_sentence_text

            def get_3grams(s: str) -> Set[Tuple[str, str, str]]:
                words = re.findall(r"\b[a-zA-Z]{2,}\b", s.lower())
                return set(zip(words, words[1:], words[2:]))

            key_3grams = get_3grams(key_text)
            anchor_3grams = get_3grams(anchor_text)

            if key_3grams and anchor_3grams:
                shared = key_3grams.intersection(anchor_3grams)
                leakage_ratio = len(shared) / len(key_3grams)

            if leakage_ratio > 0.35:
                flags.append(
                    f"Lexical Leakage Gate Violation ({leakage_ratio:.1%} shared 3-grams > 35%): "
                    f"correct answer copied verbatim phrasing from anchor sentence [{blueprint.anchor_sentence_id}]."
                )
                score -= 25.0

        # 4. Option Length Parity Gate
        lengths = [len(o.split()) for o in clean_opts] if clean_opts else [0]
        mean_len = sum(lengths) / len(lengths) if lengths else 1.0
        variance = sum((l - mean_len) ** 2 for l in lengths) / len(lengths) if lengths else 0.0
        std_dev = math.sqrt(variance)
        rel_std_dev = std_dev / mean_len if mean_len > 0 else 0.0

        if rel_std_dev > 0.25:
            flags.append(
                f"Length Parity Gate Warning: Relative StdDev is {rel_std_dev:.1%} (max allowed: 25.0%). Options: {lengths}"
            )
            score -= 10.0

        # 5. Lexical Ceiling Gate (Foundational tier)
        if blueprint.tier == CEFRDifficultyTier.FOUNDATIONAL and blueprint.passage_words:
            ADVANCED_CEFR_TIERS = {"B2", "C1", "C2"}

            def get_word_cefr_level(w: str) -> str:
                try:
                    from librarian.linguistics import LinguisticEngine
                    return str(LinguisticEngine.get_word_cefr(w) or "A1").upper()
                except Exception:
                    pass
                try:
                    from cefrpy import CEFRAnalyzer
                    analyzer = CEFRAnalyzer()
                    res = analyzer.get_word_cefr(w)
                    if res and isinstance(res, tuple) and len(res) > 0:
                        return str(res[0]).upper()
                except Exception:
                    pass
                return "A1"

            b2_violations = []
            for opt in clean_opts:
                opt_tokens = re.findall(r"\b[a-zA-Z]{4,}\b", opt.lower())
                for tok in opt_tokens:
                    if tok not in blueprint.passage_words:
                        lvl = get_word_cefr_level(tok)
                        if lvl in ADVANCED_CEFR_TIERS:
                            b2_violations.append(f"'{tok}' ({lvl})")

            if b2_violations:
                flags.append(
                    f"Lexical Ceiling Warning (Foundational Tier): Options introduced advanced out-of-passage words: {', '.join(set(b2_violations[:4]))}"
                )
                score -= 10.0

        is_valid = (score >= 80.0)
        return ReadingEvaluationResult(
            is_valid=is_valid,
            score=max(0.0, score),
            flags=flags,
            lexical_overlap_ratio=leakage_ratio,
            length_std_dev_ratio=rel_std_dev,
        )

    @classmethod
    def evaluate_reading_quiz(
        cls,
        questions: List[Dict[str, Any]],
        blueprints: List[ReadingItemBlueprint],
    ) -> Dict[str, Any]:
        """
        Evaluates an entire generated reading quiz against its blueprints.
        """
        results = []
        all_valid = True
        total_score = 0.0

        for idx, q in enumerate(questions):
            bp = blueprints[idx] if idx < len(blueprints) else blueprints[-1]
            res = cls.evaluate_reading_item(q, bp)
            if not res.is_valid:
                all_valid = False
            total_score += res.score
            results.append({
                "item_index": idx + 1,
                "skill": bp.skill_type.value,
                "is_valid": res.is_valid,
                "score": res.score,
                "flags": res.flags,
                "lexical_overlap": res.lexical_overlap_ratio,
                "length_std_dev": res.length_std_dev_ratio,
            })

        avg_score = total_score / len(questions) if questions else 0.0
        return {
            "all_valid": all_valid,
            "average_score": avg_score,
            "item_results": results,
        }


__all__ = [
    "QuizReadingEvaluatorMixin",
]
