"""Dimension 4: Uniqueness & Deduplication (W_UNIQUENESS = 20.0)."""

from __future__ import annotations

from collections import Counter
import re
from typing import Any, Dict, List, Optional, Tuple

from ..linguistics import LinguisticEngine
from .eval_helpers import W_UNIQUENESS


def _score_uniqueness(
    items: List[Dict[str, Any]],
    task_type: str,
    user_prompt: str = "",
) -> Tuple[Optional[float], List[str]]:
    """Dimension 4 (0–20). Deduplicate by the task's primary identifying field."""
    if not items:
        return 0.0, ["❌ No items to evaluate for uniqueness"]
    key_by_type = {
        "vocabulary": ("word",),
        "expressions": ("word",),
        "grammar": ("quote", "pattern_formula"),
        "quiz": ("target_word", "question", "translated_sentence", "correct_english_answer"),
        "summary": ("concept_name",),
        "mindmap": ("branch_name",),
        "expert_audit": ("item_index",),
    }
    keys = key_by_type.get(task_type, ("word", "quote", "question"))
    headwords = []
    for item in items:
        if task_type == "grammar":
            # Composite identity for grammar: quote is primary, formula is secondary
            q_val = str(item.get("quote") or "").strip().lower()
            f_val = str(item.get("pattern_formula") or "").strip().lower()
            if q_val:
                headwords.append(q_val)
            elif f_val:
                headwords.append(f_val)
            continue
        for key in keys:
            value = item.get(key)
            if value:
                headwords.append(str(value).strip().lower())
                break
    if not headwords:
        return 0.0, ["⚠️ Missing identifying keys for uniqueness check"]

    unique_flags = []
    unique = len(set(headwords))
    earned = W_UNIQUENESS
    if unique < len(headwords):
        dup = len(headwords) - unique
        # Stricter deduction: duplicate items heavily reduce score
        ratio = unique / len(headwords)
        earned = 0.0 if ratio < 0.8 else round(ratio * W_UNIQUENESS, 1)
        unique_flags.append(f"❌ Found {dup} duplicate item(s)")

    # For vocabulary: audit morphological word family collisions (e.g. 'recognize' vs 'recognition')
    if task_type == "vocabulary" and headwords:
        family_clusters: List[List[str]] = []
        for hw in headwords:
            clean_hw = re.sub(r'\[.*?\]|\(.*?\)', '', str(hw)).strip().lower()
            if not clean_hw:
                continue
            placed = False
            for fc in family_clusters:
                if any(LinguisticEngine.are_same_word_family(clean_hw, fchw) for fchw in fc):
                    fc.append(clean_hw)
                    placed = True
                    break
            if not placed:
                family_clusters.append([clean_hw])

        mult_clusters = [fc for fc in family_clusters if len(fc) > 1]
        if mult_clusters:
            coll_details = ", ".join(f"[{'/'.join(c)}]" for c in mult_clusters[:3])
            unique_flags.append(f"❌ Morphological word-family collisions detected: {coll_details}")
            # Penalize uniqueness dimension
            earned = max(0.0, earned - len(mult_clusters) * 3.0)

    # For grammar: audit duplicate common_mistakes across patterns
    if task_type == "grammar":
        all_mistakes = [
            str(it.get("common_mistakes") or "").strip()
            for it in items
            if str(it.get("common_mistakes") or "").strip()
        ]
        if len(all_mistakes) > len(set(all_mistakes)):
            m_counts = Counter(all_mistakes)
            dup_mistakes = [m for m, c in m_counts.items() if c > 1]
            if dup_mistakes:
                short_dup = [f"'{m[:40]}...' ({m_counts[m]}x)" for m in dup_mistakes[:2]]
                unique_flags.append(f"❌ Found duplicate common_mistakes across grammar patterns: {', '.join(short_dup)}")
                earned = max(0.0, earned - len(dup_mistakes) * 3.0)

    # For quizzes: penalize identical distractors recycled repeatedly across different questions
    # (Skip when options are blueprint-prescribed so deterministic distractors are not re-audited post-LLM)
    if task_type == "quiz" and "Prescribed Options:" not in (user_prompt or ""):
        all_distractors = []
        for item in items:
            opts = item.get("options", [])
            idx = item.get("correct_answer_index")
            if isinstance(opts, list) and isinstance(idx, int):
                for o_i, o in enumerate(opts):
                    if o_i != idx:
                        all_distractors.append(str(o).strip().lower())
        if all_distractors:
            counts = Counter(all_distractors)
            repeated = {w: c for w, c in counts.items() if c >= 3 and len(w) >= 3}
            if repeated:
                rep_details = ", ".join(f"'{w}' ({c}x)" for w, c in list(repeated.items())[:3])
                unique_flags.append(f"❌ Distractors recycled repeatedly across quiz: {rep_details}")
    return earned, unique_flags
