"""Dimension 3: Pedagogical & Slot-Filling Quality (W_PEDAGOGY = 25.0)."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from ..linguistics import LinguisticEngine
from ..processor import WikiProcessor
from .eval_helpers import (
    W_PEDAGOGY,
    _cefr_source_ceiling,
    _clean_core,
    _extract_target_skeletons,
    canonical_formula_key,
)
from .scorer_pedagogy_extraction import (
    validate_expert_audit_item,
    validate_expressions_item,
    validate_grammar_item,
    validate_mindmap_item,
    validate_summary_item,
    validate_vocabulary_item,
)
from .scorer_pedagogy_quiz import validate_quiz_item


def _score_pedagogy(
    items: List[Dict[str, Any]],
    task_type: str,
    user_prompt: str = "",
) -> Tuple[Optional[float], List[str]]:
    """Dimension 3 (0–25). Evaluates pedagogical quality across extraction and assessment types."""
    if task_type not in ("vocabulary", "expressions", "grammar", "quiz", "summary", "mindmap", "expert_audit"):
        return None, []
    if not items:
        return 0.0, [f"⚠️ Output list for '{task_type}' is empty."]

    flags: List[str] = []
    checks = passes = 0

    # Extract study list headwords and blueprint specifications from prompt if evaluating a quiz
    prompt_headwords = set()
    prompt_blueprints: Dict[str, Dict[str, Any]] = {}
    if task_type == "quiz" and user_prompt:
        grammar_ref_split = re.split(r"\bGRAMMAR REFERENCE:\b", user_prompt, flags=re.IGNORECASE)
        vocab_section = grammar_ref_split[0]
        for m in re.finditer(r"## \[\[(.*?)\]\]", vocab_section):
            prompt_headwords.add(m.group(1).strip().lower())
        if not prompt_headwords:
            for m in re.finditer(r"-\s*Target:\s*([^\n\r]+)", vocab_section):
                prompt_headwords.add(m.group(1).strip().lower())
        # Support Target Word from Authentic Corpus Blueprint or Translation Micro-Task Blueprint
        for m in re.finditer(
            r"-\s*Target\s+(?:Word|Keyword|Collocation\s*/\s*Lexical Frame):\s*([^\n\r(]+)",
            user_prompt,
            re.IGNORECASE,
        ):
            w = m.group(1).strip().lower()
            if w:
                prompt_headwords.add(w)

        # Parse full item blueprints: ### Item N ###
        blueprint_blocks = re.split(r"###\s*Item\s*\d+\s*###", user_prompt, flags=re.IGNORECASE)
        for block in blueprint_blocks:
            m_target = re.search(r"-\s*Target Word:\s*([^\n\r]+)", block, re.IGNORECASE)
            if not m_target:
                continue
            b_word = m_target.group(1).strip().lower()
            m_pos = re.search(r"-\s*Part of Speech:\s*([^\n\r]+)", block, re.IGNORECASE)
            m_infl = re.search(r"-\s*Inflectional Form:\s*([^\n\r]+)", block, re.IGNORECASE)
            m_def = re.search(r"-\s*(?:Locked|Contextual) Definition:\s*([^\n\r]+)", block, re.IGNORECASE)
            m_anc = re.search(r"-\s*Collocational Anchor:\s*([^\n\r]+)", block, re.IGNORECASE)
            m_ex = re.search(
                r"(?:Authentic Corpus Blueprint:\s*'(.+?)'\s*$"
                r"|Reference Example:\s*['\"](.+?)['\"]\s*$"
                r"|authentic_example['\"]\s*:\s*['\"]([^'\n\r]+)['\"]"
                r"|Model:\s*\"([^\n\r\"]+)\")",
                block,
                re.IGNORECASE | re.MULTILINE,
            )

            anchor_val = ""
            anchor_type = ""
            if m_anc:
                raw_anc = m_anc.group(1).strip()
                m_type = re.match(r"^(.*?)\s*\(([^)]+)\)$", raw_anc)
                if m_type:
                    anchor_val = m_type.group(1).strip().lower()
                    anchor_type = m_type.group(2).strip().lower()
                else:
                    anchor_val = raw_anc.lower()

            ex_val = ""
            if m_ex:
                ex_val = next((g for g in m_ex.groups() if g), "").strip()

            m_lq = re.search(
                r"Curriculum Quote \(licensed\):\s*'(.+?)'\s*$", block, re.IGNORECASE | re.MULTILINE
            )
            m_cq = re.search(
                r"Curriculum Quote \(display-only.*?\):\s*'(.+?)'\s*$", block, re.IGNORECASE | re.MULTILINE
            )
            licensed_quote_val = m_lq.group(1).strip() if m_lq else ""
            candidate_quote_val = m_cq.group(1).strip() if m_cq else ""

            m_opts = re.search(r"-\s*Prescribed Options:\s*\[([^\]]*)\]", block, re.IGNORECASE)
            prescribed_opts_val = [o.strip() for o in m_opts.group(1).split(",") if o.strip()] if m_opts else []
            m_pool = re.search(r"-\s*Attested Phrase Pool:\s*\[([^\]]*)\]", block, re.IGNORECASE)
            attested_pool_val = [o.strip() for o in m_pool.group(1).split(",") if o.strip()] if m_pool else []
            m_bk = re.search(r"-\s*Backup Symbolic Distractors:\s*\[([^\]]*)\]", block, re.IGNORECASE)
            backup_dist_val = [o.strip() for o in m_bk.group(1).split(",") if o.strip()] if m_bk else []
            m_pidx = re.search(r"-\s*Correct Answer Index:\s*(\d+)", block, re.IGNORECASE)
            prescribed_index_val = int(m_pidx.group(1)) if m_pidx else None

            prompt_blueprints[b_word] = {
                "target_word": b_word,
                "part_of_speech": m_pos.group(1).strip().lower() if m_pos else "",
                "inflection": m_infl.group(1).strip().lower() if m_infl else "",
                "definition": m_def.group(1).strip() if m_def else "",
                "anchor": anchor_val,
                "anchor_type": anchor_type,
                "example": ex_val,
                "licensed_quote": licensed_quote_val,
                "candidate_quote": candidate_quote_val,
                "quote_provenance": "weak" if candidate_quote_val else ("strong" if licensed_quote_val else ""),
                "prescribed_options": prescribed_opts_val,
                "attested_phrase_pool": attested_pool_val,
                "backup_distractors": backup_dist_val,
                "prescribed_index": prescribed_index_val,
            }

    grammar_skeletons: List[Dict[str, str]] = []
    skeleton_keys: Dict[str, List[Dict[str, str]]] = {}
    if task_type == "grammar" and user_prompt:
        grammar_skeletons = _extract_target_skeletons(user_prompt, "grammar")
        for _sk in grammar_skeletons:
            _raw_formula = str(_sk.get("formula", ""))
            _skel_keys = {
                canonical_formula_key(_raw_formula),
                canonical_formula_key(WikiProcessor.unwrap_literal_brackets(_raw_formula)),
            }
            for _key in _skel_keys:
                if _key:
                    skeleton_keys.setdefault(_key, []).append(_sk)
    consumed_skeletons: set = set()

    def _align_grammar_formula(pattern: str, category: str) -> Tuple[bool, str]:
        if not grammar_skeletons:
            return True, ""
        key = canonical_formula_key(pattern)
        matched = skeleton_keys.get(key, [])
        for sk in matched:
            if id(sk) not in consumed_skeletons:
                consumed_skeletons.add(id(sk))
                return True, str(sk.get("formula", ""))
        if matched:
            return True, ""
        wanted = ""
        for sk in grammar_skeletons:
            if id(sk) in consumed_skeletons:
                continue
            if str(sk.get("category", "")).strip().lower() == str(category).strip().lower():
                wanted = str(sk.get("formula", ""))
                break
        return False, wanted

    src_cefr_level, src_passage_words = _cefr_source_ceiling(user_prompt) if task_type in ("quiz", "grammar") else (None, set())
    src_cefr_rank = LinguisticEngine.CEFR_ORDER.get(src_cefr_level or "", 6)

    for item in items:
        checks += 1
        item_passed = False
        item_flags: List[str] = []

        if task_type == "vocabulary":
            item_passed, item_flags = validate_vocabulary_item(item)
        elif task_type == "expressions":
            item_passed, item_flags = validate_expressions_item(item)
        elif task_type == "grammar":
            item_passed, item_flags = validate_grammar_item(
                item,
                _align_grammar_formula,
                src_cefr_level=src_cefr_level,
                src_passage_words=src_passage_words,
            )
        elif task_type == "quiz":
            item_passed, item_flags = validate_quiz_item(
                item,
                prompt_headwords=prompt_headwords,
                prompt_blueprints=prompt_blueprints,
                src_cefr_level=src_cefr_level,
                src_cefr_rank=src_cefr_rank,
                src_passage_words=src_passage_words,
            )
        elif task_type == "summary":
            item_passed, item_flags = validate_summary_item(item)
        elif task_type == "mindmap":
            item_passed, item_flags = validate_mindmap_item(item)
        elif task_type == "expert_audit":
            item_passed, item_flags = validate_expert_audit_item(item)

        if item_flags:
            flags.extend(item_flags)
        if item_passed:
            passes += 1

    if checks == 0:
        return 0.0, [f"⚠️ Output list for '{task_type}' is empty."]

    # Check for copy-pasted/homogenized definitions in vocabulary
    if task_type in ("vocabulary", "expressions"):
        definitions = [_clean_core(item.get("definition", "")) for item in items if item.get("definition")]
        if len(definitions) > 1:
            dup_defs = len(definitions) - len(set(definitions))
            if dup_defs > 0:
                flags.append(f"❌ Found {dup_defs} duplicate or copy-pasted definition(s) across different terms")
                passes = max(0, passes - dup_defs)

    # Enforce 100% blueprint target coverage for vocabulary/cloze quizzes
    if task_type == "quiz" and prompt_blueprints and len(items) < len(prompt_blueprints):
        covered_targets = {
            str(it.get("target_word") or it.get("word") or "").strip().lower()
            for it in items
            if isinstance(it, dict)
        }
        missing_targets = [w for w in prompt_blueprints.keys() if w not in covered_targets]
        if missing_targets:
            flags.append(
                f"❌ Incomplete target coverage: generated {len(items)}/{len(prompt_blueprints)} items; "
                f"missing required blueprint targets: {missing_targets}"
            )
            passes = max(0, passes - len(missing_targets))

    # Enforce 100% question blueprint count for listening/reading/video quizzes with numbered Question Blueprints
    expected_q_blueprints = len(
        re.findall(r"#{3,4}\s*Question\s+\d+\s+Blueprint\s*#{3,4}", user_prompt or "", re.IGNORECASE)
    ) if task_type == "quiz" and not prompt_blueprints else 0
    if expected_q_blueprints > 0 and len(items) < expected_q_blueprints:
        missing_q_count = expected_q_blueprints - len(items)
        flags.append(
            f"❌ Incomplete question coverage: generated {len(items)}/{expected_q_blueprints} required blueprint questions "
            f"(missing {missing_q_count} question(s))"
        )
        passes = max(0, passes - missing_q_count)

    denom_count = max(
        checks,
        len(prompt_blueprints) if (task_type == "quiz" and prompt_blueprints) else expected_q_blueprints,
    )
    raw_score = round((passes / max(1, denom_count)) * W_PEDAGOGY, 1)
    # Apply soft deduction for auto-remapped grammar categories (2.0 pts per remapped item)
    if task_type == "grammar":
        remap_count = sum(1 for f in flags if "Deterministic Auto-Remap" in f)
        if remap_count > 0:
            raw_score = max(0.0, raw_score - (remap_count * 2.0))

    return max(0.0, raw_score), flags
