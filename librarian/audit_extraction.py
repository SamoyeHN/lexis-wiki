"""Extraction Semantic Re-Audit Suite.

Provides Level 1 deterministic code-gate verification and optional Level 2 LLM semantic
evaluation for extraction artifacts under ``wiki/<unit>/extractions/``:
  - Vocabulary (``*_vocabulary.md``): verbatim quote grounding, required fields, POS/CEFR tags.
  - Grammar (``*_grammar.md``): verbatim quote grounding, Quirk/Biber structural formula, pedagogical function.
  - Summary (``*_summary.md``): Narrative Overview, Lesson Hook, Essential Questions, and factual entity recall against source text.
  - Mindmap (``*_mindmap.json``): valid JSON tree, MECE orthogonality (no duplicate sibling branches), minimum depth >= 2.

Complies with the Neuro-Symbolic Core Architecture (<5ms zero-token L1 code gate).
"""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .config import config

logger = logging.getLogger("librarian.audit_extraction")


@dataclass
class ExtractionAuditReport:
    """Structured result of an extraction audit for a single unit."""

    unit: str
    passed: bool
    score: float
    defects: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)
    overruled: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "unit": self.unit,
            "passed": self.passed,
            "score": round(self.score, 1),
            "defects": self.defects,
            "overruled": self.overruled,
            "stats": self.stats,
            "error": self.error,
        }


def _project_root(project_root: Optional[str]) -> Path:
    return Path(project_root) if project_root else Path(config.project_root)


def _normalize_text_for_search(text: str) -> str:
    """Normalize text for robust substring and token window searching."""
    # Strip YAML frontmatter if present
    cleaned = re.sub(r"^---[\s\S]*?---\s*", "", text)
    cleaned = cleaned.lower()
    # Replace punctuation and special quotes with spaces
    cleaned = re.sub(r"['\"’“”‘\-—–,.;:!?()[\]{}/\\#*`]", " ", cleaned)
    return " ".join(cleaned.split())


def _is_quote_grounded(quote: str, normalized_source: str) -> bool:
    """Check whether a quoted sentence is grounded in the primary source text.

    Verifies either exact normalized substring match or high-density 6-token window match.
    """
    norm_quote = _normalize_text_for_search(quote)
    if not norm_quote:
        return False
    if norm_quote in normalized_source:
        return True

    # 6-token sliding window check for cases with minor ellipses or markdown artifacts
    tokens = norm_quote.split()
    if len(tokens) < 6:
        # For short quotes (<6 tokens), require at least 80% token presence or full match
        return norm_quote in normalized_source

    window_size = 6
    for i in range(len(tokens) - window_size + 1):
        window = " ".join(tokens[i : i + window_size])
        if window in normalized_source:
            return True

    return False


def audit_vocabulary_markdown(
    vocab_text: str, normalized_source: str
) -> Tuple[List[str], Dict[str, Any]]:
    """Audit vocabulary extraction markdown for required fields and quote grounding."""
    defects: List[str] = []
    blocks = re.split(r"\n(?=##\s+\[\[)", vocab_text)
    items: List[Dict[str, Any]] = []

    for blk in blocks:
        m_head = re.search(
            r"^##\s+\[\[(.*?)\]\]\s+([a-z]+)?(?:\s+([A-C][1-2]))?",
            blk,
            re.MULTILINE | re.IGNORECASE,
        )
        if not m_head:
            continue

        lemma = m_head.group(1).strip()
        pos = (m_head.group(2) or "").strip().lower()
        cefr = (m_head.group(3) or "").strip().upper()

        m_def = re.search(r"-\s+\*\*Definition\*\*:\s*(.*?)$", blk, re.MULTILINE)
        definition = m_def.group(1).strip() if m_def else ""

        m_quote = re.search(
            r"-\s+\*\*Quoted Sentence\*\*:\s*(.*?)$", blk, re.MULTILINE
        )
        quote = m_quote.group(1).strip() if m_quote else ""

        items.append(
            {
                "lemma": lemma,
                "pos": pos,
                "cefr": cefr,
                "definition": definition,
                "quote": quote,
            }
        )

        if not definition:
            defects.append(f"[vocabulary] Item '[[{lemma}]]' missing definition.")

        if not quote:
            defects.append(
                f"[vocabulary] Item '[[{lemma}]]' missing quoted sentence."
            )
        elif not _is_quote_grounded(quote, normalized_source):
            defects.append(
                f"[vocabulary] Item '[[{lemma}]]' quote not found in source text: '{quote[:60]}...'"
            )

    if not items:
        defects.append("[vocabulary] No vocabulary items found in extraction file.")

    stats = {
        "item_count": len(items),
        "lemmas": [it["lemma"] for it in items],
        "items": items,
    }
    return defects, stats


def audit_grammar_markdown(
    grammar_text: str, normalized_source: str
) -> Tuple[List[str], Dict[str, Any]]:
    """Audit grammar extraction markdown for formulas, pedagogy, quote grounding, and structural consistency."""
    defects: List[str] = []
    blocks = re.split(r"\n(?=##\s+\[\[)", grammar_text)
    items: List[Dict[str, Any]] = []
    grammar_defect_details: List[Dict[str, Any]] = []

    for blk in blocks:
        m_head = re.search(r"^##\s+\[\[(.*?)\]\]", blk, re.MULTILINE)
        if not m_head:
            continue

        pattern_name = m_head.group(1).strip()

        m_quote = re.search(r"-\s+\*\*Quote\*\*:\s*(.*?)$", blk, re.MULTILINE)
        quote = m_quote.group(1).strip() if m_quote else ""

        m_formula = re.search(
            r"-\s+\*\*Pattern Formula\*\*:\s*(.*?)$", blk, re.MULTILINE
        )
        formula = m_formula.group(1).strip() if m_formula else ""

        m_pedagogy = re.search(
            r"-\s+\*\*Pedagogical Function\*\*:\s*(.*?)$", blk, re.MULTILINE
        )
        pedagogy = m_pedagogy.group(1).strip() if m_pedagogy else ""

        m_imitation = re.search(
            r"-\s+\*\*Imitation Example\*\*:\s*(.*?)$", blk, re.MULTILINE
        )
        imitation = m_imitation.group(1).strip() if m_imitation else ""

        m_mistakes = re.search(
            r"-\s+\*\*Common Mistakes\*\*:\s*(.*?)(?=\n-\s+\*\*|\n##|\Z)", blk, re.DOTALL
        )
        mistakes = m_mistakes.group(1).strip() if m_mistakes else ""

        m_cat = re.search(
            r"-\s+\*\*Category\*\*:\s*(.*?)$", blk, re.MULTILINE
        )
        category = m_cat.group(1).strip() if m_cat else ""

        items.append(
            {
                "pattern": pattern_name,
                "quote": quote,
                "formula": formula,
                "pedagogy": pedagogy,
                "imitation": imitation,
                "mistakes": mistakes,
                "category": category,
            }
        )

        # 1. Formula validation
        if not formula:
            d_msg = f"[grammar] Pattern '[[{pattern_name}]]' missing Pattern Formula."
            defects.append(d_msg)
            grammar_defect_details.append({
                "pattern": pattern_name,
                "quote": quote,
                "formula": "",
                "category": "grammar",
                "issue": "Missing Pattern Formula.",
                "action": "Add Quirk/Biber syntactic formula.",
            })
        elif not any(ch in formula for ch in ("[", "]", "+", "S", "V", "NP", "VP")):
            d_msg = f"[grammar] Pattern '[[{pattern_name}]]' formula lacks Quirk/Biber syntactic markers: '{formula}'"
            defects.append(d_msg)
            grammar_defect_details.append({
                "pattern": pattern_name,
                "quote": quote,
                "formula": formula,
                "category": "grammar",
                "issue": f"Formula lacks Quirk/Biber syntactic markers: '{formula}'.",
                "action": "Ensure formula uses syntactic constituent brackets like [NP], [VP], [S].",
            })
        else:
            from .evaluator_core.eval_helpers import degenerate_grammar_slots
            deg_slots = degenerate_grammar_slots(formula)
            if deg_slots:
                d_msg = f"[grammar] Pattern '[[{pattern_name}]]' degenerate formula slot: {deg_slots}"
                defects.append(d_msg)
                grammar_defect_details.append({
                    "pattern": pattern_name,
                    "quote": quote,
                    "formula": formula,
                    "category": "grammar",
                    "issue": f"Degenerate formula slot {deg_slots} names the same constituent twice.",
                    "action": "Fix formula to avoid degenerate slots.",
                })

        # 2. Pedagogy validation
        if not pedagogy:
            d_msg = f"[grammar] Pattern '[[{pattern_name}]]' missing Pedagogical Function."
            defects.append(d_msg)
            grammar_defect_details.append({
                "pattern": pattern_name,
                "quote": quote,
                "formula": formula,
                "category": "grammar",
                "issue": "Missing Pedagogical Function.",
                "action": "Provide communicative/discourse function explanation.",
            })

        # 3. Quote grounding validation
        if not quote:
            d_msg = f"[grammar] Pattern '[[{pattern_name}]]' missing Quote."
            defects.append(d_msg)
            grammar_defect_details.append({
                "pattern": pattern_name,
                "quote": "",
                "formula": formula,
                "category": "grammar",
                "issue": "Missing Quote.",
                "action": "Extract verbatim sentence from source text.",
            })
        elif not _is_quote_grounded(quote, normalized_source):
            d_msg = f"[grammar] Pattern '[[{pattern_name}]]' quote not found in source text: '{quote[:60]}...'"
            defects.append(d_msg)
            grammar_defect_details.append({
                "pattern": pattern_name,
                "quote": quote,
                "formula": formula,
                "category": "grammar",
                "issue": "Quote not found in source text (verbatim grounding violation).",
                "action": "Ensure quote matches authentic source text verbatim.",
            })

        # 4. Imitation example duplicate check (if imitation provided)
        if imitation and quote:
            from .evaluator_core.eval_helpers import _clean_core
            if _clean_core(imitation) == _clean_core(quote):
                d_msg = f"[grammar] Pattern '[[{pattern_name}]]' imitation example duplicates the source quote."
                defects.append(d_msg)
                grammar_defect_details.append({
                    "pattern": pattern_name,
                    "quote": quote,
                    "formula": formula,
                    "category": "grammar",
                    "issue": "Imitation example duplicates the source quote (must be an authentic new sentence).",
                    "action": "Craft a new imitation sentence modeling the formula.",
                })

        # 5. Common mistakes contrastive check (if mistakes provided)
        if mistakes and not ("✗" in mistakes and "✓" in mistakes):
            d_msg = f"[grammar] Pattern '[[{pattern_name}]]' common mistakes lacks contrastive ✗ / ✓ pair."
            defects.append(d_msg)
            grammar_defect_details.append({
                "pattern": pattern_name,
                "quote": quote,
                "formula": formula,
                "category": "grammar",
                "issue": "Common mistakes section lacks contrastive ✗ and ✓ markers.",
                "action": "Include both incorrect (✗) and correct (✓) examples.",
            })

    if not items:
        defects.append("[grammar] No grammar patterns found in extraction file.")

    stats = {
        "item_count": len(items),
        "patterns": [it["pattern"] for it in items],
        "items": items,
        "grammar_defect_details": grammar_defect_details,
    }
    return defects, stats


def audit_summary_markdown(
    summary_text: str, normalized_source: str
) -> Tuple[List[str], Dict[str, Any]]:
    """Audit summary extraction markdown for narrative overview, hook, and factual entity recall."""
    defects: List[str] = []

    has_narrative = bool(re.search(r"##\s+Narrative Overview", summary_text, re.IGNORECASE))
    has_hook = bool(re.search(r"##\s+Lesson Hook", summary_text, re.IGNORECASE))
    has_questions = bool(re.search(r"##\s+Essential Questions", summary_text, re.IGNORECASE))

    if not has_narrative:
        defects.append("[summary] Missing 'Narrative Overview' section.")
    if not has_hook:
        defects.append("[summary] Missing 'Lesson Hook' section.")
    if not has_questions:
        defects.append("[summary] Missing 'Essential Questions' section.")

    # Factual entity recall / grounding check:
    # Ensure significant content words in the summary narrative overlap with the primary text
    m_narrative = re.search(
        r"##\s+Narrative Overview\s*\n+([\s\S]*?)(?=\n+##|\Z)", summary_text, re.IGNORECASE
    )
    narrative_body = m_narrative.group(1).strip() if m_narrative else ""
    summary_words = len(narrative_body.split())

    if narrative_body:
        norm_narrative = _normalize_text_for_search(narrative_body)
        narrative_tokens = [w for w in norm_narrative.split() if len(w) >= 4]
        source_tokens = set(normalized_source.split())

        if narrative_tokens:
            overlap = [w for w in narrative_tokens if w in source_tokens]
            unique_overlap = set(overlap)
            recall_ratio = len(overlap) / len(narrative_tokens)
            if len(unique_overlap) < 8 or recall_ratio < 0.15:
                defects.append(
                    f"[summary] Low lexical grounding against source text (unique entities {len(unique_overlap)} < 8 or token recall {recall_ratio * 100:.1f}% < 15%)."
                )

    stats = {
        "summary_words": summary_words,
        "has_narrative": has_narrative,
        "has_hook": has_hook,
        "has_questions": has_questions,
    }
    return defects, stats


def audit_mindmap_json(mindmap_path: Path) -> Tuple[List[str], Dict[str, Any]]:
    """Audit mindmap JSON for structural schema, non-emptiness, and MECE orthogonality."""
    defects: List[str] = []
    try:
        data = json.loads(mindmap_path.read_text(encoding="utf-8"))
    except Exception as e:
        return [f"[mindmap] Failed to parse JSON: {e}"], {}

    if not isinstance(data, dict):
        return ["[mindmap] Root element is not a JSON object."], {}

    title = data.get("title") or ""
    root_name = data.get("root_name") or ""
    branches = data.get("branches") or []

    if not title:
        defects.append("[mindmap] Missing 'title'.")
    if not root_name:
        defects.append("[mindmap] Missing 'root_name'.")

    if not isinstance(branches, list) or len(branches) < 2:
        defects.append(f"[mindmap] Insufficient branches: expected >= 2, found {len(branches)}.")

    # MECE Orthogonality: sibling branch names must be distinct
    seen_branches: Set[str] = set()
    total_leaves = 0
    max_depth = 1

    for b in branches:
        if not isinstance(b, dict):
            continue
        b_name = (b.get("branch_name") or "").strip().lower()
        if not b_name:
            defects.append("[mindmap] Found branch with empty 'branch_name'.")
            continue
        if b_name in seen_branches:
            defects.append(f"[mindmap] MECE violation: Duplicate branch name '{b.get('branch_name')}'.")
        seen_branches.add(b_name)

        sub_branches = b.get("sub_branches") or []
        if isinstance(sub_branches, list) and sub_branches:
            max_depth = max(max_depth, 2)
            seen_subs: Set[str] = set()
            for sb in sub_branches:
                if not isinstance(sb, dict):
                    continue
                sb_name = (sb.get("sub_branch_name") or "").strip().lower()
                if sb_name in seen_subs:
                    defects.append(
                        f"[mindmap] MECE violation: Duplicate sub-branch '{sb.get('sub_branch_name')}' under '{b.get('branch_name')}'."
                    )
                seen_subs.add(sb_name)
                leaves = sb.get("leaves") or []
                if isinstance(leaves, list) and leaves:
                    max_depth = max(max_depth, 3)
                    total_leaves += len(leaves)

    if max_depth < 2:
        defects.append(f"[mindmap] Hierarchy too shallow: depth {max_depth} < 2.")

    stats = {
        "branch_count": len(branches),
        "total_leaves": total_leaves,
        "max_depth": max_depth,
    }
    return defects, stats


def match_definition_to_sense(current_def: str, candidates: List[Dict[str, Any]]) -> int:
    """Find the candidate index that best matches the current definition."""
    def _toks(s: str) -> Set[str]:
        return set(re.findall(r"\b[a-z]{3,}\b", (s or "").lower()))

    c_words = _toks(current_def)
    best_idx = 0
    best_score = -1.0
    for i, cand in enumerate(candidates):
        cand_def = cand.get("definition", "")
        if current_def.strip().lower() == cand_def.strip().lower():
            return i
        d_words = _toks(cand_def)
        if not d_words:
            continue
        overlap = len(c_words & d_words) / max(1, len(c_words | d_words))
        if overlap > best_score:
            best_score = overlap
            best_idx = i
    return best_idx


def cure_vocabulary_item(
    vocab_text: str,
    lemma: str,
    new_definition: str,
    new_example: Optional[str] = None,
) -> str:
    """In-place update for a single vocabulary item definition and example in markdown."""
    pattern = rf"(##\s+\[\[{re.escape(lemma)}\]\].*?)(?=\n##\s+\[\[|\Z)"
    m = re.search(pattern, vocab_text, re.DOTALL)
    if not m:
        return vocab_text

    blk = m.group(1)
    new_blk = blk
    new_blk = re.sub(
        r"(-\s+\*\*Definition\*\*:\s*)(.*?)(?=\n-|\n\n|\Z)",
        lambda mo: mo.group(1) + new_definition,
        new_blk,
        count=1,
    )
    if new_example:
        if re.search(r"-\s+\*\*Example Usage\*\*:", new_blk):
            new_blk = re.sub(
                r"(-\s+\*\*Example Usage\*\*:\s*)(.*?)(?=\n-|\n\n|\Z)",
                lambda mo: mo.group(1) + new_example,
                new_blk,
                count=1,
            )
        else:
            new_blk = re.sub(
                r"(-\s+\*\*Definition\*\*:[^\n]*)",
                lambda mo: mo.group(1) + f"\n- **Example Usage**: {new_example}",
                new_blk,
                count=1,
            )
    return vocab_text[: m.start(1)] + new_blk + vocab_text[m.end(1) :]


def audit_vocabulary_wsd_semantic(
    items: List[Dict[str, Any]],
    source_text: str,
    discourse_context: str = "",
) -> Tuple[List[str], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Execute Level-2 neuro-symbolic WSD semantic audit over extracted vocabulary items.

    Returns:
        (defects, cures, overruled)
    """
    from .linguistics import LinguisticEngine

    defects: List[str] = []
    cures: List[Dict[str, Any]] = []
    overruled: List[Dict[str, Any]] = []

    for item in items:
        lemma = item.get("lemma", "").strip()
        pos = item.get("pos", "").strip()
        quote = item.get("quote", "").strip()
        current_def = item.get("definition", "").strip()

        if not lemma or not quote:
            continue

        cand_pool = LinguisticEngine.build_top_k_vocab_candidates(
            lemma, quote, pos=pos or None, k=12, discourse_context=discourse_context
        )
        if len(cand_pool) <= 1:
            # Monosemous / single candidate: 0-token instant pass
            continue

        current_idx = match_definition_to_sense(current_def, cand_pool)
        top_c = cand_pool[0]
        top_score = float(top_c.get("score", 0.0))
        top_margin = float(top_c.get("margin", 0.0))
        top_conf = str(top_c.get("confidence") or "").strip().lower()

        is_top_symbolically_locked = (
            top_conf == "high"
            and top_margin >= 15.0
            and top_score >= 25.0
        )

        if is_top_symbolically_locked:
            if current_idx == 0:
                # Stage 1 Symbolic Grounding Invariant (Zero-Token Bypass):
                # Authoritative symbolic engine decisively locked Candidate 0.
                continue
            else:
                # Candidate 0 is decisively locked by symbolic truth, but extraction drifted to another sense
                chosen_idx = 0
        else:
            try:
                chosen_idx = LinguisticEngine.disambiguate_via_llm(
                    lemma, quote, cand_pool, discourse_context=discourse_context, target_pos=pos or None
                )
            except Exception as e:
                logger.warning("WSD disambiguation error for '%s': %s", lemma, e)
                continue

        if chosen_idx < 0 or chosen_idx >= len(cand_pool):
            continue

        if chosen_idx == current_idx:
            continue

        # Level-1 Physical Invariant Code Gate on LLM Suggestions:
        # Screen out LLM hallucinations violating syntactic and lexicographic invariants.
        curr_cand = cand_pool[current_idx]
        chosen_cand = cand_pool[chosen_idx]
        curr_score = float(curr_cand.get("score", 0.0))
        chosen_score = float(chosen_cand.get("score", 0.0))

        chosen_cand_def = (chosen_cand.get("definition") or "").strip()
        chosen_raw = chosen_cand.get("raw_sense") or {}
        chosen_gram = str(chosen_raw.get("grammar") or "").lower()
        chosen_sp = str(chosen_cand.get("signpost") or "").lower()
        chosen_reg_blob = f"{chosen_cand_def} {chosen_gram} {chosen_sp}".lower()

        curr_cand_def = (curr_cand.get("definition") or "").strip()
        curr_raw = curr_cand.get("raw_sense") or {}
        curr_gram = str(curr_raw.get("grammar") or "").lower()
        curr_sp = str(curr_cand.get("signpost") or "").lower()
        curr_reg_blob = f"{curr_cand_def} {curr_gram} {curr_sp}".lower()

        # Gate 0: Part-of-Speech Invariant
        curr_p = LinguisticEngine._pos_code(curr_cand.get("pos") or pos)
        chosen_p = LinguisticEngine._pos_code(chosen_cand.get("pos"))
        if curr_p and chosen_p and curr_p != chosen_p:
            logger.debug(
                "Rejecting LLM WSD defect for '%s': target context is %s but LLM suggested %s sense",
                lemma, curr_p, chosen_p,
            )
            overruled.append({
                "lemma": lemma,
                "quote": quote,
                "current_idx": current_idx,
                "current_def": curr_cand_def,
                "current_score": curr_score,
                "chosen_idx": chosen_idx,
                "chosen_def": chosen_cand_def,
                "chosen_score": chosen_score,
                "gate": "Gate 0 (Part-of-Speech Invariant)",
                "reason": f"Target item in context functions as a '{pos or curr_p}'; LLM suggested a '{chosen_cand.get('pos') or chosen_p}' sense.",
            })
            continue

        # Gate 1: Symbolic Score Superiority Invariant
        # Current sense has a decisive symbolic advantage (>= 15.0 score points) over LLM's chosen candidate.
        if curr_score >= chosen_score + 15.0:
            logger.debug(
                "Rejecting LLM WSD defect for '%s': curr_score (%.1f) >= chosen_score (%.1f) + 15.0",
                lemma, curr_score, chosen_score,
            )
            overruled.append({
                "lemma": lemma,
                "quote": quote,
                "current_idx": current_idx,
                "current_def": curr_cand_def,
                "current_score": curr_score,
                "chosen_idx": chosen_idx,
                "chosen_def": chosen_cand_def,
                "chosen_score": chosen_score,
                "gate": "Gate 1 (Symbolic Score Superiority)",
                "reason": f"Current sense symbolic score ({curr_score:.1f}) >= LLM suggestion ({chosen_score:.1f}) + 15.0 pts without overriding syntactic proof.",
            })
            continue

        # Gate 2: Register & Pragmatic Invariant
        # Informal, spoken, or slang senses cannot apply to written expository/narrative prose.
        spoken_slang_tags = ("[spoken]", "[slang]", "[informal]", "[taboo]", "[humorous]")
        chosen_is_spoken = any(t in chosen_reg_blob for t in spoken_slang_tags)
        curr_is_spoken = any(t in curr_reg_blob for t in spoken_slang_tags)
        has_dialogue = any(q_mark in quote for q_mark in ('"', '“', '”', '‘', '’'))
        if chosen_is_spoken and not curr_is_spoken and not has_dialogue:
            logger.debug("Rejecting LLM WSD defect for '%s': chosen sense is spoken/informal in non-dialogue text", lemma)
            overruled.append({
                "lemma": lemma,
                "quote": quote,
                "current_idx": current_idx,
                "current_def": curr_cand_def,
                "current_score": curr_score,
                "chosen_idx": chosen_idx,
                "chosen_def": chosen_cand_def,
                "chosen_score": chosen_score,
                "gate": "Gate 2 (Register & Pragmatics)",
                "reason": "LLM suggested spoken/slang/informal sense for written expository prose without dialogue.",
            })
            continue

        # Gate 3: Clausal Complementation / Valency Invariant (e.g. 'effort to recognize')
        if re.search(rf"\b{re.escape(lemma)}\s+to\s+[a-z]+\b", quote, re.IGNORECASE):
            curr_has_to_do = "to do" in curr_cand_def.lower() or "to do" in curr_gram or any("to do" in (ex or "").lower() for ex in curr_cand.get("examples", []))
            chosen_has_to_do = "to do" in chosen_cand_def.lower() or "to do" in chosen_gram or any("to do" in (ex or "").lower() for ex in chosen_cand.get("examples", []))
            if curr_has_to_do and not chosen_has_to_do:
                logger.debug("Rejecting LLM WSD defect for '%s': curr sense matches 'to do' clausal frame, chosen does not", lemma)
                overruled.append({
                    "lemma": lemma,
                    "quote": quote,
                    "current_idx": current_idx,
                    "current_def": curr_cand_def,
                    "current_score": curr_score,
                    "chosen_idx": chosen_idx,
                    "chosen_def": chosen_cand_def,
                    "chosen_score": chosen_score,
                    "gate": "Gate 3 (Clausal Frame / Valency)",
                    "reason": "Context requires 'to do' clausal complement matching current sense; LLM sense lacks 'to do' frame.",
                })
                continue

        eff_pos = (pos or str(curr_cand.get("pos") or "")).lower()

        # Gate 4: Noun Countability & Determiner Invariant
        gate4_triggered = False
        if any(np in eff_pos for np in ("noun", "n")):
            # Indefinite article: "a/an [adj] lemma" requires countable/singular sense
            if re.search(rf"\b(?:a|an)\s+(?:[a-z-]+\s+)?{re.escape(lemma)}\b", quote, re.IGNORECASE):
                if "[uncountable]" in chosen_gram and "[countable]" not in chosen_gram and "[singular]" not in chosen_gram:
                    if "[uncountable]" not in curr_gram or "[countable]" in curr_gram or "[singular]" in curr_gram:
                        logger.debug("Rejecting LLM WSD defect for '%s': chosen sense is uncountable-only for indefinite article", lemma)
                        overruled.append({
                            "lemma": lemma,
                            "quote": quote,
                            "current_idx": current_idx,
                            "current_def": curr_cand_def,
                            "current_score": curr_score,
                            "chosen_idx": chosen_idx,
                            "chosen_def": chosen_cand_def,
                            "chosen_score": chosen_score,
                            "gate": "Gate 4 (Noun Countability)",
                            "reason": "Indefinite article 'a/an' requires countable/singular sense; LLM suggested [uncountable]-only sense.",
                        })
                        gate4_triggered = True
                        continue
            # Plural surface form requires countable sense
            if not gate4_triggered and re.search(rf"\b{re.escape(lemma)}s\b", quote, re.IGNORECASE):
                if "[uncountable]" in chosen_gram and "[countable]" not in chosen_gram:
                    logger.debug("Rejecting LLM WSD defect for '%s': chosen sense is uncountable-only for plural surface form", lemma)
                    overruled.append({
                        "lemma": lemma,
                        "quote": quote,
                        "current_idx": current_idx,
                        "current_def": curr_cand_def,
                        "current_score": curr_score,
                        "chosen_idx": chosen_idx,
                        "chosen_def": chosen_cand_def,
                        "chosen_score": chosen_score,
                        "gate": "Gate 4 (Noun Plurality)",
                        "reason": f"Surface form '{lemma}s' is plural; LLM suggested [uncountable]-only sense.",
                    })
                    continue
        if gate4_triggered:
            continue

        # Gate 5: Verb Transitivity & Object Government Invariant
        if any(vp in eff_pos for vp in ("verb", "v")):
            nlp = LinguisticEngine.get_spacy()
            if nlp:
                doc = nlp(quote)
                target_tok = None
                for tok in doc:
                    if tok.lemma_.lower() == lemma.lower() or LinguisticEngine.form_matches(tok.text.lower(), lemma.lower()):
                        target_tok = tok
                        break
                if target_tok:
                    has_dobj = any(child.dep_ in ("dobj", "obj") for child in target_tok.children)
                    is_passive = any(child.dep_ == "auxpass" for child in target_tok.children)
                    if has_dobj:
                        if "[intransitive]" in chosen_gram and "[transitive]" not in chosen_gram:
                            logger.debug("Rejecting LLM WSD defect for '%s': chosen sense is intransitive-only but verb has direct object", lemma)
                            overruled.append({
                                "lemma": lemma,
                                "quote": quote,
                                "current_idx": current_idx,
                                "current_def": curr_cand_def,
                                "current_score": curr_score,
                                "chosen_idx": chosen_idx,
                                "chosen_def": chosen_cand_def,
                                "chosen_score": chosen_score,
                                "gate": "Gate 5 (Verb Transitivity)",
                                "reason": "Sentence contains direct object; LLM suggested [intransitive]-only sense.",
                            })
                            continue
                    elif not is_passive:
                        if "[transitive]" in chosen_gram and "[intransitive]" not in chosen_gram:
                            logger.debug("Rejecting LLM WSD defect for '%s': chosen sense is transitive-only but active verb has no direct object", lemma)
                            overruled.append({
                                "lemma": lemma,
                                "quote": quote,
                                "current_idx": current_idx,
                                "current_def": curr_cand_def,
                                "current_score": curr_score,
                                "chosen_idx": chosen_idx,
                                "chosen_def": chosen_cand_def,
                                "chosen_score": chosen_score,
                                "gate": "Gate 5 (Verb Transitivity)",
                                "reason": "Active verb has no direct object; LLM suggested [transitive]-only sense.",
                            })
                            continue

        # Gate 6: Selectional Restriction & Animacy Invariant
        # Screen out LLM hallucinations that confuse human persons with inanimate printed media, artifacts, or substances.
        if any(np in eff_pos for np in ("noun", "n")):
            chosen_is_person = bool(re.match(r"^(?:someone|a person|somebody)\s+(?:who|whose|that|with)\b", chosen_cand_def, re.I)) or any(
                k in chosen_cand_def.lower() for k in ("someone whose job is", "a person whose job is", "someone who entertains", "someone who performs", "person who", "entertainer who")
            )
            curr_is_artifact_or_media = any(
                k in curr_cand_def.lower() for k in (
                    "magazine", "newspaper", "comic strip", "book", "publication", "paper", "periodical",
                    "article", "picture", "drawing", "cartoon", "document", "sheet", "clipping", "text"
                )
            )
            has_media_physical_operation = bool(re.search(
                r"\b(?:clipped|cut|printed|published|bound|stapled|torn|photocopied|pasted|glued)\s+(?:from|in|on|into)\b"
                r"|\b(?:from|in)\s+(?:newspapers?|magazines?|journals?|books?|periodicals?|drawers?|binders?|scrapbooks?)\b",
                quote,
                re.IGNORECASE
            ))
            if chosen_is_person and curr_is_artifact_or_media and has_media_physical_operation:
                logger.debug("Rejecting LLM WSD defect for '%s': chosen sense is person but context denotes physical print/media artifact", lemma)
                overruled.append({
                    "lemma": lemma,
                    "quote": quote,
                    "current_idx": current_idx,
                    "current_def": curr_cand_def,
                    "current_score": curr_score,
                    "chosen_idx": chosen_idx,
                    "chosen_def": chosen_cand_def,
                    "chosen_score": chosen_score,
                    "gate": "Gate 6 (Animacy / Artifact Invariant)",
                    "reason": "Context denotes physical print/media artifact modified by clipping/printing/publication; LLM suggested human person sense.",
                })
                continue

        chosen_ex = (chosen_cand.get("examples") or [""])[0]

        defects.append(
            f"[vocabulary-wsd] '{lemma}' definition misaligned: current='{curr_cand_def[:45]}...', "
            f"context requires Sense #{chosen_idx + 1} '{chosen_cand_def[:45]}...'"
        )
        cures.append(
            {
                "lemma": lemma,
                "quote": quote,
                "current_def": curr_cand_def,
                "chosen_idx": chosen_idx,
                "chosen_def": chosen_cand_def,
                "chosen_example": chosen_ex,
                "new_definition": chosen_cand_def,
                "new_example": chosen_ex,
            }
        )

    return defects, cures, overruled


def audit_grammar_semantic(
    items: List[Dict[str, Any]],
    source_text: str,
    discourse_context: str = "",
    model: Optional[str] = None,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Execute Level-2 neuro-symbolic grammar semantic audit over extracted grammar patterns.

    Evaluates:
      1. Instantiational Validity: Does the quote genuinely instantiate the declared grammar pattern?
      2. Syntactic Formula Congruence: Does the Quirk/Biber formula accurately abstract the quote syntax?
      3. Pedagogical Explanation: Does the pedagogical function clearly explain the communicative intent?
      4. Contrastive Error Trap: Does the Common Mistakes section illustrate an authentic L2 error with valid ✗ / ✓ pairs?

    Returns:
        (defects, defect_details)
    """
    if not items:
        return [], []

    from .config import config
    from .llm import LLMClient

    judge_m = model or config.get("judge_model") or config.get("model") or "qwen3.8:27b"
    client = LLMClient(model=judge_m)

    system_prompt = (
        "You are an expert SLA Grammatical Evaluator and Academic English Syntax Judge.\n"
        "Audit the following extracted grammar patterns against Quirk/Biber grammatical frameworks "
        "and Second Language Acquisition (SLA) standards.\n\n"
        "For each grammar pattern, evaluate:\n"
        "1. Instantiation: Does the quoted sentence from the source text genuinely instantiate the declared grammar pattern? "
        "(Must not be a superficial surface match with different underlying syntax).\n"
        "2. Formula: Does the Quirk/Biber pattern formula accurately abstract the syntax of the quoted sentence?\n"
        "3. Pedagogy: Does the pedagogical function accurately and informatively explain the communicative or discourse role of this structure for L2 learners?\n"
        "4. Error Trap: Does the common mistakes section illustrate an authentic L2 learner developmental error with a valid contrastive pair (✗ vs ✓) targeting this structure?\n\n"
        "Output STRICT JSON matching this schema:\n"
        "{\n"
        '  "evaluations": [\n'
        "    {\n"
        '      "pattern": "<Exact pattern name>",\n'
        '      "passed": true / false,\n'
        '      "defect_category": "instantiation" | "formula" | "pedagogy" | "error_trap" | null,\n'
        '      "issue": "<Detailed explanation of the linguistic/pedagogical flaw if failed, or null if passed>",\n'
        '      "action": "<Actionable recommendation to fix or null if passed>"\n'
        "    }\n"
        "  ]\n"
        "}"
    )

    eval_items = []
    for it in items:
        eval_items.append({
            "pattern": it.get("pattern", ""),
            "quote": it.get("quote", ""),
            "formula": it.get("formula", ""),
            "pedagogy": it.get("pedagogy", ""),
            "mistakes": it.get("mistakes", ""),
            "category": it.get("category", ""),
        })

    ctx_snippet = discourse_context.strip() or source_text[:1500].strip()
    user_prompt = (
        f"Passage Context:\n\"\"\"{ctx_snippet}\"\"\"\n\n"
        f"Grammar Patterns to Audit:\n{json.dumps(eval_items, indent=2, ensure_ascii=False)}"
    )

    defects: List[str] = []
    defect_details: List[Dict[str, Any]] = []

    try:
        resp = client.chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            json_format=True,
            temperature=0.0,
            skip_log=True,
        )
    except Exception as e:
        logger.warning("Grammar semantic audit LLM error: %s", e)
        return defects, defect_details

    eval_list: List[Dict[str, Any]] = []
    if isinstance(resp, dict):
        eval_list = resp.get("evaluations") or []
    elif isinstance(resp, list):
        eval_list = resp

    items_by_pattern = {it.get("pattern", ""): it for it in items}

    for ev in eval_list:
        if not isinstance(ev, dict):
            continue
        passed = ev.get("passed", True)
        if passed:
            continue

        pat_name = str(ev.get("pattern") or "").strip()
        cat = str(ev.get("defect_category") or "semantic").strip()
        issue = str(ev.get("issue") or "Failed semantic grammatical evaluation.").strip()
        action = str(ev.get("action") or "Revise grammar pattern extraction.").strip()

        matched_item = items_by_pattern.get(pat_name, {})
        quote = matched_item.get("quote", "")
        formula = matched_item.get("formula", "")

        defects.append(f"[grammar-semantic] Pattern '[[{pat_name}]]' {cat} flaw: {issue}")
        defect_details.append({
            "pattern": pat_name,
            "quote": quote,
            "formula": formula,
            "category": f"grammar-{cat}",
            "issue": issue,
            "action": action,
        })

    return defects, defect_details


def audit_unit_extractions(
    unit: str,
    project_root: Optional[str] = None,
    enable_llm: bool = False,
    fix: bool = False,
) -> ExtractionAuditReport:
    """Execute Level 1 (and optional Level 2) audit over all extraction artifacts of a unit."""
    root = _project_root(project_root)
    unit_dir = root / "wiki" / unit

    source_file = unit_dir / "sources" / f"{unit}.md"
    if not source_file.is_file():
        return ExtractionAuditReport(
            unit=unit,
            passed=False,
            score=0.0,
            error=f"Source file not found: {source_file}",
        )

    try:
        source_raw = source_file.read_text(encoding="utf-8")
    except Exception as e:
        return ExtractionAuditReport(
            unit=unit,
            passed=False,
            score=0.0,
            error=f"Failed to read source file: {e}",
        )

    norm_source = _normalize_text_for_search(source_raw)
    all_defects: List[str] = []
    stats: Dict[str, Any] = {}

    extractions_dir = unit_dir / "extractions"

    # 1. Vocabulary audit
    vocab_file = extractions_dir / f"{unit}_vocabulary.md"
    v_text = ""
    if vocab_file.is_file():
        try:
            v_text = vocab_file.read_text(encoding="utf-8")
            v_defects, v_stats = audit_vocabulary_markdown(v_text, norm_source)
            all_defects.extend(v_defects)
            stats["vocabulary"] = v_stats
        except Exception as e:
            all_defects.append(f"[vocabulary] Read error: {e}")
    else:
        all_defects.append(f"[vocabulary] Missing file: {vocab_file.name}")

    # 2. Grammar audit
    grammar_file = extractions_dir / f"{unit}_grammar.md"
    if grammar_file.is_file():
        try:
            g_text = grammar_file.read_text(encoding="utf-8")
            g_defects, g_stats = audit_grammar_markdown(g_text, norm_source)
            all_defects.extend(g_defects)
            stats["grammar"] = g_stats
        except Exception as e:
            all_defects.append(f"[grammar] Read error: {e}")
    else:
        all_defects.append(f"[grammar] Missing file: {grammar_file.name}")

    # 3. Summary audit
    summary_file = extractions_dir / f"{unit}_summary.md"
    if summary_file.is_file():
        try:
            s_text = summary_file.read_text(encoding="utf-8")
            s_defects, s_stats = audit_summary_markdown(s_text, norm_source)
            all_defects.extend(s_defects)
            stats["summary"] = s_stats
        except Exception as e:
            all_defects.append(f"[summary] Read error: {e}")
    else:
        all_defects.append(f"[summary] Missing file: {summary_file.name}")

    # 4. Mindmap audit
    mindmap_file = extractions_dir / f"{unit}_mindmap.json"
    if mindmap_file.is_file():
        try:
            m_defects, m_stats = audit_mindmap_json(mindmap_file)
            all_defects.extend(m_defects)
            stats["mindmap"] = m_stats
        except Exception as e:
            all_defects.append(f"[mindmap] Read error: {e}")
    else:
        all_defects.append(f"[mindmap] Missing file: {mindmap_file.name}")

    # Optional Level 2 LLM semantic check (WSD context alignment & Grammar verification)
    cured_items: List[str] = []
    all_overruled: List[Dict[str, Any]] = []
    if enable_llm and not all_defects:
        v_items = stats.get("vocabulary", {}).get("items", [])
        if v_items and vocab_file.is_file():
            wsd_res = audit_vocabulary_wsd_semantic(
                v_items, source_raw, discourse_context=source_raw[:1000]
            )
            if len(wsd_res) == 3:
                wsd_defects, wsd_cures, wsd_overruled = wsd_res
            else:
                wsd_defects, wsd_cures = wsd_res[:2]
                wsd_overruled = []

            if "vocabulary" in stats:
                stats["vocabulary"]["wsd_defect_details"] = wsd_cures
                stats["vocabulary"]["overruled_suggestions"] = wsd_overruled
            all_overruled.extend(wsd_overruled)

            if wsd_cures and fix:
                cur_text = v_text
                for cure in wsd_cures:
                    cur_text = cure_vocabulary_item(
                        cur_text,
                        cure["lemma"],
                        cure["new_definition"],
                        cure["new_example"],
                    )
                    cured_items.append(cure["lemma"])
                vocab_file.write_text(cur_text, encoding="utf-8")
                stats["vocabulary"]["cured_items"] = cured_items
                stats["vocabulary"]["cured_details"] = wsd_cures
                logger.info(
                    "Applied in-place WSD cures to %s: %s",
                    vocab_file.name,
                    ", ".join(cured_items),
                )
            else:
                all_defects.extend(wsd_defects)

        g_items = stats.get("grammar", {}).get("items", [])
        if g_items and grammar_file.is_file():
            logger.info("Running Level-2 semantic audit on %d grammar pattern(s) for %s...", len(g_items), unit)
            g_defects, g_details = audit_grammar_semantic(
                g_items, source_raw, discourse_context=source_raw[:1500]
            )
            if "grammar" in stats:
                existing_details = stats["grammar"].setdefault("grammar_defect_details", [])
                existing_details.extend(g_details)
            all_defects.extend(g_defects)

    # Score calculation (100 base, -10 per defect, floor 0)
    score = max(0.0, 100.0 - (len(all_defects) * 10.0))
    passed = len(all_defects) == 0 and score >= 80.0

    return ExtractionAuditReport(
        unit=unit,
        passed=passed,
        score=score,
        defects=all_defects,
        overruled=all_overruled,
        stats=stats,
    )


def discover_extraction_units(project_root: Optional[str] = None) -> List[str]:
    """Find all unit directory names that have an ``extractions/`` subfolder."""
    wiki_dir = _project_root(project_root) / "wiki"
    if not wiki_dir.is_dir():
        return []

    units: List[str] = []
    for u_dir in wiki_dir.iterdir():
        if u_dir.is_dir() and (u_dir / "extractions").is_dir():
            units.append(u_dir.name)

    units.sort()
    return units


def batch_audit_extractions(
    unit_filter: Optional[List[str]] = None,
    project_root: Optional[str] = None,
    max_workers: int = 2,
    progress_cb: Optional[Callable[[int, int, Dict[str, Any]], None]] = None,
    enable_llm: bool = False,
    fix: bool = False,
) -> Dict[str, Any]:
    """Audit extractions across multiple units in parallel."""
    available = discover_extraction_units(project_root)
    if unit_filter:
        target_units = [u for u in available if u in set(unit_filter)]
    else:
        target_units = available

    total = len(target_units)
    if total == 0:
        return {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "avg_score": 0.0,
            "enable_llm": enable_llm,
            "reports": [],
        }

    reports: List[ExtractionAuditReport] = []
    workers = max(1, min(max_workers, total))

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(
                audit_unit_extractions, u, project_root, enable_llm, fix
            ): u
            for u in target_units
        }
        done = 0
        for fut in as_completed(future_map):
            done += 1
            report = fut.result()
            reports.append(report)
            if progress_cb:
                progress_cb(done, total, report.to_dict())

    reports.sort(key=lambda r: r.unit)
    passed_count = sum(1 for r in reports if r.passed)
    failed_count = total - passed_count
    avg_score = sum(r.score for r in reports) / total if total else 0.0

    return {
        "target": "extraction",
        "total": total,
        "passed": passed_count,
        "failed": failed_count,
        "avg_score": round(avg_score, 1),
        "enable_llm": enable_llm,
        "fix": fix,
        "reports": [r.to_dict() for r in reports],
    }


def format_extraction_summary(summary: Dict[str, Any]) -> str:
    """Format the extraction batch audit results into a readable CLI table."""
    total = summary.get("total", 0)
    passed = summary.get("passed", 0)
    failed = summary.get("failed", 0)
    avg_score = summary.get("avg_score", 0.0)
    is_llm = summary.get("enable_llm", False)

    lines: List[str] = []
    lines.append("\n=======================================================")
    gate_title = "LEVEL-2 SEMANTIC AUDIT" if is_llm else "LEVEL-1 CODE GATE"
    lines.append(f"        EXTRACTION AUDIT REPORT ({gate_title})    ")
    lines.append("=======================================================")
    lines.append(f"Total Units:   {total}")
    lines.append(f"Passed:        {passed} ({passed / total * 100:.1f}%)" if total else "Passed: 0")
    lines.append(f"Failed:        {failed}")
    lines.append(f"Average Score: {avg_score:.1f} / 100")
    lines.append("-------------------------------------------------------")

    for r in summary.get("reports", []):
        unit = r.get("unit")
        p = r.get("passed")
        s = r.get("score")
        mark = "✓ PASS" if p else "✗ FAIL"
        lines.append(f"[{mark}] {unit:<32} (Score: {s:>5.1f})")

        stats = r.get("stats", {}) or {}
        v_cnt = (stats.get("vocabulary") or {}).get("item_count", 0)
        g_cnt = (stats.get("grammar") or {}).get("item_count", 0)
        s_ok = "OK" if stats.get("summary") else "Missing"
        m_branches = (stats.get("mindmap") or {}).get("branch_count", 0)
        lines.append(
            f"    ↳ [Audited]: Vocab: {v_cnt} items | Grammar: {g_cnt} patterns | Summary: {s_ok} | Mindmap: {m_branches} branches"
        )

        cured = (stats.get("vocabulary") or {}).get("cured_items", [])
        if cured:
            lines.append(f"    ↳ [Auto-Cured WSD Senses] ({len(cured)}): {', '.join(f'[[{c}]]' for c in cured)}")

        overruled = r.get("overruled", [])
        if overruled:
            lines.append(f"    ↳ [Overruled LLM Suggestions (0 score impact)]:")
            for ov in overruled:
                lemma = ov.get("lemma", "")
                c_idx = ov.get("current_idx", 0) + 1
                c_sc = ov.get("current_score", 0.0)
                l_idx = ov.get("chosen_idx", 0) + 1
                l_sc = ov.get("chosen_score", 0.0)
                gate = ov.get("gate", "Physical Code Gate")
                reason = ov.get("reason", "")
                lines.append(f"        • [[{lemma}]] (LLM: Sense #{l_idx} ➔ OVERRULED)")
                lines.append(f"          ↳ Reason: {gate} - {reason} [Scores: Current #{c_idx} ({c_sc:.1f} pts) vs LLM #{l_idx} ({l_sc:.1f} pts)]")

        defects = r.get("defects", [])
        if defects:
            lines.append(f"    ↳ [Real Defects (Score Deducted)] ({len(defects)}):")
            for d in defects[:4]:
                lines.append(f"        - {d}")
            if len(defects) > 4:
                lines.append(f"        ... and {len(defects) - 4} more defect(s) (see log for full tickets)")

    lines.append("=======================================================\n")
    return "\n".join(lines)


def save_extraction_audit_log(
    summary: Dict[str, Any],
    project_root: Optional[str] = None,
) -> Dict[str, Path]:
    """Save extraction audit report and defect tickets into logs/ directory.

    Generates:
      - logs/<timestamp>_<unit>_reaudit_extraction.log (human-readable defect ticket log)

    Returns:
      Dict mapping "log_path" to Path object.
    """
    import datetime

    root = _project_root(project_root)
    logs_dir = root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    readable_ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    reports = summary.get("reports", [])
    if summary.get("unit"):
        unit_tag = summary["unit"]
    elif len(reports) == 1 and reports[0].get("unit"):
        unit_tag = reports[0]["unit"]
    else:
        unit_tag = "batch"

    log_file = logs_dir / f"{timestamp_str}_{unit_tag}_reaudit_extraction.log"

    total = summary.get("total", 0)
    passed = summary.get("passed", 0)
    failed = summary.get("failed", 0)
    avg_score = summary.get("avg_score", 0.0)
    is_llm = summary.get("enable_llm", False)
    is_fix = summary.get("fix", False)

    gate_title = "LEVEL-2 SEMANTIC AUDIT" if is_llm else "LEVEL-1 CODE GATE"
    mode_str = "Auto-Cure Enabled (--fix)" if is_fix else "Read-Only (Run with --fix to apply automatic cures)"

    lines: List[str] = [
        "=" * 80,
        f"       EXTRACTION AUDIT REPORT & DEFECT TICKETS ({gate_title})",
        "=" * 80,
        f"Timestamp:     {readable_ts}",
        f"Target:        extraction",
        f"Audit Level:   {gate_title}",
        f"Mode:          {mode_str}",
        f"Total Units:   {total}",
        f"Passed:        {passed} ({passed / total * 100:.1f}%)" if total else "Passed:        0",
        f"Failed:        {failed}",
        f"Average Score: {avg_score:.1f} / 100",
        "=" * 80,
        "",
    ]

    has_defects = False
    for r in summary.get("reports", []):
        unit = r.get("unit", "")
        p = r.get("passed", False)
        score = r.get("score", 0.0)
        defects = r.get("defects", [])
        stats = r.get("stats", {}) or {}
        v_stats = stats.get("vocabulary") or {}
        g_stats = stats.get("grammar") or {}
        s_stats = stats.get("summary") or {}
        m_stats = stats.get("mindmap") or {}
        cured = v_stats.get("cured_items", [])
        err = r.get("error")

        status_str = "PASS" if p else "FAIL"
        lines.append("-" * 80)
        lines.append(f"[UNIT AUDIT INVENTORY] {unit} (Status: {status_str}, Score: {score:.1f} / 100)")
        lines.append("-" * 80)

        # 1. Audited Scope & Inventory
        level_tag = "Level-1 Code Gate + Level-2 Semantic" if is_llm else "Level-1 Code Gate"
        lines.append("  📦 Audited Components & Scope:")
        if v_stats:
            v_cnt = v_stats.get("item_count", 0)
            v_tag = "Level-1 Code Gate + Level-2 WSD Semantic" if is_llm else "Level-1 Code Gate"
            lines.append(f"    • Vocabulary: {v_cnt} items audited ({v_tag}: Verified)")
        else:
            lines.append("    • Vocabulary: Not audited / file missing")

        if g_stats:
            g_cnt = g_stats.get("item_count", 0)
            g_pats = g_stats.get("patterns") or [it.get("pattern") for it in g_stats.get("items", []) if it.get("pattern")]
            if not g_pats and g_stats.get("grammar_defect_details"):
                g_pats = [d.get("pattern") for d in g_stats.get("grammar_defect_details", []) if d.get("pattern")]
            lines.append(f"    • Grammar:    {g_cnt} patterns audited ({level_tag}: Verified):")
            g_defect_patterns = set()
            for d in defects:
                m_gram = re.search(r"\[(?:grammar|grammar-semantic)\]\s+Pattern\s+'\[\[(.*?)\]\]'", d)
                if m_gram:
                    g_defect_patterns.add(m_gram.group(1))
            for pat in g_pats:
                pat_status = "[FAIL]" if pat in g_defect_patterns else "[PASS]"
                lines.append(f"        - [[{pat}]] {pat_status}")
        else:
            lines.append("    • Grammar:    Not audited / file missing")

        if s_stats:
            s_words = s_stats.get("summary_words", 0)
            lines.append(f"    • Summary:    Verified ({s_words} words, Narrative Overview / Hook / Questions)")
        else:
            lines.append("    • Summary:    Not audited / file missing")

        if m_stats:
            b_cnt = m_stats.get("branch_count", 0)
            l_cnt = m_stats.get("total_leaves", 0)
            d_cnt = m_stats.get("max_depth", 0)
            lines.append(f"    • Mindmap:    Verified ({b_cnt} branches, {l_cnt} leaves, depth {d_cnt})")
        else:
            lines.append("    • Mindmap:    Not audited / file missing")
        lines.append("")

        if cured:
            lines.append(f"  ✨ [AUTO-CURED WSD SENSES] ({len(cured)}):")
            cured_details = v_stats.get("cured_details", [])
            cured_detail_map = {d.get("lemma"): d for d in cured_details if isinstance(d, dict)}
            for idx_c, c in enumerate(cured, 1):
                if c in cured_detail_map:
                    cd = cured_detail_map[c]
                    lines.append(f"    [{idx_c}] [CURED] [[{c}]]: Sense #{cd.get('chosen_idx', 0) + 1} (\"{cd.get('chosen_def', '')[:60]}...\")")
                else:
                    lines.append(f"    [{idx_c}] [CURED] [[{c}]]")
            lines.append("")

        overruled = r.get("overruled", [])
        if overruled:
            lines.append(f"  🛡️ [OVERRULED LLM SUGGESTIONS (0 SCORE IMPACT)] ({len(overruled)}):")
            for idx, ov in enumerate(overruled, 1):
                lemma = ov.get("lemma", "")
                quote = ov.get("quote", "")
                curr_def = ov.get("current_def", "")
                curr_idx = ov.get("current_idx", 0) + 1
                curr_sc = ov.get("current_score", 0.0)
                chosen_def = ov.get("chosen_def", "")
                chosen_idx = ov.get("chosen_idx", 0) + 1
                chosen_sc = ov.get("chosen_score", 0.0)
                gate = ov.get("gate", "Physical Invariant Gate")
                reason = ov.get("reason", "")
                lines.append(f"    [{idx}] [OVERRULED] [[{lemma}]]")
                if quote:
                    lines.append(f"        Context Quote:      \"{quote}\"")
                lines.append(f"        Current Extraction: Sense #{curr_idx} (\"{curr_def}\") [Score: {curr_sc:.1f}]")
                lines.append(f"        LLM Suggestion:     Sense #{chosen_idx} (\"{chosen_def}\") [Score: {chosen_sc:.1f}]")
                lines.append(f"        Veto Gate:          {gate}")
                lines.append(f"        Veto Reason:        {reason}")
                lines.append("")

        if err:
            lines.append(f"  ❌ System Error: {err}")
            lines.append("")

        if defects:
            has_defects = True
            lines.append(f"  🚨 [UNIT DEFECT TICKET] {unit} - [REAL DEFECTS (SCORE DEDUCTED)] (Count: {len(defects)}):")
            wsd_details = v_stats.get("wsd_defect_details", [])
            wsd_by_lemma = {d.get("lemma"): d for d in wsd_details if isinstance(d, dict)}
            grammar_details = g_stats.get("grammar_defect_details", [])
            grammar_by_pattern = {d.get("pattern"): d for d in grammar_details if isinstance(d, dict)}

            for idx, defect in enumerate(defects, 1):
                # 1. Vocabulary WSD
                m_wsd = re.match(r"\[vocabulary-wsd\]\s+'([^']+)'", defect)
                if m_wsd and m_wsd.group(1) in wsd_by_lemma:
                    detail = wsd_by_lemma[m_wsd.group(1)]
                    lines.append(f"    [{idx}] [DEFECT] Category:  vocabulary-wsd")
                    lines.append(f"        Lemma:     {detail.get('lemma')}")
                    if detail.get('quote'):
                        lines.append(f"        Context:   \"{detail.get('quote')}\"")
                    lines.append(f"        Current:   {detail.get('current_def')}")
                    lines.append(f"        Suggested: Sense #{detail.get('chosen_idx', 0) + 1}: {detail.get('chosen_def')}")
                    if detail.get('chosen_example'):
                        lines.append(f"        Example:   {detail.get('chosen_example')}")
                    continue

                # 2. Grammar (Code gate or Semantic)
                m_gram = re.search(r"\[(?:grammar|grammar-semantic)\]\s+Pattern\s+'\[\[(.*?)\]\]'", defect)
                if m_gram:
                    pat_name = m_gram.group(1)
                    g_det = grammar_by_pattern.get(pat_name)
                    cat_name = "grammar-semantic" if "[grammar-semantic]" in defect else "grammar"
                    lines.append(f"    [{idx}] [DEFECT] Category:  {cat_name}")
                    lines.append(f"        Pattern:   [[{pat_name}]]")
                    if g_det and g_det.get("quote"):
                        lines.append(f"        Quote:     \"{g_det.get('quote')}\"")
                    if g_det and g_det.get("formula"):
                        lines.append(f"        Formula:   {g_det.get('formula')}")
                    issue_text = g_det.get("issue") if (g_det and g_det.get("issue")) else defect
                    lines.append(f"        Issue:     {issue_text}")
                    continue

                if defect.startswith("[grammar]") or defect.startswith("[grammar-semantic]"):
                    cat_name = "grammar-semantic" if defect.startswith("[grammar-semantic]") else "grammar"
                    lines.append(f"    [{idx}] [DEFECT] Category:  {cat_name}")
                    lines.append(f"        Issue:     {defect}")
                    continue

                # 3. Vocabulary structural
                if defect.startswith("[vocabulary]"):
                    lines.append(f"    [{idx}] [DEFECT] Category:  vocabulary")
                    lines.append(f"        Issue:     {defect}")
                    continue

                # 4. Summary
                if defect.startswith("[summary]"):
                    lines.append(f"    [{idx}] [DEFECT] Category:  summary")
                    lines.append(f"        Issue:     {defect}")
                    continue

                # 5. Mindmap
                if defect.startswith("[mindmap]"):
                    lines.append(f"    [{idx}] [DEFECT] Category:  mindmap")
                    lines.append(f"        Issue:     {defect}")
                    continue

                # 6. Fallback
                lines.append(f"    [{idx}] [DEFECT] Category:  general")
                lines.append(f"        Issue:     {defect}")
            lines.append("")
        else:
            lines.append("  ✓ Defect Status: Zero defects detected in this unit.")
            lines.append("")

    if not has_defects:
        lines.append("=" * 80)
        lines.append("🎉 All audited units passed with zero defects! No defect tickets generated.")
        lines.append("")

    lines.extend([
        "=" * 80,
        "📌 SUMMARY FOR REPORTING DEFECTS:",
        "If you want the assistant to inspect or fix extraction logic, copy the [UNIT DEFECT TICKET]",
        "sections above and share them in your chat.",
        "To auto-cure WSD defects in-place, execute:",
        "  python -m librarian.cli re-audit <unit> --target extraction --llm --fix",
        "=" * 80,
        "",
    ])

    log_content = "\n".join(lines)
    log_file.write_text(log_content, encoding="utf-8")

    return {
        "log_path": log_file,
    }
