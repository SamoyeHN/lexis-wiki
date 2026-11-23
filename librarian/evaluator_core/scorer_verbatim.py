"""Dimension 2: Source Faithfulness & Verbatim Verification (W_VERBATIM = 30.0)."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from ..linguistics import LinguisticEngine
from .eval_helpers import (
    STOP_SLOTS,
    W_VERBATIM,
    _clean_core,
    _extract_source_content,
    _extract_wordlist,
    _form_in_text,
    _is_hallucinated_quote,
    _ngram_coverage,
    _wordlist_matches,
)


def _score_verbatim(
    items: List[Dict[str, Any]],
    task_type: str,
    user_prompt: str,
    context_prompt: str = "",
) -> Tuple[Optional[float], List[str]]:
    """Dimension 2 (0–30). Applies to extraction and quiz tasks; returns (None, []) otherwise.

    For extraction tasks this verifies quoted sentences are faithful to the source.
    For quiz tasks this verifies every target_word comes from the supplied word list
    (anti-hallucination: the assessment may not test fabricated vocabulary).
    """
    if task_type not in ("vocabulary", "expressions", "grammar", "quiz"):
        return None, []  # N/A -> normalized out of composite score
    if not items:
        return 0.0, ["❌ No items to evaluate for source faithfulness"]

    # --- Quiz: every target_word must be present in the supplied word list ---
    if task_type == "quiz":
        flags: List[str] = []
        effective_prompt = user_prompt if user_prompt.strip() else (context_prompt or "")
        wordlist = {_clean_core(w) for w in _extract_wordlist(effective_prompt)}
        wordlist = {w for w in wordlist if w}
        checks = matches = 0
        draft_prompt = re.split(r"\n\s*###+\s*🚨|\n\s*###+\s*\[QUALITY AUDIT REVIEW", effective_prompt, flags=re.IGNORECASE)[0]
        clean_draft = _clean_core(draft_prompt)

        for item in items:
            target = str(item.get("target_word") or "").strip()
            if not target:
                continue
            clean_target = _clean_core(target)
            if not clean_target:
                continue
            checks += 1
            if wordlist:
                if _wordlist_matches(clean_target, wordlist):
                    matches += 1
                elif clean_draft and clean_target in clean_draft:
                    # Target word matched in initial prompt draft text
                    matches += 1
                else:
                    flags.append(f"❌ Target word '{target}' not found in supplied word list (possible hallucination)")
            else:
                # If no wordlist extracted from prompt (e.g. self-correction defect ticket),
                # assume compliant with blueprint to avoid false-positive draft content warnings
                matches += 1
        if checks == 0:
            # No target_word items (e.g. reading/translation quizzes) -> nothing to verify -> N/A
            return None, flags
        return max(0.0, round((matches / checks) * W_VERBATIM, 1)), flags

    flags = []
    source = _extract_source_content(user_prompt)
    if not source and context_prompt:
        source = _extract_source_content(context_prompt)
    core_src = _clean_core(source)
    checks = matches = 0

    for item in items:
        quote = item.get("quoted_sentence") or item.get("quote") or item.get("context_sentence")
        word = str(item.get("word") or item.get("pattern_formula") or "").strip()
        checks += 1

        if not (quote and isinstance(quote, str) and quote.strip()):
            flags.append(f"❌ Missing quote/quoted_sentence for item: '{word[:40]}'")
            continue

        # Check 1: Explicit hallucination acknowledgment
        if _is_hallucinated_quote(quote):
            flags.append(f"❌ Hallucinated quote (explicitly inferred/absent): '{quote[:50]}...'")
            continue

        # Check 1.5: Reject prompt instructions / task metadata leakage
        if re.search(
            r"###\s*(?:TARGET|DETERMINISTIC|PASSAGE|SYLLABUS|OUTPUT|CORE\s+PEDAGOGICAL|JSON\s+SCHEMA)|Extract these exact|syllabus items|Do not skip or omit",
            quote,
            re.IGNORECASE,
        ):
            flags.append(f"❌ Prompt instruction leakage in quote: '{quote[:50]}...'")
            matches = max(0, matches - 1)
            continue

        # Check 2: Target word must be present in the quoted sentence (for vocabulary & expressions)
        if word and task_type in ("vocabulary", "expressions"):
            clean_word_no_slots = re.sub(r"\[.*?\]|\(.*?\)", " ", word)
            clean_word = _clean_core(clean_word_no_slots)
            clean_quote = _clean_core(quote)
            word_tokens = [w for w in clean_word.split() if w and w not in STOP_SLOTS]

            if word_tokens:
                # Match token, stem/inflection (e.g. degrade -> degradation, took -> take, pose -> poses), or irregular forms
                def _tok_in_quote(tok: str) -> bool:
                    return _form_in_text(tok, clean_quote)

                matched_count = sum(1 for w in word_tokens if _tok_in_quote(w))
                min_needed = max(1, len(word_tokens) // 2 + (1 if len(word_tokens) % 2 == 1 else 0))
                word_in_quote = (matched_count >= min_needed)
            else:
                word_in_quote = (clean_word in clean_quote)

            if not word_in_quote:
                flags.append(f"❌ Target word '{word}' does not appear in quoted sentence: '{quote[:40]}...'")
                # Severe deduction: directly penalize verbatim score rather than neutral skip
                matches = max(0, matches - 1)
                continue

            # Check 2.5: The headword itself must be a real standalone unit —
            # a recognized English word, or a standalone token in the quoted
            # sentence. Catches lemmatizer corruption (e.g. 'embe') that Check 2's
            # substring matching lets through ('embe' is a prefix of 'embedded'),
            # without false-flagging legitimate words WordNet lacks ('ice-cream',
            # "don't") or text-specific terms that appear verbatim in the quote.
            if " " not in clean_word and clean_word.isalpha():
                if LinguisticEngine.is_known_english_word(clean_word) is False:
                    if clean_word not in clean_quote.split():
                        flags.append(
                            f"❌ Headword '{word}' is not a recognized English word and does not appear "
                            f"as a standalone token in the quoted sentence (suspect lemmatization artifact)"
                        )
                        matches = max(0, matches - 1)
                        continue

        # Check 3: Cleaned quote in source or high n-gram coverage
        core_quote = _clean_core(quote)
        is_verbatim = False
        if core_quote:
            if core_quote in core_src or _ngram_coverage(core_quote, core_src, n=3) >= 0.80:
                is_verbatim = True
            elif "..." in quote or "…" in quote:
                # If model used ellipsis to omit middle parts of a long sentence, check each segment
                segments = [s.strip() for s in re.split(r'\.{3,}|…', quote) if s.strip()]
                meaningful_segs = [s for s in segments if len(_clean_core(s).split()) >= 2]
                if meaningful_segs and all(
                    _clean_core(seg) in core_src or _ngram_coverage(_clean_core(seg), core_src, n=3) >= 0.80
                    for seg in meaningful_segs
                ):
                    is_verbatim = True

        if is_verbatim:
            matches += 1
        else:
            flags.append(f"❌ Non-verbatim quote detected: '{quote[:40]}...'")

    if checks == 0:
        return 0.0, ["❌ No items to evaluate for source faithfulness"]
    return max(0.0, round((matches / checks) * W_VERBATIM, 1)), flags
