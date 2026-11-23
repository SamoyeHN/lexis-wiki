"""Deterministic Level 1 Hallucination Pruning Gate & Quote Magnetic Snapping."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from ..linguistics import LinguisticEngine
from .eval_helpers import (
    COBUILD_POS_TOKENS,
    STOP_SLOTS,
    _clean_core,
    _detect_task_type,
    _expand_contractions,
    _extract_items,
    _extract_source_content,
    _form_in_text,
    _ngram_coverage,
    _safe_str,
    auto_remap_grammar_category,
    canonical_formula_key,
)

def prune_hallucinated_items(parsed_data: Any, user_prompt: str, task_type: str = "") -> Tuple[Any, List[str]]:
    """Deterministic Level 1 Hallucination Pruning Gate.
    
    If the model extracts vocabulary/expressions where the headword neither appears
    in its quoted sentence nor in the source text, it is a pure hallucination.
    Rather than blindly looping retries that ask the model to fix nonexistent words,
    we surgically prune the hallucinated items deterministically.
    
    Returns:
        (pruned_parsed_data, list_of_pruned_flags)
    """
    if not isinstance(parsed_data, dict):
        return parsed_data, []
    
    if not task_type or task_type == "unknown" or task_type not in ("vocabulary", "expressions", "grammar"):
        detected = _detect_task_type(task_type or "", parsed_data)
        if detected in ("vocabulary", "expressions", "grammar"):
            task_type = detected
        else:
            return parsed_data, []

    items = _extract_items(parsed_data, task_type)
    if not items:
        return parsed_data, []

    source = _extract_source_content(user_prompt)
    core_src = _clean_core(source)
    if not core_src:
        return parsed_data, []

    # Initialize spaCy sentence pool for deterministic boundary snapping
    _, sentence_pool = LinguisticEngine.tokenize_and_index_sentences(source)

    key_by_type = {
        "vocabulary": "vocabulary",
        "expressions": "expressions",
        "grammar": "grammar_patterns",
    }
    array_key = key_by_type.get(task_type)
    if not array_key or array_key not in parsed_data or not isinstance(parsed_data[array_key], list):
        return parsed_data, []

    surviving_items = []
    pruned_flags = []
    seen_dedup_keys = set()

    for item in parsed_data[array_key]:
        if not isinstance(item, dict):
            surviving_items.append(item)
            continue

        # In-Place Self-Healing: Quote boundary magnetic snapping using sentence pool
        # Snaps [S-id] or partial quote with '...' to the pristine, authentic source sentence
        quote_field = "quoted_sentence" if "quoted_sentence" in item else ("quote" if "quote" in item else None)
        if quote_field and item.get(quote_field):
            raw_q = str(item[quote_field]).strip()
            snapped = LinguisticEngine.snap_to_sentence_pool(raw_q, sentence_pool)
            if snapped and snapped != raw_q:
                item[quote_field] = snapped
                pruned_flags.append(f"ℹ️ In-Place Self-Healing: Snapped quote '{raw_q[:30]}...' -> full authentic sentence")
            else:
                # Strip internal [S-id] prefix if quote was verbatim with S-id attached
                cleaned_q = re.sub(r"^\s*\[?\bS-\d+\b\]?\s*[:\-]??\s*", "", raw_q, flags=re.IGNORECASE).strip()
                item[quote_field] = cleaned_q

        # In-Place Quote Repair: the quoted sentence MUST contain the headword.
        # If the model cited the wrong sentence (headword physically present in the
        # source text but absent from the quote), deterministically swap in the
        # shortest authentic source sentence containing it — instead of burning LLM
        # retries that cannot reliably self-correct this failure mode.
        if task_type in ("vocabulary", "expressions"):
            headword = str(item.get("word") or item.get("pattern_formula") or item.get("target_word") or "").strip()
            if headword and quote_field and sentence_pool:
                rw_clean_word = _clean_core(re.sub(r"\[.*?\]|\(.*?\)", " ", headword))
                rw_tokens = [w for w in rw_clean_word.split() if w and w not in STOP_SLOTS]
                if rw_tokens:
                    def _rw_tok_in(text_core: str, tok: str) -> bool:
                        return _form_in_text(tok, text_core)

                    rw_current_core = _clean_core(str(item.get(quote_field) or ""))
                    if rw_current_core:
                        rw_matched = sum(1 for t in rw_tokens if _rw_tok_in(rw_current_core, t))
                        rw_min_needed = max(1, len(rw_tokens) // 2 + (1 if len(rw_tokens) % 2 == 1 else 0))
                        rw_needs_repair = rw_matched < rw_min_needed
                    else:
                        rw_needs_repair = True

                    # Only repair when the headword physically exists in the source;
                    # otherwise it is a pure hallucination and the prune gate handles it.
                    if rw_needs_repair and all(_rw_tok_in(core_src, t) for t in rw_tokens):
                        rw_best, rw_best_len = None, None
                        for _sid, _sent in sentence_pool.items():
                            _sent_clean = re.sub(r"^\s*\[?\bS-\d+\b\]?\s*[:\-]??\s*", "", _sent, flags=re.IGNORECASE).strip()
                            _sc = _clean_core(_sent_clean)
                            if not _sc or not all(_rw_tok_in(_sc, t) for t in rw_tokens):
                                continue
                            if rw_best_len is None or len(_sent_clean) < rw_best_len:
                                rw_best, rw_best_len = _sent_clean, len(_sent_clean)
                        if rw_best:
                            item[quote_field] = rw_best
                            pruned_flags.append(
                                f"🩹 In-Place Quote Repair: headword '{headword}' was absent from the quoted sentence; "
                                f"swapped in authentic source sentence: '{rw_best[:60]}...'"
                            )

        # Deterministic deduplication check
        if task_type == "grammar":
            raw_quote = str(item.get("quote", "")).strip()
            # 1. Physical sentence-level invariant dedup (Sentence pool ID)
            sent_id = LinguisticEngine.get_sentence_id(raw_quote, sentence_pool)
            if sent_id:
                sent_dedup_key = f"sent_id:{sent_id}"
                if sent_dedup_key in seen_dedup_keys:
                    pruned_flags.append(f"✂️ Pruned duplicate item: duplicate sentence {sent_id} ({raw_quote[:35]}...)")
                    continue
                seen_dedup_keys.add(sent_dedup_key)

            # 2. Dependency Syntax Signature Dedup (Macro domain + Dep type + Core anchor)
            if not item.get("category"):
                classified_cat = LinguisticEngine.classify_grammar_dependency(raw_quote)
                item["category"] = classified_cat or "Information Packaging"
            cat_val = str(item.get("category", "")).strip()
            fp = LinguisticEngine.extract_grammar_fingerprint(raw_quote, category=cat_val)
            if fp[1] != "generic" and fp[2]:
                fp_key = f"syntax_fp:{fp[0]}:{fp[1]}:{fp[2]}"
                if fp_key in seen_dedup_keys:
                    pruned_flags.append(f"✂️ Pruned duplicate item: homogeneous grammar pattern ({fp[0]} -> {fp[1]}:{fp[2]}): {raw_quote[:35]}...")
                    continue
                seen_dedup_keys.add(fp_key)

            formula_val = str(item.get("pattern_formula", "")).strip().lower()
            quote_val = raw_quote.lower()
            dedup_key = (formula_val, quote_val)
        else:
            word_val = str(item.get("word", "")).strip().lower()
            clean_word_val = re.sub(r'\[.*?\]|\(.*?\)', '', word_val).strip()
            dedup_key = (clean_word_val,)
            
            # Word-Family Deduplication Gate: reject items belonging to an already seen word family
            if task_type == "vocabulary" and clean_word_val:
                is_family_dup = False
                for seen_k in seen_dedup_keys:
                    if isinstance(seen_k, tuple) and seen_k and isinstance(seen_k[0], str):
                        prev_w = seen_k[0]
                        if prev_w and LinguisticEngine.are_same_word_family(clean_word_val, prev_w):
                            pruned_flags.append(f"✂️ Pruned duplicate word-family item '{clean_word_val}' (subsumed by '{prev_w}')")
                            is_family_dup = True
                            break
                if is_family_dup:
                    continue
            
        if any(dedup_key) and dedup_key in seen_dedup_keys:
            pruned_flags.append(f"✂️ Pruned duplicate item: {dedup_key[0]}")
            continue
        if any(dedup_key):
            seen_dedup_keys.add(dedup_key)

        word = str(item.get("word") or item.get("pattern_formula") or "").strip()
        quote = str(item.get("quoted_sentence") or item.get("quote") or "").strip()
        
        if not word and not quote:
            continue

        # Specialized pruning for grammar
        if task_type == "grammar":
            core_quote = _clean_core(quote)
            imitation = str(item.get("imitation_example") or "").strip()
            mistakes = str(item.get("common_mistakes") or "").strip()

            if not (core_quote and imitation and mistakes):
                missing_fields = []
                if not core_quote: missing_fields.append("quote")
                if not imitation: missing_fields.append("imitation_example")
                if not mistakes: missing_fields.append("common_mistakes")
                pruned_flags.append(f"✂️ Pruned incomplete grammar pattern '{word[:30]}' (missing required: {', '.join(missing_fields)})")
                continue

            # Auto-hydrate category if missing from simplified schema
            if not item.get("category"):
                classified_cat = LinguisticEngine.classify_grammar_dependency(quote)
                item["category"] = classified_cat or "Information Packaging"

            # Quote must exist in source text (verbatim or high n-gram coverage)
            quote_in_src = (core_quote in core_src or _ngram_coverage(core_quote, core_src, n=3) >= 0.80)
            if not quote_in_src:
                # Check segment coverage if ellipsis is present
                if "..." in quote or "…" in quote:
                    segments = [s.strip() for s in re.split(r'\.{3,}|…', quote) if s.strip()]
                    meaningful_segs = [s for s in segments if len(_clean_core(s).split()) >= 2]
                    if meaningful_segs and all(_clean_core(seg) in core_src or _ngram_coverage(_clean_core(seg), core_src, n=3) >= 0.80 for seg in meaningful_segs):
                        quote_in_src = True

            if not quote_in_src:
                pruned_flags.append(f"✂️ Pruned hallucinated grammar pattern '{word[:30]}' (quote not found in source text)")
                continue

            # Anchor verification for unprofiled/legacy items: literal functional anchors must exist in quote
            if not item.get("syntax_topic"):
                raw_lits = re.findall(r"[a-zA-Z0-9_\'/]+", re.sub(r"\[.*?\]|\(.*?\)", " ", word))
                expanded_quote = _clean_core(_expand_contractions(quote))
                quote_tokens = set(core_quote.split()) | set(expanded_quote.split())
                missing_anchors = []
                for lit in raw_lits:
                    alts = [a.strip() for a in lit.split("/") if a.strip()]
                    valid_alts = [a for a in alts if len(_clean_core(a)) >= 2 and _clean_core(a) not in COBUILD_POS_TOKENS]
                    if not valid_alts:
                        continue
                    if not any(_clean_core(a) in quote_tokens for a in valid_alts):
                        missing_anchors.append(lit)
                
                if missing_anchors:
                    # Level 1 In-Place Self-Healing: Attempt to synthesize canonical formula from quote
                    canonical_formula = LinguisticEngine.generate_cobuild_formula(quote, category=item.get("category"))
                    if canonical_formula and canonical_formula != word and canonical_formula != "[Subject] + [VP] + [Clause]":
                        item["pattern_formula"] = canonical_formula
                        pruned_flags.append(f"ℹ️ In-Place Self-Healing: Repaired mismatched formula '{word[:30]}' -> '{canonical_formula}' (anchors {missing_anchors} not in quote)")
                        word = canonical_formula
                    else:
                        pruned_flags.append(f"✂️ Pruned mismatched grammar pattern '{word[:30]}' (anchor '{missing_anchors[0]}' missing from quote)")
                        continue

            # Level 1 Code Gate: Auto-remap explicit category mismatches
            item, remap_notice = auto_remap_grammar_category(item)
            if remap_notice:
                pruned_flags.append(remap_notice)

            # Level 1 Code Gate: Auto-repair/canonicalize COBUILD formula if trivial or missing
            current_formula = str(item.get("pattern_formula", "")).strip()
            if not current_formula or "[" not in current_formula or current_formula.lower() in ("the + [noun] + [vp]", "[subject] + [vp]"):
                canonical_formula = LinguisticEngine.generate_cobuild_formula(quote, category=item.get("category"))
                if canonical_formula and canonical_formula != current_formula:
                    item["pattern_formula"] = canonical_formula
                    pruned_flags.append(f"ℹ️ In-Place Self-Healing: Standardized COBUILD formula -> '{canonical_formula}'")

            surviving_items.append(item)
            continue

        clean_word_no_slots = re.sub(r"\[.*?\]|\(.*?\)", " ", word)
        clean_word = _clean_core(clean_word_no_slots)
        clean_quote = _clean_core(quote)
        word_tokens = [w for w in clean_word.split() if w and w not in STOP_SLOTS]

        # Check A: Does the word appear in its claimed quote?
        def _tok_matches(tok: str, target_text: str) -> bool:
            return _form_in_text(tok, target_text)

        if word_tokens:
            min_needed = max(1, len(word_tokens) // 2 + (1 if len(word_tokens) % 2 == 1 else 0))
            word_in_quote = (sum(1 for w in word_tokens if _tok_matches(w, clean_quote)) >= min_needed)
            word_in_source = (sum(1 for w in word_tokens if _tok_matches(w, core_src)) >= min_needed)
        else:
            word_in_quote = (clean_word in clean_quote)
            word_in_source = (clean_word in core_src)

        # If it's absent from quote AND absent from entire source text -> pure fabrication
        if not word_in_quote and not word_in_source:
            pruned_flags.append(f"✂️ Pruned hallucinated item '{word}' (not present in quoted sentence or source text)")
            continue

        # If it's absent from its claimed quote but the quote itself is not in source
        if not word_in_quote:
            core_quote = _clean_core(quote)
            if core_quote and (core_quote not in core_src and _ngram_coverage(core_quote, core_src, n=3) < 0.85):
                pruned_flags.append(f"✂️ Pruned hallucinated item '{word}' (unmatched word in unverified quote)")
                continue

        # Level 1 In-Place Self-Healing: Canonical Lemmatization for legacy LLM single-word vocabulary
        if task_type == "vocabulary" and "word" in item:
            orig_word = str(item["word"]).strip()
            audit_str = str(item.get("design_audit", "")).upper()
            is_deterministic = "DETERMINISTIC" in audit_str
            if orig_word and not is_deterministic and "[" not in orig_word and "(" not in orig_word:
                lemmatized = LinguisticEngine.lemmatize_headword(orig_word, quote)
                if lemmatized and lemmatized.lower() != orig_word.lower():
                    item["word"] = lemmatized
                    pruned_flags.append(f"ℹ️ In-Place Self-Healing: Lemmatized headword '{orig_word}' -> '{lemmatized}'")

        surviving_items.append(item)

    if pruned_flags:
        parsed_data = dict(parsed_data)
        parsed_data[array_key] = surviving_items
    return parsed_data, pruned_flags

