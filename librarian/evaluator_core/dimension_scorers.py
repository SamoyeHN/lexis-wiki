"""Level 1 deterministic code gate scoring heuristics across 4 pedagogical quality dimensions:

1. Schema & Structural Adherence (W_SCHEMA = 25.0)
2. Source Faithfulness & Verbatim Verification (W_VERBATIM = 30.0)
3. Pedagogical & Slot-Filling Quality (W_PEDAGOGY = 25.0)
4. Uniqueness & Deduplication (W_UNIQUENESS = 20.0)
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

from ..config import config
from ..linguistics import LinguisticEngine
from .eval_helpers import (
    ALLOWED_GRAMMAR_SLOTS,
    COBUILD_POS_TOKENS,
    COMMON_IRREGULARS,
    FATAL_QA_FLAGS,
    GRAMMAR_SLOT_IDENTITIES,
    ITEM_KEYS,
    STOP_SLOTS,
    VALID_EXPR_POS_SET,
    VALID_POS_SET,
    W_PEDAGOGY,
    W_SCHEMA,
    W_UNIQUENESS,
    W_VERBATIM,
    _batch_target_conflict,
    _cefr_source_ceiling,
    _clean_core,
    _detect_task_type,
    _expand_contractions,
    _extract_items,
    _extract_json,
    _extract_source_content,
    _extract_target_skeletons,
    _extract_wordlist,
    _form_in_text,
    _formula_slot_identity,
    _inflected_variants_of,
    _is_hallucinated_quote,
    _ngram_coverage,
    _normalize_text,
    _safe_str,
    _source_token_variants,
    _wordlist_matches,
    auto_remap_grammar_category,
    canonical_formula_key,
    degenerate_grammar_slots,
)

def _score_schema(parsed: Any, raw_response: str = "", task_type: str = "") -> Tuple[Optional[float], List[str]]:
    """Dimension 1 (0–25). Always applicable."""
    if isinstance(parsed, dict):
        if len(parsed) == 0:
            return 0.0, ["❌ Valid JSON but empty object"]
        
        flags = []
        deduction = 0.0
        if raw_response and isinstance(raw_response, str):
            trimmed = raw_response.strip()
            # If closing brace was missing or raw string didn't end properly
            if trimmed.count("{") > trimmed.count("}") or trimmed.count("[") > trimmed.count("]"):
                deduction += 5.0
                flags.append("⚠️ Structural repair: unclosed braces/brackets in raw output")
        
        # Array element schema & type enforcement
        ARRAY_SPECS = {
            "mindmap": ("branches", 3, 5, "branch_name"),
            "vocabulary": ("vocabulary", 1, 50, "word"),
            "expressions": ("expressions", 1, 50, "word"),
            "grammar": ("grammar_patterns", 1, 20, "pattern_formula"),
            "summary": ("concepts", 1, 10, "concept_name"),
            "quiz": ("questions", 1, 50, "question"),
        }
        if task_type in ARRAY_SPECS:
            arr_key, min_items, max_items, required_field = ARRAY_SPECS[task_type]
            arr_val = parsed.get(arr_key)
            if not isinstance(arr_val, list):
                deduction += 15.0
                flags.append(f"❌ [INVALID_SCHEMA] Expected list for '{arr_key}', got {type(arr_val).__name__}")
            else:
                non_dict_count = sum(1 for el in arr_val if not isinstance(el, dict))
                if non_dict_count > 0:
                    deduction += 15.0
                    flags.append(f"❌ [INVALID_ITEM_TYPE] {non_dict_count}/{len(arr_val)} elements in '{arr_key}' are not objects (e.g. unparsed JSON strings)")
                valid_dict_count = len(arr_val) - non_dict_count
                if valid_dict_count < min_items:
                    deduction += 10.0
                    flags.append(f"❌ [INSUFFICIENT_ITEMS] '{arr_key}' contains only {valid_dict_count} valid objects (minimum required: {min_items})")

        score = max(0.0, W_SCHEMA - deduction)
        return score, flags
    return 0.0, ["❌ Invalid or missing JSON output"]


def _score_verbatim(items: List[Dict[str, Any]], task_type: str, user_prompt: str, context_prompt: str = "") -> Tuple[Optional[float], List[str]]:
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

    flags: List[str] = []
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
        if re.search(r"###\s*(?:TARGET|DETERMINISTIC|PASSAGE|SYLLABUS|OUTPUT|CORE\s+PEDAGOGICAL|JSON\s+SCHEMA)|Extract these exact|syllabus items|Do not skip or omit", quote, re.IGNORECASE):
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
                if LinguisticEngine.is_known_english_word(clean_word) is not True:
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
                if meaningful_segs and all(_clean_core(seg) in core_src or _ngram_coverage(_clean_core(seg), core_src, n=3) >= 0.80 for seg in meaningful_segs):
                    is_verbatim = True

        if is_verbatim:
            matches += 1
        else:
            flags.append(f"❌ Non-verbatim quote detected: '{quote[:40]}...'")
            
    if checks == 0:
        return 0.0, ["❌ No items to evaluate for source faithfulness"]
    return max(0.0, round((matches / checks) * W_VERBATIM, 1)), flags



def _score_pedagogy(items: List[Dict[str, Any]], task_type: str, user_prompt: str = "") -> Tuple[Optional[float], List[str]]:
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
    translation_bp_targets: List[str] = []
    if task_type == "quiz" and user_prompt:
        grammar_ref_split = re.split(r"\bGRAMMAR REFERENCE:\b", user_prompt, flags=re.IGNORECASE)
        vocab_section = grammar_ref_split[0]
        for m in re.finditer(r"## \[\[(.*?)\]\]", vocab_section):
            prompt_headwords.add(m.group(1).strip().lower())
        if not prompt_headwords:
            for m in re.finditer(r"-\s*Target:\s*([^\n\r]+)", vocab_section):
                prompt_headwords.add(m.group(1).strip().lower())
        # Support Target Word from Authentic Corpus Blueprint or Translation Micro-Task Blueprint
        for m in re.finditer(r"-\s*Target\s+(?:Word|Keyword|Collocation\s*/\s*Lexical Frame):\s*([^\n\r(]+)", user_prompt, re.IGNORECASE):
            w = m.group(1).strip().lower()
            if w:
                prompt_headwords.add(w)
        translation_bp_targets = [
            m.group(1).strip().lower()
            for m in re.finditer(r"-\s*Target\s+(?:Keyword|Collocation\s*/\s*Lexical Frame):\s*([^\n\r(]+)", user_prompt, re.IGNORECASE)
            if m.group(1).strip()
        ]

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

            # Quote provenance as rendered by the blueprint: a licensed quote is corpus evidence
            # strong enough to be modelled, a display-only quote may never be reused at all.
            m_lq = re.search(
                r"Curriculum Quote \(licensed\):\s*'(.+?)'\s*$", block, re.IGNORECASE | re.MULTILINE
            )
            m_cq = re.search(
                r"Curriculum Quote \(display-only.*?\):\s*'(.+?)'\s*$", block, re.IGNORECASE | re.MULTILINE
            )
            licensed_quote_val = m_lq.group(1).strip() if m_lq else ""
            candidate_quote_val = m_cq.group(1).strip() if m_cq else ""

            # the blueprint also prescribes the four options and the answer
            # index; the writer is only allowed to author stem, explanation and audit.
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

    # the formulas the lexicon pre-extracted, indexed by a presentation-
    # insensitive key. An item consumes a skeleton when it copies it; a same-category
    # skeleton left unconsumed means the model rewrote the formula it was handed.
    grammar_skeletons: List[Dict[str, str]] = []
    skeleton_keys: Dict[str, List[Dict[str, str]]] = {}
    if task_type == "grammar" and user_prompt:
        grammar_skeletons = _extract_target_skeletons(user_prompt, "grammar")
        from ..processor import WikiProcessor
        for _sk in grammar_skeletons:
            _raw_formula = str(_sk.get("formula", ""))
            # Index the skeleton both as mined and as the pipeline would canonicalize it:
            # the pedagogy branch unwraps literal brackets ('[Main Clause]' -> '[S]')
            # before comparing, so both presentations are the same formula.
            _skel_keys = {
                canonical_formula_key(_raw_formula),
                canonical_formula_key(WikiProcessor.unwrap_literal_brackets(_raw_formula)),
            }
            for _key in _skel_keys:
                if _key:
                    skeleton_keys.setdefault(_key, []).append(_sk)
    consumed_skeletons: set = set()

    def _align_grammar_formula(pattern: str, category: str) -> Tuple[bool, str]:
        """(copied a skeleton formula?, the skeleton formula this item should have copied).

        A formula that equals a mined skeleton is aligned even if another item already
        delivered that skeleton — duplicated coverage is the coverage gate's business,
        not an alignment defect. Only a formula that equals NO mined skeleton is a rewrite.
        """
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
        if task_type == "vocabulary":
            word = _safe_str(item.get("word"))
            pos = _safe_str(item.get("part_of_speech")).lower()
            quote = _safe_str(item.get("quoted_sentence"))
            is_expression = LinguisticEngine.is_multiword_expression(word)
            # A label is missing, or a multi-word unit carries the part of speech of one
            # of its tokens ('tap into' -> 'verb') instead of its expression type.
            needs_pos = bool(word) and (
                not pos or (is_expression and pos not in LinguisticEngine.EXPRESSION_TYPE_LABELS)
            )
            if needs_pos:
                # A multi-word unit is typed as an expression ('collocation', 'phrasal
                # verb', 'set phrase', 'idiom'); only a single headword is typed by the
                # parse of its quote.
                if is_expression:
                    pos = LinguisticEngine.classify_expression_type(word, quote) or "collocation"
                else:
                    pos = LinguisticEngine.determine_contextual_pos(word, quote)
                item["part_of_speech"] = pos
            definition = _safe_str(item.get("definition"))
            example = _safe_str(item.get("example_usage"))
            checks += 1
            
            # Allow words/phrases with slots, slashes, hyphens, brackets, parentheses, apostrophes
            valid_word = bool(re.search(r"^[A-Za-z\s\-'\[\]\(\)/]+$", word) and len(word) >= 2)
            # A multi-word row may carry an expression type, or (as a fallback the healing
            # above could not decide) the part of speech of a fixed frame it is taught as.
            allowed_pos = (VALID_POS_SET | VALID_EXPR_POS_SET) if is_expression else VALID_POS_SET
            valid_pos = pos in allowed_pos
            has_example = bool(example)
            is_distinct_example = bool(has_example and _clean_core(example) != _clean_core(quote))
            # 'A core academic term functioning as a noun.' is the engine's own filler,
            # not a definition. It fails the check the same way a missing one does.
            invented_definition = bool(definition) and LinguisticEngine.is_boilerplate_definition(definition)

            if valid_word and valid_pos and definition and not invented_definition and is_distinct_example:
                passes += 1
            else:
                reasons = []
                if not valid_word: reasons.append(f"invalid headword '{word}'")
                if not valid_pos: reasons.append(f"invalid PoS '{pos}'")
                if not definition: reasons.append("missing definition")
                elif invented_definition: reasons.append("invented fallback definition (no lexicon host grounds it)")
                if not has_example:
                    reasons.append("missing example_usage")
                elif not is_distinct_example:
                    reasons.append("example_usage is an unoriginal duplicate of quoted_sentence")
                flags.append(f"⚠️ Vocabulary item '{word}' failed pedagogy check: {', '.join(reasons)}")
        elif task_type == "expressions":
            word = _safe_str(item.get("word"))
            definition = _safe_str(item.get("definition"))
            example = _safe_str(item.get("example_usage"))
            quote = _safe_str(item.get("quoted_sentence"))
            checks += 1

            is_multiword = bool(re.search(r"\[.+?\]|one's", word, re.IGNORECASE) or len(word.split()) > 1)
            is_trivial = word.lower() in ("talk", "listen", "turn", "watch", "sit down", "talk to", "listen to", "look at")
            has_example = bool(example)
            is_distinct_example = bool(has_example and _clean_core(example) != _clean_core(quote))
            # the cascade's invented filler is not a definition. An expression the lexicon
            # does not define is a defect to be reported, not a row to be shipped.
            invented_definition = bool(definition) and LinguisticEngine.is_boilerplate_definition(definition)

            if is_multiword and not is_trivial and definition and not invented_definition and is_distinct_example:
                passes += 1
            else:
                reasons = []
                if not is_multiword or is_trivial:
                    reasons.append("lacks multi-word/slot form or is too basic")
                if not definition:
                    reasons.append("missing definition")
                elif invented_definition:
                    reasons.append("invented fallback definition (no lexicon host grounds it)")
                if not has_example:
                    reasons.append("missing example_usage")
                elif not is_distinct_example:
                    reasons.append("example_usage is an unoriginal duplicate of quoted_sentence")
                flags.append(f"⚠️ Expression '{word}' failed pedagogy check: {', '.join(reasons)}")
        elif task_type == "grammar":
            pattern = _safe_str(item.get("pattern_formula"))
            audit = _safe_str(item.get("design_audit"))
            quote = _safe_str(item.get("quote"))
            category = _safe_str(item.get("category"))
            if not category and quote:
                category = LinguisticEngine.classify_grammar_dependency(quote) or "Information Packaging"
                item["category"] = category
            checks += 1
            
            reasons = []
            imitation = _safe_str(item.get("imitation_example"))
            if not audit:
                reasons.append("missing design_audit")
            if not pattern:
                reasons.append("missing pattern_formula")
            if not imitation:
                reasons.append("missing imitation_example")
            else:
                if quote and _clean_core(imitation) == _clean_core(quote):
                    # an imitation example that IS the passage sentence teaches
                    # no transfer — the student copies instead of re-using the pattern.
                    reasons.append("imitation_example duplicates the source quote")
                    flags.append(
                        f"❌ Grammar pattern '{pattern[:40]}': imitation_example duplicates the source quote "
                        "(an imitation must be a new sentence, not the quoted one)"
                    )
            if not _safe_str(item.get("common_mistakes")):
                reasons.append("missing common_mistakes")
            
            # Auto-unwrap accidental brackets around literal functional words (e.g. [it], [that], [if])
            from ..processor import WikiProcessor
            pattern = WikiProcessor.unwrap_literal_brackets(pattern)

            # Minimal Anti-Triviality Gate: penalize simple patterns whose ONLY anchor is a standalone conversational filler/coordinator (e.g. 'But [S]', 'And [S]')
            slot_count = len(re.findall(r"\[.*?\]", pattern))
            lits = [
                w for w in re.sub(r"\[.*?\]|\(.*?\)|[+,/]", " ", pattern).split()
                if len(_clean_core(w)) >= 2
            ]
            TRIVIAL_ANCHOR_WORDS = frozenset({"but", "and", "so", "or", "of", "course", "well", "now", "here"})
            TRIVIAL_PHRASES = frozenset({"but", "and", "so", "or", "of course", "here is", "heres"})
            clean_lits = [re.sub(r"[^\w\s]", "", l).lower().strip() for l in lits if re.sub(r"[^\w\s]", "", l).strip()]
            combined_lit_phrase = " ".join(clean_lits)
            # Reject un-abstracted trivial formulas anchored only by sentence-connectors (e.g. 'However, [S]', 'Therefore, [Clause]')
            if slot_count <= 1:
                if (combined_lit_phrase in TRIVIAL_PHRASES) or (clean_lits and all(l in TRIVIAL_ANCHOR_WORDS for l in clean_lits)):
                    reasons.append(f"trivial formula anchored only by conversational filler or conjunction: '{pattern}'")
                elif clean_lits and any(l in {"however", "therefore", "moreover", "furthermore", "meanwhile", "nevertheless", "in fact", "actually"} for l in clean_lits):
                    reasons.append(f"un-abstracted superficial pattern anchored only on discourse adverbial connector: '{pattern}'")

            # Structural Consistency: Literal keywords in formula MUST exist in quote AND imitation_example
            # (e.g. if formula requires 'while', 'not... but...', 'which', or '[V-ing Phrase]',
            # both the quoted text and the pedagogical imitation sentence must physically instantiate them)
            clean_quote = _clean_core(quote)
            clean_quote_raw = quote.lower()
            clean_imitation = _clean_core(imitation)
            clean_imitation_raw = imitation.lower()
            if pattern:
                # Check for explicit literal connector / subordinator words in formula (supporting slash alternatives e.g. is/was)
                raw_lits = re.findall(r"[a-zA-Z0-9_\'/-]+", re.sub(r"\[.*?\]|\(.*?\)", " ", pattern))
                expanded_quote = _clean_core(_expand_contractions(quote))
                expanded_imitation = _clean_core(_expand_contractions(imitation))
                quote_tokens = set(clean_quote.split()) | set(expanded_quote.split())
                imitation_tokens = set(clean_imitation.split()) | set(expanded_imitation.split())
                missing_anchors_quote = []
                missing_anchors_imitation = []
                for lit in raw_lits:
                    alts = [a.strip() for a in lit.split("/") if a.strip()]
                    expanded_alts = []
                    for a in alts:
                        if a.lower() in ("to-v", "to-inf"):
                            expanded_alts.append("to")
                        elif len(_clean_core(a)) >= 2 and _clean_core(a) not in COBUILD_POS_TOKENS and a.lower() not in COBUILD_POS_TOKENS:
                            expanded_alts.append(a)
                    valid_alts = expanded_alts
                    if not valid_alts:
                        continue
                    quote_has_anchor = any(_clean_core(a) in quote_tokens or _form_in_text(_clean_core(a), clean_quote) for a in valid_alts)
                    if not quote_has_anchor and any(a.lower() in ("is", "was", "were", "are", "be", "been") for a in valid_alts):
                        if re.search(r"\b(?:is|was|were|are|be|been|'s|'re)\b", clean_quote, re.IGNORECASE):
                            quote_has_anchor = True
                    if not quote_has_anchor:
                        missing_anchors_quote.append(lit)
                    if imitation and len(imitation.split()) >= 3:
                        is_instantiated = any(_clean_core(a) in imitation_tokens or _form_in_text(_clean_core(a), clean_imitation) for a in valid_alts)
                        # copula paradigm (is/was/were/are/be/been)
                        if not is_instantiated and any(a.lower() in ("is", "was", "were", "are", "be", "been") for a in valid_alts):
                            if re.search(r"\b(?:is|was|were|are|be|been|'s|'re)\b", clean_imitation, re.IGNORECASE):
                                is_instantiated = True
                        # propositional encapsulation interpretive verbs (show/suggest/mean/indicate/underscore)
                        # allow cross-verb instantiation within the closed interpretive verb paradigm
                        if not is_instantiated and any(re.match(r"^(?:mean[st]?|suggest(?:s|ed)?|indicat(?:es|ed)|show(?:s|ed|n)?|demonstrat(?:es|ed)|prov(?:es|ed|en)|impl(?:ies|ied)|reveal(?:s|ed)?|underscor(?:es|ed))$", a, re.IGNORECASE) for a in valid_alts):
                            if re.search(r"\b(?:mean[st]?|suggest(?:s|ed)?|indicat(?:es|ed)|show(?:s|ed|n)?|demonstrat(?:es|ed)|prov(?:es|ed|en)|impl(?:ies|ied)|reveal(?:s|ed)?|underscor(?:es|ed))\b", imitation, re.IGNORECASE):
                                is_instantiated = True
                        # optional complementizer 'that' in sentential relative ', which + interpretive verb' when verb takes NP object
                        if not is_instantiated and any(a.lower() == "that" for a in valid_alts) and re.search(r",\s*which\b", pattern, re.IGNORECASE):
                            if re.search(r",\s*which\s+(?:mean[st]?|suggest(?:s|ed)?|indicat(?:es|ed)|show(?:s|ed|n)?|demonstrat(?:es|ed)|prov(?:es|ed|en)|impl(?:ies|ied)|reveal(?:s|ed)?|underscor(?:es|ed))\b", imitation, re.IGNORECASE):
                                is_instantiated = True
                        # concessive subordinator paradigm (including 'even if' / 'even though')
                        if not is_instantiated and (
                            any(a.lower() in ("although", "though", "even", "while", "whereas") for a in valid_alts)
                            or (any(a.lower() == "if" for a in valid_alts) and re.search(r"\beven\s+if\b", pattern, re.IGNORECASE))
                        ):
                            if re.search(r"\b(?:although|though|even though|even if|while|whereas)\b", imitation, re.IGNORECASE):
                                is_instantiated = True
                        # abstract shell-noun paradigm
                        if not is_instantiated and any(a.lower() in ("fact", "idea", "belief", "assumption", "notion", "claim", "possibility", "view", "evidence", "news", "impression", "argument", "thought", "hope", "fear", "doubt", "contention", "recognition", "realization", "premise", "hypothesis", "consensus", "conclusion", "paradox", "irony", "conviction") for a in valid_alts):
                            if re.search(r"\b(?:fact|idea|belief|assumption|notion|claim|possibility|view|evidence|news|impression|argument|thought|hope|fear|doubt|contention|recognition|realization|premise|hypothesis|consensus|conclusion|paradox|irony|conviction)\b", imitation, re.IGNORECASE):
                                is_instantiated = True
                        # complex-transitive & causative verb paradigm
                        if not is_instantiated and any(a.lower() in ("make", "makes", "made", "find", "finds", "found", "keep", "keeps", "kept", "render", "renders", "rendered", "deem", "deems", "deemed", "consider", "considers", "considered", "leave", "leaves", "left", "let", "lets", "have", "has", "had") for a in valid_alts):
                            if re.search(r"\b(?:make|makes|made|making|find|finds|found|finding|keep|keeps|kept|keeping|render|renders|rendered|rendering|deem|deems|deemed|consider|considers|considered|leave|leaves|left|leaving|let|lets|letting|have|has|had|having)\b", imitation, re.IGNORECASE):
                                is_instantiated = True
                        # conjunctive adverb transition paradigm
                        if not is_instantiated and any(a.lower() in ("however", "nevertheless", "nonetheless", "conversely") for a in valid_alts):
                            if re.search(r"\b(?:however|nevertheless|nonetheless|conversely)\b", imitation, re.IGNORECASE):
                                is_instantiated = True
                        # restrictive relative pronoun paradigm ('which'/'that'/'who')
                        if not is_instantiated and any(a.lower() in ("which", "that", "who", "whom") for a in valid_alts) and re.match(r"^\[np\]\s*\+\s*(?:which|that|who)", pattern.strip(), re.IGNORECASE):
                            if re.search(r"\b(?:which|that|who|whom)\b", imitation, re.IGNORECASE):
                                is_instantiated = True
                        if not is_instantiated:
                            missing_anchors_imitation.append(lit)

                if missing_anchors_quote:
                    reasons.append(f"formula anchor '{', '.join(missing_anchors_quote)}' ungrounded in quote")
                if missing_anchors_imitation:
                    # An imitation sentence must instantiate its own formula
                    reasons.append(f"imitation_example does not instantiate formula anchor '{', '.join(missing_anchors_imitation)}'")
                    flags.append(
                        f"❌ Grammar pattern '{pattern[:40]}': imitation_example does not instantiate formula anchor "
                        f"'{', '.join(missing_anchors_imitation)}' (an imitation must instantiate the formula it models)"
                    )

                # Check for participle requirement:
                # 1) Pure [V-ing] / [V-ing Phrase] slot without slash alternatives (e.g. not [V-ing/NP] or [V-ing / V-ed])
                ING_EXCLUDE = frozenset({"morning", "evening", "thing", "something", "nothing", "everything", "anything", "during", "ceiling", "ring", "spring", "king", "wing", "sing", "bring", "cling", "fling", "sting", "string", "swing"})
                pat_low = pattern.lower()
                requires_strict_ving = bool(re.search(r"\[v-ing(?:\s+phrase)?\]", pat_low))
                requires_any_participle = ("[v-ing" in pat_low or "[participle" in pat_low) and not requires_strict_ving
                if requires_strict_ving:
                    ing_tokens_quote = [w for w in re.findall(r"\b(?:being|doing|going|[a-zA-Z]{3,}ing)\b", clean_quote_raw) if w.lower() not in ING_EXCLUDE]
                    if not ing_tokens_quote:
                        reasons.append("formula requires '[V-ing]' but quote contains no participle verb ungrounded in quote")
                    if imitation:
                        ing_tokens_imitation = [w for w in re.findall(r"\b(?:being|doing|going|[a-zA-Z]{3,}ing)\b", clean_imitation_raw) if w.lower() not in ING_EXCLUDE]
                        if not ing_tokens_imitation:
                            reasons.append("formula requires '[V-ing]' but imitation_example contains no participle verb")
                elif requires_any_participle:
                    # Slash-alternative slots like [V-ing / V-ed Phrase], [Participle/Adj/PrepP], or [V-ing/NP]
                    # accept -ing, -ed/-en past participles, adjectives, or prepositional/noun complements.
                    pass

            # Ceiling Gate — imitation_example may not exceed the passage CEFR ceiling
            if src_cefr_level and imitation:
                over = LinguisticEngine.over_ceiling_tokens(imitation, src_cefr_level, mode="text", allow=src_passage_words)
                if over:
                    offenders = ", ".join(f"'{w}' ({lvl})" for w, lvl in over[:3])
                    band = LinguisticEngine.cefr_ceiling(src_cefr_level, mode="text")
                    reason_msg = f"imitation_example exceeds the CEFR {band} ceiling of the CEFR {src_cefr_level} source passage: {offenders}"
                    reasons.append(reason_msg)
                    flags.append(f"❌ Grammar pattern '{pattern[:40]}': {reason_msg}")

            # Degenerate Slot Gate — '[NP/NP]' names the same constituent
            # twice, teaches nothing, and is never mined by the lexicon.
            degenerate_slots = degenerate_grammar_slots(pattern)
            if degenerate_slots:
                reasons.append(f"degenerate formula slot {degenerate_slots} names the same constituent twice")
                flags.append(
                    f"❌ Grammar pattern '{pattern[:40]}': degenerate formula slot {degenerate_slots} "
                    "(a slot may not be offered to itself)"
                )

            # Skeleton Alignment Gate — the formula the lexicon mined is the
            # formula the item must ship. Casing, '+' separators and slot synonyms are
            # presentation; a rewritten structure is a deviation.
            aligned, wanted_formula = _align_grammar_formula(pattern, category)
            if not aligned and wanted_formula:
                reasons.append(f"pattern_formula deviates from the pre-extracted skeleton '{wanted_formula}'")
                flags.append(
                    f"❌ Grammar pattern '{pattern[:40]}' deviates from the pre-extracted skeleton "
                    f"formula '{wanted_formula}'"
                )
            elif not aligned:
                flags.append(
                    f"⚠️ Grammar pattern '{pattern[:40]}' matches none of the pre-extracted skeleton formulas"
                )


            # Auto-Remap explicit category mismatches if not already healed
            _, remap_notice = auto_remap_grammar_category(item)
            if remap_notice:
                flags.append(remap_notice)
                category = str(item.get("category", "")).strip()

            # Four Macro Functional Domains deterministic checks (for unprofiled / legacy items lacking syntax_topic)
            # 1. Rhetoric & Emphasis: Check for inversion markers, genuine cleft relative clauses, or parallelism coordination
            if not item.get("syntax_topic") and category == "Rhetoric & Emphasis":
                has_inversion_trigger = bool(re.search(r"\b(?:only|never|hardly|scarcely|seldom|rarely|little|not only|neither|nor|no sooner)\b", quote, re.IGNORECASE))
                has_cleft = bool(re.search(r"\bIt\s+(?:is|was|were|'s|has\s+been|had\s+been)\b", quote, re.IGNORECASE) and re.search(r"\b(?:that|who|whom|which)\b", quote, re.IGNORECASE))
                has_wh_cleft = bool(re.search(r"\bWhat\s+[a-z0-9_']+\s+(?:is|was|were)\b", quote, re.IGNORECASE))
                has_parallel_coordination = bool(
                    re.search(r"\b(?:not only\b.*?\bbut\b|either\b.*?\bor\b|neither\b.*?\bnor\b|both\b.*?\band\b|not\s+.*?\s*,\s*but\b)\b", quote, re.IGNORECASE)
                    or ";" in quote or "," in quote
                )
                if not (has_inversion_trigger or has_cleft or has_wh_cleft or has_parallel_coordination):
                    reasons.append("category 'Rhetoric & Emphasis' assigned to quote lacking rhetorical markers (inversion, cleft focus, or structural balance)")

            # 2. Logic & Stance: Check for conditional, concessive, epistemic hedging, or stance modal assertions
            elif not item.get("syntax_topic") and category == "Logic & Stance":
                has_cond = bool(re.search(r"\b(?:if|unless|provided\s+that|supposing|as\s+long\s+as|had\s+\w+\s+\w+|should\s+\w+\s+\w+|were\s+\w+\s+\w+)\b", quote, re.IGNORECASE))
                has_concessive = bool(re.search(r"\b(?:although|though|even though|even if|while|whereas|despite|in spite of)\b", quote, re.IGNORECASE))
                has_stance_or_hedging = bool(re.search(r"\b(?:can|cannot|can't|could|may|might|must|should|would|will|suggest|indicate|appear|seem|likely|probably|presumably|arguably|tends?\s+to)\b", quote, re.IGNORECASE))
                if not (has_cond or has_concessive or has_stance_or_hedging):
                    reasons.append("category 'Logic & Stance' assigned to quote lacking logical condition, concessive linker, or epistemic/stance modal assertion")

            # 3. Information Packaging: Check for non-finite verb forms, nominalization, dummy-it extraposition, relative clause, complex-transitive/causative, or passive
            elif not item.get("syntax_topic") and category == "Information Packaging":
                ING_NON_VERBS = frozenset({"morning", "evening", "thing", "something", "nothing", "everything", "anything", "ring", "spring", "king", "wing", "sing", "during", "ceiling"})
                ing_tokens = [w.lower() for w in re.findall(r"\b(?:being|doing|going|[a-zA-Z]{3,}ing)\b", quote) if w.lower() not in ING_NON_VERBS]
                has_participle = bool(ing_tokens or re.search(r"\b(?:having\s+\w+(?:ed|en|t)|surrounded|driven|given|taken|seen|known|based|built|born|located|situated|reminded|confronted|faced|with\s+\w+)\b", quote, re.IGNORECASE))
                TO_NOUNS = frozenset({"school", "work", "bed", "church", "college", "prison", "jail", "court", "sea", "town", "home", "market", "class", "them", "him", "her", "us", "me", "you", "it", "this", "that", "these", "those"})
                to_matches = [m.group(1).lower() for m in re.finditer(r"\bto\s+([a-z]{2,})\b", quote, re.IGNORECASE)]
                has_infinitive_clause = any(word not in TO_NOUNS for word in to_matches)
                has_dummy_it = bool(re.search(r"\bit(?:'s|\s+is|\s+was|\s+has\s+been)\s+(?:[a-z]{4,}\s+)?(?:that|to\s+[a-z]+)\b", quote, re.IGNORECASE))
                has_nominalization = bool(re.search(r"\b[a-z]{3,}(?:tion|sion|ment|ance|ence|ity|ness)\b", quote, re.IGNORECASE))
                has_elaborative_relative = bool(re.search(r",\s*which\s+[a-z]+|\b(?:which|that|who|whom)\s+[a-z]+", quote, re.IGNORECASE))
                has_complex_trans_or_passive = bool(re.search(r"\b(?:make|makes|made|find|finds|found|keep|keeps|kept|render|renders|rendered|deem|deems|deemed|let|lets|have|has|had|is|are|was|were|be|been)\b", quote, re.IGNORECASE))
                if not (has_participle or has_infinitive_clause or has_dummy_it or has_nominalization or has_elaborative_relative or has_complex_trans_or_passive):
                    reasons.append("category 'Information Packaging' assigned to quote lacking non-finite clauses, evaluative it-extraposition, dense nominalization, or elaborative clause")

            # 4. Cohesion & Framing: Check for shell nouns, complement that-clauses, or encapsulation
            elif not item.get("syntax_topic") and category == "Cohesion & Framing":
                has_shell_frame = bool(re.search(r"\b(?:the\s+(?:fact|idea|notion|reason|belief|claim|argument|possibility|question|view|conclusion)\s+that|this\s+(?:mean[st]?|suggest(?:s|ed)?|indicat(?:es|ed)|led\s+to|leads\s+to)|the\s+extent\s+to\s+which)\b", quote, re.IGNORECASE))
                has_anaphoric_encapsulation = bool(re.search(r"\b(?:this|these|such)\s+[a-z]{4,}\b", quote, re.IGNORECASE) or bool(re.search(r"\bthat\s+is\s+why\b", quote, re.IGNORECASE)))
                # Closed set of interpretive verbs across all inflectional forms (base, 3sg, past, participle) + optional modal auxiliaries
                INTERPRETIVE_VERBS_REGEX = (
                    r"(?:(?:would|could|might|may|can|will)\s+)?"
                    r"(?:mean[st]?|suggest(?:s|ed)?|indicat(?:es|ed)|show(?:s|ed|n)?|demonstrat(?:es|ed)|prov(?:es|ed|en)|impl(?:ies|ied)|reveal(?:s|ed)?)"
                )
                has_which_propositional = bool(re.search(rf",\s*which\s+{INTERPRETIVE_VERBS_REGEX}\b", quote, re.IGNORECASE))
                has_rel_frame = bool(re.search(r"\b(?:in\s+which|by\s+which|through\s+which|whereby)\b", quote, re.IGNORECASE))
                if not (has_shell_frame or has_anaphoric_encapsulation or has_which_propositional or has_rel_frame):
                    reasons.append("category 'Cohesion & Framing' assigned to quote lacking abstract shell frame, prepositional relative, or discourse encapsulation")

            # Boundary tolerance (any-macro-domain rule): if the strict per-category gate above
            # rejected the model's chosen category, but the quote structurally fits ANOTHER of
            # the 4 macro functional domains, it is a defensible boundary case (boundary
            # sentences sit on two domains and have no unique label). Tolerate it instead of
            # rejecting — only reject quotes that match NO domain (kept as the original fatal
            # reason, so genuinely marker-less quotes are still quarantined).
            if any(r.startswith("category '") for r in reasons):
                _ING_EXCLUDE = frozenset({"morning", "evening", "thing", "something", "nothing", "everything", "anything", "ring", "spring", "king", "wing", "sing", "during", "ceiling"})
                _TO_EXCLUDE = frozenset({"school", "work", "bed", "church", "college", "prison", "jail", "court", "sea", "town", "home", "market", "class", "them", "him", "her", "us", "me", "you", "it", "this", "that", "these", "those"})
                _fits_rhetoric = (
                    re.search(r"\b(?:only|never|hardly|scarcely|seldom|rarely|little|not only|neither|nor|no sooner)\b", quote, re.IGNORECASE)
                    or (re.search(r"\bIt\s+(?:is|was|were|'s|has\s+been|had\s+been)\b", quote, re.IGNORECASE) and re.search(r"\b(?:that|who|whom|which)\b", quote, re.IGNORECASE))
                    or re.search(r"\bWhat\s+[a-z0-9_']+\s+(?:is|was|were)\b", quote, re.IGNORECASE)
                    or re.search(r"\b(?:not only\b.*?\bbut\b|either\b.*?\bor\b|neither\b.*?\bnor\b|both\b.*?\band\b|not\s+.*?\s*,\s*but\b)\b", quote, re.IGNORECASE)
                    or ";" in quote or "," in quote
                )
                _fits_logic = (
                    re.search(r"\b(?:if|unless|provided\s+that|supposing|as\s+long\s+as|had\s+\w+\s+\w+|should\s+\w+\s+\w+|were\s+\w+\s+\w+)\b", quote, re.IGNORECASE)
                    or re.search(r"\b(?:although|though|even though|even if|while|whereas|despite|in spite of)\b", quote, re.IGNORECASE)
                    or re.search(r"\b(?:can|cannot|can't|could|may|might|must|should|would|will|suggest|indicate|appear|seem|likely|probably|presumably|arguably|tends?\s+to)\b", quote, re.IGNORECASE)
                )
                _ing_tokens = [w.lower() for w in re.findall(r"\b(?:being|doing|going|[a-zA-Z]{3,}ing)\b", quote) if w.lower() not in _ING_EXCLUDE]
                _to_words = [m.group(1).lower() for m in re.finditer(r"\bto\s+([a-z]{2,})\b", quote, re.IGNORECASE)]
                _fits_info = (
                    _ing_tokens
                    or re.search(r"\b(?:having\s+\w+(?:ed|en|t)|surrounded|driven|given|taken|seen|known|based|built|born|located|situated|reminded|confronted|faced|with\s+\w+)\b", quote, re.IGNORECASE)
                    or any(w not in _TO_EXCLUDE for w in _to_words)
                    or re.search(r"\bit(?:'s|\s+is|\s+was|\s+has\s+been)\s+(?:[a-z]{4,}\s+)?(?:that|to\s+[a-z]+)\b", quote, re.IGNORECASE)
                    or re.search(r"\b[a-z]{3,}(?:tion|sion|ment|ance|ence|ity|ness)\b", quote, re.IGNORECASE)
                    or re.search(r",\s*which\s+[a-z]+|\b(?:which|that|who|whom)\s+[a-z]+", quote, re.IGNORECASE)
                    or re.search(r"\b(?:make|makes|made|find|finds|found|keep|keeps|kept|render|renders|rendered|deem|deems|deemed|let|lets|have|has|had|is|are|was|were|be|been)\b", quote, re.IGNORECASE)
                )
                _fits_cohesion = (
                    re.search(r"\b(?:the\s+(?:fact|idea|notion|reason|belief|claim|argument|possibility|question|view|conclusion)\s+that|this\s+(?:mean[st]?|suggest(?:s|ed)?|indicat(?:es|ed)|led\s+to|leads\s+to)|the\s+extent\s+to\s+which)\b", quote, re.IGNORECASE)
                    or re.search(r"\b(?:this|these|such)\s+[a-z]{4,}\b", quote, re.IGNORECASE)
                    or re.search(r"\bthat\s+is\s+why\b", quote, re.IGNORECASE)
                    or re.search(r",\s*which\s+(?:(?:would|could|might|may|can|will)\s+)?(?:mean[st]?|suggest(?:s|ed)?|indicat(?:es|ed)|show(?:s|ed|n)?|demonstrat(?:es|ed)|prov(?:es|ed|en)|impl(?:ies|ied)|reveal(?:s|ed)?)\b", quote, re.IGNORECASE)
                    or re.search(r"\b(?:in\s+which|by\s+which|through\s+which|whereby)\b", quote, re.IGNORECASE)
                )
                if _fits_rhetoric or _fits_logic or _fits_info or _fits_cohesion:
                    reasons[:] = [r for r in reasons if not r.startswith("category '")]
                    flags.append("ℹ️ Boundary pattern: chosen category rejected by strict gate, but quote fits another macro-domain — tolerated (any-macro-domain rule)")

            # Quote cleanliness warning: trailing ellipsis
            if quote.rstrip().endswith(("...", "…")):
                flags.append(f"⚠️ Quote ends with trailing ellipsis: '{quote[:40]}...'")

            if not reasons:
                passes += 1
            else:
                flags.append(f"⚠️ Grammar pattern '{pattern[:40]}' failed pedagogy check: {', '.join(reasons)}")
        elif task_type == "quiz":
            options = item.get("options", [])
            idx = item.get("correct_answer_index")
            explanation = _safe_str(item.get("explanation") or item.get("diagnostic_critique"))
            question = _safe_str(item.get("question") or item.get("translated_sentence") or item.get("source_sentence"))
            target = _safe_str(item.get("target_word") or item.get("target_keyword") or item.get("word")).strip().lower()
            is_translation = bool(item.get("translated_sentence") or item.get("source_sentence") or item.get("english_skeleton") or item.get("target_keyword") or item.get("idiomatic_translation"))
            skeleton = _safe_str(item.get("english_skeleton"))
            is_comparative_translation = bool(item.get("idiomatic_translation") and item.get("flawed_translation"))
            if is_comparative_translation:
                # Pruned comparative translation schema strips `options`, `correct_answer_index`,
                # `explanation`, and `target_keyword` from LLM output and hydrates them post-evaluation.
                if not options:
                    options = [
                        _safe_str(item.get("idiomatic_translation")),
                        _safe_str(item.get("flawed_translation")),
                    ]
                if idx is None:
                    idx = 0
                if not target:
                    audit_str = _safe_str(item.get("design_audit"))
                    m_aud = re.search(
                        r"AUDIT:\s*\[\s*([^\]]+?)\s*\]",
                        audit_str,
                        re.IGNORECASE,
                    )
                    if m_aud:
                        cand_target = m_aud.group(1).strip().lower()
                        # Only treat as target keyword if it belongs to study headwords
                        # (Syntactic L1 items specify a grammar pattern name in brackets, not a target keyword)
                        if cand_target in prompt_headwords or any(cand_target.startswith(hw) or hw.startswith(cand_target) for hw in prompt_headwords):
                            target = cand_target
            is_comprehension = not target and not is_translation
            expected_opt_count = 2 if is_comparative_translation else 4
            checks += 1
            
            # 1. Option count and distinctness (case/space-insensitive uniqueness check)
            is_list_valid = isinstance(options, list) and len(options) == expected_opt_count
            has_no_duplicates = is_list_valid and len(set(str(o).strip().lower() for o in options)) == expected_opt_count
            
            # 2. Correct answer index validity
            valid_idx = isinstance(idx, int) and 0 <= idx < expected_opt_count
            
            # 3. Target word must be present in options and match the correct answer slot options[idx]
            target_in_options = True
            if target and is_list_valid and valid_idx:
                clean_target = _clean_core(target)
                if is_comparative_translation:
                    # In comparative translation, target_keyword must be in idiomatic_translation
                    idiomatic_ans = _clean_core(_safe_str(item.get("idiomatic_translation")).lower())
                    target_toks = [t for t in clean_target.split() if t and t not in STOP_SLOTS]
                    is_ly_adv = (" " not in clean_target and len(clean_target) >= 5 and clean_target.endswith("ly"))
                    target_in_options = (
                        bool(re.search(rf"\b{re.escape(clean_target)}\b", idiomatic_ans))
                        if is_ly_adv
                        else (
                            clean_target in idiomatic_ans
                            or (len(clean_target) >= 4 and clean_target[:4] in idiomatic_ans)
                            or (bool(target_toks) and all(_form_in_text(t, idiomatic_ans) for t in target_toks))
                        )
                    )
                elif is_translation:
                    # In legacy translation quizzes, target_keyword can be in the options slot,
                    # or in the fixed english_skeleton / correct_english_answer!
                    full_ans = _clean_core(_safe_str(item.get("correct_english_answer")).lower())
                    skel_ans = _clean_core(_safe_str(item.get("english_skeleton")).lower())
                    opt_ans = _clean_core(_safe_str(options[idx]).lower())
                    combined_ans = f"{full_ans} {skel_ans} {opt_ans}"
                    target_toks = [t for t in clean_target.split() if t and t not in STOP_SLOTS]
                    target_in_options = (
                        clean_target in full_ans or
                        clean_target in skel_ans or
                        clean_target in opt_ans or
                        (len(clean_target) >= 4 and any(
                            clean_target[:4] in text
                            for text in [full_ans, skel_ans, opt_ans]
                        )) or
                        (bool(target_toks) and all(_form_in_text(t, combined_ans) for t in target_toks))
                    )
                else:
                    selected_opt = _clean_core(str(options[idx]))
                    # Match exact or valid inflections (e.g. assert -> asserted, marathon -> marathons)
                    # or quantifier/phrase head blanking (e.g. clean_target="a piece of", selected_opt="piece")
                    target_in_options = (clean_target == selected_opt) or (
                        selected_opt in clean_target.split()
                    ) or (
                        len(clean_target) >= 4 and len(selected_opt) >= 4 and (
                            selected_opt.startswith(clean_target[:4]) or clean_target.startswith(selected_opt[:4]) or (clean_target in selected_opt)
                        )
                    )

            # 4. Blank verification for fill-in-the-blank questions (Strict Single Blank Gate)
            has_blank_when_expected = True
            is_single_blank = True
            no_stem_leak = True
            no_article_leak = True
            is_adequate_complexity = True
            # Gates below are only evaluated for lexical fill-in-the-blank items; they must
            # still be bound for translation / comprehension / comparative-translation items
            # or the shared item_passed computation below raises UnboundLocalError.
            no_scaffold_leak = True
            no_verbatim_example = True
            verbatim_label = "dictionary example"
            # the Prescribed Options Contract is satisfied by default — it only
            # binds when the blueprint actually prescribes options (cloze and comprehension
            # blueprints do not).
            options_match_blueprint = True
            index_match_blueprint = True
            no_verb_inflection_clash = True
            no_comparative_defect = True
            comp_defect_reason = ""


            if is_comparative_translation:
                has_blank_when_expected = True
                idiom_txt = _safe_str(item.get("idiomatic_translation"))
                flawed_txt = _safe_str(item.get("flawed_translation"))
                t_gram = _safe_str(item.get("target_grammar") or item.get("design_audit"))
                comp_defect = LinguisticEngine.detect_comparative_translation_defect(
                    idiom_txt, flawed_txt, target_grammar=t_gram
                )
                if comp_defect:
                    no_comparative_defect = False
                    comp_defect_reason = comp_defect
                    flags.append(
                        f"❌ Quiz item '{target or question[:25]}': Comparative translation defect — {comp_defect}"
                    )
                else:
                    mp_sim = LinguisticEngine.calculate_minimal_pair_similarity(idiom_txt, flawed_txt)
                    if mp_sim < 0.75:
                        no_comparative_defect = False
                        comp_defect_reason = f"Minimal-Pair contract violation ({mp_sim:.0%} shared wording < 75%)"
                        flags.append(
                            f"❌ Quiz item '{target or question[:25]}': Minimal-Pair contract violation — idiomatic_translation and flawed_translation share only {mp_sim:.0%} wording"
                        )
            elif is_comprehension:
                # Reading/Listening/Video comprehension questions do not require blanks
                has_blank_when_expected = True
            elif is_translation:
                if skeleton:
                    has_blank_when_expected = bool(re.search(r"\[\s*_{2,}\s*\]|_{2,}", skeleton))
            elif target and question:
                # Lexical multiple-choice fill-in-the-blank items MUST contain EXACTLY ONE blank '____'
                blanks = re.findall(r"_{2,}", question)
                if len(blanks) == 0:
                    has_blank_when_expected = False
                elif len(blanks) > 1:
                    is_single_blank = False
                    flags.append(
                        f"❌ Quiz item '{target}': Multiple blanks ({len(blanks)}) detected in question stem (only exactly ONE blank permitted)"
                    )

                # Scaffolding & Meta-Instruction Leakage Gate: stem cannot contain prompt template strings or unreplaced bracketed placeholders
                scaffold_leak = re.search(
                    r"(?:\b(?:Fit into frame|Fit into the Pattern|Plural noun slot required|Complete predicate required|Write a natural sentence|Model:)\b"
                    r"|\b(?:to\s+do|doing)\s+something\b"
                    r"|\b(?:is/are|was/were|sb/sth|sth/sb)\b"
                    r"|\(\s*should\s*\)"
                    r"|\[(?:concrete\s+physical|physical\s+(?:object|person)|sb|sth|doing\s+sth|someone|something|V-base|verb|noun)[^\]]*\])",
                    question,
                    flags=re.IGNORECASE,
                )
                if scaffold_leak:
                    no_scaffold_leak = False
                    flags.append(
                        f"❌ Quiz item '{target}': Prompt scaffolding leaked into question stem: '{scaffold_leak.group(0)}'"
                    )

                # Target Stem Leakage Gate: target word/stem or prefix-stripped derivational root cannot leak into question outside the blank
                clean_target = _clean_core(target)
                target_stem = re.sub(r'(?:ed|ing|s|es|ly|tion|ment)$', '', clean_target)
                stem_no_blank = re.sub(r'_{2,}', ' ', question)
                stem_words = re.findall(r'\b[a-zA-Z]+\b', stem_no_blank)
                leaked_words = [
                    w for w in stem_words
                    if w.lower() == clean_target
                    or (len(target_stem) >= 4 and re.sub(r'(?:ed|ing|s|es|ly|tion|ment)$', '', w.lower()) == target_stem)
                    or (" " not in clean_target and len(w) >= 4 and LinguisticEngine.shares_derivational_root(w.lower(), clean_target))
                ]
                if leaked_words:
                    no_stem_leak = False
                    flags.append(
                        f"❌ Quiz item '{target}': Target word leaks verbatim into question stem outside blank: {leaked_words}"
                    )

                # Distractor Stem Echo Gate (vocabulary single-sentence items only):
                # non-key content-word distractors (excluding function words & light verbs) must not appear verbatim in the stem
                if is_list_valid and options and valid_idx and not is_translation and len(question) <= 350:
                    _stem_words_low = {w.lower() for w in stem_words}
                    _stem_lemmas_low = {
                        (LinguisticEngine.lemma_of(w.lower()) or w.lower())
                        for w in stem_words if len(w) >= 4
                    }
                    _echo_exempt = {
                        "within", "beyond", "across", "around", "behind", "below", "beside", "between",
                        "during", "except", "inside", "outside", "through", "throughout", "toward",
                        "towards", "under", "until", "without", "against", "about", "above", "after",
                        "along", "among", "before", "beneath", "despite",
                        "make", "take", "give", "have", "keep", "find", "hold", "come", "turn", "bring",
                        "leave", "work", "play", "show", "call", "help", "need", "want", "feel", "seem",
                        "become", "start", "begin", "know", "think", "tell", "mean", "send", "use",
                        "also", "even", "just", "only", "very", "much", "more", "most", "some", "many",
                        "such", "than", "then", "that", "this", "with", "from", "into", "onto", "over",
                        "when", "what", "which", "where", "while", "will", "would", "could", "should",
                        "must", "been", "being", "were",
                    }
                    echoed_dists = []
                    for o_i, opt in enumerate(options):
                        if o_i == idx:
                            continue
                        d_clean = _clean_core(opt)
                        if not d_clean or " " in d_clean or len(d_clean) < 4 or d_clean in _echo_exempt:
                            continue
                        d_lem = LinguisticEngine.lemma_of(d_clean) or d_clean
                        if d_lem in _echo_exempt:
                            continue
                        if d_clean in _stem_words_low or (len(d_lem) >= 4 and d_lem in _stem_lemmas_low):
                            echoed_dists.append(str(opt).strip())
                    if echoed_dists:
                        no_stem_leak = False
                        flags.append(
                            f"❌ Quiz item '{target}': Distractor option(s) echo verbatim in question stem outside blank: {echoed_dists}"
                        )

                # Article Leakage Gate (Pre-blank 'a', 'an', 'a(n)', or 'a/an' leaking answer or invalidating distractors)
                no_article_leak = True
                if re.search(r"\ba\s*(?:\(\s*n\s*\)|/\s*an)\s*_{2,}\b", question, flags=re.IGNORECASE):
                    no_article_leak = False
                    flags.append(
                        f"❌ Quiz item '{target}': Question stem uses artificial 'a(n) ____' or 'a/an ____' before blank; use 'the ____', a possessive, an intervening adjective ('a remarkable ____'), or plural instead."
                    )
                elif options and re.search(r"\b(?:a|an)\s+_{2,}\b", question, flags=re.IGNORECASE):
                    initials = {str(opt).strip()[:1].lower() for opt in options if str(opt).strip()}
                    has_vowel_init = any(init in 'aeiou' for init in initials)
                    has_cons_init = any(init.isalpha() and init not in 'aeiou' for init in initials)
                    if has_vowel_init and has_cons_init:
                        no_article_leak = False
                        flags.append(
                            f"❌ Quiz item '{target}': Question stem has indefinite article ('a/an') immediately preceding blank with mixed vowel/consonant options (fatal test-wiseness leakage); use 'the ____', plural, or possessive instead."
                        )

                bp_info = prompt_blueprints.get(target.lower(), {}) if prompt_blueprints else {}
                bp_pos = bp_info.get("part_of_speech", "")

                # Verb Option Morphological Agreement Gate (动词同型校验门禁)
                # If target is an inflected verb (-ed, -ing, -s), options must not mix base verbs with inflected target
                if is_list_valid and options and len(options) == 4:
                    t_parts = target.strip().split()
                    t_head = t_parts[0].lower()
                    t_base = LinguisticEngine.lemma_of(t_head) or t_head
                    t_is_adj = any(a in bp_pos for a in ("adj", "adjective"))
                    t_is_verb = not t_is_adj and (
                        bp_pos in ("verb", "phrasal verb")
                        or (not bp_pos and LinguisticEngine._is_base_verb_lemma(t_base))
                    )
                    if t_is_verb and t_head != t_base:
                        t_verb_tag = None
                        for cand_tag in ("VBD", "VBG", "VBZ", "VBN"):
                            if (LinguisticEngine._inflect_verb(t_base, cand_tag) or "").lower() == t_head:
                                t_verb_tag = cand_tag
                                break
                        if t_verb_tag in ("VBD", "VBN", "VBG", "VBZ"):
                            base_clashing = []
                            for o in options:
                                o_str = str(o).strip()
                                o_parts = o_str.split()
                                if not o_parts:
                                    continue
                                o_head = o_parts[0].lower()
                                o_base = o_head if LinguisticEngine._is_base_verb_lemma(o_head) else (LinguisticEngine.lemma_of(o_head) or o_head)
                                o_can_be_verb = (
                                    LinguisticEngine._is_base_verb_lemma(o_base)
                                    or bool(LinguisticEngine._wordnet_pos(o_base) & {"v"})
                                )
                                if o_can_be_verb and o_head == o_base and o_head != t_head:
                                    inflected_form = LinguisticEngine._inflect_verb(o_base, t_verb_tag)
                                    if inflected_form and inflected_form.lower() == o_head:
                                        continue
                                    base_clashing.append(o_str)
                            if base_clashing:
                                no_verb_inflection_clash = False
                                flags.append(
                                    f"❌ Quiz item '{target}': Verb option inflection mismatch (动词同型错误) — target is inflected ({t_verb_tag}), but options contain base-form verbs: {base_clashing}"
                                )

                # Auxiliary & Modal Agreement Gate (助动词/补足语语法门禁)
                # Inspect the immediate left context of `____` in question stem
                q_text = str(item.get("question") or "")
                if is_list_valid and options and len(options) == 4 and "____" in q_text:
                    m_left_aux = re.search(r"\b(has|have|had)(?:\s+been)?\s+____", q_text, re.IGNORECASE)
                    if m_left_aux:
                        aux_word = m_left_aux.group(1).lower()
                        vb_clashing = []
                        for o in options:
                            o_head = str(o).strip().split()[0].lower()
                            o_base = LinguisticEngine.lemma_of(o_head) or o_head
                            if o_head == o_base and LinguisticEngine._is_base_verb_lemma(o_base):
                                vbn_form = (LinguisticEngine._inflect_verb(o_base, "VBN") or "").lower()
                                if vbn_form and vbn_form != o_base and o_head != vbn_form:
                                    vb_clashing.append(o)
                        if vb_clashing:
                            no_verb_inflection_clash = False
                            flags.append(
                                f"❌ Quiz item '{target}': Auxiliary agreement clash (助动词语法冲突) — stem has '{aux_word} ____' requiring past participle (VBN), but options contain base-form verbs: {vb_clashing}"
                            )

                    m_modal = re.search(r"\b(can|could|will|would|shall|should|may|might|must|did|does|do)\s+____", q_text, re.IGNORECASE)
                    if m_modal:
                        modal_word = m_modal.group(1).lower()
                        infl_clashing = []
                        for o in options:
                            o_head = str(o).strip().split()[0].lower()
                            o_base = LinguisticEngine.lemma_of(o_head) or o_head
                            if o_head != o_base and LinguisticEngine._is_base_verb_lemma(o_base):
                                infl_clashing.append(o)
                        if infl_clashing:
                            no_verb_inflection_clash = False
                            flags.append(
                                f"❌ Quiz item '{target}': Modal agreement clash (情态动词语法冲突) — stem has '{modal_word} ____' requiring base verb (VB), but options contain inflected verbs: {infl_clashing}"
                            )

                    m_to_inf = re.search(r"\bto\s+____", q_text, re.IGNORECASE)
                    if m_to_inf and bp_pos in ("verb", "phrase", "phrasal verb"):
                        infl_clashing = []
                        for o in options:
                            o_head = str(o).strip().split()[0].lower()
                            o_base = LinguisticEngine.lemma_of(o_head) or o_head
                            if o_head != o_base and LinguisticEngine._is_base_verb_lemma(o_base):
                                infl_clashing.append(o)
                        if infl_clashing:
                            no_verb_inflection_clash = False
                            flags.append(
                                f"❌ Quiz item '{target}': Infinitive agreement clash (不定式语法冲突) — stem has 'to ____' requiring base verb (VB), but options contain inflected verbs: {infl_clashing}"
                            )

                    # Uniform Option Inflection Gate (选项群曲折一致性门禁)
                    # Options must not mix base verbs and past participles (e.g., 3 VBNs and 1 VB)
                    vbn_count = 0
                    vb_count = 0
                    vb_opts = []
                    vbn_opts = []
                    for o in options:
                        o_head = str(o).strip().split()[0].lower()
                        o_base = LinguisticEngine.lemma_of(o_head) or o_head
                        if LinguisticEngine._is_base_verb_lemma(o_base):
                            vbn_form = (LinguisticEngine._inflect_verb(o_base, "VBN") or "").lower()
                            if o_head == o_base and vbn_form != o_base:
                                vb_count += 1
                                vb_opts.append(o)
                            elif o_head == vbn_form and vbn_form != o_base:
                                vbn_count += 1
                                vbn_opts.append(o)
                    if vb_count > 0 and vbn_count > 0 and (vb_count + vbn_count == 4):
                        no_verb_inflection_clash = False
                        flags.append(
                            f"❌ Quiz item '{target}': Mixed verb inflections among options (选项曲折形式混杂) — base verbs: {vb_opts}, past participles: {vbn_opts}"
                        )

                # Prescribed Options Contract Gate — the blueprint owns the
                # options and the answer index (CRC32-positioned distractors); the writer
                # may only author the stem, explanation and audit.
                prescribed_opts = bp_info.get("prescribed_options") or []
                prescribed_idx = bp_info.get("prescribed_index")
                if prescribed_opts and is_list_valid:
                    wanted_opts = [str(o).strip().lower() for o in prescribed_opts]
                    shipped_opts = [str(o).strip().lower() for o in options]
                    if shipped_opts != wanted_opts:
                        options_match_blueprint = False
                        substituted = sorted(set(wanted_opts) - set(shipped_opts))
                        introduced = sorted(set(shipped_opts) - set(wanted_opts))
                        flags.append(
                            f"❌ Quiz item '{target}': Prescribed options altered — blueprint prescribed "
                            f"[{', '.join(str(o) for o in prescribed_opts)}], item shipped "
                            f"[{', '.join(str(o) for o in options)}]"
                            + (f" (substituted: {substituted}; introduced: {introduced})" if substituted or introduced else "")
                        )
                    if isinstance(prescribed_idx, int) and isinstance(idx, int) and idx != prescribed_idx:
                        index_match_blueprint = False
                        flags.append(
                            f"❌ Quiz item '{target}': Prescribed answer index altered — blueprint index "
                            f"{prescribed_idx}, item shipped {idx}"
                        )

                # Verbatim Copy Gate : a stem may emulate a corpus model's syntax and
                # register, but it may never reuse its wording. Both corpus models are checked —
                # the PedLex example and the curriculum quote that was strong enough to license
                # evidence. Each source is compared twice: as written, and with the target word
                # blanked out, because the classic shortcut is to mask the blueprint's own
                # example and that hole otherwise breaks every 5-gram of a short example.
                # A display-only quote is too short for an n-gram test, so reusing any part of
                # it counts as copying outright.
                no_verbatim_example = True
                verbatim_label = "dictionary example"
                verbatim_hit = ""
                stem_clean = re.sub(r"_{2,}", " ", question.lower())
                stem_tokens = re.sub(r"[^\w\s]", " ", stem_clean).split()
                stem_5grams = {tuple(stem_tokens[i:i + 5]) for i in range(len(stem_tokens) - 4)}
                for _label, _source in (
                    ("dictionary example", bp_info.get("example", "")),
                    ("curriculum quote", bp_info.get("licensed_quote", "")),
                ):
                    for src_clean in _source_token_variants(_source, target):
                        if len(src_clean) < 5 or not stem_5grams:
                            continue
                        src_5grams = {tuple(src_clean[i:i + 5]) for i in range(len(src_clean) - 4)}
                        overlap_5grams = src_5grams.intersection(stem_5grams)
                        if overlap_5grams:
                            no_verbatim_example = False
                            verbatim_label = _label
                            verbatim_hit = " ".join(next(iter(overlap_5grams)))
                            break
                    if not no_verbatim_example:
                        break
                weak_quote = (
                    bp_info.get("candidate_quote", "")
                    if bp_info.get("quote_provenance") == "weak" else ""
                )
                weak_tokens = re.sub(r"[^\w\s]", " ", weak_quote.lower()).split()
                if no_verbatim_example and weak_tokens and len(weak_tokens) < 5:
                    if " ".join(weak_tokens) in " ".join(stem_tokens):
                        no_verbatim_example = False
                        verbatim_label = "curriculum quote"
                        verbatim_hit = " ".join(weak_tokens)
                if not no_verbatim_example:
                    flags.append(
                        f"❌ Quiz item '{target}': Stem verbatim from {verbatim_label} "
                        f"(detected n-gram overlap: '{verbatim_hit}...')!"
                    )

                # Stem Length Check (Physical Sanity Gate - Soft Warning)
                # Ensure stems provide sufficient context (> 3 words) without forcing artificial clausal complexity.
                if len(stem_words) < 4:
                    is_adequate_complexity = False
                    flags.append(f"⚠️ Quiz item '{target}': Inadequate context stem (< 4 words)")

            # 5. In-List Distractor Recycling check (Zero-Tolerance Level 1 Gate for LLM-invented distractors;
            # skipped when options are pre-computed by the blueprint).
            recycled_in_distractors = []
            bp_for_item = prompt_blueprints.get(target.lower(), {}) if (prompt_blueprints and target) else {}
            if prompt_headwords and is_list_valid and valid_idx and not bp_for_item.get("prescribed_options"):
                clean_target = _clean_core(target) if target else ""
                clean_hw_set = {_clean_core(hw) for hw in prompt_headwords if hw}

                for o_idx, opt in enumerate(options):
                    if o_idx == idx:
                        continue
                    opt_clean = str(opt).strip().lower()
                    opt_core = _clean_core(opt_clean)

                    # If this option is derived from the item's own target keyword (e.g. preserves, preserving, to preserve),
                    # it is a legitimate morphological/syntactic distractor trap, NOT recycling an external word!
                    if clean_target and (
                        opt_core == clean_target or 
                        clean_target in opt_clean or 
                        (len(clean_target) >= 4 and opt_clean.startswith(clean_target[:4]))
                    ):
                        continue

                    # Now check if it recycles another DIFFERENT headword from the study list
                    if opt_clean in prompt_headwords or (opt_core and opt_core in clean_hw_set and opt_core != clean_target):
                        recycled_in_distractors.append(opt_clean)
                    elif not is_translation:
                        # For vocabulary multiple-choice, check lemma and morphological family identity
                        # (Avoid naive 5-char prefix matching which falsely conflates 'underestimated' with 'underpin'!)
                        opt_lemma = LinguisticEngine.lemma_of(opt_clean) if " " not in opt_clean else opt_clean
                        for hw in prompt_headwords:
                            hw_clean = str(hw).strip().lower()
                            if clean_target and (hw_clean == clean_target or _clean_core(hw_clean) == clean_target):
                                continue
                            hw_lemma = LinguisticEngine.lemma_of(hw_clean) if " " not in hw_clean else hw_clean
                            if (
                                (opt_lemma and hw_lemma and opt_lemma == hw_lemma)
                                or (" " not in hw_clean and " " not in opt_clean and LinguisticEngine.are_same_word_family(opt_clean, hw_clean))
                            ):
                                recycled_in_distractors.append(opt_clean)
                                break

            no_in_list_recycling = len(recycled_in_distractors) == 0

            # 6. CEFR Lexical Ceiling Gate (flags obscure C2 vocabulary in A1/A2/B1 non-blueprint quiz options/stems)
            if src_cefr_level and src_cefr_rank <= 3 and not bp_for_item.get("prescribed_options"):
                cefr_outliers = []
                for o_idx, opt in enumerate(options if isinstance(options, list) else []):
                    if o_idx == idx:
                        continue
                    opt_w = str(opt).strip().lower()
                    if (
                        " " not in opt_w
                        and opt_w.isalpha()
                        and len(opt_w) >= 5
                        and opt_w not in src_passage_words
                        and opt_w not in prompt_headwords
                    ):
                        w_lvl = LinguisticEngine.get_word_cefr(opt_w)
                        if w_lvl and LinguisticEngine.CEFR_ORDER.get(w_lvl, 0) >= max(5, src_cefr_rank + 2):
                            cefr_outliers.append(f"{opt_w} ({w_lvl})")
                if cefr_outliers:
                    flags.append(
                        f"⚠️ Quiz item '{target or question[:25]}': Distractor exceeds passage CEFR ceiling ({src_cefr_level}): {', '.join(cefr_outliers)}"
                    )

            # Level-1 pass list. Every term below is a field-to-field or
            # string-to-string comparison on LLM-authored fields.
            item_passed = (
                is_list_valid and
                has_no_duplicates and
                valid_idx and
                target_in_options and
                explanation and
                question and
                has_blank_when_expected and
                is_single_blank and
                no_stem_leak and
                no_scaffold_leak and
                no_article_leak and
                no_verb_inflection_clash and
                no_verbatim_example and
                no_in_list_recycling and
                options_match_blueprint and
                index_match_blueprint and
                no_comparative_defect
            )

            if item_passed:
                passes += 1
            else:
                reasons = []
                if not is_list_valid: reasons.append(f"options count != {expected_opt_count}")
                elif not has_no_duplicates: reasons.append("duplicate options detected")
                if not valid_idx: reasons.append("invalid answer index")
                if not target_in_options: reasons.append(f"target '{target}' not matching options[{idx}]")
                if not explanation: reasons.append("missing explanation")
                if not question: reasons.append("missing question text")
                if not no_scaffold_leak: reasons.append("prompt scaffolding leaked into question stem")
                if not has_blank_when_expected: reasons.append("missing fill-in-the-blank slot (____)")
                if not is_single_blank: reasons.append("multiple blanks detected in stem")
                if not no_stem_leak: reasons.append("target word leaks into question stem")
                if not no_article_leak: reasons.append("indefinite article leakage before blank")
                if not no_verb_inflection_clash: reasons.append("non-base verb inflection clash before blank")
                if not no_verbatim_example: reasons.append(f"stem verbatim from {verbatim_label}")
                if not no_comparative_defect: reasons.append(comp_defect_reason or "comparative translation defect")
                if not options_match_blueprint:
                    reasons.append(f"prescribed options altered from blueprint {bp_info.get('prescribed_options')}")
                if not index_match_blueprint:
                    reasons.append(f"prescribed answer index altered from blueprint index {bp_info.get('prescribed_index')}")

                if not no_in_list_recycling:
                    reasons.append(f"recycles study list headwords in distractors {recycled_in_distractors}")
                    flags.append(f"❌ Quiz item '{target or question[:25]}' in-list distractor recycling: {recycled_in_distractors}")
                elif not has_blank_when_expected:
                    flags.append(f"❌ Quiz item '{target}': missing fill-in-the-blank slot (____)")
                elif not is_single_blank:
                    flags.append(f"❌ Quiz item '{target}': Multiple blanks detected in stem")
                elif not no_stem_leak:
                    flags.append(f"❌ Quiz item '{target}': Target word leaks into question stem: {leaked_words}")
                elif not no_article_leak:
                    flags.append(f"❌ Quiz item '{target}': indefinite article leakage before blank")
                elif not no_verbatim_example:
                    flags.append(f"❌ Quiz item '{target}': Stem verbatim from {verbatim_label}")
                else:
                    flags.append(f"⚠️ Quiz question failed pedagogy check: {', '.join(reasons)}")
        elif task_type == "summary":
            name = _safe_str(item.get("concept_name"))
            sig = _safe_str(item.get("educational_significance"))
            details = item.get("key_details", [])
            sub_concepts = item.get("sub_concepts", [])
            checks += 1
            # If sub_concepts are present, validate their internal structure
            sub_ok = True
            if isinstance(sub_concepts, list) and sub_concepts:
                for sub in sub_concepts:
                    if not isinstance(sub, dict):
                        sub_ok = False
                        break
                    s_name = _safe_str(sub.get("sub_concept_name"))
                    s_sig = _safe_str(sub.get("significance_or_takeaway"))
                    s_pts = sub.get("key_points", [])
                    if not (s_name and s_sig and isinstance(s_pts, list) and len(s_pts) > 0):
                        sub_ok = False
                        break
            if name and sig and isinstance(details, list) and len(details) > 0 and sub_ok:
                passes += 1
            else:
                reason = "missing educational significance or key details" if not (name and sig and isinstance(details, list) and len(details) > 0) else "malformed sub_concepts"
                flags.append(f"⚠️ Concept '{name}' {reason}")
        elif task_type == "mindmap":
            name = _safe_str(item.get("branch_name"))
            checks += 1
            if name:
                passes += 1
            else:
                flags.append("⚠️ MindMap branch missing branch name")
        elif task_type == "expert_audit":
            checks += 1
            raw_distractors = item.get("distractors", [])
            # Filter out empty or non-dict objects (e.g. trailing {} produced during JSON truncation)
            distractors = [
                d for d in raw_distractors 
                if isinstance(d, dict) and any(d.get(k) for k in ("option_text", "trap_type", "elimination_rationale"))
            ] if isinstance(raw_distractors, list) else []
            bs_idx = item.get("blind_solved_index")
            feedback = _safe_str(item.get("diagnostic_feedback"))
            score = item.get("pedagogical_score", 100)
            single_valid = item.get("single_fit_valid", True)
            is_valid_dist = len(distractors) in (1, 2, 3, 4)
            is_valid_idx = isinstance(bs_idx, int) and -1 <= bs_idx <= 3
            # Feedback is required only if the item has defects (score < 90 or invalid single fit);
            # for flawless items (score >= 90), empty feedback is valid and expected.
            feedback_ok = bool(feedback) if (score < 90 or not single_valid) else True
            triage = item.get("triage_action", "PASS")
            cured_q = item.get("cured_question")
            is_valid_triage = triage in ("PASS", "REPAIR", "REWRITE")
            cured_ok = bool(cured_q) if triage in ("REPAIR", "REWRITE") else True
            if is_valid_dist and is_valid_idx and feedback_ok and is_valid_triage and cured_ok:
                passes += 1
            else:
                reasons = []
                if not is_valid_dist: reasons.append(f"distractor count != 1 to 4 (got {len(distractors) if isinstance(distractors, list) else 0})")
                if not is_valid_idx: reasons.append("invalid blind solve index")
                if not feedback_ok: reasons.append("missing diagnostic feedback for defective item")
                if not is_valid_triage: reasons.append(f"invalid triage_action '{triage}'")
                if not cured_ok: reasons.append(f"missing cured_question for triage '{triage}'")
                flags.append(f"⚠️ Audit item failed validation: {', '.join(reasons)}")
                
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


def _score_uniqueness(items: List[Dict[str, Any]], task_type: str, user_prompt: str = "") -> Tuple[Optional[float], List[str]]:
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
        all_mistakes = [str(it.get("common_mistakes") or "").strip() for it in items if str(it.get("common_mistakes") or "").strip()]
        if len(all_mistakes) > len(set(all_mistakes)):
            from collections import Counter
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
            from collections import Counter
            counts = Counter(all_distractors)
            repeated = {w: c for w, c in counts.items() if c >= 3 and len(w) >= 3}
            if repeated:
                rep_details = ", ".join(f"'{w}' ({c}x)" for w, c in list(repeated.items())[:3])
                unique_flags.append(f"❌ Distractors recycled repeatedly across quiz: {rep_details}")
    return earned, unique_flags

