"""
Deterministic vocabulary extraction pipeline, verbatim surface matching, and video vocabulary.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("librarian.linguistics")


class ExtractionVocabExtractorMixin:
    @classmethod
    def extract_deterministic_vocabulary(
        cls,
        text: str,
        syllabus_vocab: Optional[List[str]] = None,
        target_count: int = 20,
        enable_llm_disambiguation: bool = True,
        exclude_words: Optional[Set[str]] = None,
        unit_name: str = ""
    ) -> List[Dict[str, Any]]:
        """
        Pure Computational Linguistic & Lexicographic Extractor (0 Tokens, 100% Deterministic).
        Extracts academic target vocabulary, locates authentic quoted sentences with Sentence IDs,
        tags contextual Part of Speech via spaCy, and hydrates authoritative definitions & authentic
        corpus examples from PedLex 6th Edition.
        """
        with cls.wsd_log_session(unit_name or "vocabulary"):
            _, pool = cls.tokenize_and_index_sentences(text)
            if not pool:
                return []

            excl_set: Set[str] = {
                re.sub(r'\[.*?\]|\(.*?\)', '', str(ew)).strip().lower()
                for ew in (exclude_words or set())
                if ew
            }
            excl_set.discard("")

            # 1. Determine target words (Syllabus primary + Passage reserve pool for Gate 2 backfill)
            target_candidates: List[Dict[str, Any]] = []
            if syllabus_vocab:
                cleaned_syllabus: List[str] = []
                seen_clean = set(excl_set)
                for w in syllabus_vocab:
                    w_clean = re.sub(r'\[.*?\]|\(.*?\)', '', str(w)).strip().lower()
                    if w_clean and w_clean not in seen_clean:
                        seen_clean.add(w_clean)
                        cleaned_syllabus.append(w_clean)

                # Verbatim Compound Resolution (Passage = Ground Truth):
                # If a bare syllabus word (e.g. 'indulgent', 'confidence', 'prone', 'reflective', 'balance')
                # has zero standalone occurrences in the passage, check if it only appears as a sub-token
                # of a hyphenated compound in the passage (e.g. 'self-indulgent', 'self-confidence', 'waste-prone').
                # Promote the candidate to the verbatim passage compound so Gate 1 & Gate 2 evaluate the true
                # passage word (accepting attested dictionary compounds like 'self-indulgent' / 'self-confidence'
                # and cleanly dropping unattested coinages like 'waste-prone' / 'self-reflective' / 'self-balance').
                if cleaned_syllabus and pool:
                    passage_hyphen_compounds: List[str] = []
                    seen_hyph: Set[str] = set()
                    for _sid, sent in pool.items():
                        for m_hc in re.finditer(r"\b([a-zA-Z]+(?:-[a-zA-Z]+)+)\b", sent):
                            hc = m_hc.group(1).lower()
                            if hc not in seen_hyph:
                                seen_hyph.add(hc)
                                passage_hyphen_compounds.append(hc)
                    if passage_hyphen_compounds:
                        resolved_syllabus: List[str] = []
                        seen_resolved: Set[str] = set()
                        for sw in cleaned_syllabus:
                            resolved_w = sw
                            if "-" not in sw and not any(cls.text_contains_form(sent, sw) for sent in pool.values()):
                                for hc in passage_hyphen_compounds:
                                    parts = hc.split("-")
                                    if any(cls.form_matches(p, sw) for p in parts):
                                        # Normalize plural hyphenated noun head to singular if attested in dictionary
                                        hc_norm = hc
                                        if not cls.get_ldoce_entry(hc) and len(parts) >= 2:
                                            last_lem = cls._surface_lemma(parts[-1])
                                            if last_lem and last_lem != parts[-1]:
                                                cand_sing = "-".join(parts[:-1] + [last_lem])
                                                if cls.get_ldoce_entry(cand_sing):
                                                    hc_norm = cand_sing
                                        resolved_w = hc_norm
                                        break
                            if resolved_w not in seen_resolved:
                                seen_resolved.add(resolved_w)
                                resolved_syllabus.append(resolved_w)
                        cleaned_syllabus = resolved_syllabus
                        seen_clean = set(cleaned_syllabus) | excl_set

                # Pre-filter same-word-family syllabus pairs (e.g. 'intrigue' + 'intrigued', 'predominant' + 'predominantly'):
                # If a syllabus word has a same-family sibling in the syllabus, drop any member that:
                #   (a) is completely unattested via text_contains_form while a sibling IS attested (e.g. 'predominant' vs 'predominantly'), OR
                #   (b) only matches passage sentences where the exact surface token is another syllabus sibling (e.g. 'intrigue' in "feeling intrigued")
                #       AND that sibling has a valid dictionary sense + example (otherwise retain the base headword).
                if len(cleaned_syllabus) > 1 and pool:
                    syl_shown_map: Dict[str, List[Tuple[str, str]]] = {
                        sw: [(sid, sent) for sid, sent in pool.items() if cls.text_contains_form(sent, sw)]
                        for sw in cleaned_syllabus
                    }
                    filtered_syllabus: List[str] = []
                    for sw in cleaned_syllabus:
                        siblings = [
                            other for other in cleaned_syllabus
                            if other != sw and cls.are_same_word_family(sw, other)
                        ]
                        if not siblings:
                            filtered_syllabus.append(sw)
                            continue
                        sw_shown = syl_shown_map.get(sw, [])
                        # Case (a): sw is not attested by any valid morphological form, while a sibling is attested
                        if not sw_shown and any(syl_shown_map.get(sib) for sib in siblings):
                            continue
                        # Case (b): sw has no exact surface match in its attested sentences, while a longer/derived sibling
                        # appears verbatim in those exact sentences AND has complete lexicon evidence (sense + example)
                        sw_has_exact = any(
                            bool(re.search(rf"\b{re.escape(sw)}\b", sent, flags=re.IGNORECASE))
                            for _sid, sent in sw_shown
                        )
                        if not sw_has_exact and sw_shown:
                            superseded = False
                            for sib in siblings:
                                sib_exact_sents = [
                                    sent for _sid, sent in sw_shown
                                    if re.search(rf"\b{re.escape(sib)}\b", sent, flags=re.IGNORECASE)
                                ]
                                if sib_exact_sents:
                                    sib_cands = cls.build_top_k_vocab_candidates(sib, sib_exact_sents[0], k=2)
                                    sib_ok = False
                                    if sib_cands:
                                        top_c = sib_cands[0]
                                        sib_def = top_c.get("definition", "")
                                        sib_ex = (top_c.get("examples") or [""])[0]
                                        sib_ex = cls.ensure_distinct_example(
                                            sib, str(top_c.get("pos") or "adjective"), sib_ex,
                                            sib_exact_sents[0], sense=top_c.get("raw_sense")
                                        )
                                        if sib_def and not cls.is_boilerplate_definition(sib_def) and sib_ex:
                                            sib_ok = True
                                    if sib_ok:
                                        superseded = True
                                        break
                            if superseded:
                                continue
                        filtered_syllabus.append(sw)
                    cleaned_syllabus = filtered_syllabus

                for w in cleaned_syllabus:
                    target_candidates.append({"word": w, "sid": "", "quote": "", "is_syllabus": True})
                # Reserve passage candidates to replace any syllabus item dropped by Gate 2 (Evidence-Completeness)
                reserve_skels = cls.mine_vocabulary_skeletons(text, target_count=max(target_count, 20) + 15)
                for sk in reserve_skels:
                    rw = sk.get("word", "").strip().lower()
                    if rw and rw not in seen_clean:
                        seen_clean.add(rw)
                        sk_copy = dict(sk)
                        sk_copy["is_syllabus"] = False
                        target_candidates.append(sk_copy)
            else:
                skels = cls.mine_vocabulary_skeletons(text, target_count=target_count + 15)
                target_candidates = skels

            results: List[Dict[str, Any]] = []
            seen_words: Set[str] = set()
            seen_defs: Set[str] = set()
            seen_word_quotes: List[Tuple[str, str]] = []

            for cand in target_candidates:
                if len(results) >= target_count:
                    break
                w = cand["word"].strip().lower()
                if not w or w in seen_words or w in excl_set:
                    continue
                if not cand.get("is_syllabus"):
                    base_adj = cls.deinflect_comparative_superlative(w)
                    cand_quote = cand.get("quote", "")
                    if base_adj and cand_quote and cls.determine_contextual_pos(w, cand_quote) == "adjective":
                        w = base_adj
                        if w in seen_words or w in excl_set:
                            continue
                if not cls.is_pedagogical_target(w):
                    continue
                w_ld_check = cls.get_ldoce_entry(w)
                if not cand.get("is_syllabus") and w_ld_check and w_ld_check.get("kind") == "derived":
                    continue
                if not cand.get("is_syllabus") and any(cls.are_same_word_family(w, sw) for sw in seen_words):
                    continue

                matching_sent = cand.get("quote", "")
                matching_sid = cand.get("sid", "")

                # If sentence not already matched, search in sentence pool
                anchor_quality = "quote" if matching_sent else "syllabus"
                shown: List[Tuple[str, str]] = []
                if not matching_sent and pool:
                    w_clean = re.sub(r'[^a-z0-9]', '', w)
                    shown = [(sid, sent) for sid, sent in pool.items() if cls.text_contains_form(sent, w)]
                    # Filter out sentences where a lowercase common-noun/adj target ONLY appears as a mid-sentence capitalized proper noun / acronym expansion (e.g., 'Placement' in 'AP (Advanced Placement)')
                    if shown and w.islower():
                        non_propn_shown = []
                        for sid_s, sent_s in shown:
                            # Fast check: if any lowercase occurrence exists, keep immediately
                            if any(re.search(rf"(?<![a-zA-Z-]){re.escape(f)}(?![a-zA-Z-])", sent_s) for f in ({w} | cls.inflected_forms(w))):
                                non_propn_shown.append((sid_s, sent_s))
                                continue
                            # Otherwise check via spaCy whether all occurrences are mid-sentence PROPN / capitalized title-case compounds
                            _nlp = cls.get_spacy()
                            doc_s = _nlp(cls._strip_lexical_quotes(sent_s))
                            alpha_s = [t_a for t_a in doc_s if t_a.is_alpha]
                            is_title_heading = bool(alpha_s) and all(
                                t_a.text[0].isupper() or t_a.pos_ in ("ADP", "DET", "CCONJ", "PART", "TO")
                                for t_a in alpha_s
                            )
                            has_valid_tok = False
                            for t_s in doc_s:
                                if cls.form_matches(t_s.text, w) or cls.form_matches(t_s.lemma_, w):
                                    is_mid_cap = (
                                        not is_title_heading
                                        and t_s.text and t_s.text[0].isupper() and t_s.i > 0
                                        and t_s.pos_ != "VERB"
                                        and doc_s[t_s.i - 1].text not in (".", "!", "?", ":", ";", '"', "\u201c", "'", "\u2018", "\u2014", "--")
                                        and (t_s.pos_ == "PROPN" or doc_s[t_s.i - 1].text == "(" or (t_s.i - 1 > 0 and doc_s[t_s.i - 1].text[0].isupper()))
                                    )
                                    if not is_mid_cap:
                                        has_valid_tok = True
                                        break
                            if has_valid_tok:
                                non_propn_shown.append((sid_s, sent_s))
                        shown = non_propn_shown
                    # Avoid sentences already claimed by a same-word-family item
                    if len(shown) > 1 and seen_word_quotes:
                        non_clash_shown = [
                            (sid, sent) for sid, sent in shown
                            if not any(cls.are_same_word_family(w, sw) and sent == sq for sw, sq in seen_word_quotes)
                        ]
                        if non_clash_shown:
                            shown = non_clash_shown
                    # For deverbal prepositions/participial adjectives (e.g. 'given', 'provided', 'considering'),
                    # prefer sentences where the word is NOT preceded by an auxiliary ('have/has/had/be/been')
                    if w in ("given", "provided", "considering", "regarding", "concerning", "following", "including", "granted") and len(shown) > 1:
                        non_aux_shown = [
                            (sid, sent) for sid, sent in shown
                            if not re.search(rf"\b(?:have|has|had|am|is|are|was|were|be|been|being)\s+(?:\w+\s+)?{re.escape(w)}\b", sent, flags=re.IGNORECASE)
                        ]
                        if non_aux_shown:
                            shown = non_aux_shown
                    valid_shown = [(sid, sent) for sid, sent in shown if cls.anchor_evidence_ok(sent, w)]
                    best_sent, _best_rich = cls.pick_anchor_sentence(w, [s for _sid, s in shown])
                    if best_sent:
                        matching_sent = best_sent
                        matching_sid = cls.get_sentence_id(best_sent, pool) or matching_sid
                        anchor_quality = "clause"
                    elif valid_shown:
                        matching_sid, matching_sent = valid_shown[0]
                        anchor_quality = "unqualified"
                    elif shown:
                        matching_sid, matching_sent = shown[0]
                        anchor_quality = "attested"

                # Ground-Truth Passage Gate: Never extract a word that is unattested as an independent word/inflection
                # in the passage (no blind substring hijacking of hyphenated compounds like 'waste-prone' -> 'prone'
                # or closed compounds like 'toothpaste' -> 'paste', 'microorganisms' -> 'organism').
                if not matching_sent:
                    continue

                if any(cls.are_same_word_family(w, sw) and matching_sent == sq for sw, sq in seen_word_quotes):
                    continue

                # Infer contextual POS
                pos = cand.get("part_of_speech")
                if (not pos or pos == "function_word") and matching_sent:
                    ctx_pos = cls.determine_contextual_pos(w, matching_sent)
                    if ctx_pos and ctx_pos != "function_word":
                        pos = ctx_pos
                if not pos or pos == "function_word":
                    ld_w = cls.get_ldoce_entry(w)
                    pos = (ld_w.get("pos") if ld_w else "") or "adverb"

                # Gate 1: Hydrate authoritative PedLex definition and authentic example via Top-K Candidate Pool + WSD
                definition = ""
                example = ""
                lock_confidence = "high"
                lock_score = 0.0
                lock_margin = 0.0
                locked_raw_sense = None

                discourse_ctx = ""
                if matching_sid and pool:
                    m_sid_num = re.match(r"S-(\d+)", str(matching_sid))
                    if m_sid_num:
                        s_num = int(m_sid_num.group(1))
                        prev_s = str(pool.get(f"S-{s_num - 1}") or "").strip()
                        next_s = str(pool.get(f"S-{s_num + 1}") or "").strip()
                        if prev_s or next_s:
                            discourse_ctx = " ".join(p for p in (prev_s, f">> {matching_sent} <<", next_s) if p)

                cand_pool = cls.build_top_k_vocab_candidates(
                    w, matching_sent, pos=pos, k=12, discourse_context=discourse_ctx
                ) if matching_sent else []
                if cand_pool:
                    chosen = None
                    top_c = cand_pool[0]
                    # If Stage 1 symbolic lock achieved high confidence with a decisive margin,
                    # respect the symbolic engine directly (0ms, 0 tokens, zero hallucination).
                    # Only invoke Stage 2 LLM arbitration when candidates compete closely (margin < 15.0 or low confidence).
                    is_symbolically_locked = (
                        top_c.get("confidence") == "high"
                        and float(top_c.get("margin", 0.0)) >= 15.0
                        and float(top_c.get("score", 0.0)) >= 25.0
                    )
                    if len(cand_pool) > 1 and enable_llm_disambiguation and not is_symbolically_locked:
                        chosen_idx = cls.disambiguate_via_llm(
                            w, matching_sent, cand_pool, discourse_context=discourse_ctx, target_pos=pos
                        )
                        if 0 <= chosen_idx < len(cand_pool):
                            chosen = cand_pool[chosen_idx]
                        elif pos and re.search(rf"\b{re.escape(w)}\b", matching_sent.lower()):
                            # Cross-POS retry is only permitted when the surface token is in base uninflected form
                            # and not syntactically locked to a prepositional/determiner noun phrase ('from drones')
                            is_locked_np = (
                                str(pos).strip().lower() == "noun"
                                and bool(re.search(rf"\b(?:from|of|in|on|at|with|by|for|into|between|among|the|a|an|this|that|these|those|my|your|his|her|its|our|their)\s+{re.escape(w)}\b", matching_sent.lower()))
                            )
                            if not is_locked_np:
                                cand_pool_all = cls.build_top_k_vocab_candidates(
                                    w, matching_sent, pos=None, k=8, discourse_context=discourse_ctx
                                )
                                alt_pool = [
                                    c for c in cand_pool_all
                                    if str(c.get("pos") or "").strip().lower() != str(pos).strip().lower()
                                ] or cand_pool_all
                                if alt_pool and alt_pool != cand_pool and len(alt_pool) > 1:
                                    retry_idx = cls.disambiguate_via_llm(
                                        w, matching_sent, alt_pool, discourse_context=discourse_ctx
                                    )
                                    if 0 <= retry_idx < len(alt_pool):
                                        chosen = alt_pool[retry_idx]
                        if chosen is None:
                            logger.info(
                                f"  [DROP_WSD_REJECTED] '{w}' rejected by WSD arbitration in [{matching_sid}], advancing to next candidate."
                            )
                            continue
                    else:
                        chosen = cand_pool[0]
                    definition = chosen.get("definition", "")
                    locked_raw_sense = chosen.get("raw_sense")
                    chosen_pos = str(chosen.get("pos") or "").strip().lower()
                    if chosen_pos:
                        for k_p, v_p in (("adv", "adverb"), ("noun", "noun"), ("verb", "verb"), ("adj", "adjective"), ("prep", "preposition"), ("conj", "conjunction")):
                            if k_p in chosen_pos:
                                pos = v_p
                                break
                    exs = chosen.get("examples") or []
                    if exs:
                        example = exs[0]
                    lock_score = chosen.get("score", 0.0)
                    lock_margin = chosen.get("margin", 0.0)
                    lock_confidence = chosen.get("confidence", "high")

                if not definition and not (matching_sent and cls.get_pedlex_entry(w) and not cand_pool):
                    definition, example = cls.get_ldoce_definition_and_example(
                        w, target_pos=pos, context_sentence=matching_sent,
                        require_sense_confidence=False
                    )
                if cls._is_pedlex_disabled() and not definition and not (matching_sent and cls.get_pedlex_entry(w) and not cand_pool):
                    # Fallback to WordNet definition ONLY when PedLex is disabled
                    try:
                        best_s = cls.disambiguate_wordnet_synset(w, pos=pos, context_sentence=matching_sent)
                        if best_s:
                            definition = best_s.definition()
                    except Exception:
                        pass

                if pos == "adjective" and definition and re.match(r"^to\s+be\s+", definition.strip(), flags=re.IGNORECASE):
                    definition = re.sub(r"^to\s+be\s+", "", definition.strip(), flags=re.IGNORECASE)
                if definition and re.match(r"^to\s+[a-z]+\b", definition.strip().lower()) and pos in ("noun", "adjective", "adverb", "function_word"):
                    if not locked_raw_sense or not locked_raw_sense.get("pos"):
                        pos = "verb"
                if pos == "function_word":
                    pos = "adverb"
                # Dictionary-First Canonical Headword Normalization:
                # If a participial/inflected word (e.g. 'absorbed', 'condensed') has no standalone adjective entry
                # in the dictionary and resolves to its base verb entry (pos == 'verb'), normalize the headword
                # to the dictionary's canonical base verb ('absorb', 'condense') so headword and verb POS/definition agree.
                if pos == "verb" and " " not in w and "-" not in w:
                    ld_e = cls.get_ldoce_entry(w)
                    if ld_e:
                        dict_hw = str(ld_e.get("word") or "").strip().lower()
                        if (
                            dict_hw
                            and dict_hw != w
                            and cls.are_same_word_family(w, dict_hw)
                            and not cls.are_spelling_variants(w, dict_hw)
                        ):
                            w = dict_hw
                    elif cls._is_pedlex_disabled():
                        lem_v = cls._surface_lemma(w)
                        if lem_v and lem_v != w:
                            wn = cls.get_wordnet()
                            if wn and wn.synsets(lem_v, pos="v"):
                                w = lem_v
                if w in seen_words or (not cand.get("is_syllabus") and any(cls.are_same_word_family(w, sw) for sw in seen_words)):
                    continue
                if not example and not locked_raw_sense:
                    example = cls.ldoce_sense_example(w, pos=pos, quote=matching_sent)
                example = cls.ensure_distinct_example(w, pos, example, matching_sent, sense=locked_raw_sense)

                # If the passage contains multiple attested sentences for `w` (`len(shown) > 1`) and the first picked sentence
                # maps to an example-less sub-sense (e.g. 'log out' vs 'log in'), evaluate alternate attested sentences in `shown`
                # before dropping the word at Gate 2.
                if (not definition or cls.is_boilerplate_definition(definition) or not example) and len(shown) > 1:
                    for alt_sid, alt_sent in shown:
                        if alt_sent == matching_sent:
                            continue
                        if any(cls.are_same_word_family(w, sw) and alt_sent == sq for sw, sq in seen_word_quotes):
                            continue
                        alt_disc = ""
                        if alt_sid and pool:
                            m_alt_num = re.match(r"S-(\d+)", str(alt_sid))
                            if m_alt_num:
                                a_num = int(m_alt_num.group(1))
                                alt_disc = " ".join(pool[f"S-{a_num + d}"] for d in (-2, -1, 1, 2) if f"S-{a_num + d}" in pool)
                        alt_pool = cls.build_top_k_vocab_candidates(w, alt_sent, pos=cand.get("part_of_speech") or pos, k=16, discourse_context=alt_disc)
                        if not alt_pool:
                            continue
                        alt_chosen = alt_pool[cls.disambiguate_via_llm(w, alt_sent, alt_pool, discourse_context=alt_disc)] if (len(alt_pool) > 1 and enable_llm_disambiguation) else alt_pool[0]
                        alt_def = alt_chosen.get("definition", "")
                        alt_raw = alt_chosen.get("raw_sense")
                        alt_pos = pos
                        alt_cpos = str(alt_chosen.get("pos") or "").strip().lower()
                        for k_p, v_p in (("adv", "adverb"), ("noun", "noun"), ("verb", "verb"), ("adj", "adjective"), ("prep", "preposition"), ("conj", "conjunction")):
                            if k_p in alt_cpos:
                                alt_pos = v_p
                                break
                        alt_exs = alt_chosen.get("examples") or []
                        alt_ex = cls.ensure_distinct_example(w, alt_pos, alt_exs[0] if alt_exs else "", alt_sent, sense=alt_raw)
                        if alt_def and not cls.is_boilerplate_definition(alt_def) and alt_ex:
                            matching_sent = alt_sent
                            matching_sid = alt_sid
                            definition = alt_def
                            pos = alt_pos
                            example = alt_ex
                            locked_raw_sense = alt_raw
                            lock_score = alt_chosen.get("score", 0.0)
                            lock_margin = alt_chosen.get("margin", 0.0)
                            lock_confidence = alt_chosen.get("confidence", "high")
                            break

                # Gate 2 (Evidence-Completeness Gate - 宁缺勿滥):
                # Drop any candidate lacking a valid definition OR an authentic, non-duplicate dictionary example
                # for its locked sense, allowing a fully attested reserve candidate to take its slot.
                # If a mandated syllabus item lacks a distinct dictionary example, use the authentic passage sentence.
                if not definition or cls.is_boilerplate_definition(definition) or not example:
                    if cand.get("is_syllabus") and definition and not cls.is_boilerplate_definition(definition) and matching_sent and cls.text_contains_form(matching_sent, w):
                        example = matching_sent
                    else:
                        continue
                # Gate 2b (Sense-Level Pedagogical Filter):
                # Drop single-word nouns locked to:
                # (1) Phrase-bound bare noun senses ('odds' -> "if two statements... are at odds", 'sake' -> "in order to help...")
                # (2) Everyday [plural] commodity aggregate senses ('electronics' -> "[plural] electronic equipment, or equipment with electronic parts")
                if pos == "noun" and " " not in w:
                    _def_s_low = definition.strip().lower()
                    _locked_gram_low = str((locked_raw_sense or {}).get("grammar") or "").lower()
                    if _def_s_low.startswith(("if two ", "if people ", "in order to help ", "used to say ")):
                        continue
                    if "[plural]" in _locked_gram_low and "[uncountable]" not in _locked_gram_low:
                        if any(
                            _def_s_low.startswith(p)
                            for p in (
                                "electronic equipment",
                                "food and other goods",
                                "substances that you put on your face",
                                "things such as soap",
                                "drinks and small amounts of food",
                            )
                        ):
                            continue
                def_norm_key = re.sub(r"[^a-z0-9]+", " ", definition.lower()).strip()
                if def_norm_key in seen_defs:
                    continue

                audit_sid = f"[{matching_sid}]" if matching_sid else "[S-1]"
                results.append({
                    "word": w,
                    "part_of_speech": pos,
                    "definition": definition,
                    "quoted_sentence": matching_sent,
                    "example_usage": example,
                    "anchor_quality": anchor_quality,
                    "sense_confidence": lock_confidence,
                    "sense_lock_score": lock_score,
                    "sense_lock_margin": lock_margin,
                    "design_audit": f"AUDIT: {audit_sid} -> [{w}] -> [VERBATIM_CONFIRMED]"
                })
                seen_words.add(w)
                seen_defs.add(def_norm_key)
                if matching_sent:
                    seen_word_quotes.append((w, matching_sent))

            return results

    @classmethod
    def _find_surface_span_in_quote(cls, headword: str, quote: str) -> str:
        """
        Deterministically resolves the exact verbatim surface substring inside `quote`
        corresponding to `headword` (handling inflections, `be` copula forms, and `[sth]`/`[sb]` slots).
        """
        if not headword or not quote:
            return headword or ""
        # Preserve internal possessive slots ([one's] -> one's) while stripping object placeholders ([sth], [sb])
        clean_hw = re.sub(r"\[(one's|someone's|somebody's)\]", "one's", headword, flags=re.IGNORECASE)
        clean_hw = re.sub(r"\[(?:sth|sb|something|somebody|[^\]]+)\]", "", clean_hw).strip()
        clean_hw = re.sub(r"\s+", " ", clean_hw).strip()
        if not clean_hw:
            return headword

        # Direct case-insensitive substring check first (with word boundaries)
        m_direct = re.search(rf"\b{re.escape(clean_hw)}\b", quote, flags=re.IGNORECASE)
        if m_direct:
            return m_direct.group(0)

        nlp = cls.get_spacy()
        if not nlp:
            return clean_hw

        hw_toks = [t.lower() for t in re.findall(r"[A-Za-z']+", clean_hw)]
        if not hw_toks:
            return clean_hw

        BE_FORMS = {"be", "am", "is", "are", "was", "were", "been", "being", "'m", "'s", "'re"}
        HAVE_FORMS = {"have", "has", "had", "having", "'ve", "'d"}
        ONES_FORMS = {"one's", "my", "your", "his", "her", "its", "our", "their"}

        q_doc = nlp(quote)
        q_words = [t for t in q_doc if not t.is_punct and not t.is_space]
        if not q_words:
            return clean_hw

        def _tok_matches(hw_t: str, doc_tok) -> bool:
            surf = doc_tok.text.lower()
            lem = doc_tok.lemma_.lower()
            if hw_t == surf or hw_t == lem:
                return True
            if hw_t == "be" and (surf in BE_FORMS or lem == "be"):
                return True
            if hw_t == "have" and (surf in HAVE_FORMS or lem == "have"):
                return True
            if hw_t in ("one's", "someone's", "somebody's") and surf in ONES_FORMS:
                return True
            if len(hw_t) >= 4 and (surf.startswith(hw_t[:4]) or lem.startswith(hw_t[:4])):
                return True
            return False

        n_hw = len(hw_toks)
        # 1. Contiguous token window match
        if n_hw == 1:
            for wt in q_words:
                if _tok_matches(hw_toks[0], wt):
                    return wt.text
        else:
            for i in range(len(q_words) - n_hw + 1):
                if all(_tok_matches(hw_toks[j], q_words[i + j]) for j in range(n_hw)):
                    start_char = q_words[i].idx
                    end_char = q_words[i + n_hw - 1].idx + len(q_words[i + n_hw - 1].text)
                    # Include contractions attached to the first word (e.g. "She's made up of" -> "'s made up of" or "is made up of")
                    return quote[start_char:end_char]
            # Allow 1 intervening modifier/adverb token inside multi-word expressions
            for i in range(len(q_words)):
                if not _tok_matches(hw_toks[0], q_words[i]):
                    continue
                cur_q = i
                matched_all = True
                for j in range(1, n_hw):
                    found_next = False
                    for step in (1, 2):
                        if cur_q + step < len(q_words) and _tok_matches(hw_toks[j], q_words[cur_q + step]):
                            cur_q = cur_q + step
                            found_next = True
                            break
                    if not found_next:
                        matched_all = False
                        break
                if matched_all:
                    start_char = q_words[i].idx
                    end_char = q_words[cur_q].idx + len(q_words[cur_q].text)
                    return quote[start_char:end_char]

        return clean_hw

    @classmethod
    def extract_video_vocabulary(
        cls,
        transcript_text: str,
        target_count: int = 12,
        enable_llm_disambiguation: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Deterministically extracts contextual single-word vocabulary and multi-word expressions
        from a timestamped video transcript after sentence-level reflow (`VideoService.reflow_transcript_lines`).
        Returns chronologically sorted items with timestamp, seconds, verbatim surface match,
        definition, example_usage, pattern, and collocations (strictly without IPA or CEFR).
        """
        if not transcript_text or not transcript_text.strip():
            return []

        try:
            from ..video import VideoService
            reflowed = VideoService.reflow_transcript_lines(transcript_text)
        except Exception:
            reflowed = transcript_text

        ts_line_re = re.compile(r"^\s*\[(\d{1,2}:\d{2}(?::\d{2})?(?:\.\d{1,3})?)\]\s*(.+)$")
        ts_sentences: List[Tuple[str, float, str]] = []
        for raw_line in reflowed.splitlines():
            m = ts_line_re.match(raw_line)
            if m:
                raw_ts = m.group(1).strip()
                sent_txt = m.group(2).strip()
                if not sent_txt:
                    continue
                # Parse seconds
                parts = raw_ts.split(":")
                try:
                    if len(parts) == 3:
                        sec = int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
                    else:
                        sec = int(parts[0]) * 60 + float(parts[1])
                except ValueError:
                    sec = 0.0
                mm = int(sec // 60)
                ss = int(sec % 60)
                display_ts = f"{mm:02d}:{ss:02d}"
                ts_sentences.append((display_ts, round(sec, 2), sent_txt))

        if not ts_sentences:
            return []

        clean_passage = "\n".join(s[2] for s in ts_sentences)
        expr_quota = max(4, target_count // 2)
        expr_items = cls.extract_deterministic_expressions(
            clean_passage,
            target_count=expr_quota,
            enable_llm_disambiguation=enable_llm_disambiguation,
        )

        # Exclude content words already covered by extracted expressions
        covered_lemmas = set()
        for ex_it in expr_items:
            w_clean = re.sub(r"\[[^\]]+\]", "", ex_it.get("word", "")).lower()
            for tok in re.findall(r"[a-z]{4,}", w_clean):
                covered_lemmas.add(tok)

        vocab_quota = max(4, target_count - len(expr_items))
        vocab_items = cls.extract_deterministic_vocabulary(
            clean_passage,
            target_count=vocab_quota,
            exclude_words=covered_lemmas,
            enable_llm_disambiguation=enable_llm_disambiguation,
        )

        combined = list(expr_items) + list(vocab_items)
        enriched_list: List[Dict[str, Any]] = []

        for item in combined:
            word = str(item.get("word") or "").strip()
            pos = str(item.get("part_of_speech") or "").strip()
            defn = str(item.get("definition") or "").strip()
            quote = str(item.get("quoted_sentence") or "").strip()
            ex_usage = str(item.get("example_usage") or "").strip()
            if not word or not defn or not quote:
                continue

            # Match quote back to timestamped sentence
            matched_ts = "00:00"
            matched_sec = 0.0
            norm_q = re.sub(r"\s+", " ", quote).strip().lower()
            for d_ts, d_sec, s_txt in ts_sentences:
                norm_s = re.sub(r"\s+", " ", s_txt).strip().lower()
                if norm_q == norm_s or norm_q in norm_s or norm_s in norm_q:
                    matched_ts = d_ts
                    matched_sec = d_sec
                    break

            surface_match = cls._find_surface_span_in_quote(word, quote)
            enrichment = {}
            try:
                enrichment = cls.get_vocab_card_enrichment(word, pos, defn, quote) or {}
            except Exception:
                enrichment = {}

            pattern_str = str(enrichment.get("pattern_note") or "").strip()
            collocs_raw = enrichment.get("collocations") or []
            if isinstance(collocs_raw, str):
                collocs_list = [c.strip() for c in collocs_raw.split(";") if c.strip()][:3]
            elif isinstance(collocs_raw, list):
                collocs_list = [str(c).strip() for c in collocs_raw if str(c).strip()][:3]
            else:
                collocs_list = []

            enriched_list.append({
                "word": word,
                "surface_match": surface_match,
                "part_of_speech": pos,
                "timestamp": matched_ts,
                "seconds": matched_sec,
                "definition": defn,
                "quoted_sentence": quote,
                "example_usage": ex_usage,
                "pattern": pattern_str,
                "collocations": collocs_list,
            })

        enriched_list.sort(key=lambda x: (x["seconds"], x["word"]))
        return enriched_list

    # =========================================================================

