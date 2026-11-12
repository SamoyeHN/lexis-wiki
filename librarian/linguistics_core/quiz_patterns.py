"""
Five-source authentic example selection and atomic pattern bundle extraction.
Pure 5 LDOCE compartments (Oxford Thesaurus and raw BNC fragments permanently excluded).
"""

import datetime
import hashlib
import json
import os
import re
import sqlite3
import threading
import zlib
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union


class QuizPatternsMixin:
    @classmethod
    def _is_complete_finite_sentence(cls, text: Optional[str]) -> bool:
        """
        Determines whether a candidate dictionary example is a complete finite clause/sentence
        (ending in `.?!`, containing >= 5 alphabetic words and a finite verb/predication)
        rather than a verbless NP/PP fragment (e.g., 'the rise of fascism', 'a team of experts').
        """
        if not text or not isinstance(text, str):
            return False
        ex_s = re.sub(r"\s*\(=[^)]*\)", "", text).strip().strip('"').strip()
        if not ex_s or ex_s[-1:] not in ".?!":
            return False
        nlp = cls.get_spacy()
        if not nlp:
            return len(re.findall(r"[A-Za-z]+", ex_s)) >= 5
        try:
            doc = nlp(ex_s)
        except Exception:
            return len(re.findall(r"[A-Za-z]+", ex_s)) >= 5
        alpha_toks = [t for t in doc if t.is_alpha]
        if len(alpha_toks) < 5:
            return False
        return any(
            (
                t.pos_ in ("VERB", "AUX")
                and (t.dep_ in ("ROOT", "relcl", "ccomp", "advcl", "conj") or t.tag_ in ("VBD", "VBP", "VBZ", "MD"))
            )
            or (
                t.dep_ == "ROOT" and any(c.dep_ in ("nsubj", "nsubjpass") for c in t.children)
            )
            or (
                t.dep_ == "amod"
                and t.head.dep_ == "ROOT"
                and t.head.pos_ in ("NOUN", "PROPN")
                and t.i > t.head.i
                and any(c.dep_ in ("prep", "advmod", "npadvmod") for c in t.children)
            )
            for t in doc
        )

    @classmethod
    def _example_attests_prep_valency(
        cls,
        ex_text: Optional[str],
        target_word: str,
        prep: Optional[str],
        prep_dir: str = "post",
    ) -> bool:
        """
        Directionally verifies that `ex_text` attests `prep` attached to `target_word`:
        - `prep_dir == "pre"`: `prep` governs `target_word` on the left (`in distress`, `on the staff`).
        - `prep_dir == "post"`: `prep` immediately follows `target_word` (`rise to`, `donation to`, `downturn in`)
          or attaches as a post-nominal `prep` child of `target_word` without another preposition intervening
          immediately after `target_word` (preventing 'rise to the top of ...' from falsely matching 'rise of').
        """
        if not ex_text or not target_word or not prep:
            return False
        tw_l = target_word.strip().lower()
        p_l = prep.strip().lower()
        ex_l = ex_text.strip().lower()
        last_t = tw_l.split()[-1] if tw_l else tw_l
        t_forms = {tw_l, last_t} | set(cls.inflected_forms(last_t))
        if prep_dir == "pre":
            return any(
                bool(re.search(rf"\b{re.escape(p_l)}\b(?:\s+(?:a|an|the|his|her|their|its|our|my|your|[a-z]+)){{0,2}}\s+\b{re.escape(tf)}\b", ex_l))
                for tf in t_forms if tf
            )
        # Post-head preposition check (`target + prep`)
        _other_preps = {"to", "of", "in", "for", "on", "with", "from", "by", "at", "into", "over", "between", "against", "towards", "upon", "about", "under"} - {p_l}
        for tf in t_forms:
            if not tf:
                continue
            if re.search(rf"\b{re.escape(tf)}\s+{re.escape(p_l)}\b", ex_l):
                return True
            if re.search(rf"\b{re.escape(tf)}\s+and\s+[a-z]+\s+{re.escape(p_l)}\b", ex_l):
                return True
        nlp = cls.get_spacy()
        if nlp:
            try:
                doc = nlp(ex_text)
                for tok in doc:
                    if tok.text.lower() in t_forms or tok.lemma_.lower() == last_t:
                        # Reject if another preposition immediately follows `tok` (`rise to the top of`)
                        if tok.i + 1 < len(doc) and doc[tok.i + 1].text.lower() in _other_preps:
                            continue
                        for ch in tok.children:
                            if ch.dep_ == "prep" and ch.i > tok.i and ch.text.lower() == p_l:
                                return True
            except Exception:
                pass
        return False

    @classmethod
    def _find_target_span_in_doc(
        cls,
        doc: Any,
        target_str: str,
        base_str: Optional[str] = None,
        t_forms: Optional[Set[str]] = None,
    ) -> Tuple[Optional[int], Optional[int], Optional[Any]]:
        """
        Locates `target_str` (either a single token or a hyphenated compound like `brand-new`,
        `world-class`, `in-person`, `real-life`) inside a spaCy `Doc`.
        Returns `(start_i, end_i, head_tok)` where `doc[start_i:end_i]` is the full surface span
        and `head_tok` is the syntactic head token of the target in `doc`.
        """
        if doc is None or not target_str:
            return None, None, None
        ft_low = target_str.strip().lower()
        bw_low = (base_str or ft_low).strip().lower()
        forms = set(t_forms or set()) | {ft_low, bw_low}

        if "-" in ft_low or "-" in bw_low:
            for tok in doc:
                if tok.text.lower() in (ft_low, bw_low) or tok.lemma_.lower() in (ft_low, bw_low):
                    return tok.i, tok.i + 1, tok
            hyph_str = ft_low if "-" in ft_low else bw_low
            parts = [p for p in hyph_str.split("-") if p]
            if len(parts) >= 2:
                span_len = 2 * len(parts) - 1
                for i in range(len(doc) - span_len + 1):
                    matched = True
                    for k, part in enumerate(parts):
                        tok_k = doc[i + 2 * k]
                        if tok_k.text.lower() != part and tok_k.lemma_.lower() != part:
                            matched = False
                            break
                        if k + 1 < len(parts):
                            sep_tok = doc[i + 2 * k + 1]
                            if sep_tok.text not in ("-", "–", "—"):
                                matched = False
                                break
                    if matched:
                        end_i = i + span_len
                        head_tok = doc[end_i - 1]
                        for sub_t in doc[i:end_i]:
                            if sub_t.dep_ == "amod" and sub_t.head.pos_ in ("NOUN", "PROPN"):
                                head_tok = sub_t
                                break
                        return i, end_i, head_tok
            return None, None, None

        for tok in doc:
            if (tok.i > 0 and doc[tok.i - 1].text == "-" and not doc[tok.i - 1].whitespace_) or (
                tok.i + 1 < len(doc) and doc[tok.i + 1].text == "-" and not tok.whitespace_
            ):
                continue
            if tok.text.lower() == ft_low:
                return tok.i, tok.i + 1, tok
        for tok in doc:
            if (tok.i > 0 and doc[tok.i - 1].text == "-" and not doc[tok.i - 1].whitespace_) or (
                tok.i + 1 < len(doc) and doc[tok.i + 1].text == "-" and not tok.whitespace_
            ):
                continue
            if tok.text.lower() in forms or tok.lemma_.lower() == bw_low:
                return tok.i, tok.i + 1, tok
        return None, None, None

    @classmethod
    def extract_sense_pattern_bundles(cls, sense: Dict[str, Any], target_word: str, quote: str = "") -> List[Dict[str, Any]]:
        """
        Extracts atomic (pattern, example, bound_prep) bundles from a locked PedLex sense.
        Prioritizes native atomic pattern_bundles directly extracted from PedLex DOM.
        Falls back to parsing raw patterns while strictly preventing pattern-example desynchronization.
        """
        if not sense:
            return []

        verb_colloc_bundles = cls.extract_sense_verified_noun_verb_collocations(sense, target_word, quote=quote)

        # 1. Native Atomic Bundles (Directly from updated PedLex DB schema)
        native_bundles = list(sense.get("pattern_bundles") or [])
        sense_examples = sense.get("examples") or []
        sense_units = [
            u for u in (sense.get("units") or [])
            if isinstance(u, str) and " " in u.strip()
        ]
        if native_bundles:
            existing_pats_low = {str(nb.get("pattern") or "").strip().lower() for nb in native_bundles}
            for u_str in sense_units:
                if u_str.strip().lower() not in existing_pats_low:
                    native_bundles.append({"pattern": u_str.strip(), "example": None, "source": "sense_unit"})
        if native_bundles:
            out: List[Dict[str, Any]] = []
            t_low = (target_word or "").strip().lower()
            t_vars = cls.orthographic_variants(t_low) if (t_low and hasattr(cls, "orthographic_variants")) else ([t_low] if t_low else [])
            t_low_esc = "(?:" + "|".join(re.escape(v) for v in (t_vars or [t_low])) + ")" if t_low else ""
            for nb in native_bundles:
                raw_pat = (nb.get("pattern") or "").strip()
                if not raw_pat:
                    continue
                m_emb = re.search(r"^(.*?)\s+([A-Z][\w\W]*\b[\.\?!])$", raw_pat)
                if not m_emb:
                    m_etc = re.search(r"^(.*?\betc)\s+([A-Za-z].*\s+.*)$", raw_pat)
                    if m_etc:
                        m_emb = m_etc
                if m_emb:
                    pat = m_emb.group(1).strip()
                    emb_ex = m_emb.group(2).strip()
                else:
                    pat = raw_pat
                    emb_ex = None
                if not pat:
                    continue
                p_clean_search = re.sub(r"\[.*?\]|\(.*?\)", "", pat).strip().lower()
                # Skip bare-NP British/idiomatic prepositional frames (`member of <target>`, `on <target>`,
                # e.g. `member of staff`, `on staff`) when the sense has full-clause examples, because
                # `member of ____` / `on ____` forces an articleless prepositional slot that clashes
                # syntactically with collective noun distractors (`a member of the crew`, `on the committee`).
                if (
                    t_low_esc
                    and re.match(rf"^(?:(?:a\s+)?members?\s+of|on)\s+{t_low_esc}$", p_clean_search)
                    and len(sense_examples) > 0
                ):
                    continue
                inf_valency = bool(re.search(r"\bto\s+do(?:\s+(?:something|somebody|sth|sb))?\b", pat.lower()))
                gerund_valency = bool(re.search(r"\bdoing(?:\s+(?:something|somebody|sth|sb))?\b", pat.lower()))
                p_for_prep = re.sub(r"\bto\s+(?:do(?:\s+something|\s+sth)?|\w+ing)\b", "", p_clean_search)
                _is_adj_sense_p = "adj" in str(sense.get("pos") or "").lower()
                if t_low_esc:
                    p_for_prep = re.sub(rf"\bto\s+{t_low_esc}\b", t_low, p_for_prep)
                    if "verb" in str(sense.get("pos") or "").lower():
                        m_t_pos = re.search(rf"\b{t_low_esc}\b", p_for_prep)
                        if m_t_pos:
                            p_for_prep = p_for_prep[m_t_pos.end():]
                    elif _is_adj_sense_p:
                        # For adjectives, only extract prepositions that immediately follow the target adjective
                        # (or its slash alternative), NEVER prepositions before the NP ('on an unprecedented scale')
                        # or after a modified noun ('in the unlikely event of something', 'a compulsory course of'),
                        # nor 'for'/'of' inside 'it is <adj> (for/of sb) to do', nor tough-movement infinitives ('simple to use').
                        if re.search(rf"\bit\s+is\s+{t_low_esc}\b.*\bto\b", p_clean_search):
                            p_for_prep = ""
                            inf_valency = True
                        else:
                            m_t_adj = re.search(rf"\b{t_low_esc}(?:\s*/\s*[a-z]+)*\b", p_for_prep)
                            if m_t_adj:
                                tail_after_adj = p_for_prep[m_t_adj.end():].strip()
                                m_to_inf_adj = re.match(r"^to\s+([a-z]+)\b", tail_after_adj)
                                if m_to_inf_adj:
                                    w_after_to = m_to_inf_adj.group(1)
                                    if w_after_to not in (
                                        "somebody", "someone", "something", "sb", "sth", "oneself", "one",
                                        "a", "an", "the", "this", "that", "these", "those", "my", "your",
                                        "his", "her", "its", "our", "their", "all", "any", "each", "every",
                                        "other", "others", "another", "most", "many", "much", "few", "both",
                                        "men", "women", "people", "children", "life", "death", "work", "school",
                                    ) and (
                                        w_after_to in ("do", "be", "use", "make", "get", "have", "see", "find", "understand", "learn", "read", "write", "follow", "operate", "manage", "control", "handle", "reach", "remember", "forget", "imagine", "believe", "explain", "solve", "fix", "build", "create", "achieve", "attain", "bear", "tolerate", "resist", "avoid", "prevent", "predict", "measure", "detect", "spot", "hear", "watch", "listen", "eat", "drink", "digest", "swallow", "carry", "lift", "move", "open", "close", "clean", "wash", "wear", "afford", "buy", "sell", "pay", "please", "satisfy", "convince", "persuade", "deal", "live", "talk")
                                        or ("v" in cls._wordnet_pos(w_after_to) and "n" not in cls._wordnet_pos(w_after_to))
                                    ):
                                        inf_valency = True
                                        tail_after_adj = ""
                                m_imm_p = re.match(
                                    r"^((?:(?:on|in|at|for|with|about|under|from|to|by|into|of|between|over|against|towards|upon)(?:\s*/\s*)?)+)\b",
                                    tail_after_adj,
                                )
                                p_for_prep = m_imm_p.group(1) if m_imm_p else ""
                            else:
                                p_for_prep = ""
                if _is_adj_sense_p:
                    all_preps = re.findall(r"\b(on|in|at|for|with|about|under|from|to|by|into|of|between|over|against|towards|upon)\b", p_for_prep)
                else:
                    all_preps = re.findall(r"\b(on|in|at|for|with|about|under|from|to|by|into|of|between|over|against|towards|upon|as)\b", p_for_prep)
                cand_preps: List[str] = []
                for cp in all_preps:
                    if cp not in cand_preps:
                        cand_preps.append(cp)
                bound_prep = cand_preps[0] if cand_preps else None
                prep_dir = "post"
                if bound_prep and t_low_esc and not _is_adj_sense_p:
                    m_tw_p = re.search(rf"\b{t_low_esc}\w*\b", p_for_prep)
                    m_bp_p = re.search(rf"\b{re.escape(bound_prep)}\b", p_for_prep)
                    if m_tw_p and m_bp_p and m_bp_p.start() < m_tw_p.start():
                        prep_dir = "pre"
                matched_nb_ex = nb.get("example") or emb_ex or None
                # If `matched_nb_ex` is missing OR is a verbless NP/PP fragment ('the rise of fascism',
                # 'our only chance of escape'), search `sense_examples` for a complete finite sentence
                # that directionally attests the pattern.
                if (not matched_nb_ex or not cls._is_complete_finite_sentence(matched_nb_ex)) and sense_examples:
                    full_sent_match = None
                    any_sent_match = None
                    for s_ex in sense_examples:
                        s_ex_str = s_ex.get("text") if isinstance(s_ex, dict) else str(s_ex or "")
                        if not s_ex_str or not cls.text_contains_form(s_ex_str, t_low):
                            continue
                        s_ex_l = s_ex_str.lower()
                        attests_pat = False
                        if inf_valency and t_low:
                            attests_pat = bool(re.search(rf"\b{t_low_esc}\w*\b(?:\s+[a-z]+){{0,3}}\s+to\s+[a-z]{{2,}}\b", s_ex_l))
                        elif gerund_valency and t_low:
                            lead_p = bound_prep or "of"
                            attests_pat = bool(re.search(rf"\b{t_low_esc}\w*\b(?:\s+[a-z]+){{0,2}}\s+{re.escape(lead_p)}\s+[a-z]{{2,}}ing\b", s_ex_l))
                        elif cand_preps:
                            attests_pat = any(
                                cls._example_attests_prep_valency(s_ex_str, t_low, cp, prep_dir=prep_dir)
                                for cp in cand_preps
                            )
                        else:
                            branches = cls.clean_and_expand_pattern_slashes(pat) or [pat]
                            for br in branches:
                                br_words = [
                                    w for w in re.split(r"\W+", br.lower())
                                    if w and w not in ("something", "somebody", "someone", "sb", "sth", "etc", "doing", "do")
                                ]
                                if br_words and all(
                                    re.search(rf"\b{re.escape(w)}\b", s_ex_l)
                                    or cls.form_matches(w, s_ex_l)
                                    or any(re.search(rf"\b{re.escape(inf)}\b", s_ex_l) for inf in cls.inflected_forms(w))
                                    for w in br_words
                                ):
                                    attests_pat = True
                                    break
                        if attests_pat:
                            if any_sent_match is None:
                                any_sent_match = s_ex_str
                            if cls._is_complete_finite_sentence(s_ex_str):
                                full_sent_match = s_ex_str
                                break
                    if full_sent_match:
                        matched_nb_ex = full_sent_match
                    elif not matched_nb_ex and any_sent_match:
                        matched_nb_ex = any_sent_match
                if matched_nb_ex and len(cand_preps) > 1:
                    for cp in cand_preps:
                        if cls._example_attests_prep_valency(matched_nb_ex, t_low, cp, prep_dir=prep_dir):
                            bound_prep = cp
                            break
                out.append({
                    "pattern": pat,
                    "example": matched_nb_ex,
                    "prep": bound_prep,
                    "preps": cand_preps,
                    "prep_dir": prep_dir,
                    "inf_valency": inf_valency,
                    "gerund_valency": gerund_valency,
                    "source": nb.get("source", "native_bundle")
                })
            return out + verb_colloc_bundles

        # 2. Fallback for older DB versions: parse flat patterns with exact scoring
        raw_patterns = list(sense.get("patterns") or []) + [
            u for u in sense_units if u not in (sense.get("patterns") or [])
        ]
        examples = sense.get("examples") or []
        bundles: List[Dict[str, Any]] = []
        seen_patterns = set()

        for raw in raw_patterns:
            raw_s = raw.strip()
            if not raw_s:
                continue
            m_emb = re.search(r"^(.*?)\s+([A-Z][\w\W]*\b[\.\?!])$", raw_s)
            if not m_emb:
                m_etc = re.search(r"^(.*?\betc)\s+([A-Za-z].*\s+.*)$", raw_s)
                if m_etc:
                    m_emb = m_etc
            if m_emb:
                clean_pat = m_emb.group(1).strip()
                emb_ex = m_emb.group(2).strip()
            else:
                clean_pat = raw_s
                emb_ex = None

            if clean_pat in seen_patterns:
                continue
            seen_patterns.add(clean_pat)

            p_clean_search = re.sub(r"\[.*?\]|\(.*?\)", "", clean_pat).strip().lower()
            p_for_prep = re.sub(r"\bto\s+(?:do\s+something|\w+ing)\b", "", p_clean_search)
            if "adj" in str(sense.get("pos") or "").lower() and target_word:
                tw_low_fb = target_word.strip().lower()
                tw_vars_fb = cls.orthographic_variants(tw_low_fb) if hasattr(cls, "orthographic_variants") else [tw_low_fb]
                tw_esc_fb = "(?:" + "|".join(re.escape(v) for v in (tw_vars_fb or [tw_low_fb])) + ")"
                if re.search(rf"\bit\s+is\s+{tw_esc_fb}\b.*\bto\b", p_clean_search):
                    p_for_prep = ""
                else:
                    m_t_adj_fb = re.search(rf"\b{tw_esc_fb}(?:\s*/\s*[a-z]+)*\b", p_for_prep)
                    if m_t_adj_fb:
                        tail_fb = p_for_prep[m_t_adj_fb.end():].strip()
                        m_imm_fb = re.match(
                            r"^((?:(?:on|in|at|for|with|about|under|from|to|by|into|of|between|over|against|towards|upon)(?:\s*/\s*)?)+)\b",
                            tail_fb,
                        )
                        p_for_prep = m_imm_fb.group(1) if m_imm_fb else ""
                    else:
                        p_for_prep = ""
            if "adj" in str(sense.get("pos") or "").lower():
                all_preps = re.findall(r"\b(on|in|at|for|with|about|under|from|to|by|into|of|between|over|against|towards|upon)\b", p_for_prep)
            else:
                all_preps = re.findall(r"\b(on|in|at|for|with|about|under|from|to|by|into|of|between|over|against|towards|upon|as)\b", p_for_prep)
            # Deduplicate while preserving order
            cand_preps: List[str] = []
            for cp in all_preps:
                if cp not in cand_preps:
                    cand_preps.append(cp)
            bound_prep = cand_preps[0] if cand_preps else None

            matched_ex = emb_ex
            if not matched_ex:
                # 1. Match 'it [verb] somebody when/how/that'
                m_it = re.search(r"^it\s+(\w+)\s+somebody\s+(when|how|that)", clean_pat.lower())
                if m_it:
                    v_stem = m_it.group(1)
                    conj = m_it.group(2)
                    for ex in examples:
                        if re.search(r"\bit\b", ex.lower()) and v_stem in ex.lower() and re.search(rf"\b{conj}\b", ex.lower()):
                            matched_ex = ex
                            break
                # 2. Direct exact or branch/keyword match of clean_pat
                if not matched_ex:
                    branches = cls.clean_and_expand_pattern_slashes(clean_pat) or [clean_pat]
                    best_ex = None
                    best_match_count = 0
                    for ex in examples:
                        ex_l = ex.lower()
                        if clean_pat.lower() in ex_l:
                            matched_ex = ex
                            break
                        for br in branches:
                            br_words = [w for w in re.split(r"\W+", br.lower()) if w and w not in ("something", "somebody", "sb", "sth", "etc", "doing")]
                            if br_words:
                                matched_words = sum(1 for w in br_words if re.search(rf"\b{re.escape(w)}\b", ex_l) or cls.form_matches(w, ex_l))
                                # Also check morphological inflections of target word in example
                                if matched_words < len(br_words):
                                    matched_words = sum(
                                        1 for w in br_words
                                        if re.search(rf"\b{re.escape(w)}\b", ex_l)
                                        or any(re.search(rf"\b{re.escape(inf)}\b", ex_l) for inf in cls.inflected_forms(w))
                                    )
                                if matched_words == len(br_words) and matched_words > best_match_count:
                                    best_match_count = matched_words
                                    best_ex = ex
                    if not matched_ex and best_ex:
                        matched_ex = best_ex

                # 3. Match gerund 'doing something'
                if not matched_ex and "doing something" in clean_pat.lower():
                    prefix = clean_pat.lower().replace("doing something", "").strip()
                    for ex in examples:
                        if prefix in ex.lower() and re.search(r"\b\w+ing\b", ex.lower()):
                            matched_ex = ex
                            break
                # 4. Match infinitive 'to do something'
                if not matched_ex and "to do something" in clean_pat.lower():
                    prefix = clean_pat.lower().replace("to do something", "").strip()
                    for ex in examples:
                        if prefix in ex.lower() and re.search(r"\bto\s+\w+\b", ex.lower()):
                            matched_ex = ex
                            break

            # Synchronize bound_prep with the preposition actually attested in matched_ex if slash alternatives exist
            if matched_ex and len(cand_preps) > 1:
                ex_l = matched_ex.lower()
                for cp in cand_preps:
                    if re.search(rf"\b{re.escape(cp)}\b", ex_l):
                        bound_prep = cp
                        break

            bundles.append({
                "pattern": clean_pat,
                "example": matched_ex,
                "prep": bound_prep,
                "preps": cand_preps,
                "inf_valency": "to do something" in clean_pat.lower(),
                "gerund_valency": "doing something" in clean_pat.lower(),
            })
        return bundles + verb_colloc_bundles

    @classmethod
    def pick_atomic_pattern_bundle(
        cls,
        sense: Optional[Dict[str, Any]],
        target_word: str,
        quote: str = "",
        preferred_anchor: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Selects the single best atomic (pattern, example, prep) bundle from a locked sense.
        Guarantees that the pattern and its illustrative example are 100% harmonized from
        the exact same dictionary sense.
        """
        if not sense:
            return None
        bundles = cls.extract_sense_pattern_bundles(sense, target_word, quote=quote)
        if not bundles:
            return None

        tw_l = target_word.lower()
        q_l = (quote or "").lower()
        is_verb_sense = "verb" in str(sense.get("pos") or "").lower()
        is_adj_sense = "adj" in str(sense.get("pos") or "").lower()
        quote_has_dobj_v = False
        quote_verb_preps: Set[str] = set()
        quote_adj_is_attributive = "only before noun" in str(sense.get("grammar") or "").lower()
        if is_verb_sense and quote and " " not in tw_l:
            try:
                nlp = cls.get_spacy()
                if nlp:
                    doc_q_pat = nlp(quote)
                    t_inf_set = {tw_l} | set(cls.inflected_forms(tw_l))
                    for tok_q in doc_q_pat:
                        if tok_q.text.lower() in t_inf_set or tok_q.lemma_.lower() == tw_l:
                            for ch in tok_q.children:
                                if ch.dep_ in ("dobj", "obj", "nsubjpass", "auxpass"):
                                    quote_has_dobj_v = True
                                elif ch.dep_ == "prep" and ch.i > tok_q.i:
                                    quote_verb_preps.add(ch.text.lower())
                            if not quote_has_dobj_v and tok_q.tag_ == "VBN" and tok_q.dep_ in ("acl", "amod"):
                                quote_has_dobj_v = True
                            if quote_has_dobj_v or quote_verb_preps:
                                break
            except Exception:
                pass
        elif is_adj_sense and quote and not quote_adj_is_attributive:
            try:
                nlp = cls.get_spacy()
                if nlp:
                    doc_q_pat = nlp(quote)
                    _, _, t_q_tok = cls._find_target_span_in_doc(doc_q_pat, tw_l)
                    if t_q_tok is not None:
                        has_q_prep_a = any(ch.dep_ == "prep" and ch.i > t_q_tok.i for ch in t_q_tok.children)
                        if t_q_tok.dep_ == "amod" and t_q_tok.head.pos_ in ("NOUN", "PROPN") and not has_q_prep_a:
                            quote_adj_is_attributive = True
            except Exception:
                pass

        scored: List[Tuple[int, Dict[str, Any]]] = []
        for b in bundles:
            pat = b["pattern"]
            ex = b["example"]
            prep = b.get("prep")
            cand_preps = b.get("preps") or ([prep] if prep else [])
            b_src = str(b.get("source") or "")
            score = 0

            # Absolute filter / priority: Target word presence
            # Genuine valency patterns contain the target headword or its inflections.
            # Idioms like 'got on his nerves' under 'annoy' do NOT contain 'annoy' and must not hijack the item!
            has_tw = bool(re.search(rf"\b{re.escape(tw_l)}\w*\b", pat.lower()))
            if " " in tw_l and not has_tw:
                tw_parts = [
                    p for p in re.findall(r"[a-z]+", tw_l)
                    if p not in ("s", "sb", "sth", "somebody", "something", "one", "ones", "oneself", "your", "his", "her", "their", "my", "our", "its")
                ]
                if not all(cls.text_contains_form(pat, p) for p in tw_parts):
                    continue
                if len(tw_parts) >= 2:
                    p_toks = re.findall(r"[a-z]+", pat.lower())
                    f_set = {tw_parts[0]} | set(cls.inflected_forms(tw_parts[0]))
                    l_set = {tw_parts[-1]} | set(cls.inflected_forms(tw_parts[-1]))
                    _valid_mw_pat = False
                    _extra_particles = {
                        "out", "in", "up", "down", "off", "on", "away", "back", "over", "into",
                        "from", "with", "for", "to", "of", "at", "by", "about", "against", "through", "around", "aside", "apart"
                    } - set(tw_parts)
                    for i_f, tk_f in enumerate(p_toks):
                        if tk_f in f_set:
                            for i_l in range(i_f + 1, min(len(p_toks), i_f + len(tw_parts) + 3)):
                                if p_toks[i_l] in l_set:
                                    mid_w = set(p_toks[i_f + 1 : i_l]) - set(tw_parts)
                                    if not (mid_w & _extra_particles):
                                        _valid_mw_pat = True
                                        break
                        if _valid_mw_pat:
                            break
                    if not _valid_mw_pat:
                        continue
            if has_tw:
                score += 50
            else:
                score -= 30

            # Source tier bonus/guard: `gramexa` is the explicit syntactic valency frame;
            # `colloinexa` coordinate binomials (`weird and wonderful`) without quote attestation or example must not hijack general items.
            if b_src in ("gramexa", "sense_unit"):
                score += 25
            elif b_src == "colloinexa":
                m_coord = re.match(rf"^(?:{re.escape(tw_l)}\w*\s+and\s+([a-z]+)|([a-z]+)\s+and\s+{re.escape(tw_l)}\w*)$", pat.strip().lower())
                if m_coord:
                    partner_w = m_coord.group(1) or m_coord.group(2)
                    if partner_w and not re.search(rf"\b{re.escape(partner_w)}\b", q_l) and not ex:
                        score -= 90

            # Check if the passage quote directly attests `target_word + prep` (e.g. 'masquerading as')
            quote_adj_preps = [
                cp for cp in cand_preps
                if cp and re.search(rf"\b{re.escape(tw_l)}\w*\s+{re.escape(cp.lower())}\b", q_l)
            ]

            # Instance-Level Transitivity Alignment Guard:
            # If the quote uses a verb transitively with a direct object (`riding a car`, `impacts some human needs`)
            # and does NOT attest `target_word + prep`, penalize bare intransitive `V + prep` bundles (`ride in/on`, `impact on/upon`)
            # that have no object slot between the verb and preposition.
            if quote_has_dobj_v and cand_preps and not quote_adj_preps:
                pat_no_opt = re.sub(r"\[.*?\]|\(.*?\)", "", pat).strip().lower()
                _is_bare_intrans_prep_pat = any(
                    bool(re.search(rf"\b{re.escape(tw_l)}\w*\s+(?:[a-z]+/)*{re.escape(cp.lower())}\b", pat_no_opt))
                    for cp in cand_preps if cp
                )
                _attested_in_quote_v_preps = any(cp and cp.lower() in quote_verb_preps for cp in cand_preps)
                if (
                    _is_bare_intrans_prep_pat
                    and not _attested_in_quote_v_preps
                    and not (preferred_anchor and any(cp.lower() == preferred_anchor.lower() for cp in cand_preps))
                ):
                    score -= 55
                elif (
                    any(tok_o in pat_no_opt for tok_o in ("something", "somebody", "someone", "sb", "sth", "importance", "significance", "attention"))
                    and (not cand_preps or any(re.search(rf"\b{re.escape(cp)}\b", q_l) for cp in cand_preps if cp))
                ):
                    score += 30
            elif is_adj_sense and quote_adj_is_attributive and not quote_adj_preps:
                # If the passage quote uses the adjective attributively (`a diplomatic move`, `identical rooms`, `invaluable experience`, `simple methods`, `a beloved national hero`)
                # without a governed preposition, skip predicative preposition/infinitive patterns (`identical in`, `invaluable in`, `simple to use`, `beloved of`)
                # unless `preferred_anchor` explicitly requests that preposition, and boost `Adj + Noun` collocation patterns (`earnest voice/request`).
                if cand_preps or b.get("inf_valency") or b.get("gerund_valency"):
                    if not (preferred_anchor and any(cp.lower() == preferred_anchor.lower() for cp in cand_preps)):
                        continue
                else:
                    score += 35

            # Massive bonus if bundle contains an authentic illustrative example (especially a complete finite sentence)
            if ex:
                if cls._is_complete_finite_sentence(ex):
                    score += 55
                else:
                    score -= 20
            elif quote_adj_preps:
                # Directly attested immediately after target word in passage quote: do NOT penalize
                score += 20
            elif cand_preps:
                # If pattern claims a bound preposition (e.g. 'identical in') but has NO matching example in the sense,
                # penalize heavily so it doesn't pair with a non-prepositional fallback sentence ('four identical houses')!
                score -= 35

            # Infinitive / Gerund shell-noun or verb valency alignment with quote / preferred_anchor
            b_prep_dir = b.get("prep_dir", "post")
            if b.get("inf_valency"):
                if (
                    (preferred_anchor and preferred_anchor.lower() == "to")
                    or re.search(rf"\b{re.escape(tw_l)}\w*\b(?:\s+[a-z]+){{0,3}}\s+to\s+[a-z]{{2,}}\b", q_l)
                ):
                    score += 65
            elif b.get("gerund_valency"):
                lead_p = prep or "of"
                if re.search(rf"\b{re.escape(tw_l)}\w*\b(?:\s+[a-z]+){{0,2}}\s+{re.escape(lead_p)}\s+[a-z]{{2,}}ing\b", q_l):
                    score += 65

            # Quote / preferred_anchor alignment: if curriculum quote or anchor uses one of the bundle's prepositions
            if preferred_anchor and any(cp.lower() == preferred_anchor.lower() for cp in cand_preps):
                if not ex or cls._example_attests_prep_valency(ex, tw_l, preferred_anchor.lower(), prep_dir=b_prep_dir):
                    score += 40
                    b = dict(b, prep=preferred_anchor.lower())
                    prep = b["prep"]
                else:
                    score -= 25
            elif quote_adj_preps:
                for cp in quote_adj_preps:
                    if not ex or cls._example_attests_prep_valency(ex, tw_l, cp.lower(), prep_dir=b_prep_dir):
                        score += 55
                        b = dict(b, prep=cp.lower())
                        prep = b["prep"]
                        break
            elif any(re.search(rf"\b{re.escape(cp)}\b", q_l) for cp in cand_preps):
                for cp in cand_preps:
                    if re.search(rf"\b{re.escape(cp)}\b", q_l) and (
                        not ex or cls._example_attests_prep_valency(ex, tw_l, cp.lower(), prep_dir=b_prep_dir)
                    ):
                        score += 35
                        b = dict(b, prep=cp)
                        prep = cp
                        break
            # General prepositional valency bonus
            if prep:
                score += 10
            # Sense-verified `Verb + a/an + <noun>` collocation bonus (beats bare `noun + of` frames like `loan of`)
            if b.get("verb_anchor"):
                if b.get("from_quote_verb") or (preferred_anchor and b["verb_anchor"].lower() == preferred_anchor.lower()):
                    score += 55
                elif b.get("from_sense_example"):
                    score += 40
                else:
                    score += 25
                if ex and re.search(rf"\b(?:a|an)\b[^.?!;]*\b{re.escape(tw_l)}\b", ex.lower()):
                    score += 10
            # Syntactic valency frames (gerunds, infinitives, complement clauses)
            if any(v in pat.lower() for v in ("doing something", "to do something", "somebody when", "somebody that", "somebody how")):
                score += 20
            # Clean length preference (compact, high-utility frames)
            p_len = len(re.sub(r"\[.*?\]|\(.*?\)", "", pat).strip())
            if 5 <= p_len <= 35:
                score += 5
            scored.append((score, b))

        scored.sort(key=lambda x: (x[0], -len(x[1]["pattern"])), reverse=True)
        if not scored or scored[0][0] < 0:
            return None
        best_b = scored[0][1]
        # Do not return a preposition-claiming pattern that lacks any example AND is not attested in the quote
        if best_b.get("prep") and not best_b.get("example"):
            if not any(re.search(rf"\b{re.escape(cp)}\b", q_l) for cp in (best_b.get("preps") or [best_b["prep"]])):
                return None
        return best_b

    @classmethod
    def select_optimal_five_source_example(
        cls,
        final_target: str,
        base_word: str,
        canonical_pos: str,
        target_entry: Optional[Dict[str, Any]],
        locked_sense: Optional[Dict[str, Any]],
        t_s_idx: Optional[int],
        definition: Optional[str] = None,
        quote: Optional[str] = None,
        anchor: Optional[str] = None,
        anchor_type: Optional[str] = None,
        best_pattern: Optional[str] = None,
        current_example: Optional[str] = None,
        is_attributive_only_adj: bool = False,
        is_copula_adj: bool = False,
        is_plural_noun: bool = False,
        has_uncountable_contrast_foil: bool = False,
    ) -> Tuple[Optional[str], Optional[str], str]:
        """
        Five-Source Optimal Example Pre-Selector (GEMINI.md §3.2).
        Evaluates sense-isolated candidate examples across pure 5 LDOCE compartments in `pedlex.db`:
          1. Sense-locked `grammar_boxes`
          2. Locked `Sense.pattern_bundles` (`gramexa`) & `Sense.examples`
          3. Sense-matched `LDOCE Thesaurus`
          4. Sense-matched `Language Activator`
          5. Sense-isolated Curated `Collocations` & `Cross-Collocations`
        (Note: Oxford Thesaurus and raw BNC corpus dumps are permanently excluded for hygiene).
        Returns `(winning_example, synchronized_pattern, winning_source_tier)`.
        """
        ft_low = (final_target or "").strip().lower()
        bw_low = (base_word or ft_low).strip().lower()
        q_norm = re.sub(r"\s+", " ", (quote or "").strip()).lower()
        senses_all = (target_entry or {}).get("senses") or []
        is_poly = len(senses_all) > 1 and locked_sense is not None

        # Precompute sense-isolation keyword sets (`locked_def_kws` vs `other_sense_kws`)
        s_kws: Set[str] = set()
        o_kws: Set[str] = set()
        if locked_sense is not None:
            l_blob = " ".join([
                str(locked_sense.get("signpost") or ""),
                str(definition or locked_sense.get("definition") or ""),
                "" if is_poly else " ".join(str(x) for x in (locked_sense.get("examples") or [])),
                " ".join(str(x) for x in (locked_sense.get("synonyms") or [])),
                " ".join(str(x) for x in (locked_sense.get("patterns") or [])),
            ]).lower()
            _def_meta_stopwords = {
                "member", "members", "group", "groups", "organization", "organizations",
                "system", "systems", "part", "parts", "number", "amount", "kind", "type", "form"
            }
            s_kws = {
                w for w in re.findall(r"[a-z]{4,}", l_blob)
                if w not in (ft_low, bw_low) and w not in cls._ANCHOR_STOPWORDS and w not in _def_meta_stopwords
            }
            if is_poly:
                for idx_o, s_o in enumerate(senses_all):
                    if s_o is locked_sense or (t_s_idx is not None and idx_o == t_s_idx):
                        continue
                    o_blob = " ".join([
                        str(s_o.get("signpost") or ""),
                        str(s_o.get("definition") or ""),
                        " ".join(str(x) for x in (s_o.get("examples") or [])[:4]),
                        " ".join(str(x) for x in (s_o.get("synonyms") or [])),
                    ]).lower()
                    o_kws |= {
                        w for w in re.findall(r"[a-z]{4,}", o_blob)
                        if w not in (ft_low, bw_low) and w not in cls._ANCHOR_STOPWORDS
                    }
                disc_s_kws = s_kws - o_kws
                if disc_s_kws:
                    s_kws = disc_s_kws
                o_kws -= s_kws

        def _passes_sense_isolation(probe_text: str, explicit_indices: Optional[List[int]] = None, strict_positive: bool = False) -> bool:
            if not is_poly:
                return True
            if isinstance(explicit_indices, list) and explicit_indices and t_s_idx is not None:
                return t_s_idx in explicit_indices
            words = {
                w for w in re.findall(r"[a-z]{4,}", (probe_text or "").lower())
                if w not in (ft_low, bw_low) and w not in cls._ANCHOR_STOPWORDS
            }
            pos_cnt = len(words & s_kws)
            neg_cnt = len(words & o_kws)
            if neg_cnt > pos_cnt:
                return False
            if strict_positive and s_kws and pos_cnt == 0:
                return False
            return True

        # Gather `(example_text, pattern_hint, source_tier)` candidates
        raw_candidates: List[Tuple[str, Optional[str], str]] = []

        # Source 1: Sense-locked `grammar_boxes`
        if target_entry:
            for gb in (target_entry.get("grammar_boxes") or []):
                if not isinstance(gb, dict):
                    continue
                gb_pos = str(gb.get("homograph_pos") or "").strip().lower()
                if gb_pos and canonical_pos in ("noun", "verb", "adj", "adv") and cls._pos_code(gb_pos) != cls._pos_code(canonical_pos):
                    continue
                if not _passes_sense_isolation(f"{gb.get('heading') or ''} {' '.join(str(r) for r in (gb.get('notes') or gb.get('rules') or []))}", gb.get("sense_indices")):
                    continue
                bad_set = {str(bx).strip().lower() for bx in (gb.get("bad_examples") or []) if bx}
                for g_ex in (gb.get("examples") or []):
                    g_ex_s = str(g_ex or "").strip()
                    if g_ex_s and g_ex_s.lower() not in bad_set and "✗" not in g_ex_s and "don't say" not in g_ex_s.lower():
                        raw_candidates.append((g_ex_s, best_pattern, "grammar_boxes"))

        # Source 2: Locked `Sense.pattern_bundles` & `Sense.examples`
        if locked_sense:
            for pb in cls.extract_sense_pattern_bundles(locked_sense, final_target, quote=quote or ""):
                if pb.get("example"):
                    src_tag = f"pattern_bundle_{pb.get('source') or 'native'}"
                    raw_candidates.append((str(pb["example"]), pb.get("pattern"), src_tag))
            for s_ex in (locked_sense.get("examples") or []):
                s_ex_txt = s_ex.get("text") if isinstance(s_ex, dict) else str(s_ex or "")
                if s_ex_txt:
                    raw_candidates.append((s_ex_txt, best_pattern, "sense_example"))

        # Source 3: Sense-matched `LDOCE Thesaurus`
        if target_entry:
            for th in (target_entry.get("thesaurus") or []):
                if not isinstance(th, dict):
                    continue
                if not (cls._thesaurus_variant_matches(th.get("word"), final_target) or cls._thesaurus_variant_matches(th.get("word"), bw_low)):
                    continue
                th_probe = f"{th.get('section') or ''} {th.get('concept') or ''} {th.get('distinction') or ''}"
                if not _passes_sense_isolation(th_probe, strict_positive=True):
                    continue
                for th_ex in (th.get("examples") or []):
                    if th_ex:
                        raw_candidates.append((str(th_ex), None, "ldoce_thesaurus"))

        # Source 4: Sense-matched `Language Activator`
        if target_entry:
            def_is_neg = bool(re.search(r"\b(?:not|no|never|without|lack|fail|stop|prevent|refuse)\b", (definition or "").lower()))
            for la in (target_entry.get("language_activator") or []):
                if not isinstance(la, dict):
                    continue
                concept_str = str(la.get("concept") or "").strip().lower()
                if (concept_str.startswith("not ") or " not " in concept_str) and not def_is_neg:
                    continue
                for exp in (la.get("exponents") or la.get("words") or []):
                    if not isinstance(exp, dict):
                        continue
                    if not (cls._thesaurus_variant_matches(exp.get("word"), final_target) or cls._thesaurus_variant_matches(exp.get("word"), bw_low)):
                        continue
                    la_probe = f"{concept_str} {la.get('section_gloss') or ''} {exp.get('definition') or ''}"
                    if not _passes_sense_isolation(la_probe, strict_positive=True):
                        continue
                    for la_ex in (exp.get("examples") or []):
                        if la_ex:
                            raw_candidates.append((str(la_ex), None, "language_activator"))

        # Source 5: Curated `Collocations`, `Cross-Collocations` & `current_example`
        # Note: Oxford Thesaurus (`oxford_thesaurus`) is intentionally excluded because it contains unedited
        # BNC corpus dumps and lacks sense-level valency structure. Within `target_entry["collocations"]`,
        # curated LDOCE Collocation Box items (`source in ("entry", "sense", None)`) are strictly separated
        # from raw corpus dumps (`source == "corpus"`).
        if current_example and (" " in ft_low or not is_poly or _passes_sense_isolation(current_example)):
            raw_candidates.append((current_example, best_pattern, "sense_example" if " " in ft_low else "collocation_anchor"))
        corpus_candidates: List[Tuple[str, Optional[str], str]] = []
        _sense_ex_pat_blob = " ".join([
            " ".join(str(x) for x in ((locked_sense or {}).get("examples") or [])),
            " ".join(str(x) for x in ((locked_sense or {}).get("patterns") or [])),
        ]).lower()
        _sense_ex_pat_kws = {
            w for w in re.findall(r"[a-z]{4,}", _sense_ex_pat_blob)
            if w not in (ft_low, bw_low) and w not in cls._ANCHOR_STOPWORDS
        }
        if target_entry:
            for c_list in ((target_entry.get("collocations") or {}).values()):
                for c_it in (c_list or []):
                    if not isinstance(c_it, dict) or not c_it.get("example"):
                        continue
                    c_src = str(c_it.get("source") or "entry").strip().lower()
                    c_colloc_str = str(c_it.get("collocation") or "").strip()
                    c_colloc_kws = {
                        w for w in re.findall(r"[a-z]{4,}", c_colloc_str.lower())
                        if w not in (ft_low, bw_low) and w not in cls._ANCHOR_STOPWORDS
                    }
                    _colloc_attested_in_sense = bool(
                        c_src != "corpus"
                        and c_colloc_kws
                        and (c_colloc_kws & _sense_ex_pat_kws)
                        and not (c_colloc_kws & o_kws)
                    )
                    c_probe = f"{c_colloc_str} {c_it.get('gloss') or ''} {c_it.get('example') or ''}"
                    if _colloc_attested_in_sense or _passes_sense_isolation(c_probe, c_it.get("sense_indices"), strict_positive=True):
                        if is_poly and s_kws and not _colloc_attested_in_sense:
                            c_w = {w for w in re.findall(r"[a-z]{4,}", c_probe.lower()) if w not in (ft_low, bw_low) and w not in cls._ANCHOR_STOPWORDS}
                            if not (c_w & s_kws) and not c_it.get("sense_indices"):
                                continue
                        c_pat_hint = (
                            c_colloc_str
                            if (
                                c_colloc_str
                                and " " in c_colloc_str
                                and c_src != "corpus"
                                and (cls.text_contains_form(c_colloc_str, bw_low) or cls.text_contains_form(c_colloc_str, ft_low))
                            )
                            else None
                        )
                        if c_src == "corpus":
                            corpus_candidates.append((str(c_it["example"]), None, "collocations_corpus"))
                        else:
                            raw_candidates.append((str(c_it["example"]), c_pat_hint, "collocations_entry"))
            for cc_it in (target_entry.get("cross_collocations") or []):
                if not isinstance(cc_it, dict):
                    continue
                cc_exs = ([cc_it["example"]] if cc_it.get("example") else []) + (cc_it.get("examples") or [])
                for cc_ex in cc_exs:
                    if not cc_ex:
                        continue
                    cc_probe = f"{cc_it.get('collocation') or ''} {cc_it.get('gloss') or ''} {cc_ex}"
                    if _passes_sense_isolation(cc_probe, cc_it.get("sense_indices"), strict_positive=True):
                        cc_colloc_str = str(cc_it.get("collocation") or "").strip()
                        cc_pat_hint = (
                            cc_colloc_str
                            if (
                                cc_colloc_str
                                and " " in cc_colloc_str
                                and (cls.text_contains_form(cc_colloc_str, bw_low) or cls.text_contains_form(cc_colloc_str, ft_low))
                            )
                            else None
                        )
                        raw_candidates.append((str(cc_ex), cc_pat_hint, "cross_collocations"))

        # Only fall back to unedited BNC corpus collocation dumps if no curated dictionary compartment
        # provides a complete finite sentence containing the target word/noun form.
        if corpus_candidates and not any(
            cls._is_complete_finite_sentence(r_ex)
            and (
                re.search(rf"\b{re.escape(ft_low)}\b", str(r_ex or "").lower())
                if canonical_pos == "adv"
                else (cls.text_contains_form(r_ex, ft_low) or cls.text_contains_form(r_ex, bw_low))
            )
            for r_ex, _, _ in raw_candidates
        ):
            raw_candidates.extend(corpus_candidates)

        # For derived or sparse entries (especially `-ly` adverbs whose `target_entry` resolved to the base adjective
        # or entries like `thereafter` whose primary sense only stores a short fragment), harvest authentic examples
        # from Oxford Thesaurus, WordNet (`oewn:2025+`), and PedLex cross-entry corpus examples (`search_corpus_examples`)
        # whenever no curated compartment provides a complete finite sentence containing `ft_low`.
        if " " not in ft_low and not any(
            len(re.findall(r"[A-Za-z]+", str(r_ex or ""))) >= 5
            and str(r_ex or "").strip()[-1:] in ".?!"
            and (
                re.search(rf"\b{re.escape(ft_low)}\b", str(r_ex or "").lower())
                if canonical_pos == "adv"
                else (cls.text_contains_form(r_ex, ft_low) or cls.text_contains_form(r_ex, bw_low))
            )
            and re.sub(r"\s+", " ", str(r_ex or "").strip()).lower() != q_norm
            for r_ex, _, _ in raw_candidates
        ):
            try:
                for ox_ent in (cls.get_oxford_thesaurus_entries(ft_low) or []):
                    for ox_s in (ox_ent.get("senses") or []):
                        ox_ex = str(ox_s.get("example") or "").strip()
                        if ox_ex and re.search(rf"\b{re.escape(ft_low)}\b", ox_ex.lower()):
                            raw_candidates.append((ox_ex, None, "ldoce_thesaurus"))
            except Exception:
                pass
            try:
                _wn_pos_code = {"noun": "n", "verb": "v", "adj": "a", "adv": "r"}.get(canonical_pos, canonical_pos)
                for wn_syn in (cls.get_wordnet_synsets(ft_low, pos=_wn_pos_code) or []):
                    _wn_exs = wn_syn.examples() if hasattr(wn_syn, "examples") and callable(wn_syn.examples) else (wn_syn.get("examples") if isinstance(wn_syn, dict) else [])
                    for wn_ex in (_wn_exs or []):
                        wn_ex_s = str(wn_ex or "").strip()
                        if wn_ex_s and re.search(rf"\b{re.escape(ft_low)}\b", wn_ex_s.lower()):
                            raw_candidates.append((wn_ex_s, None, "sense_example"))
            except Exception:
                pass
            if not is_poly:
                try:
                    for ch_ex in (cls.search_corpus_examples(ft_low, limit=8) or []):
                        ch_s = str(ch_ex or "").strip()
                        if ch_s and re.search(rf"\b{re.escape(ft_low)}\b", ch_s.lower()):
                            raw_candidates.append((ch_s, None, "sense_example"))
                except Exception:
                    pass

        if not raw_candidates:
            return current_example, best_pattern, "none"

        nlp = cls.get_spacy()
        if canonical_pos == "adv" and " " not in ft_low:
            t_forms = {ft_low}
        else:
            t_forms = {ft_low, bw_low} | set(cls.inflected_forms(bw_low)) | set(cls.inflected_forms(ft_low))
        if canonical_pos == "noun" and " " not in ft_low:
            t_forms = {
                f for f in t_forms
                if not (
                    f.endswith(("ed", "ing"))
                    and not bw_low.endswith(("ed", "ing"))
                    and not ft_low.endswith(("ed", "ing"))
                )
            }
        scored_exs: List[Tuple[int, str, Optional[str], str]] = []
        seen_ex_norm: Set[str] = set()

        tier_bonuses = {
            "pattern_bundle_gramexa": 32,
            "grammar_boxes": 28,
            "sense_example": 28,
            "collocations_entry": 26,
            "pattern_bundle_sense_verb_colloc": 26,
            "ldoce_thesaurus": 24,
            "language_activator": 22,
            "cross_collocations": 22,
            "collocation_anchor": 12,
            "pattern_bundle_colloinexa": 10,
            "collocations_corpus": -25,
        }

        for raw_ex, pat_hint, tier in raw_candidates:
            ex_clean = re.sub(r"\s*\(=[^)]*\)", "", str(raw_ex or "")).strip().strip('"').strip()
            if not ex_clean:
                continue
            # Filter out unedited BNC spoken transcript headers ('Male speaker ...', 'Female speaker ...')
            # and comma-heavy list enumerations ('piles of newspapers, heaps of books, manuscripts, labels, ...')
            if re.match(r"^(?:male|female|unidentified)\s+speaker\b", ex_clean, re.IGNORECASE):
                continue
            if ex_clean.count(",") >= 4:
                continue
            ex_norm = re.sub(r"\s+", " ", ex_clean).lower()
            if ex_norm == q_norm or ex_norm in seen_ex_norm:
                continue
            # Must contain target word/expression
            _is_passive_3w_ex = False
            _passive_cop_matched = False
            if " " in ft_low:
                ft_parts = [
                    p for p in re.findall(r"[a-z]+", ft_low)
                    if p not in ("s", "sb", "sth", "somebody", "something", "one", "ones", "oneself", "your", "his", "her", "their", "my", "our", "its")
                ]
                if (
                    len(ft_parts) == 3
                    and ft_parts[0] in ("be", "get", "feel", "am", "is", "are", "was", "were", "been", "being")
                    and ft_parts[2] in ("with", "in", "to", "of", "on", "for", "from", "into", "at", "by", "about", "as", "against", "under", "over")
                ):
                    _is_passive_3w_ex = True
                    _passive_cop_matched = cls.text_contains_form(ex_clean, ft_parts[0])
                    if not _passive_cop_matched:
                        _v_lem_p = cls._surface_lemma(ft_parts[1]) or cls.lemma_of(ft_parts[1]) or ft_parts[1]
                        ft_parts = [_v_lem_p, ft_parts[2]]
                if ft_parts:
                    if not all(cls.text_contains_form(ex_clean, p) for p in ft_parts):
                        continue
                    if len(ft_parts) >= 2:
                        ex_toks = re.findall(r"[a-z]+", ex_norm)
                        f_set = {ft_parts[0]} | set(cls.inflected_forms(ft_parts[0]))
                        l_set = {ft_parts[-1]} | set(cls.inflected_forms(ft_parts[-1]))
                        _extra_particles = {
                            "out", "in", "up", "down", "off", "on", "away", "back", "over", "into",
                            "from", "with", "for", "to", "of", "at", "by", "about", "against", "through", "around", "aside", "apart"
                        } - set(ft_parts)
                        _valid_mw_ex = False
                        for i_f, tk_f in enumerate(ex_toks):
                            if tk_f in f_set:
                                for i_l in range(i_f + 1, min(len(ex_toks), i_f + len(ft_parts) + 4)):
                                    if ex_toks[i_l] in l_set:
                                        mid_w = set(ex_toks[i_f + 1 : i_l]) - set(ft_parts)
                                        if not (mid_w & _extra_particles):
                                            _valid_mw_ex = True
                                            break
                            if _valid_mw_ex:
                                break
                        if not _valid_mw_ex:
                            continue
            else:
                if canonical_pos == "noun":
                    if not any(re.search(rf"\b{re.escape(nf)}\b", ex_norm) for nf in t_forms):
                        continue
                elif canonical_pos == "adv":
                    if not re.search(rf"\b{re.escape(ft_low)}\b", ex_norm):
                        continue
                elif not (cls.text_contains_form(ex_clean, ft_low) or cls.text_contains_form(ex_clean, bw_low)):
                    continue
                if canonical_pos == "verb" and ft_low == bw_low and not current_example and quote:
                    _has_non_noun_exact = any(
                        re.search(rf"\b{re.escape(ft_low)}\b", str(r_ex or "").lower())
                        and not re.search(rf"\b(?:a|an|the|his|her|their|its|my|our|your|this|that)\s+{re.escape(ft_low)}\b", str(r_ex or "").lower())
                        and re.sub(r"\s+", " ", str(r_ex or "").strip()).lower() != q_norm
                        for r_ex, _, _ in raw_candidates
                    )
                    if _has_non_noun_exact and not re.search(rf"\b{re.escape(ft_low)}\b", ex_norm):
                        continue

            if anchor and anchor_type == "partitive":
                anc_req = str(anchor).strip().lower()
                if anc_req and not re.search(rf"\b{re.escape(anc_req)}\w*\b", ex_norm):
                    continue

            seen_ex_norm.add(ex_norm)
            sc = tier_bonuses.get(tier, 15)

            # Exact inflected form preference (especially for plural nouns or participial adjectives/verbs)
            if re.search(rf"\b{re.escape(ft_low)}\b", ex_norm):
                sc += 35 if (is_plural_noun or ft_low != bw_low) else 28
            elif is_plural_noun or canonical_pos == "verb":
                sc -= 25
                if canonical_pos == "verb" and ft_low == bw_low and re.search(rf"\b{re.escape(ft_low)}\b", q_norm):
                    if not re.search(rf"\b{re.escape(ft_low)}s?\b", ex_norm):
                        sc -= 65

            # Parse with spaCy to score clause completeness & syntactic role around target word
            try:
                doc = nlp(ex_clean) if nlp else None
            except Exception:
                doc = None

            if doc is not None:
                alpha_toks = [t for t in doc if t.is_alpha]
                n_words = len(alpha_toks)
                has_end_punct = ex_clean[-1:] in ".?!"
                has_finite_verb = any(
                    (
                        t.pos_ in ("VERB", "AUX")
                        and (
                            t.tag_ in ("VBD", "VBP", "VBZ", "MD")
                            or (
                                t.dep_ in ("ROOT", "relcl", "ccomp", "advcl", "conj")
                                and (
                                    t.tag_ == "VB"
                                    or any(c.dep_ in ("nsubj", "nsubjpass", "aux", "auxpass", "expl") for c in t.children)
                                )
                            )
                        )
                    ) or (
                        t.dep_ == "ROOT" and any(c.dep_ in ("nsubj", "nsubjpass") for c in t.children)
                    ) or (
                        t.dep_ == "amod"
                        and t.head.dep_ == "ROOT"
                        and t.head.pos_ in ("NOUN", "PROPN")
                        and (t.i > t.head.i)
                        and any(c.dep_ in ("prep", "advmod", "npadvmod") for c in t.children)
                    )
                    for t in doc
                )
                if has_end_punct and has_finite_verb and n_words >= 3:
                    sc += 75
                elif not has_end_punct or not has_finite_verb or n_words < 3:
                    sc -= 140  # Strict penalty against verbless/unpunctuated NP/PP fragments ('earthy language', 'a team of experts')
                if 6 <= n_words <= 22:
                    sc += 15

                # Find target token (or hyphenated compound span) in `doc`
                t_start_i, t_end_i, t_tok = (None, None, None)
                if " " not in ft_low:
                    t_start_i, t_end_i, t_tok = cls._find_target_span_in_doc(
                        doc, ft_low, base_str=bw_low, t_forms=t_forms
                    )
                if t_tok is not None:
                    if canonical_pos == "adj":
                        eff_head = t_tok.head
                        while (
                            eff_head is not None
                            and eff_head.dep_ == "conj"
                            and eff_head.head.pos_ == "ADJ"
                            and eff_head.head != eff_head
                        ):
                            eff_head = eff_head.head
                        if eff_head is not None and eff_head.pos_ == "ADJ" and eff_head.head != eff_head:
                            eff_head = eff_head.head
                        if anchor_type == "verb_copula" and anchor:
                            if (
                                eff_head is not None
                                and eff_head.pos_ in ("VERB", "AUX")
                                and eff_head.lemma_.lower() == str(anchor).lower()
                            ):
                                sc += 55
                            elif (
                                t_tok.dep_ in ("acomp", "oprd", "ccomp", "xcomp", "conj")
                                and eff_head is not None
                                and eff_head.pos_ in ("VERB", "AUX")
                            ):
                                sc += 25
                            elif t_tok.dep_ == "amod":
                                sc -= 30
                        elif is_copula_adj or anchor_type == "prep":
                            if t_tok.dep_ in ("acomp", "ROOT", "attr", "oprd", "conj") or any(c.dep_ == "prep" for c in t_tok.children):
                                sc += 45
                                if anchor_type == "prep" and anchor and any(
                                    c.dep_ == "prep" and c.i > t_tok.i and c.text.lower() == str(anchor).lower()
                                    for c in t_tok.children
                                ):
                                    sc += 35
                            elif t_tok.dep_ == "amod":
                                sc -= 20
                        else:
                            # Canonical pre-nominal `Adj + Noun` (`... their ____ achievement`, `... an ____ opportunity`, `... the ____ task`)
                            t_left_idx = t_start_i if t_start_i is not None else t_tok.i
                            if (
                                t_tok.dep_ in ("amod", "conj")
                                and eff_head is not None
                                and eff_head.pos_ in ("NOUN", "PROPN")
                                and t_left_idx < eff_head.i
                            ):
                                sc += 55
                                if any(c.dep_ in ("det", "poss") and c.i < t_left_idx for c in eff_head.children):
                                    sc += 15
                            elif eff_head is not None and eff_head.pos_ in ("NOUN", "PROPN") and t_left_idx > eff_head.i:
                                # Postpositive adjective phrase ('an achievement unparalleled in sporting history')
                                sc -= 35
                            elif t_tok.dep_ in ("acomp", "ROOT", "oprd") and not any(c.dep_ == "prep" for c in t_tok.children):
                                # Bare predicative ('The work was tiring and tedious') is weaker than attributive 'Adj + Noun'
                                sc -= 35 if (anchor_type == "modified_noun" or is_attributive_only_adj) else 15
                    elif canonical_pos == "noun":
                        if (
                            t_tok.pos_ in ("VERB", "AUX", "ADJ", "ADV")
                            or t_tok.tag_ in ("VBD", "VBG", "VBN", "VBP", "VBZ", "JJ", "JJR", "JJS")
                            or t_tok.dep_ == "amod"
                        ):
                            sc -= 140
                        elif " " not in ft_low and t_tok.dep_ == "compound" and t_tok.head.pos_ in ("NOUN", "PROPN") and t_tok.i < t_tok.head.i:
                            sc -= 65
                        else:
                            if any(c.dep_ == "conj" for c in t_tok.children):
                                sc -= 25
                            _SUPPORT_VERBS_NOUN = {
                                "make", "take", "give", "pay", "hold", "lose", "gain", "keep",
                                "put", "set", "bear", "lay", "draw", "reach", "strike", "pose", "stand",
                            }
                            if t_tok.dep_ in ("dobj", "obj") and t_tok.head.pos_ == "VERB":
                                v_lem_n = t_tok.head.lemma_.lower()
                                if v_lem_n not in cls._DELEXICAL_VERBS or v_lem_n in _SUPPORT_VERBS_NOUN:
                                    sc += 45
                            elif t_tok.dep_ in ("nsubj", "nsubjpass") and t_tok.head.pos_ in ("VERB", "AUX") and t_tok.head.lemma_.lower() not in cls._DELEXICAL_VERBS:
                                sc += 35
                            if any(c.dep_ in ("amod", "prep") for c in t_tok.children):
                                sc += 20
                            # C1: Shell noun complementation (`N + to + V`, `N + of + V-ing`, `N + that + clause`)
                            if anchor_type == "shell_to_do" or (best_pattern and "to do" in best_pattern.lower()):
                                if re.search(rf"\b{re.escape(t_tok.text.lower())}\b(?:\s+[a-z]+){{0,3}}\s+to\s+[a-z]{{2,}}\b", ex_norm):
                                    sc += 65
                                else:
                                    sc -= 40
                            elif anchor_type == "shell_of_doing" or (best_pattern and "doing" in best_pattern.lower()):
                                if re.search(rf"\b{re.escape(t_tok.text.lower())}\b(?:\s+[a-z]+){{0,2}}\s+[a-z]+\s+[a-z]{{2,}}ing\b", ex_norm):
                                    sc += 65
                                else:
                                    sc -= 40
                            elif anchor_type == "shell_that":
                                if re.search(rf"\b{re.escape(t_tok.text.lower())}\b(?:\s+[a-z]+){{0,2}}\s+that\s+[a-z]+\b", ex_norm):
                                    sc += 65
                                else:
                                    sc -= 40
                            # C3/C4: Directional prepositional valency (`target + prep` vs `prep + target`)
                            if anchor and anchor_type == "prep":
                                anc_p = str(anchor).strip().lower()
                                _bp_low = str(best_pattern or "").strip().lower()
                                _req_prep_dir = (
                                    "pre"
                                    if (
                                        _bp_low.startswith(anc_p + " ")
                                        or (
                                            q_norm
                                            and re.search(rf"\b{re.escape(anc_p)}\s+(?:(?:a|an|the|his|her|their|our|my|your|its)\s+)?{re.escape(ft_low)}\b", q_norm)
                                            and not re.search(rf"\b{re.escape(ft_low)}\s+{re.escape(anc_p)}\b", q_norm)
                                        )
                                    )
                                    else "post"
                                )
                                if cls._example_attests_prep_valency(ex_clean, ft_low, anc_p, prep_dir=_req_prep_dir):
                                    sc += 55
                                elif _req_prep_dir == "post" and any(c.dep_ == "prep" and c.i > t_tok.i and c.text.lower() != anc_p for c in t_tok.children):
                                    # Example attests a different post-nominal valency preposition (e.g. 'rise to')
                                    sc += 30
                            if has_uncountable_contrast_foil and not is_plural_noun:
                                if any(c.dep_ == "det" and c.text.lower() in ("a", "an", "each", "every") for c in t_tok.children):
                                    sc += 25
                    elif canonical_pos == "verb":
                        _is_postnom_root_verb = (
                            t_tok.dep_ == "amod"
                            and t_tok.head.dep_ == "ROOT"
                            and t_tok.head.pos_ in ("NOUN", "PROPN")
                            and t_tok.i > t_tok.head.i
                        )
                        if t_tok.pos_ == "NOUN" and (
                            t_tok.dep_ in ("pobj", "dobj", "nsubj", "attr")
                            or any(c.dep_ in ("det", "poss") for c in t_tok.children)
                        ):
                            sc -= 220 if any(c.dep_ in ("det", "poss") for c in t_tok.children) else 80
                        elif t_tok.dep_ == "amod" and not _is_postnom_root_verb:
                            sc -= 120 if ft_low == bw_low else 45
                        else:
                            has_post_to_inf = (
                                any((c.dep_ == "xcomp" or c.text.lower() == "to") and c.i > t_tok.i for c in t_tok.children)
                                or bool(re.search(rf"\b{re.escape(t_tok.text.lower())}\s+to\s+[a-z]+\b", ex_norm))
                            )
                            if has_post_to_inf:
                                if best_pattern and "to do" in best_pattern.lower():
                                    sc += 55
                                else:
                                    sc += 25
                            elif best_pattern and "to do" in best_pattern.lower():
                                sc -= 45
                            if anchor_type == "prep" and anchor and re.search(rf"\b{re.escape(anchor.lower())}\b", ex_norm):
                                sc += 50
                            has_ex_dobj = any(
                                c.dep_ in ("dobj", "obj") and c.pos_ in ("NOUN", "PROPN", "PRON") and c.i > t_tok.i
                                for c in t_tok.children
                            )
                            has_ex_lex_dobj = any(
                                c.dep_ in ("dobj", "obj") and c.pos_ in ("NOUN", "PROPN") and c.i > t_tok.i
                                for c in t_tok.children
                            )
                            has_fronted_wh_dobj = (
                                ex_clean.endswith("?")
                                and not has_ex_dobj
                                and any(c.dep_ in ("dobj", "obj") and c.i < t_tok.i for c in t_tok.children)
                            )
                            if has_fronted_wh_dobj:
                                sc -= 45
                            is_ex_passive = any(c.dep_ in ("auxpass", "nsubjpass") for c in t_tok.children)
                            has_fronted_rel_dobj = (
                                not has_ex_dobj
                                and not is_ex_passive
                                and (
                                    t_tok.dep_ == "relcl"
                                    or any(c.dep_ in ("dobj", "obj") and c.i < t_tok.i for c in t_tok.children)
                                )
                            )
                            if has_fronted_rel_dobj and anchor_type == "object":
                                sc -= 40
                            has_ex_active_subj = _is_postnom_root_verb or any(
                                c.dep_ == "nsubj" and c.pos_ in ("NOUN", "PROPN")
                                for c in (list(t_tok.children) + (list(t_tok.head.children) if t_tok.dep_ == "xcomp" else []))
                            )
                            if anchor_type == "sb_to_do":
                                if re.search(rf"\b{re.escape(t_tok.text.lower())}\s+(?:[a-z]+\s+){{1,3}}to\s+[a-z]+\b", ex_norm):
                                    sc += 65
                                elif not has_post_to_inf:
                                    sc -= 45
                            elif anchor_type == "verb_subject":
                                if not has_ex_dobj and not is_ex_passive and has_ex_active_subj:
                                    sc += 55
                                elif has_ex_dobj or is_ex_passive:
                                    sc -= 30
                            elif anchor_type == "object":
                                if has_ex_lex_dobj:
                                    sc += 55
                                elif not has_ex_dobj and not is_ex_passive:
                                    sc -= 35
                            else:
                                if has_ex_lex_dobj:
                                    sc += 40
                                elif ft_low != bw_low and is_ex_passive:
                                    sc += 40
                    elif canonical_pos == "adv":
                        if t_tok.dep_ in ("acomp", "attr", "nsubj", "dobj", "pobj") or (
                            t_tok.head.lemma_.lower() == "be"
                            and not any(c.dep_ in ("acomp", "attr") and c != t_tok for c in t_tok.head.children)
                        ):
                            sc -= 90
                        else:
                            _adv_h = t_tok.head
                            _is_sent_initial_discourse = (
                                (
                                    t_tok.i == 0
                                    or getattr(t_tok, "is_sent_start", False)
                                    or (t_tok.i > 0 and doc[t_tok.i - 1].text in (".", "!", "?", ";"))
                                )
                                and t_tok.i + 1 < len(doc)
                                and doc[t_tok.i + 1].text in (",", ";")
                            )
                            # Detect `be/get + Adv + VBN` stative/adjectival participle ("are happily married")
                            _is_be_adv_vbn = bool(
                                _adv_h is not None
                                and _adv_h.tag_ == "VBN"
                                and t_tok.i < _adv_h.i
                                and (
                                    any(c.lemma_.lower() in ("be", "get", "seem", "look", "remain") and c.dep_ in ("aux", "auxpass") for c in _adv_h.children)
                                    or (_adv_h.dep_ in ("acomp", "amod", "oprd") and _adv_h.head.lemma_.lower() in ("be", "get", "seem", "look", "remain"))
                                )
                            )
                            if anchor_type == "discourse":
                                if _is_sent_initial_discourse:
                                    sc += 65
                                elif t_tok.i > 0 and doc[t_tok.i - 1].text in (",", ";", "—", "-"):
                                    sc += 45
                            elif anchor_type == "modifies_adj":
                                if (
                                    t_tok.dep_ == "advmod"
                                    and _adv_h is not None
                                    and (_adv_h.pos_ in ("ADJ", "ADV") or _is_be_adv_vbn)
                                    and _adv_h.lemma_.lower() not in cls._ANCHOR_STOPWORDS
                                ):
                                    sc += 60
                                    if _adv_h.text[:1].isupper() and _adv_h.i > 0:
                                        sc -= 35
                                elif t_tok.i + 1 < len(doc) and doc[t_tok.i + 1].pos_ == "ADJ":
                                    sc += 50
                                    if doc[t_tok.i + 1].text[:1].isupper():
                                        sc -= 35
                                elif _adv_h is not None and _adv_h.pos_ == "VERB" and not _is_be_adv_vbn:
                                    sc -= 20
                            elif anchor_type == "modifies_verb":
                                if _is_be_adv_vbn:
                                    sc -= 40
                                elif _is_sent_initial_discourse:
                                    sc -= 30
                                elif (
                                    _adv_h is not None
                                    and _adv_h.pos_ == "VERB"
                                    and _adv_h.lemma_.lower() not in cls._ANCHOR_STOPWORDS
                                ):
                                    sc += 60
                                    if any(c.dep_ == "prep" and c.i > t_tok.i for c in _adv_h.children):
                                        sc += 15
                                elif t_tok.i > 0 and doc[t_tok.i - 1].pos_ == "VERB" and doc[t_tok.i - 1].lemma_.lower() not in cls._ANCHOR_STOPWORDS:
                                    sc += 45
                            else:
                                if (
                                    (t_tok.dep_ == "advmod" and _adv_h is not None and _adv_h.pos_ in ("VERB", "ADJ") and _adv_h.lemma_.lower() not in cls._ANCHOR_STOPWORDS)
                                    or (t_tok.i > 0 and doc[t_tok.i - 1].pos_ in ("ADJ", "VERB") and doc[t_tok.i - 1].lemma_.lower() not in cls._ANCHOR_STOPWORDS)
                                ):
                                    sc += 45
                                    if _adv_h is not None and _adv_h.pos_ == "ADJ" and _adv_h.text[:1].isupper() and _adv_h.i > 0:
                                        sc -= 30

            # Anchor & pattern alignment
            if anchor and anchor_type not in ("contextual", None):
                anc_l = str(anchor).strip().lower()
                if anc_l:
                    if anchor_type == "prep" and canonical_pos == "noun":
                        anc_match = (
                            cls._example_attests_prep_valency(ex_clean, ft_low, anc_l, prep_dir="post")
                            or cls._example_attests_prep_valency(ex_clean, ft_low, anc_l, prep_dir="pre")
                        )
                    elif anchor_type == "prep":
                        anc_match = bool(re.search(rf"\b{re.escape(anc_l)}\b", ex_norm))
                    else:
                        anc_match = bool(re.search(rf"\b{re.escape(anc_l)}\w*\b", ex_norm)) or cls.text_contains_form(ex_clean, anc_l)
                    if anc_match:
                        sc += 30
            if pat_hint and best_pattern and pat_hint == best_pattern:
                sc += 15

            scored_exs.append((sc, ex_clean, pat_hint, tier))

        if not scored_exs:
            return current_example, best_pattern, "none"

        scored_exs.sort(key=lambda x: x[0], reverse=True)
        win_score, win_ex, win_pat, win_tier = scored_exs[0]

        # Synchronize `best_pattern` with `win_ex`: if `best_pattern` was a `colloinexa` binomial or a fragment-only
        # pattern whose lexical words/prepositions/infinitive complements do not appear in `win_ex`, update or clear `best_pattern`
        def _pattern_matches_example(pat_str: Optional[str], ex_str: Optional[str]) -> bool:
            if not pat_str or not ex_str:
                return False
            p_c = re.sub(r"\[.*?\]|\(.*?\)", "", pat_str).strip().lower()
            ex_l = ex_str.lower()
            # 1. Multi-word phrase targets must have all phrase particles preserved in `pat_str`
            if " " in ft_low:
                ft_parts = [p for p in re.findall(r"[a-z]+", ft_low) if p not in ("sb", "sth", "somebody", "something", "one", "ones")]
                if not all(re.search(rf"\b{re.escape(p)}\b", p_c) for p in ft_parts):
                    return False
            # 2. Infinitive complement (`to do something`) must appear AFTER the target in `ex_str`
            if re.search(r"\bto\s+do\b", p_c):
                last_t = ft_low.split()[-1] if ft_low else bw_low
                last_forms = {last_t} | set(cls.inflected_forms(last_t))
                if not any(
                    re.search(rf"\b{re.escape(lf)}\b(?:\s+[a-z]+){{0,3}}\s+to\s+[a-z]+\b", ex_l)
                    for lf in last_forms if lf
                ):
                    return False
            # 3. Trailing preposition in `pat_str` (`clutter of`, `exposure to`) must directionally attach AFTER the target in `ex_str`
            p_no_inf = re.sub(r"\bto\s+do(?:\s+something|\s+somebody)?\b", "", p_c).strip()
            m_trail_p = re.search(r"\b(of|for|in|to|on|with|from|into|about|against|over|under|between|at|by|upon|towards)$", p_no_inf)
            if m_trail_p:
                prep_w = m_trail_p.group(1)
                last_t = ft_low.split()[-1] if ft_low else bw_low
                if prep_w != last_t and not cls._example_attests_prep_valency(ex_str, last_t, prep_w, prep_dir="post"):
                    return False
            # 4. Content words in `pat_str` (respecting slash alternatives like `importance/significance`) must appear in `ex_str`
            branches = (cls.clean_and_expand_pattern_slashes(p_c) if "/" in p_c else [p_c]) or [p_c]
            for br in branches:
                p_words = [
                    w for w in re.findall(r"[a-z]+", br)
                    if w not in ("something", "somebody", "someone", "sb", "sth", "do", "doing", "etc", "a", "an", "the", "one", "ones", "your", "his", "her", "their")
                    and w not in t_forms
                    and w not in set(ft_low.split())
                ]
                if not p_words or all(
                    re.search(rf"\b{re.escape(pw)}\w*\b", ex_l) or cls.text_contains_form(ex_str, pw)
                    for pw in p_words
                ):
                    return True
            return False

        synced_pattern = best_pattern
        if synced_pattern and win_ex and not _pattern_matches_example(synced_pattern, win_ex):
            if win_pat and _pattern_matches_example(win_pat, win_ex):
                synced_pattern = win_pat
            else:
                synced_pattern = None
                if locked_sense:
                    for pb_cand in cls.extract_sense_pattern_bundles(locked_sense, final_target, quote=quote or ""):
                        pb_p = pb_cand.get("pattern")
                        if pb_p and _pattern_matches_example(pb_p, win_ex):
                            synced_pattern = pb_p
                            break

        return win_ex, synced_pattern, win_tier

    # Backward compatibility alias
    select_optimal_six_source_example = select_optimal_five_source_example


    @classmethod
    def select_best_single_pattern(cls, patterns: List[str], target_word: str, preferred_anchor: Optional[str] = None) -> Optional[str]:
        """
        Selects exactly ONE best, high-value syntactic pattern from PedLex.
        Filters out noise like single verbs ('have') or repeated fragments.
        If preferred_anchor is given, boosts patterns containing that anchor (e.g. 'serious about' when anchor='about').
        """
        if not patterns:
            return None

        scored_patterns = []
        target_lower = target_word.lower()
        for p in patterns:
            p_clean = p.strip()
            if not p_clean:
                continue
            if " " not in p_clean and "/" not in p_clean:
                continue
            score = 0
            p_lower = p_clean.lower()
            if target_lower in p_lower:
                score += 10
            if preferred_anchor and re.search(rf"\b{re.escape(preferred_anchor.lower())}\b", p_lower):
                score += 15
            if "/" in p_clean:
                score += 5
            if any(prep in p_lower for prep in ["of", "to", "over", "for", "in", "on", "with", "from", "that", "about"]):
                score += 5
            if 6 <= len(p_clean) <= 40:
                score += 3
            scored_patterns.append((score, p_clean))

        if scored_patterns:
            scored_patterns.sort(key=lambda x: (x[0], -len(x[1])), reverse=True)
            return scored_patterns[0][1]

        return patterns[0].strip() if patterns else None
