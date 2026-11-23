"""
Deterministic contextual part-of-speech (POS) disambiguation and ranking using spaCy and morphology.
"""

import logging
import math
import os
import re
import sqlite3
import threading
from functools import lru_cache
from typing import Iterable, Any, Callable, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger("librarian.linguistics")


class ContextualPosMixin:
    @classmethod
    def determine_all_contextual_poses(cls, word: str, quoted_sentence: str) -> List[Tuple[int, str, str]]:
        """
        Returns a list of `(token_index, surface_text, contextual_pos)` for every occurrence
        of `word` (or its inflected forms) in `quoted_sentence` in linear sentence order.
        Useful when a homograph appears multiple times in the same sentence with different
        syntactic roles (e.g. "The cost of the damage will cost millions.").
        """
        sent_text = re.sub(r"^\s*\[?\bS-\d+\b\]?\s*[:\-]??\s*", "", quoted_sentence or "").strip()
        if sent_text.startswith("**") and sent_text.endswith("**") and len(sent_text) > 4:
            sent_text = sent_text[2:-2].strip()
        elif sent_text.startswith("*") and sent_text.endswith("*") and len(sent_text) > 2:
            sent_text = sent_text[1:-1].strip()
        if not sent_text or not word:
            return []
        sent_text = re.sub(r'(\w)(["“”])(\w)', r'\1 \2 \3', sent_text)
        sent_text = re.sub(r'(\S)(—|–|--)(\S)', r'\1 — \3', sent_text)
        sent_text = cls._strip_lexical_quotes(sent_text)
        caps_words = re.findall(r"\b[A-Z]{2,}\b", sent_text)
        if len(caps_words) >= 2:
            def _truecase_caps_match(m: re.Match) -> str:
                tok_str = m.group(0)
                start_i = m.start()
                prefix = sent_text[:start_i]
                if not prefix.strip() or re.search(r"(?:^|[:.!?—–])\s*[“\"'(\[]*\s*$", prefix):
                    return tok_str.capitalize()
                return tok_str.lower()
            sent_text = re.sub(r"\b[A-Z]{2,}\b", _truecase_caps_match, sent_text)

        nlp = cls.get_spacy()
        doc = nlp(sent_text)
        clean_w = re.sub(r"\[.*?\]|\(.*?\)", "", word).strip().lower()
        w_toks = clean_w.split()
        target_tok = w_toks[0] if w_toks else clean_w
        target_forms = cls.inflected_forms(clean_w) | {clean_w}
        matched_toks = []
        for tok in doc:
            tok_lower = tok.text.lower()
            tok_lem = tok.lemma_.lower()
            if (
                tok_lower == target_tok or tok_lem == target_tok or
                tok_lower in target_forms or tok_lem in target_forms or
                cls.form_matches(tok_lower, clean_w) or cls.form_matches(tok_lem, clean_w)
            ):
                matched_toks.append(tok)
        return [
            (tok.i, tok.text, cls.determine_contextual_pos(clean_w, quoted_sentence, occurrence=idx))
            for idx, tok in enumerate(matched_toks)
        ]

    @classmethod
    def determine_contextual_pos(
        cls,
        word: str,
        quoted_sentence: str,
        preferred_pos: Optional[str] = None,
        occurrence: Optional[int] = None,
    ) -> str:
        """
        Deterministically infers the contextual part of speech (noun, verb, adjective, adverb,
        preposition, conjunction, interjection) using spaCy dependency parsing and POS tagging.
        Runs entirely offline at zero token cost (<1ms).
        """
        VALID_POS_MAP = {
            "NOUN": "noun", "PROPN": "noun",
            "VERB": "verb", "AUX": "verb",
            "ADJ": "adjective",
            "ADV": "adverb",
            "ADP": "preposition",
            "CCONJ": "conjunction", "SCONJ": "conjunction",
            "DET": "determiner",
            "INTJ": "interjection",
        }
        _HINT_POS_MAP = {
            "n": "noun", "noun": "noun",
            "v": "verb", "vt": "verb", "vi": "verb", "verb": "verb",
            "adj": "adjective", "adjective": "adjective",
            "adv": "adverb", "adverb": "adverb",
            "prep": "preposition", "preposition": "preposition",
            "conj": "conjunction", "conjunction": "conjunction",
            "det": "determiner", "determiner": "determiner",
            "intj": "interjection", "interjection": "interjection",
        }
        if preferred_pos:
            preferred_pos = _HINT_POS_MAP.get(preferred_pos.strip().lower().rstrip("."), preferred_pos.strip().lower())
        elif word:
            m_hint = re.search(
                r"[\(\[]\s*(n\.?|noun|v\.?|vt\.?|vi\.?|verb|adj\.?|adjective|adv\.?|adverb|prep\.?|preposition|conj\.?|conjunction)\s*[\)\]]",
                word,
                re.IGNORECASE,
            )
            if m_hint:
                preferred_pos = _HINT_POS_MAP.get(m_hint.group(1).lower().rstrip("."))
        sent_text = re.sub(r"^\s*\[?\bS-\d+\b\]?\s*[:\-]??\s*", "", quoted_sentence or "").strip()
        # Strip outer markdown bold or italic wrappers (e.g. **Heading!**)
        if sent_text.startswith("**") and sent_text.endswith("**") and len(sent_text) > 4:
            sent_text = sent_text[2:-2].strip()
        elif sent_text.startswith("*") and sent_text.endswith("*") and len(sent_text) > 2:
            sent_text = sent_text[1:-1].strip()
        if not sent_text or not word:
            return "noun"

        # Normalize missing spaces around quotation marks (e.g. an"upright"message -> an "upright" message)
        sent_text = re.sub(r'(\w)(["“”])(\w)', r'\1 \2 \3', sent_text)
        # Normalize tight em-dashes / en-dashes / double-hyphens (e.g. crisis—which lasted three years—devastated)
        sent_text = re.sub(r'(\S)(—|–|--)(\S)', r'\1 — \3', sent_text)
        # Strip word-level double/single quotation marks so spaCy tags quoted nouns/verbs natively
        sent_text = cls._strip_lexical_quotes(sent_text)

        # Dual-View Non-Destructive ALL-CAPS Truecasing for POS Tagging:
        # When a sentence or clause has >= 2 ALL-CAPS words (e.g. 'WARNING: HIGH VOLTAGE CAN CAUSE SEVERE INJURY.',
        # 'DO NOT THROW AWAY.'), spaCy's statistical tagger mis-tags ALL-CAPS tokens as PROPN (NNP).
        # Normalize ALL-CAPS words to lowercase (preserving sentence/clause-initial title case) for syntactic parsing.
        caps_words = re.findall(r"\b[A-Z]{2,}\b", sent_text)
        if len(caps_words) >= 2:
            def _truecase_caps_match(m: re.Match) -> str:
                tok_str = m.group(0)
                start_i = m.start()
                prefix = sent_text[:start_i]
                if not prefix.strip() or re.search(r"(?:^|[:.!?—–])\s*[“\"'(\[]*\s*$", prefix):
                    return tok_str.capitalize()
                return tok_str.lower()
            sent_text = re.sub(r"\b[A-Z]{2,}\b", _truecase_caps_match, sent_text)

        nlp = cls.get_spacy()
        doc = nlp(sent_text)
        clean_w = re.sub(r"\[.*?\]|\(.*?\)", "", word).strip().lower()
        w_toks = clean_w.split()
        target_tok = w_toks[0] if w_toks else clean_w

        # 0. Contextual compound & hyphenated-word resolution:
        # Collect attested POS tags across PedLex and WordNet for hyphenated/open/closed variants,
        # and disambiguate multi-POS compounds (e.g. 'part-time', 'full-time', 'second-hand', 'follow-up',
        # 'grown-up', 'run-down', 'warm-up', 'high-rise', 'face-to-face', 'one-way') using the compound span's
        # syntactic slot in the sentence rather than blindly returning 'noun' or 'adjective' by fixed priority.
        wn = cls.get_wordnet()
        if "-" in clean_w or " " in clean_w:
            comp_poses: Set[str] = set()
            ld_cands = [clean_w, clean_w.replace("-", ""), clean_w.replace(" ", ""), clean_w.replace("-", " "), clean_w.replace(" ", "-")]
            for c_cand in ld_cands:
                ld_c_entry = cls.get_ldoce_entry(c_cand)
                if ld_c_entry:
                    for p in (ld_c_entry.get("all_poses") or [ld_c_entry.get("pos", "")]):
                        p_low = str(p).lower()
                        if "noun" in p_low and "pronoun" not in p_low:
                            comp_poses.add("noun")
                        if "adj" in p_low:
                            comp_poses.add("adjective")
                        if "adv" in p_low:
                            comp_poses.add("adverb")
                        if "verb" in p_low and "adverb" not in p_low:
                            comp_poses.add("verb")
                    for s in (ld_c_entry.get("senses") or []):
                        sp = str(s.get("pos") or "").lower()
                        if "noun" in sp and "pronoun" not in sp:
                            comp_poses.add("noun")
                        elif "adj" in sp:
                            comp_poses.add("adjective")
                        elif "adv" in sp:
                            comp_poses.add("adverb")
                        elif "verb" in sp:
                            comp_poses.add("verb")
            if wn:
                wn_map = {"n": "noun", "v": "verb", "a": "adjective", "s": "adjective", "r": "adverb"}
                for c_cand in (clean_w, clean_w.replace(" ", "-"), clean_w.replace("-", " "), clean_w.replace("-", "_")):
                    for syn in wn.synsets(c_cand):
                        if syn.pos in wn_map:
                            comp_poses.add(wn_map[syn.pos])
            # Productive adverbial use of 'X-to-X' / 'one-way' / 'part-time' / 'second-hand' compounds
            parts_hy = [p for p in re.split(r"[\s\-]+", clean_w) if p]
            if len(parts_hy) == 3 and parts_hy[1] == "to" and parts_hy[0] == parts_hy[2]:
                comp_poses.add("adverb")
            elif clean_w in ("one-way", "part-time", "full-time", "second-hand", "first-hand", "first-class", "off-peak", "full-scale"):
                comp_poses.add("adverb")

            if len(comp_poses) == 1:
                return next(iter(comp_poses))

            if comp_poses:
                # Locate the hyphenated/multi-word span in the parsed sentence
                comp_tokens = [p for p in re.split(r"[\s\-]+", clean_w) if p]
                span_start = None
                span_end = None
                if comp_tokens:
                    for idx_t in range(len(doc)):
                        if doc[idx_t].lower_ == comp_tokens[0] or doc[idx_t].lemma_.lower() == comp_tokens[0]:
                            cur_j = idx_t
                            ok_match = True
                            for ct in comp_tokens[1:]:
                                cur_j += 1
                                while cur_j < len(doc) and doc[cur_j].text == "-":
                                    cur_j += 1
                                if cur_j >= len(doc) or (doc[cur_j].lower_ != ct and doc[cur_j].lemma_.lower() != ct):
                                    ok_match = False
                                    break
                            if ok_match:
                                span_start = idx_t
                                span_end = cur_j
                                break
                if span_start is not None and span_end is not None:
                    span_slice = doc[span_start : span_end + 1]
                    span_set = {t.i for t in span_slice}
                    next_tok = doc[span_end + 1] if span_end + 1 < len(doc) else None
                    prev_tok = doc[span_start - 1] if span_start > 0 else None
                    ext_heads = [t.head for t in span_slice if t.head.i not in span_set]
                    ext_head = ext_heads[0] if ext_heads else span_slice[-1].head

                    # 1) Attributive modifier immediately preceding a head noun (or modifying an external noun to its right):
                    is_attr_before_noun = bool(
                        next_tok is not None
                        and (
                            next_tok.pos_ in ("NOUN", "PROPN")
                            or (next_tok.pos_ == "ADJ" and span_end + 2 < len(doc) and doc[span_end + 2].pos_ in ("NOUN", "PROPN"))
                        )
                        and (
                            any(t.dep_ in ("amod", "compound", "nmod") and t.head.i > span_end for t in span_slice)
                            or (ext_head.i >= next_tok.i and ext_head.pos_ in ("NOUN", "PROPN"))
                        )
                    )
                    if is_attr_before_noun and "adjective" in comp_poses:
                        return "adjective"

                    # 2) Predicative adjective after copula/linking verb WITHOUT a determiner ('is very grown-up', 'looks run-down'):
                    has_ext_det_or_poss = any(
                        c.dep_ in ("det", "poss") and c.i < span_start
                        for t in span_slice for c in t.children
                    ) or (prev_tok is not None and prev_tok.pos_ == "DET")
                    if (
                        not has_ext_det_or_poss
                        and ext_head.lemma_.lower() in ("be", "become", "seem", "look", "feel", "get", "remain", "stay", "appear", "sound")
                        and not (next_tok is not None and next_tok.pos_ in ("NOUN", "PROPN"))
                        and "adjective" in comp_poses
                    ):
                        return "adjective"

                    # 3) Nominal head slot: governed by a determiner/possessive/adjective/preposition and NOT modifying a right-hand noun
                    if not is_attr_before_noun and "noun" in comp_poses:
                        has_left_adj_or_det = has_ext_det_or_poss or (prev_tok is not None and prev_tok.pos_ in ("ADJ", "ADP"))
                        has_right_pp = next_tok is not None and next_tok.pos_ == "ADP"
                        if has_left_adj_or_det or has_right_pp or any(t.dep_ in ("nsubj", "nsubjpass", "pobj", "attr") for t in span_slice):
                            return "noun"

                    # 4) Post-verbal adverbial modifier ('works part-time', 'working full-time', 'bought furniture second-hand', 'met face-to-face', 'flows one-way'):
                    if not is_attr_before_noun and "adverb" in comp_poses:
                        is_modifying_verb = (
                            any(t.dep_ in ("advmod", "npadvmod") and t.head.pos_ in ("VERB", "AUX", "ADP") for t in span_slice)
                            or (
                                ext_head.pos_ in ("VERB", "AUX", "ADP")
                                and ext_head.i < span_start
                                and not has_ext_det_or_poss
                            )
                        )
                        if is_modifying_verb:
                            return "adverb"

                # Fallback priority if span slot is inconclusive
                for pref_p in ("adjective", "noun", "adverb", "verb"):
                    if pref_p in comp_poses:
                        return pref_p

        # 1. Exact match on token text, lemma, or morphological inflection
        # Prioritize exact surface matches over lemma-only matches when multiple forms occur in one sentence
        # (e.g. 'costs' vs 'cost' in "The high cost of materials costs the company millions."),
        # and support explicit occurrence index or preferred_pos disambiguation for identical surface homographs
        # (e.g. "The cost of the damage will cost millions.").
        target_forms = cls.inflected_forms(clean_w) | {clean_w}
        exact_surface_toks = []
        lemma_or_inflected_toks = []
        all_matched_in_order = []
        for _cand_tok in doc:
            _tl = _cand_tok.text.lower()
            _tlem = _cand_tok.lemma_.lower()
            if _tl == target_tok:
                exact_surface_toks.append(_cand_tok)
                all_matched_in_order.append(_cand_tok)
            elif (
                _tlem == target_tok or
                _tl in target_forms or _tlem in target_forms or
                cls.form_matches(_tl, clean_w) or cls.form_matches(_tlem, clean_w)
            ):
                lemma_or_inflected_toks.append(_cand_tok)
                all_matched_in_order.append(_cand_tok)

        if occurrence is not None and 0 <= occurrence < len(all_matched_in_order):
            ordered_cand_toks = [all_matched_in_order[occurrence]]
        else:
            if preferred_pos and len(all_matched_in_order) > 1:
                for _occ_i in range(len(all_matched_in_order)):
                    if cls.determine_contextual_pos(clean_w, quoted_sentence, occurrence=_occ_i) == preferred_pos:
                        return preferred_pos
            ordered_cand_toks = exact_surface_toks + lemma_or_inflected_toks

        for tok in ordered_cand_toks:
            tok_lower = tok.text.lower()
            tok_lem = tok.lemma_.lower()
            if True:
                # Special contextual overrides for degree/directional adverbs and compound prepositions
                if clean_w in ("outside", "inside") and tok.i + 1 < len(doc) and doc[tok.i + 1].lower_ == "of":
                    return "preposition"
                if clean_w == "way" and "way too" in sent_text.lower():
                    return "adverb"
                if clean_w in ("clockwise", "counterclockwise", "anticlockwise"):
                    if tok.head.pos_ in ("VERB", "AUX") or tok.dep_ in ("advmod", "npadvmod", "dobj"):
                        return "adverb"
                if clean_w in ("interconnected", "intrigued", "oppressed", "glittering"):
                    return "adjective"
                if clean_w == "verge" and "on the verge of" in sent_text.lower():
                    return "noun"
                if clean_w == "retail" and "retail jobs" in sent_text.lower():
                    return "noun"
                if clean_w in ("vaguely", "furiously", "tearfully"):
                    return "adverb"

                # Collect dictionary-attested POS tags for clean_w (if clean_w is in PedLex)
                ld_target = cls.get_ldoce_entry(clean_w)
                dict_poses: Set[str] = set()
                if ld_target:
                    for p in (ld_target.get("all_poses") or []):
                        p_low = str(p).lower()
                        if "noun" in p_low and "pronoun" not in p_low:
                            dict_poses.add("noun")
                        if "adv" in p_low:
                            dict_poses.add("adverb")
                        if "verb" in p_low and "adverb" not in p_low:
                            dict_poses.add("verb")
                        if "adj" in p_low:
                            dict_poses.add("adjective")
                        if "prep" in p_low:
                            dict_poses.add("preposition")
                        if "conj" in p_low:
                            dict_poses.add("conjunction")
                        if "det" in p_low:
                            dict_poses.add("determiner")
                    for s in (ld_target.get("senses") or []):
                        sp = str(s.get("pos") or "").lower()
                        if "noun" in sp and "pronoun" not in sp:
                            dict_poses.add("noun")
                        elif "adv" in sp:
                            dict_poses.add("adverb")
                        elif "verb" in sp:
                            dict_poses.add("verb")
                        elif "adj" in sp:
                            dict_poses.add("adjective")
                        elif "prep" in sp:
                            dict_poses.add("preposition")
                        elif "conj" in sp:
                            dict_poses.add("conjunction")
                        elif "det" in sp:
                            dict_poses.add("determiner")

                # Compute whether the matched token in the sentence exhibits an unmistakable verbal argument/auxiliary frame
                # (used to distinguish verbal gerunds & present participles from lexicalized -ing/-ed nouns and adjectives).
                has_contracted_s_ing_frame = bool(
                    tok_lower.endswith("ing")
                    and tok.dep_ == "ROOT"
                    and tok.i > 0
                    and doc[tok.i - 1].lower_ == "'s"
                    and not any(
                        t.pos_ in ("VERB", "AUX") and t.dep_ in ("ROOT", "conj") and t.i != tok.i
                        for t in doc
                    )
                    and any(
                        c.dep_ in ("acl", "xcomp", "ccomp", "dobj", "obj", "prep", "advmod", "npadvmod", "prt")
                        for c in tok.children
                    )
                )
                # Post-Parenthetical Main-Verb Counterfactual Projection:
                # When an intervening parenthetical ('—which lasted three years—', ', which lasted three years,', '(...)')
                # separates the subject NP from an '-ed' transitive main verb ('devastated trade'), spaCy mis-attaches
                # the finite verb as 'amod' of its bare object noun and leaves the subject NP as a verbless ROOT.
                has_post_paren_main_verb = False
                if (
                    tok.i > 1
                    and doc[tok.i - 1].text in ("—", "–", "--", ",", ")")
                    and tok.dep_ in ("amod", "acl", "dep", "conj")
                ):
                    stripped_paren = re.sub(
                        r"(?:\s*[—–]\s*[^—–]+\s*[—–]\s*|\s*,\s*[^,]+\s*,\s*|\s*\([^)]+\)\s*)",
                        " ",
                        sent_text,
                    )
                    stripped_paren = re.sub(r"\s{2,}", " ", stripped_paren).strip()
                    if stripped_paren and stripped_paren != sent_text:
                        for st in nlp(stripped_paren):
                            if st.text.lower() == tok_lower and (
                                st.dep_ in ("ROOT", "conj", "ccomp", "advcl")
                                or st.tag_ in ("VBD", "VBP", "VBZ")
                                or any(c.dep_ in ("dobj", "obj", "nsubj") for c in st.children)
                            ):
                                has_post_paren_main_verb = True
                                break

                has_verbal_frame = (
                    has_contracted_s_ing_frame
                    or has_post_paren_main_verb
                    or any(
                        c.dep_ in ("aux", "auxpass", "agent", "ccomp", "xcomp", "prt")
                        or (c.dep_ == "advmod" and tok.dep_ not in ("amod", "acomp", "oprd"))
                        or (c.dep_ in ("dobj", "obj", "iobj", "oprd") and tok.dep_ != "amod")
                        or (c.dep_ in ("nsubj", "nsubjpass", "csubj") and not any(ch.dep_ in ("det", "poss") for ch in tok.children))
                        for c in tok.children
                    )
                    or (
                        tok.dep_ in ("ROOT", "advcl", "acl", "relcl")
                        and not any(c.dep_ in ("det", "poss") for c in tok.children)
                        and any(c.dep_ in ("nsubj", "nsubjpass", "agent", "prep", "npadvmod") for c in tok.children)
                    )
                    or (
                        tok.tag_ == "VBG"
                        and tok.i == 0
                        and tok.head.tag_ in ("NNS", "NNPS")
                        and tok.head.dep_ in ("nsubj", "nsubjpass", "csubj")
                        and (
                            (tok.dep_ in ("compound", "amod") and tok.head.head.tag_ == "VBZ")
                            or (tok.dep_ == "compound" and not cls.get_ldoce_entry(f"{tok_lower} {tok.head.lemma_.lower()}"))
                        )
                    )
                )

                # If clean_w is a base lemma (e.g. 'exist', 'remain', 'warn') while the actual token in the sentence
                # (tok_lower, e.g. 'existing', 'remaining', 'warning') has its own dedicated lexicalized PedLex entry
                # and is NOT in a verbal frame, merge the surface entry's POS tags so single-POS base verbs don't
                # prematurely override participial adjectives or verbal nouns.
                surf_ld = cls.get_ldoce_entry(tok_lower) if tok_lower != clean_w else ld_target
                surf_poses_set: Set[str] = set()
                surf_has_pred_adj = False
                if surf_ld and str(surf_ld.get("word") or "").lower() == tok_lower:
                    for p in (surf_ld.get("all_poses") or []):
                        p_low = str(p).lower()
                        if "noun" in p_low:
                            surf_poses_set.add("noun")
                        if "adj" in p_low:
                            surf_poses_set.add("adjective")
                        if "verb" in p_low and "adverb" not in p_low:
                            surf_poses_set.add("verb")
                    for s in (surf_ld.get("senses") or []):
                        sp = str(s.get("pos") or "").lower()
                        if "noun" in sp:
                            surf_poses_set.add("noun")
                        elif "adj" in sp:
                            surf_poses_set.add("adjective")
                            if "only before noun" not in str(s.get("grammar") or "").lower():
                                surf_has_pred_adj = True
                        elif "verb" in sp:
                            surf_poses_set.add("verb")
                    if tok_lower != clean_w and surf_poses_set:
                        if not has_verbal_frame:
                            dict_poses |= surf_poses_set

                # Conversely, if clean_w is an inflected -ing/-ed form (e.g. 'opening', 'rising', 'demanding', 'understanding', 'planning')
                # whose surface entry in PedLex only lists noun/adjective, fuse 'verb' from its base verb lemma when it
                # appears in a verbal syntactic frame (e.g. 'Opening the heavy door...', 'The company's planning to launch...').
                if "verb" not in dict_poses and (tok.pos_ in ("VERB", "AUX") or has_contracted_s_ing_frame):
                    cand_verb_lem = tok_lem if tok_lem != clean_w else (tok_lower[:-3] if tok_lower.endswith("ing") and len(tok_lower) > 5 else tok_lem)
                    if cand_verb_lem.endswith(("nn", "mm", "pp", "tt", "gg", "dd", "bb", "rr")) and len(cand_verb_lem) >= 4:
                        cand_verb_lem_trimmed = cand_verb_lem[:-1]
                    else:
                        cand_verb_lem_trimmed = cand_verb_lem
                    lem_ld = cls.get_ldoce_entry(cand_verb_lem_trimmed) or (cls.get_ldoce_entry(tok_lem) if tok_lem != clean_w else None)
                    lem_v_senses = [
                        s for s in ((lem_ld.get("senses") or []) if lem_ld else [])
                        if isinstance(s, dict) and "verb" in str(s.get("pos") or lem_ld.get("pos") or "").lower()
                    ]
                    lem_is_strictly_intransitive = bool(
                        lem_v_senses
                        and all(
                            "intransitive" in str(s.get("grammar") or "").lower()
                            and not re.search(r"(?<!in)transitive", str(s.get("grammar") or "").lower())
                            for s in lem_v_senses
                        )
                    )
                    lem_has_verb = bool(
                        (lem_ld and any("verb" in str(p).lower() and "adverb" not in str(p).lower() for p in (lem_ld.get("all_poses") or [lem_ld.get("pos", "")])))
                        or (wn and (wn.synsets(clean_w, pos="v") or wn.synsets(tok_lem, pos="v") or wn.synsets(cand_verb_lem_trimmed, pos="v")))
                    )
                    eff_verbal_frame = has_verbal_frame
                    if (
                        eff_verbal_frame
                        and lem_is_strictly_intransitive
                        and tok.tag_ == "VBG"
                        and not has_contracted_s_ing_frame
                        and not any(c.dep_ in ("aux", "auxpass", "agent", "ccomp", "xcomp", "prt", "nsubj", "advmod", "prep") for c in tok.children)
                    ):
                        # Strictly [intransitive] base verb cannot take a direct object ('with persevering industry')
                        eff_verbal_frame = False
                    if lem_has_verb and (clean_w == tok_lem or eff_verbal_frame):
                        dict_poses.add("verb")
                        if has_contracted_s_ing_frame or has_post_paren_main_verb:
                            return "verb"
                if (
                    "adjective" not in dict_poses
                    and wn
                    and not has_verbal_frame
                    and ((tok.pos_ == "ADJ" and tok.dep_ != "compound") or (tok.dep_ == "amod" and tok.head.pos_ in ("NOUN", "PROPN")))
                    and (wn.synsets(clean_w, pos="a") or wn.synsets(clean_w, pos="s") or wn.synsets(tok_lower, pos="a") or wn.synsets(tok_lower, pos="s"))
                ):
                    dict_poses.add("adjective")

                # Causative/Resultative, Copular Predicative, or Attributive Comparative Adjective Invariant:
                # When a surface token is an inflected comparative (-er) or superlative (-est) adjective
                # (e.g. 'warmer' -> 'warm', 'colder' -> 'cold', 'cleaner' -> 'clean', 'faster' -> 'fast'),
                # resolve to 'adjective' if it occupies an adjective syntactic slot, preventing accidental
                # noun homographs (e.g. 'warmer' [plate warmer], 'cooler' [cool box], 'cleaner' [carpet cleaner])
                # from hijacking predicative or causative object complement slots.
                base_adj = cls.deinflect_comparative_superlative(clean_w) or cls.deinflect_comparative_superlative(tok_lower)
                if base_adj:
                    is_causative = cls.is_causative_object_complement(tok, doc)
                    is_copula_pred = (
                        tok.head.lemma_.lower() in ("be", "become", "seem", "look", "feel", "get", "remain", "stay", "grow", "appear", "sound")
                        and not any(c.pos_ == "VERB" and c.tag_ in ("VBN", "VBG") for c in tok.head.children)
                    )
                    is_attr_before_noun = (
                        tok.dep_ == "amod"
                        or (tok.i + 1 < len(doc) and doc[tok.i + 1].pos_ in ("NOUN", "PROPN") and not doc[tok.i + 1].text.startswith((".", ",", "!", "?", ":", ";", '"', "'", "”", "’")))
                    )
                    if is_causative or is_copula_pred or is_attr_before_noun or tok.dep_ in ("acomp", "oprd"):
                        return "adjective"
                    dict_poses.add("adjective")

                # If PedLex exclusively defines this headword under a single POS (e.g. 'elicit' -> verb, 'namely' -> adverb, 'stripe' -> noun),
                # return that authoritative dictionary POS immediately so headword and definition never desynchronize!
                if len(dict_poses) == 1:
                    return next(iter(dict_poses))

                # 0a. Morphological Comparative/Superlative Inflection Gate (-er / -est / -ier / -iest):
                # Verbs never inflect with -er/-est; when "noun" is not in dict_poses and "adjective" is in dict_poses,
                # a comparative/superlative surface form (e.g. 'plumper' from 'plump') is unambiguously an adjective.
                if "adjective" in dict_poses and "noun" not in dict_poses and tok_lower != clean_w and len(clean_w) >= 3:
                    expected_deg = {
                        clean_w + "er", clean_w + "r",
                        clean_w + "est", clean_w + "st",
                    }
                    if clean_w.endswith("y"):
                        expected_deg.update({clean_w[:-1] + "ier", clean_w[:-1] + "iest"})
                    if clean_w[-1] in "bdfgklmnprstz" and clean_w[-2] in "aeiou":
                        expected_deg.update({clean_w + clean_w[-1] + "er", clean_w + clean_w[-1] + "est"})
                    if tok_lower in expected_deg:
                        surf_e = cls.get_ldoce_entry(tok_lower)
                        is_standalone_noun = bool(
                            surf_e and surf_e.get("word", "").lower() == tok_lower
                            and any("noun" in str(p).lower() for p in (surf_e.get("all_poses") or [surf_e.get("pos", "")]))
                        )
                        if not is_standalone_noun:
                            return "adjective"

                # 0b. Copula-Ellipsis / Gapping Coordination Gate:
                # e.g. "The leaves were straighter and taller than ordinary ones, and the grains plumper."
                # where spaCy tags the gapped predicate 'plumper' as NOUN conj of 'were' (with 'straighter' as acomp).
                if tok.pos_ == "NOUN" and tok.dep_ == "conj" and "adjective" in dict_poses and "noun" not in dict_poses:
                    if tok.head.pos_ == "ADJ" or (
                        tok.head.lemma_.lower() in ("be", "become", "get", "grow", "seem", "look", "feel", "remain", "stay")
                        and any(ch.dep_ in ("acomp", "oprd") or ch.pos_ == "ADJ" for ch in tok.head.children)
                    ):
                        return "adjective"

                # 0. Dictionary POS hard gate for spaCy mistagged VERB:
                if tok.pos_ in ("VERB", "AUX"):
                    # Verbal noun / gerund governed by a determiner or possessive ('the striving', 'his constant striving for...')
                    if "noun" in dict_poses and any(c.dep_ in ("det", "poss") or (c.dep_ == "amod" and c.i < tok.i) for c in tok.children):
                        return "noun"
                    # Bare uncountable activity gerund noun without verbal arguments ('Reading is good for the mind', 'She enjoys reading and painting')
                    if (
                        tok.tag_ == "VBG"
                        and not has_verbal_frame
                        and "noun" in surf_poses_set
                        and tok.dep_ in ("nsubj", "nsubjpass", "dobj", "obj", "pobj", "conj", "attr")
                        and not any(c.dep_ in ("dobj", "obj", "aux", "auxpass", "advmod", "prep", "prt", "ccomp", "xcomp") for c in tok.children)
                    ):
                        return "noun"
                    # Coordinated Subject Noun Parallelism ('Exercise and rest lower blood pressure naturally'):
                    # A sentence-initial NOUN ROOT ('Exercise') cannot coordinate via 'and' with a finite VERB ('rest')
                    # when followed by a transitive verb homograph ('lower') and its object ('blood pressure').
                    if (
                        "noun" in dict_poses
                        and tok.dep_ == "conj"
                        and tok.head.dep_ == "ROOT"
                        and tok.head.pos_ in ("NOUN", "PROPN")
                        and any(c.dep_ in ("dobj", "obj") for c in tok.children)
                    ):
                        return "noun"
                    # Plural-Noun Subject + Reduced Relative / Downstream Subjectless Predicate Invariant
                    # (e.g. "issued the joint declaration that 'objects acquired in earlier times must be viewed...'"):
                    # When spaCy mis-tags a plural noun ('objects', 'results', 'claims', 'records') after 'that' as VBZ ('relcl'),
                    # attaching its reduced-relative passive participle ('acquired') as 'xcomp' and stranding the true clause
                    # predicate ('must be viewed') as a subjectless, conjunctionless verb.
                    if "noun" in dict_poses and tok.tag_ == "VBZ":
                        _vbn_catenatives = {
                            "be", "get", "have", "become", "seem", "look", "feel", "remain", "stay",
                            "keep", "leave", "make", "find", "see", "hear", "watch", "notice",
                            "want", "need", "like", "prefer", "consider", "deem", "declare", "prove", "render",
                        }
                        _has_bare_vbn_mod = (
                            clean_w not in _vbn_catenatives
                            and any(
                                c.tag_ == "VBN"
                                and c.dep_ in ("xcomp", "acl", "amod", "dep")
                                and c.i > tok.i
                                for c in tok.children
                            )
                        )
                        _has_stranded_modal_or_plural_pred = False
                        for _k in range(tok.i + 1, len(doc)):
                            _tk = doc[_k]
                            if _tk.text in (";", ":", ".", "!", "?"):
                                break
                            if _tk.pos_ in ("VERB", "AUX") and (
                                any(ch.dep_ in ("aux", "auxpass") or ch.tag_ == "MD" for ch in _tk.children)
                                or _tk.tag_ in ("VBP", "MD")
                            ):
                                _has_pred_subj = any(
                                    ch.dep_ in ("nsubj", "nsubjpass", "csubj", "csubjpass", "expl")
                                    for ch in _tk.children
                                )
                                _has_pred_cc = (
                                    any(ch.dep_ == "cc" for ch in _tk.children)
                                    or (_k > 0 and doc[_k - 1].lower_ in ("and", "or", "but"))
                                )
                                if not _has_pred_subj and not _has_pred_cc:
                                    _has_stranded_modal_or_plural_pred = True
                                    break
                        if _has_bare_vbn_mod or (
                            _has_stranded_modal_or_plural_pred
                            and tok.i > 0
                            and doc[tok.i - 1].lower_ in ("that", "whether", "if", "although", "though", "because", "while", "since", "as")
                        ):
                            return "noun"
                    dict_available = bool(dict_poses) or bool(wn and getattr(wn, "synsets", None))
                    if dict_available:
                        has_v = ("verb" in dict_poses) if dict_poses else bool(wn.synsets(clean_w, pos="v") or wn.synsets(tok_lem, pos="v"))
                        if not has_v:
                            if dict_poses:
                                if tok.dep_ in ("amod", "acomp", "oprd") and "adjective" in dict_poses:
                                    return "adjective"
                                for pref in ("noun", "adjective", "adverb"):
                                    if pref in dict_poses:
                                        return pref
                            return "noun"

                # 0c. Cross-POS Syntactic Gates for ADV / ADJ / PREP / CONJ / VERB / NOUN Chameleons:
                # 1) Nominal Slot Invariant for Mistagged ADV/ADP/SCONJ ('from the deep well', 'his own good'):
                if tok.pos_ in ("ADV", "ADP", "SCONJ") and "noun" in dict_poses:
                    _has_nom_mod = any(
                        c.dep_ in ("det", "poss", "nummod", "relcl", "case")
                        or (c.dep_ in ("amod", "compound") and c.i < tok.i)
                        for c in tok.children
                    )
                    if _has_nom_mod and tok.dep_ in ("pobj", "dobj", "obj", "iobj", "nsubj", "nsubjpass", "attr", "appos", "conj", "ROOT"):
                        return "noun"

                # 2) Verbal Clause / Infinitive / Causative-Help Invariant for Mistagged ADV/ADP
                # ('Many exiled writers long for their homeland', 'Most children like sweet fruit',
                #  'helped to still the panicked crowd', 'helped the visiting team even the score'):
                if tok.pos_ in ("ADV", "ADP", "SCONJ") and "verb" in dict_poses:
                    _has_clause_verb = any(
                        t.pos_ in ("VERB", "AUX")
                        and t.dep_ in ("ROOT", "conj", "advcl", "ccomp")
                        and t.i != tok.i
                        for t in doc
                    )
                    # Case 2a: Sentence ROOT or zero-verb main clause ('writers long for their homeland', 'children like sweet fruit')
                    if (
                        (tok.dep_ == "ROOT" and any(c.dep_ in ("nsubj", "nsubjpass", "dobj", "obj", "prep") for c in tok.children))
                        or (
                            not _has_clause_verb
                            and doc[-1].text in (".", "!", "?")
                            and tok.dep_ == "prep"
                            and tok.head.dep_ == "ROOT"
                            and tok.head.pos_ in ("NOUN", "PROPN", "PRON")
                            and tok.head.i < tok.i
                            and any(c.dep_ in ("pobj", "dobj", "obj") for c in tok.children)
                        )
                    ):
                        return "verb"
                    # Case 2b: 'to + V + NP' misparsed as 'to (ADP) + pcomp (ADV) + dobj on governing verb' ('helped to still the panicked crowd')
                    if (
                        tok.dep_ in ("pcomp", "pobj", "advmod")
                        and tok.i > 0
                        and doc[tok.i - 1].lower_ == "to"
                        and tok.i + 1 < len(doc)
                        and doc[tok.i + 1].pos_ in ("DET", "PRON", "ADJ", "NOUN", "PROPN")
                    ):
                        return "verb"
                    # Case 2c: Causative / Help / Perception bare infinitive + NP ('helped the visiting team even the score')
                    if (
                        tok.dep_ == "advmod"
                        and tok.head.pos_ in ("NOUN", "PROPN")
                        and tok.head.i > tok.i
                        and any(c.dep_ in ("det", "poss") and c.i > tok.i for c in tok.head.children)
                        and tok.i >= 2
                        and doc[tok.i - 1].pos_ in ("NOUN", "PROPN", "PRON")
                    ):
                        _anc_verbs = [
                            t for t in (tok.head.head, tok.head.head.head if tok.head.head else None)
                            if t is not None and t.pos_ == "VERB" and t.i < tok.i - 1
                        ]
                        if any(av.lemma_.lower() in ("help", "make", "let", "have", "watch", "see", "hear") for av in _anc_verbs):
                            return "verb"

                # 3) Preposition vs. Subordinating Conjunction vs. Bare Adverb Invariant
                # ('since 1985' [prep] vs 'since the treaty was signed' [conj] vs 'has not been seen since' [adv];
                #  'flew past overhead' [adv] vs 'sprinted past the hall' [prep]):
                if ("preposition" in dict_poses or "conjunction" in dict_poses) and tok.pos_ in ("ADP", "SCONJ", "CCONJ", "ADV", "ADJ"):
                    # Do NOT treat an ADJ inside an ADP phrase ('of like mind', 'in like manner') as a preposition!
                    if tok.pos_ == "ADJ" and tok.head.pos_ == "ADP" and "adjective" in dict_poses:
                        return "adjective"
                    _has_pobj = any(c.dep_ == "pobj" for c in tok.children) and tok.pos_ != "ADJ"
                    _has_clausal = (
                        tok.dep_ == "mark"
                        or any(c.dep_ in ("advcl", "ccomp", "pcomp") and c.pos_ in ("VERB", "AUX") for c in tok.children)
                        or (
                            tok.dep_ == "advmod"
                            and tok.i == 0
                            and any(c.dep_ in ("nsubj", "nsubjpass") or c.pos_ in ("VERB", "AUX") for c in tok.children)
                        )
                    )
                    if _has_pobj and not _has_clausal and "preposition" in dict_poses:
                        return "preposition"
                    if _has_clausal and not _has_pobj and "conjunction" in dict_poses:
                        return "conjunction"
                    if (
                        not _has_pobj
                        and not _has_clausal
                        and "adverb" in dict_poses
                        and (
                            (
                                tok.dep_ in ("advmod", "prt", "prep")
                                and (
                                    tok.head.pos_ in ("VERB", "AUX", "ADV")
                                    or (tok.head.pos_ == "NOUN" and tok.head.dep_ in ("advmod", "npadvmod") and tok.i < tok.head.i)
                                )
                            )
                            or (tok.dep_ == "pobj" and tok.head.pos_ == "ADP")
                        )
                    ):
                        return "adverb"

                # 4) Linking-Verb Predicative Complement for 'well' ('feel well', 'be well', 'look well', 'get well'):
                # In PedLex, 'well' after 'be / feel / look / seem / get / become / remain / stay' without a passive main verb
                # is the predicative [not before noun] adjective meaning 'healthy / in good condition'.
                if (
                    clean_w == "well"
                    and "adjective" in dict_poses
                    and tok.dep_ in ("advmod", "acomp", "oprd")
                    and tok.head.lemma_.lower() in ("feel", "be", "look", "seem", "get", "become", "remain", "stay", "appear", "sound")
                    and tok.head.pos_ in ("VERB", "AUX")
                    and not any(c.pos_ == "VERB" and c.tag_ in ("VBN", "VBG") for c in tok.head.children)
                ):
                    return "adjective"

                # If token is an adverb (ADV) or only has adverb/noun in dict, do NOT force it to adjective!
                if tok.pos_ == "ADV" and ("adverb" in dict_poses or not ("adjective" in dict_poses)):
                    return "adverb"

                # Coordinated Subject + Transitive ADJ/VERB Homograph Invariant
                # (e.g. 'Exercise and rest lower blood pressure naturally'):
                if (
                    tok.dep_ == "amod"
                    and "verb" in dict_poses
                    and tok.head.pos_ in ("NOUN", "PROPN")
                    and tok.i >= 2
                    and doc[tok.i - 1].dep_ == "conj"
                    and doc[tok.i - 1].head.dep_ == "ROOT"
                    and doc[tok.i - 1].head.pos_ in ("NOUN", "PROPN")
                    and tok.head.dep_ in ("dobj", "obj", "npadvmod")
                ):
                    return "verb"

                # Coordinated Transitive Verb + Bare Object Invariant
                # (e.g. 'Workers had to dig trenches and pipe water to the village'):
                if (
                    tok.dep_ == "compound"
                    and "verb" in dict_poses
                    and tok.head.pos_ in ("NOUN", "PROPN")
                    and tok.head.dep_ == "conj"
                    and tok.head.head.dep_ in ("dobj", "obj")
                    and tok.i > 0
                    and doc[tok.i - 1].lower_ in ("and", "or")
                ):
                    _pair_str = f"{tok.lemma_.lower()} {tok.head.lemma_.lower()}"
                    _is_comp_pair = bool(
                        cls.get_ldoce_entry(_pair_str)
                        or (wn and (wn.synsets(_pair_str.replace(" ", "_"), pos="n") or wn.synsets(_pair_str.replace(" ", "-"), pos="n")))
                    )
                    if not _is_comp_pair and ld_target:
                        _gov_v = tok.head.head.head
                        _post_preps = [
                            c.lemma_.lower() for c in list(_gov_v.children) + list(tok.head.children)
                            if c.dep_ == "prep" and c.i > tok.head.i
                        ]
                        if _post_preps and any(
                            "verb" in str(s.get("pos", "")).lower()
                            and (
                                any(
                                    ("something" in p.lower() or "sth" in p.lower())
                                    and any(pp in re.findall(r"\b[a-z]+\b", p.lower()) for pp in _post_preps)
                                    for p in (s.get("patterns") or [])
                                )
                                or (
                                    "transitive" in str(s.get("grammar", "")).lower()
                                    and tok.head.lemma_.lower() in re.findall(r"\b[a-z]+\b", " ".join(s.get("patterns") or []) + " " + " ".join(s.get("examples") or []))
                                )
                            )
                            for s in (ld_target.get("senses") or [])
                        ):
                            return "verb"

                # Directional / spatial adverbial modifier mistagged as dobj/npadvmod NOUN after an intransitive motion verb
                # (e.g. 'walk uphill', 'went downhill', 'flew overhead', 'travelled overland', 'headed upstream'):
                if (
                    "adverb" in dict_poses
                    and tok.pos_ in ("NOUN", "PROPN", "ADJ")
                    and tok.dep_ in ("dobj", "obj", "npadvmod", "advmod")
                    and tok.head.pos_ == "VERB"
                    and tok.head.i < tok.i
                    and not any(c.dep_ in ("det", "poss", "amod", "nummod") for c in tok.children)
                    and tok.head.lemma_.lower() in (
                        "walk", "go", "run", "move", "travel", "flow", "head", "climb", "hike",
                        "march", "drive", "ride", "cycle", "roll", "slide", "drift", "swim", "fly", "work", "live"
                    )
                ):
                    return "adverb"

                # Title Case / Heading Infinitive ('How to Master Time Management', 'Steps to Build Trust'):
                # When spaCy tags a capitalized base verb immediately after 'to (PART/TO)' as PROPN/NOUN inside a compound chain
                if (
                    "verb" in dict_poses
                    and tok.i > 0
                    and doc[tok.i - 1].lower_ == "to"
                    and (doc[tok.i - 1].pos_ == "PART" or doc[tok.i - 1].tag_ == "TO")
                    and not any(c.dep_ in ("det", "poss") for c in tok.children)
                ):
                    return "verb"

                # If modifying a noun as an adjectival modifier (amod) or compound modifier (e.g. Title Case 'Take Corporate Action', 'premier hotels', 'amateur runners')
                if (tok.dep_ == "amod" or tok.dep_ == "compound") and tok.head.pos_ in ("NOUN", "PROPN"):
                    if tok.dep_ == "compound":
                        has_adj = ("adjective" in dict_poses) if dict_poses else bool(wn and (wn.synsets(clean_w, pos="a") or wn.synsets(clean_w, pos="s")))
                        has_noun = ("noun" in dict_poses) if dict_poses else bool(wn and wn.synsets(clean_w, pos="n"))
                        if has_adj and not has_noun:
                            return "adjective"
                        # If PedLex lists adjective BEFORE noun in all_poses (or primary pos is adjective, e.g. 'premier', 'amateur', 'veteran', 'ancient', 'key', 'major'),
                        # treat as adjective when modifying a common NOUN ('premier hotels', 'amateur runners') OR a Title-Case common noun mistagged as PROPN
                        # ('Why Ancient Civilizations Collapsed', 'Key Findings', 'Major Changes'), but NOT a genuine PERSON proper name ('Premier Zhou Enlai', 'Major Tom').
                        head_ld = cls.get_ldoce_entry(tok.head.lemma_.lower()) or cls.get_ldoce_entry(tok.head.text.lower())
                        head_is_common_noun = (
                            tok.head.pos_ == "NOUN"
                            or tok.head.tag_ == "NNPS"
                            or (
                                head_ld is not None
                                and tok.head.ent_type_ != "PERSON"
                                and any("noun" in str(p).lower() for p in (head_ld.get("all_poses") or [head_ld.get("pos", "")]))
                            )
                        )
                        if head_is_common_noun and has_adj and ld_target and (
                            "adj" in str((ld_target.get("all_poses") or [ld_target.get("pos", "")])[0]).lower()
                            or (
                                tok.head.tag_ in ("NNS", "NNPS")
                                and cls.get_ldoce_entry(f"{clean_w} {tok.head.lemma_.lower()}") is None
                                and any(
                                    "adj" in str(s.get("pos", "")).lower()
                                    and any(g in str(s.get("grammar", "")).lower() for g in ("only before noun", "usually before noun", "no comparative"))
                                    for s in (ld_target.get("senses") or [])
                                )
                            )
                        ):
                            return "adjective"
                        if has_noun and not has_verbal_frame and (
                            tok.pos_ in ("NOUN", "PROPN")
                            or (tok.i == 0 and tok.tag_ in ("JJ", "VB", "VBP") and not has_adj)
                            or (tok.tag_ == "VBG" and "noun" in surf_poses_set and "verb" not in surf_poses_set)
                        ):
                            return "noun"
                    else:
                        if tok.tag_ in ("VBN", "VBG", "VBD") or tok.pos_ == "VERB":
                            if has_verbal_frame and "verb" in dict_poses:
                                return "verb"
                            # Post-Parenthetical Main-Verb Rescue:
                            # When an intervening parenthetical ('—which lasted three years—', ', which lasted three years,', '(...)')
                            # separates the subject NP from an '-ed' transitive main verb ('devastated trade'), spaCy mis-attaches
                            # the finite verb as 'amod' of its bare object noun and leaves the subject NP as a verbless ROOT.
                            # Stripping the preceding parenthetical projects the main clause back to its canonical SVO dependency tree.
                            if (
                                ("verb" in dict_poses or bool(cls.get_ldoce_entry(tok.lemma_.lower())))
                                and tok.i > 1
                                and doc[tok.i - 1].text in ("—", "–", "--", ",", ")")
                            ):
                                stripped_paren = re.sub(
                                    r"(?:\s*[—–]\s*[^—–]+\s*[—–]\s*|\s*,\s*[^,]+\s*,\s*|\s*\([^)]+\)\s*)",
                                    " ",
                                    sent_text,
                                )
                                stripped_paren = re.sub(r"\s{2,}", " ", stripped_paren).strip()
                                if stripped_paren and stripped_paren != sent_text:
                                    for st in nlp(stripped_paren):
                                        if st.text.lower() == tok.text.lower() and (
                                            st.dep_ in ("ROOT", "conj", "ccomp", "advcl")
                                            or st.tag_ in ("VBD", "VBP", "VBZ")
                                            or any(c.dep_ in ("dobj", "obj", "nsubj") for c in st.children)
                                        ):
                                            return "verb"
                            has_adj = ("adjective" in dict_poses) if dict_poses else bool(wn and (wn.synsets(clean_w, pos="a") or wn.synsets(clean_w, pos="s")))
                            if has_adj:
                                return "adjective"
                            else:
                                return "verb"
                        # Material / substance noun adjunct at sentence start mistagged as ADJ (amod) by spaCy
                        # (e.g. 'Silver coins were used...', 'Plastic pollution is choking...') when PedLex primary POS is noun or the pair is a compound noun:
                        if (
                            tok.i == 0
                            and "noun" in dict_poses
                            and ld_target
                            and (
                                "noun" in str((ld_target.get("all_poses") or [ld_target.get("pos", "")])[0]).lower()
                                or cls.get_ldoce_entry(f"{clean_w} {tok.head.lemma_.lower()}") is not None
                                or clean_w in ("plastic", "silver", "gold", "iron", "steel", "stone", "glass", "paper", "wood", "cotton", "rubber")
                            )
                        ):
                            return "noun"
                        if not dict_poses or "adjective" in dict_poses:
                            return "adjective"
                        if "determiner" in dict_poses:
                            return "determiner"
                        if "noun" in dict_poses:
                            return "noun"

                # Finite Clause Predicate Invariant for ADJ/VERB Homographs
                # (e.g. 'When winds calm and skies clear...', 'Forest fires and droughts thin the canopy.'):
                # An adjective never takes a direct object ('dobj'/'obj'), nor can it act as a copula-less finite clause predicate.
                if tok.pos_ == "ADJ" and ("verb" in dict_poses or (not dict_poses and wn and wn.synsets(clean_w, pos="v"))):
                    if any(c.dep_ in ("dobj", "obj") for c in tok.children):
                        return "verb"
                    _is_clause_pred = tok.dep_ in ("ROOT", "advcl", "ccomp", "relcl") or (
                        tok.dep_ == "conj" and tok.head.dep_ in ("ROOT", "advcl", "ccomp", "relcl")
                    )
                    if _is_clause_pred and doc[-1].text in (".", "!", "?"):
                        _pred_toks = [tok, tok.head] if tok.dep_ == "conj" else [tok]
                        _has_subj = any(c.dep_ in ("nsubj", "nsubjpass") for pt in _pred_toks for c in pt.children)
                        _has_cop_aux = any(c.dep_ in ("cop", "aux", "auxpass") for pt in _pred_toks for c in pt.children)
                        if _has_subj and not _has_cop_aux:
                            return "verb"

                # Participle adjective / predicate adjective / prepositional participle check vs Passive Voice:
                if tok.tag_ in ("VBN", "VBG") or tok.pos_ == "VERB":
                    # Prepositional participles (e.g. 'Given that...', 'Given the difficulty...', 'Considering...')
                    if "preposition" in dict_poses and tok.dep_ in ("prep", "mark", "advcl") and not any(c.dep_ in ("aux", "auxpass", "nsubj", "nsubjpass") for c in tok.children):
                        if any(c.dep_ in ("pobj", "dobj", "ccomp", "mark") or c.text.lower() == "that" for c in tok.children) or (tok.i + 1 < len(doc) and doc[tok.i + 1].text.lower() == "that"):
                            return "preposition"
                if tok.tag_ == "VBN" or tok.pos_ == "VERB":
                    surf = tok.text.lower()
                    if any(c.dep_ in ("dobj", "obj", "iobj", "prt") for c in tok.children):
                        return "verb"
                    ld_entry = cls.get_ldoce_entry(surf)
                    is_surf_entry = bool(ld_entry and str(ld_entry.get("word") or "").lower() == surf)
                    surf_poses = {cls._pos_code(s.get("pos")) for s in (ld_entry.get("senses") or []) if s.get("pos")} if is_surf_entry else set()
                    adj_senses = [s for s in (ld_entry.get("senses") or []) if "adj" in str(s.get("pos") or "").lower()] if is_surf_entry else []
                    has_pred_adj = any("only before noun" not in str(s.get("grammar") or "").lower() for s in adj_senses)
                    if not has_pred_adj and is_surf_entry and not adj_senses:
                        has_pred_adj = any("adj" in str(p).lower() for p in (ld_entry.get("all_poses") or []))
                    if not has_pred_adj and not ld_entry and wn:
                        has_pred_adj = bool(wn.synsets(surf, pos="a") or wn.synsets(surf, pos="s"))

                    has_auxpass = any(c.dep_ == "auxpass" for c in tok.children) or (tok.dep_ == "auxpass")
                    has_be_gov = any(c.lemma_ == "be" for c in tok.children if c.dep_ in ("aux", "cop")) or (tok.head.lemma_ == "be")
                    if any(c.dep_ in ("ccomp", "xcomp") for c in tok.children):
                        if not (tok.tag_ == "VBN" and has_pred_adj and (has_auxpass or has_be_gov)):
                            return "verb"
                    if clean_w != surf and dict_poses and "adjective" not in dict_poses and "verb" in dict_poses:
                        return "verb"
                    if has_auxpass:
                        if has_pred_adj and surf_poses == {"adj"}:
                            has_passive_event_cue = any(
                                c.dep_ in ("agent", "npadvmod")
                                or (c.dep_ == "prep" and c.lemma_.lower() == "by")
                                or (c.dep_ in ("aux", "auxpass") and (c.lemma_.lower() != "be" or c.text.lower() == "being"))
                                or (
                                    c.dep_ == "advmod"
                                    and c.lemma_.lower() in (
                                        "finally", "eventually", "suddenly", "gradually", "quickly", "rapidly",
                                        "slowly", "immediately", "recently", "newly", "carefully", "deliberately",
                                        "accidentally", "officially", "formally", "illegally", "secretly",
                                    )
                                )
                                for c in tok.children
                            )
                            if not has_passive_event_cue:
                                return "adjective"
                        return "verb"
                    if has_be_gov:
                        if has_pred_adj:
                            if surf_poses == {"adj"} and not (
                                tok.tag_ == "VBG"
                                and any(c.dep_ == "prep" and c.lemma_.lower() in ("from", "into", "onto", "out", "through", "across", "toward", "towards") for c in tok.children)
                            ):
                                return "adjective"
                            if tok.tag_ == "VBG" and "verb" in dict_poses and tok.dep_ not in ("acomp", "oprd"):
                                has_deg_adv = any(
                                    c.dep_ == "advmod" and c.lemma_.lower() in ("very", "more", "most", "quite", "rather", "too", "so", "extremely", "highly", "truly")
                                    for c in tok.children
                                )
                                if not has_deg_adv:
                                    return "verb"
                            if dict_poses and "adjective" not in dict_poses and "verb" in dict_poses:
                                return "verb"
                            return "adjective"
                    # Post-nominal [not before noun] or preposition-bound participial adjectives in acl position
                    # (e.g. 'scientists involved in the research', 'parties concerned with public safety')
                    if (
                        tok.tag_ == "VBN"
                        and tok.dep_ == "acl"
                        and has_pred_adj
                        and adj_senses
                        and not any(c.dep_ in ("agent", "aux", "auxpass", "npadvmod") for c in tok.children)
                    ):
                        prep_lemmas = [c.lemma_.lower() for c in tok.children if c.dep_ == "prep" and c.lemma_.lower() != "by"]
                        if prep_lemmas and any(
                            any(f"{surf} {pl}" in str(pat).lower() for pat in (s.get("patterns") or []))
                            for s in adj_senses
                            for pl in prep_lemmas
                        ):
                            return "adjective"

                # Syntactic dependency repair for misparsed NOUN/PROPN (e.g. ROOT/predicate verbs like 'trap energy', 'swells', or 'to + V' infinitivals like 'income to deposit'):
                if tok.pos_ in ("NOUN", "PROPN"):
                    has_verb_entry = ("verb" in dict_poses) if dict_poses else bool(wn and (wn.synsets(clean_w, pos="v") or wn.synsets(tok.lemma_.lower(), pos="v")))
                    if has_verb_entry:
                        has_to_inf_child = any(
                            (c.tag_ == "TO" or (c.lower_ == "to" and c.pos_ == "PART"))
                            and c.dep_ in ("aux", "mark")
                            and c.i < tok.i
                            for c in tok.children
                        ) or (
                            tok.dep_ == "pobj"
                            and tok.head.lower_ == "to"
                            and (tok.head.pos_ == "PART" or tok.head.tag_ == "TO")
                            and not any(c.dep_ in ("det", "poss", "amod", "nummod") for c in tok.children)
                        )
                        has_dobj = any(c.dep_ in ("dobj", "obj") for c in tok.children)
                        has_nsubj_child = any(c.dep_ in ("nsubj", "nsubjpass") for c in tok.children)
                        if has_to_inf_child or has_dobj or has_nsubj_child:
                            return "verb"

                        # Coordinate Predicate Symmetry & Plural-Subject Base-Verb Invariant:
                        # 1. A bare plural/base verb following a plural subject NP (e.g. 'Human relationships bruise, collide, and comfort')
                        # is often mis-tagged NOUN (tag='NN', dep='ROOT') with the plural subject attached as 'compound' (tag='NNS'),
                        # and its coordinated predicates ('collide', 'comfort') also mis-tagged NOUN.
                        has_det_or_poss = any(
                            c.dep_ in ("det", "poss", "nummod") or (c.dep_ == "amod" and c.i < tok.i)
                            for c in tok.children
                        )
                        has_verb_conj_child = any(
                            c.dep_ == "conj" and (
                                c.pos_ == "VERB"
                                or "verb" in ((cls.get_ldoce_entry(c.lemma_.lower()) or {}).get("all_poses") or [])
                            )
                            for c in tok.children
                        )
                        is_conj_of_verb = (
                            tok.dep_ == "conj"
                            and not has_det_or_poss
                            and (
                                tok.head.pos_ == "VERB"
                                or (
                                    tok.head.dep_ == "ROOT"
                                    and any(hc.dep_ == "compound" and hc.tag_ in ("NNS", "NNPS") and hc.i < tok.head.i for hc in tok.head.children)
                                )
                            )
                        )
                        has_plural_compound_subj = (
                            tok.dep_ == "ROOT"
                            and tok.tag_ == "NN"
                            and not has_det_or_poss
                            and any(c.dep_ == "compound" and c.tag_ in ("NNS", "NNPS") and c.i < tok.i for c in tok.children)
                        )
                        is_followed_by_coord_verb = (
                            not has_det_or_poss
                            and tok.i > 0
                            and doc[tok.i - 1].pos_ in ("NOUN", "PROPN", "PRON")
                            and tok.i + 2 < len(doc)
                            and doc[tok.i + 1].text == ","
                            and doc[tok.i + 2].pos_ == "VERB"
                        )
                        if (has_plural_compound_subj and has_verb_conj_child) or is_conj_of_verb or is_followed_by_coord_verb:
                            return "verb"

                        raw_quote_followed = bool(
                            re.search(
                                rf"(?<!['\"“”‘’])\b{re.escape(tok.text)}\s*,?\s+[\"“‘'«`]\s*[A-Za-z]",
                                quoted_sentence or "",
                            )
                        )
                        quote_followed = raw_quote_followed
                        is_nominal_arg = (
                            tok.dep_ in ("nsubj", "nsubjpass", "csubj", "csubjpass", "dobj", "obj", "pobj", "compound", "appos", "nmod", "attr")
                            and tok.head != tok
                            and tok.head.pos_ in ("VERB", "AUX", "ADP", "NOUN", "PROPN")
                        )
                        if quote_followed and not has_det_or_poss and not is_nominal_arg and tok.tag_ != "NNS":
                            has_coord_verb = any(t.pos_ == "VERB" for t in doc if t.i != tok.i)
                            if has_coord_verb:
                                return "verb"

                        if (
                            (tok.i > 0 and doc[tok.i - 1].text == ",")
                            and not has_det_or_poss
                            and not is_nominal_arg
                            and tok.tag_ != "NNS"
                            and (quote_followed or any(c.dep_ in ("ccomp", "xcomp") and c.pos_ in ("VERB", "AUX") for c in tok.children))
                        ):
                            has_coord_action = any(t.text.lower() in ("then", "and", "next") or t.pos_ == "VERB" for t in doc[tok.i+1:])
                            if has_coord_action:
                                return "verb"

                        has_clause_finite_verb = any(
                            t.pos_ in ("VERB", "AUX")
                            and t.dep_ in ("ROOT", "conj", "advcl", "ccomp")
                            and t.i != tok.i
                            for t in doc
                        )

                        def _is_lex_compound(prev_token) -> bool:
                            cand_pairs = {
                                f"{prev_token.lemma_.lower()} {clean_w}",
                                f"{prev_token.text.lower()} {tok.text.lower()}",
                            }
                            for cp in cand_pairs:
                                ld_cp = cls.get_ldoce_entry(cp)
                                if ld_cp and str(ld_cp.get("word") or "").lower() == cp:
                                    return True
                                if wn and wn.synsets(cp.replace(" ", "_"), pos="n"):
                                    return True
                            return False

                        if tok.dep_ == "ROOT":
                            has_nsubj = any(c.dep_ in ("nsubj", "nsubjpass") for c in tok.children)
                            has_compound_subj = any(c.dep_ in ("compound", "nmod") and c.head == tok and c.i < tok.i for c in tok.children)
                            has_prep_comp = any(c.dep_ in ("prep", "advmod", "npadvmod") for c in tok.children)
                            if (has_nsubj or has_plural_compound_subj or (has_compound_subj and has_prep_comp)) and not has_clause_finite_verb:
                                return "verb"

                        # Main-Clause Coordinated Subject Predicate Invariant (e.g. 'The oceans and land collapse under extreme stress.',
                        # 'In a recession, tech stocks and gold drop rapidly.', 'Limits on water use and power drop during the rainy season.',
                        # 'Emissions from coal plants and traffic harm public health.'):
                        # When the entire sentence has 0 finite verbs, ends with sentence punctuation, and spaCy attaches a bare verb
                        # as 'conj' of a sentence-initial subject noun (or its post-nominal PP pobj) with the 2nd coordinated noun attached as 'compound' of tok.
                        _root_subj = (
                            tok.head if tok.head.dep_ == "ROOT"
                            else (
                                tok.head.head.head
                                if tok.head.dep_ == "pobj" and tok.head.head.dep_ == "prep" and tok.head.head.head.dep_ == "ROOT"
                                else None
                            )
                        )
                        if (
                            tok.dep_ == "conj"
                            and _root_subj is not None
                            and not has_clause_finite_verb
                            and not has_det_or_poss
                            and not tok.text.istitle()
                            and doc[-1].text in (".", "!", "?")
                            and tok.i >= 2
                            and doc[tok.i - 1].pos_ in ("NOUN", "PROPN")
                            and any(c.dep_ in ("compound", "nmod") and c.i == tok.i - 1 for c in tok.children)
                            and (
                                any(c.dep_ in ("prep", "advmod", "npadvmod") for c in tok.children)
                                or any(c.dep_ in ("prep", "advmod", "npadvmod") and (c.i < _root_subj.i or c.i > tok.i) for c in _root_subj.children)
                                or any(c.dep_ == "appos" and c.i > tok.i and doc[tok.i + 1].text != "," for c in _root_subj.children)
                            )
                        ):
                            if not _is_lex_compound(doc[tok.i - 1]):
                                return "verb"

                        # AcI Perception / Causative Bare-Infinitive Complement Invariant
                        # (e.g. 'Investors watched tech stocks and gold drop.', 'High tariffs made foreign exports and demand decline.'):
                        _aci_verbs = {"watch", "see", "hear", "feel", "notice", "observe", "witness", "make", "let", "have"}
                        gov_v = tok.head if tok.head.pos_ == "VERB" else (tok.head.head if tok.dep_ == "conj" and tok.head.head.pos_ == "VERB" else None)
                        if (
                            gov_v is not None
                            and gov_v.lemma_.lower() in _aci_verbs
                            and gov_v.i < tok.i - 1
                            and not has_det_or_poss
                            and not any(c.dep_ == "amod" for c in tok.children)
                            and doc[tok.i - 1].pos_ in ("NOUN", "PROPN")
                            and any(c.dep_ in ("compound", "nmod") and c.i == tok.i - 1 for c in tok.children)
                        ):
                            if not _is_lex_compound(doc[tok.i - 1]):
                                return "verb"

                        if tok.dep_ == "pobj" and tok.head.lemma_.lower() in ("until", "before", "after", "since", "as", "when", "while"):
                            has_det_child = any(c.dep_ == "det" for c in tok.children)
                            has_comp_subj = any(c.dep_ in ("compound", "amod", "poss") and c.i < tok.i for c in tok.children)
                            if has_det_child and has_comp_subj and tok.tag_ == "NNS":
                                return "verb"

                        # Constraint Grammar "Missing-Predicate Invariant" for Subordinate & Complement Clauses:
                        # Every finite clause introduced by a subordinating conjunction ('if, when, although, because, while, as, once, unless, that, whether...')
                        # must contain at least one finite verb. When spaCy mis-attaches a bare noun-verb homograph at the end of a
                        # subject NP (especially coordinated subjects like 'If the oceans and land collapse,' or 'warn that heat and pressure rise')
                        # as a compound/conj/pobj/dobj NOUN, promote it to 'verb' if the subclause lacks any other finite verb and (prev_noun + tok)
                        # is not a lexicalized compound noun in PedLex/WordNet.
                        _pure_sconjs = {
                            "if", "when", "whenever", "although", "though", "because",
                            "while", "whilst", "once", "unless", "whereas", "wherever", "whether",
                        }
                        _dual_sconjs = {"as", "since", "until", "before", "after"}
                        _clause_punct = {",", ";", ":", "—", "--", ".", "!", "?"}
                        if not has_det_or_poss and tok.i >= 2 and doc[tok.i - 1].pos_ in ("NOUN", "PROPN", "PRON"):
                            sub_idx = None
                            for j in range(tok.i - 2, -1, -1):
                                tj = doc[j]
                                if tj.text in _clause_punct:
                                    break
                                is_that_comp = (
                                    tj.lower_ == "that"
                                    and (tj.dep_ == "mark" or (j > 0 and doc[j - 1].pos_ in ("VERB", "ADJ", "AUX")))
                                )
                                if tj.lower_ in (_pure_sconjs | _dual_sconjs) or is_that_comp:
                                    if j + 1 < len(doc) and doc[j + 1].lower_ in ("of", "to", "for", "from", "with"):
                                        continue
                                    sub_idx = j
                                    break
                            if sub_idx is not None:
                                clause_end = len(doc)
                                for k_idx in range(tok.i + 1, len(doc)):
                                    tk = doc[k_idx]
                                    if tk.text in _clause_punct or tk.lower_ in (_pure_sconjs | _dual_sconjs):
                                        clause_end = k_idx
                                        break
                                sub_span = doc[sub_idx + 1 : clause_end]
                                has_other_subclause_verb = any(
                                    t.i != tok.i and (
                                        t.pos_ in ("VERB", "AUX")
                                        or t.tag_ in ("VBD", "VBZ", "VBP", "MD")
                                    )
                                    for t in sub_span
                                )
                                has_main_clause_verb = any(
                                    (t.i < sub_idx or t.i > clause_end)
                                    and (t.pos_ in ("VERB", "AUX") or t.tag_ in ("VBD", "VBZ", "VBP", "MD"))
                                    for t in doc
                                )
                                if not has_other_subclause_verb and has_main_clause_verb:
                                    sub_tok_low = doc[sub_idx].lower_
                                    pre_np_tokens = doc[sub_idx + 1 : tok.i]
                                    is_valid_subclause_frame = (
                                        sub_tok_low in (_pure_sconjs | {"that"})
                                        or any(t.lower_ in ("and", "or") or t.tag_ in ("NNS", "NNPS") for t in pre_np_tokens)
                                        or (tok.tag_ == "NNS" and any(t.dep_ in ("det", "poss") for t in pre_np_tokens))
                                    )
                                    if is_valid_subclause_frame:
                                        if not _is_lex_compound(doc[tok.i - 1]):
                                            return "verb"

                # Syntactic dependency repair for misparsed ADJ coordinated with a finite verb (e.g. 'bodies wear out and degenerate'):
                if tok.pos_ == "ADJ" and tok.dep_ == "conj" and tok.head.pos_ == "VERB":
                    has_verb_entry = ("verb" in dict_poses) if dict_poses else bool(wn and wn.synsets(clean_w, pos="v"))
                    if has_verb_entry and not any(c.dep_ == "cop" for c in tok.children):
                        return "verb"

                # Syntactic dependency repair for misparsed ADJ occupying a nominal head slot (e.g. 'more like a "lunatic"', 'in need of "lunatics"', 'the lunatic\'s'):
                if tok.pos_ == "ADJ" and ("noun" in dict_poses or (not dict_poses and wn and (wn.synsets(clean_w, pos="n") or wn.synsets(tok.lemma_.lower(), pos="n")))):
                    has_nom_child = any(c.dep_ in ("det", "poss", "nummod", "relcl", "case") or (c.dep_ in ("amod", "compound") and c.i < tok.i) for c in tok.children)
                    is_bare_pobj_head = (
                        tok.dep_ == "pobj"
                        and tok.head.pos_ == "ADP"
                        and not any(c.pos_ in ("NOUN", "PROPN") for c in tok.children)
                        and (tok.i + 1 >= len(doc) or doc[tok.i + 1].pos_ not in ("NOUN", "PROPN"))
                    )
                    if (has_nom_child and tok.dep_ in ("pobj", "dobj", "obj", "iobj", "nsubj", "nsubjpass", "attr", "oprd", "appos", "conj", "poss", "nmod", "ROOT")) or is_bare_pobj_head:
                        return "noun"

                mapped_pos = VALID_POS_MAP.get(tok.pos_, "noun")
                if dict_poses and mapped_pos not in dict_poses:
                    for pref in ("verb", "noun", "adjective", "adverb", "determiner", "preposition", "conjunction"):
                        if pref in dict_poses:
                            return pref
                return mapped_pos

        # 2. Hyphen-insensitive phrase span match (e.g. 'reschedule' in sentence with 're-schedule')
        target_no_hyphen = target_tok.replace("-", "")
        ld_target_fallback = cls.get_ldoce_entry(clean_w) or cls.get_ldoce_entry(target_no_hyphen)
        dict_poses_fb: Set[str] = set()
        if ld_target_fallback:
            for p in (ld_target_fallback.get("all_poses") or []):
                p_low = str(p).lower()
                for k, v in (("noun", "noun"), ("adv", "adverb"), ("verb", "verb"), ("adj", "adjective"), ("prep", "preposition"), ("conj", "conjunction"), ("det", "determiner")):
                    if k in p_low and not (k == "verb" and "adverb" in p_low) and not (k == "noun" and "pronoun" in p_low):
                        dict_poses_fb.add(v)
            for s in (ld_target_fallback.get("senses") or []):
                sp = str(s.get("pos") or "").lower()
                for k, v in (("noun", "noun"), ("adv", "adverb"), ("verb", "verb"), ("adj", "adjective"), ("prep", "preposition"), ("conj", "conjunction"), ("det", "determiner")):
                    if k in sp and not (k == "verb" and "adverb" in sp) and not (k == "noun" and "pronoun" in sp):
                        dict_poses_fb.add(v)
        if len(dict_poses_fb) == 1:
            return next(iter(dict_poses_fb))

        # Check regex span in sentence allowing hyphens/spaces
        hyphen_pat = r"\b" + r"[\s\-]*".join(re.escape(c) for c in target_no_hyphen) + r"\b"
        m_span = re.search(hyphen_pat, sent_text, re.IGNORECASE)
        if m_span:
            char_span = doc.char_span(m_span.start(), m_span.end(), alignment_mode="expand")
            if char_span:
                if char_span.root.head.pos_ in ("NOUN", "PROPN") and char_span.root.dep_ in ("amod", "nmod", "compound"):
                    if not dict_poses_fb or "adjective" in dict_poses_fb:
                        return "adjective"
                span_root_pos = char_span.root.pos_
                if span_root_pos in VALID_POS_MAP:
                    cand_p = VALID_POS_MAP[span_root_pos]
                    if not dict_poses_fb or cand_p in dict_poses_fb:
                        return cand_p

        # 3. Substring match on tokens (only when token shares stem >= 4 chars)
        if len(target_tok) >= 4:
            for tok in doc:
                if target_tok in tok.text.lower() or target_tok in tok.lemma_.lower():
                    if tok.dep_ == "amod" and tok.head.pos_ in ("NOUN", "PROPN"):
                        if not dict_poses_fb or "adjective" in dict_poses_fb:
                            return "adjective"
                    cand_p = VALID_POS_MAP.get(tok.pos_, "noun")
                    if not dict_poses_fb or cand_p in dict_poses_fb:
                        return cand_p

        # 4. WordNet dictionary fallback for word/headword POS
        wn = cls.get_wordnet()
        if wn:
            synsets = wn.synsets(target_tok) or (wn.synsets(target_no_hyphen) if target_no_hyphen != target_tok else [])
            if synsets:
                wn_pos = synsets[0].pos
                wn_map = {"n": "noun", "v": "verb", "a": "adjective", "s": "adjective", "r": "adverb"}
                if wn_pos in wn_map:
                    return wn_map[wn_pos]

        return "noun"

    @classmethod
    def resolve_item_pos(
        cls,
        word: str,
        quote: Optional[str] = None,
        manual_pos: Optional[str] = None
    ) -> str:
        """
        Unified Single Source of Truth for resolving the canonical part of speech of a vocabulary item.
        Harmonizes contextual syntactic evidence (quote + spaCy) with PedLex dictionary lexical truth.

        Lifecycle Entry Points:
          1. Extraction stage (before writing to markdown)
          2. Quiz skeletons builder (runtime calibration, without touching markdown files)
        """
        raw_word = str(word or "").strip()
        w_clean = re.sub(r"\[.*?\]|\(.*?\)", "", raw_word).strip().lower()
        if not w_clean:
            return "noun"

        # 1. Multi-word Expressions Gate (preserve slot-bearing formulas like 'reverse [oneself]' or 'name [sb] [sth]')
        m_pos_low = str(manual_pos or "").strip().lower()
        if (
            " " in w_clean
            or cls.is_multiword_expression(raw_word)
            or cls.is_multiword_expression(w_clean)
        ):
            return cls.classify_expression_type(raw_word, quote or "") or (m_pos_low if m_pos_low in cls.EXPRESSION_TYPE_LABELS else "collocation")

        # 2. Contextual Evidence Gate (Quote + spaCy)
        contextual_pos = None
        if quote and str(quote).strip():
            contextual_pos = cls.determine_contextual_pos(w_clean, str(quote).strip())

        # Normalize manual_pos if provided
        norm_manual = None
        if manual_pos and str(manual_pos).strip():
            m_lower = str(manual_pos).strip().lower()
            if "adv" in m_lower or m_lower.startswith("r"):
                norm_manual = "adverb"
            elif "verb" in m_lower or m_lower.startswith("v"):
                norm_manual = "verb"
            elif "adj" in m_lower or m_lower.startswith("a"):
                norm_manual = "adjective"
            elif "prep" in m_lower or m_lower.startswith("p"):
                norm_manual = "preposition"
            elif "conj" in m_lower or m_lower.startswith("c"):
                norm_manual = "conjunction"
            elif "noun" in m_lower or m_lower.startswith("n"):
                norm_manual = "noun"

        ld_entry = cls.get_ldoce_entry(w_clean)
        ld_poses = []
        if ld_entry:
            ld_poses = [p.lower() for p in ld_entry.get("all_poses", [ld_entry.get("pos", "")])]

        # 2b. Lexicon POS Hard Gate:
        # If the headword in PedLex has entries and strictly NO verb entry (e.g. 'emblem', 'climate', 'table'),
        # it can never be classified as a verb regardless of spaCy misparses or manual hints.
        if ld_poses and not any("verb" in p and "adverb" not in p for p in ld_poses):
            wn = cls.get_wordnet()
            if not wn or not wn.synsets(w_clean, pos="v"):
                if contextual_pos in ("adjective", "adverb", "preposition", "conjunction") and any(contextual_pos[:4] in p or contextual_pos[:3] in p for p in ld_poses):
                    return contextual_pos
                if norm_manual in ("adjective", "adverb", "preposition", "conjunction") and any(norm_manual[:4] in p or norm_manual[:3] in p for p in ld_poses):
                    return norm_manual
                if any("noun" in p for p in ld_poses):
                    return "noun"
                if any("adj" in p for p in ld_poses):
                    return "adjective"
                if any("prep" in p for p in ld_poses):
                    return "preposition"
                if any("conj" in p for p in ld_poses):
                    return "conjunction"
                if any("adv" in p for p in ld_poses):
                    return "adverb"
            elif norm_manual == "verb":
                return "verb"

        # Contextual Predicate / Finite Verb Primacy:
        # If sentence context clearly marks the word as a predicate/finite verb (e.g. 'You may shrug your shoulders')
        # and PedLex confirms verb entry, context strictly overrides manual misclassification
        if contextual_pos == "verb":
            wn = cls.get_wordnet()
            if not ld_poses or any("verb" in p and "adverb" not in p for p in ld_poses) or (wn and wn.synsets(w_clean, pos="v")):
                return "verb"

        # If context indicates an adjective, adverb, preposition, or conjunction and PedLex confirms
        if contextual_pos in ("adjective", "adverb", "preposition", "conjunction"):
            prefix = "adj" if contextual_pos == "adjective" else ("adv" if contextual_pos == "adverb" else contextual_pos[:4])
            has_pos = not ld_poses or any(prefix in p for p in ld_poses)
            if has_pos:
                return contextual_pos

        # If manual_pos is supported by PedLex and context does not strongly contradict
        if norm_manual:
            prefix = "adj" if norm_manual == "adjective" else ("adv" if norm_manual == "adverb" else norm_manual[:4])
            if not ld_poses or any(prefix in p for p in ld_poses):
                return norm_manual

        # Fallback to contextual_pos if available
        if contextual_pos:
            return contextual_pos

        # Fallback to PedLex primary pos
        if ld_poses:
            first_p = ld_poses[0]
            if "adv" in first_p:
                return "adverb"
            elif "verb" in first_p:
                return "verb"
            elif "adj" in first_p:
                return "adjective"
            elif "prep" in first_p:
                return "preposition"
            elif "conj" in first_p:
                return "conjunction"

        return "noun"

