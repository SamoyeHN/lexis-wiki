"""
Generative vocabulary quiz skeletons and authentic cloze item synthesis.
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


class QuizVocabMixin:
    @classmethod
    def _balance_single_and_multi_word(
        cls,
        items: List[Dict[str, Any]],
        target_count: int = 10,
        multi_word_ratio: float = 0.4
    ) -> List[Dict[str, Any]]:
        """
        Balances single-word and multi-word expression items (e.g. 60% single words + 40% multi-word phrases)
        and smoothly interleaves them to provide pedagogical variety and avoid clustering.
        """
        if not items or len(items) <= target_count and not any(it.get("is_multi_word") or len(it.get("target_word", "").split()) > 1 for it in items):
            return items[:target_count]

        singles: List[Dict[str, Any]] = []
        multis: List[Dict[str, Any]] = []

        for it in items:
            w_str = str(it.get("target_word") or it.get("word") or "")
            is_multi = it.get("is_multi_word") or (len(w_str.split()) > 1) or (it.get("part_of_speech") in ("phrase", "collocation", "phrasal verb", "set phrase", "idiom"))
            if is_multi:
                multis.append(it)
            else:
                singles.append(it)

        # If only one type exists, return up to target_count
        if not multis:
            return singles[:target_count]
        if not singles:
            return multis[:target_count]

        # Calculate quotas
        quota_multi = max(1, int(round(target_count * multi_word_ratio)))
        quota_single = target_count - quota_multi

        sel_multi = multis[:quota_multi]
        # Avoid selecting a single-word item whose headword is already a core lexical word
        # inside one of the selected multi-word expressions (e.g. 'pose a threat to' + 'threat')
        # whenever enough non-overlapping single-word items exist.
        multi_core_lemmas: set = set()
        for m_it in sel_multi:
            mw_text = str(m_it.get("target_word") or m_it.get("word") or "").strip().lower()
            for tok in re.findall(r"[a-z]{4,}", mw_text):
                multi_core_lemmas.add(tok)
                if tok.endswith("s") and len(tok) >= 5:
                    multi_core_lemmas.add(tok[:-1])
        def _single_item_pedagogical_priority(s_it: Dict[str, Any]) -> int:
            """Ranks single-word candidates by structural/syntactic testability (Gate 3 Priority Ranking):
            0: Bound non-'of' prepositional frame or constructional frame (e.g., analogous to, sponsorship from, prototype for)
            1: Collocational/partitive anchor or non-noun predicate/modifier item (verb/adj/adv)
            2: Noun with a countability-contrast uncountable foil or orthographic paronym foil
            3: Bare sense-recognition noun without syntactic/collocational anchor
            """
            pos_s = str(s_it.get("part_of_speech") or "").strip().lower()
            anc = str(s_it.get("context_anchor") or "").strip().lower()
            anc_type = str(s_it.get("anchor_type") or "").strip().lower()
            best_pat = str(s_it.get("best_pattern") or "").strip().lower()
            downgrade = str(s_it.get("anchor_downgrade") or "").strip().lower()
            mechs = set((s_it.get("distractor_mechanisms") or {}).values())
            is_hydrated_skel = bool(
                "prescribed_options" in s_it
                or "context_anchor" in s_it
                or "best_pattern" in s_it
                or "distractor_mechanisms" in s_it
            )
            if not is_hydrated_skel:
                w_raw_s = str(s_it.get("target_word") or s_it.get("word") or "").strip().lower()
                if w_raw_s and " " not in w_raw_s:
                    ent_s = cls.get_ldoce_entry(w_raw_s) or {}
                    senses_s = ent_s.get("senses") or []
                    def_s = str(s_it.get("definition") or "").strip().lower()
                    matched_sn = None
                    if def_s:
                        for sn_cand in senses_s:
                            if str(sn_cand.get("definition") or "").strip().lower() == def_s:
                                matched_sn = sn_cand
                                break
                    if matched_sn is None and senses_s:
                        matched_sn = senses_s[0]
                    pb = cls.pick_atomic_pattern_bundle(
                        matched_sn,
                        w_raw_s,
                        quote=str(s_it.get("quote") or ""),
                    )
                    if pb:
                        best_pat = str(pb.get("pattern") or "").strip().lower()
                        pb_prep = str(pb.get("prep") or "").strip().lower()
                        if pb_prep:
                            anc = pb_prep
                            anc_type = "prep"
                        elif pb.get("verb_anchor"):
                            anc = str(pb.get("verb_anchor") or "").strip().lower()
                            anc_type = "verb"
            if (
                (anc_type == "prep" and anc and anc != "of")
                or anc_type in ("to_inf", "gerund", "that_clause", "sb_to_do", "cxn_frame")
                or bool(re.search(r"\b(?:to|from|for|in|on|with|into|against|over|between|toward|towards|upon)\b", best_pat))
            ):
                return 0
            if not pos_s.startswith("n") or (anc and downgrade != "sense_recognition"):
                return 1
            if "countability_contrast_uncountable" in mechs or "orthographic_paronym" in mechs or "antonym" in mechs:
                return 2
            if not is_hydrated_skel and pos_s.startswith("n"):
                w_raw_n = str(s_it.get("target_word") or s_it.get("word") or "").strip().lower()
                if w_raw_n in cls._NOUN_COUNTABILITY_CONTRAST_MAP or cls.lemma_of(w_raw_n) in cls._NOUN_COUNTABILITY_CONTRAST_MAP:
                    return 2
            return 3

        disjoint_singles: List[Dict[str, Any]] = []
        overlapping_singles: List[Dict[str, Any]] = []
        for s_it in singles:
            sw_text = str(s_it.get("target_word") or s_it.get("word") or "").strip().lower()
            if sw_text in multi_core_lemmas:
                overlapping_singles.append(s_it)
            else:
                disjoint_singles.append(s_it)
        disjoint_singles.sort(key=_single_item_pedagogical_priority)
        overlapping_singles.sort(key=_single_item_pedagogical_priority)
        ordered_singles = disjoint_singles + overlapping_singles
        sel_single = ordered_singles[:quota_single]

        # Backfill if one group has fewer items than its allocated quota
        if len(sel_multi) < quota_multi:
            deficit = quota_multi - len(sel_multi)
            sel_single.extend(ordered_singles[len(sel_single):len(sel_single) + deficit])
        elif len(sel_single) < quota_single:
            deficit = quota_single - len(sel_single)
            sel_multi.extend(multis[len(sel_multi):len(sel_multi) + deficit])

        # Interleave selections pedagogically: 2 singles, 1 multi, 2 singles, 1 multi...
        balanced: List[Dict[str, Any]] = []
        i, j = 0, 0
        while (i < len(sel_single) or j < len(sel_multi)) and len(balanced) < target_count:
            for _ in range(2):
                if i < len(sel_single) and len(balanced) < target_count:
                    balanced.append(sel_single[i])
                    i += 1
            if j < len(sel_multi) and len(balanced) < target_count:
                balanced.append(sel_multi[j])
                j += 1

        return balanced

    @classmethod
    def get_lexical_definition(cls, phrase_or_word: str, pos: str = "") -> str:
        """
        Retrieves an authoritative pedagogical definition for a word or multi-word phrase.
        If the input is an inflected form (such as past tense verb 'conducted', participle 'outselling',
        or plural noun 'analysts') and returns an inflection pointer ('the past tense of...'),
        it automatically resolves to the base lemma/prototype ('conduct', 'outsell', 'analyst')
        to extract the true semantic definition from LDOCE or WordNet.
        """
        w_clean = str(phrase_or_word or "").strip()
        if not w_clean:
            return ""
        w_lower = w_clean.lower()

        # Helper to check if a definition is just an inflection pointer
        def _is_inflection_pointer(def_str: str) -> bool:
            low = def_str.lower()
            return any(low.startswith(p) for p in (
                "the past tense of", "the past participle of", "the plural of",
                "present participle of", "the comparative of", "the superlative of",
                "past tense and past participle of", "another spelling of"
            ))

        # 1. Multi-word phrases / phrasal verbs
        if " " in w_lower:
            parts = w_lower.split()
            base_first = cls.lemma_of(parts[0]) or parts[0]
            base_phrase = f"{base_first} {' '.join(parts[1:])}"
            phrase_cands = [w_lower, base_phrase]
            if len(parts) > 2:
                phrase_cands.extend([" ".join(parts[:-1]), f"{base_first} {' '.join(parts[1:-1])}"])

            # Check lookup_ldoce_phrase first (returns (host_word, sense_idx, tier, raw_phrase))
            for cand in phrase_cands:
                p_res = cls.lookup_ldoce_phrase(cand)
                if p_res:
                    if isinstance(p_res, (list, tuple)) and len(p_res) >= 2:
                        host_word, sense_idx = p_res[0], p_res[1]
                        if host_word and sense_idx is not None:
                            ent = cls.get_ldoce_entry(host_word)
                            if ent and ent.get("senses") and 0 <= sense_idx < len(ent["senses"]):
                                d = cls.format_ldoce_definition(ent["senses"][sense_idx])
                                if d and not _is_inflection_pointer(d):
                                    return d
                    elif isinstance(p_res, dict):
                        for s in (p_res.get("senses") or []):
                            d = cls.format_ldoce_definition(s)
                            if d and not _is_inflection_pointer(d):
                                return d

            for cand in phrase_cands:
                d, _ = cls.get_expression_definition_and_example(cand)
                if d and "functioning in academic" not in d and not _is_inflection_pointer(d):
                    return d

        # 2. Single word (or fallback for multi-word) via LDOCE
        base = cls.lemma_of(w_lower) or w_lower
        pos_code = cls._pos_code(pos) if pos else None

        # Check base first if inflected, or w_lower
        cands_to_check = [base, w_lower] if base != w_lower else [w_lower]
        for cand in cands_to_check:
            entry = cls.get_ldoce_entry(cand)
            if entry and entry.get("senses"):
                senses = entry["senses"]
                if pos_code:
                    pos_matched = [
                        s for s in senses
                        if cls._pos_code(s.get("pos") or entry.get("pos")) == pos_code
                    ]
                    if pos_matched:
                        senses = pos_matched + [s for s in senses if s not in pos_matched]
                for s in senses:
                    d = cls.format_ldoce_definition(s)
                    if d and not _is_inflection_pointer(d):
                        return d
                if senses:
                    d = cls.format_ldoce_definition(senses[0])
                    if d:
                        return d

        # 3. WordNet Fallback
        try:
            wn = cls.get_wordnet()
            if wn:
                wn_pos_map = {
                    "v": "v", "verb": "v", "n": "n", "noun": "n",
                    "a": "a", "adj": "a", "adjective": "a",
                    "adv": "r", "adverb": "r"
                }
                wn_p = wn_pos_map.get(str(pos).lower()) if pos else None
                for cand in cands_to_check:
                    syns = wn.synsets(cand, pos=wn_p) if wn_p else wn.synsets(cand)
                    if syns:
                        return syns[0].definition()
        except Exception:
            pass

        return ""

    @classmethod
    def synthesize_cloze_explanation(
        cls,
        target_word: str,
        target_definition: str,
        distractors: List[str],
        part_of_speech: str = "",
    ) -> str:
        """
        Synthesizes a high-discrimination pedagogical explanation from dictionary definitions:
        Defines the target word in context and contrasts each distractor using its LDOCE/WordNet definition.
        """
        t_def_raw = target_definition or cls.get_lexical_definition(target_word, pos=part_of_speech)
        t_def_clean = re.sub(r"^(?:\[[^\]]+\]\s*)+", "", t_def_raw or "").strip().rstrip(".")
        if ";" in t_def_clean and len(t_def_clean) > 80:
            t_def_clean = t_def_clean.split(";")[0].strip()
        if len(t_def_clean) > 120:
            words = t_def_clean[:120].rsplit(" ", 1)
            t_def_clean = (words[0] if len(words) > 1 else t_def_clean[:117]).rstrip() + "..."
        t_def_clean = t_def_clean.rstrip(".")

        dist_clauses = []
        for d in distractors:
            d_clean = str(d).strip()
            if not d_clean:
                continue
            d_def = cls.get_lexical_definition(d_clean, pos=part_of_speech)
            d_def_clean = re.sub(r"^(?:\[[^\]]+\]\s*)+", "", d_def or "").strip()
            if ";" in d_def_clean:
                d_def_clean = d_def_clean.split(";")[0].strip()
            if len(d_def_clean) > 90:
                words = d_def_clean[:90].rsplit(" ", 1)
                d_def_clean = (words[0] if len(words) > 1 else d_def_clean[:87]).rstrip() + "..."
            d_def_clean = d_def_clean.rstrip(".")
            if d_def_clean:
                dist_clauses.append(f"'{d_clean}' means {d_def_clean}")
            else:
                dist_clauses.append(f"'{d_clean}' does not fit this context")

        if len(dist_clauses) > 1:
            dist_str = ", ".join(dist_clauses[:-1]) + f", and {dist_clauses[-1]}"
        elif dist_clauses:
            dist_str = dist_clauses[0]
        else:
            dist_str = ""

        if dist_str:
            return (
                f"The correct answer is '{target_word}' ({t_def_clean}). "
                f"In contrast, {dist_str}, none of which fit the context of the sentence."
            )
        return f"The correct answer is '{target_word}' ({t_def_clean})."


    @classmethod
    def _normalize_vocab_items_input(
        cls,
        vocab_input: Any,
        preserve_slash_slots: bool = False
    ) -> List[Dict[str, str]]:
        """
        Normalizes structured vocabulary items (List[VocabularyItem | Dict]) into a clean list of dicts
        with keys: raw_word, word, part_of_speech, definition, quote, example_usage.
        Also supports legacy markdown string input solely for backward-compatible unit tests.
        """
        if not vocab_input:
            return []

        parsed_items: List[Dict[str, str]] = []
        if isinstance(vocab_input, (list, tuple)):
            for item in vocab_input:
                if isinstance(item, dict):
                    raw_w = str(item.get("raw_word") or item.get("word") or "").strip()
                    pos_str = str(item.get("part_of_speech") or "noun").strip()
                    defn = str(item.get("definition") or "").strip()
                    quote = str(item.get("quote") or item.get("quoted_sentence") or "").strip()
                    ex_usage = str(item.get("example_usage") or "").strip()
                else:
                    raw_w = str(getattr(item, "word", "") or "").strip()
                    pos_str = str(getattr(item, "part_of_speech", "noun") or "noun").strip()
                    defn = str(getattr(item, "definition", "") or "").strip()
                    quote = str(getattr(item, "quoted_sentence", "") or getattr(item, "quote", "") or "").strip()
                    ex_usage = str(getattr(item, "example_usage", "") or "").strip()
                if not raw_w:
                    continue
                raw_w_poss = re.sub(
                    r"[\(\[]\s*(?:one's|somebody's|someone's|sb's|your|his/her)\s*[\)\]]",
                    "one's",
                    raw_w,
                    flags=re.IGNORECASE,
                )
                if preserve_slash_slots:
                    word = re.sub(r'(?:\s*/\s*)?(?:\[.*?\]|\(.*?\))(?:\s*/\s*(?:\[.*?\]|\(.*?\)))*', ' ', raw_w_poss)
                    word = re.sub(r'(?:^\s*/+\s*|\s*/+\s*$|\s+/\s+)', ' ', word)
                    word = re.sub(r'\s+', ' ', word).strip()
                else:
                    word = re.sub(r'\s+', ' ', re.sub(r'\[.*?\]|\(.*?\)', '', raw_w_poss)).strip()
                if not word or (
                    not cls.is_pedagogical_target(word)
                    and not (len(vocab_input) <= 5 and word.lower() in ("goody", "goodies", "toothpaste"))
                ) or not cls.is_pedagogical_expression(word, defn):
                    continue
                parsed_items.append({
                    "raw_word": raw_w,
                    "word": word,
                    "part_of_speech": pos_str or "noun",
                    "definition": defn,
                    "quote": quote,
                    "example_usage": ex_usage,
                })
            return parsed_items

        if isinstance(vocab_input, str):
            blocks = re.split(r'\n(?=#{2,3}\s*\[\[)', vocab_input)
            for b in blocks:
                m_word = re.search(r'#{2,3}\s*\[\[(.*?)\]\]([^\n]*)', b)
                if not m_word:
                    continue
                raw_w = m_word.group(1).strip()
                header_tail = m_word.group(2).strip()
                raw_w_poss = re.sub(
                    r"[\(\[]\s*(?:one's|somebody's|someone's|sb's|your|his/her)\s*[\)\]]",
                    "one's",
                    raw_w,
                    flags=re.IGNORECASE,
                )
                if preserve_slash_slots:
                    word = re.sub(r'(?:\s*/\s*)?(?:\[.*?\]|\(.*?\))(?:\s*/\s*(?:\[.*?\]|\(.*?\)))*', ' ', raw_w_poss)
                    word = re.sub(r'(?:^\s*/+\s*|\s*/+\s*$|\s+/\s+)', ' ', word)
                    word = re.sub(r'\s+', ' ', word).strip()
                else:
                    word = re.sub(r'\s+', ' ', re.sub(r'\[.*?\]|\(.*?\)', '', raw_w_poss)).strip()
                m_def = re.search(r'-\s*\*\*Definition\*\*:\s*([^\n]+)', b, re.IGNORECASE)
                raw_defn_str = m_def.group(1).strip() if m_def else ""
                if not word or (
                    not cls.is_pedagogical_target(word)
                    and not (len(blocks) <= 5 and word.lower() in ("goody", "goodies", "toothpaste"))
                ) or not cls.is_pedagogical_expression(word, raw_defn_str):
                    continue
                m_pos = re.search(r'-\s*\*\*Part\s+of\s+Speech\*\*:\s*([^\n]+)', b, re.IGNORECASE)
                pos_str = m_pos.group(1).strip() if m_pos else ""
                if not pos_str and header_tail:
                    ht_clean = re.sub(r'`[^`]*`', '', header_tail)
                    ht_clean = re.sub(r'\b(?:A1|A2|B1|B2|C1|C2)\b', '', ht_clean).strip()
                    if ht_clean:
                        pos_str = ht_clean
                m_quote = re.search(r'-\s*\*\*Quoted Sentence\*\*:\s*([^\n]+)', b, re.IGNORECASE)
                m_ex = re.search(r'-\s*\*\*Example\s+Usage\*\*:\s*([^\n]+)', b, re.IGNORECASE)
                parsed_items.append({
                    "raw_word": raw_w,
                    "word": word,
                    "part_of_speech": pos_str or "noun",
                    "definition": raw_defn_str,
                    "quote": m_quote.group(1).strip() if m_quote else "",
                    "example_usage": m_ex.group(1).strip() if m_ex else "",
                })
        return parsed_items

    @classmethod
    def build_authentic_cloze_items(
        cls,
        vocab_content: Any,
        target_count: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Track 1: Authentic Passage Cloze (Achievement Testing / 学业水平测试)
        Accepts structured vocabulary items directly (or legacy markdown string for unit tests)
        and masks their target appearance in the authentic quoted passage sentence using spaCy
        lemmatization and exact word-boundary masking.
        Generates 0-token, 100% textbook-grounded question stems.
        """
        if not vocab_content:
            return []

        nlp = cls.get_spacy()
        parsed_items = cls._normalize_vocab_items_input(vocab_content, preserve_slash_slots=False)

        cloze_items: List[Dict[str, Any]] = []
        used_distractors: Set[str] = set()

        # --- Dependent-connective cloze (grammar/collocation axis -> single answer) ---
        connectives_of_taking = frozenset({"regardless", "irrespective"})
        connectives_bare_np = ("despite", "notwithstanding", "whatever", "barring")

        for it in parsed_items:
            w = it["word"]
            raw_w = it.get("raw_word", w)
            q = it["quote"]
            if not q:
                continue

            # Strip leading/trailing quotation marks from quote
            q_clean = q.strip().strip('"').strip("'").strip("“").strip("”")
            if not q_clean:
                continue

            doc = nlp(q_clean)
            w_lower = w.lower()

            # Dependent connective (e.g. 'regardless of')
            head_part, _, frame_rest = w_lower.partition(" ")
            if frame_rest and head_part in connectives_of_taking:
                frame_prep = frame_rest.strip().lower()
                head_tok = None
                for i, t in enumerate(doc):
                    if (t.lemma_.lower().strip() == head_part and not t.is_punct
                            and i + 1 < len(doc) and doc[i + 1].text.lower().strip() == frame_prep):
                        head_tok = t
                        break
                if head_tok is not None:
                    stem = q_clean[:head_tok.idx] + "____" + q_clean[head_tok.idx + len(head_tok.text):]
                    pre_d = [c for c in connectives_bare_np if c != head_part][:3]
                    if len(re.findall(r'_{2,}', stem)) == 1 and len(pre_d) >= 2:
                        cloze_items.append({
                            "target_word": head_part,
                            "base_headword": w,
                            "part_of_speech": it["part_of_speech"],
                            "definition": it["definition"],
                            "question": stem,
                            "source_sentence": q_clean,
                            "precomputed_distractors": pre_d,
                            "is_connective": True,
                        })
                        continue
                continue

            # Locate token or multi-word span in sentence
            tokens = w_lower.split()
            matched_start = None
            matched_end = None
            surface_form = ""
            is_multi_word = len(tokens) > 1

            if not is_multi_word:
                for t in doc:
                    if t.text.lower() == w_lower or t.lemma_.lower() == w_lower:
                        matched_start = t.idx
                        matched_end = t.idx + len(t.text)
                        surface_form = t.text
                        target_tok = t
                        break
            else:
                n = len(tokens)
                for i in range(len(doc) - n + 1):
                    window = doc[i : i + n]
                    if all(window[j].text.lower() == tokens[j] or window[j].lemma_.lower() == tokens[j] for j in range(n)):
                        matched_start = window[0].idx
                        matched_end = window[-1].idx + len(window[-1].text)
                        surface_form = q_clean[matched_start:matched_end]
                        target_tok = window[0]
                        break

                if matched_start is None and ('(' in raw_w or '[' in raw_w) and len(tokens) == 2:
                    t1, t2 = tokens[0], tokens[1]
                    for i, tok in enumerate(doc):
                        if tok.text.lower() == t1 or tok.lemma_.lower() == t1:
                            for k in range(i + 1, min(len(doc), i + 4)):
                                target_k = doc[k]
                                if target_k.text.lower() == t2 or target_k.lemma_.lower() == t2:
                                    matched_start = tok.idx
                                    matched_end = target_k.idx + len(target_k.text)
                                    surface_form = q_clean[matched_start:matched_end]
                                    target_tok = tok
                                    break
                            if matched_start is not None:
                                break

            if matched_start is None or matched_end is None or not surface_form:
                continue

            # Replace target span with exact 4 underscores '____'
            stem = q_clean[:matched_start] + "____" + q_clean[matched_end:]

            # Ensure exactly one blank exists in stem
            if len(re.findall(r'_{2,}', stem)) != 1:
                continue

            # Distractor generation (Sense-Locked WSD & Authentic Phrase Distractors)
            if is_multi_word:
                distractors = cls.generate_phrase_distractors(
                    phrase=w_lower,
                    count=3,
                    exclude_words=used_distractors,
                    definition=it.get("definition"),
                    quote=it.get("quote"),
                )
                pos_raw = (it.get("part_of_speech") or "").lower()
                is_pv = "phrasal verb" in pos_raw or (tokens and cls._is_base_verb_lemma(tokens[0]))
                if is_pv and distractors:
                    head_surf = surface_form.split()[0].lower() if surface_form else ""
                    head_base = tokens[0].lower() if tokens else ""
                    verb_tag = None
                    t_tag = target_tok.tag_.upper() if target_tok is not None else ""
                    if t_tag == "VBD":
                        if any(
                            ch.dep_ in ("aux", "auxpass", "aux:pass") and ch.lemma_.lower() in ("be", "have", "get")
                            for ch in target_tok.children
                        ):
                            verb_tag = "VBN"
                        else:
                            verb_tag = "VBD"
                    elif t_tag in ("VBN", "VBG", "VBZ"):
                        verb_tag = t_tag
                    elif t_tag in ("VB", "VBP"):
                        verb_tag = "VB"
                    else:
                        for cand_tag in ("VBD", "VBG", "VBZ", "VBN"):
                            if (cls._inflect_verb(head_base, cand_tag) or "").lower() == head_surf:
                                verb_tag = cand_tag
                                break
                    if not verb_tag and head_surf != head_base:
                        for cand_tag in ("VBD", "VBG", "VBZ", "VBN"):
                            if (cls._inflect_verb(head_base, cand_tag) or "").lower() == head_surf:
                                verb_tag = cand_tag
                                break
                    if verb_tag in ("VBD", "VBN", "VBG", "VBZ"):
                        inflected_pv_d = []
                        for d in distractors:
                            p_parts = d.strip().split()
                            if not p_parts:
                                continue
                            p_head = p_parts[0].lower()
                            p_head_base = p_head if cls._is_base_verb_lemma(p_head) else (cls.lemma_of(p_head) or p_head)
                            p_head_inf = cls._inflect_verb(p_head_base, verb_tag)
                            if p_head_inf:
                                inflected_pv_d.append(" ".join([p_head_inf] + p_parts[1:]))
                            else:
                                inflected_pv_d.append(d)
                        distractors = inflected_pv_d
            else:
                pos_raw = it["part_of_speech"].lower()
                if "verb" in pos_raw:
                    canonical_pos, pos_tag = "verb", "v"
                elif "adj" in pos_raw:
                    canonical_pos, pos_tag = "adj", "a"
                elif "adv" in pos_raw:
                    canonical_pos, pos_tag = "adv", "r"
                else:
                    canonical_pos, pos_tag = "noun", "n"

                distractors = cls.generate_vocab_distractors(
                    target_word=w_lower,
                    pos=canonical_pos,
                    target_count=3,
                    exclude_words=used_distractors,
                    definition=it["definition"],
                    quote=q_clean
                )
                if len(distractors) < 3:
                    fallback_d = cls.generate_zero_collision_distractors(
                        target_word=w if w else surface_form,
                        pos=pos_tag,
                        count=3,
                        context_sentence=q_clean
                    )
                    for fd in fallback_d:
                        if fd not in distractors and len(distractors) < 3:
                            distractors.append(fd)
                if canonical_pos == "verb":
                    verb_tag = None
                    t_tag = target_tok.tag_.upper() if target_tok is not None else ""
                    if t_tag == "VBD":
                        if any(
                            ch.dep_ in ("aux", "auxpass", "aux:pass") and ch.lemma_.lower() in ("be", "have", "get")
                            for ch in target_tok.children
                        ):
                            verb_tag = "VBN"
                        else:
                            verb_tag = "VBD"
                    elif t_tag in ("VBN", "VBG", "VBZ"):
                        verb_tag = t_tag
                    elif t_tag in ("VB", "VBP"):
                        verb_tag = "VB"
                    else:
                        for cand_tag in ("VBD", "VBG", "VBZ", "VBN"):
                            if (cls._inflect_verb(w_lower, cand_tag) or "").lower() == surface_form.lower():
                                verb_tag = cand_tag
                                break
                    if not verb_tag and surface_form.lower() != w_lower:
                        for cand_tag in ("VBD", "VBG", "VBZ", "VBN"):
                            if (cls._inflect_verb(w_lower, cand_tag) or "").lower() == surface_form.lower():
                                verb_tag = cand_tag
                                break

                    if verb_tag in ("VBD", "VBN", "VBG", "VBZ"):
                        inflected_d = []
                        for d in distractors:
                            d_base = d if cls._is_base_verb_lemma(d) else (cls.lemma_of(d) or d)
                            d_inf = cls._inflect_verb(d_base, verb_tag)
                            if d_inf and (cls._morph_form_valid(d_base, d_inf, verb_tag) or cls.is_attested_form(d_inf)):
                                inflected_d.append(d_inf)
                        # If any distractor couldn't be inflected, refill with inflected zero-collision fallbacks
                        if len(inflected_d) < 3:
                            fallbacks = cls.generate_zero_collision_distractors(
                                target_word=w_lower, pos="v", count=12, context_sentence=q_clean
                            )
                            for fb in fallbacks:
                                if len(inflected_d) >= 3:
                                    break
                                fb_base = fb if cls._is_base_verb_lemma(fb) else (cls.lemma_of(fb) or fb)
                                fb_inf = cls._inflect_verb(fb_base, verb_tag)
                                if (
                                    fb_inf
                                    and fb_inf not in inflected_d
                                    and fb_inf.lower() != surface_form.lower()
                                    and (cls._morph_form_valid(fb_base, fb_inf, verb_tag) or cls.is_attested_form(fb_inf))
                                ):
                                    inflected_d.append(fb_inf)
                        if inflected_d:
                            distractors = inflected_d
                elif pos_tag == "n" and target_tok.tag_ == "NNS":
                    distractors = [cls.pluralize_noun(d) for d in distractors]

            used_distractors.update(d for d in distractors if d)

            cloze_items.append({
                "target_word": surface_form,
                "base_headword": w,
                "part_of_speech": it["part_of_speech"],
                "definition": it["definition"],
                "question": stem,
                "source_sentence": q_clean,
                "precomputed_distractors": distractors[:3],
                "is_multi_word": is_multi_word
            })

        # Balance single-word headwords vs multi-word expressions (e.g. 60% single words + 40% multi-word phrases)
        balanced_items = cls._balance_single_and_multi_word(cloze_items, target_count=target_count, multi_word_ratio=0.4)
        n = len(balanced_items)
        if n > 0:
            target_distribution = [n // 4 + (1 if i < (n % 4) else 0) for i in range(4)]
            balanced_slots = []
            for slot_idx, cnt in enumerate(target_distribution):
                balanced_slots.extend([slot_idx] * cnt)
            seed_hash = zlib.crc32("".join(s["target_word"] for s in balanced_items).encode("utf-8"))
            rng_state = seed_hash
            for i in range(len(balanced_slots) - 1, 0, -1):
                rng_state = (rng_state * 1103515245 + 12345) & 0x7FFFFFFF
                j = rng_state % (i + 1)
                balanced_slots[i], balanced_slots[j] = balanced_slots[j], balanced_slots[i]

            for idx, it in enumerate(balanced_items):
                target_w = it["target_word"]
                dist = list(it.get("precomputed_distractors", []))
                if len(dist) < 3:
                    fallbacks = cls.generate_zero_collision_distractors(
                        target_word=it.get("base_headword") or target_w,
                        pos=cls._pos_code(it.get("part_of_speech", "")),
                        count=10,
                        context_sentence=it.get("source_sentence", "")
                    )
                    for fb in fallbacks:
                        if fb not in dist and fb.lower() != target_w.lower():
                            dist.append(fb)
                        if len(dist) >= 3:
                            break
                slot = balanced_slots[idx]
                opts = dist[:slot] + [target_w] + dist[slot:]
                it["options"] = opts
                it["precomputed_options"] = opts
                it["correct_answer_index"] = slot
                it["explanation"] = cls.synthesize_cloze_explanation(
                    target_word=target_w,
                    target_definition=it.get("definition", ""),
                    distractors=[o for i_o, o in enumerate(opts) if i_o != slot],
                    part_of_speech=it.get("part_of_speech", "")
                )
        return balanced_items

    @classmethod
    def extract_sense_verified_noun_verb_collocations(
        cls,
        sense: Optional[Dict[str, Any]],
        target_word: str,
        quote: str = "",
    ) -> List[Dict[str, Any]]:
        """
        Extracts canonical singular `Verb + a/an + <noun>` collocation bundles for countable nouns,
        strictly gated by:
          (A) Single-sense noun entry (`len(noun_senses) == 1`), OR
          (B) Direct attestation in `sense["examples"]` or `quote` (`Sense 对应例句`).
        """
        if not sense or not target_word:
            return []
        t_clean = (target_word or "").strip().lower()
        if not t_clean or " " in t_clean:
            return []
        s_pos = (sense.get("pos") or "").lower()
        s_gram = (sense.get("grammar") or "").lower()
        if "noun" not in s_pos or "uncountable" in s_gram or "plural" in s_gram:
            return []

        sense_examples = [
            re.sub(r"\s*\(=[^)]*\)", "", str(ex)).strip()
            for ex in (sense.get("examples") or [])
            if str(ex).strip()
        ]
        # Require the locked sense's own examples to attest the singular form `t_clean`
        if not any(re.search(rf"\b{re.escape(t_clean)}\b", ex.lower()) for ex in sense_examples):
            return []

        entry = cls.get_ldoce_entry(t_clean)
        noun_senses = [
            s for s in ((entry or {}).get("senses") or [])
            if "noun" in (s.get("pos") or (entry or {}).get("pos") or "").lower()
        ]
        is_single_noun_sense = len(noun_senses) == 1
        q_low = (quote or "").lower()
        out_bundles: List[Dict[str, Any]] = []
        seen_pats: Set[str] = set()

        t_forms = {t_clean, cls.pluralize_noun(t_clean)} | set(cls.inflected_forms(t_clean))
        t_rx = r"(?:" + "|".join(re.escape(f) for f in sorted(t_forms, key=len, reverse=True) if f) + r")"

        def _verb_in_text(v_phrase: str, txt: str) -> bool:
            vp = v_phrase.strip().lower()
            parts = vp.split()
            if not parts:
                return False
            v_forms = {parts[0]} | set(cls.inflected_forms(parts[0]))
            v_head_rx = r"(?:" + "|".join(re.escape(f) for f in sorted(v_forms, key=len, reverse=True) if f) + r")"
            np_window = r"(?:\s+(?!that\b|which\b|who\b|whom\b|whose\b|when\b|where\b|because\b|although\b|though\b|while\b|if\b|whether\b)[a-z'\-]+){0,4}"
            if len(parts) == 1:
                # Do not match bare 'pay' against phrasal 'pay off / pay back', and require same-clause NP window
                return bool(
                    re.search(rf"\b{v_head_rx}\b(?!\s+(?:off|out|back|up|down|in|over)\b){np_window}\s+\b{t_rx}\b", txt.lower())
                )
            tail_rx = r"\s+".join(re.escape(p) for p in parts[1:])
            return bool(re.search(rf"\b{v_head_rx}\s+{tail_rx}\b{np_window}\s+\b{t_rx}\b", txt.lower()))

        # 1. Check `entry["collocations"]["verbs"]` for `Verb + a/an + <noun>`
        if entry:
            senses_all = entry.get("senses") or []
            locked_s_idx = None
            for idx_s, s_c in enumerate(senses_all):
                if s_c is sense or (
                    s_c.get("definition") == sense.get("definition")
                    and s_c.get("pos") == sense.get("pos")
                ):
                    locked_s_idx = idx_s
                    break
            s_kws: Set[str] = set()
            o_kws: Set[str] = set()
            if not is_single_noun_sense:
                l_str = f"{sense.get('signpost') or ''} {sense.get('definition') or ''} {' '.join(sense_examples)} {' '.join(str(x) for x in (sense.get('synonyms') or []))}".lower()
                s_kws = {w for w in re.findall(r"[a-z]{4,}", l_str) if w != t_clean and w not in cls._ANCHOR_STOPWORDS}
                for idx_o, s_o in enumerate(senses_all):
                    if s_o is sense or idx_o == locked_s_idx:
                        continue
                    o_str = f"{s_o.get('signpost') or ''} {s_o.get('definition') or ''} {' '.join(str(x) for x in (s_o.get('examples') or [])[:4])}".lower()
                    o_kws |= {w for w in re.findall(r"[a-z]{4,}", o_str) if w != t_clean and w not in cls._ANCHOR_STOPWORDS and w not in s_kws}
            verb_collocs = (entry.get("collocations") or {}).get("verbs") or []
            for vc in verb_collocs:
                if vc.get("source") == "corpus":
                    continue
                c_raw = str(vc.get("collocation") or "").strip()
                if not c_raw:
                    continue
                if not is_single_noun_sense:
                    vc_s_inds = vc.get("sense_indices")
                    if isinstance(vc_s_inds, list) and vc_s_inds and locked_s_idx is not None and locked_s_idx not in vc_s_inds:
                        continue
                m_vc = re.match(
                    rf"^([a-zA-Z]+(?:\s+[a-zA-Z]+)?(?:/[a-zA-Z]+(?:\s+[a-zA-Z]+)?)*)\s+(a|an)\s+{re.escape(t_clean)}$",
                    c_raw.strip(),
                    flags=re.IGNORECASE,
                )
                if not m_vc:
                    continue
                verb_alts = [v.strip().lower() for v in m_vc.group(1).split("/") if v.strip()]
                art = m_vc.group(2).lower()
                raw_vc_exs = ([vc["example"]] if vc.get("example") else []) + (vc.get("examples") or [])
                vc_exs = [
                    re.sub(r"\s*\(=[^)]*\)", "", str(e)).strip()
                    for e in raw_vc_exs
                    if str(e).strip()
                ]
                if not is_single_noun_sense and o_kws:
                    vc_probe = f"{c_raw} {vc.get('gloss') or ''} {' '.join(vc_exs)}".lower()
                    vc_words = {w for w in re.findall(r"[a-z]{4,}", vc_probe) if w != t_clean and w not in cls._ANCHOR_STOPWORDS}
                    if len(vc_words & o_kws) > len(vc_words & s_kws):
                        continue
                # Check if any verb alternative is attested in quote or in the locked sense's own examples
                matched_quote_v = next(
                    (v for v in verb_alts if _verb_in_text(v, q_low)),
                    None,
                )
                matched_sense_ex_Pair = next(
                    (
                        (v, ex)
                        for v in verb_alts
                        for ex in sense_examples
                        if _verb_in_text(v, ex)
                    ),
                    None,
                )
                # Strict Sense Gate: must be either single-sense noun OR verb attested in sense_examples / quote
                if not (is_single_noun_sense or matched_sense_ex_Pair or matched_quote_v):
                    continue
                chosen_v = (
                    matched_quote_v
                    or (matched_sense_ex_Pair[0] if matched_sense_ex_Pair else verb_alts[0])
                )
                # Ensure chosen_ex prefers an example with chosen_v, falling back to vc_exs[0] or sense_examples[0]
                chosen_ex = next(
                    (ex for ex in (sense_examples + vc_exs) if _verb_in_text(chosen_v, ex)),
                    None,
                )
                if not chosen_ex:
                    if matched_sense_ex_Pair and not matched_quote_v:
                        chosen_v, chosen_ex = matched_sense_ex_Pair
                    else:
                        chosen_ex = vc_exs[0] if vc_exs else (sense_examples[0] if sense_examples else "")
                if not chosen_ex:
                    continue
                pat_str = f"{chosen_v} {art} {t_clean}"
                if pat_str not in seen_pats:
                    seen_pats.add(pat_str)
                    out_bundles.append({
                        "pattern": pat_str,
                        "example": chosen_ex,
                        "prep": None,
                        "preps": [],
                        "verb_anchor": chosen_v,
                        "from_quote_verb": bool(matched_quote_v and chosen_v == matched_quote_v),
                        "from_sense_example": bool(any(_verb_in_text(chosen_v, s_ex) for s_ex in sense_examples)),
                        "source": "sense_verb_colloc",
                    })

        # 2. Directly mine `sense["examples"]` for `Verb (+ particle) + a/an + <noun>` verified by lexical POS
        _skip_tokens = {
            "be", "is", "are", "was", "were", "been", "being",
            "have", "has", "had", "having", "do", "does", "did",
            "get", "got", "getting", "make", "made", "making",
            "in", "on", "at", "by", "with", "from", "to", "for", "of", "as",
            "into", "onto", "upon", "out", "off", "up", "down", "over", "under",
            "after", "before", "without", "within", "between", "among", "through",
            "during", "including", "against", "about", "around", "across", "behind",
            "beyond", "despite", "except", "like", "near", "since", "than", " toward",
            "towards", "until", "via", "and", "or", "but", "if", "when", "while",
            "because", "although", "though", "unless", "whether", "that", "which",
            "who", "whom", "whose", "where", "what", "how", "why", "not", "never",
            "always", "often", "usually", "just", "only", "also", "even", "still",
            "already", "almost", "nearly", "quite", "rather", "very", "too", "so",
            "such", "more", "most", "less", "least", "much", "many", "some", "any",
            "no", "each", "every", "either", "neither", "both", "all", "half",
        }
        for ex in sense_examples:
            m_ex = re.search(
                rf"\b([a-zA-Z]+(?:\s+(?:out|off|back|up|in|on|over|down))?)\s+(a|an)\s+{re.escape(t_clean)}\b",
                ex,
                flags=re.IGNORECASE,
            )
            if m_ex:
                raw_v = m_ex.group(1).strip().lower()
                v_parts = raw_v.split()
                if v_parts[0] in _skip_tokens or len(v_parts[0]) <= 2:
                    continue
                v_head_lem = cls.lemma_of(v_parts[0])
                if v_head_lem in _skip_tokens:
                    continue
                v_entry = cls.get_ldoce_entry(v_head_lem)
                v_pos_str = " ".join(
                    [(v_entry or {}).get("pos") or ""]
                    + [s.get("pos") or "" for s in ((v_entry or {}).get("senses") or [])]
                ).lower()
                if "verb" not in v_pos_str:
                    continue
                art = "an" if t_clean[0] in "aeiou" else "a"
                v_norm = " ".join([v_head_lem] + v_parts[1:])
                pat_str = f"{v_norm} {art} {t_clean}"
                if pat_str not in seen_pats:
                    seen_pats.add(pat_str)
                    out_bundles.append({
                        "pattern": pat_str,
                        "example": ex,
                        "prep": None,
                        "preps": [],
                        "verb_anchor": v_norm,
                        "from_quote_verb": bool(re.search(rf"\b{re.escape(v_norm)}\w*\b", q_low)),
                        "from_sense_example": True,
                        "source": "sense_verb_colloc",
                    })

        return out_bundles

    @classmethod
    def build_precomputed_target_skeletons(
        cls,
        vocab_content: Any,
        target_count: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Track 2: Pre-Computed Generative Skeletons (Proficiency Testing / 语境迁移单选题)
        Accepts structured vocabulary items directly (or legacy markdown string for unit tests)
        and synthesizes authoritative collocational anchors (via PedLex 6th Edition) and
        collision-free distractors (via WordNet + PedLex).
        Produces pre-computed item skeletons to strictly constrain LLM generation, guaranteeing zero
        duplicate options, zero synonym pile double-keys, and perfect stem-option-explanation alignment.
        """
        if not vocab_content:
            return []

        nlp = cls.get_spacy()
        parsed_items = cls._normalize_vocab_items_input(vocab_content, preserve_slash_slots=True)
        if len(parsed_items) > target_count + 4:
            parsed_items = cls._balance_single_and_multi_word(
                parsed_items, target_count=target_count + 4, multi_word_ratio=0.4
            )

        skeletons: List[Dict[str, Any]] = []
        used_distractors: Set[str] = set()

        # Batch Anchor Collision Set : every headword that is itself a target of
        # this same batch. An anchor (or a prepositional object welded into the micro-task
        # frame) drawn from this set produces an unsatisfiable item — the evaluator's
        # anchor-presence gate demands the word inside the stem while its cross-target
        # leakage gate bans it with a fatal flag. Anchors must therefore be drawn from
        # OUTSIDE this set; when nothing outside the set fits, the blueprint degrades to
        # the neutral semantic-field template instead of welding two gates into a deadlock.
        batch_target_words: Set[str] = set()
        single_word_targets: Set[str] = set()
        for _it in parsed_items:
            _w = _it["word"].lower().strip()
            if not _w:
                continue
            batch_target_words.add(_w)
            if " " not in _w:
                single_word_targets.add(_w)
            else:
                for _tok in _w.split():
                    if len(_tok) >= 3 and _tok not in cls._ANCHOR_STOPWORDS:
                        batch_target_words.add(_tok)

        for it in parsed_items:
            w = it["word"]
            raw_w_item = it.get("raw_word") or w
            w_lower = w.lower()
            raw_pos = it["part_of_speech"].lower()

            # Discontinuous Verb + [sb/sth] + Prep Frame Resolution (e.g. "remind [sb] of [sb] / [sth]"):
            # Stripping internal object slots leaves an ungrammatical adjacent string ('remind of') that cannot
            # fill a single blank in English ('*reminds of me'). Decompose into head verb ('remind') + bound
            # preposition anchor ('of') so the blank tests the head verb in '... ____ [sb/sth] of ...'.
            discontinuous_prep = None
            causative_extraposition_pat = None
            causative_extraposition_ex = None
            correlative_second_arm = None
            correlative_pat = None
            correlative_frame = None
            lv_triad_prep = None
            lv_triad_pat = None
            lv_triad_frame = None
            cxn_override_pat = None
            cxn_override_frame = None

            _ditrans_prep_verbs = {
                "remind", "convince", "inform", "warn", "assure", "rob", "deprive", "rid", "cure",
                "prevent", "stop", "prohibit", "discourage", "deter", "accuse", "suspect",
                "congratulate", "blame", "thank", "forgive", "talk", "trick", "deceive", "fool",
                "equate", "compare", "liken", "confuse", "conflate", "associate", "connect",
                "link", "relate", "attribute", "ascribe", "credit", "expose", "subject",
                "confine", "restrict", "limit", "reduce", "convert", "transform", "translate",
                "integrate", "incorporate", "merge", "blend", "combine", "adapt", "adjust",
                "tailor", "accustom", "reconcile", "resign", "dedicate", "devote", "commit",
                "apply", "attach", "fasten", "tie", "bind", "fix", "add", "append",
                "define", "describe", "regard", "view", "see", "treat", "use", "hail",
                "dismiss", "denounce", "condemn", "portray", "depict", "characterize",
                "classify", "categorize", "label", "brand", "cast", "cite", "quote",
                "base", "model", "ground", "center", "centre", "focus", "concentrate",
                "hinge", "depend", "rely", "count", "bank", "pride", "avail", "endow",
                "inflict", "impose", "lavish", "bestow", "confer",
            }
            orig_disc_prep = None
            v_obj_pp_tail = None
            m_v_obj_pp = re.match(
                r'^([a-z]+)\s+(?:\[[^\]]+\]|\([^)]+\)|\.\.\.)\s+((?:of|to|with|from|into|for|on|in|about|against|at|upon|over|as|under|out\s+of|beyond|within)\s+(?!sb\b|sth\b|somebody\b|something\b)[a-z]+(?:\s+[a-z]+)?)$',
                raw_w_item.strip(),
                re.IGNORECASE,
            )
            if m_v_obj_pp:
                head_v_pp = m_v_obj_pp.group(1).lower()
                if head_v_pp not in single_word_targets:
                    w = head_v_pp
                    w_lower = head_v_pp
                    raw_pos = "verb"
                    v_obj_pp_tail = re.sub(r"\s+", " ", m_v_obj_pp.group(2).lower()).strip()
                    cxn_override_pat = f"{head_v_pp} [sth] {v_obj_pp_tail}"
                    cxn_override_frame = f"... ____ [sth] {v_obj_pp_tail}"
                    cur_v_def = str(it.get("definition") or "").strip()
                    if cur_v_def and v_obj_pp_tail not in cur_v_def.lower():
                        it["definition"] = f"{cur_v_def} (in the phrase '{head_v_pp} something {v_obj_pp_tail}')"

            m_disc = None
            if not v_obj_pp_tail:
                m_disc = re.match(
                    r'^([a-z]+)\s+(?:\[[^\]]+\]|\([^)]+\)|\.\.\.)\s+(of|to|with|from|into|for|on|in|about|against|at|upon|over|as)\s+(?:\[[^\]]+\]|\([^)]+\)|\b(?:sb|sth|somebody|something)\b)\s*$',
                    raw_w_item.strip(),
                    re.IGNORECASE
                )
                if not m_disc:
                    m_disc_cand = re.match(
                        r'^([a-z]+)\s+(?:\[[^\]]+\]|\([^)]+\)|\.\.\.)\s+(of|to|with|from|into|for|on|in|about|against|at|upon|over|as)\s*$',
                        raw_w_item.strip(),
                        re.IGNORECASE
                    )
                    if m_disc_cand and (
                        m_disc_cand.group(1).lower() in _ditrans_prep_verbs
                        or m_disc_cand.group(2).lower() in ("of", "to", "with", "from", "into", "for", "about", "against", "at", "upon", "as")
                    ):
                        m_disc = m_disc_cand
                if not m_disc and str(raw_pos or "").strip().lower() != "phrasal verb" and len(w_lower.split()) == 2:
                    w_p0, w_p1 = w_lower.split()
                    _contiguous_intrans_prep_verbs = {
                        "talk", "speak", "think", "dream", "hear", "know", "complain", "boast",
                        "depend", "rely", "count", "bank", "hinge", "focus", "concentrate",
                        "insist", "elaborate", "embark", "impinge", "infringe", "intrude",
                    }
                    _is_contiguous_raw = bool(
                        re.search(rf"\b{re.escape(w_p0)}\s+{re.escape(w_p1)}\b", raw_w_item.lower())
                    )
                    if (
                        w_p1 in ("of", "from", "into", "for", "on", "to", "with")
                        and w_p0 in _ditrans_prep_verbs
                        and not (_is_contiguous_raw and w_p0 in _contiguous_intrans_prep_verbs)
                    ):
                        m_disc = re.match(r'^([a-z]+)\s+([a-z]+)$', w_lower)
            if m_disc:
                head_v = m_disc.group(1).lower()
                disc_p = m_disc.group(2).lower()
                # Only decompose if head_v is not already a standalone single-word target in the batch
                if head_v not in single_word_targets:
                    w = head_v
                    w_lower = head_v
                    discontinuous_prep = disc_p
                    orig_disc_prep = disc_p
                    raw_pos = "verb"

            # Family 1B: Catenative Verb + `to do (sth)` Placeholder Decomposition
            # (e.g. 'tend to do (sth)' where 'do (sth)' is a dictionary placeholder for '<infinitive verb>'
            # and the passage quote uses 'tend to think', not literal 'tend to do')
            m_cat_to_do = re.match(
                r"^([a-z]+)\s+to\s+do(?:\s+[\(\[][^\)\]]+[\)\]])?$",
                raw_w_item.strip(),
                re.IGNORECASE,
            )
            if m_cat_to_do and "to do" not in (it.get("quote") or "").lower():
                cat_v = m_cat_to_do.group(1).lower()
                if cat_v not in single_word_targets:
                    w = cat_v
                    w_lower = cat_v
                    raw_pos = "verb"

            # Family 1C: Copular Adjective Phrase Decomposition (e.g. "be full of (sth)", "be short of (sth)", "be proud of", "be fond of")
            # In dictionaries/syllabi, adjective complementation is often recorded with a pedagogical placeholder 'be'
            # (e.g. 'be full of (sth)'). In authentic English usage and psychometric assessment, the copula is finite
            # ('was full of', 'are short of') and the adjective phrase ('full of', 'short of') occupies the predicate.
            # Decouple 'be + <Adj> + <Prep>' into '<Adj> + <Prep>' as a predicate adjective phrase, preserving
            # the finite copula in the stem's cloze slot frame ('... was/is ____ [something]') and generating
            # syntactically parallel adjective phrase distractors (e.g. 'short of', 'devoid of', 'aware of').
            # Guard: Distinguish from 3-word passive verbal constructions ('be equipped with', 'be based on', 'be associated with')
            # where the head word is a passive participle verb.
            m_cop_adj_prep = re.match(
                r"^be\s+([a-z]+)\s+([a-z]+)(?:\s+[\(\[][^\)\]]+[\)\]])?$",
                raw_w_item.strip(),
                re.IGNORECASE,
            )
            if not m_cop_adj_prep and re.match(r"^be\s+([a-z]+)\s+([a-z]+)$", w_lower):
                m_cop_adj_prep = re.match(r"^be\s+([a-z]+)\s+([a-z]+)$", w_lower)
            if m_cop_adj_prep:
                cand_adj_w = m_cop_adj_prep.group(1).lower()
                cand_prep_w = m_cop_adj_prep.group(2).lower()
                cand_adj_entry = cls.get_ldoce_entry(cand_adj_w)
                cand_pos = str((cand_adj_entry or {}).get("pos") or "").lower()
                is_true_adj = not (cand_adj_w.endswith("ed") or cand_adj_w.endswith("en")) and (
                    cand_pos == "adjective" or bool(cls._wordnet_pos(cand_adj_w) & {"a", "s"})
                )
                if is_true_adj:
                    w = f"{cand_adj_w} {cand_prep_w}"
                    w_lower = w
                    raw_pos = "phrase"

            # Family 1A: Discontinuous Causative Verb + [sth/it] + Adjective Extraposition Frame Resolution
            # (e.g. "make [sth] possible", "find [sth] easier", "keep it simple", "leave it clear"):
            # Stripping '[sth]' into bare 2-word 'make possible' produces an ungrammatical contiguous string ('*make possible it').
            # Transform into the single-blank preparatory-object extraposition construction '(make|find|keep|leave) it <adj>'
            # with frame '<verb> it <adj> ... to do [sth]'.
            _comp_adj_to_base = {
                "easier": "easy",
                "harder": "hard",
                "clearer": "clear",
                "simpler": "simple",
                "safer": "safe",
            }
            m_caus_adj = re.match(
                r'^(make|find|keep|leave)\s+(?:it|\[[^\]]+\]|\([^)]+\)|\.\.\.)\s+(possible|impossible|clear|clearer|easy|easier|difficult|hard|harder|certain|plain|simple|simpler|safe|safer|worthwhile|necessary|unlikely|likely)$',
                raw_w_item.strip(),
                re.IGNORECASE
            )
            if not m_caus_adj and re.match(
                r'^(make|find|keep|leave)\s+(?:it\s+)?(possible|impossible|clear|clearer|easy|easier|difficult|hard|harder|certain|plain|simple|simpler|safe|safer)$',
                w_lower,
            ):
                m_caus_adj = re.match(r'^(make|find|keep|leave)\s+(?:it\s+)?([a-z]+)$', w_lower)
            if m_caus_adj:
                caus_v = m_caus_adj.group(1).lower()
                raw_adj_w = m_caus_adj.group(2).lower()
                adj_w = _comp_adj_to_base.get(raw_adj_w, raw_adj_w)
                w = f"{caus_v} it {adj_w}"
                w_lower = w
                raw_pos = "collocation"
                causative_extraposition_pat = f"{caus_v} it {adj_w} ... to do [sth]"
                adj_entry = cls.get_ldoce_entry(adj_w)
                if adj_entry and adj_entry.get("senses"):
                    for s_cand in adj_entry["senses"]:
                        if not isinstance(s_cand, dict):
                            continue
                        for pb in (s_cand.get("pattern_bundles") or []):
                            if not isinstance(pb, dict):
                                continue
                            pb_pat = str(pb.get("pattern") or "").lower()
                            if re.search(rf"\b{re.escape(caus_v)}s?\s+it\s+{re.escape(adj_w)}\b.*\bto\b", pb_pat):
                                raw_s_def = cls.format_ldoce_definition(s_cand)
                                comp_def = cls._compose_collocation_definition(
                                    f"{caus_v} {adj_w}", "collocation", adj_w, s_cand, raw_s_def, it.get("quote")
                                )
                                if comp_def:
                                    it["definition"] = comp_def
                                causative_extraposition_ex = str(pb.get("example") or "").strip() or None
                                break
                        if causative_extraposition_ex:
                            break

            # Family 2: Discontinuous Paired Correlatives ('not only ... but (also)', 'neither ... nor', 'no sooner ... than', etc.)
            # Decompose discontinuous correlative headwords into the first arm as the single-blank target while locking the second arm
            # in the stem's Cloze Slot Frame ('... ____ ... but also ...').
            _correl_decomp_map = [
                (r"^not\s+only\b.*\bbut(?:\s+also)?\b", "not only", "but also", "not only ... but also ...", "... ____ ... but also ..."),
                (r"^neither\b.*\bnor\b", "neither", "nor", "neither ... nor ...", "... ____ ... nor ..."),
                (r"^either\b.*\bor\b", "either", "or", "either ... or ...", "... ____ ... or ..."),
                (r"^whether\b.*\bor\b", "whether", "or", "whether ... or ...", "... ____ ... or ..."),
                (r"^no\s+sooner\b.*\bthan\b", "no sooner", "than", "no sooner ... than ...", "... ____ ... than ..."),
                (r"^hardly\b.*\bwhen\b", "hardly", "when", "hardly ... when ...", "... ____ ... when ..."),
                (r"^scarcely\b.*\bwhen\b", "scarcely", "when", "scarcely ... when ...", "... ____ ... when ..."),
                (r"^would\s+rather\b.*\bthan\b", "would rather", "than", "would rather ... than ...", "... ____ ... than ..."),
            ]
            for c_rx, first_arm, sec_arm, c_pat, c_frame in _correl_decomp_map:
                if re.search(c_rx, raw_w_item.strip(), re.IGNORECASE) or re.search(c_rx, w_lower, re.IGNORECASE):
                    w = first_arm
                    w_lower = first_arm
                    raw_pos = "phrase"
                    correlative_second_arm = sec_arm
                    correlative_pat = c_pat
                    correlative_frame = c_frame
                    break
            if not correlative_frame and w_lower in ("not only", "neither", "either", "no sooner", "hardly", "scarcely", "would rather") and it.get("quote"):
                q_low_corr = it["quote"].lower()
                _quote_correl_map = {
                    "not only": ("but", "not only ... but also ...", "... ____ ... but also ..."),
                    "neither": ("nor", "neither ... nor ...", "... ____ ... nor ..."),
                    "either": ("or", "either ... or ...", "... ____ ... or ..."),
                    "no sooner": ("than", "no sooner ... than ...", "... ____ ... than ..."),
                    "hardly": ("when", "hardly ... when ...", "... ____ ... when ..."),
                    "scarcely": ("when", "scarcely ... when ...", "... ____ ... when ..."),
                    "would rather": ("than", "would rather ... than ...", "... ____ ... than ..."),
                }
                sec_arm, c_pat, c_frame = _quote_correl_map[w_lower]
                if re.search(rf"\b{re.escape(w_lower)}\b[^.?!;]*\b{re.escape(sec_arm)}\b", q_low_corr):
                    raw_pos = "phrase"
                    correlative_second_arm = sec_arm
                    correlative_pat = c_pat
                    correlative_frame = c_frame

            # Family 6: Light-Verb + (Det/Adj) + Noun + Prep Triad Internal-Modifier Decomposition
            # When the passage quote inserts an internal adjective modifier before the triad head noun
            # (e.g. 'pay close attention to', 'have a profound impact on', 'play a vital role in'),
            # a contiguous 3-word phrase option cannot fit without colliding with or dropping the modifier.
            # Decompose to the triad head noun ('attention', 'impact', 'role') with locked triad frame.
            _lv_triads = {
                "pay attention to": ("pay", "attention", "to"),
                "have an impact on": ("have", "impact", "on"),
                "have an effect on": ("have", "effect", "on"),
                "play a role in": ("play", "role", "in"),
                "play a part in": ("play", "part", "in"),
                "take advantage of": ("take", "advantage", "of"),
                "make use of": ("make", "use", "of"),
            }
            if w_lower in _lv_triads and it.get("quote"):
                lv_v, lv_n, lv_p = _lv_triads[w_lower]
                m_lv_mod = re.search(
                    rf"\b({re.escape(lv_v)}(?:s|es|ed|d|ing)?|had|has|having|paid|paying|played|playing|took|taken|taking|made|making)\s+(?:(a|an|the)\s+)?([a-z]+)\s+{re.escape(lv_n)}\s+{re.escape(lv_p)}\b",
                    it["quote"],
                    re.IGNORECASE,
                )
                if m_lv_mod:
                    det_part = (m_lv_mod.group(2) or "").lower()
                    adj_mod = m_lv_mod.group(3).lower()
                    if adj_mod not in ("a", "an", "the", "no", "any", "some", "much", "more", "little", "less") and lv_n not in single_word_targets:
                        w = lv_n
                        w_lower = lv_n
                        raw_pos = "noun"
                        lv_triad_prep = lv_p
                        prefix_str = f"{lv_v} {det_part} [{adj_mod}]".replace("  ", " ") if det_part else f"{lv_v} [{adj_mod}]"
                        frame_prefix = f"{lv_v} {det_part} {adj_mod}".replace("  ", " ") if det_part else f"{lv_v} {adj_mod}"
                        lv_triad_pat = f"{prefix_str} {lv_n} {lv_p} [sth]"
                        lv_triad_frame = f"... {frame_prefix} ____ {lv_p} [sth]"

            # Prepositional Phrase with Obligatory Internal Modifier Slot (e.g. `on a (sth) scale` -> `on a large scale`):
            # Stripping `(sth)` leaves an ungrammatical bare singular count noun (`*on a scale`).
            # Realize the authentic 4-word phrase from the passage quote or dictionary example usage.
            _m_prep_mod_slot = re.match(
                r"^((?:on|in|at|to|by|under|with|from)\s+(?:a|an|the))\s+(?:\[[^\]]+\]|\([^)]+\)|\.\.\.)\s+([a-z]+)$",
                raw_w_item,
                re.IGNORECASE,
            )
            if _m_prep_mod_slot:
                _pm_prefix = _m_prep_mod_slot.group(1).lower()
                _pm_prep = _pm_prefix.split()[0]
                _pm_tail = _m_prep_mod_slot.group(2).lower()
                for _src_txt in (it.get("quote") or "", it.get("example_usage") or ""):
                    if not _src_txt:
                        continue
                    _m_realized = re.search(
                        rf"\b({re.escape(_pm_prep)}\s+(?:a|an)\s+[a-z]+\s+{re.escape(_pm_tail)})\b",
                        _src_txt,
                        re.IGNORECASE,
                    )
                    if _m_realized:
                        w = _m_realized.group(1).lower()
                        w_lower = w
                        break

            # Map POS category
            # Idiomatic Slot Gate: If single word is embedded in a fixed idiomatic construction in authentic quote
            # (e.g. 'granted' in 'take ... for granted'), prioritize the authentic idiomatic slot over isolated POS/function-word
            is_idiom_slot = False
            if w_lower == "granted" and it.get("quote") and re.search(r'\btake\b(?:\W+\w+){0,6}\W+for\W+granted\b', it["quote"], re.IGNORECASE):
                is_idiom_slot = True
                canonical_pos = "idiom_slot"
                anchor = "take for"
                anchor_type = "idiom"

            if not is_idiom_slot:
                if discontinuous_prep:
                    canonical_pos = "verb"
                elif correlative_frame:
                    canonical_pos = "phrase"
                elif lv_triad_frame:
                    canonical_pos = "noun"
                else:
                    resolved = cls.resolve_item_pos(w_lower, it.get("quote"), manual_pos=raw_pos)
                    if resolved in ("phrase", "collocation", "phrasal verb", "set phrase", "idiom"):
                        canonical_pos = "phrase"
                    elif resolved == "function_word" or cls.is_function_word(w_lower):
                        canonical_pos = "function_word"
                    elif resolved == "verb":
                        canonical_pos = "verb"
                    elif resolved in ("adjective", "adj"):
                        canonical_pos = "adj"
                    elif resolved in ("adverb", "adv"):
                        canonical_pos = "adv"
                    else:
                        canonical_pos = "noun"

            # Evidence-Grade Gate (P2): a curriculum 'Quoted Sentence' licenses an anchor or a
            # valency frame only when it actually contains a clause. Fragments lifted from
            # headings ('Punctuality Pays!') were treated as authentic usage, so the blueprint
            # welded anchors and frames onto a two-word title that never demonstrated them.
            # A weak quote is demoted to display-only evidence: it stays visible to the writer
            # as candidate_quote, but every anchor must then come from PedLex instead.
            quote_provenance, quote_wordcount = cls._quote_evidence_strength(it.get("quote"))
            it["raw_quote"] = it.get("quote") or ""
            if quote_provenance == "weak":
                it["quote"] = ""

            # 0. Sense-First Grounding (Sense Primacy — GEMINI.md §1.10):
            # Lock the authoritative Pedagogical Sense BEFORE anchor selection or distractor generation.
            target_p_code = cls._pos_code(canonical_pos)
            entry_base = cls.get_ldoce_entry(w_lower)
            if entry_base and entry_base.get("kind") == "derived":
                continue
            target_entry = entry_base
            entry_p_code = cls._pos_code((target_entry or {}).get("pos")) if target_entry else None
            if (
                not target_entry
                or not target_entry.get("senses")
                or (target_p_code and entry_p_code and entry_p_code != target_p_code)
            ) and " " not in w_lower:
                lem_w = cls.lemma_of(w_lower)
                if lem_w and lem_w != w_lower and ("-" not in w_lower or "-" in lem_w):
                    lem_entry = cls.get_ldoce_entry(lem_w)
                    if lem_entry and lem_entry.get("kind") != "derived" and lem_entry.get("senses"):
                        target_entry = lem_entry
            locked_sense = None
            t_s_idx = None
            use_pv_blocks = canonical_pos == "phrase" and (
                not target_entry
                or raw_pos == "phrasal verb"
                or str(target_entry.get("word", "")).strip().lower() != w_lower
            )
            if use_pv_blocks:
                pv_blocks = cls._phrasal_verb_blocks(w_lower)
                pv_senses = []
                if pv_blocks:
                    for blk in pv_blocks:
                        blk_hw = str(blk.get("headword") or "").strip()
                        for s_obj in (blk.get("senses") or []):
                            if isinstance(s_obj, dict):
                                s_copy = dict(s_obj)
                                if blk_hw and "phrasal_headword" not in s_copy:
                                    s_copy["phrasal_headword"] = blk_hw
                                pv_senses.append(s_copy)
                # Unified Literal + Figurative Phrasal Verb Sense Pool:
                # In LDOCE/PedLex, literal physical senses of separable phrasal verbs (e.g. 'nail something down'
                # under 'nail' Sense 1) are often stored inside the main verb entry's `pattern_bundles`/`patterns`
                # rather than the figurative `phrasal_verbs` sub-block. Fuse matching main-verb senses into `pv_senses`
                # so `_lock_sense` can lock onto the literal physical sense whenever attested by the passage/definition.
                w_parts_pv = w_lower.split()
                is_passive_3w = (
                    len(w_parts_pv) == 3
                    and w_parts_pv[0] in ("be", "get", "feel")
                    and (w_parts_pv[1].endswith("ed") or w_parts_pv[1].endswith("en") or cls.lemma_of(w_parts_pv[1]) != w_parts_pv[1])
                )
                if len(w_parts_pv) == 2 or is_passive_3w:
                    if is_passive_3w:
                        v_head_pv, part_w_pv = cls.lemma_of(w_parts_pv[1]), w_parts_pv[2]
                        pv_pat_re = re.compile(
                            rf"\b(?:{re.escape(v_head_pv)}|{re.escape(w_parts_pv[1])})\b(?:\s+(?:something|somebody|sb|sth|oneself|it|them|up|down|out|in|on|off|away|back|/|\s)*)?\b{re.escape(part_w_pv)}\b",
                            re.IGNORECASE,
                        )
                    else:
                        v_head_pv, part_w_pv = w_parts_pv[0], w_parts_pv[1]
                        pv_pat_re = re.compile(
                            rf"\b{re.escape(v_head_pv)}\b(?:\s+(?:something|somebody|sb|sth|oneself|it|them|up|down|out|in|on|off|away|back|/|\s)*)?\b{re.escape(part_w_pv)}\b",
                            re.IGNORECASE,
                        )
                    base_v_ent = cls.get_ldoce_entry(v_head_pv)
                    for cand_ent in ([base_v_ent] if base_v_ent else []):
                        for s_obj in (cand_ent.get("senses") or []):
                            if not isinstance(s_obj, dict):
                                continue
                            if "verb" not in str(s_obj.get("pos") or cand_ent.get("pos") or "").lower():
                                continue
                            s_pats = [str(p) for p in (s_obj.get("patterns") or [])] + [
                                str(pb.get("pattern") or "")
                                for pb in (s_obj.get("pattern_bundles") or [])
                                if isinstance(pb, dict)
                            ]
                            if any(pv_pat_re.search(p_str) for p_str in s_pats):
                                s_copy = dict(s_obj)
                                match_pat = next((p_str for p_str in s_pats if pv_pat_re.search(p_str)), f"{v_head_pv} something {part_w_pv}")
                                s_copy["phrasal_headword"] = match_pat
                                # Filter examples/pattern_bundles on this copy to those attesting the particle if available
                                part_re = re.compile(rf"\b{re.escape(part_w_pv)}\b", re.IGNORECASE)
                                matching_pbs = [
                                    pb for pb in (s_obj.get("pattern_bundles") or [])
                                    if isinstance(pb, dict) and (
                                        pv_pat_re.search(str(pb.get("pattern") or ""))
                                        or any(part_re.search(str(ex)) for ex in (pb.get("examples") or []))
                                    )
                                ]
                                matching_exs = [
                                    ex for ex in (s_obj.get("examples") or [])
                                    if part_re.search(str(ex))
                                ]
                                for pb in matching_pbs:
                                    for ex in (pb.get("examples") or []):
                                        if part_re.search(str(ex)) and ex not in matching_exs:
                                            matching_exs.append(ex)
                                if matching_pbs:
                                    s_copy["pattern_bundles"] = matching_pbs
                                if matching_exs:
                                    s_copy["examples"] = matching_exs
                                pv_senses.append(s_copy)
                if pv_senses:
                    synth_pv_entry = {"word": w_lower, "pos": "phrasal verb", "senses": pv_senses}
                    target_entry = synth_pv_entry
                    t_s_idx = cls._lock_sense(
                        synth_pv_entry,
                        definition=it.get("definition"),
                        quote=it.get("quote") or it.get("raw_quote"),
                        target_pos="phrase"
                    )
                    if t_s_idx is not None and 0 <= t_s_idx < len(pv_senses):
                        locked_sense = pv_senses[t_s_idx]
            if locked_sense is None and canonical_pos == "phrase" and len(w_lower.split()) == 2:
                w_p0, w_p1 = w_lower.split()
                p0_entry = cls.get_ldoce_entry(w_p0)
                p0_pos = str((p0_entry or {}).get("pos") or "").lower()
                p0_is_adj = not (w_p0.endswith("ed") or w_p0.endswith("en")) and (
                    p0_pos == "adjective" or bool(cls._wordnet_pos(w_p0) & {"a", "s"})
                )
                if p0_is_adj and p0_entry and p0_entry.get("senses"):
                    target_entry = p0_entry
                    t_s_idx = cls._lock_sense(
                        p0_entry,
                        definition=it.get("definition"),
                        quote=it.get("quote") or it.get("raw_quote"),
                        target_pos="adj"
                    )
                    if t_s_idx is not None and 0 <= t_s_idx < len(p0_entry.get("senses", [])):
                        locked_sense = p0_entry["senses"][t_s_idx]
            if locked_sense is None and target_entry and target_entry.get("senses"):
                # When decomposed from a discontinuous trivalent frame ('V + (sth/sb) + Prep'),
                # include transitive phrasal_verbs senses (for verbs like 'rid', 'endow', 'deprive')
                # alongside transitive main-entry verb senses, and filter out [intransitive, no passive] senses.
                if discontinuous_prep:
                    trans_pool = []
                    for s_obj in (target_entry.get("senses") or []):
                        if not isinstance(s_obj, dict):
                            continue
                        s_gram = f"{s_obj.get('grammar') or ''} {s_obj.get('syntax_codes') or ''}".lower()
                        if "intransitive" in s_gram and "transitive" not in s_gram.replace("intransitive", ""):
                            continue
                        trans_pool.append(s_obj)
                    for blk in cls._phrasal_verb_blocks(f"{w_lower} {discontinuous_prep}"):
                        blk_hw_l = str(blk.get("headword") or "").lower()
                        for s_obj in (blk.get("senses") or []):
                            if not isinstance(s_obj, dict):
                                continue
                            s_gram = f"{s_obj.get('grammar') or ''} {s_obj.get('syntax_codes') or ''}".lower()
                            if ("intransitive" in s_gram and "transitive" not in s_gram.replace("intransitive", "")) or "no passive" in s_gram:
                                continue
                            if trans_pool and not any(
                                re.search(
                                    rf"\b{re.escape(w_lower)}\s+(?:somebody|something|someone|sb|sth)\b.*\b{re.escape(discontinuous_prep)}\b",
                                    str(ph).lower(),
                                )
                                for ph in ([blk_hw_l] + list(s_obj.get("patterns") or []))
                            ):
                                continue
                            trans_pool.append(s_obj)
                    if trans_pool:
                        synth_trans_entry = dict(target_entry, senses=trans_pool)
                        t_s_idx = cls._lock_sense(
                            synth_trans_entry,
                            definition=it.get("definition"),
                            quote=it.get("quote") or it.get("raw_quote"),
                            target_pos=canonical_pos,
                        )
                        if t_s_idx is not None and 0 <= t_s_idx < len(trans_pool):
                            locked_sense = trans_pool[t_s_idx]
                            # Replace stale intransitive phrasal-verb definition (e.g. 'to be equal to something' for 'equate')
                            locked_def_fmt = cls.format_ldoce_definition(locked_sense)
                            if locked_def_fmt:
                                it["definition"] = locked_def_fmt
                if locked_sense is None:
                    t_s_idx = cls._lock_sense(
                        target_entry,
                        definition=it.get("definition"),
                        quote=it.get("quote") or it.get("raw_quote"),
                        target_pos=canonical_pos
                    )
                    if t_s_idx is None and target_entry is not entry_base:
                        t_s_idx = cls._lock_sense(
                            target_entry,
                            definition=it.get("definition"),
                            quote=it.get("quote") or it.get("raw_quote"),
                            target_pos=None
                        )
                    if t_s_idx is not None and 0 <= t_s_idx < len(target_entry["senses"]):
                        locked_sense = target_entry["senses"][t_s_idx]
                        # Guard against stale single-word definitions in cached markdown when the quote
                        # explicitly attests a multi-word sense unit (e.g. 'attaches great importance to' -> 'attach importance/significance etc to something')
                        q_raw_for_unit = (it.get("quote") or it.get("raw_quote") or "").strip()
                        if q_raw_for_unit and not locked_sense.get("units") and any(s_c.get("units") for s_c in target_entry["senses"]):
                            q_only_idx = cls._lock_sense(
                                target_entry,
                                definition=None,
                                quote=q_raw_for_unit,
                                target_pos=canonical_pos,
                            )
                            if q_only_idx is not None and q_only_idx != t_s_idx and 0 <= q_only_idx < len(target_entry["senses"]):
                                cand_u_sense = target_entry["senses"][q_only_idx]
                                for u_str in (cand_u_sense.get("units") or []):
                                    u_brs = cls.clean_and_expand_pattern_slashes(str(u_str).lower()) or [str(u_str).lower()]
                                    if any(
                                        len([w for w in re.findall(r"[a-z]{2,}", br) if w not in ("something", "somebody", "someone", "sb", "sth", "etc", "one", "ones", "do", "doing") and w != w_lower]) >= 2
                                        and all(
                                            cls.text_contains_form(q_raw_for_unit, w)
                                            for w in re.findall(r"[a-z]{2,}", br)
                                            if w not in ("something", "somebody", "someone", "sb", "sth", "etc", "one", "ones", "do", "doing")
                                        )
                                        for br in u_brs
                                    ):
                                        t_s_idx = q_only_idx
                                        locked_sense = cand_u_sense
                                        it["definition"] = cls.format_ldoce_definition(locked_sense)
                                        break
            if locked_sense and not it.get("definition"):
                it["definition"] = cls.format_ldoce_definition(locked_sense)

            # Enrich bare-gloss definitions (e.g. 'luncheon' -> '[formal] lunch') using PedLex Activator/Thesaurus distinction
            raw_def_curr = str(it.get("definition") or "").strip()
            def_body_curr = re.sub(r"^(?:\[[^\]]+\]\s*)+", "", raw_def_curr).strip()
            if def_body_curr and len(def_body_curr.split()) <= 2 and target_entry:
                for la in (target_entry.get("language_activator") or []):
                    if isinstance(la, dict):
                        for exp in (la.get("words") or la.get("exponents") or []):
                            if isinstance(exp, dict) and str(exp.get("word", "")).strip().lower() == w_lower:
                                la_def = str(exp.get("definition", "")).strip()
                                if la_def and len(la_def.split()) > 3 and la_def.lower() not in raw_def_curr.lower():
                                    it["definition"] = f"{raw_def_curr} ({la_def})"
                                    break
                    if "(" in str(it.get("definition", "")):
                        break
                if "(" not in str(it.get("definition", "")):
                    for th in (target_entry.get("thesaurus") or []):
                        if isinstance(th, dict) and str(th.get("word", "")).strip().lower() == w_lower:
                            th_def = str(th.get("distinction", "")).strip()
                            if th_def and len(th_def.split()) > 3 and th_def.lower() not in raw_def_curr.lower():
                                it["definition"] = f"{raw_def_curr} ({th_def})"
                                break

            # 0.1 Pre-bind Atomic Sense Pattern Bundle (Pattern + Corpus Example + Bound Preposition)
            # before anchor selection and distractor generation so that sense-native prepositions
            # participate directly in Step 1 anchor resolution and Step 3 double-key clearance.
            ldoce_pattern_example = None
            ldoce_cloze_frame = None
            pattern_anchor = None
            best_pattern = None
            initial_atomic_bundle = cls.pick_atomic_pattern_bundle(
                locked_sense, w_lower, quote=it.get("quote"), preferred_anchor=discontinuous_prep
            )
            if initial_atomic_bundle and initial_atomic_bundle.get("prep"):
                sense_native_prep = initial_atomic_bundle["prep"]
                if discontinuous_prep and discontinuous_prep != sense_native_prep:
                    discontinuous_prep = sense_native_prep
            else:
                sense_native_prep = discontinuous_prep

            # 1. Retrieve Authentic Oxford Collocational Anchor (Sense-Guided)
            if not is_idiom_slot:
                anchor = discontinuous_prep
                anchor_type = "prep" if discontinuous_prep else "collocation"
            prep_obj = None
            quote_governed_prep: Optional[str] = discontinuous_prep if discontinuous_prep else None
            quote_is_transitive = False

            if correlative_frame:
                anchor = correlative_second_arm
                anchor_type = "correl_pair"
                cxn_override_pat = correlative_pat
                cxn_override_frame = correlative_frame
            elif lv_triad_frame:
                anchor = lv_triad_prep
                anchor_type = "lv_triad"
                cxn_override_pat = lv_triad_pat
                cxn_override_frame = lv_triad_frame
            elif v_obj_pp_tail:
                anchor = v_obj_pp_tail
                anchor_type = "v_obj_pp"

            word_cxn_specs_raw = cls.get_word_cxn_specs(w_lower)
            word_cxn_specs = [
                sp for sp in word_cxn_specs_raw
                if int(sp.get("sense_idx", -1)) < 0
                or t_s_idx is None
                or int(sp.get("sense_idx", -1)) == int(t_s_idx)
            ]
            word_subcats = {str(sp.get("subcat_group") or "") for sp in word_cxn_specs}

            target_inflections_set = {w_lower} | set(cls.inflected_forms(w_lower))

            def _tok_matches_target(tok, doc_obj) -> bool:
                if "-" not in w_lower:
                    if (tok.i > 0 and doc_obj[tok.i - 1].text == "-" and not doc_obj[tok.i - 1].whitespace_) or (
                        tok.i + 1 < len(doc_obj) and doc_obj[tok.i + 1].text == "-" and not tok.whitespace_
                    ):
                        return False
                t_l = tok.text.lower()
                return t_l == w_lower or tok.lemma_.lower() == w_lower or t_l in target_inflections_set

            quote_adj_is_attributive = False

            if canonical_pos == "verb":
                q_low_v = (it.get("quote") or "").lower()
                # Family 8: Mandative Subjunctive Verb ('insist/suggest/demand/recommend that [sb] (should) [V-base]')
                if not anchor and "V_MANDATIVE_SUBJ" in word_subcats and q_low_v:
                    if re.search(
                        rf"\b{re.escape(w_lower)}(?:s|es|ed|d|ing)?\s+that\s+(?:[a-z]+\s+){{1,4}}(?:should\s+[a-z]+|be\b|not\s+[a-z]+|[a-z]+\b)",
                        q_low_v,
                    ) and not re.search(rf"\b{re.escape(w_lower)}(?:s|es|ed|d|ing)?\s+(?:on|upon|to|with)\b", q_low_v):
                        anchor = "that"
                        anchor_type = "mandative_subj"
                        cxn_override_pat = f"{w_lower} that [sb] (should) [V-base]"
                        cxn_override_frame = "... ____ that [sb] (should) [V-base]"
                # Family 7: Negative Polarity Verb ('budge' under negation)
                if not anchor and "NPI_ITEM" in word_subcats and q_low_v:
                    if re.search(r"\b(?:not|n't|never|hardly|scarcely|barely|refused?\s+to|would\s+not|could\s+not|will\s+not|cannot|can't|won't|wouldn't|couldn't)\b", q_low_v):
                        anchor = "not"
                        anchor_type = "npi"
                        cxn_override_pat = f"not/never {w_lower}"
                        cxn_override_frame = "... not/never ____"
                # Family 3: Gerund-Only vs Infinitive-Only Verb Complementation
                if not anchor and "V_GERUND_ONLY" in word_subcats and q_low_v:
                    if re.search(rf"\b{re.escape(w_lower)}(?:s|es|ed|d|ing)?\s+(?:not\s+)?[a-z]+ing\b", q_low_v):
                        anchor = "doing"
                        anchor_type = "gerund"
                        cxn_override_pat = f"{w_lower} [doing sth]"
                        cxn_override_frame = "... ____ [doing sth]"
                if not anchor and "V_INF_ONLY" in word_subcats:
                    sense_pats_low = [str(p).lower() for p in ((locked_sense or {}).get("patterns") or [])]
                    sense_is_inf_only = bool(
                        sense_pats_low
                        and all("to do" in p and not re.search(r"\b(?:sb|somebody|someone)\s+to\s+do\b", p) for p in sense_pats_low)
                    )
                    if sense_is_inf_only or (
                        q_low_v and re.search(rf"\b{re.escape(w_lower)}(?:s|es|ed|d|ing)?\s+(?:not\s+)?to\s+[a-z]+\b", q_low_v)
                    ):
                        anchor = "to"
                        anchor_type = "to_inf"
                        cxn_override_pat = f"{w_lower} to do [sth]"
                        cxn_override_frame = "... ____ to do [sth]"
                # Family 4: Verb + [sb] + to do [sth]
                _has_sb_to_do_pat = "V_OBJ_TO_DO" in word_subcats or any(
                    re.search(r"\b(?:sb|somebody|someone)\s+to\s+do\b", str(p).lower())
                    for p in ((locked_sense or {}).get("patterns") or [])
                )
                if not anchor and _has_sb_to_do_pat and q_low_v:
                    if re.search(rf"\b{re.escape(w_lower)}(?:s|es|ed|d|ing)?\s+(?:[a-z]+\s+){{1,3}}to\s+[a-z]+\b", q_low_v):
                        anchor = "to"
                        anchor_type = "sb_to_do"
                        cxn_override_pat = f"{w_lower} [sb] to do [sth]"
                        cxn_override_frame = "... ____ [sb] to do [sth]"

                # Instance-Level Syntactic Frame Resolution (F1..F5):
                # Inspect the quote's syntactic dependency frame around the target verb BEFORE selecting an anchor:
                #   F1_TRIVALENT_NP_PP / F2_MONOTRANS_NP: Verb governs a direct object (`dobj`/`obj` or quoted-speech object) -> Transitive!
                #   F4_INTRANS_GOV_PREP: Verb has NO direct object and governs a preposition (`prep` outside quoted speech) -> Intransitive + Prep!
                #   F5_INTRANS_BARE_OR_ERGATIVE: Verb has neither `dobj` nor `prep` -> Ergative Inchoative (`S_patient + V`) or Max-PMI Fallback!
                if not anchor and it["quote"]:
                    try:
                        doc = nlp(it["quote"])
                        v_collocs = cls.get_oxford_collocations(w_lower, pos="verb")
                        consensus_preps = set(v_collocs.get("prep", []))
                        if sense_native_prep:
                            consensus_preps.add(sense_native_prep)
                        for tok in doc:
                            if not _tok_matches_target(tok, doc):
                                continue
                            # Detect quotation marks opened after `tok.i` (`print "Here's a pat on the back for You..."`)
                            # Tokens inside a quoted utterance after `tok` are NOT prepositional dependents of `tok`.
                            quote_open_i = next(
                                (t_q.i for t_q in doc[tok.i + 1 :] if t_q.text in ('"', "“", "”", "‘", "’") or (t_q.text == "'" and t_q.i == tok.i + 1)),
                                None,
                            )
                            quote_has_quoted_obj = bool(quote_open_i is not None and quote_open_i <= tok.i + 2)
                            dobj_children = [
                                c for c in tok.children
                                if c.dep_ in ("dobj", "obj") and (quote_open_i is None or c.i < quote_open_i)
                            ]
                            has_quote_dobj = bool(dobj_children) or quote_has_quoted_obj

                            # Collect post-verbal prepositional children outside quoted speech
                            verified_prep = None   # (token_index, prep, pobj) confirmed by OCD/sense valency
                            unverified_prep = None  # (token_index, prep, pobj) bare syntactic complement
                            verified_prep_has_gerund = False
                            _cand_prep_children = [ch for ch in tok.children if ch.dep_ == "prep"]
                            for _dc in dobj_children:
                                for _d_prep in _dc.children:
                                    if _d_prep.dep_ == "prep" and _d_prep not in _cand_prep_children:
                                        _cand_prep_children.append(_d_prep)
                            for child in _cand_prep_children:
                                if quote_open_i is not None and child.i > quote_open_i:
                                    continue
                                prep_tok = child.text.lower()
                                if prep_tok not in ("to", "with", "from", "on", "for", "in", "into", "of", "against", "at", "upon", "towards", "as", "under", "about", "over", "between", "by"):
                                    continue
                                if child.i <= tok.i:
                                    continue  # Post-head linear constraint
                                child_pobj = None
                                has_pcomp_vbg = any(pc.dep_ == "pcomp" or pc.tag_ == "VBG" for pc in child.children)
                                for pchild in child.children:
                                    if pchild.dep_ in ("pobj", "pcomp") and pchild.lemma_.lower() not in cls._ANCHOR_STOPWORDS:
                                        child_pobj = pchild.lemma_.lower()
                                        break
                                if child_pobj is not None and (prep_tok, child_pobj) in cls._ADJUNCT_PREP_PAIRS:
                                    continue  # Fixed adjunct / interjection (e.g. 'for example')
                                if (
                                    child_pobj in (
                                        "morning", "afternoon", "evening", "night", "midnight", "noon", "day", "week",
                                        "weekend", "month", "year", "decade", "century", "hour", "minute", "moment",
                                        "sunset", "sunrise", "dawn", "dusk", "spring", "summer", "autumn", "fall", "winter",
                                        "o'clock", "oclock",
                                    )
                                    or (prep_tok == "at" and any(pc.pos_ == "NUM" for pc in child.children))
                                ) and prep_tok != sense_native_prep:
                                    continue  # Temporal adjunct PP (e.g. 'on such a freezing morning', 'at six o'clock')
                                if prep_tok in consensus_preps and (child.head == tok or has_pcomp_vbg or prep_tok == sense_native_prep or prep_tok != "of"):
                                    if verified_prep is None:
                                        verified_prep = (child.i, prep_tok, child_pobj)
                                        verified_prep_has_gerund = has_pcomp_vbg
                                elif child.head == tok and unverified_prep is None:
                                    unverified_prep = (child.i, prep_tok, child_pobj)

                            if has_quote_dobj:
                                # F1 (`V + NP + PP`) or F2 (`V + NP`): Instance is TRANSITIVE.
                                quote_is_transitive = True
                                if verified_prep is not None and (verified_prep_has_gerund or verified_prep[1] == sense_native_prep):
                                    anchor, prep_obj = verified_prep[1], verified_prep[2]
                                    anchor_type = "prep"
                                    quote_governed_prep = verified_prep[1]
                                else:
                                    for child in dobj_children:
                                        if (
                                            child.pos_ in ("NOUN", "PROPN")
                                            and child.lemma_.lower() not in cls._ANCHOR_STOPWORDS
                                            and not cls._in_forbidden_word_set(child.lemma_.lower(), batch_target_words)
                                        ):
                                            obj_form = child.text.lower() if child.tag_ == "NNS" else child.lemma_.lower()
                                            anchor = obj_form
                                            anchor_type = "object"
                                            break
                                    if verified_prep is not None:
                                        quote_governed_prep = verified_prep[1]
                                        # If `dobj` was a human pronoun (`warned him against`) and the sense explicitly has a `V + sb + prep` frame,
                                        # allow the verified preposition if no lexical object was extracted from the quote.
                                        if not anchor and any(
                                            c.pos_ == "PRON" and c.lemma_.lower() in ("he", "him", "she", "her", "me", "us", "you", "they", "them", "someone", "somebody")
                                            for c in dobj_children
                                        ):
                                            anchor, prep_obj = verified_prep[1], verified_prep[2]
                                            anchor_type = "prep"
                            else:
                                # F4 (`V + PP`) vs F5 (`S_patient + V` ergative inchoative):
                                chosen = None
                                if verified_prep is not None and unverified_prep is not None:
                                    chosen = verified_prep if verified_prep[0] < unverified_prep[0] else unverified_prep
                                elif verified_prep is not None:
                                    chosen = verified_prep
                                elif unverified_prep is not None:
                                    chosen = unverified_prep

                                _generic_human_subjs = {
                                    "boy", "girl", "man", "woman", "person", "people", "child", "children",
                                    "student", "teacher", "volunteer", "worker", "soldier", "player", "friend",
                                    "doctor", "parent", "father", "mother", "son", "daughter", "guy", "kid",
                                    "member", "leader", "author", "writer", "reader", "speaker", "listener",
                                    "user", "customer", "villager", "passenger", "driver", "guest", "visitor",
                                }
                                subj_candidates = [c for c in tok.children if c.dep_ in ("nsubj", "nsubjpass") and c.i < tok.i]
                                if not subj_candidates and tok.dep_ == "xcomp" and tok.head.pos_ in ("VERB", "AUX") and tok.head.lemma_.lower() in (
                                    "begin", "start", "continue", "cease", "tend", "fail", "seem", "appear",
                                    "refuse", "manage", "threaten", "used", "going", "supposed", "expected",
                                ):
                                    subj_candidates = [c for c in tok.head.children if c.dep_ in ("nsubj", "nsubjpass") and c.i < tok.head.i]
                                patient_subj_form: Optional[str] = None
                                for child in subj_candidates:
                                    c_lem = child.lemma_.lower()
                                    if (
                                        child.pos_ in ("NOUN", "PROPN")
                                        and child.ent_type_ != "PERSON"
                                        and c_lem not in cls._ANCHOR_STOPWORDS
                                        and c_lem not in _generic_human_subjs
                                        and not cls._in_forbidden_word_set(c_lem, batch_target_words)
                                    ):
                                        c_supers = cls.get_wordnet_supersenses(c_lem, pos="n")
                                        if c_supers and all(ss == "noun.person" for ss in c_supers):
                                            continue
                                        patient_subj_form = child.text.lower() if child.tag_ == "NNS" else c_lem
                                        break

                                s_def_low_erg = str((locked_sense or {}).get("definition") or it.get("definition") or "").lower().strip()
                                s_gram_low_v = str((locked_sense or {}).get("grammar") or "").lower()
                                _has_have_aux = any(c.dep_ == "aux" and c.lemma_.lower() == "have" for c in tok.children)
                                is_quote_passive = (
                                    any(c.dep_ in ("auxpass", "nsubjpass") for c in tok.children)
                                    or (tok.tag_ == "VBN" and not _has_have_aux)
                                )
                                is_strictly_trans_sense = (
                                    "transitive" in s_gram_low_v
                                    and "intransitive" not in s_gram_low_v.replace("transitive", "")
                                )
                                is_ergative_inchoative_sense = bool(
                                    not is_strictly_trans_sense
                                    and not is_quote_passive
                                    and (
                                        re.match(
                                            r"^(?:\[[^\]]+\]\s*)*(?:if\s+(?:a|an|the|something|somebody's|someone's|your)\b|to\s+(?:become|grow|fall|rise|sink|shrink|expand|disappear|appear|cease|stop|begin\s+to\s+work)\b)",
                                            s_def_low_erg,
                                        )
                                        or re.search(
                                            r"\bor\s+if\s+(?:it|they|something|a|an)\s+[a-z]+s\b|\bor\s+to\s+become\s+[a-z]+\b",
                                            s_def_low_erg,
                                        )
                                        or (
                                            "intransitive" in s_gram_low_v
                                            and chosen is None
                                        )
                                        or w_lower in (
                                            "drop", "rise", "fall", "melt", "freeze", "boil", "evaporate", "condense",
                                            "shrink", "expand", "collapse", "explode", "burst", "crack", "shatter",
                                            "fade", "vanish", "disappear", "emerge", "subside", "open", "close",
                                        )
                                    )
                                )
                                if patient_subj_form and is_ergative_inchoative_sense and (
                                    chosen is None or chosen[1] != sense_native_prep
                                ):
                                    anchor = patient_subj_form
                                    anchor_type = "verb_subject"
                                elif patient_subj_form and is_quote_passive and (
                                    chosen is None or chosen[1] != sense_native_prep
                                ):
                                    # Deep Passive Projection: in passive voice (`[Patient] be V-ed`),
                                    # `patient_subj_form` (`nsubjpass`) is the underlying direct object (`受动者`), NOT `verb_subject`.
                                    anchor = patient_subj_form
                                    anchor_type = "object"
                                elif chosen is not None and not (
                                    is_quote_passive
                                    and is_strictly_trans_sense
                                    and verified_prep is None
                                    and str(raw_w_item or w_lower).strip().lower() == w_lower
                                ):
                                    anchor, prep_obj = chosen[1], chosen[2]
                                    anchor_type = "prep"
                                    quote_governed_prep = anchor
                            break
                    except Exception:
                        pass

                # Verb Step B3: Fallback to sense-aligned direct noun object (from locked sense examples + OCD) or sense-native prep
                if not anchor:
                    has_sense_inf_or_gerund = bool(
                        not quote_is_transitive
                        and initial_atomic_bundle
                        and (initial_atomic_bundle.get("inf_valency") or initial_atomic_bundle.get("gerund_valency"))
                        and initial_atomic_bundle.get("example")
                    )
                    if not has_sense_inf_or_gerund:
                        sense_ex_dobj_nouns: List[str] = []
                        if locked_sense and locked_sense.get("examples"):
                            try:
                                for s_ex_str in (locked_sense.get("examples") or []):
                                    doc_sex = nlp(str(s_ex_str))
                                    for t_sex in doc_sex:
                                        if _tok_matches_target(t_sex, doc_sex):
                                            is_sex_passive = any(c.dep_ in ("auxpass", "nsubjpass") for c in t_sex.children)
                                            for ch_sex in t_sex.children:
                                                # Deep Passive Projection: include `nsubjpass` when the dictionary example is passive
                                                # so the underlying Patient ('受动者') is extracted as a direct object candidate.
                                                if (
                                                    (ch_sex.dep_ in ("dobj", "obj") or (is_sex_passive and ch_sex.dep_ == "nsubjpass"))
                                                    and ch_sex.pos_ in ("NOUN", "PROPN")
                                                    and ch_sex.ent_type_ != "PERSON"
                                                    and ch_sex.lemma_.lower() not in cls._ANCHOR_STOPWORDS
                                                    and not cls._in_forbidden_word_set(ch_sex.lemma_.lower(), batch_target_words)
                                                ):
                                                    n_f = ch_sex.text.lower() if ch_sex.tag_ == "NNS" else ch_sex.lemma_.lower()
                                                    if n_f not in sense_ex_dobj_nouns:
                                                        sense_ex_dobj_nouns.append(n_f)
                            except Exception:
                                pass
                        v_entry = cls.get_oxford_collocations(w_lower, pos="verb")
                        ocd_nouns = v_entry.get("colloc_nouns", []) + v_entry.get("noun_after", [])
                        candidates = sense_ex_dobj_nouns + [n for n in ocd_nouns if n not in sense_ex_dobj_nouns]
                        if candidates:
                            anchor = cls._select_sense_aligned_anchor(candidates, definition=it.get("definition"), quote=it.get("quote"), target_word=w_lower, forbidden=batch_target_words)
                            if not anchor and sense_ex_dobj_nouns:
                                anchor = sense_ex_dobj_nouns[0]
                            if anchor:
                                anchor_type = "object"
                    if not anchor and sense_native_prep and not cls._in_forbidden_word_set(sense_native_prep, batch_target_words):
                        anchor = sense_native_prep
                        anchor_type = "prep"

            elif canonical_pos == "noun":
                n_entry = cls.get_oxford_collocations(w_lower, pos="noun")
                q_low_n = (it.get("quote") or "").lower()
                # Family 5: Shell Nouns & Abstract Nouns Taking Post-Nominal Complements
                if not anchor and q_low_n and word_subcats & {"N_SHELL_TO_DO", "N_SHELL_OF_DOING", "N_SHELL_THAT_ONLY"}:
                    _is_shell_to_inf = False
                    if "N_SHELL_TO_DO" in word_subcats and re.search(rf"\b{re.escape(w_lower)}(?:s|es|ies)?\s+to\s+[a-z]+\b", q_low_n):
                        try:
                            doc_n_shell = nlp(it["quote"])
                            for tok_n in doc_n_shell:
                                if _tok_matches_target(tok_n, doc_n_shell) and tok_n.i + 2 < len(doc_n_shell):
                                    nxt1 = doc_n_shell[tok_n.i + 1]
                                    nxt2 = doc_n_shell[tok_n.i + 2]
                                    if nxt1.text.lower() == "to" and (
                                        nxt1.tag_ == "TO"
                                        or nxt2.tag_ == "VB"
                                        or nxt2.pos_ in ("VERB", "AUX")
                                    ) and nxt2.pos_ not in ("DET", "PRON", "PROPN", "NOUN", "NUM"):
                                        _is_shell_to_inf = True
                                        break
                        except Exception:
                            pass
                    if _is_shell_to_inf:
                        anchor = "to"
                        anchor_type = "shell_to_do"
                        cxn_override_pat = f"{w_lower} to do [sth]"
                        cxn_override_frame = "... ____ to do [sth]"
                    elif "N_SHELL_OF_DOING" in word_subcats and re.search(rf"\b{re.escape(w_lower)}(?:s|es|ies)?\s+of\s+(?:[a-z]+\s+)?[a-z]+ing\b", q_low_n):
                        anchor = "of"
                        anchor_type = "shell_of_doing"
                        cxn_override_pat = f"{w_lower} of [doing sth]"
                        cxn_override_frame = "... ____ of [doing sth]"
                    elif "N_SHELL_THAT_ONLY" in word_subcats and re.search(rf"\b{re.escape(w_lower)}(?:s|es|ies)?\s+that\s+[a-z]+\b", q_low_n):
                        anchor = "that"
                        anchor_type = "shell_that"
                        cxn_override_pat = f"the {w_lower} that ..."
                        cxn_override_frame = "... the ____ that ..."
                if not anchor and it["quote"]:
                    try:
                        doc = nlp(it["quote"])
                        ocd_prep_tokens = set()
                        for p in n_entry.get("prep", []):
                            cleaned = re.sub(r'[\(\)]', '', p).lower()
                            for prep_tok in cleaned.replace("/", " ").split():
                                if prep_tok in ("to", "with", "from", "on", "for", "in", "into", "of", "against", "at", "upon", "towards", "toward", "over", "under", "about"):
                                    ocd_prep_tokens.add(prep_tok)
                        if locked_sense is not None:
                            for eb in cls.extract_sense_pattern_bundles(locked_sense, w_lower):
                                if eb.get("prep"):
                                    ocd_prep_tokens.add(eb["prep"])
                        else:
                            for lp in cls.get_ldoce_preps(w_lower):
                                ocd_prep_tokens.add(lp)

                        for tok in doc:
                            if _tok_matches_target(tok, doc):
                                for child in tok.children:
                                    if child.dep_ == "prep" and child.text.lower() in ("to", "with", "from", "on", "for", "in", "into", "of", "against", "at", "upon", "towards", "toward", "over", "under", "about"):
                                        prep_word = child.text.lower()
                                        if ocd_prep_tokens and prep_word not in ocd_prep_tokens:
                                            continue
                                        pobj_tok = None
                                        for pchild in child.children:
                                            if pchild.dep_ == "pobj" and pchild.pos_ != "PRON" and pchild.lemma_.lower() not in cls._ANCHOR_STOPWORDS and pchild.lemma_.lower() != w_lower:
                                                pobj_tok = pchild.lemma_.lower()
                                                break
                                        if pobj_tok is not None and (prep_word, pobj_tok) in cls._ADJUNCT_PREP_PAIRS:
                                            continue
                                        anchor = prep_word
                                        anchor_type = "prep"
                                        prep_obj = pobj_tok
                                        quote_governed_prep = anchor
                                        break
                                if not anchor:
                                    has_prep_or_bundle_ev = bool(
                                        sense_native_prep
                                        or (locked_sense and locked_sense.get("pattern_bundles"))
                                        or (n_entry and n_entry.get("prep"))
                                    )
                                    attested_adj_text = " ".join([
                                        " ".join(str(a) for a in (n_entry.get("adj") or [])),
                                        " ".join(str(u) for u in ((locked_sense or {}).get("units") or [])),
                                        " ".join(str(p) for p in ((locked_sense or {}).get("patterns") or [])),
                                        " ".join(str(ex) for ex in ((locked_sense or {}).get("examples") or [])),
                                    ]).lower()
                                    for child in tok.children:
                                        if child.dep_ in ("amod", "compound"):
                                            mod_tok = (
                                                child.text.lower()
                                                if child.dep_ == "compound" and child.tag_ in ("NNS", "NNPS")
                                                else child.lemma_.lower()
                                            )
                                            if (
                                                child.pos_ != "PRON"
                                                and mod_tok not in cls._ANCHOR_STOPWORDS
                                                and mod_tok not in cls.GENERIC_COLLOCATION_MODIFIERS
                                                and not cls._is_delexical_anchor(mod_tok, "adj", "noun")
                                                and mod_tok != w_lower
                                                and not cls._in_forbidden_word_set(mod_tok, batch_target_words)
                                                and (
                                                    not has_prep_or_bundle_ev
                                                    or bool(re.search(rf"\b{re.escape(mod_tok)}\b", attested_adj_text))
                                                )
                                            ):
                                                anchor = mod_tok
                                                anchor_type = "adj"
                                                break
                                if not anchor and tok.dep_ in ("dobj", "pobj", "attr"):
                                    head_tok = tok.head
                                    if head_tok.pos_ == "VERB":
                                        head_tok_lemma = head_tok.lemma_.lower()
                                        if (head_tok.pos_ != "PRON"
                                                and head_tok_lemma not in cls._ANCHOR_STOPWORDS
                                                and head_tok_lemma != w_lower
                                                and head_tok_lemma not in cls._DELEXICAL_VERBS
                                                and not cls._in_forbidden_word_set(head_tok_lemma, batch_target_words)):
                                            verb_before_set = {v.lower().split()[0] for v in n_entry.get("verb_before", [])}
                                            if not verb_before_set or head_tok_lemma in verb_before_set:
                                                anchor = head_tok_lemma
                                                anchor_type = "verb"
                                break
                    except Exception:
                        pass

                # Priority D: dictionary candidates, strictly gated by quote/definition evidence
                if not anchor and n_entry:
                    filtered_adj = [
                        a for a in (n_entry.get("adj") or [])
                        if str(a).strip().lower() not in cls.GENERIC_COLLOCATION_MODIFIERS
                    ]
                    sense_mod_cands = []
                    if isinstance(locked_sense, dict):
                        for pat_s in (locked_sense.get("patterns") or []):
                            pat_toks = [t.lower() for t in re.findall(r"[a-z]+", str(pat_s)) if t.lower() not in ("a", "an", "the", "some", "any", "one")]
                            if len(pat_toks) == 2 and cls.form_matches(pat_toks[1], w_lower) and pat_toks[0] not in cls.GENERIC_COLLOCATION_MODIFIERS:
                                sense_mod_cands.append(pat_toks[0])
                    if filtered_adj or sense_mod_cands or n_entry.get("verb_before"):
                        candidates = sense_mod_cands + filtered_adj + n_entry.get("verb_before", [])
                        anchor = cls._select_sense_aligned_anchor(candidates, definition=it.get("definition"), quote=it.get("quote"), target_word=w_lower, forbidden=batch_target_words)
                        if anchor:
                            anchor_type = "adj" if cls._token_in_phrases(anchor, sense_mod_cands + filtered_adj) else "verb"
                            if cls._is_delexical_anchor(anchor, anchor_type, "noun"):
                                anchor, anchor_type = None, "collocation"
                    if not anchor and sense_native_prep and not cls._in_forbidden_word_set(sense_native_prep, batch_target_words):
                        anchor = sense_native_prep
                        anchor_type = "prep"
                    if not anchor and n_entry.get("prep"):
                        p_cand = cls._pick_anchor_token(n_entry["prep"], target_word=w_lower, forbidden=batch_target_words)
                        if p_cand:
                            anchor = p_cand
                            anchor_type = "prep"

                # Priority E (B2): the partitive compound the passage itself shows.
                if not anchor and it["quote"]:
                    pt_anchor, pt_type, pt_head = cls._partitive_compound_anchor(
                        w_lower,
                        it["quote"],
                        forbidden=batch_target_words
                    )
                    if pt_anchor:
                        anchor, anchor_type, prep_obj = pt_anchor, pt_type, pt_head

            elif canonical_pos == "adj":
                adj_locked_gram = str((locked_sense or {}).get("grammar") or "").lower()
                adj_prof = cls.get_word_gram_profile(w_lower, pos="adj")
                is_not_before_noun_adj = (
                    "not before noun" in adj_locked_gram
                    or "never before noun" in adj_locked_gram
                    or bool(adj_prof.get("not_before_noun"))
                )
                q_low_adj = (it.get("quote") or "").lower()
                # Family 8: Mandative Subjunctive Adjective ('It is essential/vital/imperative that [sb] (should) [V-base]')
                if not anchor and "ADJ_MANDATIVE_SUBJ" in word_subcats and q_low_adj:
                    if re.search(rf"\bit\s+(?:is|was)\s+(?:not\s+|very\s+|quite\s+)?{re.escape(w_lower)}\s+that\b", q_low_adj):
                        anchor = "that"
                        anchor_type = "mandative_subj"
                        cxn_override_pat = f"it is {w_lower} that [sb] (should) [V-base]"
                        cxn_override_frame = "It is ____ that [sb] (should) [V-base]"
                # Family 1B: Subject Extraposition 'It is <adj> (for/of [sb]) to do [sth]' & For-To Infinitival Guard
                # Never extract 'for' or 'of' inside 'It is <adj> for/of sb to do' as a lexical preposition anchor ('*necessary for [sth]').
                if not anchor and q_low_adj:
                    m_it_adj_inf = re.search(
                        rf"\bit\s+(?:is|was|seems|appears|becomes|gets|'s|’s|would\s+be|will\s+be|can\s+be|could\s+be)\s+(?:not\s+|very\s+|quite\s+|extremely\s+|particularly\s+|especially\s+|too\s+|so\s+)?{re.escape(w_lower)}\b(?:[^.?!;]*?\b(for|of)\s+[^.?!;]+?)?\s+to\s+[a-z]+\b",
                        q_low_adj,
                    )
                    if m_it_adj_inf:
                        anchor = "to"
                        anchor_type = "it_extrapos"
                        for_or_of = (m_it_adj_inf.group(1) or "").lower()
                        if for_or_of in ("for", "of"):
                            cxn_override_pat = f"it is {w_lower} {for_or_of} [sb] to do [sth]"
                            cxn_override_frame = f"It is ____ {for_or_of} [sb] to do [sth]"
                        else:
                            cxn_override_pat = f"it is {w_lower} to do [sth]"
                            cxn_override_frame = "It is ____ to do [sth]"
                # Family 1C: Finite `that`-clause Extraposition ('It is <adj> that ...' or coordinated 'It is ... and/but <adj> that ...')
                _it_that_extrapos_adjs = {
                    "likely", "unlikely", "clear", "obvious", "possible", "impossible",
                    "probable", "improbable", "certain", "uncertain", "true", "evident",
                    "apparent", "natural", "strange", "odd", "surprising", "fortunate", "unfortunate",
                }
                if not anchor and q_low_adj and ("ADJ_IT_EXTRAPOS" in word_subcats or w_lower in _it_that_extrapos_adjs):
                    if re.search(
                        rf"\bit\s*(?:is|was|seems|appears|'s|’s|would\s+be|will\s+be)\b[^.?!;]*?\b{re.escape(w_lower)}\s+that\b",
                        q_low_adj,
                    ):
                        anchor = "that"
                        anchor_type = "it_extrapos"
                        cxn_override_pat = f"it is {w_lower} that ..."
                        cxn_override_frame = "It is ____ that ..."
                # Pre-fetch OCD entry & attested prepositions so Adjective Step A1 never mistakes a clause-level
                # temporal/circumstantial adjunct (e.g. 'remained unsolved for 1,400 years') for a bound adjective preposition.
                adj_entry = cls.get_oxford_collocations(w_lower, pos="adj")
                attested_adj_preps = set()
                if sense_native_prep:
                    attested_adj_preps.add(sense_native_prep.lower())
                if adj_entry and adj_entry.get("prep"):
                    for raw_p in adj_entry["prep"]:
                        raw_p_clean = re.sub(r"\[.*?\]|\(.*?\)", " ", str(raw_p).lower())
                        for m_ap in re.finditer(
                            rf"\b{re.escape(w_lower)}\s+(to|for|with|of|in|from|at|about|on|by|into|towards|against|over)\b",
                            raw_p_clean,
                        ):
                            attested_adj_preps.add(m_ap.group(1))
                if locked_sense:
                    for b in cls.extract_sense_pattern_bundles(locked_sense, w_lower):
                        for p_tok in (b.get("preps") or []):
                            attested_adj_preps.add(p_tok.lower())
                else:
                    for p_tok in cls.get_ldoce_preps(w_lower, pos="adj"):
                        attested_adj_preps.add(p_tok.lower())
                for p_key, adj_set in cls._ADJ_PREP_VALENCY_MAP.items():
                    if w_lower in adj_set:
                        attested_adj_preps.add(p_key)
                if w_lower in {
                    "inaudible", "audible", "invisible", "visible", "imperceptible",
                    "immune", "allergic", "susceptible", "proportional", "akin",
                    "analogous", "adjacent", "prior", "subsequent", "preferable",
                    "superior", "inferior", "identical", "relevant", "loyal",
                    "secondary", "accustomed",
                }:
                    attested_adj_preps.add("to")

                # Adjective Step A: Syntactic Dependency Analysis on Authentic Quote
                if not anchor and it["quote"]:
                    try:
                        doc = nlp(it["quote"])
                        _temporal_pobj_lemmas = {
                            "year", "month", "week", "day", "hour", "minute", "second",
                            "decade", "century", "millennium", "period", "time", "while", "moment", "age", "era"
                        }
                        q_start_i, q_end_i, tok = cls._find_target_span_in_doc(
                            doc, w_lower, base_str=w_lower, t_forms=target_inflections_set
                        )
                        if tok is not None and q_start_i is not None and q_end_i is not None:
                            eff_head = tok
                            while eff_head.dep_ == "conj" and eff_head.head and eff_head.head.pos_ == "ADJ" and eff_head.head.i < eff_head.i:
                                eff_head = eff_head.head
                            # A1. Predicative Valency: Check if adjective governs a lexicographically attested bound preposition
                            for child in tok.children:
                                c_low = child.text.lower()
                                if child.dep_ == "prep" and child.i >= q_end_i and c_low in ("to", "for", "with", "of", "in", "from", "at", "about", "on"):
                                    if c_low not in attested_adj_preps:
                                        continue
                                    pobj_tokens = [pc for pc in child.children if pc.dep_ == "pobj"]
                                    if pobj_tokens and any(
                                        pc.ent_type_ in ("DATE", "TIME") or pc.lemma_.lower() in _temporal_pobj_lemmas
                                        for pc in pobj_tokens
                                    ):
                                        continue
                                    anchor = c_low
                                    anchor_type = "prep"
                                    quote_governed_prep = anchor
                                    for pchild in pobj_tokens:
                                        if pchild.pos_ != "PRON" and pchild.text.lower() not in cls._ANCHOR_STOPWORDS and pchild.lemma_.lower() != w_lower:
                                            prep_obj = pchild.lemma_.lower()
                                            break
                                    break
                            # A2. Attributive Modification: Check if adjective modifies a head noun (dep_ == "amod" / "compound")
                            if not anchor and not is_not_before_noun_adj:
                                head_n = None
                                if eff_head.dep_ in ("amod", "compound") and eff_head.head and eff_head.head.pos_ in ("NOUN", "PROPN") and q_end_i <= eff_head.head.i:
                                    head_n = eff_head.head
                                elif q_end_i < len(doc) and doc[q_end_i].pos_ in ("NOUN", "PROPN"):
                                    head_n = doc[q_end_i]
                                elif eff_head.dep_ in ("amod", "compound") and q_start_i > 0 and doc[q_start_i - 1].pos_ == "DET":
                                    # Quoted nominalized expression preceded by determiner (e.g. 'a simple "Hello"')
                                    quote_adj_is_attributive = True
                                if head_n is not None:
                                    quote_adj_is_attributive = True
                                    if (
                                        head_n.pos_ != "PRON"
                                        and head_n.text.lower() not in cls._ANCHOR_STOPWORDS
                                        and head_n.lemma_.lower() != w_lower
                                        and not cls._in_forbidden_word_set(head_n.lemma_.lower(), batch_target_words)
                                    ):
                                        anchor = head_n.lemma_.lower()
                                        anchor_type = "modified_noun"
                            # A3. Predicative / Causative Complement: Check if adjective is complement of a causative/linking verb
                            if not anchor and eff_head.dep_ in ("ccomp", "xcomp", "acomp", "oprd") and eff_head.head and eff_head.head.pos_ in ("VERB", "AUX"):
                                head_v = eff_head.head.lemma_.lower()
                                if head_v in (
                                    "make", "render", "find", "deem", "keep", "consider",
                                    "feel", "sound", "seem", "remain", "stay", "become", "grow", "turn", "prove", "look", "appear",
                                ):
                                    anchor = head_v
                                    anchor_type = "verb_copula"
                    except Exception:
                        pass

                # Adjective Step B: Sense-Native Preposition Valency & Collocation Lexicon Lookup (OCD) if no anchor found in quote
                if not anchor and not quote_adj_is_attributive and sense_native_prep and not cls._in_forbidden_word_set(sense_native_prep, batch_target_words):
                    anchor = sense_native_prep
                    anchor_type = "prep"
                if not anchor and adj_entry:
                    if not is_not_before_noun_adj:
                        # Use `colloc_nouns` (NEVER `noun_after`, which stores `colls["phrases"]` like "keep silent")
                        candidates = list(adj_entry.get("colloc_nouns", []))
                        if candidates:
                            anchor = cls._select_sense_aligned_anchor(candidates, definition=it.get("definition"), quote=it.get("quote"), target_word=w_lower, forbidden=batch_target_words)
                            if anchor:
                                anchor_type = "modified_noun"
                    if not anchor and not quote_adj_is_attributive and sense_native_prep and not cls._in_forbidden_word_set(sense_native_prep, batch_target_words):
                        anchor = sense_native_prep
                        anchor_type = "prep"
                    if not anchor and not quote_adj_is_attributive and adj_entry.get("prep"):
                        for raw_p in adj_entry["prep"]:
                            raw_p_clean = re.sub(r"\[.*?\]|\(.*?\)", " ", str(raw_p).lower())
                            m_ap = re.search(
                                rf"\b{re.escape(w_lower)}\s+(to|for|with|of|in|from|at|about|on|by|into|towards|against|over)\b",
                                raw_p_clean,
                            )
                            if m_ap and not cls._in_forbidden_word_set(m_ap.group(1), batch_target_words):
                                anchor = m_ap.group(1)
                                anchor_type = "prep"
                                break

                # Adjective Step C: Fall back to locked sense pattern first, then curated academic adjective preposition valency map
                if not anchor and not quote_adj_is_attributive:
                    ld_entry = target_entry or cls.get_ldoce_entry(w_lower)
                    if ld_entry:
                        candidate_senses = [locked_sense] if locked_sense else ld_entry.get("senses", [])
                        for s in candidate_senses:
                            if not isinstance(s, dict):
                                continue
                            for pat in s.get("patterns", []):
                                pat_low_c = re.sub(r"\[.*?\]|\(.*?\)", " ", str(pat).lower())
                                # For-To Infinitival Guard: do not extract 'for'/'of'/'to' as a lexical prep from 'it is <adj> (for/of sb) to do'
                                if re.search(rf"\bit\s+is\s+{re.escape(w_lower)}\b.*\bto\b", pat_low_c):
                                    continue
                                m_adj_prep = re.search(
                                    rf"\b{re.escape(w_lower)}\s+(of|to|for|with|about|from|in|on|at)\b",
                                    pat_low_c,
                                )
                                if m_adj_prep:
                                    p = m_adj_prep.group(1)
                                    if p == "to":
                                        m_inf = re.search(r"\bto\s+([a-z]+)", pat_low_c)
                                        if m_inf and m_inf.group(1) not in (
                                            "sb", "sth", "someone", "something", "oneself", "himself", "herself", "them", "the", "a", "an"
                                        ):
                                            continue
                                    anchor = p
                                    anchor_type = "prep"
                                    break
                            if anchor:
                                break
                    if not anchor:
                        for p, adjs in cls._ADJ_PREP_VALENCY_MAP.items():
                            if w_lower in adjs:
                                anchor = p
                                anchor_type = "prep"
                                break

            elif canonical_pos in ("adv", "function_word", "phrase"):
                q_low_adv = (it.get("quote") or "").lower()
                # Family 7: Fronted Negative Inversion ('Seldom/Rarely/Never/Under no circumstances + Aux + Subj + V')
                if not anchor and word_subcats & {"ADV_FRONTED_INV", "PHR_FRONTED_INV"}:
                    if canonical_pos == "phrase" or (
                        q_low_adv
                        and re.search(
                            rf"(?:^|[.;!?]\s*|\b(?:and|but)\s+){re.escape(w_lower)}\s+(?:do|does|did|have|has|had|am|is|are|was|were|can|could|will|would|shall|should|may|might|must)\b",
                            q_low_adv,
                        )
                    ):
                        anchor = "inversion"
                        anchor_type = "fronted_inv"
                        cxn_override_pat = f"{w_lower} + Aux + Subj + V"
                        cxn_override_frame = "____ + Aux + Subj + V"
                # Family 7: Negative Polarity Adverb/Phrase ('remotely', 'whatsoever', 'at all', 'lift a finger', etc.)
                if not anchor and "NPI_ITEM" in word_subcats:
                    if canonical_pos == "phrase" or (
                        q_low_adv
                        and re.search(
                            r"\b(?:not|n't|never|hardly|scarcely|barely|seldom|rarely|little|few|no|neither|nor|without|doubt|deny|refused?|failed?|unlikely|impossible)\b",
                            q_low_adv,
                        )
                    ):
                        anchor = "not"
                        anchor_type = "npi"
                        cxn_override_pat = f"not/never ... {w_lower}"
                        cxn_override_frame = "... not/never ... ____"
                # Adverb Syntactic Dependency Walker & Sense-Lock Gate
                if not anchor and canonical_pos == "adv" and it.get("quote"):
                    _discourse_conjunct_advs = {
                        "similarly", "likewise", "conversely", "otherwise", "instead", "meanwhile",
                        "moreover", "furthermore", "however", "nevertheless", "nonetheless", "therefore",
                        "consequently", "thus", "hence", "accordingly",
                    }
                    _temporal_stance_advs = {
                        "nowadays", "currently", "recently", "lately", "previously", "formerly",
                        "originally", "initially", "eventually", "ultimately", "finally", "traditionally",
                        "thereafter",
                    }
                    if w_lower in _discourse_conjunct_advs:
                        anchor_type = "discourse"
                    elif w_lower not in _temporal_stance_advs:
                        try:
                            doc = nlp(it["quote"])
                            for tok in doc:
                                if _tok_matches_target(tok, doc):
                                    if tok.i == 0 and tok.i + 1 < len(doc) and doc[tok.i + 1].text == ",":
                                        anchor_type = "discourse"
                                        break
                                    if (
                                        tok.i + 1 < len(doc)
                                        and doc[tok.i + 1].pos_ == "ADJ"
                                        and doc[tok.i + 1].lemma_.lower() not in cls._ANCHOR_STOPWORDS
                                        and doc[tok.i + 1].lemma_.lower() != w_lower
                                    ):
                                        anchor_type = "modifies_adj"
                                        if not doc[tok.i + 1].text[:1].isupper():
                                            anchor = doc[tok.i + 1].lemma_.lower()
                                        break
                                    head = tok.head
                                    if (
                                        head
                                        and abs(tok.i - head.i) <= 4
                                        and head.lemma_.lower() not in cls._ANCHOR_STOPWORDS
                                        and head.lemma_.lower() != w_lower
                                    ):
                                        h_lem = head.lemma_.lower()
                                        if head.pos_ in ("ADJ", "ADV"):
                                            anchor_type = "modifies_adj"
                                            if not (head.text[:1].isupper() and head.i > 0):
                                                anchor = h_lem
                                            break
                                        elif head.pos_ == "VERB":
                                            anchor = h_lem
                                            anchor_type = "modifies_verb"
                                            break
                        except Exception:
                            pass

                # Priority B: Dictionary lookup from PedLex zero collision anchor
                if not anchor and canonical_pos == "adv" and anchor_type != "discourse":
                    ld_anc, ld_atype, ld_frame, _ = cls.find_ldoce_zero_collision_anchor(
                        target_word=w_lower,
                        pos="adv",
                        definition=it.get("definition"),
                        quote=it.get("quote"),
                        locked_sense=locked_sense
                    )
                    if ld_anc and not (anchor_type == "modifies_adj" and ld_atype == "modifies_verb"):
                        aligned = cls._select_sense_aligned_anchor(
                            [ld_anc],
                            definition=it.get("definition"),
                            quote=it.get("quote"),
                            target_word=w_lower,
                            forbidden=batch_target_words
                        )
                        if aligned:
                            anchor = aligned
                            anchor_type = anchor_type or ld_atype or "modifies_verb"

            _cxn_anchor_types = {
                "it_extrapos", "gerund", "to_inf", "sb_to_do",
                "shell_to_do", "shell_of_doing", "shell_that",
                "correl_pair", "fronted_inv", "npi", "mandative_subj", "lv_triad", "v_obj_pp"
            }

            # 1.5 Sense-Native Atomic Pattern Anchor Inheritance (GEMINI.md §1.10):
            # If Step 1 found no anchor (or only a weak fallback) and the locked sense's
            # atomic pattern bundle specifies a bound preposition, inherit it BEFORE Step 3
            # distractor generation so double-key clearance runs against the true anchor.
            if not is_idiom_slot and sense_native_prep and anchor_type not in _cxn_anchor_types and (not anchor or anchor_type == "collocation"):
                if not cls._in_forbidden_word_set(sense_native_prep, batch_target_words):
                    anchor = sense_native_prep
                    anchor_type = "prep"
                    pattern_anchor = sense_native_prep

            # 2. Syntactic Fallback: Extract anchor from quoted sentence if OCD / prep lookup was empty
            if not anchor and anchor_type != "discourse" and it["quote"]:
                q_clean = it["quote"].strip().strip('"').strip("'").strip("“").strip("”")
                try:
                    doc = nlp(q_clean)
                    for tok in doc:
                        if _tok_matches_target(tok, doc):
                            for child in tok.children:
                                if child.dep_ in ("dobj", "pobj") and child.pos_ != "PRON" and child.lemma_.lower() not in cls._ANCHOR_STOPWORDS and child.lemma_.lower() != w_lower:
                                    anchor = child.lemma_.lower()
                                    anchor_type = "contextual"
                                    break
                                elif child.dep_ == "prep":
                                    for pchild in child.children:
                                        if pchild.dep_ == "pobj" and pchild.pos_ != "PRON" and pchild.lemma_.lower() not in cls._ANCHOR_STOPWORDS and pchild.lemma_.lower() != w_lower:
                                            anchor = pchild.lemma_.lower()
                                            anchor_type = "contextual"
                                            break
                            if not anchor and tok.head and tok.head.pos_ in ("NOUN", "VERB") and tok.dep_ not in ("conj", "parataxis", "advcl", "ccomp"):
                                h_lem = tok.head.lemma_.lower()
                                if tok.head.pos_ != "PRON" and h_lem not in cls._ANCHOR_STOPWORDS and h_lem not in cls._DELEXICAL_VERBS and h_lem != w_lower:
                                    anchor = h_lem
                                    anchor_type = "contextual"
                            break
                except Exception:
                    pass

            # 3. Generate 3 Zero-Collision Distractors with Definition-Locked WSD & Sense Antonyms
            is_quantifier_frame = False
            quant_head_noun = None
            if canonical_pos == "phrase":
                q_match = re.match(r"^(a|an)\s+([a-z]+)\s+of$", w_lower)
                if q_match:
                    is_quantifier_frame = True
                    quant_head_noun = q_match.group(2)
                    q_siblings_all = [
                        "piece", "bit", "slice", "sheet", "block", "drop",
                        "pair", "pack", "stroke", "portion", "set", "share"
                    ]
                    q_cands = [s for s in q_siblings_all if s != quant_head_noun and s not in used_distractors]
                    h_rank = cls.CEFR_ORDER.get(cls.get_word_cefr(quant_head_noun) or "A2", 2)
                    if h_rank <= 2:
                        foundational_pool = ["bit", "slice", "sheet", "block", "drop", "pair", "pack", "set"]
                        q_cands = [s for s in q_cands if s in foundational_pool] + [s for s in q_cands if s not in foundational_pool]
                    distractors = q_cands[:3]
                    dist_meta = {d: "quantifier_collocation_contrast" for d in distractors}
                else:
                    p_dists = cls.generate_phrase_distractors(
                        w_lower,
                        pos=raw_pos or canonical_pos,
                        count=3,
                        exclude_words=used_distractors,
                        definition=it.get("definition"),
                        quote=it.get("quote"),
                    )
                    distractors = p_dists
                    dist_meta = {d: "phrase_contrast" for d in p_dists}
            elif canonical_pos == "idiom_slot" and w_lower == "granted":
                i_cands = ["accepted", "given", "known", "certain", "true"]
                distractors = [c for c in i_cands if c not in used_distractors][:3]
                dist_meta = {d: "idiom_collocation_contrast" for d in distractors}
            elif anchor_type == "v_obj_pp" and v_obj_pp_tail:
                lv_pool = ["take", "bring", "keep", "make", "give", "leave", "set", "call", "turn", "hold", "draw"]
                distractors = [
                    c for c in lv_pool
                    if c != w_lower
                    and c not in used_distractors
                    and not cls.lookup_ldoce_phrase(f"{c} {v_obj_pp_tail}")
                    and not cls.lookup_ldoce_phrase(f"{c} something {v_obj_pp_tail}")
                ][:3]
                dist_meta = {d: "idiom_collocation_contrast" for d in distractors}
            else:
                distractors, dist_meta = cls.generate_vocab_distractors(
                    target_word=w_lower,
                    pos=canonical_pos,
                    context_anchor=anchor,
                    anchor_type=anchor_type,
                    target_count=3,
                    exclude_words=used_distractors,
                    definition=it.get("definition"),
                    quote=it.get("quote"),
                    return_metadata=True
                )

            # Ensure we have valid distractors; fallback to zero_collision if needed
            if len(distractors) < 3 and canonical_pos != "phrase":
                pos_char = "v" if canonical_pos == "verb" else ("a" if canonical_pos == "adj" else ("r" if canonical_pos == "adv" else "n"))
                distractors = cls.generate_zero_collision_distractors(w_lower, pos=pos_char, count=3)

            verb_requires_object = cls._verb_takes_object(w_lower) if canonical_pos == "verb" else None

            # Determine plural-noun target status before distractor screening so _is_valid_distractor
            # rejects uncountable or invariant-plural noun candidates before they enter the final 3 slots.
            is_plural_noun = False
            plural_target = w_lower
            if canonical_pos == "noun":
                is_spacy_plural = cls._is_genuine_plural_noun(w_lower)
                plural_tantum = {"goods", "customs", "clothes", "belongings", "surroundings", "fireworks", "premises", "congratulations"}
                locked_noun_gram = str((locked_sense or {}).get("grammar") or "").lower()
                is_pedlex_plural = bool(
                    re.search(r"\bplural\b", locked_noun_gram)
                    and not re.search(r"\b(?:countable|singular|uncountable)\b", locked_noun_gram)
                )

                quote_plural = False
                if it["quote"]:
                    try:
                        q_doc = nlp(it["quote"])
                        quote_plural = any(
                            (tok.lemma_.lower() == w_lower or tok.text.lower() == w_lower)
                            and tok.tag_ == "NNS"
                            and (tok.text.lower() != w_lower or is_spacy_plural or is_pedlex_plural)
                            for tok in q_doc
                        )
                    except Exception:
                        quote_plural = False

                # Do NOT pluralize via quote_plural if the locked atomic pattern/example uses the singular base headword
                # in a fixed singular construction (e.g. 'matter of routine' -> do not corrupt into 'matter of routines')
                # or if the locked sense's own examples exclusively attest the singular form and a singular pattern bundle exists.
                if quote_plural and initial_atomic_bundle:
                    init_pat = (initial_atomic_bundle.get("pattern") or "").lower()
                    init_ex = (initial_atomic_bundle.get("example") or "").lower()
                    p_cand = cls.pluralize_noun(w_lower)
                    if (
                        init_pat
                        and re.search(rf"\b(?:a|an|of|in|on|by|at|for|to|into|under|without)\s+{re.escape(w_lower)}\b", init_pat)
                        and not re.search(rf"\b{re.escape(p_cand)}\b", init_pat)
                    ):
                        if not init_ex or not re.search(rf"\b{re.escape(p_cand)}\b", init_ex):
                            quote_plural = False
                    elif locked_sense and (not init_pat or not re.search(rf"\b{re.escape(p_cand)}\b", init_pat)):
                        s_exs = [str(e).lower() for e in (locked_sense.get("examples") or []) if str(e).strip()]
                        if (
                            s_exs
                            and any(re.search(rf"\b{re.escape(w_lower)}\b", e) for e in s_exs)
                            and not any(re.search(rf"\b{re.escape(p_cand)}\b", e) for e in s_exs)
                        ):
                            quote_plural = False

                if is_spacy_plural or is_pedlex_plural or w_lower in plural_tantum or quote_plural:
                    is_plural_noun = True
                    if (is_spacy_plural or is_pedlex_plural or w_lower in plural_tantum) and w_lower.endswith("s"):
                        plural_target = w_lower
                    else:
                        plural_target = cls.pluralize_noun(w_lower)

            def _is_double_key(cand: str) -> bool:
                if canonical_pos == "function_word":
                    return False
                if is_quantifier_frame and quant_head_noun:
                    c_low_p = cand.strip().lower()
                    return cls.double_key_collision(quant_head_noun, c_low_p, anchor, anchor_type, pos="noun", quote=it.get("quote"))[0]
                if canonical_pos in ("phrase", "idiom_slot"):
                    c_low_p = cand.strip().lower()
                    return cls.double_key_collision(w_lower, c_low_p, pos="phrase", quote=it.get("quote"))[0]
                if anchor_type == "v_obj_pp" and v_obj_pp_tail:
                    c_low_v = cand.strip().lower()
                    return bool(
                        cls.lookup_ldoce_phrase(f"{c_low_v} {v_obj_pp_tail}")
                        or cls.lookup_ldoce_phrase(f"{c_low_v} something {v_obj_pp_tail}")
                    )
                flag, _kind = cls.double_key_collision(
                    w_lower, cand, anchor, anchor_type, pos=canonical_pos, quote=it.get("quote")
                )
                if flag:
                    return True
                if orig_disc_prep and orig_disc_prep != anchor:
                    flag_orig_p, _ = cls.double_key_collision(
                        w_lower, cand, orig_disc_prep, "prep", pos=canonical_pos, quote=it.get("quote")
                    )
                    if flag_orig_p:
                        return True
                raw_w_low = str(raw_w_item or "").strip().lower()
                if raw_w_low and raw_w_low != w_lower:
                    flag_raw, _ = cls.double_key_collision(
                        raw_w_low, cand, anchor, anchor_type, pos=canonical_pos, quote=it.get("quote")
                    )
                    if flag_raw:
                        return True
                ft_low = str(locals().get("final_target") or "").strip().lower()
                if ft_low and ft_low not in (w_lower, raw_w_low):
                    flag_ft, _ = cls.double_key_collision(
                        ft_low, cand, anchor, anchor_type, pos=canonical_pos, quote=it.get("quote")
                    )
                    if flag_ft:
                        return True
                if canonical_pos == "noun":
                    p_tw = cls.pluralize_noun(w_lower)
                    p_cand = cls.pluralize_noun(cand)
                    if p_tw != w_lower or p_cand != cand:
                        flag_p, _ = cls.double_key_collision(
                            p_tw, p_cand, anchor, anchor_type, pos=canonical_pos, quote=it.get("quote")
                        )
                        if flag_p:
                            return True
                return False

            def _is_valid_distractor(cand: str) -> bool:
                if not cand or _is_double_key(cand):
                    return False
                c_low = cand.strip().lower()
                if canonical_pos in ("function_word", "phrase", "idiom_slot"):
                    if c_low == w_lower:
                        return False
                    if canonical_pos == "phrase" and not is_quantifier_frame and (" " in w_lower) != (" " in c_low):
                        if anchor_type != "correl_pair" and w_lower not in cls._CORREL_FIRST_ARMS:
                            return False
                    return True
                if len(w_lower) >= 4 and len(c_low) >= 4:
                    if w_lower in c_low or c_low in w_lower:
                        if cls.is_cross_ontology_derivational_noun_pair(w_lower, c_low):
                            pass
                        elif cls.are_same_word_family(w_lower, c_low) or cls.shares_derivational_root(w_lower, c_low):
                            return False
                        else:
                            is_rhyme_containment_paronym = (
                                canonical_pos == "noun"
                                and min(len(w_lower), len(c_low)) >= 5
                                and abs(len(w_lower) - len(c_low)) <= 3
                                and (c_low.endswith(w_lower) or w_lower.endswith(c_low))
                                and cls._phonetic_rhyme_tail(w_lower) == cls._phonetic_rhyme_tail(c_low)
                            )
                            if not is_rhyme_containment_paronym and abs(len(w_lower) - len(c_low)) >= 2:
                                return False
                if (
                    is_plural_noun
                    and not cls.can_pluralize_noun_distractor(c_low)
                    and c_low not in set(cls._NOUN_COUNTABILITY_CONTRAST_MAP.get(w_lower, []))
                ):
                    return False

                if not cls._distractor_occupies_slot(
                    cand, canonical_pos, requires_object=(verb_requires_object is True)
                ):
                    return False
                return cls.is_cefr_compliant_distractor(cand, w_lower)

            def _clear_and_refill_distractors(curr_distractors: List[str], curr_meta: Dict[str, str], extra_exclude: Optional[Set[str]] = None) -> Tuple[List[str], Dict[str, str]]:
                near_s, ants_s = cls.semantic_fields(w_lower, pos=canonical_pos)
                near_only_s = near_s - ants_s
                excl_set = set(used_distractors) | set(extra_exclude or set())
                cleared_list: List[str] = []
                for d in curr_distractors:
                    if d in excl_set or not _is_valid_distractor(d):
                        continue
                    has_d_clash, _ = cls.evaluate_syntactic_slot_clash(
                        w_lower, d, anchor=anchor, anchor_type=anchor_type, pos=canonical_pos, quote=it.get("quote")
                    )
                    if (
                        not has_d_clash
                        and d.lower() in near_only_s
                        and sum(1 for x in cleared_list if x.lower() in near_only_s and curr_meta.get(x) != "antonym") >= 1
                    ):
                        continue
                    cleared_list.append(d)
                r_meta: Dict[str, str] = {}
                if len(cleared_list) < 3:
                    try:
                        if is_quantifier_frame and quant_head_noun:
                            refill_pool = [s for s in q_siblings_all if s != quant_head_noun and s not in excl_set and s not in cleared_list]
                            r_meta = {d: "quantifier_collocation_contrast" for d in refill_pool}
                        elif canonical_pos == "phrase":
                            refill_pool = cls.generate_phrase_distractors(
                                w_lower,
                                pos=raw_pos or canonical_pos,
                                count=8,
                                exclude_words=excl_set | set(cleared_list),
                                definition=it.get("definition"),
                                quote=it.get("quote"),
                            )
                            r_meta = {}
                        else:
                            refill_pool, r_meta = cls.generate_vocab_distractors(
                                target_word=w_lower, pos=canonical_pos,
                                context_anchor=anchor, anchor_type=anchor_type,
                                target_count=8, exclude_words=excl_set,
                                definition=it.get("definition"), quote=it.get("quote"),
                                return_metadata=True
                            )
                            r_meta = r_meta or {}
                    except Exception:
                        refill_pool = []
                    for cand in refill_pool:
                        if len(cleared_list) >= 3:
                            break
                        if cand and cand not in cleared_list and cand not in excl_set and _is_valid_distractor(cand):
                            has_c_clash, _ = cls.evaluate_syntactic_slot_clash(
                                w_lower, cand, anchor=anchor, anchor_type=anchor_type, pos=canonical_pos, quote=it.get("quote")
                            )
                            if (
                                not has_c_clash
                                and cand.lower() in near_only_s
                                and sum(1 for x in cleared_list if x.lower() in near_only_s and (curr_meta.get(x) or r_meta.get(x)) != "antonym") >= 1
                            ):
                                continue
                            cleared_list.append(cand)
                    if len(cleared_list) < 3 and canonical_pos != "phrase":
                        pos_char = "v" if canonical_pos == "verb" else ("a" if canonical_pos == "adj" else ("r" if canonical_pos == "adv" else "n"))
                        try:
                            for cand in cls.generate_zero_collision_distractors(w_lower, pos=pos_char, count=3):
                                if len(cleared_list) >= 3:
                                    break
                                if cand and cand not in cleared_list and cand not in excl_set and _is_valid_distractor(cand):
                                    if cand.lower() in near_only_s and sum(1 for x in cleared_list if x.lower() in near_only_s) >= 1:
                                        continue
                                    cleared_list.append(cand)
                        except Exception:
                            pass
                    if len(cleared_list) < 3:
                        pos_char = "v" if canonical_pos == "verb" else ("a" if canonical_pos == "adj" else ("r" if canonical_pos == "adv" else "n"))
                        needed = 3 - len(cleared_list)
                        pref_vowel: Optional[bool] = None
                        if canonical_pos == "noun" and not w_lower.endswith("s") and w_lower[:1].isalpha():
                            pref_vowel = w_lower[0].lower() in "aeiou"
                        fb_cands = cls.get_corpus_fallback_candidates(
                            target_word=w_lower,
                            pos=pos_char,
                            target_count=needed,
                            exclude_words=excl_set | set(cleared_list) | {w_lower},
                            prefer_initial_vowel=pref_vowel,
                        )
                        for cand in fb_cands:
                            if len(cleared_list) >= 3:
                                break
                            if cand != w_lower and cand not in cleared_list and cand not in excl_set and _is_valid_distractor(cand):
                                cleared_list.append(cand)
                    if len(cleared_list) < 3:
                        needed = 3 - len(cleared_list)
                        fb_cands = cls.get_corpus_fallback_candidates(
                            target_word=w_lower,
                            pos=pos_char,
                            target_count=needed,
                            exclude_words=excl_set | set(cleared_list) | {w_lower},
                            prefer_initial_vowel=None,
                        )
                        for cand in fb_cands:
                            if len(cleared_list) >= 3:
                                break
                            if cand != w_lower and cand not in cleared_list and cand not in excl_set and not _is_double_key(cand):
                                cleared_list.append(cand)
                merged_meta = {
                    d: (curr_meta.get(d) or r_meta.get(d) or "distant_contrast")
                    for d in cleared_list
                }
                return cleared_list, merged_meta

            distractors, dist_meta = _clear_and_refill_distractors(distractors, dist_meta)

            # Assemble full options (target + 3 distractors)
            distractors3 = distractors[:3]
            slot_target = quant_head_noun if is_quantifier_frame else w_lower
            slot = zlib.crc32(slot_target.encode("utf-8")) % 4
            raw_options = distractors3[:slot] + [slot_target] + distractors3[slot:]
            correct_answer_index = slot
            used_distractors.update(d for d in distractors3 if d)

            def _pluralize_or_keep_uncountable(d_word: str) -> str:
                if (
                    dist_meta.get(d_word) == "countability_contrast_uncountable"
                    or d_word in set(cls._NOUN_COUNTABILITY_CONTRAST_MAP.get(w_lower, []))
                ):
                    return d_word
                return cls.pluralize_noun(d_word)

            # 4. Plan A: All options and target word maintain parallel inflection
            if is_plural_noun:
                inflection_desc = "plural form (NNS)"
                final_target = plural_target
                inflected_distractors3 = [_pluralize_or_keep_uncountable(d) for d in distractors3]
                inflected_distractors3 = cls.screen_inflected_options(
                    inflected_distractors3,
                    [_pluralize_or_keep_uncountable(d) for d in distractors[3:]],
                    final_target,
                )
                used_distractors.update(inflected_distractors3)
                slot = min(slot, len(inflected_distractors3))
                correct_answer_index = slot
                final_options = inflected_distractors3[:slot] + [final_target] + inflected_distractors3[slot:]
            elif is_quantifier_frame:
                inflection_desc = "singular quantifier noun"
                final_target = quant_head_noun
                final_options = raw_options
            else:
                inflection_desc = "base form"
                final_target = w_lower
                final_options = raw_options

            # 5. Sense-First Grounding & Synthesize Itemized Micro-Task for LLM
            dist_list = [opt for opt in final_options if opt.lower() != final_target.lower()]
            dist_str = ", ".join(dist_list)

            # 5.1 Re-verify PedLex Entry & Sense if final_target differs and Step 0 had no entry
            if locked_sense is None:
                entry_final = cls.get_ldoce_entry(final_target)
                if entry_base and cls._pos_code(entry_base.get("pos")) == target_p_code:
                    target_entry = entry_base
                elif entry_final and cls._pos_code(entry_final.get("pos")) == target_p_code:
                    target_entry = entry_final
                else:
                    target_entry = entry_base or entry_final

                if target_entry and target_entry.get("senses"):
                    t_s_idx = cls._lock_sense(
                        target_entry,
                        definition=it.get("definition"),
                        quote=it.get("quote") or it.get("raw_quote"),
                        target_pos=canonical_pos
                    )
                    if t_s_idx is not None and 0 <= t_s_idx < len(target_entry["senses"]):
                        locked_sense = target_entry["senses"][t_s_idx]

            # Hydrate authoritative formatted definition (with [spoken], [formal], [variety] labels).
            # If it["definition"] was already extracted/disambiguated and is non-boilerplate,
            # preserve it whenever locked_sense is absent or represents a different/fallback sense.
            ext_defn = (it.get("definition") or "").strip()
            locked_defn = cls.format_ldoce_definition(locked_sense) if locked_sense else ""
            if ext_defn and not cls.is_boilerplate_definition(ext_defn):
                ext_norm = re.sub(r"^\s*(?:\[[^\]]+\]\s*)+", "", ext_defn).strip().lower()
                lock_norm = re.sub(r"^\s*(?:\[[^\]]+\]\s*)+", "", locked_defn).strip().lower()
                if locked_defn and ext_norm == lock_norm:
                    skeleton_definition = locked_defn
                elif locked_defn and canonical_pos not in ("phrase", "idiom_slot") and not m_caus_adj:
                    skeleton_definition = locked_defn
                else:
                    skeleton_definition = ext_defn
            else:
                skeleton_definition = locked_defn or ext_defn
            if skeleton_definition and final_target:
                skeleton_definition = re.sub(
                    rf"^\s*{re.escape(final_target)}\s*:\s*",
                    "",
                    skeleton_definition,
                    flags=re.IGNORECASE,
                ).strip()

            # 5.2 Atomic Sense Pattern Bundle (Pattern + Corpus Example + Bound Preposition as an indivisible bundle)
            anchor_before_step5 = anchor
            atomic_bundle = cls.pick_atomic_pattern_bundle(
                locked_sense,
                final_target,
                quote=it.get("quote") or it.get("raw_quote"),
                preferred_anchor=anchor
            )
            if atomic_bundle:
                best_pattern = atomic_bundle.get("pattern")
                if atomic_bundle.get("example"):
                    ldoce_pattern_example = atomic_bundle["example"]
                if atomic_bundle.get("verb_anchor") and anchor_type not in _cxn_anchor_types:
                    anchor = atomic_bundle["verb_anchor"]
                    anchor_type = "verb"
                    pattern_anchor = anchor
                elif atomic_bundle.get("prep") and anchor_type not in _cxn_anchor_types:
                    pat_prep = atomic_bundle["prep"]
                    anchor_in_bundle_ex = bool(
                        anchor
                        and ldoce_pattern_example
                        and re.search(rf"\b{re.escape(str(anchor).lower())}\w*\b", ldoce_pattern_example.lower())
                    )
                    if (not anchor or anchor_type in ("collocation", "contextual")) and not (
                        canonical_pos == "adj" and quote_adj_is_attributive and pat_prep != quote_governed_prep
                    ):
                        if not any(cls.double_key_collision(final_target, d, anchor=pat_prep, anchor_type="prep", pos=canonical_pos)[0] for d in dist_list):
                            anchor = pat_prep
                            anchor_type = "prep"
                            pattern_anchor = pat_prep
                    elif (
                        not anchor_in_bundle_ex
                        and anchor_type in ("adj", "verb", "object", "modified_noun", "verb_subject")
                        and not (
                            canonical_pos == "verb"
                            and (quote_is_transitive or anchor_type in ("object", "verb_subject"))
                            and pat_prep != quote_governed_prep
                        )
                        and not (
                            canonical_pos == "adj"
                            and (quote_adj_is_attributive or anchor_type in ("modified_noun", "verb_copula"))
                            and pat_prep != quote_governed_prep
                        )
                    ):
                        # The locked sense's atomic pattern and corpus example test a bound prepositional frame
                        # (e.g., 'on benefit' -> 'families on benefit') rather than the quote's collocation modifier;
                        # synchronize anchor to `pat_prep` so Step 5 re-verifies distractors against `on ____`.
                        anchor = pat_prep
                        anchor_type = "prep"
                        pattern_anchor = pat_prep
            if causative_extraposition_pat:
                best_pattern = causative_extraposition_pat
                if causative_extraposition_ex:
                    ldoce_pattern_example = causative_extraposition_ex
            elif cxn_override_pat:
                best_pattern = cxn_override_pat
                if canonical_pos == "adj" and anchor_type == "it_extrapos" and locked_sense and not ldoce_pattern_example:
                    _q_norm_ex = (it.get("quote") or "").strip().lower()
                    for _pb in (locked_sense.get("pattern_bundles") or []):
                        if not isinstance(_pb, dict):
                            continue
                        _pb_ex = str(_pb.get("example") or "").strip()
                        if _pb_ex and _pb_ex.lower() != _q_norm_ex and re.search(
                            rf"\bit\s*(?:is|was|seems|appears|'s|’s|would\s+be|will\s+be)\b[^.?!;]*?\b{re.escape(final_target)}\b[^.?!;]*?\b{re.escape(str(anchor or 'to'))}\b",
                            _pb_ex.lower(),
                        ):
                            ldoce_pattern_example = _pb_ex
                            break
                    if not ldoce_pattern_example:
                        for _raw_ex in (locked_sense.get("examples") or []):
                            _ex_txt = (_raw_ex.get("text") if isinstance(_raw_ex, dict) else str(_raw_ex or "")).strip()
                            if _ex_txt and _ex_txt.lower() != _q_norm_ex and re.search(
                                rf"\bit\s*(?:is|was|seems|appears|'s|’s|would\s+be|will\s+be)\b[^.?!;]*?\b{re.escape(final_target)}\b[^.?!;]*?\b{re.escape(str(anchor or 'to'))}\b",
                                _ex_txt.lower(),
                            ):
                                ldoce_pattern_example = _ex_txt
                                break

            quote_shown_prep = bool(anchor) and anchor_type == "prep" and bool(
                quote_governed_prep and str(anchor).lower() == quote_governed_prep.lower()
            )
            quote_shown_partitive = bool(anchor) and anchor_type == "partitive" and bool(re.search(
                rf"\b{re.escape(str(anchor).lower())}\b", (it.get("quote") or "").lower()))

            if anchor and anchor_type != "contextual" and anchor_type not in _cxn_anchor_types and not ldoce_pattern_example:
                lookup_anchor = prep_obj if anchor_type == "partitive" else anchor
                ldoce_pattern_example = cls.ldoce_example_for_anchor(
                    final_target,
                    lookup_anchor,
                    locked_sense=locked_sense,
                    definition=it.get("definition"),
                    quote=it.get("quote"),
                    pos=canonical_pos
                )
                pattern_anchor = anchor
            has_atomic_valency_ex = bool(
                atomic_bundle
                and (atomic_bundle.get("inf_valency") or atomic_bundle.get("gerund_valency"))
                and ldoce_pattern_example
            )
            if anchor_type not in _cxn_anchor_types and anchor_type not in ("verb_subject", "discourse") and not has_atomic_valency_ex and (
                    not anchor or anchor_type == "contextual"
                    or (not ldoce_pattern_example and not quote_shown_prep and not quote_shown_partitive)):
                ld_anchor, ld_atype, ld_fdesc, ld_ex = cls.find_ldoce_zero_collision_anchor(
                    target_word=final_target,
                    pos=canonical_pos,
                    distractors=dist_list,
                    definition=it.get("definition"),
                    quote=it.get("quote"),
                    locked_sense=locked_sense
                )
                if ld_anchor and not (
                    canonical_pos == "verb"
                    and (quote_is_transitive or anchor_type == "object")
                    and ld_atype == "prep"
                    and ld_anchor != quote_governed_prep
                ) and not (
                    canonical_pos == "adj"
                    and (quote_adj_is_attributive or anchor_type in ("modified_noun", "verb_copula"))
                    and ld_atype == "prep"
                    and ld_anchor != quote_governed_prep
                ):
                    anchor = ld_anchor
                    anchor_type = ld_atype
                    ldoce_pattern_example = ld_ex
                    pattern_anchor = ld_anchor

            # Batch Anchor Collision Gate 
            if batch_target_words and not is_idiom_slot and anchor_type not in _cxn_anchor_types and (anchor or prep_obj):
                if prep_obj and cls._in_forbidden_word_set(prep_obj, batch_target_words):
                    prep_obj = None
                if (anchor and anchor_type == "partitive"
                        and cls._in_forbidden_word_set(
                            prep_obj or (anchor.split()[0] if anchor.split() else ""),
                            batch_target_words)):
                    anchor, anchor_type = None, "collocation"
                    prep_obj = None
                if anchor and cls._in_forbidden_word_set(anchor, batch_target_words):
                    anchor, anchor_type = cls._batch_safe_reanchor(
                        final_target,
                        canonical_pos,
                        definition=it.get("definition"),
                        quote=it.get("quote"),
                        forbidden=batch_target_words
                    )
                    if not anchor_type:
                        anchor_type = "collocation"
                    ldoce_pattern_example = None
                    ldoce_cloze_frame = None

            # Locked-Sense Frame Concordance Gate 
            if anchor_type == "prep" and anchor and not is_idiom_slot and cls._prep_anchor_sense_conflict(
                final_target,
                anchor,
                definition=it.get("definition"),
                quote=it.get("quote"),
                canonical_pos=canonical_pos
            ):
                alt_anchor, alt_type, _alt_frame, alt_ex = cls.find_ldoce_zero_collision_anchor(
                    target_word=final_target,
                    pos=canonical_pos,
                    distractors=dist_list,
                    definition=it.get("definition"),
                    quote=it.get("quote"),
                    locked_sense=locked_sense
                )
                if alt_anchor and not cls._prep_anchor_sense_conflict(
                    final_target,
                    alt_anchor,
                    definition=it.get("definition"),
                    quote=it.get("quote"),
                    canonical_pos=canonical_pos
                ):
                    anchor, anchor_type = alt_anchor, (alt_type or "collocation")
                    ldoce_pattern_example = alt_ex
                    prep_obj = None
                else:
                    anchor, anchor_type = None, "collocation"
                    prep_obj = None
                    ldoce_pattern_example = None
                ldoce_cloze_frame = None

            # If Step 5 re-anchoring changed the anchor after Step 3 (including clearing anchor to None),
            # re-verify distractors against the final anchor so zero-double-key invariance holds 100%.
            if anchor != anchor_before_step5 and canonical_pos not in ("phrase", "idiom_slot", "function_word"):
                if any(_is_double_key(d) for d in distractors[:3]):
                    distractors, dist_meta = _clear_and_refill_distractors(distractors, dist_meta)
                    distractors3 = distractors[:3]
                    raw_options = distractors3[:slot] + [slot_target] + distractors3[slot:]
                    if is_plural_noun:
                        inflected_distractors3 = cls.screen_inflected_options(
                            [_pluralize_or_keep_uncountable(d) for d in distractors3],
                            [_pluralize_or_keep_uncountable(d) for d in distractors[3:]],
                            final_target,
                        )
                        used_distractors.update(inflected_distractors3)
                        final_options = inflected_distractors3[:slot] + [final_target] + inflected_distractors3[slot:]
                    else:
                        used_distractors.update(distractors3)
                        final_options = raw_options
                    dist_list = [opt for opt in final_options if opt.lower() != final_target.lower()]
                    dist_str = ", ".join(dist_list)

            # If a re-anchoring produced a new anchor, look up the PedLex sentence that
            # actually demonstrates this (target, anchor) pair within the locked sense.
            if anchor and not ldoce_pattern_example and anchor != pattern_anchor and anchor_type not in _cxn_anchor_types:
                ldoce_pattern_example = cls.ldoce_example_for_anchor(
                    final_target,
                    prep_obj if anchor_type == "partitive" else anchor,
                    locked_sense=locked_sense,
                    definition=it.get("definition"),
                    quote=it.get("quote"),
                    pos=canonical_pos
                )

            # Fallback to the locked sense's own example if no anchor-specific example survived
            if not ldoce_pattern_example:
                ldoce_pattern_example = cls.ldoce_sense_example(
                    final_target,
                    pos=canonical_pos,
                    definition=it.get("definition"),
                    quote=it.get("quote"),
                    locked_sense=locked_sense
                )

            # Fallback to upstream-verified example_usage (or expression_definition_evidence for phrases
            # stored in PedLex sub-blocks/run-ons rather than top-level headwords)
            if not ldoce_pattern_example:
                upstream_ex = (it.get("example_usage") or "").strip().strip('"').strip()
                if upstream_ex:
                    ldoce_pattern_example = upstream_ex
                elif canonical_pos in ("phrase", "idiom_slot"):
                    _, expr_ex, _ = cls.expression_definition_evidence(
                        raw_w_item or final_target,
                        raw_pos,
                        context_sentence=it.get("quote")
                    )
                    if expr_ex:
                        ldoce_pattern_example = expr_ex.strip().strip('"').strip()

            # Co-variant Example Guard for Multi-Word Expressions:
            # If final_target is a multi-word expression (e.g. 'date back to' / 'date back', 'talk of') sharing a PedLex block
            # with a co-variant ('date from', 'talk ... out of'), ensure ldoce_pattern_example actually instantiates final_target
            # without inserting an unrelated particle/preposition.
            if ldoce_pattern_example and " " in final_target:
                ft_toks = [
                    t for t in re.findall(r"[a-z]+", final_target.lower())
                    if t not in {"s", "sb", "sth", "somebody", "something", "one", "ones", "oneself", "your", "his", "her", "their", "my", "our", "its"}
                ]
                if len(ft_toks) >= 2:
                    def _mw_ex_matches(cand_s: str) -> bool:
                        c_low = (cand_s or "").lower()
                        req_parts = ft_toks
                        if (
                            len(ft_toks) == 3
                            and ft_toks[0] in ("be", "get", "feel", "am", "is", "are", "was", "were", "been", "being")
                            and ft_toks[2] in ("with", "in", "to", "of", "on", "for", "from", "into", "at", "by", "about", "as", "against", "under", "over")
                        ):
                            if not cls.text_contains_form(cand_s, ft_toks[0]):
                                _v_lem_p = cls._surface_lemma(ft_toks[1]) or cls.lemma_of(ft_toks[1]) or ft_toks[1]
                                req_parts = [_v_lem_p, ft_toks[2]]
                        if not all(cls.text_contains_form(cand_s, tk) for tk in req_parts):
                            return False
                        c_toks = re.findall(r"[a-z]+", c_low)
                        f_set = {req_parts[0]} | set(cls.inflected_forms(req_parts[0]))
                        l_set = {req_parts[-1]} | set(cls.inflected_forms(req_parts[-1]))
                        _extra_p = {
                            "out", "in", "up", "down", "off", "on", "away", "back", "over", "into",
                            "from", "with", "for", "to", "of", "at", "by", "about", "against", "through", "around", "aside", "apart"
                        } - set(req_parts)
                        for i_f, tk_f in enumerate(c_toks):
                            if tk_f in f_set:
                                for i_l in range(i_f + 1, min(len(c_toks), i_f + len(req_parts) + 4)):
                                    if c_toks[i_l] in l_set:
                                        if not ((set(c_toks[i_f + 1 : i_l]) - set(req_parts)) & _extra_p):
                                            return True
                        return False

                    if not _mw_ex_matches(ldoce_pattern_example):
                        replacement_ex = None
                        upstream_ex_cand = (it.get("example_usage") or "").strip().strip('"').strip()
                        if (
                            upstream_ex_cand
                            and upstream_ex_cand.lower() != (it.get("quote") or "").strip().lower()
                            and _mw_ex_matches(upstream_ex_cand)
                        ):
                            replacement_ex = upstream_ex_cand
                        if not replacement_ex:
                            for raw_ex in ((locked_sense or {}).get("examples") or []):
                                ex_txt = (raw_ex.get("text") if isinstance(raw_ex, dict) else str(raw_ex or "")).strip()
                                if ex_txt and ex_txt.lower() != (it.get("quote") or "").strip().lower():
                                    if _mw_ex_matches(ex_txt):
                                        replacement_ex = ex_txt
                                        break
                        if not replacement_ex:
                            corpus_hits = cls.search_corpus_examples(final_target, limit=5)
                            for ch in (corpus_hits or []):
                                if ch.strip().lower() != (it.get("quote") or "").strip().lower():
                                    if raw_pos in ("phrasal verb", "verb") and re.search(
                                        rf"\b(?:a|an|the|one|major|key|main|important|crucial|decisive)\s+{re.escape(ft_toks[0])}\b",
                                        ch.lower(),
                                    ):
                                        continue
                                    if _mw_ex_matches(ch):
                                        replacement_ex = ch
                                        break
                        ldoce_pattern_example = replacement_ex

            # Pre-compute syntactic slot flags (copular/attributive adjective & noun countability)
            # so Five-Source Optimal Example Pre-Selection and Direct Example-to-Frame Slicing honor them.
            is_copula_adj = False
            is_attributive_only_adj = False
            if canonical_pos == "adj":
                adj_sense_gram = str((locked_sense or {}).get("grammar") or "").lower()
                adj_gram_prof = cls.get_word_gram_profile(w_lower, pos="adj")
                if (
                    "not before noun" in adj_sense_gram
                    or "never before noun" in adj_sense_gram
                    or bool(adj_gram_prof.get("not_before_noun"))
                ):
                    is_copula_adj = True
                elif "only before noun" in adj_sense_gram or bool(adj_gram_prof.get("only_before_noun")):
                    is_attributive_only_adj = True
                if not is_copula_adj and not is_attributive_only_adj and distractors:
                    for d_cand in distractors:
                        d_clash, d_reason = cls.evaluate_syntactic_slot_clash(
                            w_lower, d_cand, anchor=anchor, anchor_type=anchor_type, pos="adj", quote=it.get("quote")
                        )
                        if d_clash and d_reason == "adj_not_before_noun_in_attributive_slot" and anchor_type != "prep":
                            is_attributive_only_adj = True
                            break
                        if d_clash and d_reason == "adj_only_before_noun_in_predicative_slot":
                            is_copula_adj = True
                            break

            # 5.3 Five-Source Optimal Example Pre-Selection (GEMINI.md §3.2):
            # Evaluate pure 5 LDOCE compartments (grammar_boxes, pattern_bundles & sense_examples,
            # LDOCE Thesaurus, Language Activator, curated Collocations/cross_collocations) to select
            # the single best, sense-isolated, complete authentic sentence before deriving cloze_slot_frame.
            # (Oxford Thesaurus and raw BNC fragments permanently excluded).
            winning_example_source = "none"
            _shell_noun_anchors = ("shell_to_do", "shell_of_doing", "shell_that")
            _verb_cxn_anchors = ("sb_to_do", "to_inf", "gerund")
            if not causative_extraposition_pat and (
                (not cxn_override_pat and anchor_type not in _cxn_anchor_types)
                or (canonical_pos == "noun" and anchor_type in _shell_noun_anchors)
                or (canonical_pos == "verb" and anchor_type in _verb_cxn_anchors)
            ):
                opt_ex, opt_pat, winning_example_source = cls.select_optimal_five_source_example(
                    target_entry=target_entry,
                    locked_sense=locked_sense,
                    t_s_idx=t_s_idx,
                    final_target=final_target,
                    base_word=str((target_entry or {}).get("word") or w_lower).strip().lower(),
                    canonical_pos=canonical_pos,
                    definition=skeleton_definition,
                    quote=it.get("quote") or it.get("raw_quote"),
                    anchor=anchor,
                    anchor_type=anchor_type,
                    best_pattern=best_pattern,
                    current_example=ldoce_pattern_example,
                    is_copula_adj=is_copula_adj,
                    is_attributive_only_adj=is_attributive_only_adj,
                )
                if opt_ex:
                    ldoce_pattern_example = opt_ex
                if not cxn_override_pat and anchor_type not in _cxn_anchor_types:
                    best_pattern = opt_pat
                elif opt_pat and not best_pattern:
                    best_pattern = opt_pat

            # Synchronize `modified_noun` anchor to the syntactic partner in `ldoce_pattern_example` when `anchor`
            # is absent from the winning example AND came from:
            #   (1) a generic light noun in the quote ('diplomatic move'),
            #   (2) an unattested dictionary fallback absent from the quote ('outdated' -> 'method'),
            #   (3) a non-attributive quote fallback ('analogous' -> 'sponsorship'), or
            #   (4) a sense whose winning example instantiates a bound prepositional frame ('silent on').
            if (
                canonical_pos == "adj"
                and anchor_type == "modified_noun"
                and anchor
                and ldoce_pattern_example
                and not re.search(rf"\b{re.escape(str(anchor).lower())}\w*\b", ldoce_pattern_example.lower())
            ):
                try:
                    _sync_doc = nlp(ldoce_pattern_example)
                    _, _, _sync_tok = cls._find_target_span_in_doc(
                        _sync_doc,
                        final_target.lower(),
                        base_str=str((target_entry or {}).get("word") or w_lower).strip().lower(),
                        t_forms=target_inflections_set,
                    )
                    if _sync_tok is not None:
                        _ex_partner, _ex_ptype, _ = cls._example_partner(_sync_tok, final_target, "adj")
                        if _ex_ptype == "prep" and _ex_partner not in attested_adj_preps:
                            _ex_partner, _ex_ptype = None, None
                            if _sync_tok.head and _sync_tok.head.pos_ in ("AUX", "VERB"):
                                for _ch in _sync_tok.head.children:
                                    if _ch.dep_ in ("nsubj", "nsubjpass") and _ch.pos_ in ("NOUN", "PROPN"):
                                        _c_lem = (_ch.lemma_ or _ch.text).lower()
                                        if len(_c_lem) >= 3 and _c_lem not in cls._ANCHOR_STOPWORDS and _c_lem != final_target.lower():
                                            _ex_partner, _ex_ptype = _c_lem, "modified_noun"
                                            break
                        _should_sync_adj_anchor = (
                            not quote_adj_is_attributive
                            or _ex_ptype == "prep"
                            or str(anchor).strip().lower() in {"move", "thing", "one", "ones", "way", "part", "place", "time", "point", "case", "type", "kind", "sort"}
                            or not re.search(rf"\b{re.escape(str(anchor).lower())}\w*\b", str(it.get("quote") or "").lower())
                        )
                        if (
                            _should_sync_adj_anchor
                            and _ex_partner
                            and _ex_ptype in ("modified_noun", "prep", "verb_copula")
                            and not cls._in_forbidden_word_set(_ex_partner, batch_target_words)
                        ):
                            anchor = _ex_partner
                            anchor_type = _ex_ptype
                            if _ex_ptype == "prep" and not best_pattern:
                                best_pattern = f"{final_target.lower()} {_ex_partner}"
                            if any(
                                cls.double_key_collision(final_target, d, anchor=anchor, anchor_type=anchor_type, pos="adj")[0]
                                for d in dist_list
                            ):
                                distractors, dist_meta = _clear_and_refill_distractors(distractors, dist_meta)
                                distractors3 = distractors[:3]
                                raw_options = distractors3[:slot] + [slot_target] + distractors3[slot:]
                                used_distractors.update(distractors3)
                                final_options = raw_options
                                dist_list = [opt for opt in final_options if opt.lower() != final_target.lower()]
                                dist_str = ", ".join(dist_list)
                except Exception:
                    pass

            # Synchronize adverb `anchor` & `anchor_type` to the syntactic partner in `ldoce_pattern_example`
            # whenever the winning example instantiates a concrete modified verb/adjective different from (or filling in) `anchor`.
            if (
                canonical_pos == "adv"
                and anchor_type != "discourse"
                and ldoce_pattern_example
                and (
                    not anchor
                    or not re.search(rf"\b{re.escape(str(anchor).lower())}\w*\b", ldoce_pattern_example.lower())
                )
            ):
                try:
                    _sync_adv_doc = nlp(ldoce_pattern_example)
                    _, _, _sync_adv_tok = cls._find_target_span_in_doc(
                        _sync_adv_doc,
                        final_target.lower(),
                        base_str=final_target.lower(),
                        t_forms={final_target.lower()},
                    )
                    if _sync_adv_tok is not None:
                        _adv_partner, _adv_ptype, _ = cls._example_partner(_sync_adv_tok, final_target, "adv")
                        if (
                            _adv_partner
                            and _adv_ptype in ("modifies_adj", "modifies_verb")
                            and not cls._in_forbidden_word_set(_adv_partner, batch_target_words)
                        ):
                            anchor = _adv_partner
                            anchor_type = _adv_ptype
                except Exception:
                    pass

            # Deep Passive & Transitive Phrasal-Verb Object Anchor Synchronization:
            # When a verbal phrase or transitive verb has no anchor yet (or only a contextual quote fallback absent from the example)
            # and `ldoce_pattern_example` instantiates a concrete direct object (`dobj`) or passive Patient (`nsubjpass`),
            # project that Patient noun as `anchor` (`anchor_type = "object"`) so the LLM prompt locks onto the concrete Patient ('受动者').
            if (
                canonical_pos in ("verb", "phrase")
                and anchor_type not in _cxn_anchor_types
                and ldoce_pattern_example
                and (
                    not anchor
                    or anchor_type in ("contextual", "collocation")
                    or (
                        anchor_type in ("prep", "object")
                        and not re.search(rf"\b{re.escape(str(anchor).lower())}\w*\b", ldoce_pattern_example.lower())
                    )
                )
            ):
                try:
                    _sync_v_doc = nlp(ldoce_pattern_example)
                    _head_v_tok = None
                    if " " in final_target.lower():
                        _ft_words = [t for t in re.findall(r"[a-z]+", final_target.lower()) if t not in ("sb", "sth", "somebody", "something", "one", "ones")]
                        if _ft_words:
                            _f_forms = {_ft_words[0]} | set(cls.inflected_forms(_ft_words[0]))
                            _l_w = _ft_words[-1]
                            for _tk in _sync_v_doc:
                                if (_tk.text.lower() in _f_forms or _tk.lemma_.lower() == _ft_words[0]) and _tk.pos_ in ("VERB", "AUX"):
                                    if any(
                                        _sync_v_doc[_j].text.lower() == _l_w or _sync_v_doc[_j].lemma_.lower() == _l_w
                                        for _j in range(_tk.i + 1, min(len(_sync_v_doc), _tk.i + len(_ft_words) + 4))
                                    ):
                                        _head_v_tok = _tk
                                        break
                    else:
                        _, _, _head_v_tok = cls._find_target_span_in_doc(
                            _sync_v_doc,
                            final_target.lower(),
                            base_str=str((target_entry or {}).get("word") or w_lower).strip().lower(),
                            t_forms=target_inflections_set,
                        )
                    if _head_v_tok is not None and _head_v_tok.pos_ in ("VERB", "AUX"):
                        _is_v_pass = any(c.dep_ in ("auxpass", "nsubjpass") for c in _head_v_tok.children)
                        _pat_obj_tok = next(
                            (
                                c for c in _head_v_tok.children
                                if (c.dep_ in ("dobj", "obj") or (_is_v_pass and c.dep_ == "nsubjpass"))
                                and c.pos_ in ("NOUN", "PROPN")
                                and c.ent_type_ != "PERSON"
                                and c.lemma_.lower() not in cls._ANCHOR_STOPWORDS
                                and not cls._in_forbidden_word_set(c.lemma_.lower(), batch_target_words)
                            ),
                            None,
                        )
                        if _pat_obj_tok is not None:
                            anchor = _pat_obj_tok.text.lower() if _pat_obj_tok.tag_ == "NNS" else _pat_obj_tok.lemma_.lower()
                            anchor_type = "object"
                except Exception:
                    pass

            # Final Atomic Pattern-Anchor-Example Synchronization (GEMINI.md §1.10):
            # Ensure best_pattern never claims a preposition that clashes with final anchor or is absent from ldoce_pattern_example.
            if best_pattern and anchor_type not in _cxn_anchor_types and not causative_extraposition_pat and not cxn_override_pat:
                p_clean_sync = re.sub(r"\[.*?\]|\(.*?\)", "", best_pattern).strip().lower()
                p_for_prep_sync = re.sub(r"\bto\s+(?:do(?:\s+something)?|\w+ing)\b", "", p_clean_sync)
                pat_preps_sync = re.findall(
                    r"\b(on|in|at|for|with|about|under|from|to|by|into|of|between|over|against|towards|upon)\b",
                    p_for_prep_sync
                )
                if pat_preps_sync:
                    if anchor and anchor_type == "prep" and anchor.lower() not in pat_preps_sync:
                        # Re-select a sense pattern that matches the final anchor, or clear best_pattern
                        re_bundle = cls.pick_atomic_pattern_bundle(
                            locked_sense, final_target, quote=it.get("quote") or it.get("raw_quote"), preferred_anchor=anchor
                        )
                        if re_bundle and re_bundle.get("pattern") and anchor.lower() in (re_bundle.get("preps") or []):
                            re_ex = re_bundle.get("example")
                            if re_ex and (
                                cls._is_complete_finite_sentence(re_ex)
                                or not cls._is_complete_finite_sentence(ldoce_pattern_example)
                            ):
                                best_pattern = re_bundle["pattern"]
                                ldoce_pattern_example = re_ex
                            elif ldoce_pattern_example and cls._example_attests_prep_valency(
                                ldoce_pattern_example, final_target, anchor.lower(), prep_dir="any"
                            ):
                                best_pattern = re_bundle["pattern"]
                            else:
                                best_pattern = None
                        else:
                            best_pattern = None
                    elif ldoce_pattern_example:
                        if canonical_pos == "noun":
                            _p_dir = "pre" if any(p_for_prep_sync.startswith(pp + " ") for pp in pat_preps_sync) else "post"
                            if not any(
                                cls._example_attests_prep_valency(ldoce_pattern_example, final_target, pp, prep_dir=_p_dir)
                                for pp in pat_preps_sync
                            ):
                                best_pattern = None
                        elif not any(re.search(rf"\b{re.escape(pp)}\b", ldoce_pattern_example.lower()) for pp in pat_preps_sync):
                            best_pattern = None

            # Derive Authentic Cloze Sentence Frame Prototype if an authentic example is available
            if ldoce_pattern_example:
                try:
                    ex_doc = nlp(ldoce_pattern_example)
                    ex_s_i, ex_e_i, _ = cls._find_target_span_in_doc(
                        ex_doc,
                        final_target.lower(),
                        base_str=str((target_entry or {}).get("word") or w_lower).strip().lower(),
                        t_forms=target_inflections_set,
                    )
                    if ex_s_i is not None and ex_e_i is not None:
                        tok_texts = (
                            [t.text for t in ex_doc[:ex_s_i]]
                            + ["____"]
                            + [t.text for t in ex_doc[ex_e_i:]]
                        )
                        ldoce_cloze_frame = " ".join(tok_texts)
                except Exception:
                    ldoce_cloze_frame = None

            if anchor_type == "partitive" and anchor and ldoce_cloze_frame:
                quant_lem = (prep_obj or (anchor.split()[0] if anchor.split() else "")).lower()
                if quant_lem and not re.search(rf"\b{re.escape(quant_lem)}\w*\b", ldoce_cloze_frame.lower()):
                    ldoce_cloze_frame = None

            # Prescribed-Inflection Concordance 
            _base_options = list(final_options)
            _orig_inflection_desc = inflection_desc
            _reconcile_example = ldoce_pattern_example
            if (
                canonical_pos == "verb"
                and quote_shown_prep
                and _reconcile_example
                and re.search(rf"\b{re.escape(final_target.lower())}\b", (it.get("quote") or "").lower())
                and not re.search(rf"\b{re.escape(final_target.lower())}\b", _reconcile_example.lower())
            ):
                _reconcile_example = None
            inflection_desc, final_options, final_target, verb_form_tag = cls._reconcile_prescribed_inflection(
                base_options=_base_options,
                target_index=correct_answer_index,
                target_word=final_target,
                canonical_pos=canonical_pos,
                inflection_desc=_orig_inflection_desc,
                authentic_example=_reconcile_example,
                quote=it.get("quote") or it.get("raw_quote"),
            )
            eff_pos = "noun" if is_quantifier_frame else canonical_pos
            has_double_key_collision = (
                canonical_pos != "function_word"
                and any(
                    opt.lower() != final_target.lower() and cls.double_key_collision(
                        final_target, opt, anchor, anchor_type, pos=eff_pos, quote=it.get("quote")
                    )[0]
                    for opt in final_options
                )
            )
            if has_double_key_collision:
                bad_bases = {
                    b for b, f in zip(_base_options, final_options)
                    if f.lower() != final_target.lower() and cls.double_key_collision(
                        final_target, f, anchor, anchor_type, pos=eff_pos, quote=it.get("quote")
                    )[0]
                }
                filtered_base_dists = [d for d in distractors if d not in bad_bases]
                distractors, dist_meta = _clear_and_refill_distractors(filtered_base_dists, dist_meta, extra_exclude=bad_bases)
                distractors3 = distractors[:3]
                _base_options = distractors3[:slot] + [slot_target] + distractors3[slot:]
                inflection_desc, final_options, final_target, verb_form_tag = cls._reconcile_prescribed_inflection(
                    base_options=_base_options,
                    target_index=correct_answer_index,
                    target_word=slot_target,
                    canonical_pos=canonical_pos,
                    inflection_desc=_orig_inflection_desc,
                    authentic_example=ldoce_pattern_example,
                    quote=it.get("quote") or it.get("raw_quote"),
                )
            if final_options != _base_options or has_double_key_collision:
                _form_remap = {b: f for b, f in zip(_base_options, final_options)}
                dist_meta = {_form_remap.get(k, k): v for k, v in dist_meta.items()}
                used_distractors.update(o for o in final_options if o.lower() != final_target.lower())
                dist_list = [opt for opt in final_options if opt.lower() != final_target.lower()]
                dist_str = ", ".join(dist_list)

            anchor_downgrade = None
            if not anchor and canonical_pos == "noun" and not is_idiom_slot:
                anchor_downgrade = "sense_recognition"

            if best_pattern:
                best_pattern = re.sub(
                    r"\b(?:American|British|Australian|North American)\s+English\b",
                    "",
                    best_pattern,
                    flags=re.IGNORECASE,
                )
                best_pattern = re.sub(
                    r"\b(?:formal|informal|spoken|written|technical|literary|dialect)\b",
                    "",
                    best_pattern,
                    flags=re.IGNORECASE,
                )
                # Normalize separable phrasal verb patterns (e.g. 'pack something ↔ up' -> 'pack up [sth]')
                m_sep_pv = re.match(
                    r"^([a-zA-Z]+)\s+(something\s+or\s+somebody|somebody\s+or\s+something|somebody/something|something/somebody|somebody|something|sb/sth|sth/sb|sb|sth)\s*(?:↔|<->|↔)?\s*([a-zA-Z]+(?:\s+[a-zA-Z]+)*)$",
                    best_pattern.strip(),
                    flags=re.IGNORECASE,
                )
                if m_sep_pv and f"{m_sep_pv.group(1)} {m_sep_pv.group(3)}".lower() == w_lower:
                    if verb_form_tag == "VBN":
                        best_pattern = ""
                    else:
                        obj_raw = m_sep_pv.group(2).lower()
                        obj_tag = "[sb/sth]" if ("somebody" in obj_raw and "something" in obj_raw) or "/" in obj_raw else ("[sb]" if obj_raw in ("somebody", "sb") else "[sth]")
                        best_pattern = f"{final_target} {obj_tag}"
                best_pattern = re.sub(r"\s+", " ", best_pattern).strip()

            # Synthesize explicit object-slot pattern for transitive/prepositional phrasal verbs
            # (e.g. 'catch up with' -> 'catch up with [sb/sth]') when PedLex stores the object slot in phrasal_headword or raw_w_item
            if not best_pattern and canonical_pos == "phrase" and verb_form_tag != "VBN":
                pv_hw = str((locked_sense or {}).get("phrasal_headword") or "").strip()
                raw_hw_src = pv_hw if pv_hw and w_lower in pv_hw.lower() else (raw_w_item or "")
                m_trail_obj = re.search(
                    rf"\b{re.escape(w_lower)}\s+(\[?(?:somebody/something|something/somebody|sb/sth|sth/sb|somebody|something|sb|sth)\]?)\b",
                    raw_hw_src,
                    flags=re.IGNORECASE,
                )
                if m_trail_obj:
                    obj_tok = m_trail_obj.group(1).strip("[]").lower()
                    if obj_tok in ("somebody/something", "something/somebody", "sb/sth", "sth/sb"):
                        obj_tag = "[sb/sth]"
                    elif obj_tok in ("somebody", "sb"):
                        obj_tag = "[sb]"
                    else:
                        obj_tag = "[sth]"
                    best_pattern = f"{final_target} {obj_tag}"
                elif any(
                    w_lower.endswith(" " + p)
                    for p in ("with", "to", "for", "of", "into", "from", "on", "at", "about", "against", "upon", "over", "after")
                ):
                    best_pattern = f"{final_target} [sb/sth]"

            if best_pattern and (is_plural_noun or (canonical_pos in ("verb", "phrase") and final_target.lower() != w_lower)):
                best_pattern = re.sub(r"\b" + re.escape(w_lower) + r"\b", final_target, best_pattern, flags=re.IGNORECASE)

            # Resolve slash-separated lexical alternatives and object animacy directly on best_pattern
            # (e.g. 'regulatory body/authority/agency' -> 'regulatory authority', 'wind up in/at/with etc' -> 'wind up in',
            # 'catch up with [sb/sth]' -> 'catch up with someone' or 'catch up with something' matching the example)
            # so Pattern and Fit into frame are 100% single-form and never expose raw slashes.
            resolved_obj_anim = cls._resolve_object_animacy_from_example(
                ldoce_pattern_example,
                final_target,
                quote=it.get("quote") or it.get("raw_quote"),
                default_tag="[sb/sth]",
            )
            if best_pattern:
                best_pattern = re.sub(r"\s+\betc\.?(?=\s|$)", "", best_pattern, flags=re.IGNORECASE).strip()
                best_pattern = re.sub(
                    r"\[(?:sb/sth|sth/sb|somebody/something|something/somebody)\]|\b(?:somebody/something|something/somebody|sb/sth|sth/sb)\b",
                    resolved_obj_anim,
                    best_pattern,
                    flags=re.IGNORECASE,
                )
                if "/" in best_pattern:
                    ex_low_for_slash = (ldoce_pattern_example or it.get("quote") or "").lower()
                    anc_low_for_slash = str(anchor or "").strip().lower()
                    # Resolve multi-word preposition slash alternatives (`into/out of`, `in/out of`) before single-word slash splitting
                    def _pick_multiword_prep_slash(m: re.Match) -> str:
                        p1 = m.group(1)
                        if anc_low_for_slash == "out of" or (ex_low_for_slash and "out of" in ex_low_for_slash and p1.lower() not in ex_low_for_slash):
                            return "out of"
                        return p1
                    best_pattern = re.sub(r"\b(into|in)/out\s+of\b", _pick_multiword_prep_slash, best_pattern, flags=re.IGNORECASE)
                    def _pick_slash_alt(m: re.Match) -> str:
                        alts = [a for a in m.group(0).split("/") if a]
                        if set(a.lower() for a in alts) <= {"sb", "sth", "somebody", "something"}:
                            return resolved_obj_anim
                        if anc_low_for_slash and anc_low_for_slash in [a.lower() for a in alts]:
                            return next(a for a in alts if a.lower() == anc_low_for_slash)
                        for a in alts:
                            if ex_low_for_slash and re.search(rf"\b{re.escape(a.lower())}\w*\b", ex_low_for_slash):
                                return a
                        return alts[0]
                    best_pattern = re.sub(r"\b[a-zA-Z]+(?:/[a-zA-Z]+)+\b", _pick_slash_alt, best_pattern).strip()
                if "one's" in w_lower.split():
                    best_pattern = re.sub(
                        r"\b(?:somebody's|someone's|sb's|your)\b",
                        "one's",
                        best_pattern,
                        flags=re.IGNORECASE,
                    )
                    if final_target.lower() != w_lower:
                        best_pattern = re.sub(
                            r"\b" + re.escape(w_lower) + r"\b",
                            final_target,
                            best_pattern,
                            flags=re.IGNORECASE,
                        )

            # Compute noun countability and option initial flags before finalizing `cloze_slot_frame`
            t_noun_gram_str = str((locked_sense or {}).get("grammar") or "").lower() if canonical_pos == "noun" else ""
            t_noun_gram_prof = cls.get_word_gram_profile(w_lower, pos="noun") if canonical_pos == "noun" else {}
            is_pure_uncountable_target = (
                canonical_pos == "noun"
                and not is_plural_noun
                and (
                    bool(t_noun_gram_prof.get("pure_uncountable"))
                    or (
                        "uncountable" in t_noun_gram_str
                        and not re.search(r"\bcountable\b", t_noun_gram_str)
                        and not re.search(r"\bplural\b", t_noun_gram_str)
                    )
                )
            )
            is_target_strictly_countable = (
                canonical_pos == "noun"
                and not is_pure_uncountable_target
                and (
                    is_plural_noun
                    or (
                        "uncountable" not in t_noun_gram_str
                        and (
                            bool(re.search(r"\bcountable\b", t_noun_gram_str))
                            or bool(t_noun_gram_prof.get("pure_countable"))
                        )
                    )
                )
            )
            has_uncountable_contrast_foil = (
                is_target_strictly_countable
                and any(
                    dist_meta.get(o) == "countability_contrast_uncountable"
                    or dist_meta.get(o.lower()) == "countability_contrast_uncountable"
                    or o.lower() in set(cls._NOUN_COUNTABILITY_CONTRAST_MAP.get(w_lower, []))
                    or (is_plural_noun and cls.get_word_gram_profile(o.lower(), pos="noun").get("pure_uncountable"))
                    for o in dist_list
                )
            )
            has_mixed_vc_initials = (
                canonical_pos in ("noun", "adj")
                and bool(final_options)
                and any(str(o).strip()[:1].lower() in "aeiou" for o in final_options if str(o).strip())
                and any(str(o).strip()[:1].lower().isalpha() and str(o).strip()[:1].lower() not in "aeiou" for o in final_options if str(o).strip())
            )

            # 5.4 Direct Example-to-Slot-Frame Slicing & Context-Rich Frame Synthesis
            patient_ontology_info = cls._extract_verb_patient_ontology(
                word=w_lower,
                canonical_pos=canonical_pos,
                target_entry=target_entry,
                locked_sense=locked_sense,
                definition=str(it.get("definition") or ""),
                authentic_example=ldoce_pattern_example or it.get("quote") or it.get("raw_quote") or "",
                quote=str(it.get("quote") or it.get("raw_quote") or ""),
                forbidden_words=batch_target_words,
            )
            patient_ontology_slot = (patient_ontology_info or {}).get("slot_label")
            _is_cxn_family = bool(
                cxn_override_frame
                or causative_extraposition_pat
                or cxn_override_pat
                or anchor_type in _cxn_anchor_types
            )
            cloze_slot_frame = None
            if is_quantifier_frame and quant_head_noun:
                q_det = w_lower.split()[0] if w_lower.split() and w_lower.split()[0] in ("a", "an") else "a"
                cloze_slot_frame = f"... {q_det} ____ of"
            elif cxn_override_frame:
                cloze_slot_frame = cxn_override_frame
            elif best_pattern and (_is_cxn_family or ("..." not in best_pattern and "…" not in best_pattern)):
                # Preserve trailing object placeholders ([sb/sth], [sb], [sth], [doing sth], [V-base]) and optional prepositions while stripping editorial notes
                p_clean = re.sub(
                    r"\((from|to|into|of|in|on|with|for|about|at|by|as)\)",
                    r"\1",
                    best_pattern,
                    flags=re.IGNORECASE,
                )
                p_clean = re.sub(r"\((?!should\)).*?\)", "", p_clean).strip()
                p_clean = re.sub(r"\s{2,}", " ", p_clean)
                if _is_cxn_family:
                    p_clean = re.sub(r"\bdoing\s+something\b", "[doing sth]", p_clean, flags=re.IGNORECASE)
                    p_clean = re.sub(r"\bto\s+do\s+something\b", "to do [sth]", p_clean, flags=re.IGNORECASE)
                    p_clean = re.sub(r"\bsomebody\b", "[sb]", p_clean, flags=re.IGNORECASE)
                    p_clean = re.sub(r"\bsomething\b", "[sth]", p_clean, flags=re.IGNORECASE)
                else:
                    p_clean = re.sub(r"\bdoing\s+(?:something|\[sth\])\b", "doing something", p_clean, flags=re.IGNORECASE)
                    p_clean = re.sub(r"\bto\s+do\s+(?:something|\[sth\])\b", "to do something", p_clean, flags=re.IGNORECASE)
                    p_clean = re.sub(r"\bsomebody\b", "someone", p_clean, flags=re.IGNORECASE)
                    p_clean = re.sub(r"\bsomething\b", "something", p_clean, flags=re.IGNORECASE)
                    p_clean = re.sub(r"\[sb\]", "someone", p_clean, flags=re.IGNORECASE)
                    p_clean = re.sub(r"\[sth\]", "something", p_clean, flags=re.IGNORECASE)
                    p_clean = re.sub(r"\[doing\s+sth\]", "doing something", p_clean, flags=re.IGNORECASE)
                    p_clean = re.sub(r"\bto\s+\[verb\]", "to do something", p_clean, flags=re.IGNORECASE)
                    p_clean = re.sub(r"\[(?:sb/sth|sth/sb)\]", resolved_obj_anim, p_clean, flags=re.IGNORECASE)
                p_clean = re.sub(
                    r"\[(?!(?:sb|sth|somebody|something|doing sth|V-base|verb|noun|adj|close|great|special)\]).*?\]",
                    "",
                    p_clean,
                    flags=re.IGNORECASE,
                ).strip()
                if p_clean.lower() == w_lower or p_clean.lower() == final_target.lower():
                    best_pattern = None
                    p_clean = ""
                else:
                    w_lemma = nlp(w_lower)[0].lemma_.lower() if (nlp and " " not in w_lower) else w_lower
                    target_forms = set(cls.inflected_forms(w_lower)) | {w_lower, final_target, w_lemma}
                    if w_lemma != w_lower and " " not in w_lemma:
                        target_forms |= set(cls.inflected_forms(w_lemma))
                    sorted_forms = sorted((f for f in target_forms if f), key=len, reverse=True)
                    t_pat = r"\b(?:" + "|".join(re.escape(f) for f in sorted_forms) + r")\b"
                    if re.search(t_pat, p_clean, re.IGNORECASE):
                        cloze_slot_frame = re.sub(t_pat, "____", p_clean, flags=re.IGNORECASE).strip()
                        if is_plural_noun:
                            cloze_slot_frame = re.sub(r"\b(?:a|an)\s+_{2,}\b", "____", cloze_slot_frame, flags=re.IGNORECASE).strip()
                        if not cloze_slot_frame.startswith("...") and not cloze_slot_frame.lower().startswith("it "):
                            cloze_slot_frame = f"... {cloze_slot_frame}"
                    elif any(p_clean.lower().startswith(prep) for prep in ("on ", "in ", "at ", "for ", "with ", "about ", "under ", "from ", "to ", "by ")):
                        cloze_slot_frame = f"... {p_clean} ____"
                        if is_plural_noun:
                            cloze_slot_frame = re.sub(r"\b(?:a|an)\s+_{2,}\b", "____", cloze_slot_frame, flags=re.IGNORECASE).strip()
                    elif (
                        not any(final_target.lower().endswith(" " + prep) for prep in ("on", "in", "at", "for", "with", "about", "under", "from", "to", "by", "of", "into"))
                        and any(p_clean.lower().endswith(" " + prep) for prep in ("on", "in", "at", "for", "with", "about", "under", "from", "to", "by", "of", "into"))
                    ):
                        m_end_prep = re.search(r"\b(on|in|at|for|with|about|under|from|to|by|of|into)$", p_clean.lower())
                        if m_end_prep:
                            cloze_slot_frame = f"... ____ {m_end_prep.group(1)}"

            if cloze_slot_frame and len(re.findall(r"_{2,}", cloze_slot_frame)) > 1:
                cloze_slot_frame = None
            if canonical_pos == "noun" and anchor and anchor_type == "prep" and ldoce_pattern_example:
                _anc_in_ex = cls._example_attests_prep_valency(
                    ldoce_pattern_example, final_target, str(anchor).strip().lower(), prep_dir="any"
                )
            else:
                _anc_in_ex = bool(
                    anchor
                    and (
                        not ldoce_pattern_example
                        or re.search(rf"\b{re.escape(str(anchor).strip().lower())}\b", str(ldoce_pattern_example).lower())
                    )
                )
            if not cloze_slot_frame and anchor and anchor_type == "prep" and canonical_pos in ("verb", "adj") and _anc_in_ex:
                if orig_disc_prep and canonical_pos == "verb":
                    cloze_slot_frame = f"... ____ {resolved_obj_anim} {anchor}"
                    if not best_pattern:
                        best_pattern = f"{final_target} {resolved_obj_anim} {anchor}"
                else:
                    cloze_slot_frame = f"... ____ {anchor}"
                    if not best_pattern:
                        best_pattern = f"{final_target} {anchor}"

            #Check if `sliced_frame` from `derive_cloze_slot_frame_from_example` can provide a richer,
            # example-synchronized frame directly (for all non-`_is_cxn_family`, non-`is_quantifier_frame`,
            # non-`partitive` items), while preserving `best_pattern`'s frame if `sliced_frame` misses a required `anchor`.
            _is_bare_det_frame = bool(
                cloze_slot_frame
                and re.match(
                    r"^\.\.\.\s*(?:a|an|the|each|every|his|her|their|its|our|my|your)?\s*_{2,}\s*$",
                    cloze_slot_frame.strip(),
                    re.IGNORECASE,
                )
            )
            if not _is_cxn_family and not is_quantifier_frame and anchor_type != "partitive":
                sliced_frame = cls.derive_cloze_slot_frame_from_example(
                    example_text=ldoce_pattern_example or it.get("quote") or it.get("raw_quote"),
                    final_target=final_target,
                    base_word=str((target_entry or {}).get("word") or w_lower).strip().lower(),
                    canonical_pos=canonical_pos,
                    anchor=anchor,
                    anchor_type=anchor_type,
                    is_copula_adj=is_copula_adj,
                    is_attributive_only_adj=is_attributive_only_adj,
                    is_plural_noun=is_plural_noun,
                    is_pure_uncountable_target=is_pure_uncountable_target,
                    is_target_strictly_countable=is_target_strictly_countable,
                    has_mixed_vc_initials=has_mixed_vc_initials,
                    has_uncountable_contrast_foil=has_uncountable_contrast_foil,
                    verb_form_tag=verb_form_tag,
                    forbidden_words=batch_target_words,
                    patient_ontology_slot=patient_ontology_slot,
                )
                if sliced_frame:
                    _sliced_is_bare = bool(
                        re.match(
                            r"^\.\.\.\s*(?:a|an|the|each|every|his|her|their|its|our|my|your|to|someone|somebody|he|she|they)?\s*_{2,}\s*$",
                            sliced_frame.strip(),
                            re.IGNORECASE,
                        )
                    )
                    if not cloze_slot_frame or _is_bare_det_frame:
                        cloze_slot_frame = sliced_frame
                    elif not _sliced_is_bare:
                        # Prefer `sliced_frame` unless `cloze_slot_frame` preserves a required `anchor` (or intervening trivalent object) that is attested in the example
                        _anc_low = str(anchor or "").strip().lower()
                        _pat_has_anc = bool(_anc_low and _anc_in_ex and re.search(rf"\b{re.escape(_anc_low)}\b", cloze_slot_frame.lower()))
                        _sliced_has_anc = bool(_anc_low and re.search(rf"\b{re.escape(_anc_low)}\b", sliced_frame.lower()))
                        _pat_has_mid_obj = bool(
                            orig_disc_prep
                            and re.search(r"_{2,}\s+(?:someone|something|oneself|\[sb\]|\[sth\]|\[sb/sth\])\s+[a-z]+", cloze_slot_frame.lower())
                            and not re.search(r"_{2,}\s+(?:someone|something|oneself|\[sb\]|\[sth\]|\[sb/sth\])\s+[a-z]+", sliced_frame.lower())
                        )
                        if not (_pat_has_anc and not _sliced_has_anc) and not _pat_has_mid_obj:
                            cloze_slot_frame = sliced_frame

            # Guaranteed POS-grounded fallback if no example/quote token matched
            if not cloze_slot_frame:
                if canonical_pos == "adj":
                    if is_copula_adj:
                        cloze_slot_frame = f"... be ____ {anchor}" if (anchor and anchor_type == "prep" and _anc_in_ex) else "... be ____"
                    elif anchor and anchor_type == "modified_noun":
                        cloze_slot_frame = f"... the ____ {anchor}"
                    elif anchor and anchor_type == "prep" and _anc_in_ex:
                        cloze_slot_frame = f"... ____ {anchor}"
                    else:
                        cloze_slot_frame = "... the ____ one"
                elif canonical_pos == "verb":
                    if anchor and anchor_type == "prep":
                        cloze_slot_frame = f"... ____ {anchor}"
                    elif anchor and anchor_type == "object":
                        cloze_slot_frame = f"... ____ {anchor}"
                    else:
                        cloze_slot_frame = f"... ____ {resolved_obj_anim}"
                elif canonical_pos == "noun":
                    if anchor and anchor_type == "verb":
                        det_fb = "" if (is_pure_uncountable_target or is_plural_noun) else ("the " if has_mixed_vc_initials else "a ")
                        cloze_slot_frame = f"... {anchor} {det_fb}____"
                    elif anchor and anchor_type == "adj":
                        det_fb = "" if (is_pure_uncountable_target or is_plural_noun) else ("the " if has_mixed_vc_initials else "a ")
                        cloze_slot_frame = f"... {det_fb}{anchor} ____"
                    elif anchor and anchor_type == "prep":
                        det_fb = "" if is_pure_uncountable_target else "the "
                        cloze_slot_frame = f"... {det_fb}____ {anchor}"
                    else:
                        det_fb = "their" if is_pure_uncountable_target else ("each" if has_uncountable_contrast_foil and not is_plural_noun else "the")
                        cloze_slot_frame = f"... {det_fb} ____"
                elif canonical_pos == "adv":
                    if anchor_type == "discourse":
                        cloze_slot_frame = "____, ..."
                    elif anchor and anchor_type == "modifies_adj":
                        cloze_slot_frame = f"... ____ {anchor}"
                    elif anchor and anchor_type == "modifies_verb":
                        cloze_slot_frame = f"... {anchor} ____"
                    else:
                        cloze_slot_frame = "... act ____"
                else:
                    cloze_slot_frame = f"... ____ {resolved_obj_anim}"

            _frame_has_copula = bool(
                cloze_slot_frame
                and re.search(
                    r"\b(?:be|am|is|are|was|were|been|being|become|became|fall|fell|fallen|remain|remained|stay|stayed|seem|seemed|appear|appeared|look|looked|sound|sounded|feel|felt|get|got|turn|turned|grow|grew|keep|kept)\s+(?:[a-z]+\s+)?_{2,}",
                    cloze_slot_frame.lower(),
                )
            )
            _frame_has_post_prep = bool(
                cloze_slot_frame
                and re.search(
                    r"_{2,}\s+(?:to|in|of|with|for|from|at|on|about|into|by|as|towards?|against)\b",
                    cloze_slot_frame.lower(),
                )
            )
            if canonical_pos == "adj" and (_frame_has_copula or _frame_has_post_prep):
                is_copula_adj = True
                is_attributive_only_adj = False
            elif not is_copula_adj and not is_attributive_only_adj and canonical_pos == "adj" and (cloze_slot_frame or anchor):
                copula_lemmas = {"be", "become", "fall", "remain", "stay", "seem", "appear", "look", "sound", "smell", "taste", "feel", "get", "turn", "grow", "keep"}
                check_tokens = re.findall(r"\b[a-zA-Z]+\b", (cloze_slot_frame or "") + " " + str(anchor or ""))
                for t in check_tokens:
                    doc_t = nlp(t)
                    if any(tok.lemma_.lower() in copula_lemmas for tok in doc_t):
                        is_copula_adj = True
                        break

            if best_pattern and not cloze_slot_frame:
                w_lemma = nlp(w_lower)[0].lemma_.lower() if nlp else w_lower
                target_forms = set(cls.inflected_forms(w_lower)) | {w_lower, final_target, w_lemma}
                if not any(re.search(rf"\b{re.escape(f)}\b", best_pattern, re.IGNORECASE) for f in target_forms if f):
                    best_pattern = None

            is_be_led_phrase = (
                canonical_pos == "phrase"
                and (inflection_desc or "base form") == "base form"
                and bool(final_target.split())
                and final_target.split()[0].lower() == "be"
            )
            _ft_words = final_target.lower().split()
            _ft_head_ent = cls.get_ldoce_entry(_ft_words[0]) if _ft_words else None
            _ft_head_pos = str((_ft_head_ent or {}).get("pos") or "").strip().lower()
            _ft_is_non_verbal_idiom = (
                (len(_ft_words) == 3 and _ft_words[1] in ("and", "or", "by", "to") and (_ft_words[0] == _ft_words[2] or _ft_words[0] in ("back", "up", "down", "in", "out", "on", "off", "more", "sooner", "now", "first", "day", "time", "side", "face", "hand", "step", "word", "pros", "ups", "odds", "ins")))
                or (len(_ft_words) >= 3 and _ft_words[1] == "of")
                or _ft_head_pos in ("noun", "adjective", "adverb", "preposition", "determiner")
            )
            is_base_verb_slot = (
                (inflection_desc or "base form") == "base form"
                and not is_plural_noun
                and not is_be_led_phrase
                and (
                    canonical_pos == "verb"
                    or (
                        canonical_pos == "phrase"
                        and str((target_entry or {}).get("pos", "")).strip().lower() != "noun"
                        and bool(_ft_words)
                        and not _ft_is_non_verbal_idiom
                        and _ft_words[0] not in {"a", "an", "the", "in", "on", "at", "by", "for", "with", "from", "to", "of"}
                        and bool(cls._wordnet_pos(_ft_words[0]) & {"v"})
                    )
                )
            )
            be_led_slot_hint = (
                " (Bare-infinitive 'be' phrase slot required: precede '____' with infinitive 'to ____' "
                "or a modal verb 'can/will/would/should/must/may/might ____' — NEVER place '____' after "
                "finite 'am/is/are/was/were/been' or directly after a bare subject without a modal)."
            )
            non_base_verb_slot_hint = ""
            if canonical_pos in ("verb", "phrase") and not is_base_verb_slot and not is_be_led_phrase:
                if verb_form_tag == "VBD":
                    non_base_verb_slot_hint = (
                        f" (Simple past-tense verb slot required for '{final_target}': place '____' directly after the subject "
                        "as a simple past verb — do not put 'have/has/had' or 'to' before '____')."
                    )
                elif verb_form_tag == "VBZ":
                    non_base_verb_slot_hint = (
                        f" (Present 3rd-person singular '-s' verb slot required for '{final_target}': place '____' directly after "
                        "a singular subject — do not put any modal or 'to' before '____')."
                    )
                elif verb_form_tag == "VBN":
                    non_base_verb_slot_hint = (
                        f" (Past participle verb slot required for '{final_target}': place 'has/have/had' or "
                        "passive 'is/are/was/were' immediately before '____')."
                    )
                elif verb_form_tag == "VBG":
                    non_base_verb_slot_hint = (
                        f" (Present participle / '-ing' verb slot required for '{final_target}': place '____' after a preposition "
                        "or progressive 'is/are/was/were')."
                    )

            if is_pure_uncountable_target and cloze_slot_frame:
                cloze_slot_frame = re.sub(r"\b(?:a|an|each|every)\s+_{2,}\b", "____", cloze_slot_frame, flags=re.IGNORECASE).strip()
            elif has_mixed_vc_initials and cloze_slot_frame:
                if has_uncountable_contrast_foil and not is_plural_noun:
                    cloze_slot_frame = re.sub(r"\b(?:a|an)\s+_{2,}\b", "each ____", cloze_slot_frame, flags=re.IGNORECASE).strip()
                else:
                    cloze_slot_frame = re.sub(r"\b(?:a|an)\s+_{2,}\b", "the ____", cloze_slot_frame, flags=re.IGNORECASE).strip()

            if cloze_slot_frame:
                if not _is_cxn_family:
                    # Resolve any remaining metasyntax brackets ('[sb]', '[sb/sth]', '[sth]', 'to do [sth]') to concrete single-choice cues
                    cloze_slot_frame = re.sub(r"\bto\s+do\s+\[sth\]", "to [verb]", cloze_slot_frame, flags=re.IGNORECASE)
                    cloze_slot_frame = re.sub(r"\[sb\]", "someone", cloze_slot_frame, flags=re.IGNORECASE)
                    cloze_slot_frame = re.sub(r"\[(?:sb/sth|sth/sb)\]", resolved_obj_anim, cloze_slot_frame, flags=re.IGNORECASE)
                    cloze_slot_frame = re.sub(r"\[sth\]", "something", cloze_slot_frame, flags=re.IGNORECASE)

                _NON_VERB_FRAME_HEADS = {
                    "in", "on", "at", "by", "with", "from", "to", "for", "of", "as",
                    "into", "onto", "upon", "out", "off", "up", "down", "over", "under",
                    "after", "before", "without", "within", "between", "among", "through",
                    "during", "against", "about", "around", "across", "behind", "beyond",
                    "despite", "except", "like", "near", "since", "than", "towards", "toward",
                    "until", "via", "and", "or", "but", "if", "when", "while", "because",
                }
                m_adj_noun = re.search(r"_{2,}\s+([a-zA-Z]+)\b", cloze_slot_frame) if canonical_pos == "adj" else None
                if m_adj_noun and m_adj_noun.group(1).lower() in _NON_VERB_FRAME_HEADS:
                    m_adj_noun = None
                m_verb_det_noun = (
                    re.match(r"^\.\.\.\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)?)\s+(a|an|the|each|every)\s+_{2,}$", cloze_slot_frame.strip())
                    if canonical_pos == "noun"
                    else None
                )
                if m_verb_det_noun and m_verb_det_noun.group(1).strip().split()[0].lower() in _NON_VERB_FRAME_HEADS:
                    m_verb_det_noun = None
                if m_verb_det_noun and not is_plural_noun:
                    verb_part = m_verb_det_noun.group(1).strip()
                    det_part = m_verb_det_noun.group(2).strip()
                    v_toks = verb_part.split()
                    v_past_first = cls._inflect_verb(v_toks[0], "VBD") if v_toks else ""
                    v_past = " ".join([v_past_first] + v_toks[1:]) if v_past_first and v_past_first.lower() != v_toks[0].lower() else ""
                    ex_low_v = (ldoce_pattern_example or "").lower()
                    chosen_verb_part = v_past if (v_past and re.search(rf"\b{re.escape(v_past.lower())}\b", ex_low_v)) else verb_part
                    cloze_slot_frame = f"... {chosen_verb_part} {det_part} ____"
                    stem_core = f"Fit into frame '{cloze_slot_frame}'."
                else:
                    stem_core = f"Fit into frame '{cloze_slot_frame}'."
                    _frame_already_has_vbn_aux = bool(
                        verb_form_tag == "VBN"
                        and re.search(r"\b(?:has|have|had|is|are|was|were|been|be|got|get)\b", cloze_slot_frame.lower())
                    )
                    _frame_already_has_vbg_gov = bool(
                        verb_form_tag == "VBG"
                        and re.search(
                            r"\b(?:of|in|on|at|for|from|by|with|without|about|after|before|into|upon|is|are|was|were|been|be)\b",
                            cloze_slot_frame.lower(),
                        )
                    )
                    if is_quantifier_frame and quant_head_noun:
                        stem_core += f" (Quantifier head noun slot: blank only '{quant_head_noun}' inside '{w_lower}')."
                    elif is_plural_noun and not re.search(r"\b(?:several|many|few|fewer|two|three|four|these|those|various)\s+_{2,}", cloze_slot_frame.lower()):
                        stem_core += f" (Plural noun slot required: e.g. '{final_target}')."
                    elif is_copula_adj and not _frame_has_copula:
                        stem_core += " (Complete predicate required: stem MUST include the linking verb so the adjective predicate is complete; NEVER place before a noun)."
                    elif is_attributive_only_adj and not m_adj_noun:
                        stem_core += " (Attributive pre-noun slot required: place '____' directly before a modified noun; NEVER use after a linking verb)."
                    elif is_be_led_phrase and not re.search(r"\b(?:to|will|would|can|could|may|might|should|must)\s+_{2,}", cloze_slot_frame.lower()):
                        stem_core += be_led_slot_hint
                    elif is_base_verb_slot:
                        _has_left_aux_in_frame = bool(
                            re.search(
                                r"\b(?:to|will|would|can|could|may|might|should|must|do|does|did|let|make|made|help|helped)\s+(?:[a-z]+\s+)?_{2,}",
                                cloze_slot_frame.lower(),
                            )
                        )
                        if ("one's" in final_target.lower() or "oneself" in final_target.lower()) and not _has_left_aux_in_frame:
                            stem_core += (
                                " (Base-form verbal idiom that already includes 'one's': place '____' after 'to' or a modal verb)."
                            )
                        elif not _has_left_aux_in_frame:
                            stem_core += " (Base-form verb slot: use after a modal verb, infinitive 'to', or a plural/1st-person subject — never after singular 'he/she/it' or 'ended up')."
                    elif non_base_verb_slot_hint and not _frame_already_has_vbn_aux and not _frame_already_has_vbg_gov:
                        stem_core += non_base_verb_slot_hint
            elif best_pattern:
                stem_core = "Fit into the Required Stem Frame above."
                if is_plural_noun:
                    stem_core += f" (Plural noun slot required: e.g. '{final_target}')."
                elif is_copula_adj:
                    stem_core += " (Complete predicate required: stem MUST include the linking verb; NEVER place before a noun)."
                elif is_attributive_only_adj:
                    stem_core += " (Attributive pre-noun slot required: place '____' directly before a modified noun; NEVER use after a linking verb)."
                elif is_be_led_phrase:
                    stem_core += be_led_slot_hint
                elif is_base_verb_slot:
                    stem_core += " (Base-form verb slot: use after a modal verb, infinitive 'to', or a plural/1st-person subject — never after singular 'he/she/it' or 'ended up')."
                elif non_base_verb_slot_hint:
                    stem_core += non_base_verb_slot_hint
            else:
                stem_core = "Write a natural sentence containing '____' that matches the Definition above."
                if is_plural_noun:
                    stem_core += f" (Plural noun slot required: e.g. '{final_target}')."
                elif is_copula_adj:
                    stem_core += " (Complete predicate required: stem MUST include a linking/copular verb; NEVER place before a noun)."
                elif is_attributive_only_adj:
                    stem_core += " (Attributive pre-noun slot required: place '____' directly before a modified noun; NEVER use after a linking verb)."
                elif is_be_led_phrase:
                    stem_core += be_led_slot_hint
                elif is_base_verb_slot:
                    stem_core += " (Base-form verb slot: use after a modal verb, infinitive 'to', or a plural/1st-person subject — never after singular 'he/she/it' or 'ended up')."
                elif non_base_verb_slot_hint:
                    stem_core += non_base_verb_slot_hint

            if "one's" in w_lower.split():
                m_poss_sync = re.search(r"\b(her|his|their|its)\b", final_target.lower())
                if m_poss_sync:
                    _sync_poss = m_poss_sync.group(1)
                    _ant_hint = (
                        "a female person ('she/her')"
                        if _sync_poss == "her"
                        else ("a male person ('he/him')" if _sync_poss == "his" else ("a plural subject ('they/them')" if _sync_poss == "their" else "a singular thing ('it')"))
                    )
                    stem_core += (
                        f" (Whole-idiom slot already contains '{_sync_poss}' as in '{final_target}': "
                        f"ensure the sentence includes {_ant_hint} for '{_sync_poss}' to refer to, and NEVER repeat '{_sync_poss}' or the idiom's tail noun outside '____')."
                    )

            _NON_INDIVIDUAL_COLLECTIVE_NOUNS = {
                "staff", "personnel", "workforce", "police", "cattle",
                "poultry", "clergy", "clientele", "livestock", "foliage", "gentry",
            }
            _blank_after_prep = bool(
                cloze_slot_frame
                and re.search(
                    r"\b(?:a|an|each|every|of|on|in|at|for|with|from|by|to|into|under|over|the|his|her|their|our|your|my|its|own|no)\s+____",
                    cloze_slot_frame.lower(),
                )
            )
            _blank_after_bare_prep = bool(
                cloze_slot_frame
                and re.search(
                    r"\b(?:of|on|in|at|for|with|from|by|to|into|under|over)\s+____",
                    cloze_slot_frame.lower(),
                )
            )
            is_non_individual_collective = (
                canonical_pos == "noun"
                and final_target.lower() in _NON_INDIVIDUAL_COLLECTIVE_NOUNS
                and not is_plural_noun
            )
            if is_non_individual_collective and not _blank_after_bare_prep:
                stem_core += (
                    " (Collective noun slot required: precede '____' with 'the', a possessive like 'their/our', "
                    "or 'the entire ____' — NEVER use singular individual determiners 'a', 'an', 'one', 'each', "
                    "or 'every' before '____')."
                )
            elif has_uncountable_contrast_foil and not _blank_after_prep:
                if is_plural_noun:
                    stem_core += (
                        " (Countability contrast active against uncountable distractor: precede '____' with an explicit plural quantifier/numeral "
                        "like 'several / many / two / three / fewer / these ____' or pair '____' with a plural verb 'are/were/have' "
                        "— NEVER use 'the / some / any / various' alone without explicit plural agreement)."
                    )
                else:
                    stem_core += (
                        " (Countability contrast active against uncountable distractor: precede '____' with a singular countable determiner "
                        "like 'each ____', 'every ____', or 'a single ____' — NEVER use 'the', 'some', 'any', or a bare noun before '____')."
                    )
            elif is_pure_uncountable_target and not _blank_after_prep:
                stem_core += (
                    " (Uncountable noun slot required: precede '____' with 'the', a possessive like 'their/its', "
                    "or use bare uncountable '____' — NEVER use 'a', 'an', 'each', or 'every' before '____')."
                )

            if (
                canonical_pos == "noun"
                and w_lower.endswith("ics")
                and is_pedlex_plural
                and not re.search(
                    r"\b(?:study|science|scientific|branch|subject|field|theory|art)\b",
                    str(it.get("definition") or "").lower(),
                )
            ):
                stem_core += (
                    f" (Concrete [plural] noun sense required for '{final_target}' matching \"{it.get('definition')}\": "
                    "frame a concrete physical/commercial context with plural agreement where applicable — "
                    "NEVER write an academic subject/university frame like 'department of ____', 'professor of ____', or 'study of ____')."
                )

            # Warn against leaking derivational root cognates (e.g. 'solve' when testing 'unsolved')
            if " " not in w_lower and len(w_lower) >= 6:
                for _pfx in ("un", "in", "im", "ir", "il", "dis", "non", "re", "mis", "over", "under"):
                    if w_lower.startswith(_pfx) and len(w_lower) - len(_pfx) >= 4:
                        _stripped_base = w_lower[len(_pfx):]
                        _base_lem = re.sub(r"(?:ed|ing|s|es|ly|able|ible)$", "", _stripped_base)
                        if len(_base_lem) < 4:
                            _base_lem = _stripped_base
                        if cls.get_ldoce_entry(_base_lem) or cls._wordnet_pos(_base_lem):
                            stem_core += f" (Do NOT use root cognates like '{_base_lem}' or '{_stripped_base}' anywhere in the stem)."
                        break

            task_parts = [stem_core]

            # Data-Driven Grammatical & Usage Discriminator: only emit in Micro-Task when it carries
            # a concrete grammatical slot constraint (`[...]`) or a sense-locked `Grammar Box:` rule
            target_base = quant_head_noun if is_quantifier_frame else w_lower
            dist_base_list = [
                o for o in _base_options
                if o.lower() not in {target_base.lower(), final_target.lower(), slot_target.lower()}
            ]
            dist_base_meta = {b: dist_meta.get(f) or dist_meta.get(b) for b, f in zip(_base_options, final_options)}
            discriminator_note = cls._semantic_discriminator_note(
                target_base,
                it.get("definition"),
                dist_base_list,
                dist_base_meta,
                pos=canonical_pos,
                sense_index=t_s_idx,
                quote=it.get("quote") or it.get("raw_quote")
            )
            if discriminator_note:
                # Strip generic non-grammatical Contrast labels ('polar opposite', 'orthographic look-alike', 'different degree/scale')
                filtered_disc = re.sub(
                    r"Contrast:\s*vs\s*'[^']+'\s*\((?:polar opposite of target meaning|orthographic look-alike / paronym with distinct meaning|different degree/scale or collocational frame|polar opposite)\)\.\s*",
                    "",
                    discriminator_note,
                ).strip()
                if filtered_disc:
                    task_parts.append(filtered_disc)

            micro_task = " ".join(task_parts)

            # Determine anchor source: "quote" if anchor appeared in authentic quote text, else "dictionary"
            anchor_source = None
            if anchor:
                if anchor_type == "prep":
                    anchor_source = "quote" if quote_shown_prep else "dictionary"
                elif it.get("quote") and re.search(rf"\b{re.escape(anchor.lower())}\b", it["quote"].lower()):
                    anchor_source = "quote"
                else:
                    anchor_source = "dictionary"

            skeletons.append({
                "target_word": final_target,
                "base_headword": w,
                "part_of_speech": canonical_pos,
                "inflection": inflection_desc,
                "verb_form_tag": verb_form_tag,
                "verb_requires_object": verb_requires_object,
                "context_anchor": anchor,
                "anchor_type": anchor_type,
                "anchor_source": anchor_source,
                # B2: 'anchor_downgrade' records that no collocation frame survived the gates,
                # so this item is planned as a sense-recognition item and the writer is told not
                # to invent an anchor. None means an ordinary frame-driven cloze.
                "anchor_downgrade": anchor_downgrade,
                "item_type": anchor_downgrade or "cloze",
                "authentic_example": ldoce_pattern_example,
                "best_pattern": best_pattern,
                "cloze_slot_frame": cloze_slot_frame,
                "cloze_frame_prototype": ldoce_cloze_frame,
                "micro_task": micro_task,
                "distractor_mechanisms": dist_meta,
                "prescribed_options": final_options,
                "correct_answer_index": correct_answer_index,
                "definition": skeleton_definition,
                # Quote provenance tiering: 'candidate_quote' is the raw curriculum quote and
                # is display-only when 'quote_provenance' == 'weak'; only 'licensed_quote' was
                # strong enough to license sense-locking, anchors and valency evidence.
                "candidate_quote": it.get("raw_quote") or it.get("quote") or "",
                "licensed_quote": it.get("quote") or "",
                "quote_provenance": quote_provenance,
                "quote_wordcount": quote_wordcount,
            })

        # Balance single-word headwords vs multi-word expressions (e.g. 60% single words + 40% multi-word phrases)
        skeletons = cls._balance_single_and_multi_word(skeletons, target_count=target_count, multi_word_ratio=0.4)

        # Balance Correct Answer Index Distribution evenly across 0, 1, 2, 3 (e.g. 3/3/2/2 for 10 items)
        # Prevents CRC32 clustering where one slot gets 4+ occurrences and disrupts psychometric balance
        if skeletons:
            n = len(skeletons)
            # Target distribution counts: base = n // 4, rem = n % 4
            target_distribution = [n // 4 + (1 if i < (n % 4) else 0) for i in range(4)]
            # Assign balanced slot to each item deterministically
            # Create a pool of slots [0, 1, 2, 3, 0, 1, 2, 3, 0, 1] shuffled deterministically
            balanced_slots = []
            for slot_idx, count in enumerate(target_distribution):
                balanced_slots.extend([slot_idx] * count)
            # Deterministic pseudo-shuffle based on passage words
            seed_hash = zlib.crc32("".join(s["target_word"] for s in skeletons).encode("utf-8"))
            # Standard Knuth shuffle with linear congruential generator
            rng_state = seed_hash
            for i in range(len(balanced_slots) - 1, 0, -1):
                rng_state = (rng_state * 1103515245 + 12345) & 0x7FFFFFFF
                j = rng_state % (i + 1)
                balanced_slots[i], balanced_slots[j] = balanced_slots[j], balanced_slots[i]

            # Re-map options for each skeleton to match its balanced slot
            for idx, skel in enumerate(skeletons):
                target_w = skel["target_word"]
                cur_opts = skel["prescribed_options"]
                new_slot = balanced_slots[idx]
                
                # Extract distractors in order
                other_opts = [opt for opt in cur_opts if opt.lower() != target_w.lower()]
                # If target was duplicated or missing, safeguard
                if len(other_opts) < 3:
                    continue
                new_opts = other_opts[:new_slot] + [target_w] + other_opts[new_slot:]
                skel["prescribed_options"] = new_opts
                skel["correct_answer_index"] = new_slot

        return skeletons
