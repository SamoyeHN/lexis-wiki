"""
Atomic Pattern-Example-Anchor binding and precomputed target skeleton blueprints for downstream quizzes.
"""

"""
Linguistic Engine for Lexis Wiki.
Integrates spaCy (computational dependency syntax) and WordNet (lexical relations)
to achieve deterministic sentence indexing, macro grammar domain classification,
boundary-accurate phrase extraction, canonical lemmatization, and zero-collision distractors.
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



class QuizSkeletonsMixin:

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
                if not word or not cls.is_pedagogical_target(word):
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
            blocks = re.split(r'\n(?=##\s*\[\[)', vocab_input)
            for b in blocks:
                m_word = re.search(r'##\s*\[\[(.*?)\]\]([^\n]*)', b)
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
                if not word or not cls.is_pedagogical_target(word):
                    continue
                m_pos = re.search(r'-\s*\*\*Part\s+of\s+Speech\*\*:\s*([^\n]+)', b, re.IGNORECASE)
                pos_str = m_pos.group(1).strip() if m_pos else ""
                if not pos_str and header_tail:
                    ht_clean = re.sub(r'`[^`]*`', '', header_tail)
                    ht_clean = re.sub(r'\b(?:A1|A2|B1|B2|C1|C2)\b', '', ht_clean).strip()
                    if ht_clean:
                        pos_str = ht_clean
                m_def = re.search(r'-\s*\*\*Definition\*\*:\s*([^\n]+)', b, re.IGNORECASE)
                m_quote = re.search(r'-\s*\*\*Quoted Sentence\*\*:\s*([^\n]+)', b, re.IGNORECASE)
                m_ex = re.search(r'-\s*\*\*Example\s+Usage\*\*:\s*([^\n]+)', b, re.IGNORECASE)
                parsed_items.append({
                    "raw_word": raw_w,
                    "word": word,
                    "part_of_speech": pos_str or "noun",
                    "definition": m_def.group(1).strip() if m_def else "",
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
                if pos_tag == "n" and target_tok.tag_ == "NNS":
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
        return cls._balance_single_and_multi_word(cloze_items, target_count=target_count, multi_word_ratio=0.4)

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
        native_bundles = sense.get("pattern_bundles")
        sense_examples = sense.get("examples") or []
        if native_bundles:
            out: List[Dict[str, Any]] = []
            t_low = (target_word or "").strip().lower()
            t_low_esc = re.escape(t_low)
            for nb in native_bundles:
                pat = (nb.get("pattern") or "").strip()
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
                all_preps = re.findall(r"\b(on|in|at|for|with|about|under|from|to|by|into|of|between|over|against|towards|upon|as)\b", p_for_prep)
                cand_preps: List[str] = []
                for cp in all_preps:
                    if cp not in cand_preps:
                        cand_preps.append(cp)
                bound_prep = cand_preps[0] if cand_preps else None
                matched_nb_ex = nb.get("example") or None
                if not matched_nb_ex and sense_examples:
                    if inf_valency and t_low:
                        for s_ex in sense_examples:
                            if cls.text_contains_form(s_ex, t_low) and re.search(r"\bto\s+[a-z]{2,}\b", s_ex.lower()):
                                matched_nb_ex = s_ex
                                break
                    elif gerund_valency and t_low:
                        for s_ex in sense_examples:
                            if cls.text_contains_form(s_ex, t_low) and re.search(r"\b[a-z]{2,}ing\b", s_ex.lower()):
                                matched_nb_ex = s_ex
                                break
                    else:
                        branches = cls.clean_and_expand_pattern_slashes(pat) or [pat]
                        for s_ex in sense_examples:
                            s_ex_l = s_ex.lower()
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
                                    matched_nb_ex = s_ex
                                    break
                            if matched_nb_ex:
                                break
                if matched_nb_ex and len(cand_preps) > 1:
                    ex_l = matched_nb_ex.lower()
                    for cp in cand_preps:
                        if re.search(rf"\b{re.escape(cp)}\b", ex_l):
                            bound_prep = cp
                            break
                out.append({
                    "pattern": pat,
                    "example": matched_nb_ex,
                    "prep": bound_prep,
                    "preps": cand_preps,
                    "inf_valency": inf_valency,
                    "gerund_valency": gerund_valency,
                    "source": nb.get("source", "native_bundle")
                })
            return out + verb_colloc_bundles

        # 2. Fallback for older DB versions: parse flat patterns with exact scoring
        raw_patterns = sense.get("patterns") or []
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
            if has_tw:
                score += 50
            else:
                score -= 30

            # Source tier bonus/guard: `gramexa` is the explicit syntactic valency frame;
            # `colloinexa` coordinate binomials (`weird and wonderful`) without quote attestation or example must not hijack general items.
            if b_src == "gramexa":
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

            # Massive bonus if bundle contains an authentic illustrative example
            if ex:
                score += 25
            elif quote_adj_preps:
                # Directly attested immediately after target word in passage quote: do NOT penalize
                score += 20
            elif cand_preps:
                # If pattern claims a bound preposition (e.g. 'identical in') but has NO matching example in the sense,
                # penalize heavily so it doesn't pair with a non-prepositional fallback sentence ('four identical houses')!
                score -= 35

            # Quote / preferred_anchor alignment: if curriculum quote or anchor uses one of the bundle's prepositions
            if preferred_anchor and any(cp.lower() == preferred_anchor.lower() for cp in cand_preps):
                score += 40
                # Pin the bundle's active prep to preferred_anchor if it's in cand_preps and attested in ex (or no ex)
                if not ex or re.search(rf"\b{re.escape(preferred_anchor.lower())}\b", ex.lower()):
                    b = dict(b, prep=preferred_anchor.lower())
                    prep = b["prep"]
            elif quote_adj_preps:
                score += 55
                for cp in quote_adj_preps:
                    if not ex or re.search(rf"\b{re.escape(cp.lower())}\b", ex.lower()):
                        b = dict(b, prep=cp.lower())
                        prep = b["prep"]
                        break
            elif any(re.search(rf"\b{re.escape(cp)}\b", q_l) for cp in cand_preps):
                score += 35
                for cp in cand_preps:
                    if re.search(rf"\b{re.escape(cp)}\b", q_l) and (not ex or re.search(rf"\b{re.escape(cp)}\b", ex.lower())):
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
    def select_optimal_six_source_example(
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
        Six-Source Optimal Example Pre-Selector (GEMINI.md §3.2).
        Evaluates sense-isolated candidate examples across all 6 dictionary compartments in `pedlex.db`:
          1. Sense-locked `grammar_boxes`
          2. Locked `Sense.pattern_bundles` (`gramexa`) & `Sense.examples`
          3. Sense-matched `LDOCE Thesaurus`
          4. Sense-matched `Language Activator`
          5. Sense-matched `Oxford Thesaurus`
          6. Sense-isolated `Collocations` & `Cross-Collocations`
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
                " ".join(str(x) for x in (locked_sense.get("examples") or [])),
                " ".join(str(x) for x in (locked_sense.get("synonyms") or [])),
                " ".join(str(x) for x in (locked_sense.get("patterns") or [])),
            ]).lower()
            s_kws = {
                w for w in re.findall(r"[a-z]{4,}", l_blob)
                if w not in (ft_low, bw_low) and w not in cls._ANCHOR_STOPWORDS
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
                        if w not in (ft_low, bw_low) and w not in cls._ANCHOR_STOPWORDS and w not in s_kws
                    }

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
            if strict_positive and s_kws and pos_cnt == 0 and neg_cnt > 0:
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

        # Source 5: Sense-matched `Oxford Thesaurus`
        if canonical_pos in ("noun", "verb", "adj", "adv"):
            ox_entries = cls.get_oxford_thesaurus_entries(bw_low) or cls.get_oxford_thesaurus_entries(ft_low)
            req_p_code = cls._pos_code(canonical_pos)
            best_ox_ex = None
            best_ox_score = -1
            for ox_e in (ox_entries or []):
                if req_p_code and cls._pos_code(ox_e.get("pos")) != req_p_code:
                    continue
                for s_blk in (ox_e.get("senses") or []):
                    ox_ex = str(s_blk.get("example") or "").strip()
                    if not ox_ex:
                        continue
                    ox_probe = f"{s_blk.get('definition') or ''} {' '.join(str(x) for x in (s_blk.get('synonyms') or [])[:12])} {ox_ex}".lower()
                    ox_words = {w for w in re.findall(r"[a-z]{4,}", ox_probe) if w not in (ft_low, bw_low) and w not in cls._ANCHOR_STOPWORDS}
                    pos_c = len(ox_words & s_kws)
                    neg_c = len(ox_words & o_kws)
                    if is_poly and (neg_c > pos_c or (s_kws and pos_c == 0)):
                        continue
                    sc_ox = pos_c * 2 - neg_c
                    if sc_ox > best_ox_score:
                        best_ox_score = sc_ox
                        best_ox_ex = ox_ex
            if best_ox_ex:
                raw_candidates.append((best_ox_ex, None, "oxford_thesaurus"))

        # Source 6: Sense-isolated `Collocations` & `current_example`
        if current_example:
            raw_candidates.append((current_example, best_pattern, "collocation_anchor"))
        if target_entry:
            for c_list in ((target_entry.get("collocations") or {}).values()):
                for c_it in (c_list or []):
                    if not isinstance(c_it, dict) or not c_it.get("example"):
                        continue
                    c_probe = f"{c_it.get('collocation') or ''} {c_it.get('gloss') or ''} {c_it.get('example') or ''}"
                    if _passes_sense_isolation(c_probe, c_it.get("sense_indices"), strict_positive=True):
                        if is_poly and s_kws:
                            c_w = {w for w in re.findall(r"[a-z]{4,}", c_probe.lower()) if w not in (ft_low, bw_low) and w not in cls._ANCHOR_STOPWORDS}
                            if not (c_w & s_kws) and not c_it.get("sense_indices"):
                                continue
                        c_colloc_str = str(c_it.get("collocation") or "").strip()
                        c_pat_hint = (
                            c_colloc_str
                            if (
                                c_colloc_str
                                and " " in c_colloc_str
                                and (cls.text_contains_form(c_colloc_str, bw_low) or cls.text_contains_form(c_colloc_str, ft_low))
                            )
                            else None
                        )
                        raw_candidates.append((str(c_it["example"]), c_pat_hint, "collocations"))

        if not raw_candidates:
            return current_example, best_pattern, "none"

        nlp = cls.get_spacy()
        t_forms = {ft_low, bw_low} | set(cls.inflected_forms(bw_low)) | set(cls.inflected_forms(ft_low))
        scored_exs: List[Tuple[int, str, Optional[str], str]] = []
        seen_ex_norm: Set[str] = set()

        tier_bonuses = {
            "pattern_bundle_gramexa": 32,
            "grammar_boxes": 28,
            "sense_example": 28,
            "ldoce_thesaurus": 24,
            "language_activator": 22,
            "oxford_thesaurus": 20,
            "pattern_bundle_sense_verb_colloc": 26,
            "collocation_anchor": 18,
            "collocations": 14,
            "pattern_bundle_colloinexa": 10,
        }

        for raw_ex, pat_hint, tier in raw_candidates:
            ex_clean = re.sub(r"\s*\(=[^)]*\)", "", str(raw_ex or "")).strip().strip('"').strip()
            if not ex_clean:
                continue
            ex_norm = re.sub(r"\s+", " ", ex_clean).lower()
            if ex_norm == q_norm or ex_norm in seen_ex_norm:
                continue
            # Must contain target word/expression
            if " " in ft_low:
                ft_parts = [p for p in re.findall(r"[a-z]+", ft_low) if p not in ("sb", "sth", "somebody", "something", "one", "ones")]
                if ft_parts and not all(cls.text_contains_form(ex_clean, p) for p in ft_parts[:2]):
                    continue
            else:
                if not (cls.text_contains_form(ex_clean, ft_low) or cls.text_contains_form(ex_clean, bw_low)):
                    continue
                if canonical_pos == "verb" and ft_low == bw_low and not current_example and quote:
                    if not re.search(rf"\b{re.escape(ft_low)}\b", ex_norm):
                        continue

            if anchor and anchor_type in ("prep", "partitive"):
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
                    t.pos_ in ("VERB", "AUX") and (t.dep_ in ("ROOT", "relcl", "ccomp", "advcl", "conj") or t.tag_ in ("VBD", "VBP", "VBZ", "MD"))
                    for t in doc
                )
                if has_end_punct and has_finite_verb and n_words >= 5:
                    sc += 55
                elif not has_end_punct or not has_finite_verb or n_words < 4:
                    sc -= 55  # Heavy penalty against verbless/unpunctuated NP/PP fragments ('earthy language', 'a loan of £60,000')
                if 6 <= n_words <= 22:
                    sc += 15

                # Find target token in `doc`
                t_tok = None
                if " " not in ft_low:
                    for tok in doc:
                        if tok.text.lower() == ft_low or tok.text.lower() == bw_low or tok.lemma_.lower() == bw_low or tok.text.lower() in t_forms:
                            t_tok = tok
                            break
                if t_tok is not None:
                    if canonical_pos == "adj":
                        if is_copula_adj or anchor_type == "prep":
                            if t_tok.dep_ in ("acomp", "ROOT", "attr", "oprd") or any(c.dep_ == "prep" for c in t_tok.children):
                                sc += 45
                            elif t_tok.dep_ == "amod":
                                sc -= 20
                        else:
                            # Canonical pre-nominal `Adj + Noun` (`... their ____ achievement`, `... an ____ opportunity`, `... the ____ task`)
                            if t_tok.dep_ == "amod" and t_tok.head.pos_ in ("NOUN", "PROPN") and t_tok.i < t_tok.head.i:
                                sc += 45
                                if any(c.dep_ in ("det", "poss") and c.i < t_tok.i for c in t_tok.head.children):
                                    sc += 15
                            elif t_tok.head.pos_ in ("NOUN", "PROPN") and t_tok.i > t_tok.head.i:
                                # Postpositive adjective phrase ('an achievement unparalleled in sporting history')
                                sc -= 30
                            elif t_tok.dep_ in ("acomp", "ROOT") and not any(c.dep_ == "prep" for c in t_tok.children):
                                # Bare predicative ('The work was tiring and tedious') is weaker than attributive 'Adj + Noun'
                                sc -= 12
                    elif canonical_pos == "noun":
                        if t_tok.pos_ in ("VERB", "AUX") or t_tok.tag_ in ("VBD", "VBG", "VBN", "VBP", "VBZ"):
                            sc -= 120
                        else:
                            if t_tok.dep_ in ("dobj", "obj") and t_tok.head.pos_ == "VERB" and t_tok.head.lemma_.lower() not in cls._DELEXICAL_VERBS:
                                sc += 45
                            elif t_tok.dep_ in ("nsubj", "nsubjpass") and t_tok.head.pos_ in ("VERB", "AUX") and t_tok.head.lemma_.lower() not in cls._DELEXICAL_VERBS:
                                sc += 35
                            if any(c.dep_ in ("amod", "prep") for c in t_tok.children):
                                sc += 20
                            if has_uncountable_contrast_foil and not is_plural_noun:
                                if any(c.dep_ == "det" and c.text.lower() in ("a", "an", "each", "every") for c in t_tok.children):
                                    sc += 25
                    elif canonical_pos == "verb":
                        if t_tok.pos_ == "NOUN" and t_tok.dep_ in ("pobj", "dobj", "nsubj"):
                            sc -= 80
                        elif t_tok.dep_ == "amod":
                            sc -= 45
                        else:
                            if any(c.dep_ == "xcomp" or c.text.lower() == "to" for c in t_tok.children) or re.search(rf"\b{re.escape(t_tok.text.lower())}\s+to\s+[a-z]+\b", ex_norm):
                                if best_pattern and "to do" in best_pattern.lower():
                                    sc += 55
                                else:
                                    sc += 25
                            if anchor_type == "prep" and anchor and re.search(rf"\b{re.escape(anchor.lower())}\b", ex_norm):
                                sc += 50
                            if any(c.dep_ in ("dobj", "obj") and c.pos_ in ("NOUN", "PROPN") for c in t_tok.children):
                                sc += 40
                            elif ft_low != bw_low and any(c.dep_ == "auxpass" for c in t_tok.children):
                                sc += 40

            # Anchor & pattern alignment
            if anchor and anchor_type not in ("contextual", None):
                anc_l = str(anchor).strip().lower()
                if anc_l and (re.search(rf"\b{re.escape(anc_l)}\w*\b", ex_norm) or cls.text_contains_form(ex_clean, anc_l)):
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
            # 3. Trailing preposition in `pat_str` (`clutter of`, `exposure to`) must appear AFTER the target in `ex_str`
            p_no_inf = re.sub(r"\bto\s+do(?:\s+something|\s+somebody)?\b", "", p_c).strip()
            m_trail_p = re.search(r"\b(of|for|in|to|on|with|from|into|about|against|over|under|between|at|by|upon|towards)$", p_no_inf)
            if m_trail_p:
                prep_w = m_trail_p.group(1)
                last_t = ft_low.split()[-1] if ft_low else bw_low
                last_forms = {last_t} | set(cls.inflected_forms(last_t))
                if prep_w != last_t and not any(
                    re.search(rf"\b{re.escape(lf)}\b(?:\s+[a-z]+){{0,4}}\s+\b{re.escape(prep_w)}\b", ex_l)
                    for lf in last_forms if lf
                ):
                    return False
            # 4. Content words in `pat_str` must appear in `ex_str`
            p_words = [
                w for w in re.findall(r"[a-z]+", p_c)
                if w not in ("something", "somebody", "someone", "sb", "sth", "do", "doing", "etc", "a", "an", "the", "one", "ones", "your", "his", "her", "their")
                and w not in t_forms
                and w not in set(ft_low.split())
            ]
            if p_words and not all(
                re.search(rf"\b{re.escape(pw)}\w*\b", ex_l) or cls.text_contains_form(ex_str, pw)
                for pw in p_words
            ):
                return False
            return True

        synced_pattern = best_pattern
        if synced_pattern and win_ex and not _pattern_matches_example(synced_pattern, win_ex):
            synced_pattern = win_pat if (win_pat and _pattern_matches_example(win_pat, win_ex)) else None

        return win_ex, synced_pattern, win_tier

    @classmethod
    def _resolve_object_animacy_from_example(
        cls,
        example_text: Optional[str],
        target_word: str,
        quote: Optional[str] = None,
        default_tag: str = "[sb/sth]",
    ) -> str:
        """
        Resolves object animacy (`someone` vs. `something`) directly from the winning example sentence
        (falling back to `quote`) so `cloze_slot_frame` never emits slash choices like `someone/something`,
        `him/her/them`, or `it/them`.
        """
        tag_l = (default_tag or "").strip().lower()
        if tag_l in ("[sb]", "sb", "somebody", "someone"):
            return "someone"
        if tag_l in ("[sth]", "sth", "something"):
            return "something"

        human_pronouns = {
            "him", "her", "me", "us", "you", "whom", "who",
            "someone", "somebody", "anyone", "anybody", "everyone", "everybody", "nobody",
            "himself", "herself", "themselves", "ourselves", "yourself", "yourselves",
        }
        human_nouns = {
            "person", "people", "man", "men", "woman", "women", "child", "children", "kid", "boy", "girl",
            "student", "teacher", "professor", "worker", "employee", "employer", "staff", "colleague",
            "customer", "client", "patient", "voter", "reader", "viewer", "listener", "audience", "guest",
            "friend", "family", "parent", "mother", "father", "son", "daughter", "brother", "sister",
            "citizen", "resident", "victim", "suspect", "witness", "prisoner", "leader", "manager",
            "director", "president", "governor", "minister", "official", "politician", "candidate",
            "soldier", "troop", "army", "enemy", "opponent", "rival", "partner", "ally", "neighbor",
            "neighbour", "stranger", "visitor", "tourist", "passenger", "driver", "pilot", "doctor",
            "nurse", "lawyer", "judge", "police", "officer", "author", "writer", "poet", "artist",
            "actor", "actress", "singer", "musician", "player", "athlete", "fan", "crowd", "public",
            "community", "population", "generation", "youth", "adult", "teenager", "baby", "infant",
        }

        nlp = cls.get_spacy()
        tw_l = (target_word or "").strip().lower()
        tw_last = tw_l.split()[-1] if tw_l else ""

        for src_txt in (example_text, quote):
            if not src_txt or not nlp:
                continue
            try:
                doc = nlp(src_txt)
                for tok in doc:
                    if tok.text.lower() == tw_last or tok.lemma_.lower() == tw_last:
                        # Inspect the 1..4 tokens following the target word/particle
                        window = doc[tok.i + 1 : min(len(doc), tok.i + 5)]
                        for w_tok in window:
                            if w_tok.is_punct:
                                break
                            w_low = w_tok.text.lower()
                            w_lem = w_tok.lemma_.lower()
                            if w_low in human_pronouns or w_lem in human_nouns or w_tok.ent_type_ == "PERSON":
                                return "someone"
                            if w_low in ("it", "this", "that", "which", "what") or (
                                w_tok.pos_ == "NOUN" and w_lem not in human_nouns
                            ):
                                return "something"
            except Exception:
                pass
        return "something"

    @classmethod
    def derive_cloze_slot_frame_from_example(
        cls,
        example_text: Optional[str],
        final_target: str,
        base_word: str,
        canonical_pos: str,
        anchor: Optional[str] = None,
        anchor_type: Optional[str] = None,
        is_copula_adj: bool = False,
        is_attributive_only_adj: bool = False,
        is_plural_noun: bool = False,
        is_pure_uncountable_target: bool = False,
        is_target_strictly_countable: bool = False,
        has_mixed_vc_initials: bool = False,
        has_uncountable_contrast_foil: bool = False,
        verb_form_tag: Optional[str] = None,
        forbidden_words: Optional[Set[str]] = None,
    ) -> Optional[str]:
        """
        Slices a compact, authentic 2-to-5-word `cloze_slot_frame` directly from the winning example's
        syntactic neighborhood around the target word, guaranteeing at least one concrete left or right
        context word/POS anchor (`... their ____ achievement`, `... take out a ____`, `... ____ to [verb]`,
        `... ____ for`, `... is ____ in`) and zero bare `None` frames.
        """
        if not example_text or not final_target:
            return None
        nlp = cls.get_spacy()
        if not nlp:
            return None

        ex_clean = re.sub(r"\s*\(=[^)]*\)", "", str(example_text)).strip()
        ft_low = final_target.strip().lower()
        bw_low = (base_word or ft_low).strip().lower()
        t_forms = {ft_low, bw_low} | set(cls.inflected_forms(bw_low)) | set(cls.inflected_forms(ft_low))

        def _is_forbidden(word_str: str) -> bool:
            if not word_str or not forbidden_words:
                return False
            return cls._in_forbidden_word_set(word_str.lower(), forbidden_words)

        def _normalize_det(det_str: str, det_tok_obj=None) -> str:
            if det_tok_obj is not None and det_tok_obj.dep_ == "poss" and det_tok_obj.pos_ in ("NOUN", "PROPN", "ADJ"):
                return "their"
            d_l = (det_str or "").strip().lower()
            if not d_l:
                return ""
            if d_l.endswith("'s") or d_l.endswith("’s"):
                return "their"
            if canonical_pos == "noun":
                if is_pure_uncountable_target or is_plural_noun:
                    if d_l in ("a", "an", "each", "every", "one"):
                        return "the" if is_plural_noun else ""
                elif has_uncountable_contrast_foil and not is_plural_noun:
                    if d_l in ("a", "an", "the"):
                        return "each" if has_mixed_vc_initials else "a"
                elif has_mixed_vc_initials and d_l in ("a", "an"):
                    return "each" if has_uncountable_contrast_foil else "the"
            elif canonical_pos == "adj":
                if has_mixed_vc_initials and d_l in ("a", "an"):
                    return "the"
            return d_l

        try:
            doc = nlp(ex_clean)
        except Exception:
            return None

        # 1. Multi-word phrase / expression
        if " " in ft_low or canonical_pos == "phrase":
            if "one's" in bw_low or "one's" in ft_low:
                ft_clean_for_toks = re.sub(
                    r"\b(?:one's|oneself|someone's|somebody's|my|your|his|her|its|our|their)\b",
                    " ",
                    ft_low,
                )
            else:
                ft_clean_for_toks = re.sub(r"\b(?:one's|oneself|someone's|somebody's)\b", " ", ft_low)
            ft_toks = [t for t in re.findall(r"[a-z]+", ft_clean_for_toks) if t not in ("sb", "sth", "somebody", "something", "one", "ones")]
            if ft_toks:
                first_forms = {ft_toks[0]} | set(cls.inflected_forms(ft_toks[0]))
                last_w = ft_toks[-1]
                start_i, end_i = None, None
                for tok in doc:
                    if tok.text.lower() in first_forms or tok.lemma_.lower() == ft_toks[0]:
                        if ft_toks[0] in ("a", "an", "the") and len(ft_toks) >= 2:
                            if tok.i + 1 >= len(doc) or (
                                doc[tok.i + 1].text.lower() != ft_toks[1]
                                and doc[tok.i + 1].lemma_.lower() != ft_toks[1]
                            ):
                                continue
                        for j in range(tok.i, min(len(doc), tok.i + len(ft_toks) + 4)):
                            if doc[j].text.lower() == last_w or doc[j].lemma_.lower() == last_w:
                                start_i, end_i = tok.i, j + 1
                                break
                        if start_i is not None:
                            break
                if start_i is not None and end_i is not None:
                    _head_ent = cls.get_ldoce_entry(ft_toks[0])
                    _head_primary_pos = str((_head_ent or {}).get("pos") or "").strip().lower()
                    _is_binomial_or_of_np = (
                        (
                            len(ft_toks) == 3
                            and ft_toks[1] in ("and", "or", "by", "to")
                            and (
                                ft_toks[0] == ft_toks[2]
                                or ft_toks[0] in ("back", "up", "down", "in", "out", "on", "off", "more", "sooner", "now", "first", "day", "time", "side", "face", "hand", "step", "word", "pros", "ups", "odds", "ins")
                            )
                        )
                        or (len(ft_toks) >= 3 and ft_toks[1] == "of")
                    )
                    is_verbal_phrase = (
                        not _is_binomial_or_of_np
                        and (
                            doc[start_i].pos_ == "VERB"
                            or (
                                bool(cls._wordnet_pos(ft_toks[0]) & {"v"})
                                and _head_primary_pos not in ("noun", "adjective", "adverb", "preposition", "determiner")
                            )
                        )
                    )
                    left_aux = ""
                    if is_verbal_phrase and start_i > 0 and verb_form_tag not in ("VBD", "VBZ", "VBG", "VBN"):
                        prev_raw = doc[start_i - 1].text.lower()
                        if prev_raw == "'ll":
                            left_aux = "will "
                        elif prev_raw == "'d":
                            left_aux = "would "
                        elif prev_raw in ("to", "will", "would", "could", "can", "might", "may", "should", "must", "did", "do", "does"):
                            left_aux = f"{prev_raw} "
                    # Check if a direct object sits inside a separable phrasal verb ('turned the whole house over', 'wake you up')
                    mid_toks = doc[start_i + 1 : end_i - 1]
                    if len(ft_toks) == 2 and any(t.pos_ in ("PRON", "NOUN", "PROPN") for t in mid_toks):
                        anim = cls._resolve_object_animacy_from_example(ex_clean, ft_toks[0], default_tag="[sb/sth]")
                        return f"... {left_aux}____ {anim}"
                    # If the phrase ends with a transitive preposition ('set out on', 'catch up with', 'listen to')
                    if last_w in ("with", "to", "for", "of", "into", "from", "on", "at", "about", "against", "upon", "over", "after"):
                        anim = cls._resolve_object_animacy_from_example(ex_clean, ft_low, default_tag="[sb/sth]")
                        return f"... {left_aux}____ {anim}"
                    # Check if the phrase starts with a verb and takes a direct object right after `end_i`
                    if is_verbal_phrase and end_i < len(doc):
                        nxt_t = doc[end_i]
                        if nxt_t.pos_ == "ADP" and nxt_t.text.lower() not in ("between", "during", "since", "until", "before", "after"):
                            return f"... {left_aux}____ {nxt_t.text.lower()}"
                        has_dobj_after = (
                            nxt_t.dep_ in ("dobj", "obj")
                            or (
                                nxt_t.pos_ in ("DET", "PRON", "NOUN", "PROPN")
                                and nxt_t.text.lower() not in ("all", "every", "each", "both", "today", "yesterday", "tomorrow")
                                and any(t.dep_ in ("dobj", "obj") for t in doc[end_i : min(len(doc), end_i + 4)])
                            )
                        )
                        if has_dobj_after:
                            anim = cls._resolve_object_animacy_from_example(ex_clean, ft_low, default_tag="[sb/sth]")
                            return f"... {left_aux}____ {anim}"
                    if is_verbal_phrase:
                        if left_aux:
                            return f"... {left_aux}____"
                        if verb_form_tag in ("VBD", "VBZ", "VBG", "VBN"):
                            if end_i < len(doc) and doc[end_i].is_alpha and not _is_forbidden(doc[end_i].lemma_.lower()):
                                return f"... ____ {doc[end_i].text.lower()}"
                            subj_cands = [
                                c for c in doc[start_i].children
                                if c.dep_ in ("nsubj", "nsubjpass") and c.i < start_i and c.is_alpha
                            ]
                            subj_tok = subj_cands[-1] if subj_cands else (
                                doc[start_i - 1]
                                if start_i > 0 and doc[start_i - 1].is_alpha and doc[start_i - 1].pos_ in ("NOUN", "PROPN", "PRON")
                                else None
                            )
                            if subj_tok and subj_tok.pos_ in ("NOUN", "PROPN") and not _is_forbidden(subj_tok.lemma_.lower()):
                                s_low = subj_tok.text.lower()
                                det_cands = [d for d in subj_tok.children if d.dep_ == "det" and d.i < subj_tok.i and d.is_alpha]
                                if det_cands:
                                    d_low = det_cands[-1].text.lower()
                                    if d_low in ("my", "your", "our"):
                                        d_low = "a" if subj_tok.tag_ == "NN" else "the"
                                    return f"... {d_low} {s_low} ____"
                                if subj_tok.tag_ == "NN":
                                    return f"... a {s_low} ____"
                                return f"... {s_low} ____"
                            if re.search(r"\bher\b", ft_low):
                                return "... she ____"
                            if re.search(r"\bhis\b", ft_low):
                                return "... he ____"
                            if re.search(r"\btheir\b", ft_low):
                                return "... they ____"
                            return "... someone ____"
                        return "... to ____"
                    if start_i > 0 and doc[start_i - 1].is_alpha and not _is_forbidden(doc[start_i - 1].lemma_.lower()):
                        prev_w = doc[start_i - 1].text.lower()
                        if prev_w in ("and", "or", "got"):
                            return "... have ____"
                        return f"... {prev_w} ____"
                    if end_i < len(doc) and doc[end_i].is_alpha and not _is_forbidden(doc[end_i].lemma_.lower()):
                        return f"... ____ {doc[end_i].text.lower()}"
            return None

        # Locate target token in `doc`
        t_tok = None
        for tok in doc:
            if tok.text.lower() == ft_low:
                t_tok = tok
                break
        if t_tok is None:
            for tok in doc:
                if tok.text.lower() in t_forms or tok.lemma_.lower() == bw_low:
                    t_tok = tok
                    break
        if t_tok is None:
            return None

        # 2. Adjective (`canonical_pos == "adj"`)
        if canonical_pos == "adj":
            # Check for bound preposition immediately following `t_tok`
            prep_child = next(
                (c for c in t_tok.children if c.dep_ == "prep" and c.i > t_tok.i and c.text.lower() in ("in", "of", "to", "for", "with", "about", "at", "on", "from", "by", "as", "into", "over", "towards", "against")),
                None,
            )
            if is_copula_adj or (prep_child is not None and t_tok.dep_ != "amod") or t_tok.dep_ in ("acomp", "oprd"):
                cop_word = "be"
                if t_tok.head and t_tok.head.pos_ in ("AUX", "VERB") and t_tok.head.i < t_tok.i:
                    cop_word = t_tok.head.text.lower()
                else:
                    for c in t_tok.children:
                        if c.dep_ in ("cop", "aux", "auxpass") and c.i < t_tok.i:
                            cop_word = c.text.lower()
                            break
                if prep_child is not None:
                    return f"... {cop_word} ____ {prep_child.text.lower()}"
                if anchor_type == "prep" and anchor:
                    return f"... {cop_word} ____ {anchor.lower()}"
                return f"... {cop_word} ____"

            # Attributive pre-nominal `Adj + Noun` (`tok.dep_ == "amod"`)
            if t_tok.dep_ == "amod" and t_tok.head and t_tok.head.pos_ in ("NOUN", "PROPN") and t_tok.i < t_tok.head.i:
                head_n = t_tok.head
                head_txt = head_n.text.lower()
                if _is_forbidden(head_n.lemma_.lower()):
                    head_txt = "[noun]"
                det_tok = next(
                    (c for c in head_n.children if c.dep_ in ("det", "poss") and c.i < t_tok.i),
                    None,
                )
                det_str = _normalize_det(det_tok.text if det_tok else "", det_tok_obj=det_tok)
                if det_str:
                    return f"... {det_str} ____ {head_txt}"
                return f"... ____ {head_txt}"

            # Fallback if immediately followed by a noun in surface order
            if t_tok.i + 1 < len(doc) and doc[t_tok.i + 1].pos_ in ("NOUN", "PROPN"):
                nxt_n = doc[t_tok.i + 1]
                head_txt = "[noun]" if _is_forbidden(nxt_n.lemma_.lower()) else nxt_n.text.lower()
                prev_det = doc[t_tok.i - 1].text if (t_tok.i > 0 and doc[t_tok.i - 1].pos_ in ("DET", "PRON")) else ""
                det_str = _normalize_det(prev_det)
                if det_str:
                    return f"... {det_str} ____ {head_txt}"
                return f"... ____ {head_txt}"

            if prep_child is not None:
                return f"... ____ {prep_child.text.lower()}"
            if t_tok.i > 0 and doc[t_tok.i - 1].is_alpha:
                return f"... {doc[t_tok.i - 1].text.lower()} ____"
            return "... the ____ [noun]"

        # 3. Verb (`canonical_pos == "verb"`)
        if canonical_pos == "verb":
            left_aux = ""
            if t_tok.i > 0:
                prev_t = doc[t_tok.i - 1]
                if (
                    verb_form_tag in ("VBN", "VBG")
                    and (prev_t.pos_ in ("AUX", "ADP") or prev_t.text.lower() in ("has", "have", "had", "is", "are", "was", "were", "been", "being", "be", "get", "got"))
                ) or (
                    verb_form_tag == "VBD"
                    and prev_t.text.lower() in ("has", "have", "had", "is", "are", "was", "were", "been", "being", "be", "get", "got")
                ):
                    if prev_t.text.lower() in ("been", "being") and t_tok.i > 1 and doc[t_tok.i - 2].text.lower() in ("has", "have", "had", "is", "are", "was", "were"):
                        left_aux = f"{doc[t_tok.i - 2].text.lower()} {prev_t.text.lower()} "
                    else:
                        left_aux = prev_t.text.lower() + " "

            # 3a. Infinitive complement (`to + V-base`)
            if (
                t_tok.i + 2 < len(doc)
                and doc[t_tok.i + 1].text.lower() == "to"
                and (doc[t_tok.i + 1].tag_ == "TO" or doc[t_tok.i + 2].pos_ in ("VERB", "AUX"))
            ):
                return f"... {left_aux}____ to [verb]"

            # 3b. Bound preposition complement (`pose for`, `correlate with`, `originate in`)
            prep_c = next(
                (c for c in t_tok.children if c.dep_ in ("prep", "agent") and c.i > t_tok.i and c.text.lower() in ("for", "with", "in", "into", "to", "from", "on", "of", "at", "about", "against", "over", "under", "between", "as", "upon", "towards", "by")),
                None,
            )
            if prep_c is not None and (anchor_type == "prep" or not any(c.dep_ in ("dobj", "obj") and c.i < prep_c.i for c in t_tok.children)):
                return f"... {left_aux}____ {prep_c.text.lower()}"

            # 3c. Direct object (`dobj` / `obj` / `oprd`)
            dobj_c = next(
                (c for c in t_tok.children if c.dep_ in ("dobj", "obj", "oprd") and c.i > t_tok.i),
                None,
            )
            if dobj_c is not None:
                if dobj_c.pos_ == "PRON" or dobj_c.ent_type_ == "PERSON" or _is_forbidden(dobj_c.lemma_.lower()):
                    anim = cls._resolve_object_animacy_from_example(ex_clean, ft_low, default_tag="[sb/sth]")
                    return f"... {left_aux}____ {anim}"
                if dobj_c.pos_ in ("NOUN", "PROPN"):
                    # Collect determiners/quantifiers between `t_tok` and `dobj_c`
                    raw_span = list(doc[t_tok.i + 1 : dobj_c.i + 1])
                    poss_idx = next((idx for idx, t in enumerate(raw_span[:-1]) if t.dep_ == "poss" and t.pos_ in ("NOUN", "PROPN")), None)
                    if poss_idx is not None:
                        after_poss = [t.text.lower() for t in raw_span[poss_idx + 1 :] if t.is_alpha and not _is_forbidden(t.lemma_.lower())]
                        span_toks = ["the"] + after_poss if after_poss else [dobj_c.text.lower()]
                    else:
                        span_toks = [
                            t.text.lower()
                            for t in raw_span
                            if t.is_alpha and not _is_forbidden(t.lemma_.lower())
                        ]
                    if 1 <= len(span_toks) <= 4:
                        return f"... {left_aux}____ {' '.join(span_toks)}"
                    return f"... {left_aux}____ {dobj_c.text.lower()}"

            if left_aux:
                return f"... {left_aux}____"

            # 3d. Intransitive verb with subject (`the custom ____`)
            subj_c = next(
                (c for c in t_tok.children if c.dep_ in ("nsubj", "nsubjpass") and c.i < t_tok.i),
                None,
            )
            if subj_c is not None and not _is_forbidden(subj_c.lemma_.lower()):
                if subj_c.pos_ == "PRON":
                    return f"... {subj_c.text.lower()} {left_aux}____"
                s_det = next((c.text.lower() for c in subj_c.children if c.dep_ in ("det", "poss") and c.i < subj_c.i), "")
                s_prefix = f"{s_det} {subj_c.text.lower()}".strip()
                return f"... {s_prefix} {left_aux}____"

            if prep_c is not None:
                return f"... {left_aux}____ {prep_c.text.lower()}"
            if t_tok.i + 1 < len(doc) and doc[t_tok.i + 1].is_alpha and not _is_forbidden(doc[t_tok.i + 1].lemma_.lower()):
                return f"... {left_aux}____ {doc[t_tok.i + 1].text.lower()}"
            if t_tok.i > 0 and doc[t_tok.i - 1].is_alpha and not _is_forbidden(doc[t_tok.i - 1].lemma_.lower()):
                return f"... {doc[t_tok.i - 1].text.lower()} ____"
            return f"... {left_aux}____ something"

        # 4. Noun (`canonical_pos == "noun"`)
        if canonical_pos == "noun":
            det_tok = next(
                (c for c in t_tok.children if c.dep_ in ("det", "poss") and c.i < t_tok.i),
                None,
            )
            amod_tok = next(
                (c for c in t_tok.children if c.dep_ in ("amod", "compound") and c.i < t_tok.i and c.is_alpha and not _is_forbidden(c.lemma_.lower())),
                None,
            )
            if amod_tok is None and t_tok.i > 0 and doc[t_tok.i - 1].pos_ == "ADJ" and doc[t_tok.i - 1].is_alpha and not _is_forbidden(doc[t_tok.i - 1].lemma_.lower()):
                amod_tok = doc[t_tok.i - 1]
            if det_tok is None:
                prev_det_i = (amod_tok.i - 1) if amod_tok is not None else (t_tok.i - 1)
                if prev_det_i >= 0 and doc[prev_det_i].text.lower() in ("a", "an", "the", "each", "every", "his", "her", "their", "its", "our", "my", "your", "no"):
                    det_tok = doc[prev_det_i]
            default_noun_det = (
                ("each" if has_mixed_vc_initials else "a")
                if (has_uncountable_contrast_foil and not is_plural_noun)
                else ("a" if is_target_strictly_countable and not is_plural_noun and not has_mixed_vc_initials else ("the" if not is_pure_uncountable_target else ""))
            )
            det_str = _normalize_det(
                det_tok.text if det_tok else default_noun_det,
                det_tok_obj=det_tok,
            )
            prep_after = next(
                (c for c in t_tok.children if c.dep_ == "prep" and c.i == t_tok.i + 1 and c.text.lower() in ("of", "for", "in", "to", "on", "with", "over", "between", "from", "into", "against", "towards")),
                None,
            )
            if prep_after is None and t_tok.i + 1 < len(doc) and doc[t_tok.i + 1].pos_ == "ADP" and doc[t_tok.i + 1].text.lower() in ("of", "for", "in", "to", "on", "with", "over", "between", "from"):
                prep_after = doc[t_tok.i + 1]

            # 4a. Direct object of a lexical verb (`take out a ____`, `paid off the ____`)
            if t_tok.dep_ in ("dobj", "obj") and t_tok.head and t_tok.head.pos_ == "VERB":
                v_head = t_tok.head
                v_lem = v_head.lemma_.lower()
                if v_lem not in cls._DELEXICAL_VERBS and not _is_forbidden(v_lem):
                    prt_tok = next((c.text.lower() for c in v_head.children if c.dep_ == "prt" and c.i < t_tok.i), "")
                    v_surface = v_head.text.lower() if v_head.tag_ in ("VBD", "VBN", "VB", "VBP") else v_lem
                    v_phrase = f"{v_surface} {prt_tok}".strip()
                    np_mid = f"{det_str} " if det_str else ""
                    if prep_after is not None:
                        return f"... {v_phrase} {np_mid}____ {prep_after.text.lower()}"
                    return f"... {v_phrase} {np_mid}____"

            # 4b. Pre-nominal adjective modifier + noun (+ optional prep) (`a personal ____ for`, `an economic ____`)
            if amod_tok is not None:
                amod_txt = amod_tok.text.lower()
                det_for_amod = det_str
                if det_for_amod in ("a", "an") and amod_txt:
                    det_for_amod = "an" if amod_txt[0] in "aeiou" else "a"
                np_mid = f"{det_for_amod} {amod_txt}".strip()
                if prep_after is not None:
                    return f"... {np_mid} ____ {prep_after.text.lower()}"
                return f"... {np_mid} ____"

            # 4c. Noun + bound preposition (`the ____ of`, `a ____ for`)
            if prep_after is not None:
                d_pref = det_str or (default_noun_det if not is_pure_uncountable_target else "")
                return f"... {d_pref + ' ' if d_pref else ''}____ {prep_after.text.lower()}"

            # 4d. Subject of a lexical verb (`the ____ unfolds`)
            if t_tok.dep_ in ("nsubj", "nsubjpass") and t_tok.head and t_tok.head.pos_ in ("VERB", "AUX") and t_tok.head.i > t_tok.i:
                v_head = t_tok.head
                if v_head.lemma_.lower() not in cls._DELEXICAL_VERBS and not _is_forbidden(v_head.lemma_.lower()):
                    d_pref = det_str or "the"
                    return f"... {d_pref} ____ {v_head.text.lower()}"

            # 4e. Object of a preposition (`in ____`, `on ____`, `under ____`)
            if t_tok.dep_ == "pobj" and t_tok.head and t_tok.head.dep_ == "prep":
                p_left = t_tok.head.text.lower()
                return f"... {p_left} {det_str + ' ' if det_str else ''}____"

            d_pref = det_str or "the"
            return f"... {d_pref} ____"

        # 5. Adverb (`canonical_pos == "adv"`)
        if canonical_pos == "adv":
            if t_tok.head and t_tok.head != t_tok and t_tok.head.is_alpha and not _is_forbidden(t_tok.head.lemma_.lower()):
                if t_tok.i < t_tok.head.i:
                    return f"... ____ {t_tok.head.text.lower()}"
                return f"... {t_tok.head.text.lower()} ____"
            if t_tok.i + 1 < len(doc) and doc[t_tok.i + 1].is_alpha:
                return f"... ____ {doc[t_tok.i + 1].text.lower()}"
            if t_tok.i > 0 and doc[t_tok.i - 1].is_alpha:
                return f"... {doc[t_tok.i - 1].text.lower()} ____"

        return None

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
                    if w_p1 in ("of", "from", "into", "for", "on", "to", "with") and w_p0 in _ditrans_prep_verbs:
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
                r'^(make|find|keep|leave)\s+(?:it|\[[^\]]+\]|\([^)]+\)|\.\.\.)\s+([a-z]+)$',
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
                if lem_w and lem_w != w_lower:
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
                if pv_blocks:
                    pv_senses = []
                    for blk in pv_blocks:
                        blk_hw = str(blk.get("headword") or "").strip()
                        for s_obj in (blk.get("senses") or []):
                            if isinstance(s_obj, dict):
                                s_copy = dict(s_obj)
                                if blk_hw and "phrasal_headword" not in s_copy:
                                    s_copy["phrasal_headword"] = blk_hw
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

            def _tok_matches_target(tok, doc_obj) -> bool:
                if "-" not in w_lower:
                    if (tok.i > 0 and doc_obj[tok.i - 1].text == "-" and not doc_obj[tok.i - 1].whitespace_) or (
                        tok.i + 1 < len(doc_obj) and doc_obj[tok.i + 1].text == "-" and not tok.whitespace_
                    ):
                        return False
                return tok.text.lower() == w_lower or tok.lemma_.lower() == w_lower

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
                if not anchor and "V_INF_ONLY" in word_subcats and q_low_v:
                    if re.search(rf"\b{re.escape(w_lower)}(?:s|es|ed|d|ing)?\s+(?:not\s+)?to\s+[a-z]+\b", q_low_v):
                        anchor = "to"
                        anchor_type = "to_inf"
                        cxn_override_pat = f"{w_lower} to do [sth]"
                        cxn_override_frame = "... ____ to do [sth]"
                # Family 4: Verb + [sb] + to do [sth]
                if not anchor and "V_OBJ_TO_DO" in word_subcats and q_low_v:
                    if re.search(rf"\b{re.escape(w_lower)}(?:s|es|ed|d|ing)?\s+(?:[a-z]+\s+){{1,3}}to\s+[a-z]+\b", q_low_v):
                        anchor = "to"
                        anchor_type = "sb_to_do"
                        cxn_override_pat = f"{w_lower} [sb] to do [sth]"
                        cxn_override_frame = "... ____ [sb] to do [sth]"

                # Verb Step A: Extract the valency complement PP from the authentic quote.
                if not anchor and it["quote"]:
                    try:
                        doc = nlp(it["quote"])
                        v_collocs = cls.get_oxford_collocations(w_lower, pos="verb")
                        consensus_preps = set(v_collocs.get("prep", []))
                        if sense_native_prep:
                            consensus_preps.add(sense_native_prep)
                        for tok in doc:
                            if _tok_matches_target(tok, doc):
                                verified_prep = None   # (token_index, prep, pobj) confirmed by OCD/sense valency
                                unverified_prep = None  # (token_index, prep, pobj) bare syntactic complement
                                for child in tok.children:
                                    if child.dep_ != "prep":
                                        continue
                                    prep_tok = child.text.lower()
                                    if prep_tok not in ("to", "with", "from", "on", "for", "in", "into", "of", "against", "at", "upon", "towards", "as", "under", "about", "over", "between", "by"):
                                        continue
                                    if child.i <= tok.i:
                                        continue  # Post-head linear constraint
                                    child_pobj = None
                                    for pchild in child.children:
                                        if pchild.dep_ == "pobj" and pchild.lemma_.lower() not in cls._ANCHOR_STOPWORDS:
                                            child_pobj = pchild.lemma_.lower()
                                            break
                                    if child_pobj is not None and (prep_tok, child_pobj) in cls._ADJUNCT_PREP_PAIRS:
                                        continue  # Fixed adjunct / interjection (e.g. 'for example')
                                    if prep_tok in consensus_preps:
                                        if verified_prep is None:
                                            verified_prep = (child.i, prep_tok, child_pobj)
                                    elif unverified_prep is None:
                                        unverified_prep = (child.i, prep_tok, child_pobj)
                                chosen = None
                                if verified_prep is not None and unverified_prep is not None:
                                    chosen = verified_prep if verified_prep[0] < unverified_prep[0] else unverified_prep
                                elif verified_prep is not None:
                                    chosen = verified_prep
                                elif unverified_prep is not None:
                                    chosen = unverified_prep
                                if chosen is not None:
                                    anchor, prep_obj = chosen[1], chosen[2]
                                    anchor_type = "prep"
                                    quote_governed_prep = anchor
                                break
                    except Exception:
                        pass

                # Verb Step B1: If no bound preposition found in quote, inspect quote for authentic direct object patient (dobj)
                if not anchor and it["quote"]:
                    try:
                        doc = nlp(it["quote"])
                        for tok in doc:
                            if _tok_matches_target(tok, doc):
                                for child in tok.children:
                                    if (
                                        child.dep_ == "dobj"
                                        and child.lemma_.lower() not in cls._ANCHOR_STOPWORDS
                                        and not cls._in_forbidden_word_set(child.lemma_.lower(), batch_target_words)
                                    ):
                                        obj_form = child.text.lower() if child.tag_ == "NNS" else child.lemma_.lower()
                                        anchor = obj_form
                                        anchor_type = "object"
                                        break
                                if anchor:
                                    break
                    except Exception:
                        pass

                # Verb Step B2: If no bound prep or dobj, check if the verb has an authentic subject noun in quote (nsubj)
                if not anchor and it["quote"]:
                    try:
                        doc = nlp(it["quote"])
                        for tok in doc:
                            if _tok_matches_target(tok, doc):
                                for child in tok.children:
                                    if (
                                        child.dep_ in ("nsubj", "nsubjpass")
                                        and child.lemma_.lower() not in cls._ANCHOR_STOPWORDS
                                        and not cls._in_forbidden_word_set(child.lemma_.lower(), batch_target_words)
                                    ):
                                        subj_form = child.text.lower() if child.tag_ == "NNS" else child.lemma_.lower()
                                        anchor = subj_form
                                        anchor_type = "verb_subject"
                                        break
                                if anchor:
                                    break
                    except Exception:
                        pass

                # Verb Step B3: Fallback to sense-aligned direct noun object from OCD or sense-native prep
                if not anchor:
                    has_sense_inf_or_gerund = bool(
                        initial_atomic_bundle
                        and (initial_atomic_bundle.get("inf_valency") or initial_atomic_bundle.get("gerund_valency"))
                        and initial_atomic_bundle.get("example")
                    )
                    if not has_sense_inf_or_gerund:
                        v_entry = cls.get_oxford_collocations(w_lower, pos="verb")
                        candidates = v_entry.get("colloc_nouns", []) + v_entry.get("noun_after", [])
                        if candidates:
                            anchor = cls._select_sense_aligned_anchor(candidates, definition=it.get("definition"), quote=it.get("quote"), target_word=w_lower, forbidden=batch_target_words)
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
                    elif sense_native_prep and not cls._in_forbidden_word_set(sense_native_prep, batch_target_words):
                        anchor = sense_native_prep
                        anchor_type = "prep"
                    elif n_entry.get("prep"):
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
                        rf"\bit\s+(?:is|was|seems|appears|becomes|gets|'s|would\s+be|will\s+be|can\s+be|could\s+be)\s+(?:not\s+|very\s+|quite\s+|extremely\s+|particularly\s+|especially\s+|too\s+|so\s+)?{re.escape(w_lower)}\b(?:[^.?!;]*?\b(for|of)\s+[^.?!;]+?)?\s+to\s+[a-z]+\b",
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
                # Pre-fetch OCD entry & attested prepositions so Adjective Step A1 never mistakes a clause-level
                # temporal/circumstantial adjunct (e.g. 'remained unsolved for 1,400 years') for a bound adjective preposition.
                adj_entry = cls.get_oxford_collocations(w_lower, pos="adj")
                attested_adj_preps = set()
                if sense_native_prep:
                    attested_adj_preps.add(sense_native_prep.lower())
                if adj_entry and adj_entry.get("prep"):
                    for raw_p in adj_entry["prep"]:
                        for p_tok in re.findall(r"\b(to|for|with|of|in|from|at|about|on|by|into|towards|against|over)\b", str(raw_p).lower()):
                            attested_adj_preps.add(p_tok)
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

                # Adjective Step A: Syntactic Dependency Analysis on Authentic Quote
                if not anchor and it["quote"]:
                    try:
                        doc = nlp(it["quote"])
                        _temporal_pobj_lemmas = {
                            "year", "month", "week", "day", "hour", "minute", "second",
                            "decade", "century", "millennium", "period", "time", "while", "moment", "age", "era"
                        }
                        for tok in doc:
                            if _tok_matches_target(tok, doc):
                                # A1. Predicative Valency: Check if adjective governs a lexicographically attested bound preposition
                                for child in tok.children:
                                    c_low = child.text.lower()
                                    if child.dep_ == "prep" and c_low in ("to", "for", "with", "of", "in", "from", "at", "about", "on"):
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
                                # A2. Attributive Modification: Check if adjective modifies a head noun (dep_ == "amod")
                                if not anchor and not is_not_before_noun_adj and tok.dep_ == "amod" and tok.head and tok.head.pos_ in ("NOUN", "PROPN"):
                                    if (
                                        tok.head.pos_ != "PRON"
                                        and tok.head.text.lower() not in cls._ANCHOR_STOPWORDS
                                        and tok.head.lemma_.lower() != w_lower
                                        and not cls._in_forbidden_word_set(tok.head.lemma_.lower(), batch_target_words)
                                    ):
                                        anchor = tok.head.lemma_.lower()
                                        anchor_type = "modified_noun"
                                # A3. Predicative / Causative Complement: Check if adjective is complement of a causative/copula verb
                                if not anchor and tok.dep_ in ("ccomp", "xcomp", "acomp", "oprd") and tok.head and tok.head.pos_ == "VERB":
                                    head_v = tok.head.lemma_.lower()
                                    if head_v in ("make", "render", "find", "deem", "keep", "consider"):
                                        anchor = head_v
                                        anchor_type = "verb_copula"
                                break
                    except Exception:
                        pass

                # Adjective Step B: Collocation Lexicon Lookup (OCD) if no anchor found in quote
                if not anchor and adj_entry:
                    if not is_not_before_noun_adj:
                        candidates = adj_entry.get("noun_after", []) + adj_entry.get("colloc_nouns", [])
                        if candidates:
                            anchor = cls._select_sense_aligned_anchor(candidates, definition=it.get("definition"), quote=it.get("quote"), target_word=w_lower, forbidden=batch_target_words)
                            if anchor:
                                anchor_type = "modified_noun"
                    if not anchor and sense_native_prep and not cls._in_forbidden_word_set(sense_native_prep, batch_target_words):
                        anchor = sense_native_prep
                        anchor_type = "prep"
                    if not anchor and adj_entry.get("prep"):
                        p_cand = cls._pick_anchor_token(adj_entry["prep"], target_word=w_lower, forbidden=batch_target_words)
                        if p_cand:
                            anchor = p_cand
                            anchor_type = "prep"

                # Adjective Step C: Fall back to locked sense pattern first, then curated academic adjective preposition valency map
                if not anchor:
                    ld_entry = target_entry or cls.get_ldoce_entry(w_lower)
                    if ld_entry:
                        candidate_senses = [locked_sense] if locked_sense else ld_entry.get("senses", [])
                        for s in candidate_senses:
                            for pat in s.get("patterns", []):
                                pat_low_c = pat.lower()
                                # For-To Infinitival Guard: do not extract 'for'/'of'/'to' as a lexical prep from 'it is <adj> (for/of sb) to do'
                                if re.search(rf"\bit\s+is\s+{re.escape(w_lower)}\b.*\bto\b", pat_low_c):
                                    continue
                                for p in ("of", "to", "for", "with", "about", "from", "in", "on", "at"):
                                    if re.search(rf"\b{p}\b", pat_low_c):
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
                    try:
                        doc = nlp(it["quote"])
                        for tok in doc:
                            if _tok_matches_target(tok, doc):
                                head = tok.head
                                if head and head.lemma_.lower() not in cls._ANCHOR_STOPWORDS and head.lemma_.lower() != w_lower:
                                    h_lem = head.lemma_.lower()
                                    if head.pos_ in ("ADJ", "ADV"):
                                        anchor = h_lem
                                        anchor_type = "modifies_adj"
                                        break
                                    elif head.pos_ == "VERB":
                                        anchor = h_lem
                                        anchor_type = "modifies_verb"
                                        break
                    except Exception:
                        pass

                # Priority B: Dictionary lookup from PedLex zero collision anchor
                if not anchor and canonical_pos == "adv":
                    ld_anc, ld_atype, ld_frame, _ = cls.find_ldoce_zero_collision_anchor(
                        target_word=w_lower,
                        pos="adv",
                        definition=it.get("definition"),
                        quote=it.get("quote"),
                        locked_sense=locked_sense
                    )
                    if ld_anc:
                        aligned = cls._select_sense_aligned_anchor(
                            [ld_anc],
                            definition=it.get("definition"),
                            quote=it.get("quote"),
                            target_word=w_lower,
                            forbidden=batch_target_words
                        )
                        if aligned:
                            anchor = aligned
                            anchor_type = ld_atype or "modifies_verb"

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
            if not anchor and it["quote"]:
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
                            if not anchor and tok.head and tok.head.pos_ in ("NOUN", "VERB"):
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
                    elif locked_sense and init_pat and re.search(rf"\b{re.escape(w_lower)}\b", init_pat) and not re.search(rf"\b{re.escape(p_cand)}\b", init_pat):
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
                if canonical_pos in ("function_word", "phrase", "idiom_slot"):
                    return False
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
                if (
                    canonical_pos == "noun"
                    and not is_plural_noun
                    and initial_atomic_bundle
                    and re.search(rf"\b(?:a|an)\s+{re.escape(w_lower)}\b", (initial_atomic_bundle.get("pattern") or "").lower())
                    and c_low[:1].isalpha()
                    and w_lower[:1].isalpha()
                    and (c_low[0] in "aeiou") != (w_lower[0] in "aeiou")
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
                        if canonical_pos == "phrase":
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
                        t_lvl_str = cls.get_word_cefr(w_lower) or "B2"
                        t_rank = cls.CEFR_ORDER.get(t_lvl_str, 4)
                        if t_rank <= 3:
                            fallback_pool = {
                                "v": ["choose", "accept", "decide", "expect", "follow", "notice", "explain", "remain", "manage", "allow"],
                                "n": ["choice", "reason", "matter", "situation", "effort", "problem", "action", "change", "moment", "condition"],
                                "a": ["simple", "common", "certain", "different", "similar", "natural", "clear", "direct", "actual", "special"],
                                "r": ["deeply", "strictly", "tightly", "widely", "carefully", "clearly", "readily", "sharply", "directly", "steadily"]
                            }
                        else:
                            fallback_pool = {
                                "v": ["indicate", "facilitate", "establish", "evaluate", "demonstrate", "generate", "enhance", "maintain", "assess"],
                                "n": ["framework", "perspective", "dimension", "mechanism", "principle", "criterion", "strategy", "phenomenon", "capacity"],
                                "a": ["apparent", "consistent", "distinct", "variable", "significant", "plausible", "rigid", "modest", "tentative"],
                                "r": ["deeply", "strictly", "tightly", "widely", "carefully", "clearly", "readily", "sharply", "directly", "steadily"]
                            }
                        for cand in fallback_pool.get(pos_char, []):
                            if len(cleared_list) >= 3:
                                break
                            if cand != w_lower and cand not in cleared_list and cand not in excl_set and _is_valid_distractor(cand):
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
                else:
                    skeleton_definition = ext_defn
            else:
                skeleton_definition = locked_defn or ext_defn

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
                    if not anchor or anchor_type in ("collocation", "contextual"):
                        if not any(cls.double_key_collision(final_target, d, anchor=pat_prep, anchor_type="prep", pos=canonical_pos)[0] for d in dist_list):
                            anchor = pat_prep
                            anchor_type = "prep"
                            pattern_anchor = pat_prep
                    elif not anchor_in_bundle_ex and anchor_type in ("adj", "verb", "object", "modified_noun", "verb_subject"):
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
            if anchor_type not in _cxn_anchor_types and not has_atomic_valency_ex and (
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
                if ld_anchor:
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
            # If final_target is a multi-word expression (e.g. 'date back to' / 'date back') sharing a PedLex block
            # with a co-variant ('date from'), ensure ldoce_pattern_example actually instantiates final_target's particle/preposition
            # rather than the co-variant ('a custom that dates from the 17th century').
            if ldoce_pattern_example and " " in final_target:
                ft_toks = [t for t in re.findall(r"[a-z]+", final_target.lower()) if t not in {"sb", "sth", "somebody", "something", "one", "ones"}]
                if len(ft_toks) >= 2:
                    second_tok = ft_toks[1]
                    if not re.search(rf"\b{re.escape(second_tok)}\b", ldoce_pattern_example.lower()):
                        replacement_ex = None
                        for raw_ex in ((locked_sense or {}).get("examples") or []):
                            ex_txt = (raw_ex.get("text") if isinstance(raw_ex, dict) else str(raw_ex or "")).strip()
                            if ex_txt and ex_txt.lower() != (it.get("quote") or "").strip().lower():
                                if all(re.search(rf"\b{re.escape(tok)}\w*\b", ex_txt.lower()) for tok in ft_toks[:2]):
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
                                    if all(re.search(rf"\b{re.escape(tok)}\w*\b", ch.lower()) for tok in ft_toks[:2]):
                                        replacement_ex = ch
                                        break
                        if replacement_ex:
                            ldoce_pattern_example = replacement_ex

            # Pre-compute syntactic slot flags (copular/attributive adjective & noun countability)
            # so Six-Source Optimal Example Pre-Selection and Direct Example-to-Frame Slicing honor them.
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

            # 5.3 Six-Source Optimal Example Pre-Selection (GEMINI.md §3.2):
            # Evaluate all 6 PedLex compartments (pattern_bundles, grammar_boxes, collocations/cross_collocations,
            # LDOCE Thesaurus, Language Activator, Oxford Thesaurus) to select the single best, sense-isolated,
            # complete authentic sentence before deriving cloze_slot_frame.
            winning_example_source = "none"
            if not causative_extraposition_pat and not cxn_override_pat and anchor_type not in _cxn_anchor_types:
                opt_ex, opt_pat, winning_example_source = cls.select_optimal_six_source_example(
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
                best_pattern = opt_pat

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
                            best_pattern = re_bundle["pattern"]
                            if re_bundle.get("example"):
                                ldoce_pattern_example = re_bundle["example"]
                        else:
                            best_pattern = None
                    elif ldoce_pattern_example and not any(re.search(rf"\b{re.escape(pp)}\b", ldoce_pattern_example.lower()) for pp in pat_preps_sync):
                        best_pattern = None

            # Derive Authentic Cloze Sentence Frame Prototype if an authentic example is available
            if ldoce_pattern_example:
                try:
                    ex_doc = nlp(ldoce_pattern_example)
                    tok_texts = []
                    replaced = False
                    for tok in ex_doc:
                        if not replaced and (tok.lemma_.lower() == final_target.lower() or tok.text.lower() == final_target.lower()):
                            tok_texts.append("____")
                            replaced = True
                        else:
                            tok_texts.append(tok.text)
                    if replaced:
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
            if final_options != _base_options:
                if canonical_pos not in ("phrase", "idiom_slot", "function_word") and any(
                    opt.lower() != final_target.lower() and cls.double_key_collision(
                        final_target, opt, anchor, anchor_type, pos=canonical_pos, quote=it.get("quote")
                    )[0]
                    for opt in final_options
                ):
                    bad_bases = {
                        b for b, f in zip(_base_options, final_options)
                        if f.lower() != final_target.lower() and cls.double_key_collision(
                            final_target, f, anchor, anchor_type, pos=canonical_pos, quote=it.get("quote")
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
                    obj_raw = m_sep_pv.group(2).lower()
                    obj_tag = "[sb/sth]" if ("somebody" in obj_raw and "something" in obj_raw) or "/" in obj_raw else ("[sb]" if obj_raw in ("somebody", "sb") else "[sth]")
                    best_pattern = f"{final_target} {obj_tag}"
                best_pattern = re.sub(r"\s+", " ", best_pattern).strip()

            # Synthesize explicit object-slot pattern for transitive/prepositional phrasal verbs
            # (e.g. 'catch up with' -> 'catch up with [sb/sth]') when PedLex stores the object slot in phrasal_headword or raw_w_item
            if not best_pattern and canonical_pos == "phrase":
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
                p_clean = re.sub(r"\bdoing\s+something\b", "[doing sth]", p_clean, flags=re.IGNORECASE)
                if _is_cxn_family:
                    p_clean = re.sub(r"\bto\s+do\s+something\b", "to do [sth]", p_clean, flags=re.IGNORECASE)
                    p_clean = re.sub(r"\bsomebody\b", "[sb]", p_clean, flags=re.IGNORECASE)
                    p_clean = re.sub(r"\bsomething\b", "[sth]", p_clean, flags=re.IGNORECASE)
                else:
                    p_clean = re.sub(r"\bto\s+do\s+(?:something|\[sth\])\b", "to [verb]", p_clean, flags=re.IGNORECASE)
                    p_clean = re.sub(r"\bsomebody\b", "someone", p_clean, flags=re.IGNORECASE)
                    p_clean = re.sub(r"\bsomething\b", "something", p_clean, flags=re.IGNORECASE)
                    p_clean = re.sub(r"\[sb\]", "someone", p_clean, flags=re.IGNORECASE)
                    p_clean = re.sub(r"\[sth\]", "something", p_clean, flags=re.IGNORECASE)
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
                    elif any(p_clean.lower().endswith(" " + prep) for prep in ("on", "in", "at", "for", "with", "about", "under", "from", "to", "by", "of", "into")):
                        m_end_prep = re.search(r"\b(on|in|at|for|with|about|under|from|to|by|of|into)$", p_clean.lower())
                        if m_end_prep:
                            cloze_slot_frame = f"... ____ {m_end_prep.group(1)}"

            if cloze_slot_frame and len(re.findall(r"_{2,}", cloze_slot_frame)) > 1:
                cloze_slot_frame = None
            if not cloze_slot_frame and anchor and anchor_type == "prep" and canonical_pos in ("verb", "adj"):
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
                )
                if sliced_frame:
                    _sliced_is_bare = bool(
                        re.match(
                            r"^\.\.\.\s*(?:a|an|the|each|every|his|her|their|its|our|my|your)?\s*_{2,}\s*$",
                            sliced_frame.strip(),
                            re.IGNORECASE,
                        )
                    )
                    if not cloze_slot_frame or _is_bare_det_frame:
                        cloze_slot_frame = sliced_frame
                    elif not _sliced_is_bare:
                        # Prefer `sliced_frame` unless `cloze_slot_frame` preserves a required `anchor` that `sliced_frame` omitted
                        _anc_low = str(anchor or "").strip().lower()
                        _pat_has_anc = bool(_anc_low and re.search(rf"\b{re.escape(_anc_low)}\b", cloze_slot_frame.lower()))
                        _sliced_has_anc = bool(_anc_low and re.search(rf"\b{re.escape(_anc_low)}\b", sliced_frame.lower()))
                        if not (_pat_has_anc and not _sliced_has_anc):
                            cloze_slot_frame = sliced_frame

            # Guaranteed POS-grounded fallback if no example/quote token matched
            if not cloze_slot_frame:
                if canonical_pos == "adj":
                    if is_copula_adj:
                        cloze_slot_frame = f"... be ____ {anchor}" if (anchor and anchor_type == "prep") else "... be ____"
                    elif anchor and anchor_type == "modified_noun":
                        cloze_slot_frame = f"... the ____ {anchor}"
                    elif anchor and anchor_type == "prep":
                        cloze_slot_frame = f"... ____ {anchor}"
                    else:
                        cloze_slot_frame = "... the ____ [noun]"
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
                else:
                    cloze_slot_frame = f"... ____ {resolved_obj_anim}"

            if not is_copula_adj and not is_attributive_only_adj and canonical_pos == "adj" and (cloze_slot_frame or anchor):
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
                        "as a simple past verb, e.g. 'the custom ____' or 'they ____' — do not put 'have/has/had' or 'to' before '____')."
                    )
                elif verb_form_tag == "VBZ":
                    non_base_verb_slot_hint = (
                        f" (Present 3rd-person singular '-s' verb slot required for '{final_target}': place '____' directly after "
                        "a singular subject in the present tense, e.g. 'it ____' or 'the contract ____' — do not put any modal or 'to' before '____')."
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
                    if is_quantifier_frame and quant_head_noun:
                        stem_core += f" (Quantifier head noun slot: blank only '{quant_head_noun}' inside '{w_lower}')."
                    elif is_plural_noun:
                        stem_core += f" (Plural noun slot required: e.g. '{final_target}')."
                    elif is_copula_adj:
                        stem_core += " (Complete predicate required: stem MUST include the linking verb so the adjective predicate is complete; NEVER place before a noun)."
                    elif is_attributive_only_adj:
                        if m_adj_noun:
                            stem_core += f" (Attributive pre-noun slot required: you MUST include the modified noun '{m_adj_noun.group(1)}' immediately after '____' as in '____ {m_adj_noun.group(1)}'; NEVER leave '____' without its head noun)."
                        else:
                            stem_core += " (Attributive pre-noun slot required: place '____' directly before a modified noun; NEVER use after a linking verb)."
                    elif is_be_led_phrase:
                        stem_core += be_led_slot_hint
                    elif is_base_verb_slot:
                        _has_left_aux_in_frame = bool(
                            re.search(
                                r"\b(?:to|will|would|can|could|may|might|should|must|do|does|did)\s+_{2,}",
                                cloze_slot_frame.lower(),
                            )
                        )
                        if "one's" in final_target.lower() or "oneself" in final_target.lower():
                            stem_core += (
                                " (Base-form verbal idiom that already includes 'one's': place '____' after 'to' or a modal verb "
                                "at the end of a clause or before a 'when/if/in' phrase, e.g. 'ready to ____ when needed')."
                            )
                        elif not _has_left_aux_in_frame:
                            stem_core += " (Base-form verb slot: use after a modal verb, infinitive 'to', or a plural/1st-person subject — never after singular 'he/she/it' or 'ended up')."
                    elif non_base_verb_slot_hint:
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
            elif has_mixed_vc_initials and not is_plural_noun and not _blank_after_prep:
                stem_core += " (Mixed vowel/consonant option initials: precede '____' with 'the' or a possessive like 'his/their')."

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

            # Data-Driven Semantic & Grammatical Discriminator (PedLex <GRAM>, Grammar Boxes, Thesaurus & Language Activator)
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
                task_parts.append(discriminator_note)

            if not any(p.startswith("Contrast:") for p in task_parts):
                t_entry_data = cls.get_ldoce_entry(final_target) or cls.get_ldoce_entry(target_base)
                if t_entry_data and t_entry_data.get("language_activator"):
                    for act in t_entry_data["language_activator"]:
                        for act_w in act.get("words", []):
                            act_word = (act_w.get("word") or "").lower()
                            if act_word in dist_base_list:
                                act_def = (act_w.get("definition") or "").strip()
                                if act_def:
                                    task_parts.append(f"Contrast: vs '{act_word}' ({act_def}).")
                                else:
                                    task_parts.append(f"Contrast: vs '{act_word}' (nuance distinction).")
                                break
                        if any(p.startswith("Contrast:") for p in task_parts):
                            break

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

    # =========================================================================
    # SYMBOLIC TRANSLATION MICRO-TASK BLUEPRINTS (translation_quiz)
    # =========================================================================

    @classmethod
    def _classify_vocab_register(cls, word: str, pos: str = "", definition: str = "") -> str:
        """
        Deterministically classifies a vocabulary item's pragmatic register into
        'formal', 'informal', or 'neutral' using PedLex register_tags, definition labels,
        phrasal verb morphology, and CEFR/Latinate profile.
        """
        w_clean = re.sub(r"\[.*?\]|\(.*?\)", "", str(word or "")).strip().lower()
        w_clean = re.sub(r"\s+", " ", w_clean)
        def_low = str(definition or "").lower()
        pos_low = str(pos or "").lower()

        # 1. Explicit bracketed register tags in definition
        if any(t in def_low for t in ("[spoken]", "[informal]", "[slang]", "[colloquial]", "spoken,", "informal]")):
            return "informal"
        if any(t in def_low for t in ("[formal]", "[written]", "[technical]", "[literary]", "[law]", "[medical]")):
            return "formal"

        # 2. PedLex entry & sense register_tags lookup
        entry = cls.get_ldoce_entry(w_clean) or (cls.get_ldoce_entry(w_clean.split()[0]) if " " in w_clean else None)
        if isinstance(entry, dict):
            senses = entry.get("senses") or []
            matched_sense = None
            if def_low:
                norm_target = re.sub(r"[^a-z0-9 ]", "", def_low).strip()
                for s in senses:
                    s_def = re.sub(r"[^a-z0-9 ]", "", str(s.get("definition") or "").lower()).strip()
                    if norm_target and s_def and (norm_target in s_def or s_def in norm_target):
                        matched_sense = s
                        break
            target_senses = [matched_sense] if matched_sense else senses[:2]
            for s in target_senses:
                if not isinstance(s, dict):
                    continue
                reg_tags = [str(r).lower() for r in (s.get("register_tags") or [])]
                dom_tags = [str(d).lower() for d in (s.get("domain_tags") or [])]
                s_def_txt = str(s.get("definition") or "").lower()
                if any(r in reg_tags for r in ("spoken", "informal", "slang", "colloquial", "everyday")) or "[spoken]" in s_def_txt or "[informal]" in s_def_txt:
                    return "informal"
                if any(r in reg_tags for r in ("formal", "written", "technical", "literary", "academic")) or any(d in dom_tags for d in ("law", "science", "medicine", "economics")) or "[formal]" in s_def_txt:
                    return "formal"

        # 3. Multi-word phrasal verbs with conversational particles default to informal
        conversational_particles = {"up", "out", "off", "down", "around", "away", "back", "over", "along", "in", "through"}
        tokens = w_clean.split()
        if pos_low == "phrasal verb" or (len(tokens) == 2 and tokens[1] in conversational_particles):
            # Exclude formal prepositional verbs like "originate in", "comply with", "consist of", "result in"
            formal_prep_verbs = {"originate", "comply", "consist", "culminate", "pertain", "adhere", "concur", "abstain", "deviate", "reside", "prevail"}
            if tokens[0] not in formal_prep_verbs:
                return "informal"

        # 4. Formal academic / Latinate headwords
        formal_academic_roots = {
            "originate", "regulatory", "comply", "subsequent", "consequently", "notwithstanding",
            "constitute", "implement", "predominant", "facilitate", "mandate", "jurisdiction",
            "stipulate", "commence", "endeavor", "ascertain", "proliferate", "mitigate",
            "exacerbate", "unprecedented", "indispensable", "autonomous", "empirical", "paradigm",
        }
        if w_clean in formal_academic_roots:
            return "formal"

        if " " not in w_clean and len(w_clean) >= 8:
            cefr_lvl = cls.get_word_cefr(w_clean)
            if cefr_lvl in ("C1", "C2") and w_clean.endswith(("ate", "ory", "tion", "ive", "ity", "ence", "ance", "ous")):
                return "formal"

        return "neutral"

    @classmethod
    def _classify_grammar_register(cls, name: str, formula: str = "", pedagogical_function: str = "") -> str:
        """
        Deterministically classifies a grammar pattern's stylistic register into
        'formal', 'informal', or 'neutral'.
        """
        combined = f"{name or ''} {formula or ''} {pedagogical_function or ''}".lower()

        # Check informal / conversational markers first
        if any(k in combined for k in (
            "[spoken]", "[informal]", "conversational", "spoken", "informal",
            "question tag", "ellipsis", "get-passive", "everyday coordination",
            "simple coordination", "colloquial", "casual"
        )):
            return "informal"

        # Check formal / elevated academic syntax
        if any(k in combined for k in (
            "inversion", "not only", "seldom", "rarely", "never ", "little + aux",
            "only when", "only by", "only after", "participial", "participle",
            "adjunct", "having +", "v-ing,", "v-ed,", "shell noun", "appositive",
            "cleft", "nominalization", "extraposition", "preposition + which",
            "preposition + whom", "formal", "academic", "elevated"
        )):
            return "formal"

        return "neutral"

    @classmethod
    def _resolve_translation_lexical_trap(
        cls,
        word: str,
        pos: str = "",
        definition: str = "",
        quote: str = "",
        example_usage: str = "",
    ) -> Dict[str, Any]:
        """
        Resolves the target keyword's Required Lexical Frame, Model Example, and
        optional single-point Lexical Error Trap from `pedlex_error_traps` (with
        deterministic symbolic fallback when pedlex.db is absent or unindexed).
        Enforces strict POS and `sense_idx` isolation.
        """
        w_clean = re.sub(r"\[.*?\]|\(.*?\)", "", str(word or "")).strip()
        w_clean = re.sub(r"\s+", " ", w_clean)
        def_str = str(definition or "").strip()
        pos_str = str(pos or "").strip()

        # 1. Query atomic pattern bundle for canonical pattern & authentic example
        locked_sense = None
        s_idx: Optional[int] = None
        entry = cls.get_ldoce_entry(w_clean)
        if entry and entry.get("senses"):
            s_idx = cls._lock_sense(
                entry,
                definition=def_str,
                quote=quote,
                target_pos=pos_str or None,
            )
            if s_idx is not None and 0 <= s_idx < len(entry["senses"]):
                locked_sense = entry["senses"][s_idx]

        atomic_bundle = (
            cls.pick_atomic_pattern_bundle(locked_sense, w_clean, quote=quote)
            if locked_sense
            else None
        )
        bundle_pat = str((atomic_bundle.get("pattern") if isinstance(atomic_bundle, dict) else "") or "").strip()
        bundle_ex = str((atomic_bundle.get("example") if isinstance(atomic_bundle, dict) else "") or "").strip()
        if bundle_pat:
            bundle_pat = re.sub(r"\s+\betc\.?\b", "", bundle_pat).strip()

        if not bundle_pat and isinstance(locked_sense, dict):
            sense_pats = [
                re.sub(r"\s+\betc\.?\b", "", str(p)).strip()
                for p in (locked_sense.get("patterns") or [])
                if str(p).strip()
            ]
            if sense_pats:
                q_low = f"{quote} {example_usage}".lower()
                matched_pats = [p for p in sense_pats if p.lower() in q_low]
                bundle_pat = matched_pats[0] if matched_pats else sense_pats[0]

        # Filter out conversational speech-act formulas, personal deictic phrases,
        # or noisy proper-noun exemplar strings so narrative sentences are clean.
        _dialogic_deictic_re = re.compile(
            r"\b(?:thanks|thank\s+you|hello|hi|sorry|please|excuse\s+me|pardon|your|my|our|you're|i'm|i\s+am|don't\s+worry|having\s+us)\b",
            re.IGNORECASE,
        )
        if bundle_pat and (
            _dialogic_deictic_re.search(bundle_pat)
            or "/" in bundle_pat and any(c.isupper() for c in bundle_pat)
            or len(bundle_pat.split()) > 5
        ):
            bundle_pat = ""

        # Ensure model_example actually contains the target word stem and is not a dialogic greeting/thanks formula
        w_stem = w_clean.lower()[: max(3, min(len(w_clean), 5))]
        candidate_examples = [bundle_ex, example_usage, quote]
        if isinstance(locked_sense, dict):
            candidate_examples.extend(str(ex) for ex in (locked_sense.get("examples") or []) if str(ex).strip())
        model_example = ""
        for cand_ex in candidate_examples:
            cand_s = str(cand_ex or "").strip()
            if not cand_s:
                continue
            if w_stem in cand_s.lower() and not _dialogic_deictic_re.search(cand_s):
                model_example = cand_s
                break
        if not model_example:
            for cand_ex in candidate_examples:
                cand_s = str(cand_ex or "").strip()
                if cand_s and not _dialogic_deictic_re.search(cand_s):
                    model_example = cand_s
                    break

        # For nouns, do not force a bare generic genitive 'Noun + of' frame (e.g. 'emblem of', 'population of')
        # as Required Lexical Frame, since standalone modified nouns ('its official emblem', 'the growing population')
        # would otherwise force the LLM to invent tautological 'of + NP' tails ('population of residents').
        _generic_noun_of_re = re.compile(
            rf"^{re.escape(w_clean)}\s+of$",
            re.IGNORECASE,
        )
        if pos_str.lower() in ("noun", "n") and bundle_pat and _generic_noun_of_re.match(bundle_pat.strip()):
            bundle_pat = ""

        if not bundle_pat:
            ctx_sample = f"{model_example} {quote} {definition}"
            if pos_str.lower() == "noun" and re.search(rf"\bthe\s+{re.escape(w_clean)}\s+of\b", ctx_sample, re.IGNORECASE):
                bundle_pat = f"the {w_clean} of [sth]"
            elif pos_str.lower() == "verb":
                m_vp = re.search(rf"\b{re.escape(w_clean)}(?:s|ed|ing|med|ming)?\s+(into|in|on|to|with|from|for|of)\b", ctx_sample, re.IGNORECASE)
                if m_vp:
                    bundle_pat = f"{w_clean} {m_vp.group(1).lower()} [sth]"
        required_frame = bundle_pat or w_clean

        # Helper to build canonical uncountable pluralization form (e.g. hospitality -> hospitalities)
        def _pluralize_uncountable_noun(noun_str: str) -> str:
            n_low = noun_str.strip()
            if re.search(r"[^aeiou]y$", n_low, re.IGNORECASE):
                return n_low[:-1] + "ies"
            if re.search(r"(?:s|x|z|ch|sh)$", n_low, re.IGNORECASE):
                return n_low + "es"
            return n_low + "s"

        # 2. Query pedlex_error_traps with strict POS and sense_idx isolation
        traps = cls.get_word_error_traps(w_clean, pos=pos_str or None, sense_idx=s_idx)
        best_trap: Optional[Dict[str, Any]] = None
        if traps:
            priority_order = {
                "GRAM_BOX_DONTSAY": 0,
                "GRAMBOX_BADEXA": 0,
                "GRAMBOX_RULE_TRAP": 1,
                "PREP_COLLOC_TRAP": 2,
                "SYNTAX_COMP_TRAP": 3,
                "GRAM_ATTRIBUTIVE_ONLY": 4,
                "GRAM_NOT_BEFORE_NOUN": 4,
                "GRAM_UNCOUNTABLE": 5,
                "GRAM_INTRANSITIVE": 6,
                "L1_CONJUNCTION_REDUNDANCY": 7,
                "CLAUSE_PREP_CONFUSION": 7,
                "GRAM_PROPERTY_TRAP": 8,
            }
            def _is_invalid_prep_or_dialogic_trap(t_obj: Dict[str, Any]) -> bool:
                t_type = str(t_obj.get("trap_type") or "")
                cf = str(t_obj.get("correct_frame") or "").strip()
                cf_low = cf.lower()
                wf_low = str(t_obj.get("wrong_frame") or "").strip().lower()
                if _dialogic_deictic_re.search(cf):
                    return True
                if t_type != "PREP_COLLOC_TRAP":
                    return False
                # Reject generic nominal 'Noun + of' preposition traps (e.g. 'emblem of' vs '*emblem from*',
                # 'population of' vs '*population from*'), which force artificial tautologies on standalone nouns.
                if pos_str.lower() in ("noun", "n") and _generic_noun_of_re.match(cf):
                    return True
                if not re.search(r"\bto\b", cf_low):
                    return False
                # If 'to' is at the end of the frame (e.g. 'Is it possible to', 'able to') or followed by a bare verb ('to do', 'to be'),
                # it is an infinitive marker rather than a preposition governing a noun phrase.
                if re.search(r"\bto\s*$", cf_low) or re.search(r"\bto\s+(?:do|be|have|get|make|take|go|see|find|say|know)\b", cf_low):
                    return True
                if ("for/with" in wf_low or "for" in wf_low) and pos_str.lower() in ("adjective", "adj") and not re.search(r"\bto\s+(?:somebody|something|sb|sth|\[)", cf_low):
                    return True
                return False

            valid_traps = [
                t for t in traps
                if str(t.get("trap_type") or "") in priority_order and not _is_invalid_prep_or_dialogic_trap(t)
            ]
            if valid_traps:
                sorted_traps = sorted(
                    valid_traps,
                    key=lambda t: priority_order.get(str(t.get("trap_type") or ""), 99)
                )
                best_trap = dict(sorted_traps[0])
                if str(best_trap.get("trap_type") or "") == "GRAM_UNCOUNTABLE":
                    pl_form = _pluralize_uncountable_noun(w_clean)
                    best_trap["wrong_frame"] = f"*{pl_form}* (uncountable pluralization)"
                c_f = str(best_trap.get("correct_frame") or "").strip()
                t_type_sel = str(best_trap.get("trap_type") or "")
                if (
                    c_f
                    and required_frame == w_clean
                    and t_type_sel in ("PREP_COLLOC_TRAP", "SYNTAX_COMP_TRAP")
                    and "[" not in c_f
                    and not c_f.endswith((".", "!", "?"))
                    and not _dialogic_deictic_re.search(c_f)
                    and not (pos_str.lower() in ("noun", "n") and _generic_noun_of_re.match(c_f))
                    and len(c_f.split()) <= 6
                ):
                    required_frame = c_f
                trap_mod_ex = str(best_trap.get("model_example") or "").strip()
                if trap_mod_ex and not model_example and not _dialogic_deictic_re.search(trap_mod_ex):
                    model_example = trap_mod_ex

        # 3. Deterministic fallback for explicit <gram> properties or bound prepositions
        if not best_trap:
            def_low = def_str.lower()
            if ("[only before noun]" in def_low or "attributive" in pos_str.lower()) and pos_str.lower() in ("adjective", "adj", ""):
                best_trap = {
                    "trap_type": "GRAM_ATTRIBUTIVE_ONLY",
                    "correct_frame": f"[only before noun] ({w_clean} + Noun)",
                    "wrong_frame": f"*be + {w_clean}* (predicative misuse)",
                    "pedagogical_note": f"'{w_clean}' is an attributive-only adjective ([only before noun]) and cannot be used as a predicate after a linking verb.",
                }
            elif "[uncountable]" in def_low and pos_str.lower() in ("noun", "n", ""):
                pl_form = _pluralize_uncountable_noun(w_clean)
                best_trap = {
                    "trap_type": "GRAM_UNCOUNTABLE",
                    "correct_frame": f"[uncountable] ({w_clean})",
                    "wrong_frame": f"*{pl_form}* (uncountable pluralization)",
                    "pedagogical_note": f"'{w_clean}' is an uncountable noun and must not be pluralized as '*{pl_form}*'.",
                }
            else:
                # Check common bound-preposition collocation frames
                prep_fallbacks = {
                    "originate": ("originate in [place/situation]", "*originate from / originate at*", "Many Christmas traditions originated in Germany."),
                    "comply": ("comply with [rule/law]", "*comply to / comply by*", "All companies must comply with the new safety regulations."),
                    "depend": ("depend on [sb/sth]", "*depend of / depend from*", "Our success depends on everyone's cooperation."),
                    "rely": ("rely on [sb/sth]", "*rely in / rely at*", "Many rural communities rely on public transport."),
                    "consist": ("consist of [parts]", "*be consisted of / consist in*", "The committee consists of ten senior members."),
                    "result": ("result in [outcome]", "*result to*", "The policy change resulted in a sharp drop in emissions."),
                    "contribute": ("contribute to [sth/V-ing]", "*contribute for*", "Regular exercise contributes to better mental health."),
                    "adapt": ("adapt to [new situation]", "*adapt with*", "Species must adapt to changing environments."),
                    "participate": ("participate in [activity]", "*participate on / participate to*", "Students are encouraged to participate in seminar discussions."),
                    "succeed": ("succeed in [sth/V-ing]", "*succeed to do*", "She succeeded in persuading the board."),
                    "insist": ("insist on [sth/V-ing]", "*insist to do*", "He insisted on paying for the meal."),
                }
                w_low = w_clean.lower()
                if w_low in prep_fallbacks:
                    c_frame, w_frame, mod_ex = prep_fallbacks[w_low]
                    if required_frame == w_clean or required_frame.startswith(f"{w_low} "):
                        required_frame = c_frame
                    if not model_example:
                        model_example = mod_ex
                    best_trap = {
                        "trap_type": "PREP_COLLOC_TRAP",
                        "correct_frame": c_frame,
                        "wrong_frame": w_frame,
                        "pedagogical_note": f"Use the collocation '{c_frame}', not {w_frame}.",
                    }
                elif bundle_pat and " " in bundle_pat:
                    m_prep = re.search(rf"\b{re.escape(w_low)}\s+(in|on|at|to|with|for|from|of|into|about|over|against)\b", bundle_pat.lower())
                    if m_prep:
                        prep_tok = m_prep.group(1)
                        bp_low = bundle_pat.strip().lower()
                        is_inf_to = (
                            prep_tok == "to"
                            and (
                                bp_low.endswith(" to")
                                or bool(re.search(r"\bto\s+(?:do|be|have|get|make|take|go|see|find|say|know)\b", bp_low))
                                or (pos_str.lower() in ("adjective", "adj") and not re.search(r"\bto\s+(?:somebody|something|sb|sth|\[)", bp_low))
                            )
                        )
                        is_generic_noun_of = pos_str.lower() in ("noun", "n") and prep_tok == "of"
                        if not is_inf_to and not is_generic_noun_of:
                            alt_prep = {"in": "from/at", "into": "at/on", "with": "to", "on": "of", "to": "for/with", "of": "from", "from": "of", "for": "to"}.get(prep_tok, "to")
                            best_trap = {
                                "trap_type": "PREP_COLLOC_TRAP",
                                "correct_frame": bundle_pat,
                                "wrong_frame": f"*{w_clean} {alt_prep}*",
                                "pedagogical_note": f"Use '{bundle_pat}' rather than substituting '*{w_clean} {alt_prep}*'.",
                            }

        # Format human-readable lexical flaw label and corruption instruction
        lex_flaw_label = ""
        lex_corruption_step = ""
        if best_trap:
            t_type = str(best_trap.get("trap_type") or "")
            c_frame = str(best_trap.get("correct_frame") or required_frame).strip()
            w_frame = str(best_trap.get("wrong_frame") or "").strip()
            dont_say = str(best_trap.get("dont_say") or best_trap.get("bad_example") or "").strip()
            if t_type == "PREP_COLLOC_TRAP":
                lex_flaw_label = f"Lexical Valency / Preposition Trap (`{c_frame}` vs `{w_frame}`)"
                lex_corruption_step = f"substituting the wrong preposition `{w_frame}` instead of `{c_frame}`"
            elif t_type == "SYNTAX_COMP_TRAP":
                lex_flaw_label = f"Subcategorization Complement Trap (`{c_frame}` vs `{w_frame}`)"
                lex_corruption_step = f"replacing the required complement frame `{c_frame}` with `{w_frame}`"
            elif t_type == "GRAM_UNCOUNTABLE":
                pl_form = _pluralize_uncountable_noun(w_clean)
                lex_flaw_label = f"Uncountable Noun Pluralization Trap (`{c_frame}` -> `*{pl_form}*`)"
                lex_corruption_step = (
                    f"pluralizing the uncountable noun `{w_clean}` into `*{pl_form}*` while keeping all surrounding words and determiners identical"
                )
            elif t_type in ("GRAM_ATTRIBUTIVE_ONLY", "GRAM_NOT_BEFORE_NOUN", "GRAM_INTRANSITIVE", "GRAM_PROPERTY_TRAP"):
                lex_flaw_label = f"Lexical Grammar Property Violation (`{c_frame}` -> `{w_frame}`)"
                lex_corruption_step = f"violating `{c_frame}` by using `{w_frame}`"
            elif t_type in ("GRAM_BOX_DONTSAY", "GRAMBOX_BADEXA", "GRAMBOX_RULE_TRAP"):
                bad_ex = f" (Do NOT say: {dont_say})" if dont_say else f" (`{w_frame}`)"
                lex_flaw_label = f"Lexical Collocation / Usage Trap on `{w_clean}`{bad_ex}"
                lex_corruption_step = f"introducing the authentic learner error `{w_frame or dont_say}` instead of `{c_frame}`"

        return {
            "required_frame": required_frame,
            "model_example": model_example,
            "trap": best_trap,
            "flaw_label": lex_flaw_label,
            "corruption_step": lex_corruption_step,
        }

    @classmethod
    def _resolve_translation_syntactic_trap(cls, gram_item: Dict[str, str]) -> Dict[str, str]:
        """
        Deterministically extracts a Single-Point Syntactic L1 Negative-Transfer Trap
        from a grammar pattern and its `Grammar Warning (<TOPIC>)` common mistake.
        Matches strictly against the pattern name, formula, and warning topic header
        (never matching incidental prose words inside the explanation body).
        """
        name = str(gram_item.get("name") or "Target Grammar Pattern").strip()
        formula = str(gram_item.get("formula") or name).strip()
        mistakes = str(gram_item.get("common_mistakes") or "").strip()

        m_topic = re.search(r"(?:Grammar\s+Warning\s*)?\(([^)]+)\)", mistakes, re.IGNORECASE)
        warn_topic = m_topic.group(1).strip() if m_topic else ""
        header_low = f"{name} {formula} {warn_topic}".lower()
        name_form_low = f"{name} {formula}".lower()
        form_low = formula.lower()

        if any(k in header_low for k in ("although", "even though", "though")) and "despite" not in header_low:
            return {
                "flaw_label": "L1 Conjunction Redundancy (`Although ... but ...` double conjunction)",
                "corruption_step": "adding redundant `but` before the main clause in an `Although / Even though` sentence",
                "constraint_note": "NO 'but' in main clause",
                "zh_cue": "虽然...，但...",
            }
        if any(k in header_low for k in ("despite", "in spite of")):
            return {
                "flaw_label": "Clause-Preposition Confusion (`Despite / In spite of + *finite clause*`)",
                "corruption_step": "placing a bare finite clause directly after `Despite / In spite of` instead of a noun phrase, `-ing` gerund, or `the fact that`",
                "constraint_note": "followed by NP / V-ing / the fact that, NOT a bare finite clause",
                "zh_cue": "尽管...，...",
            }
        if (
            "resultative particip" in name_form_low
            or "post-positioned present particip" in name_form_low
            or "comma splice" in header_low
            or bool(re.search(r"\[vp\]\s*,\s*\[v-ing", form_low))
        ):
            return {
                "flaw_label": "Resultative Participle vs. Comma Splice / Voice Trap (replacing `, [V-ing]` with passive `*, V-ed*` or an uncoordinated finite verb)",
                "corruption_step": "replacing the post-positioned present participle `, [V-ing]` with a passive participle `*, V-ed*` or a bare finite verb (comma splice)",
                "constraint_note": "attach `, [V-ing Phrase]` directly after the main clause without a conjunction",
                "zh_cue": "……，从而造成了/象征着……",
            }
        if (
            "passive" in name_form_low
            or "unaccusative" in name_form_low
            or "be/modal be" in form_low
            or bool(re.search(r"\[be[^\]]*\]\s*\+\s*\[v(?:-ed|3)\]", form_low))
        ):
            return {
                "flaw_label": "Passive Voice / Unaccusative Verb Trap (malformed passive auxiliary/participle or passivized intransitive verb)",
                "corruption_step": "corrupting the passive verb form (`[be] + [V-ed]` -> active/base verb or missing `be`) or passivizing an intransitive verb",
                "constraint_note": "use `[be/modal be] + past participle [V-ed]`",
                "zh_cue": "被……/由……",
            }
        if (
            "causative" in header_low
            or "bare infinitive" in header_low
            or "bare verb" in header_low
            or bool(re.search(r"\b(?:let|make|have)\s*\+\s*\[", form_low))
        ):
            return {
                "flaw_label": "Causative Infinitive Trap (`let/make/have + Object + *to do*` instead of bare infinitive)",
                "corruption_step": "inserting ungrammatical `*to*` before the infinitive verb after causative `let / make / have` (e.g., `*let sb. to do*`)",
                "constraint_note": "use a bare infinitive (NO 'to') after `let / make / have + Object`",
                "zh_cue": "让/使某人做……",
            }
        if (
            any(k in header_low for k in ("inversion", "negative adv", "seldom", "rarely", "never", "little", "only when", "only by", "only after"))
            or ("not only" in header_low and "correlative" not in header_low and "aux" in form_low)
        ):
            return {
                "flaw_label": "Failed Subject-Auxiliary Inversion (keeping declarative Subject + Verb order)",
                "corruption_step": "failing to invert the auxiliary verb and subject after the fronted negative/restrictive adverbial",
                "constraint_note": "invert Auxiliary + Subject in the main/fronted clause",
                "zh_cue": "不仅/很少/只有...",
            }
        if "correlative" in header_low or ("not only" in header_low and "but also" in header_low):
            return {
                "flaw_label": "Correlative Parallelism Misplacement (`not only ... but also` unbalanced constituent slots)",
                "corruption_step": "breaking parallel constituent structure after `not only` and `but also`",
                "constraint_note": "place parallel grammatical constituents immediately after `not only` and `but also`",
                "zh_cue": "不仅……，而且……",
            }
        if any(k in header_low for k in ("particip", "v-ing", "v-ed", "adjunct", "having +")):
            return {
                "flaw_label": "Dangling Participle / Participle Voice Confusion",
                "corruption_step": "misusing the participle voice (`-ing` vs `-ed`) or inserting a redundant conjunction (`and`/`so`) before the main clause",
                "constraint_note": "align participle subject with main-clause subject without redundant conjunction",
                "zh_cue": "伴随/分词状语...",
            }
        if any(k in header_low for k in ("shell noun", "appositive")):
            return {
                "flaw_label": "Appositive Clause Connector Trap (using `*which*` instead of appositive `that`)",
                "corruption_step": "replacing the appositive complementizer `that` with `*which*` after the shell noun",
                "constraint_note": "use appositive `that` + complete clause",
                "zh_cue": "……这一事实/观点",
            }
        if any(k in header_low for k in (", which", "non-restrictive", "anaphoric", "sentential relative")):
            return {
                "flaw_label": "Non-Restrictive Relative Clause Trap (`*, that*` or comma splice)",
                "corruption_step": "replacing `, which` with ungrammatical `*, that*` or omitting the relative pronoun in a comma splice",
                "constraint_note": "use `, which` + singular verb",
                "zh_cue": "……，这意味着/表明……",
            }
        if (
            "restrictive relative" in name_form_low
            or "relative clause" in name_form_low
            or bool(re.search(r"\[np\]\s*\+\s*(?:that|who|which)\s*\+\s*\[vp\]", form_low))
        ):
            return {
                "flaw_label": "Relative Clause Verb Agreement Trap (mismatching verb number agreement with the antecedent `[NP]` inside the `that/who/which` clause)",
                "corruption_step": "mismatching singular/plural verb agreement inside the relative clause (`that/who/which + *singular/plural verb mismatch*`) against the antecedent `[NP]`",
                "constraint_note": "strictly match relative-clause verb number agreement to the antecedent `[NP]` in `[NP] + that/who/which + [VP]`",
                "zh_cue": "……的（定语从句）",
            }
        if any(k in header_low for k in ("conditional", "unless", "as long as", "provided that", "stipulative")) or form_low.startswith("if "):
            return {
                "flaw_label": "Conditional Tense / Modal Trap (inserting modal `*will / would*` inside the conditional clause)",
                "corruption_step": "inserting modal `*will*` or `*would*` inside the conditional subordinate clause",
                "constraint_note": "use present/past indicative tense (NOT 'will/would') inside the conditional clause",
                "zh_cue": "如果/除非……，就……",
            }
        if any(k in header_low for k in ("cleft", "extraposition", "dummy-it", "it + [be]", "it is")):
            return {
                "flaw_label": "Extraposition / Cleft Structure Trap (missing dummy `It` or malformed connector)",
                "corruption_step": "omitting dummy subject `It` or replacing the required `that/to` complementizer",
                "constraint_note": "preserve `It is ... that/to` frame",
                "zh_cue": "正是/……是必要的",
            }
        if (
            any(k in header_low for k in ("preparatory", "object complement", "complex-transitive", "complex transitive"))
            or bool(re.search(r"\b(?:find|make|keep|consider|think|deem)\s*\+\s*\[(?:np|it|object)\]\s*\+\s*\[adj", form_low))
        ):
            return {
                "flaw_label": "Complex-Transitive / Preparatory-It Trap (omitting dummy object `it` before `[adj] + to-V` or using an `*-ly*` adverb instead of `[adj]`)",
                "corruption_step": "omitting dummy object `it` in `find/make it + [adj] + to do` (e.g., `*find hard to do*`) or replacing the object-complement adjective `[adj]` with an `*-ly*` adverb",
                "constraint_note": "keep `find/make + [NP / dummy it] + [adj]` intact",
                "zh_cue": "觉得/发现……是……的",
            }
        if any(
            k in header_low
            for k in (
                "conjunctive adverb",
                "transition adverb",
                "discourse stance",
                "however,",
                "therefore,",
                "nevertheless,",
                "consequently,",
                "moreover,",
            )
        ):
            return {
                "flaw_label": "Conjunctive Adverb Comma Splice (using a comma instead of `;` or `.` before `however / therefore / nevertheless` when joining two independent clauses)",
                "corruption_step": "joining two independent clauses with only a comma before `however / therefore / nevertheless` (comma splice) or adding a redundant conjunction (`*However, ... but ...*`)",
                "constraint_note": "use a period or semicolon before `However, / Therefore,` when linking two independent clauses",
                "zh_cue": "然而/因此，……",
            }

        clean_warn = re.sub(r"^(?:Grammar\s+Warning\s*)?\([^)]*\)\s*:?\s*", "", mistakes, flags=re.IGNORECASE).strip()
        if clean_warn:
            # Split at sentence boundary, colon+quote example, or example markers without splitting on 'sb.', 'sth.', 'vs.', 'e.g.'
            parts = re.split(
                r"(?<!\bvs)(?<!\bsb)(?<!\bsth)(?<!\be\.g)(?<!\bi\.e)\.\s+|:\s*[\"'\u201c\u2018]|\s+;\s+|\n|\[INCORRECT:|✗",
                clean_warn,
            )
            short_warn = parts[0].strip().rstrip(":;,./ ")
            # Strip trailing ', or ...' clause if it pushes the label past 100 chars
            if len(short_warn) > 95 and ", or " in short_warn:
                short_warn = short_warn.split(", or ")[0].strip()
            topic_label = f"{warn_topic}: {short_warn}" if warn_topic and warn_topic.lower() not in short_warn.lower() else short_warn
            if len(topic_label) > 120:
                topic_label = topic_label[:120].rsplit(" ", 1)[0].rstrip(",;:./ ")
            # Balance or strip any unclosed quotes and parentheses
            if topic_label.count("'") % 2 == 1:
                topic_label = topic_label.rstrip("'/ ") + "'"
            if topic_label.count('"') % 2 == 1:
                topic_label = topic_label.rstrip('"/ ') + '"'
            if topic_label.count("(") > topic_label.count(")"):
                topic_label += ")" * (topic_label.count("(") - topic_label.count(")"))
            return {
                "flaw_label": f"Syntactic Structure Trap ({topic_label})",
                "corruption_step": f"violating the `{name}` rule ({topic_label})",
                "constraint_note": f"strictly follow `{formula}`",
                "zh_cue": "",
            }

        return {
            "flaw_label": f"Syntactic Formula Breakdown (`{name}`)",
            "corruption_step": f"distorting the required syntactic formula `{formula}` while keeping all other words identical",
            "constraint_note": f"strictly follow `{formula}`",
            "zh_cue": "",
        }

    @classmethod
    def calculate_minimal_pair_similarity(cls, idiomatic: str, flawed: str) -> float:
        """
        Computes the word-level overlap similarity (0.0 to 1.0) between
        `idiomatic_translation` and `flawed_translation` to verify the Minimal-Pair
        corruption contract (≥85% shared non-target wording).
        """
        import difflib
        from collections import Counter

        t1 = re.findall(r"[a-z0-9']+", str(idiomatic or "").lower())
        t2 = re.findall(r"[a-z0-9']+", str(flawed or "").lower())
        if not t1 or not t2:
            return 0.0

        c1, c2 = Counter(t1), Counter(t2)
        shared = sum((c1 & c2).values())
        denom = max(min(len(t1), len(t2)), 1)
        multiset_overlap = shared / denom
        seq_ratio = difflib.SequenceMatcher(None, t1, t2).ratio()
        raw_sim = round(min(1.0, max(multiset_overlap, seq_ratio)), 4)
        # In short A1/A2 sentences (4-6 words), a strict single-word minimal-pair edit
        # (e.g. 5/6 shared = 0.8333) has only 1 unshared token and is a valid Minimal Pair.
        max_len = max(len(t1), len(t2))
        if min(len(t1), len(t2)) >= 4 and (max_len - shared) <= 1 and abs(len(t1) - len(t2)) <= 1:
            return max(0.86, raw_sim)
        return raw_sim

    @classmethod
    def build_translation_micro_task_blueprints(
        cls,
        vocab_content: Any,
        grammar_content: Any = None,
        target_count: int = 5,
        target_language: str = "Chinese",
        cefr_level: str = "B2",
    ) -> List[Dict[str, Any]]:
        """
        Builds Symbolic Translation Micro-Task Blueprints (`translation_quiz`):
        1. Step 1 — Register-Aligned Lexical & Syntactic Pairing (`Symbolic Pair Matching`):
           Pairs each `Target Keyword` with a register-compatible `Target Grammar Pattern`
           (preventing awkward forced marriages like `[spoken]` phrasal verbs with formal inversions).
        2. Step 2 — Single-Point Minimal-Pair Corruption Operator (`pedlex_error_traps` & `Grammar Warning`):
           Assigns a deterministic single-point flaw operator and CRC32-balanced `correct_answer_index` (0 or 1).
        3. Step 3 — Formats the canonical `### Item N Specification ###` with `🎯 Micro-Task for LLM`.
        """
        # 1. Parse vocabulary items (structured list or Markdown string)
        vocab_items: List[Dict[str, str]] = []
        for norm_it in cls._normalize_vocab_items_input(vocab_content):
            vocab_items.append({
                "word": norm_it.get("word") or "",
                "pos": norm_it.get("part_of_speech") or "word",
                "definition": norm_it.get("definition") or "",
                "quote": norm_it.get("quote") or "",
                "example_usage": norm_it.get("example_usage") or "",
            })

        if not vocab_items:
            return []

        # 2. Parse grammar items (structured list or Markdown string)
        grammar_items: List[Dict[str, str]] = []
        if isinstance(grammar_content, (list, tuple)):
            for git in grammar_content:
                if isinstance(git, dict):
                    explicit_topic = str(git.get("syntax_topic") or git.get("name") or "").strip()
                    raw_f = str(git.get("pattern_formula") or explicit_topic or "").strip()
                    explicit_form = str(git.get("formula") or git.get("pattern_formula") or "").strip()
                    ped_f = str(git.get("pedagogical_function") or "").strip()
                    mist = str(git.get("common_mistakes") or git.get("mistakes") or "").strip()
                else:
                    explicit_topic = str(getattr(git, "syntax_topic", "") or getattr(git, "name", "") or "").strip()
                    raw_f = str(getattr(git, "pattern_formula", "") or explicit_topic or "").strip()
                    explicit_form = str(getattr(git, "formula", "") or getattr(git, "pattern_formula", "") or "").strip()
                    ped_f = str(getattr(git, "pedagogical_function", "") or "").strip()
                    mist = str(getattr(git, "common_mistakes", "") or getattr(git, "mistakes", "") or "").strip()
                if not raw_f:
                    continue
                if explicit_topic and explicit_form and explicit_topic != explicit_form:
                    g_name, g_form = explicit_topic, explicit_form
                elif ":" in raw_f:
                    g_name, g_form = [p.strip() for p in raw_f.split(":", 1)]
                else:
                    g_name, g_form = raw_f, explicit_form or raw_f
                grammar_items.append({
                    "name": g_name,
                    "formula": g_form,
                    "pedagogical_function": ped_f,
                    "common_mistakes": mist,
                })
        elif isinstance(grammar_content, str) and grammar_content.strip():
            g_blocks = re.split(r"(?=^##\s*\[\[)", grammar_content.strip(), flags=re.MULTILINE)
            for gb in g_blocks:
                m_gn = re.search(r"^##\s*\[\[(.*)\]\]\s*$", gb, re.MULTILINE)
                if not m_gn:
                    continue
                g_name = m_gn.group(1).strip()
                m_gf = re.search(r"-\s*\*\*Pattern Formula\*\*:\s*(.*)", gb, re.IGNORECASE)
                g_form = m_gf.group(1).strip() if m_gf else g_name
                m_gp = re.search(r"-\s*\*\*Pedagogical Function\*\*:\s*(.*)", gb, re.IGNORECASE)
                ped_f = m_gp.group(1).strip() if m_gp else ""
                m_gm = re.search(r"-\s*\*\*Common Mistakes\*\*:\s*(.*)", gb, re.IGNORECASE)
                mist = m_gm.group(1).strip() if m_gm else ""
                grammar_items.append({
                    "name": g_name,
                    "formula": g_form,
                    "pedagogical_function": ped_f,
                    "common_mistakes": mist,
                })

        # Fallback grammar pool if none provided
        if not grammar_items:
            grammar_items = [
                {
                    "name": "Concessive Clause",
                    "formula": "Although / Even though + Subject + Verb, Main Clause",
                    "pedagogical_function": "Frames a contrast between an obstacle and an outcome.",
                    "common_mistakes": "Grammar Warning (CONCESSIVE CONNECTIVES): Never pair 'Although / Even though' with 'but' in the same sentence.",
                }
            ]

        effective_count = min(max(1, target_count), len(vocab_items))
        selected_vocab = vocab_items[:effective_count]

        # Classify registers on selected vocabulary and grammar candidates
        for vit in selected_vocab:
            vit["register"] = cls._classify_vocab_register(vit["word"], vit["pos"], vit["definition"])
        for git in grammar_items:
            git["register"] = cls._classify_grammar_register(git["name"], git["formula"], git["pedagogical_function"])

        # 3. Step 1: Register-Aligned Lexical & Syntactic Pairing (Symbolic Pair Matching)
        gram_usage_counts = [0] * len(grammar_items)
        paired_specs: List[Tuple[Dict[str, str], Dict[str, str]]] = []

        for v_it in selected_vocab:
            v_reg = v_it["register"]
            best_g_idx = 0
            best_score = -10**9
            for g_idx, g_it in enumerate(grammar_items):
                g_reg = g_it["register"]
                g_comb = f"{g_it['name']} {g_it['formula']}".lower()
                score = 0
                # Penalize forced marriages between opposite registers
                if v_reg == "informal" and g_reg == "formal":
                    score -= 100
                elif v_reg == "formal" and g_reg == "informal":
                    score -= 100
                elif v_reg == g_reg:
                    score += 30
                elif v_reg == "formal" and g_reg == "neutral":
                    # Concessive / conditional / relative clauses pair naturally with formal academic words
                    score += 25 if any(k in g_comb for k in ("concessive", "although", "despite", "while", "whereas")) else 18
                elif v_reg == "informal" and g_reg == "neutral":
                    score += 20
                else:
                    # neutral vocab works cleanly with any grammar pattern
                    score += 22 if g_reg == "neutral" else 15

                # Encourage even rotation across compatible grammar patterns
                score -= gram_usage_counts[g_idx] * 18
                if score > best_score:
                    best_score = score
                    best_g_idx = g_idx

            gram_usage_counts[best_g_idx] += 1
            paired_specs.append((v_it, grammar_items[best_g_idx]))

        # 4. Compute deterministic CRC32-balanced 2-option slots (0 vs 1, 50/50 split)
        n_items = len(paired_specs)
        slot_0_count = n_items // 2 + (n_items % 2)
        slot_1_count = n_items // 2
        balanced_slots = [0] * slot_0_count + [1] * slot_1_count
        seed_str = "|".join(f"{v['word']}:{g['name']}" for v, g in paired_specs)
        rng_state = zlib.crc32(seed_str.encode("utf-8"))
        for i in range(len(balanced_slots) - 1, 0, -1):
            rng_state = (rng_state * 1103515245 + 12345) & 0x7FFFFFFF
            j = rng_state % (i + 1)
            balanced_slots[i], balanced_slots[j] = balanced_slots[j], balanced_slots[i]

        # 5. Step 2 & Step 3: Build Single-Point Minimal-Pair Corruption Operator & Micro-Task Blueprint
        blueprints: List[Dict[str, Any]] = []
        cefr_upper = str(cefr_level or "B2").upper()

        for idx, (v_it, g_it) in enumerate(paired_specs, 1):
            word = v_it["word"]
            pos = v_it["pos"]
            defn = v_it["definition"]
            v_reg = v_it["register"]

            g_name = g_it["name"]
            g_formula = g_it["formula"]
            g_reg = g_it["register"]
            target_grammar_display = (
                f"{g_name} (`{g_formula}`)"
                if g_formula and g_formula.lower() != g_name.lower()
                else g_name
            )

            lex_info = cls._resolve_translation_lexical_trap(
                word, pos=pos, definition=defn, quote=v_it.get("quote", ""), example_usage=v_it.get("example_usage", "")
            )
            syn_info = cls._resolve_translation_syntactic_trap(g_it)

            req_frame = lex_info["required_frame"] or word
            model_ex = lex_info["model_example"]
            frame_line = f"{req_frame} (Model: \"{model_ex}\")" if model_ex else req_frame

            # Select primary Prescribed Flaw Type:
            # - If lexical trap is an explicit property/dontsay trap (GRAM_ATTRIBUTIVE_ONLY, GRAM_UNCOUNTABLE, GRAM_BOX_DONTSAY),
            #   or on odd/even items when a lexical trap exists and we balance lexical vs syntactic traps across the quiz.
            lex_trap = lex_info.get("trap")
            lex_trap_type = str(lex_trap.get("trap_type") or "") if isinstance(lex_trap, dict) else ""
            use_lexical_primary = False
            if lex_trap and lex_info.get("flaw_label"):
                if lex_trap_type in ("GRAM_ATTRIBUTIVE_ONLY", "GRAM_UNCOUNTABLE", "GRAM_BOX_DONTSAY"):
                    use_lexical_primary = True
                elif idx % 2 == 1 and not ("although" in g_formula.lower() and word.lower() == "originate"):
                    use_lexical_primary = True

            if use_lexical_primary:
                prescribed_flaw_type = lex_info["flaw_label"]
                flaw_category = "lexical"
                primary_corruption = lex_info["corruption_step"]
                secondary_alt = f", or {syn_info['corruption_step']}" if syn_info.get("corruption_step") else ""
                critique_focus = (
                    f"pointing out the exact `{prescribed_flaw_type}` error and confirming the `{req_frame}` collocation and `{g_name}` structure"
                )
            else:
                prescribed_flaw_type = syn_info["flaw_label"]
                flaw_category = "syntactic"
                primary_corruption = syn_info["corruption_step"]
                secondary_alt = (
                    f", or misusing `{word}` ({lex_info['corruption_step']})"
                    if lex_info.get("corruption_step")
                    else f", or misusing `{word}`"
                )
                critique_focus = (
                    f"pointing out the exact `{prescribed_flaw_type}` error and confirming the `{req_frame}` collocation"
                )

            zh_cue_str = f" (matching `{g_name}` + `{word}` meaning: \"{defn}\")"
            gram_constraint = f" ({syn_info['constraint_note']})" if syn_info.get("constraint_note") else ""

            micro_task_lines = [
                f"### Item {idx} Specification ###",
                f"- Target Keyword: {word} ({pos}) — \"{defn}\"",
                f"- Required Lexical Frame: {frame_line}",
                f"- Target Grammar Pattern: {target_grammar_display}",
                f"- Prescribed Flaw Type: {prescribed_flaw_type}",
                f"- 🎯 Micro-Task for LLM:",
                f"  1. `translated_sentence`: Write a natural {target_language} sentence{zh_cue_str}.",
                f"  2. `idiomatic_translation`: Translate into natural CEFR {cefr_upper} English integrating BOTH `{req_frame}` and `{g_formula}`{gram_constraint}.",
                f"  3. `flawed_translation`: Keep the sentence nearly identical as a Minimal Pair (≥85% shared wording), ONLY introducing the prescribed flaw ({primary_corruption}{secondary_alt}).",
                f"  4. `diagnostic_critique`: Contrast the two versions by {critique_focus}.",
            ]
            micro_task_block = "\n".join(micro_task_lines)

            blueprints.append({
                "item_index": idx,
                "target_keyword": word,
                "part_of_speech": pos,
                "definition": defn,
                "vocab_register": v_reg,
                "required_lexical_frame": req_frame,
                "model_example": model_ex,
                "target_grammar": f"{g_name}: {g_formula}" if g_formula and g_formula.lower() != g_name.lower() else g_name,
                "target_grammar_name": g_name,
                "target_grammar_formula": g_formula,
                "grammar_register": g_reg,
                "flaw_type": prescribed_flaw_type,
                "flaw_category": flaw_category,
                "lexical_trap": lex_trap,
                "syntactic_trap": syn_info,
                "correct_answer_index": balanced_slots[idx - 1],
                "micro_task_block": micro_task_block,
            })

        return blueprints
