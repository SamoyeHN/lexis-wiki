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
            is_multi = it.get("is_multi_word") or (len(it.get("target_word", "").split()) > 1) or (it.get("part_of_speech") in ("phrase", "collocation", "phrasal verb", "set phrase", "idiom"))
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
        sel_single = singles[:quota_single]

        # Backfill if one group has fewer items than its allocated quota
        if len(sel_multi) < quota_multi:
            deficit = quota_multi - len(sel_multi)
            sel_single.extend(singles[len(sel_single):len(sel_single) + deficit])
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
                    exclude_words=used_distractors
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
    def extract_sense_pattern_bundles(cls, sense: Dict[str, Any], target_word: str) -> List[Dict[str, Any]]:
        """
        Extracts atomic (pattern, example, bound_prep) bundles from a locked PedLex sense.
        Prioritizes native atomic pattern_bundles directly extracted from PedLex DOM.
        Falls back to parsing raw patterns while strictly preventing pattern-example desynchronization.
        """
        if not sense:
            return []

        # 1. Native Atomic Bundles (Directly from updated PedLex DB schema)
        native_bundles = sense.get("pattern_bundles")
        if native_bundles:
            out: List[Dict[str, Any]] = []
            for nb in native_bundles:
                pat = (nb.get("pattern") or "").strip()
                if not pat:
                    continue
                p_clean_search = re.sub(r"\[.*?\]|\(.*?\)", "", pat).strip().lower()
                p_for_prep = re.sub(r"\bto\s+(?:do\s+something|\w+ing)\b", "", p_clean_search)
                m_prep = re.search(r"\b(on|in|at|for|with|about|under|from|to|by|into|of)\b", p_for_prep)
                bound_prep = m_prep.group(1) if m_prep else None
                out.append({
                    "pattern": pat,
                    "example": nb.get("example") or None,
                    "prep": bound_prep,
                    "source": nb.get("source", "native_bundle")
                })
            if out:
                return out

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
            all_preps = re.findall(r"\b(on|in|at|for|with|about|under|from|to|by|into|of|between|over|against|towards|upon)\b", p_for_prep)
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
            })
        return bundles

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
        bundles = cls.extract_sense_pattern_bundles(sense, target_word)
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
            score = 0

            # Absolute filter / priority: Target word presence
            # Genuine valency patterns contain the target headword or its inflections.
            # Idioms like 'got on his nerves' under 'annoy' do NOT contain 'annoy' and must not hijack the item!
            has_tw = bool(re.search(rf"\b{re.escape(tw_l)}\w*\b", pat.lower()))
            if has_tw:
                score += 50
            else:
                score -= 30

            # Massive bonus if bundle contains an authentic illustrative example
            if ex:
                score += 25
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
            # Syntactic valency frames (gerunds, infinitives, complement clauses)
            if any(v in pat.lower() for v in ("doing something", "to do something", "somebody when", "somebody that", "somebody how")):
                score += 15
            # Clean length preference (compact, high-utility frames)
            p_len = len(re.sub(r"\[.*?\]|\(.*?\)", "", pat).strip())
            if 5 <= p_len <= 35:
                score += 5
            scored.append((score, b))

        scored.sort(key=lambda x: (x[0], -len(x[1]["pattern"])), reverse=True)
        if not scored:
            return None
        best_b = scored[0][1]
        # Do not return a preposition-claiming pattern that lacks any example AND is not attested in the quote
        if best_b.get("prep") and not best_b.get("example"):
            if not any(re.search(rf"\b{re.escape(cp)}\b", q_l) for cp in (best_b.get("preps") or [best_b["prep"]])):
                return None
        return best_b

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

            m_disc = re.match(
                r'^([a-z]+)\s+(?:\[[^\]]+\]|\([^)]+\)|\.\.\.)\s+(of|to|with|from|into|for|on|in|about|against|at|upon|over)\b',
                raw_w_item.strip(),
                re.IGNORECASE
            )
            if not m_disc and len(w_lower.split()) == 2:
                w_p0, w_p1 = w_lower.split()
                if w_p1 in ("of", "from", "into", "for", "on", "to", "with") and w_p0 in {
                    "remind", "convince", "inform", "warn", "assure", "rob", "deprive", "rid", "cure",
                    "prevent", "stop", "prohibit", "discourage", "deter", "accuse", "suspect",
                    "congratulate", "blame", "thank", "forgive", "talk", "trick", "deceive", "fool",
                }:
                    m_disc = re.match(r'^([a-z]+)\s+([a-z]+)$', w_lower)
            if m_disc:
                head_v = m_disc.group(1).lower()
                disc_p = m_disc.group(2).lower()
                # Only decompose if head_v is not already a standalone single-word target in the batch
                if head_v not in single_word_targets:
                    w = head_v
                    w_lower = head_v
                    discontinuous_prep = disc_p
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
            target_entry = entry_base
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
                t_s_idx = cls._lock_sense(
                    target_entry,
                    definition=it.get("definition"),
                    quote=it.get("quote") or it.get("raw_quote"),
                    target_pos=canonical_pos
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
            sense_native_prep = discontinuous_prep or (initial_atomic_bundle.get("prep") if initial_atomic_bundle else None)

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

            word_cxn_specs = cls.get_word_cxn_specs(w_lower)
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
                                    if prep_tok not in ("to", "with", "from", "on", "for", "in", "into", "of", "against", "at", "upon", "towards"):
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
                                    if child.dep_ == "dobj" and child.lemma_.lower() not in cls._ANCHOR_STOPWORDS:
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
                                    if child.dep_ in ("nsubj", "nsubjpass") and child.lemma_.lower() not in cls._ANCHOR_STOPWORDS:
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
                    if "N_SHELL_TO_DO" in word_subcats and re.search(rf"\b{re.escape(w_lower)}(?:s|es|ies)?\s+to\s+[a-z]+\b", q_low_n):
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
                                    for child in tok.children:
                                        if child.dep_ in ("amod", "compound"):
                                            mod_tok = child.lemma_.lower()
                                            if child.pos_ != "PRON" and mod_tok not in cls._ANCHOR_STOPWORDS and mod_tok != w_lower:
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
                                                and head_tok_lemma not in cls._DELEXICAL_VERBS):
                                            verb_before_set = {v.lower().split()[0] for v in n_entry.get("verb_before", [])}
                                            if not verb_before_set or head_tok_lemma in verb_before_set:
                                                anchor = head_tok_lemma
                                                anchor_type = "verb"
                                break
                    except Exception:
                        pass

                # Priority D: dictionary candidates, strictly gated by quote/definition evidence
                if not anchor and n_entry:
                    if n_entry.get("adj") or n_entry.get("verb_before"):
                        candidates = n_entry.get("adj", []) + n_entry.get("verb_before", [])
                        anchor = cls._select_sense_aligned_anchor(candidates, definition=it.get("definition"), quote=it.get("quote"), target_word=w_lower, forbidden=batch_target_words)
                        if anchor:
                            anchor_type = "adj" if cls._token_in_phrases(anchor, n_entry.get("adj", [])) else "verb"
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
                # Adjective Step A: Syntactic Dependency Analysis on Authentic Quote
                if not anchor and it["quote"]:
                    try:
                        doc = nlp(it["quote"])
                        for tok in doc:
                            if _tok_matches_target(tok, doc):
                                # A1. Predicative Valency: Check if adjective governs a bound preposition
                                for child in tok.children:
                                    if child.dep_ == "prep" and child.text.lower() in ("to", "for", "with", "of", "in", "from", "at", "about", "on"):
                                        anchor = child.text.lower()
                                        anchor_type = "prep"
                                        quote_governed_prep = anchor
                                        for pchild in child.children:
                                            if pchild.dep_ == "pobj" and pchild.pos_ != "PRON" and pchild.text.lower() not in cls._ANCHOR_STOPWORDS and pchild.lemma_.lower() != w_lower:
                                                prep_obj = pchild.lemma_.lower()
                                                break
                                        break
                                # A2. Attributive Modification: Check if adjective modifies a head noun (dep_ == "amod")
                                if not anchor and not is_not_before_noun_adj and tok.dep_ == "amod" and tok.head and tok.head.pos_ in ("NOUN", "PROPN"):
                                    if tok.head.pos_ != "PRON" and tok.head.text.lower() not in cls._ANCHOR_STOPWORDS and tok.head.lemma_.lower() != w_lower:
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
                adj_entry = cls.get_oxford_collocations(w_lower, pos="adj")
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
                "correl_pair", "fronted_inv", "npi", "mandative_subj", "lv_triad"
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
                    )
                    distractors = p_dists
                    dist_meta = {d: "phrase_contrast" for d in p_dists}
            elif canonical_pos == "idiom_slot" and w_lower == "granted":
                i_cands = ["accepted", "given", "known", "certain", "true"]
                distractors = [c for c in i_cands if c not in used_distractors][:3]
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
                if quote_plural and initial_atomic_bundle:
                    init_pat = (initial_atomic_bundle.get("pattern") or "").lower()
                    init_ex = (initial_atomic_bundle.get("example") or "").lower()
                    p_cand = cls.pluralize_noun(w_lower)
                    if init_pat and re.search(rf"\b{re.escape(w_lower)}\b", init_pat) and not re.search(rf"\b{re.escape(p_cand)}\b", init_pat):
                        if not init_ex or not re.search(rf"\b{re.escape(p_cand)}\b", init_ex):
                            quote_plural = False

                if is_spacy_plural or is_pedlex_plural or w_lower in plural_tantum or quote_plural:
                    is_plural_noun = True
                    plural_target = cls.pluralize_noun(w_lower)

            def _is_double_key(cand: str) -> bool:
                if canonical_pos in ("function_word", "phrase", "idiom_slot"):
                    return False
                flag, _kind = cls.double_key_collision(
                    w_lower, cand, anchor, anchor_type, pos=canonical_pos, quote=it.get("quote")
                )
                if flag:
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
                if len(w_lower) >= 4 and len(c_low) >= 4:
                    if (w_lower in c_low or c_low in w_lower) and (
                        abs(len(w_lower) - len(c_low)) >= 2 or cls.are_same_word_family(w_lower, c_low)
                    ):
                        return False
                if canonical_pos in ("function_word", "phrase", "idiom_slot"):
                    return True
                if is_plural_noun and not cls.can_pluralize_noun_distractor(c_low):
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
                    if (
                        curr_meta.get(d) != "ldoce_thesaurus"
                        and d.lower() in near_only_s
                        and sum(1 for x in cleared_list if x.lower() in near_only_s and curr_meta.get(x) != "ldoce_thesaurus") >= 1
                    ):
                        continue
                    cleared_list.append(d)
                r_meta: Dict[str, str] = {}
                if len(cleared_list) < 3:
                    try:
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
                            if (
                                r_meta.get(cand) != "ldoce_thesaurus"
                                and cand.lower() in near_only_s
                                and sum(1 for x in cleared_list if x.lower() in near_only_s and (curr_meta.get(x) or r_meta.get(x)) != "ldoce_thesaurus") >= 1
                            ):
                                continue
                            cleared_list.append(cand)
                    if len(cleared_list) < 3:
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

            # 4. Plan A: All options and target word maintain parallel inflection
            if is_plural_noun:
                inflection_desc = "plural form (NNS)"
                final_target = plural_target
                inflected_distractors3 = [cls.pluralize_noun(d) for d in distractors3]
                inflected_distractors3 = cls.screen_inflected_options(
                    inflected_distractors3,
                    [cls.pluralize_noun(d) for d in distractors[3:]],
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
                if atomic_bundle.get("prep") and anchor_type not in _cxn_anchor_types:
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
            if anchor_type not in _cxn_anchor_types and (
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
                            [cls.pluralize_noun(d) for d in distractors3],
                            [cls.pluralize_noun(d) for d in distractors[3:]],
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
            inflection_desc, final_options, final_target, verb_form_tag = cls._reconcile_prescribed_inflection(
                base_options=_base_options,
                target_index=correct_answer_index,
                target_word=final_target,
                canonical_pos=canonical_pos,
                inflection_desc=_orig_inflection_desc,
                authentic_example=ldoce_pattern_example
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
                        authentic_example=ldoce_pattern_example
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
                    # If the passage quote or dictionary example attest both human and non-human objects, generalize to [sb/sth]
                    if obj_tag in ("[sb]", "[sth]"):
                        best_pattern = f"{final_target} [sb/sth]"
                    else:
                        best_pattern = f"{final_target} {obj_tag}"
                elif any(
                    w_lower.endswith(" " + p)
                    for p in ("with", "to", "for", "of", "into", "from", "on", "at", "about", "against", "upon", "over", "after")
                ):
                    best_pattern = f"{final_target} [sb/sth]"

            if best_pattern and is_plural_noun:
                best_pattern = re.sub(r"\b" + re.escape(w_lower) + r"\b", final_target, best_pattern, flags=re.IGNORECASE)

            # Resolve slash-separated lexical alternatives directly on best_pattern
            # (e.g. 'regulatory body/authority/agency' -> 'regulatory authority', 'wind up in/at/with etc' -> 'wind up in')
            # so Pattern and Fit into frame are 100% single-form and never expose raw slashes.
            if best_pattern:
                best_pattern = re.sub(r"\s+\betc\.?(?=\s|$)", "", best_pattern, flags=re.IGNORECASE).strip()
                if "/" in best_pattern:
                    ex_low_for_slash = (ldoce_pattern_example or it.get("quote") or "").lower()
                    anc_low_for_slash = str(anchor or "").strip().lower()
                    def _pick_slash_alt(m: re.Match) -> str:
                        # Preserve bracketed object placeholders like [sb/sth]
                        start_idx = m.start()
                        if start_idx > 0 and best_pattern[start_idx - 1] == "[":
                            return m.group(0)
                        alts = [a for a in m.group(0).split("/") if a]
                        if set(a.lower() for a in alts) <= {"sb", "sth", "somebody", "something"}:
                            return m.group(0)
                        if anc_low_for_slash and anc_low_for_slash in [a.lower() for a in alts]:
                            return next(a for a in alts if a.lower() == anc_low_for_slash)
                        for a in alts:
                            if ex_low_for_slash and re.search(rf"\b{re.escape(a.lower())}\w*\b", ex_low_for_slash):
                                return a
                        return alts[0]
                    best_pattern = re.sub(r"\b[a-zA-Z]+(?:/[a-zA-Z]+)+\b", _pick_slash_alt, best_pattern).strip()

            # Streamlined Context Stem Directive
            cloze_slot_frame = None
            if cxn_override_frame:
                cloze_slot_frame = cxn_override_frame
            elif best_pattern:
                # Preserve trailing object placeholders ([sb/sth], [sb], [sth], [doing sth], [V-base]) while stripping editorial notes
                p_clean = re.sub(r"\(.*?\)", "", best_pattern).strip()
                p_clean = re.sub(r"\bdoing\s+something\b", "[doing sth]", p_clean, flags=re.IGNORECASE)
                p_clean = re.sub(r"\bto\s+do\s+something\b", "to do [sth]", p_clean, flags=re.IGNORECASE)
                p_clean = re.sub(r"\bsomebody\b", "[sb]", p_clean, flags=re.IGNORECASE)
                p_clean = re.sub(r"\bsomething\b", "[sth]", p_clean, flags=re.IGNORECASE)
                p_clean = re.sub(
                    r"\[(?!(?:sb/sth|sth/sb|sb|sth|somebody|something|doing sth|V-base|adj|close|great|special)\]).*?\]",
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
                if not is_copula_adj and not is_attributive_only_adj and (cloze_slot_frame or anchor):
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
            is_base_verb_slot = (
                (inflection_desc or "base form") == "base form"
                and not is_plural_noun
                and not is_be_led_phrase
                and (
                    canonical_pos == "verb"
                    or (
                        canonical_pos == "phrase"
                        and str((target_entry or {}).get("pos", "")).strip().lower() != "noun"
                        and final_target.split()
                        and final_target.split()[0] not in {"a", "an", "the", "in", "on", "at", "by", "for", "with", "from", "to", "of"}
                        and bool(cls._wordnet_pos(final_target.split()[0]) & {"v"})
                    )
                )
            )
            be_led_slot_hint = (
                " (Bare-infinitive 'be' phrase slot required: precede '____' with infinitive 'to ____' "
                "or a modal verb 'can/will/would/should/must/may/might ____' — NEVER place '____' after "
                "finite 'am/is/are/was/were/been' or directly after a bare subject without a modal)."
            )

            if cloze_slot_frame:
                stem_core = f"Fit into frame '{cloze_slot_frame}'."
                m_adj_noun = re.search(r"_{2,}\s+([a-zA-Z]+)\b", cloze_slot_frame) if canonical_pos == "adj" else None
                if is_plural_noun:
                    stem_core += f" (Plural noun slot required: e.g. '{final_target}')."
                elif is_copula_adj:
                    stem_core += " (Complete predicate required: stem MUST include the linking verb so the adjective predicate is complete; NEVER place before a noun)."
                elif is_attributive_only_adj:
                    if m_adj_noun:
                        stem_core += f" (Attributive pre-noun slot required: you MUST include the modified noun '{m_adj_noun.group(1)}' immediately after '____' as in '____ {m_adj_noun.group(1)}'; NEVER leave '____' without its head noun)."
                    else:
                        stem_core += " (Attributive pre-noun slot required: place '____' directly before a modified noun; NEVER use after a linking verb)."
                elif m_adj_noun and m_adj_noun.group(1).lower() not in {"to", "in", "on", "at", "for", "with", "of", "from", "by", "about", "that"}:
                    stem_core += f" (You MUST include the modified noun '{m_adj_noun.group(1)}' immediately after '____' as in '____ {m_adj_noun.group(1)}')."
                elif is_be_led_phrase:
                    stem_core += be_led_slot_hint
                elif is_base_verb_slot:
                    stem_core += " (Base-form verb slot required: use 'to ____', modal 'can/will/should ____', or plural/1st-person subject 'They/We/I ____' — NEVER singular 'He/She/It ____' or perfect 'has/have/had ____' without 'to')."
            elif best_pattern:
                stem_core = "Fit into the Pattern above."
                if is_plural_noun:
                    stem_core += f" (Plural noun slot required: e.g. '{final_target}')."
                elif is_copula_adj:
                    stem_core += " (Complete predicate required: stem MUST include the linking verb; NEVER place before a noun)."
                elif is_attributive_only_adj:
                    stem_core += " (Attributive pre-noun slot required: place '____' directly before a modified noun; NEVER use after a linking verb)."
                elif is_be_led_phrase:
                    stem_core += be_led_slot_hint
                elif is_base_verb_slot:
                    stem_core += " (Base-form verb slot required: use 'to ____', modal 'can/will/should ____', or plural/1st-person subject 'They/We/I ____' — NEVER singular 'He/She/It ____' or perfect 'has/have/had ____' without 'to')."
            else:
                stem_core = "Write a natural sentence matching the Definition above."
                if is_plural_noun:
                    stem_core += f" (Plural noun slot required: e.g. '{final_target}')."
                elif is_copula_adj:
                    stem_core += " (Complete predicate required: stem MUST include a linking/copular verb; NEVER place before a noun)."
                elif is_attributive_only_adj:
                    stem_core += " (Attributive pre-noun slot required: place '____' directly before a modified noun; NEVER use after a linking verb)."
                elif is_be_led_phrase:
                    stem_core += be_led_slot_hint
                elif is_base_verb_slot:
                    stem_core += " (Base-form verb slot required: use 'to ____', modal 'can/will/should ____', or plural/1st-person subject 'They/We/I ____' — NEVER singular 'He/She/It ____' or perfect 'has/have/had ____' without 'to')."

            task_parts = [stem_core]

            if ldoce_pattern_example:
                ex_clean = ldoce_pattern_example.strip()
                skip_model = False
                if is_plural_noun and canonical_pos == "noun":
                    if re.search(rf"\b{re.escape(w_lower)}\b", ex_clean, re.IGNORECASE) and not re.search(rf"\b{re.escape(final_target)}\b", ex_clean, re.IGNORECASE):
                        skip_model = True
                if not skip_model:
                    task_parts.append(f'Model: "{ex_clean}"')

            # Data-Driven Semantic & Grammatical Discriminator (PedLex <GRAM>, Grammar Boxes, Thesaurus & Language Activator)
            target_base = quant_head_noun if is_quantifier_frame else w_lower
            dist_base_list = [o for o in _base_options if o.lower() != target_base]
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
