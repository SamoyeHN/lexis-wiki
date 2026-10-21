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
            m_disc = re.match(
                r'^([a-z]+)\s+(?:\[[^\]]+\]|\([^)]+\))\s+(of|to|with|from|into|for|on|in|about|against|at|upon|over)\b',
                raw_w_item.strip(),
                re.IGNORECASE
            )
            if m_disc:
                head_v = m_disc.group(1).lower()
                disc_p = m_disc.group(2).lower()
                # Only decompose if head_v is not already a standalone single-word target in the batch
                if head_v not in single_word_targets:
                    w = head_v
                    w_lower = head_v
                    discontinuous_prep = disc_p
                    raw_pos = "verb"

            # Discontinuous Causative Verb + [sth] + Adjective Extraposition Frame Resolution (e.g. "make [sth] possible"):
            # Stripping '[sth]' into bare 2-word 'make possible' produces an ungrammatical contiguous string ('*make possible it').
            # Transform into the single-blank preparatory-object extraposition construction 'make it <adj>' with frame
            # 'make it <adj> ... to do [sth]'.
            m_caus_adj = re.match(
                r'^(make|find|keep|leave)\s+(?:\[[^\]]+\]|\([^)]+\)|\.\.\.)\s+([a-z]+)$',
                raw_w_item.strip(),
                re.IGNORECASE
            )
            if not m_caus_adj and re.match(r'^(make|find)\s+(possible|impossible|clear|easy|difficult|hard|certain|plain)$', w_lower):
                m_caus_adj = re.match(r'^(make|find)\s+([a-z]+)$', w_lower)
            if m_caus_adj:
                caus_v = m_caus_adj.group(1).lower()
                adj_w = m_caus_adj.group(2).lower()
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
                                w = f"{caus_v} it {adj_w}"
                                w_lower = w
                                raw_pos = "collocation"
                                raw_s_def = cls.format_ldoce_definition(s_cand)
                                comp_def = cls._compose_collocation_definition(
                                    f"{caus_v} {adj_w}", "collocation", adj_w, s_cand, raw_s_def, it.get("quote")
                                )
                                if comp_def:
                                    it["definition"] = comp_def
                                causative_extraposition_pat = f"{caus_v} it {adj_w} ... to do [sth]"
                                causative_extraposition_ex = str(pb.get("example") or "").strip() or None
                                break
                        if causative_extraposition_pat:
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

            def _tok_matches_target(tok, doc_obj) -> bool:
                if "-" not in w_lower:
                    if (tok.i > 0 and doc_obj[tok.i - 1].text == "-" and not doc_obj[tok.i - 1].whitespace_) or (
                        tok.i + 1 < len(doc_obj) and doc_obj[tok.i + 1].text == "-" and not tok.whitespace_
                    ):
                        return False
                return tok.text.lower() == w_lower or tok.lemma_.lower() == w_lower

            if canonical_pos == "verb":
                # Verb Step A: Extract the valency complement PP from the authentic quote.
                if it["quote"]:
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
                if it["quote"]:
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
                # Adjective Step A: Syntactic Dependency Analysis on Authentic Quote
                if it["quote"]:
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
                                for p in ("of", "to", "for", "with", "about", "from", "in", "on", "at"):
                                    if re.search(rf"\b{p}\b", pat.lower()):
                                        if p == "to":
                                            m_inf = re.search(r"\bto\s+([a-z]+)", pat.lower())
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

            elif canonical_pos == "adv":
                # Adverb Syntactic Dependency Walker & Sense-Lock Gate
                if it.get("quote"):
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
                if not anchor:
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

            # 1.5 Sense-Native Atomic Pattern Anchor Inheritance (GEMINI.md §1.10):
            # If Step 1 found no anchor (or only a weak fallback) and the locked sense's
            # atomic pattern bundle specifies a bound preposition, inherit it BEFORE Step 3
            # distractor generation so double-key clearance runs against the true anchor.
            if not is_idiom_slot and sense_native_prep and (not anchor or anchor_type == "collocation"):
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
                    p_dists = cls.generate_phrase_distractors(w_lower, count=3, exclude_words=used_distractors)
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
                if atomic_bundle.get("prep"):
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

            quote_shown_prep = bool(anchor) and anchor_type == "prep" and bool(
                quote_governed_prep and str(anchor).lower() == quote_governed_prep.lower()
            )
            quote_shown_partitive = bool(anchor) and anchor_type == "partitive" and bool(re.search(
                rf"\b{re.escape(str(anchor).lower())}\b", (it.get("quote") or "").lower()))

            if anchor and anchor_type != "contextual" and not ldoce_pattern_example:
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
            if (not anchor or anchor_type == "contextual"
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
            if batch_target_words and not is_idiom_slot and (anchor or prep_obj):
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
            if anchor and not ldoce_pattern_example and anchor != pattern_anchor:
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
            if best_pattern:
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
            if best_pattern:
                # Preserve trailing object placeholders ([sb/sth], [sb], [sth]) while stripping editorial notes
                p_clean = re.sub(r"\(.*?\)", "", best_pattern).strip()
                p_clean = re.sub(
                    r"\[(?!(?:sb/sth|sth/sb|sb|sth|somebody|something)\]).*?\]",
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

            is_base_verb_slot = (
                (inflection_desc or "base form") == "base form"
                and not is_plural_noun
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
