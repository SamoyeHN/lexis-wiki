"""
Deterministic academic vocabulary mining, Top-K candidate pool, distinct authentic example selection, and extraction.
"""

import datetime
import hashlib
import json
import logging
import os
import re
import sqlite3
import threading
import zlib
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

logger = logging.getLogger("librarian.linguistics")


class ExtractionVocabMixin:
    # -------------------------------------------------------------------------
    # 3c. Deterministic Academic Vocabulary Mining (AWL + Lemmatization)
    # -------------------------------------------------------------------------
    @classmethod
    def mine_vocabulary_skeletons(cls, text: str, target_count: int = 20) -> List[Dict[str, Any]]:
        """
        Deterministically mines genuine academic vocabulary targets from the indexed sentence pool.
        Uses spaCy context-aware lemmatization to extract base dictionary headwords,
        prioritizes the Academic Word List (AWL 560), and filters out low-value everyday words.

        Returns list of dicts:
            [{
                "sid": "S-1",
                "quote": "full sentence text",
                "word": "autonomy",
                "part_of_speech": "noun",
                "is_awl": True
            }, ...]
        """
        _, pool = cls.tokenize_and_index_sentences(text)
        if not pool:
            return []

        nlp = cls.get_spacy()

        # Closed whitelist of parts of speech accepted by schema
        VALID_POS_MAP = {
            "NOUN": "noun",
            "VERB": "verb",
            "ADJ": "adjective",
            "ADV": "adverb",
        }

        # Exclude trivial grammaticalized or ultra-common verbs/words
        TRIVIAL_WORDS = frozenset({
            "be", "have", "do", "say", "get", "make", "go", "know", "take", "see",
            "come", "think", "look", "want", "give", "use", "find", "tell", "ask",
            "work", "seem", "feel", "try", "leave", "call", "good", "new", "first",
            "last", "long", "great", "little", "own", "other", "old", "right", "big",
            "high", "different", "small", "large", "next", "early", "young", "important",
            "few", "public", "bad", "same", "able", "man", "woman", "person", "people",
            "child", "time", "year", "day", "way", "thing", "life", "hand", "part",
            "eye", "place", "case", "week", "company", "system", "program", "question",
            "government", "number", "night", "point", "home", "water", "room", "mother",
            "area", "money", "story", "fact", "month", "lot", "study", "book",
            "word", "business", "issue", "side", "kind", "head", "house", "service",
            "friend", "father", "power", "hour", "game", "line", "end", "member", "law",
            "car", "city", "community", "name", "president", "team", "minute", "idea",
            "kid", "body", "information", "back", "parent", "face", "others", "level",
            "office", "door", "health", "art", "war", "history", "party",
            "result", "change", "morning", "reason", "research", "girl", "guy", "moment",
            "air", "teacher", "force", "education",
            "yesterday", "today", "tomorrow", "tonight", "later", "daily"
        })

        # Dependent adverbial connectives: these heads are grammatically incomplete
        # without their obligatory prepositional complement. They are only valid as
        # the fixed 'head + of' phrase (POS = preposition), never as a bare adverb —
        # a bare 'regardless' is not a testable single-blank target and, being absent
        # from the Oxford/ACL collocation model, would yield a broken distractor space.
        # This is a small, closed, stable set of 'of'-licensing connective heads
        # (deliberately NOT an open-ended valency map): any connective whose bare
        # form is incomplete on its own belongs here.
        DEPENDENT_CONNECTIVES = {
            "regardless": "of",
            "irrespective": "of",
        }

        candidates: List[Dict[str, Any]] = []
        seen_lemmas: Set[str] = set()

        cand_map: Dict[str, Dict[str, Any]] = {}

        def _add_or_update_candidate(
            word_key: str,
            sid_val: str,
            quote_val: str,
            pos_val: str,
            is_awl_val: bool,
            score_val: float,
            is_connective_val: bool = False,
        ) -> None:
            if word_key in cand_map:
                existing_occs = cand_map[word_key]["occurrences"]
                if all(o[0] != sid_val for o in existing_occs):
                    existing_occs.append((sid_val, quote_val, pos_val))
                if score_val > cand_map[word_key]["score"]:
                    cand_map[word_key]["score"] = score_val
                return
            entry_dict: Dict[str, Any] = {
                "sid": sid_val,
                "quote": quote_val,
                "word": word_key,
                "part_of_speech": pos_val,
                "is_awl": is_awl_val,
                "score": score_val,
                "occurrences": [(sid_val, quote_val, pos_val)],
            }
            if is_connective_val:
                entry_dict["is_connective"] = True
            cand_map[word_key] = entry_dict

        for sid, sentence in pool.items():
            sent_clean = sentence.strip()
            if len(sent_clean.split()) < 5:
                continue

            doc = nlp(cls._strip_lexical_quotes(sent_clean))
            # Mask out token spans belonging to PedLex-attested non-compositional idioms and set phrases
            # (e.g. 'pros and cons', 'peace of mind', 'keep in touch with') so bound phrase constituents
            # like 'cons' in "There are pros and cons in this change." are never decomposed into single words.
            from ..expression_patterns import ExpressionPatternEngine
            expr_engine = ExpressionPatternEngine.get_engine(nlp)
            bound_phrase_token_indices: Set[int] = set()
            for em in expr_engine.match_sentence(doc):
                if em.get("type") in ("idiom", "set phrase"):
                    bound_phrase_token_indices.update(em.get("token_indices") or [])

            # First pass: mine any attested hyphenated compounds in the sentence as intact units
            for m_hc in re.finditer(r"\b([a-zA-Z]+(?:-[a-zA-Z]+)+)\b", sent_clean):
                hc = m_hc.group(1).lower()
                hc_entry = cls.get_ldoce_entry(hc)
                if hc_entry and hc_entry.get("kind") != "derived" and hc_entry.get("senses"):
                    hc_pos_raw = (hc_entry.get("pos") or "adjective").lower()
                    hc_pos = "noun" if "noun" in hc_pos_raw else ("verb" if "verb" in hc_pos_raw else ("adverb" if "adverb" in hc_pos_raw else "adjective"))
                    _add_or_update_candidate(
                        hc, sid, sent_clean, hc_pos, True, round(30.0 + min(len(hc), 10) * 0.3, 2)
                    )

            WH_AND_PRONOUN_STOPS = frozenset({
                "what", "whatever", "who", "whoever", "whom", "whose", "which", "whichever",
                "where", "wherever", "when", "whenever", "why", "how", "however"
            })
            for token in doc:
                if token.i in bound_phrase_token_indices:
                    continue
                # Never extract a severed sub-token that is attached to a hyphen (e.g. 'waste' or 'prone' in 'waste-prone')
                if (token.i > 0 and doc[token.i - 1].text == "-" and not doc[token.i - 1].whitespace_) or (
                    token.i + 1 < len(doc) and doc[token.i + 1].text == "-" and not token.whitespace_
                ):
                    continue
                surf_low = token.text.lower().strip()
                lemma = token.lemma_.lower().strip()
                if (lemma == "accord" and surf_low == "according") or (lemma == "depend" and surf_low == "depending"):
                    continue
                if lemma == "prime" and token.i + 1 < len(doc) and doc[token.i + 1].text.lower() in ("minister", "ministers"):
                    continue
                if token.dep_ == "amod" and surf_low.endswith(("ing", "ed")) and surf_low != lemma:
                    surf_entry = cls.get_ldoce_entry(surf_low)
                    if surf_entry and surf_entry.get("kind") != "derived" and any("adj" in str(s.get("pos") or surf_entry.get("pos") or "").lower() for s in (surf_entry.get("senses") or [])):
                        lemma = surf_low
                if len(lemma) < 3 or not lemma.isalpha() or lemma in WH_AND_PRONOUN_STOPS:
                    continue

                # 1. Academic Closed-Paradigm Function Word Gate:
                # If lemma belongs to our curated high-utility closed paradigms (e.g. 'whereas',
                # 'nonetheless', 'notwithstanding', 'albeit'), bypass spaCy's stopword filter and POS restriction!
                if cls.is_function_word(lemma):
                    # Filter out elementary/common prepositions and conjunctions so passage mining focuses on academic lexis
                    ULTRA_BASIC_FUNC_WORDS = frozenset({
                        "in", "at", "on", "to", "for", "with", "from", "by", "of",
                        "and", "but", "or", "nor", "so", "for", "yet",
                        "if", "when", "as", "than", "that", "because",
                        "without", "within", "upon", "onto", "into", "until", "till",
                        "since", "while", "though", "although", "despite", "throughout",
                        "during", "whether", "before", "after", "under", "above", "below",
                        "between", "among", "around", "against", "towards", "toward",
                        "through", "across", "behind", "beyond", "besides", "except",
                        "unless", "once", "where", "wherever", "whenever"
                    })
                    if lemma in ULTRA_BASIC_FUNC_WORDS:
                        continue

                    f_idx = cls._get_function_word_index()
                    _fam, stype = f_idx[lemma][0]
                    # Map syntactic complement type to schema-supported POS
                    assigned_pos = "preposition" if stype == "prepositional" else ("conjunction" if stype == "clausal" else "adverb")
                    w_cefr = cls.get_word_cefr(lemma)
                    is_awl = w_cefr in ("B2", "C1", "C2")
                    score = (24.0 if is_awl else 12.0) + min(len(lemma), 8) * 0.3
                    _add_or_update_candidate(lemma, sid, sent_clean, assigned_pos, is_awl, round(score, 2), is_connective_val=True)
                    continue

                # 2. Open-Class Content Words (Noun, Verb, Adjective, Adverb)
                if token.pos_ not in VALID_POS_MAP:
                    continue
                if token.is_stop:
                    continue
                # Reject mid-sentence capitalized proper nouns / acronym expansions (e.g., 'Placement' in 'AP (Advanced Placement)')
                if token.text and token.text[0].isupper() and token.i > 0 and doc[token.i - 1].text not in (".", "!", "?", ":", ";", '"', "\u201c", "'", "\u2018", "\u2014", "--"):
                    if token.pos_ == "PROPN" or doc[token.i - 1].text == "(" or (token.i > 0 and doc[token.i - 1].text[0].isupper()):
                        continue
                if not cls.is_pedagogical_target(lemma, ent_type=token.ent_type_):
                    continue
                if lemma in TRIVIAL_WORDS:
                    continue

                w_cefr = cls.get_word_cefr(lemma) or "B1"
                is_awl = w_cefr in ("B2", "C1", "C2")

                # Dependent-connective gate (extraction-stage quality, Pillar 0).
                if token.pos_ == "ADV" and lemma in DEPENDENT_CONNECTIVES:
                    required_prep = DEPENDENT_CONNECTIVES[lemma]
                    has_prep = (
                        any(ch.dep_ == "prep" and ch.text.lower() == required_prep for ch in token.children)
                        or (token.i + 1 < len(doc) and doc[token.i + 1].text.lower() == required_prep)
                    )
                    if not has_prep:
                        continue
                    phrase = f"{lemma} {required_prep}"
                    score = (28.0 if is_awl else 16.0) + min(len(phrase), 10) * 0.3
                    _add_or_update_candidate(phrase, sid, sent_clean, "preposition", is_awl, round(score, 2), is_connective_val=True)
                    continue

                if lemma == "following" and re.search(r"\bthe\s+following\b", sent_clean, re.IGNORECASE):
                    continue
                l_entry = cls.get_ldoce_entry(lemma) or {}
                # Comparative/Superlative Adjective De-inflection Fallback:
                # If a token is an inflected comparative (-er) or superlative (-est) adjective (e.g. 'plumper' -> 'plump', 'warmer' -> 'warm'),
                # resolve to the attested base adjective lemma in PedLex and prevent spurious agent/instrument noun entries
                # (e.g. 'warmer' [noun = plate warmer], 'cooler' [noun = cool box]) from hijacking predicative or causative adjective slots.
                base_adj = cls.deinflect_comparative_superlative(surf_low) or cls.deinflect_comparative_superlative(lemma)
                if base_adj:
                    is_causative = cls.is_causative_object_complement(token, doc)
                    is_copula_pred = (
                        token.head.lemma_.lower() in ("be", "become", "seem", "look", "feel", "get", "remain", "stay", "grow", "appear", "sound")
                        and not any(c.pos_ == "VERB" and c.tag_ in ("VBN", "VBG") for c in token.head.children)
                    )
                    is_attr_before_noun = (
                        token.dep_ == "amod"
                        or (token.i + 1 < len(doc) and doc[token.i + 1].pos_ in ("NOUN", "PROPN") and not doc[token.i + 1].text.startswith((".", ",", "!", "?", ":", ";", '"', "'", "”", "’")))
                    )
                    if (
                        token.pos_ == "ADJ"
                        or token.tag_ in ("JJR", "JJS")
                        or is_causative
                        or is_copula_pred
                        or is_attr_before_noun
                        or token.dep_ in ("acomp", "oprd")
                    ):
                        lemma = base_adj
                        l_entry = cls.get_ldoce_entry(base_adj) or {}
                        w_cefr = cls.get_word_cefr(base_adj) or "B1"
                        is_awl = w_cefr in ("B2", "C1", "C2")
                if l_entry.get("kind") == "derived":
                    continue
                if cls._ldoce_conn is not None and not l_entry.get("senses"):
                    continue
                # Bound Idiom Constituent Gate:
                if (
                    token.pos_ == "NOUN"
                    and token.dep_ in ("dobj", "obj", "pobj", "attr")
                    and not any(ch.dep_ in ("det", "poss", "amod", "nummod") for ch in token.children)
                    and token.head is not token
                ):
                    head_lem = token.head.lemma_.lower()
                    pair_forms = {f"{head_lem} {lemma}", f"{head_lem} {surf_low}"}
                    is_bound_verbal_unit = False
                    for s_u in (l_entry.get("senses") or []):
                        u_list_s = s_u.get("units") or []
                        s_def_u = str(s_u.get("definition") or "").strip().lower()
                        if u_list_s and s_def_u.startswith(("to ", "if ", "used ")):
                            for u_raw in u_list_s:
                                u_str = (u_raw.get("phrase") if isinstance(u_raw, dict) else str(u_raw or "")).strip().lower()
                                u_clean_p = re.sub(r"\[.*?\]|\(.*?\)", "", u_str).strip()
                                if u_clean_p in pair_forms:
                                    is_bound_verbal_unit = True
                                    break
                        if is_bound_verbal_unit:
                            break
                    if is_bound_verbal_unit:
                        continue

                # Comprehension-First Tri-Factor Lexical Ranking (Longman Core 3000/1000 + CEFR + AWL):
                # Balances:
                #   1. Longman Core 3000 (`core_level == 1` [2001-3000] & `core_level == 2` [1001-2000])
                #   2. Academic Word List (`is_awl`)
                #   3. Beyond-3000 Comprehension Blockers (`core_level == 0`) & CEFR (`B1`..`C2`)
                #   4. Top-1000 Everyday Filter (`core_level == 3` [1-1000]: filters out A1/A2 everyday words
                #      like 'science', 'language', 'section', 'similar', 'establish', 'reveal', and dampens
                #      B1/B2 Top-1000 words like 'object', 'defeat' so they never crowd out Core 2000/3000 or AWL).
                core_raw = l_entry.get("core_level")
                core_lv = 0 if core_raw is None else int(core_raw)
                pedlex_awl = bool(l_entry.get("is_awl"))

                # Transparent Gerund / Verbal Noun De-inflation:
                # If a '-ing' noun (e.g. 'learning', 'understanding', 'finding', 'opening') derives from an
                # A1/A2 Top-1000 (`core_level == 3`) base verb, align its difficulty to its base verb.
                if lemma.endswith("ing") and len(lemma) > 5:
                    for b_cand in (lemma[:-3], lemma[:-4], lemma[:-3] + "e"):
                        if len(b_cand) >= 3:
                            b_ld = cls.get_ldoce_entry(b_cand)
                            if b_ld and int(b_ld.get("core_level") or 0) == 3 and (cls.get_word_cefr(b_cand) or "A1") in ("A1", "A2"):
                                core_lv = 3
                                w_cefr = "A2"
                                break

                # Compositional Closed Compound De-inflation:
                # If a `core_level == 0` word splits cleanly into two `core_level == 3` (Top-1000) halves
                # AND its primary definition explicitly contains one of those halves (e.g. 'birthplace' -> 'birth' + 'place'),
                # treat its frequency band as `core_level = 2` rather than an opaque beyond-3000 root.
                if core_lv == 0 and len(lemma) >= 8:
                    first_def = str(((l_entry.get("senses") or [{}])[0]).get("definition") or "").lower()
                    for split_i in range(4, len(lemma) - 3):
                        p1, p2 = lemma[:split_i], lemma[split_i:]
                        ld1, ld2 = cls.get_ldoce_entry(p1), cls.get_ldoce_entry(p2)
                        if (
                            ld1 and ld2
                            and int(ld1.get("core_level") or 0) == 3
                            and int(ld2.get("core_level") or 0) == 3
                            and (p1 in first_def or p2 in first_def)
                        ):
                            core_lv = 2
                            break

                # Filter out elementary A1 Top-1000 words ('science', 'language', 'famous', etc.) and short A1/A2 non-AWL words;
                # keep A2 / AWL Top-1000 words at negative base score (-6.0) only as last-resort fallback for A1/A2 beginner texts.
                if core_lv == 3 and w_cefr == "A1" and not pedlex_awl:
                    continue
                if w_cefr in ("A1", "A2") and core_lv >= 2 and not pedlex_awl and len(lemma) < 5:
                    continue

                ctx_pos = cls.determine_contextual_pos(lemma, sent_clean) or VALID_POS_MAP[token.pos_]
                reg_tags = [str(r).lower() for r in (l_entry.get("register_tags") or [])]
                is_informal = any("informal" in r or "spoken" in r or "slang" in r for r in reg_tags)

                if core_lv == 0:
                    if is_informal:
                        core_pts = 4.0
                    elif ctx_pos in ("verb", "adjective", "adverb"):
                        core_pts = 16.0
                    else:
                        core_pts = 15.0
                elif core_lv == 1:
                    core_pts = 16.0
                elif core_lv == 2:
                    core_pts = 12.0
                else:
                    core_pts = -6.0

                raw_cefr_pts = {
                    "C2": 22.0 if (core_lv >= 1 or ctx_pos in ("verb", "adjective")) else 20.0,
                    "C1": 20.0,
                    "B2": 16.0,
                    "B1": 12.0 if core_lv <= 2 else 6.0,
                    "A2": 6.0 if core_lv <= 2 else 0.0,
                    "A1": 0.0,
                }.get(w_cefr, 8.0)
                # Cross-calibrate cefrpy under-ratings using Longman frequency scarcity floor:
                # Beyond-3000 (`core=0`, e.g. 'unearth', 'rightful') and Core 2001-3000 (`core=1`, e.g. 'legacy', 'proclaim') floor at 16.0 (B2).
                core_floor_pts = {0: 16.0 if not is_informal else 8.0, 1: 16.0, 2: 6.0, 3: 0.0}.get(core_lv, 0.0)
                cefr_pts = max(raw_cefr_pts, core_floor_pts)

                # Intra-Tier AWL Priority ("同级加分，不能越级加分"):
                # Adjacent tiers of `core_pts` (16 vs 12) and `cefr_pts` (22 / 20 / 16 / 12 / 6) are separated by >= 4.0 pts.
                # Keeping `awl_pts` at +2.0 (strictly < 4.0) ensures AWL words always win within their own (core_level, CEFR) tier,
                # while never leapfrogging across tiers over higher-difficulty `core=0/1` or higher-CEFR targets.
                if pedlex_awl:
                    awl_pts = 2.0 if core_lv <= 2 else (1.0 if w_cefr not in ("A1", "A2") else 0.0)
                else:
                    awl_pts = 0.0

                score = round(core_pts + cefr_pts + awl_pts + min(len(lemma), 10) * 0.2, 2)
                is_awl_flag = bool(pedlex_awl or w_cefr in ("B2", "C1", "C2") or core_lv <= 1)

                _add_or_update_candidate(lemma, sid, sent_clean, ctx_pos, is_awl_flag, score)

        if not cand_map:
            return []

        candidates = list(cand_map.values())
        candidates.sort(key=lambda x: x["score"], reverse=True)

        selected: List[Dict[str, Any]] = []
        selected_words: Set[str] = set()
        sid_counts: Dict[str, int] = {}

        def _get_sid_index(sid_str: str) -> int:
            m = re.search(r'\d+', sid_str)
            return int(m.group(0)) if m else -999

        # Dynamic Sentence-Load Selection:
        # Balances intrinsic word comprehension value (`score`) against sentence density (`sid_counts[sid]`).
        # A 2nd word in the same sentence incurs a -2.5 pt intra-tier penalty (spreading same-tier words across
        # sentences first, while never forcing a higher-tier word to lose to a lower-tier word from another sentence);
        # 3rd+ words in the same sentence incur a steep -14.0 * cnt penalty.
        while len(selected) < target_count:
            best_cand: Optional[Dict[str, Any]] = None
            best_occ: Optional[Tuple[str, str, str]] = None
            best_eff_score = -999.0

            for item in candidates:
                w = item["word"]
                if w in selected_words or any(cls.are_same_word_family(w, sw) for sw in selected_words):
                    continue
                occs = item.get("occurrences") or [(item["sid"], item["quote"], item["part_of_speech"])]

                def _occ_penalty(occ: Tuple[str, str, str]) -> float:
                    s_id = occ[0]
                    cnt = sid_counts.get(s_id, 0)
                    s_num = _get_sid_index(s_id)
                    adj_clash = (
                        cnt == 0
                        and any(abs(s_num - _get_sid_index(ssid)) <= 1 for ssid, c_v in sid_counts.items() if c_v > 0)
                    )
                    density_pen = 2.5 * cnt if cnt <= 1 else 14.0 * cnt
                    return density_pen + (0.8 if adj_clash else 0.0)

                chosen_occ = min(occs, key=_occ_penalty)
                eff_score = item["score"] - _occ_penalty(chosen_occ)
                if eff_score > best_eff_score:
                    best_eff_score = eff_score
                    best_cand = item
                    best_occ = chosen_occ

            if best_cand is None or best_occ is None:
                break

            chosen_sid, chosen_quote, chosen_pos = best_occ
            selected.append({
                "sid": chosen_sid,
                "quote": chosen_quote,
                "word": best_cand["word"],
                "part_of_speech": chosen_pos,
                "is_awl": best_cand["is_awl"],
                "score": best_cand["score"],
                **({"is_connective": True} if best_cand.get("is_connective") else {}),
            })
            selected_words.add(best_cand["word"])
            sid_counts[chosen_sid] = sid_counts.get(chosen_sid, 0) + 1

        return selected[:target_count]

    @classmethod
    def build_top_k_vocab_candidates(
        cls,
        word: str,
        quote: str,
        pos: Optional[str] = None,
        k: int = 16,
        discourse_context: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Morphology-Gated Full-Spectrum Cross-POS Candidate Selection (NO Embedding Model).
        Retrieves top-k (k <= 16) candidate senses from PedLex across all morphologically
        compatible parts of speech using:
        1. Surface-token morphological POS compatibility (eliminating single-POS tagger lock-in)
        2. Syntactic valency / frame / particle agreement
        3. Simplified Lesk overlap, 2nd-order collocate projection, and discourse window cue scoring
        4. PedLex natural frequency order fallback
        """
        w_low_clean = (word or "").strip().lower()
        entry = cls.get_pedlex_entry(word)
        if not entry and ("-" in w_low_clean or " " in w_low_clean):
            u_res = cls._unit_owning_sense(
                w_low_clean.replace("-", " "), context_sentence=quote
            )
            if u_res and u_res[0] and u_res[1]:
                u_owner, u_sense, _u_row = u_res
                entry = {"headword": word, "pos": pos or u_sense.get("pos"), "senses": [dict(u_sense, pos=pos or u_sense.get("pos"))]}
        if entry and entry.get("kind") == "derived" and (
            not entry.get("senses")
            or all(s.get("gloss_source") in ("wordnet", "empty") or not s.get("definition") for s in entry.get("senses", []))
        ):
            return []
        raw_senses = entry.get("senses", []) if entry else []

        # Filter out morphological pointer senses ("the past participle of...", "the plural of...", "= a goodie")
        # whenever substantive semantic definitions exist in the entry
        _morph_ptr_re = re.compile(
            r"^(?:(?:the\s+)?(?:past\s+participle|past\s+tense|present\s+participle|plural|comparative|superlative)\s+of\b|=)",
            re.IGNORECASE
        )
        non_ptr_senses = [
            s for s in raw_senses
            if not _morph_ptr_re.match((s.get("definition") or "").strip())
        ]
        senses = non_ptr_senses if non_ptr_senses else raw_senses

        target_pos = pos or (entry.get("pos") if entry else None)
        target_p = cls._pos_code(target_pos)
        q_low = (quote or "").lower()

        # Determine morphologically compatible POS codes from the surface token(s) of `word` in `quote`
        all_sense_poses = {cls._pos_code(s.get("pos")) for s in senses if s.get("pos")}
        q_toks_raw = re.findall(r"\b[a-z]+(?:-[a-z]+)*\b", q_low)
        matched_surf_toks = [t for t in q_toks_raw if t == w_low_clean or cls.form_matches(t, w_low_clean)]
        morph_poses: Set[str] = set()
        if (
            target_p == "adj"
            and "adj" not in all_sense_poses
            and "n" in all_sense_poses
            and not w_low_clean.endswith(("ed", "ing"))
        ):
            # Attributive noun adjunct (e.g. 'fossil fuels', 'rival options') where PedLex records
            # noun (+ verb) senses but no standalone adjective sense: lock to noun senses.
            morph_poses = {"n"}
        elif not matched_surf_toks or w_low_clean in matched_surf_toks:
            # Base uninflected form (e.g. 'trap', 'ride', 'stamp', 'present') is morphologically compatible with all entry POS
            morph_poses = set(all_sense_poses) | ({target_p} if target_p else set())
        else:
            for st in matched_surf_toks:
                if st.endswith("ed"):
                    # An inflected -ed form of a non-ed base headword ('presented' != 'present') can ONLY inflect a verb
                    morph_poses.add("v")
                elif st.endswith("ing"):
                    morph_poses.add("v")
                elif st.endswith("s"):
                    morph_poses.update({"n", "v"})
                elif st.endswith(("er", "est")):
                    morph_poses.update({"adj", "adv"})
                else:
                    morph_poses.update({"v", "n", "adj"})
            if not (morph_poses & all_sense_poses) and target_p:
                morph_poses.add(target_p)
        if not morph_poses and target_p:
            morph_poses = {target_p}

        has_pos_match = bool(
            any(cls._sense_matches_pos(s, target_p) for s in senses)
            if target_p
            else (morph_poses and any(cls._pos_code(s.get("pos")) in morph_poses for s in senses))
        )
        # Zero-derived run-on guard: just like suffixed `kind == "derived"` entries (line 2805),
        # a zero-derived conversion listed in `derived_forms` (e.g. `showcase` verb under `showcase` noun)
        # has no independent dictionary definition in `senses` and must not steal the base POS definition.
        if (
            target_p
            and not has_pos_match
            and entry
            and any(
                isinstance(df, dict)
                and str(df.get("word") or "").strip().lower() == w_low_clean
                and cls._pos_code(df.get("pos")) == target_p
                for df in (entry.get("derived_forms") or [])
            )
        ):
            return []

        if not senses:
            candidates = []
        else:
            entry_filtered = dict(entry, senses=senses) if senses is not raw_senses else entry
            # Always pass target_pos=None so _lock_sense scores every sense under its own POS (`eff_p`),
            # while `pos_filtered_trace` enforces `_sense_in_morph_poses`.
            res = cls._lock_sense(
                entry_filtered,
                quote=quote,
                target_pos=None,
                return_scores=True,
                discourse_context=discourse_context
            )

            def _is_unattested_phrase_unit_sense(s_obj: Dict[str, Any]) -> bool:
                u_list = s_obj.get("units") or []
                if not u_list:
                    return False
                w_low = word.strip().lower()
                s_defn_low = str(s_obj.get("definition") or "").strip().lower()
                s_p = cls._pos_code(s_obj.get("pos")) or target_p
                # When extracting a single-word noun, any unit sense with a verbal/clausal/pragmatic definition
                # ('to ...', 'if ...', 'used ...', e.g. 'make sense' -> 'to be a sensible thing to do') defines
                # the multi-word idiom rather than the standalone noun.
                if " " not in w_low and s_p == "n" and s_defn_low.startswith(("to ", "if ", "used ")):
                    return True
                # If sense carries a core semantic signpost (e.g. [MAIL] on 'stamp' Sense 0) and is not a conditional/usage idiom,
                # its unit (e.g. 'postage stamp') is an illustrative compound rather than an obligatory multi-word idiom.
                if s_obj.get("signpost") and not s_defn_low.startswith(("to ", "if ", "used ")):
                    return False
                # If a verb sense explicitly declares [transitive], its prepositional unit (e.g. 'tend to sb/sth')
                # is an optional prepositional variant, not an obligatory multi-word idiom
                s_gram_low = str(s_obj.get("grammar") or "").lower()
                if s_p == "v" and re.search(r"(?<!in)transitive", s_gram_low):
                    return False
                # Check if any unit phrase is attested or shares its grammatical preposition/frame with the quote
                all_multi_unattested = True
                def _unit_tok_matches(q_text: str, u_tok: str) -> bool:
                    if cls.text_contains_form(q_text, u_tok):
                        return True
                    if u_tok == "be":
                        return bool(re.search(r"\b(?:be|am|is|are|was|were|been|being|'m|'s|'re)\b", q_text))
                    if u_tok == "have":
                        return bool(re.search(r"\b(?:have|has|had|having|'ve|'d)\b", q_text))
                    if u_tok == "get":
                        return bool(re.search(r"\b(?:get|gets|got|gotten|getting)\b", q_text))
                    if u_tok == "do":
                        return bool(re.search(r"\b(?:do|does|did|done|doing)\b", q_text))
                    if u_tok in ("yourself", "oneself"):
                        return bool(re.search(r"\b(?:myself|yourself|himself|herself|itself|ourselves|yourselves|themselves|oneself)\b", q_text))
                    if u_tok == "your":
                        return bool(re.search(r"\b(?:my|your|his|her|its|our|their|one's)\b", q_text))
                    return False

                for u_it in u_list:
                    u_phr = (u_it.get("phrase") if isinstance(u_it, dict) else str(u_it or "")).strip()
                    u_clean = re.sub(r'\[.*?\]|\(.*?\)', '', u_phr).strip().lower()
                    if " " not in w_low and u_clean.startswith(("not ", "no ", "never ")) and s_defn_low.startswith(("no ", "not ", "never ")):
                        continue
                    u_clean_no_slot = re.sub(r"\b(?:somebody's|someone's|something's|somebody|someone|something|sb's|sb|sth|one's|oneself|etc)\b\.?", "", u_clean).strip()
                    u_toks = [t for t in re.findall(r"[a-z]+", u_clean_no_slot) if t not in ("a", "an", "the", "some", "any", "doing")]
                    # For predicative adjective valency units starting with a copula/linking verb
                    # (e.g. 'be accustomed to (doing) something', 'be aware of', 'be fond of'),
                    # strip the leading copula so the unit is evaluated on its '<adj> + <prep>' core.
                    if (
                        s_p == "adj"
                        and len(u_toks) >= 2
                        and u_toks[0] in ("be", "get", "become", "grow", "feel", "seem", "look", "remain", "stay")
                        and cls.form_matches(u_toks[1], w_low)
                    ):
                        u_toks = u_toks[1:]
                    # Preserve 2-word nominal compound <modifier> + <headword> or <headword> + <noun> (e.g. 'postage stamp', 'comic book')
                    if (
                        s_p == "n"
                        and not s_defn_low.startswith(("to ", "if ", "used "))
                        and len(u_toks) == 2
                        and (
                            cls.form_matches(u_toks[0], w_low)
                            or (
                                cls.form_matches(u_toks[-1], w_low)
                                and u_toks[0] not in ("in", "on", "at", "by", "for", "of", "to", "with", "from", "under", "out", "beyond", "without", "within")
                            )
                        )
                    ):
                        all_multi_unattested = False
                        break
                    # When extracting a single-word headword (e.g. 'likelihood'), do NOT allow a prepositional/adverbial idiom unit
                    # starting with another word (e.g. 'in all likelihood') to define the bare noun!
                    if " " not in w_low and len(u_toks) >= 2 and not cls.form_matches(u_toks[0], w_low):
                        continue
                    if cls.match_typed_slot_pattern(u_phr, quote, headword=w_low):
                        all_multi_unattested = False
                        break
                    if len(u_toks) <= 1:
                        all_multi_unattested = False
                        break
                    if " " not in w_low and u_toks[0] in ("not", "no", "never") and s_defn_low.startswith(("no ", "not ", "never ")):
                        continue
                    if all(_unit_tok_matches(q_low, t) for t in u_toks):
                        all_multi_unattested = False
                        break
                    # Preserve unit sense if it is a <headword> + <preposition> frame (e.g. 'an embarrassment of riches')
                    # and the quote contains <headword> + <preposition> (e.g. 'an embarrassment of possibilities')
                    if len(u_toks) >= 2 and u_toks[0] == w_low and u_toks[1] in ("of", "to", "for", "with", "in", "on", "from", "about"):
                        if f"{w_low} {u_toks[1]}" in q_low:
                            all_multi_unattested = False
                            break
                if not all_multi_unattested:
                    return False
                return True

            def _sense_in_morph_poses(s_obj: Dict[str, Any]) -> bool:
                sp_code = cls._pos_code(s_obj.get("pos"))
                if not sp_code:
                    return True
                if target_p and target_p in all_sense_poses:
                    if cls._sense_matches_pos(s_obj, target_p):
                        return True
                    # Allow cross-POS sense ONLY if it declares an explicit multi-word pattern/unit matching the quote
                    for u_it in (s_obj.get("units") or []):
                        u_phr = (u_it.get("phrase") if isinstance(u_it, dict) else str(u_it or "")).strip()
                        if u_phr and " " in u_phr and cls.match_typed_slot_pattern(u_phr, quote, headword=w_low_clean):
                            return True
                    for p_it in (s_obj.get("patterns") or []):
                        p_str = str(p_it or "").strip()
                        if p_str and " " in p_str and cls.match_typed_slot_pattern(p_str, quote, headword=w_low_clean):
                            return True
                    return False
                if not morph_poses:
                    return not has_pos_match or cls._sense_matches_pos(s_obj, target_p)
                return sp_code in morph_poses or any(cls._sense_matches_pos(s_obj, mp) for mp in morph_poses)

            if not res or len(res) < 5:
                matched = [
                    s for s in senses
                    if _sense_in_morph_poses(s)
                    and not _is_unattested_phrase_unit_sense(s)
                ] or [s for s in senses if _sense_in_morph_poses(s)] or senses
                candidates = [{
                    "definition": cls.format_ldoce_definition(s) or s.get("definition", ""),
                    "pos": s.get("pos") or pos or target_pos,
                    "sense_idx": i,
                    "signpost": s.get("signpost", ""),
                    "patterns": list(dict.fromkeys(list(s.get("patterns") or []) + list(s.get("units") or []))),
                    "examples": s.get("examples", []),
                    "raw_sense": s,
                    "score": 0.0,
                    "margin": 0.0,
                    "confidence": "high" if len(matched) <= 1 else "low",
                } for i, s in enumerate(matched[:k])]
            else:
                idx, score, margin, conf, score_trace = res
                # Enforce morphological POS compatibility gate and filter unattested multi-word idiom senses
                pos_filtered_trace = [
                    (s_i, sc) for s_i, sc in score_trace
                    if _sense_in_morph_poses(senses[s_i])
                ] or score_trace
                non_idiom_trace = [
                    (s_i, sc) for s_i, sc in pos_filtered_trace
                    if not _is_unattested_phrase_unit_sense(senses[s_i])
                ] or pos_filtered_trace

                # Symbolic Hard Pruning (Grammar / Number / Valency / Register / Domain / Phrasal Particle):
                # Eliminate senses that violate hard syntactic/morphological invariants in the quote
                # whenever at least one grammatically compatible sense survives.
                q_seq_prune = [w.lower() for w in re.findall(r"\b[a-zA-Z']+\b", quote or "")]
                target_w_clean_p = (entry.get("word") or word or "").strip().lower()
                headword_tokens_p = set(re.findall(r"\b[a-z]+\b", target_w_clean_p))

                _pv_slot_stop_p = {"something", "somebody", "someone", "sth", "sb", "<->", "\u2194", "oneself", "one's", "etc"}
                pv_sense_map_p: Dict[int, List[str]] = {}
                for pv in entry.get("phrasal_verbs") or []:
                    phr = re.sub(r"[\(\)\[\]]", " ", pv.get("phrase", ""))
                    p_words = [
                        w for w in phr.lower().split()
                        if w != target_w_clean_p and not cls.form_matches(w, target_w_clean_p) and w not in _pv_slot_stop_p
                    ]
                    if p_words:
                        for pvs in pv.get("senses") or []:
                            p_defn = pvs.get("definition")
                            for s_idx_p, s_cand_p in enumerate(senses):
                                if (
                                    s_cand_p.get("definition") == p_defn
                                    and not s_cand_p.get("sensenum")
                                    and s_cand_p.get("homograph_num") in (None, 0)
                                ):
                                    pv_sense_map_p[s_idx_p] = p_words

                q_morph_num: List[str] = []
                q_has_count_det = False
                q_is_pred = False
                q_is_attr = False
                q_has_dobj = False
                q_is_passive = False
                q_has_adv_or_prep_comp = False
                if quote and len(non_idiom_trace) > 1:
                    try:
                        nlp_p = cls.get_spacy()
                        doc_p = nlp_p(cls._strip_lexical_quotes(quote))
                        m_toks = [
                            t for t in doc_p
                            if t.lemma_.lower() == target_w_clean_p or t.text.lower() == target_w_clean_p
                            or cls.form_matches(t.text.lower(), target_w_clean_p)
                            or cls.form_matches(t.lemma_.lower(), target_w_clean_p)
                            or (headword_tokens_p and (t.lemma_.lower() in headword_tokens_p or t.text.lower() in headword_tokens_p))
                        ]
                        if m_toks:
                            mt = m_toks[0]
                            if mt.pos_ == "NOUN":
                                q_morph_num = list(mt.morph.get("Number") or [])
                                q_has_count_det = any(
                                    ch.dep_ in ("det", "nummod")
                                    and ch.lemma_.lower() in ("a", "an", "one", "two", "three", "many", "few", "several", "each", "every", "another", "both")
                                    for ch in mt.children
                                )
                            if mt.dep_ in ("acomp", "oprd") or (
                                mt.head != mt
                                and mt.head.lemma_.lower() in ("be", "become", "seem", "feel", "get", "remain", "stay", "grow", "turn", "appear", "look", "sound")
                                and mt.dep_ != "amod"
                            ):
                                q_is_pred = True
                            elif mt.dep_ == "amod" and mt.head.pos_ in ("NOUN", "PROPN") and mt.i < mt.head.i:
                                q_is_attr = True
                            if mt.pos_ == "VERB" or "v" in morph_poses:
                                q_is_passive = any(c.dep_ in ("auxpass", "aux:pass", "nsubjpass", "nsubj:pass") for c in mt.children)
                                q_has_dobj = any(c.dep_ in ("dobj", "obj") for c in mt.children) or q_is_passive
                                q_has_adv_or_prep_comp = any(
                                    c.dep_ in ("prep", "prt", "npadvmod")
                                    or (c.dep_ == "advmod" and c.i > mt.i)
                                    for c in mt.children
                                )
                    except Exception:
                        pass

                def _is_grammatically_hard_pruned_sense(s_idx_c: int, sc_c: float) -> bool:
                    if sc_c <= -40.0:
                        return True
                    s_c = senses[s_idx_c]
                    s_p_c = cls._pos_code(s_c.get("pos")) or target_p
                    s_gram_c = (s_c.get("grammar") or "").lower()
                    s_reg_c = (s_c.get("register") or "").lower()
                    s_sp_c = (s_c.get("signpost") or "").lower()
                    s_defn_c = (s_c.get("definition") or "").lower()

                    # 1. Missing mandatory phrasal verb particle
                    if s_idx_c in pv_sense_map_p:
                        req_ps = pv_sense_map_p[s_idx_c]
                        if not all(
                            any(cls.form_matches(tok_q, alt) for tok_q in q_seq_prune for alt in rp.split("/") if alt)
                            for rp in req_ps
                        ):
                            return True
                    elif s_p_c == "v":
                        m_pv_def = re.match(
                            rf"^if\s+[^,]+?\s+(?:{re.escape(target_w_clean_p)}[a-z]*)\s+(up|down|out|off|away|back|over|through|around|round|apart|aside|ahead|along|about)\b",
                            s_defn_c.strip(),
                        )
                        if m_pv_def and not any(cls.form_matches(tok_q, m_pv_def.group(1)) for tok_q in q_seq_prune):
                            return True

                    # 2. Noun countability, number & verbal idiom unit hard clash (only for noun senses)
                    if s_p_c == "n":
                        if " " not in target_w_clean_p and s_c.get("units") and s_defn_c.strip().startswith(("to ", "if ", "used ")):
                            return True
                        if (("Plur" in q_morph_num and not target_w_clean_p.endswith("s")) or q_has_count_det) and "[uncountable]" in s_gram_c and "[countable]" not in s_gram_c:
                            return True
                        if ("Plur" in q_morph_num and not target_w_clean_p.endswith("s")) and "[singular]" in s_gram_c and "[countable]" not in s_gram_c and "[plural]" not in s_gram_c:
                            return True
                        if ("Sing" in q_morph_num or q_has_count_det) and "[plural]" in s_gram_c and "[singular]" not in s_gram_c and "[countable]" not in s_gram_c:
                            return True

                    # 3. Adjective syntactic slot hard clash (only for adjective senses)
                    if s_p_c == "adj":
                        if q_is_pred and "[only before noun]" in s_gram_c:
                            s_ex_pat_c = " ".join(list(s_c.get("examples") or []) + list(s_c.get("patterns") or [])).lower()
                            s_ex_toks_c = {w for w in re.findall(r"\b[a-z]{4,}\b", s_ex_pat_c) if w != target_w_clean_p}
                            if not any(cls.form_matches(tok_q, w_ex) for tok_q in q_seq_prune for w_ex in s_ex_toks_c):
                                return True
                        if q_is_attr and ("[not before noun]" in s_gram_c or "[never before noun]" in s_gram_c):
                            return True

                    # 4. Verb transitivity & mandatory locative/prepositional complement clash (only for verb senses)
                    if s_p_c == "v":
                        if q_has_dobj and "[intransitive]" in s_gram_c and not re.search(r"(?<!in)transitive", s_gram_c):
                            return True
                        if "always + adverb/preposition" in s_gram_c and not q_has_adv_or_prep_comp:
                            if not (q_has_dobj and re.search(r",\s*transitive\b", s_gram_c)):
                                return True

                    # 5. Spoken register gate in non-dialogue narrative
                    if "spoken" in s_reg_c or "[spoken]" in s_gram_c:
                        is_dialogue = any(qc in (quote or "") for qc in ('"', "'", "“", "”", "‘", "’"))
                        if not is_dialogue:
                            return True

                    # 6. Specialized domain gate (medical / law / technical) without domain triggers
                    is_med = s_reg_c == "medical" or "[medical]" in s_gram_c or s_sp_c in ("illness", "medical", "disease")
                    if is_med and not any(w in q_low for w in (
                        "illness", "disease", "symptom", "patient", "doctor", "hospital", "medicine",
                        "syndrome", "fever", "pain", "clinic", "clinical", "infection", "treatment",
                        "suffer", "suffered", "suffering", "mild", "severe", "fatal", "heart", "brain",
                        "blood", "attack", "paralyz", "recover", "health"
                    )):
                        return True
                    is_law = s_reg_c in ("law", "legal") or "[law]" in s_gram_c or "[legal]" in s_gram_c or s_sp_c in ("law", "legal", "court")
                    if is_law and not any(w in q_low for w in (
                        "court", "judge", "jury", "trial", "lawyer", "attorney", "lawsuit", "legal", "statute", "prosecution", "defendant", "verdict"
                    )):
                        return True
                    is_tech = (
                        s_reg_c in ("chemistry", "physics", "biology", "computing")
                        or (s_reg_c == "technical" and s_idx_c > 0)
                        or any(tag in s_gram_c for tag in ("[chemistry]", "[physics]", "[computing]"))
                        or ("[technical]" in s_gram_c and s_idx_c > 0)
                        or s_sp_c in ("chemistry", "physics", "technical", "chemical")
                    )
                    if is_tech and not any(w in q_low for w in (
                        "chemical", "acid", "substance", "atom", "molecule", "reaction", "physics",
                        "laboratory", "voltage", "circuit", "alkali", "gas", "metal", "compound",
                        "technology", "technological", "scientific", "science", "engineering",
                        "robot", "robotics", "inventor", "inventors", "technician", "technicians",
                        "experiment", "printing", "device", "machine", "software", "hardware",
                        "digital", "electronic", "electronics", "computer"
                    )):
                        return True
                    is_grammar_meta = (
                        s_reg_c in ("grammar", "linguistics")
                        or "[grammar]" in s_gram_c
                        or s_sp_c in ("grammar", "linguistics")
                    )
                    if is_grammar_meta and not any(w in q_low for w in (
                        "grammar", "grammatical", "sentence", "verb", "noun", "adjective", "adverb",
                        "pronoun", "preposition", "clause", "phrase", "tense", "plural", "singular",
                        "syntax", "linguistic", "linguistics", "conjugat", "inflect"
                    )):
                        return True

                    # 6b. Weather/elements domain restriction in WordNet/PedLex glosses ('(of the elements) ...')
                    if s_defn_c.strip().startswith(("(of the elements)", "(of weather)", "(of wind")) and not any(
                        w in q_low for w in ("storm", "wind", "rain", "wave", "sea", "weather", "element", "sky", "tempest", "gale")
                    ):
                        return True

                    # 7. Cross-reference inflection or variant stub
                    if re.match(r"^(?:(?:the\s+)?(?:past\s+participle|past\s+tense|present\s+participle|plural|comparative|superlative)\s+of\b|=)", s_defn_c.strip()):
                        return True

                    return False

                pruned_trace = [
                    (s_i, sc) for s_i, sc in non_idiom_trace
                    if not _is_grammatically_hard_pruned_sense(s_i, sc)
                ] or non_idiom_trace

                # 8. Reflexive verb frame hard lock ('reverse myself' -> 'reverse yourself'):
                if target_p == "v" and len(pruned_trace) > 1 and quote:
                    surf_alts = "|".join(re.escape(t) for t in (matched_surf_toks or [w_low_clean]))
                    if re.search(rf"\b(?:{surf_alts})\s+(?:myself|yourself|himself|herself|itself|ourselves|yourselves|themselves|oneself)\b", q_low):
                        refl_trace = []
                        for s_i, sc in pruned_trace:
                            s_cand_r = senses[s_i]
                            u_strs = [str(u.get("phrase") if isinstance(u, dict) else u).lower() for u in (s_cand_r.get("units") or [])]
                            p_strs = [str(p).lower() for p in (s_cand_r.get("patterns") or [])]
                            if any("yourself" in up or "oneself" in up for up in (u_strs + p_strs)):
                                refl_trace.append((s_i, sc))
                        if refl_trace:
                            pruned_trace = refl_trace

                sorted_trace = sorted(pruned_trace, key=lambda x: -x[1])
                valid_idx_set = {s_i for s_i, _ in pruned_trace}
                ordered_indices = [s_i for s_i, _ in sorted_trace]
                if idx is not None and idx in valid_idx_set:
                    ordered_indices = [idx] + [s_i for s_i in ordered_indices if s_i != idx]
                elif not ordered_indices:
                    ordered_indices = [0]

                candidates = []
                seen_cand_keys = set()
                for s_i in ordered_indices:
                    if len(candidates) >= k:
                        break
                    s = senses[s_i]
                    cand_def = cls.format_ldoce_definition(s) or s.get("definition", "")
                    cand_pos = s.get("pos") or pos or target_pos
                    dedup_key = (
                        str(cand_pos or "").strip().lower(),
                        re.sub(r"[^a-z0-9]+", " ", str(s.get("definition") or cand_def).lower()).strip(),
                    )
                    if dedup_key[1] and dedup_key in seen_cand_keys:
                        continue
                    seen_cand_keys.add(dedup_key)
                    s_exs = list(s.get("examples") or [])
                    if quote and len(s_exs) > 1:
                        _ex_stop = set(cls._PATTERN_FUNCTION_WORDS) | set(cls._PATTERN_PREPS) | {
                            "want", "they", "have", "from", "with", "that", "this", "what", "when", "where",
                            "which", "their", "there", "also", "very", "more", "most", "some", "many", "much",
                            "other", "such", "only", "just", "even", "into", "over", "after", "before", "about",
                        }
                        q_open_toks = {
                            w for w in re.findall(r"[a-z]{4,}", q_low)
                            if w not in _ex_stop and w != w_low_clean and w not in cls.inflected_forms(w_low_clean)
                        }
                        if q_open_toks:
                            s_exs.sort(
                                key=lambda ex_str: -sum(1 for qt in q_open_toks if re.search(rf"\b{re.escape(qt)}\b", str(ex_str).lower()))
                            )
                    if quote and s.get("pattern_bundles"):
                        for pb in (s.get("pattern_bundles") or []):
                            if isinstance(pb, dict) and pb.get("pattern") and pb.get("example"):
                                pb_pat = str(pb["pattern"]).strip()
                                pb_ex = str(pb["example"]).strip()
                                if pb_ex and cls.match_typed_slot_pattern(pb_pat, quote, headword=(entry.get("word") or word)):
                                    s_exs = [pb_ex] + [e for e in s_exs if e != pb_ex]
                                    break
                    s_unit_strs = [u.get("phrase") if isinstance(u, dict) else str(u or "") for u in (s.get("units") or [])]
                    candidates.append({
                        "definition": cand_def,
                        "pos": cand_pos,
                        "sense_idx": s_i,
                        "signpost": s.get("signpost", ""),
                        "patterns": list(dict.fromkeys([str(p) for p in (s.get("patterns") or []) if p] + [u for u in s_unit_strs if u])),
                        "examples": s_exs,
                        "raw_sense": s,
                        "score": next((sc for s_idx, sc in score_trace if s_idx == s_i), 0.0),
                        "margin": float(margin or 0.0),
                        "confidence": conf or "high",
                    })

        # Cross-Lexicon Pool Fusion (WordNet oewn:2025+):
        # 1. Trigger when PedLex has NO senses matching the target POS.
        # 2. Also trigger when PedLex has only 1 sense for the target POS AND that sole sense is pragmatically
        #    restricted ([disapproving], [spoken], [informal], [humorous], [old-fashioned]) or a narrow physical
        #    conditional subject ('if a piece of ...'), so neutral/academic senses attested in WordNet with
        #    authentic examples can enter the pool for Stage 2 1-token LLM arbitration.
        sole_pedlex_restricted = False
        if candidates and has_pos_match:
            q_content_toks = set(re.findall(r"[a-z]{3,}", q_low))
            if re.search(r"\b(?:myself|yourself|himself|herself|itself|ourselves|yourselves|themselves|oneself)\b", q_low):
                q_content_toks.update({"yourself", "oneself"})
            _pat_meta = {"somebody", "something", "sb", "sth", "your", "one", "etc", "and", "or", "with", "for", "from", "into", "about", "over", "under", "between", "among", "against", "towards", "toward", "upon", "within", "without", "doing", "that", "what", "where", "when", "whether", "how", "adjective", "noun", "verb", "adverb"}
            all_restricted_or_mismatched = True
            for c_item in candidates:
                c0_def = str(c_item.get("definition") or "").lower()
                c0_raw = c_item.get("raw_sense") if isinstance(c_item.get("raw_sense"), dict) else {}
                c0_reg = str(c0_raw.get("register") or "").lower()
                c0_gram = str(c0_raw.get("grammar") or "").lower()
                is_restr = (
                    any(tag in c0_reg or tag in c0_gram or f"[{tag}]" in c0_def for tag in ("disapproving", "humorous"))
                    or c0_def.startswith(("if a piece of ", "if a chemical substance "))
                )
                c0_pats = [str(p) for p in (c0_raw.get("patterns") or []) if isinstance(p, str) and str(p).strip()]
                has_explicit_colloc_mismatch = False
                if c0_pats and target_p == "v":
                    q_lemmas = q_content_toks | {
                        lem for qw in q_content_toks for lem in (cls._surface_lemma(qw),) if lem
                    }
                    w_lemma = cls._surface_lemma(word) or word.lower()
                    all_pats_mismatched = True
                    for p_str in c0_pats:
                        p_collocs = set()
                        for pw in re.findall(r"[a-z]{4,}", p_str.lower()):
                            pw_lem = cls._surface_lemma(pw)
                            if pw != word.lower() and (pw_lem or pw) != w_lemma and pw not in _pat_meta:
                                p_collocs.add(pw)
                                if pw_lem:
                                    p_collocs.add(pw_lem)
                        if not p_collocs or (p_collocs & q_lemmas):
                            all_pats_mismatched = False
                            break
                    if all_pats_mismatched:
                        has_explicit_colloc_mismatch = True
                if not is_restr and not has_explicit_colloc_mismatch:
                    all_restricted_or_mismatched = False
                    break
            if all_restricted_or_mismatched:
                sole_pedlex_restricted = True

        needs_wn_fallback = (
            not candidates
            or not has_pos_match
            or sole_pedlex_restricted
        )
        if cls._is_pedlex_disabled() and needs_wn_fallback:
            try:
                wn = cls.get_wordnet()
                if wn:
                    wn_pos_map = {"n": ("n",), "v": ("v",), "a": ("a", "s"), "adj": ("a", "s"), "r": ("r",), "adv": ("r",)}
                    wn_poses = wn_pos_map.get(target_p) if target_p else None
                    syns = [s for p_code in wn_poses for s in (wn.synsets(word, pos=p_code) or [])] if wn_poses else wn.synsets(word)
                    lem_w = cls._surface_lemma(word)
                    if not syns and lem_w and lem_w != word:
                        syns = [s for p_code in wn_poses for s in (wn.synsets(lem_w, pos=p_code) or [])] if wn_poses else wn.synsets(lem_w)
                    if syns:
                        seen_norm = {re.sub(r"[^a-z0-9]+", " ", (c.get("definition") or "").lower()).strip() for c in candidates}
                        q_words = set(re.findall(r"[a-z]{3,}", q_low))
                        scored_syns = []
                        for syn in syns:
                            s_def = (syn.definition() or "").strip()
                            if not s_def:
                                continue
                            s_norm = re.sub(r"[^a-z0-9]+", " ", s_def.lower()).strip()
                            if s_norm in seen_norm:
                                continue
                            s_exs = [str(e).strip().strip('"“”') for e in (syn.examples() or []) if e]
                            # Only include WordNet synsets that either have examples or when PedLex has 0 POS-matched candidates
                            if not s_exs and candidates and has_pos_match:
                                continue
                            seen_norm.add(s_norm)
                            s_bag = set(re.findall(r"[a-z]{3,}", (s_def + " " + " ".join(s_exs)).lower()))
                            overlap = len(q_words & s_bag)
                            if f"{word.lower()} of " in q_low and any(kw in s_def.lower() for kw in ("quantity", "excess", "number", "amount", "multitude", "group", "competitors")):
                                overlap += 3
                            scored_syns.append((overlap, syn, s_def, s_exs, s_norm))
                        scored_syns.sort(key=lambda x: -x[0])
                        if not has_pos_match and scored_syns:
                            candidates = []
                        is_wn_sentinel = bool(candidates and has_pos_match)
                        wn_slots = min(2, len(scored_syns)) if is_wn_sentinel else min(k, len(scored_syns))
                        if is_wn_sentinel and len(candidates) + wn_slots > k:
                            candidates = candidates[:max(1, k - wn_slots)]
                        for _ov, _syn, s_def, s_exs, s_norm in scored_syns[:wn_slots]:
                            if len(candidates) >= k:
                                break
                            if target_p == "v" and not s_def.lower().startswith("to "):
                                s_def = f"to {s_def}"
                            cand_exs = [] if is_wn_sentinel else s_exs
                            candidates.append({
                                "definition": s_def,
                                "pos": target_pos or pos or "noun",
                                "sense_idx": -1,
                                "signpost": "",
                                "patterns": [],
                                "examples": cand_exs,
                                "raw_sense": {"definition": s_def, "pos": target_pos or pos or "noun", "examples": cand_exs},
                                "score": 0.0,
                                "margin": 0.0,
                                "confidence": "high",
                            })
            except Exception:
                pass

        return candidates[:k]


    @classmethod
    def ensure_distinct_example(
        cls,
        word_or_formula: str,
        pos: str,
        current_example: str,
        quoted_sentence: str,
        sense: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Guarantees that Example Usage is NEVER a verbatim duplicate of Quoted Sentence,
        and ALWAYS demonstrates the target word or expression in an authentic dictionary sentence
        from its locked sense (or from monosemous corpus entries). Returns '' if no valid example exists.
        """
        clean_target = re.sub(r'\[.*?\]|\(.*?\)', '', str(word_or_formula or '')).strip()
        clean_target = re.sub(r"\b(?:sb's|sb|sth|one's|oneself)\b", "", clean_target).strip()
        clean_target = re.sub(r"\s+", " ", clean_target).strip()
        clean_low = clean_target.lower()
        u_key = cls._unit_key(clean_target)
        u_key_no_cop = " ".join(u_key.split()[1:]) if (len(u_key.split()) >= 3 and u_key.split()[0] in ("be", "feel", "get")) else ""
        is_passive_cop = bool(u_key_no_cop and u_key_no_cop.split()[0].endswith("ed"))
        is_multi = len(u_key.split()) >= 2 and "-" not in clean_low
        q_norm = re.sub(r'\s+', ' ', str(quoted_sentence or '').strip()).lower()
        pos_low = str(pos or '').strip().lower()

        def _is_valid_distinct(ex_str: str, check_pos: bool = True) -> bool:
            if not ex_str or not str(ex_str).strip():
                return False
            ex_clean = str(ex_str).strip().strip('"“”')
            e_norm = re.sub(r'\s+', ' ', ex_clean).lower()
            if e_norm == q_norm or (q_norm and (e_norm in q_norm or q_norm in e_norm) and abs(len(e_norm) - len(q_norm)) <= 12):
                return False
            if is_multi:
                if pos_low in ("phrasal verb", "verb") and u_key:
                    v_head = u_key.split()[0]
                    if re.search(rf"\b(?:a|an|the|this|that|each|every|one|another)\s+(?:\w+\s+)?{re.escape(v_head)}\b", e_norm):
                        return False
                    if check_pos and cls.determine_contextual_pos(v_head, ex_clean) == "noun":
                        return False
                if is_passive_cop:
                    part_tok = u_key_no_cop.split()[0]
                    v_stems = [c for c in ([cls._surface_lemma(part_tok)] + cls._headword_candidates(part_tok)) if c and c != part_tok and cls.get_ldoce_entry(c)]
                    v_stem = v_stems[0] if v_stems else cls._surface_lemma(part_tok)
                    # Reject singular noun false positives (e.g. 'a standard feature in' for 'be featured in')
                    if v_stem and v_stem != part_tok and re.search(rf"\b(?:a|an|the)\s+(?:\w+\s+)?{re.escape(v_stem)}\b", e_norm) and part_tok not in e_norm:
                        return False
                if cls._unit_example(u_key, [ex_clean], allow_modifier=True) or cls._relaxed_unit_example(u_key, [ex_clean]):
                    return True
                u_toks_ex = u_key.split()
                if len(u_toks_ex) == 3 and u_toks_ex[0] in ("do", "make", "take", "give", "show", "have", "play", "pay", "put", "keep", "get"):
                    vn_core_ex = f"{u_toks_ex[0]} {cls._surface_lemma(u_toks_ex[2]) or u_toks_ex[2]}"
                    if cls._unit_example(vn_core_ex, [ex_clean], allow_modifier=True) or cls._unit_example(f"{u_toks_ex[0]} {u_toks_ex[2]}", [ex_clean], allow_modifier=True):
                        return True
                if u_key_no_cop:
                    if cls._unit_example(u_key_no_cop, [ex_clean], allow_modifier=True):
                        return True
                    # British/American doubled-consonant participle equivalence ('modeled on' <-> 'modelled on')
                    p0 = u_key_no_cop.split()[0]
                    if p0.endswith("led") and not p0.endswith("lled"):
                        uk_br = p0[:-3] + "lled " + " ".join(u_key_no_cop.split()[1:])
                        if cls._unit_example(uk_br, [ex_clean], allow_modifier=True):
                            return True
                    # Transitive active / slash-preposition pattern equivalence on the locked verb sense (e.g. 'measure something in something' for 'be measured in', 'admit somebody to/into something' for 'be admitted to')
                    if is_passive_cop and sense and isinstance(sense, dict):
                        p_word = u_key_no_cop.split()[-1]
                        v_stems = [c for c in ([cls._surface_lemma(p0)] + cls._headword_candidates(p0)) if c and c != p0 and cls.get_ldoce_entry(c)]
                        v_stem = v_stems[0] if v_stems else cls._surface_lemma(p0)
                        for pb in (sense.get("pattern_bundles") or []):
                            if isinstance(pb, dict) and pb.get("pattern") and pb.get("example"):
                                if str(pb["example"]).strip().strip('"“”') == ex_clean:
                                    for p_var in cls._slash_variants(str(pb["pattern"])):
                                        pat_words = re.findall(r"[a-z]+", p_var.lower())
                                        if p_word in pat_words and (p0 in pat_words or (v_stem and v_stem in pat_words)):
                                            return True
                        all_s_pats = list(sense.get("patterns") or []) + [
                            pb.get("pattern") for pb in (sense.get("pattern_bundles") or [])
                            if isinstance(pb, dict) and pb.get("pattern")
                        ]
                        for pat in all_s_pats:
                            for p_var in cls._slash_variants(str(pat)):
                                pat_l = p_var.lower()
                                if p_word in re.findall(r"[a-z]+", pat_l) and (p0 in pat_l or (v_stem and v_stem in pat_l)):
                                    if cls.match_typed_slot_pattern(str(pat), ex_clean):
                                        return True
                    # Trivalent split-frame verb sense ('equate (sth) to (sth)' grounded on transitive 'equate' sense whose example uses a sister preposition 'equate wealth with success')
                    if (
                        sense
                        and isinstance(sense, dict)
                        and cls._pos_code(sense.get("pos")) == "v"
                        and len(u_key_no_cop.split()) == 2
                        and u_key_no_cop.split()[1] in {"to", "with", "into", "from", "of", "for", "on", "upon", "as", "against"}
                    ):
                        v_head = u_key_no_cop.split()[0]
                        if ex_clean in [str(e).strip().strip('"“”') for e in (sense.get("examples") or [])] and cls.text_contains_form(ex_clean, v_head):
                            return True
                norm_clean = re.sub(r"[^a-z0-9]+", " ", clean_low).strip()
                norm_ex = re.sub(r"[^a-z0-9]+", " ", e_norm).strip()
                return bool(norm_clean and norm_clean in norm_ex)
            if not cls.text_contains_form(ex_clean, clean_target):
                return False
            if check_pos and pos_low in ("noun", "verb", "adjective", "adverb"):
                ex_pos = cls.determine_contextual_pos(clean_target, ex_clean)
                if ex_pos and ex_pos in ("noun", "verb", "adjective", "adverb") and ex_pos != pos_low:
                    return False
            return True

        # 1. Check locked sense examples, units, and pattern_bundles first (Strict Sense-Isolation)
        cand_pool: List[str] = []
        if sense and isinstance(sense, dict):
            # When u_key_no_cop has a specific preposition (e.g. 'featured in', 'measured in'), promote pattern_bundles matching participle+prep first, then stem+prep
            target_prep = u_key_no_cop.split()[-1] if u_key_no_cop else ""
            part_tok = u_key_no_cop.split()[0] if u_key_no_cop else ""
            part_stems = [c for c in ([cls._surface_lemma(part_tok)] + cls._headword_candidates(part_tok)) if c and c != part_tok and cls.get_ldoce_entry(c)] if part_tok else []
            part_stem = part_stems[0] if part_stems else (cls._surface_lemma(part_tok) if part_tok else "")
            exact_pb_exs = []
            matching_pb_exs = []
            other_pb_exs = []
            for pb in (sense.get("pattern_bundles") or []):
                if isinstance(pb, dict) and pb.get("example"):
                    pb_ex = str(pb["example"]).strip()
                    pb_pat = str(pb.get("pattern") or "").lower()
                    pb_words = re.findall(r"[a-z]+", pb_pat)
                    if target_prep and target_prep in pb_words and part_tok and part_tok in pb_words:
                        exact_pb_exs.append(pb_ex)
                    elif target_prep and target_prep in pb_words and ((part_stem and part_stem in pb_words) or not is_passive_cop):
                        matching_pb_exs.append(pb_ex)
                    else:
                        other_pb_exs.append(pb_ex)
            cand_pool.extend(exact_pb_exs)
            cand_pool.extend(matching_pb_exs)
            cand_pool.extend(other_pb_exs)
            for u_obj in (sense.get("units") or []):
                if isinstance(u_obj, dict):
                    for ex in (u_obj.get("examples") or []):
                        if ex:
                            cand_pool.append(str(ex).strip())
            for ex in (sense.get("examples") or []):
                if ex:
                    cand_pool.append(str(ex).strip())
            # Also include sense-isolated full-sentence examples from cross_collocations, thesaurus, and collocations
            # when explicitly bound to this sense index or matching a sense-exclusive lexical cue from this sense
            _ent_for_sense = cls.get_ldoce_entry(clean_target) if not is_multi else None
            if _ent_for_sense:
                _s_pats_toks: Set[str] = set()
                for _p in (sense.get("patterns") or []):
                    for _t in re.findall(r"[a-z]+", str(_p).lower()):
                        if _t != clean_target and len(_t) >= 3:
                            _s_pats_toks.add(_t)
                for _pb in (sense.get("pattern_bundles") or []):
                    if isinstance(_pb, dict) and _pb.get("pattern"):
                        for _t in re.findall(r"[a-z]+", str(_pb["pattern"]).lower()):
                            if _t != clean_target and len(_t) >= 3:
                                _s_pats_toks.add(_t)
                _ent_senses = _ent_for_sense.get("senses") or []
                _locked_idx = next((i for i, _es in enumerate(_ent_senses) if _es is sense or _es.get("definition") == sense.get("definition")), -1)
                _generic_def_meta = {
                    "someone", "something", "somebody", "person", "people", "thing", "things",
                    "used", "make", "very", "much", "many", "more", "most", "good", "well",
                    "able", "produce", "have", "having", "with", "from", "that", "which",
                    "into", "about", "especially", "particular", "usually",
                }
                _s_own_text = " ".join(
                    [str(sense.get("definition") or ""), str(sense.get("signpost") or "")]
                    + [str(_ex_i) for _ex_i in (sense.get("examples") or []) if _ex_i]
                ).lower()
                _s_own_toks: Set[str] = set()
                for _w_o in re.findall(r"\b[a-z]{4,}\b", _s_own_text):
                    if _w_o != clean_target and _w_o not in _generic_def_meta and not cls.form_matches(_w_o, clean_target):
                        _s_own_toks.add(_w_o)
                        _lem_o = cls._surface_lemma(_w_o)
                        if _lem_o and len(_lem_o) >= 4 and _lem_o not in _generic_def_meta:
                            _s_own_toks.add(_lem_o)
                _other_s_toks: Set[str] = set()
                for _idx_o, _es_o in enumerate(_ent_senses):
                    if _idx_o == _locked_idx or _es_o is sense:
                        continue
                    _o_txt = " ".join(
                        [str(_es_o.get("definition") or ""), str(_es_o.get("signpost") or "")]
                        + [str(_ex_o) for _ex_o in (_es_o.get("examples") or []) if _ex_o]
                    ).lower()
                    for _w_ot in re.findall(r"\b[a-z]{4,}\b", _o_txt):
                        _other_s_toks.add(_w_ot)
                        _lem_ot = cls._surface_lemma(_w_ot)
                        if _lem_ot and len(_lem_ot) >= 4:
                            _other_s_toks.add(_lem_ot)
                _allowed_sense_toks = _s_pats_toks | (_s_own_toks - _other_s_toks)

                def _colloc_matches_locked_sense(_c_str: str) -> bool:
                    if len(_ent_senses) <= 1 or all(not s.get("examples") for s in _ent_senses):
                        return True
                    for _ct in re.findall(r"\b[a-z]{3,}\b", (_c_str or "").lower()):
                        if _ct == clean_target or cls.form_matches(_ct, clean_target):
                            continue
                        if _ct in _allowed_sense_toks:
                            return True
                        _ct_lem = cls._surface_lemma(_ct)
                        if _ct_lem and _ct_lem in _allowed_sense_toks:
                            return True
                    return False

                for _xc in (_ent_for_sense.get("cross_collocations") or []):
                    if isinstance(_xc, dict) and _xc.get("example"):
                        if _colloc_matches_locked_sense(str(_xc.get("collocation") or "")):
                            cand_pool.append(str(_xc["example"]).strip())

                for _th in (_ent_for_sense.get("thesaurus") or []):
                    if isinstance(_th, dict) and cls._thesaurus_variant_matches(_th.get("word"), clean_target):
                        if len(_ent_senses) <= 1 or all(not s.get("examples") for s in _ent_senses):
                            for _th_ex in (_th.get("examples") or ([_th["example"]] if _th.get("example") else [])):
                                if _th_ex:
                                    cand_pool.append(str(_th_ex).strip())

                if isinstance(_ent_for_sense.get("collocations"), dict):
                    for _c_list in _ent_for_sense["collocations"].values():
                        if not isinstance(_c_list, list):
                            continue
                        for _cc in _c_list:
                            if not isinstance(_cc, dict) or not _cc.get("example"):
                                continue
                            _cc_s_idxs = _cc.get("sense_indices")
                            _cc_word = str(_cc.get("collocation") or "").strip().lower()
                            if (isinstance(_cc_s_idxs, list) and _locked_idx >= 0 and _locked_idx in _cc_s_idxs) or (
                                (len(_ent_senses) <= 1 or all(not s.get("examples") for s in _ent_senses)) and _cc_word and _colloc_matches_locked_sense(_cc_word)
                            ):
                                cand_pool.append(str(_cc["example"]).strip())

            def _is_full_sentence_example(ex_s: Optional[str]) -> bool:
                if not ex_s:
                    return False
                s_clean = str(ex_s).strip().strip('"“”')
                w_cnt = len(s_clean.split())
                return w_cnt >= 4 and s_clean.endswith((".", "!", "?"))

            fallback_short_ex: Optional[str] = None
            # If multi-word expression has an extended preposition pattern attested in quoted_sentence (e.g. 'get by on'),
            # prefer an example from the locked sense that demonstrates that exact extended pattern.
            if is_multi and q_norm and cand_pool:
                for pat in (sense.get("patterns") or []):
                    for p_var in cls._slash_variants(str(pat)):
                        p_k = cls._unit_key(p_var)
                        if p_k and len(p_k.split()) > len(u_key.split()) and cls._unit_example(p_k, [quoted_sentence], allow_modifier=True):
                            for cand_ex in cand_pool:
                                if _is_valid_distinct(cand_ex, check_pos=False) and cls._unit_example(p_k, [cand_ex], allow_modifier=True):
                                    c_ex_str = cand_ex.strip().strip('"“”')
                                    if _is_full_sentence_example(c_ex_str):
                                        return c_ex_str
                                    if not fallback_short_ex:
                                        fallback_short_ex = c_ex_str
            for exact_ex in exact_pb_exs:
                if _is_valid_distinct(exact_ex, check_pos=False):
                    c_ex_str = exact_ex.strip().strip('"“”')
                    if _is_full_sentence_example(c_ex_str):
                        return c_ex_str
                    if not fallback_short_ex:
                        fallback_short_ex = c_ex_str
            # For copular/participial expressions ('be headed for [sth]'), prioritize examples from the locked sense
            # that contain the exact inflected surface phrase ('headed for') ahead of lemmatized active variants ('heading for').
            if u_key_no_cop:
                for surf_ex in ([current_example] if current_example else []) + cand_pool:
                    if surf_ex and _is_valid_distinct(surf_ex, check_pos=False) and re.search(rf"\b{re.escape(u_key_no_cop)}\b", str(surf_ex), re.I):
                        c_ex_str = str(surf_ex).strip().strip('"“”')
                        if _is_full_sentence_example(c_ex_str):
                            return c_ex_str
                        if not fallback_short_ex:
                            fallback_short_ex = c_ex_str
                sense_unit_strs = [
                    (str(u.get("phrase") or u.get("unit") or "") if isinstance(u, dict) else str(u)).lower()
                    for u in ((sense.get("units") or []) if isinstance(sense, dict) else [])
                ]
                u_cop_prefix = " ".join(u_key.split()[:2])
                if any(u_cop_prefix in su for su in sense_unit_strs):
                    for corp_ex in cls.search_corpus_examples(f"be {u_key_no_cop}", limit=10):
                        if (
                            _is_valid_distinct(corp_ex, check_pos=False)
                            and re.search(rf"\b(?:be|am|is|are|was|were|been|being)\s+(?:[a-z]+\s+)?{re.escape(u_key_no_cop)}\b", corp_ex, re.I)
                        ):
                            return corp_ex.strip().strip('"“”')
            for match_ex in matching_pb_exs:
                if _is_valid_distinct(match_ex, check_pos=False):
                    c_ex_str = match_ex.strip().strip('"“”')
                    if _is_full_sentence_example(c_ex_str):
                        return c_ex_str
                    if not fallback_short_ex:
                        fallback_short_ex = c_ex_str
        else:
            def _is_full_sentence_example(ex_s: Optional[str]) -> bool:
                if not ex_s:
                    return False
                s_clean = str(ex_s).strip().strip('"“”')
                w_cnt = len(s_clean.split())
                return w_cnt >= 4 and s_clean.endswith((".", "!", "?"))

            fallback_short_ex = None

        if _is_valid_distinct(current_example, check_pos=(sense is None)):
            c_ex_str = str(current_example).strip().strip('"“”')
            if _is_full_sentence_example(c_ex_str):
                return c_ex_str
            if not fallback_short_ex:
                fallback_short_ex = c_ex_str

        if cand_pool:
            for cand_ex in cand_pool:
                if _is_valid_distinct(cand_ex, check_pos=False):
                    c_ex_str = cand_ex.strip().strip('"“”')
                    if _is_full_sentence_example(c_ex_str):
                        return c_ex_str
                    if not fallback_short_ex:
                        fallback_short_ex = c_ex_str

        # Determine whether the entry (or multi-word unit) is monosemous under the target POS
        entry = cls.get_ldoce_entry(clean_target)
        target_p = cls._pos_code(pos_low)
        entry_senses = (entry.get("senses") or []) if entry else []
        pos_senses = [s for s in entry_senses if not target_p or cls._sense_matches_pos(s, target_p)]
        is_monosemous = len(pos_senses) <= 1

        u_blocks = cls._unit_blocks(u_key) if is_multi else []
        u_rows = cls._unit_phrase_rows(u_key) if is_multi else []
        has_dedicated_unit = bool(u_rows or u_blocks or (entry and len(entry_senses) == 1))
        has_poly_block = any(len(cls._unit_senses(b, u_key)) > 1 for b in u_blocks)
        is_monosemous_multi = is_multi and has_dedicated_unit and not has_poly_block and len(u_rows) <= 1
        # Check if a copular/prepositional multi-word expression ('be exhausted from', 'be smart about')
        # is grounded in a host entry where only 1 sense owns that headword + preposition pattern.
        if is_multi and not is_monosemous_multi and sense and isinstance(sense, dict):
            uk_check = u_key_no_cop or u_key
            uk_toks = uk_check.split()
            if len(uk_toks) == 2:
                h_word, p_word = uk_toks[0], uk_toks[1]
                for h_cand in (h_word, cls._surface_lemma(h_word)):
                    if not h_cand:
                        continue
                    h_ent = cls.get_ldoce_entry(h_cand)
                    if not h_ent:
                        continue
                    matching_pat_senses = []
                    for hs in (h_ent.get("senses") or []):
                        hs_pats = []
                        for hp in (hs.get("patterns") or []):
                            hs_pats.extend(v.lower() for v in cls._slash_variants(str(hp)))
                        for hpb in (hs.get("pattern_bundles") or []):
                            if isinstance(hpb, dict) and hpb.get("pattern"):
                                hs_pats.extend(v.lower() for v in cls._slash_variants(str(hpb["pattern"])))
                        if any(p_word in re.findall(r"[a-z]+", p_str) for p_str in hs_pats):
                            matching_pat_senses.append(hs)
                    if len(matching_pat_senses) == 1 and matching_pat_senses[0].get("definition") == sense.get("definition"):
                        is_monosemous_multi = True
                        break

        # If sense was locked to a WordNet fallback sense while PedLex already has senses for this POS,
        # do NOT steal examples from PedLex's different senses.
        if sense and isinstance(sense, dict):
            if is_multi:
                if not is_monosemous_multi:
                    return fallback_short_ex or ""
            else:
                s_def_norm = re.sub(r"[^a-z0-9]+", " ", str(sense.get("definition") or "").lower()).strip()
                in_pedlex = any(
                    ps is sense
                    or re.sub(r"[^a-z0-9]+", " ", str(cls.format_ldoce_definition(ps) or "").lower()).strip() == s_def_norm
                    or re.sub(r"[^a-z0-9]+", " ", str(ps.get("definition") or "").lower()).strip() == s_def_norm
                    for ps in pos_senses
                )
                if pos_senses and not in_pedlex:
                    return fallback_short_ex or ""
                all_senses_empty_examples = bool(pos_senses and all(not s.get("examples") for s in pos_senses))
                if not is_monosemous and not all_senses_empty_examples:
                    # Polysemous word whose locked sense has 0 full-sentence examples: do NOT steal from another sense or corpus
                    return fallback_short_ex or ""

        # 2. Monosemous or unlocked entry-level & cross-entry corpus examples
        all_senses_empty_examples = bool(pos_senses and all(not s.get("examples") for s in pos_senses))
        if entry and (is_monosemous or all_senses_empty_examples):
            for s in pos_senses:
                for pb in (s.get("pattern_bundles") or []):
                    if isinstance(pb, dict) and pb.get("example"):
                        cand_pool.append(str(pb["example"]).strip())
                for ex in (s.get("examples") or []):
                    if ex:
                        cand_pool.append(str(ex).strip())
            for cc in (entry.get("cross_collocations") or []):
                if isinstance(cc, dict) and cc.get("example"):
                    cand_pool.append(str(cc["example"]).strip())
            if isinstance(entry.get("collocations"), dict):
                for _c_list in entry["collocations"].values():
                    if isinstance(_c_list, list):
                        for cc in _c_list:
                            if isinstance(cc, dict) and cc.get("example"):
                                cand_pool.append(str(cc["example"]).strip())

        # 3. For multiword units, check _unit_phrase_rows and _unit_blocks when not polysemously conflicting
        if is_multi and (not sense or is_monosemous_multi):
            for _h, row in u_rows:
                for ex in (row.get("examples") or []):
                    if ex:
                        cand_pool.append(str(ex).strip())
            for block in u_blocks:
                u_senses = cls._unit_senses(block, u_key)
                if not sense or len(u_senses) <= 1:
                    for s in u_senses:
                        for ex in (s.get("examples") or []):
                            if ex:
                                cand_pool.append(str(ex).strip())

        for cand_ex in cand_pool:
            if _is_valid_distinct(cand_ex):
                c_ex_str = cand_ex.strip().strip('"“”')
                if _is_full_sentence_example(c_ex_str):
                    return c_ex_str
                if not fallback_short_ex:
                    fallback_short_ex = c_ex_str

        # 4. Search corpus examples in PedLex SQLite (only for monosemous words or monosemous dedicated multi-word units)
        if (not is_multi and (is_monosemous or all_senses_empty_examples)) or (is_multi and (is_monosemous_multi or (not sense and not has_poly_block))):
            try:
                search_queries = [u_key] if is_multi else [clean_target]
                if is_multi and u_key_no_cop and u_key_no_cop not in search_queries:
                    search_queries.append(u_key_no_cop)
                for sq in search_queries:
                    for ex in (cls.search_corpus_examples(sq, limit=15) or []):
                        if ex and _is_valid_distinct(str(ex).strip()):
                            c_ex_str = str(ex).strip().strip('"“”')
                            if _is_full_sentence_example(c_ex_str):
                                return c_ex_str
                            if not fallback_short_ex:
                                fallback_short_ex = c_ex_str
            except Exception:
                pass

        # 5. Check WordNet synset examples ONLY when PedLex is disabled and monosemous in WordNet
        allow_wn_examples = bool(cls._is_pedlex_disabled() and (not entry or not pos_senses))
        if not is_multi and is_monosemous and allow_wn_examples:
            try:
                wn = cls.get_wordnet()
                if wn:
                    wn_pos_map = {"noun": ("n",), "verb": ("v",), "adjective": ("a", "s"), "adverb": ("r",)}
                    wn_poses = wn_pos_map.get(pos_low)
                    synsets = [s for p_code in wn_poses for s in (wn.synsets(clean_target.replace(" ", "_"), pos=p_code) or [])] if wn_poses else wn.synsets(clean_target.replace(" ", "_"))
                    if len(synsets) <= 1:
                        for syn in synsets:
                            for ex in (syn.examples() or []):
                                if not ex:
                                    continue
                                ex_s = re.sub(r"\s*\((?:or|e\.g\.)\s+[^)]+\)", "", str(ex)).strip()
                                if ex_s and _is_valid_distinct(ex_s):
                                    c_ex_str = ex_s.strip().strip('"“”')
                                    if _is_full_sentence_example(c_ex_str):
                                        return c_ex_str
                                    if not fallback_short_ex:
                                        fallback_short_ex = c_ex_str
            except Exception:
                pass
        elif cls._is_pedlex_disabled() and is_multi and is_monosemous_multi and u_key:
            try:
                wn = cls.get_wordnet()
                if wn:
                    synsets_m = list(wn.synsets(u_key.replace(" ", "_")) or [])
                    for syn in synsets_m:
                        for ex in (syn.examples() or []):
                            if not ex:
                                continue
                            ex_s = re.sub(r"\s*\((?:or|e\.g\.)\s+[^)]+\)", "", str(ex)).strip()
                            if ex_s and _is_valid_distinct(ex_s):
                                c_ex_str = ex_s.strip().strip('"“”')
                                if _is_full_sentence_example(c_ex_str):
                                    return c_ex_str
                                if not fallback_short_ex:
                                    fallback_short_ex = c_ex_str
            except Exception:
                pass

        return fallback_short_ex or ""

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
                    w, matching_sent, pos=pos, k=6, discourse_context=discourse_ctx
                ) if matching_sent else []
                if cand_pool:
                    chosen = None
                    if len(cand_pool) > 1 and enable_llm_disambiguation:
                        chosen_idx = cls.disambiguate_via_llm(w, matching_sent, cand_pool, discourse_context=discourse_ctx)
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
    # DECLARATIVE 3-BAND COBUILD GRAMMAR PROFILES & HOT-RELOAD LOADER
    # Grounded in librarian/data/grammar_profiles.json (A1_A2 / B1_B2 / C1_C2)
