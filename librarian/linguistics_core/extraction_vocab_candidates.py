"""
Deterministic academic vocabulary candidate mining and Top-K candidate pool generation.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("librarian.linguistics")


class ExtractionVocabCandidatesMixin:
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



