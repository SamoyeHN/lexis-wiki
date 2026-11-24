"""
Authentic cloze assessment item synthesis and lexical definition support.
"""

import logging
import re
import zlib
from typing import Any, Dict, List, Optional, Set, Tuple


class QuizVocabClozeMixin:
    @classmethod
    def _balance_single_and_multi_word(
        cls,
        items: List[Dict[str, Any]],
        target_count: int = 10,
        multi_word_ratio: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Balances single-word and multi-word expression items (e.g. 100% single words by default for vocabulary quiz)
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

        # If no single words exist, return multis up to target_count
        if not singles:
            return multis[:target_count]

        # Calculate quotas (if multi_word_ratio <= 0.0 or no multis, prioritize single words)
        quota_multi = 0 if (multi_word_ratio <= 0.0 or not multis) else min(len(multis), max(1, int(round(target_count * multi_word_ratio))))
        quota_single = target_count - quota_multi

        sel_multi = multis[:quota_multi] if quota_multi > 0 else []
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
            if mechs and all(str(m or "").endswith("fallback") for m in mechs):
                return 4
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
        # Mutual Collocation Partner Eviction:
        # When an item with a concrete collocational partner (e.g. 'closely' with anchor 'follow')
        # is selected, evict the partner from remaining candidates to avoid redundant testing,
        # effectively testing both words within one item via anchor absorption.
        sel_single: List[Dict[str, Any]] = []
        selected_anchors: Set[str] = set()
        selected_targets: Set[str] = set()
        evicted_singles: List[Dict[str, Any]] = []

        for s_it in ordered_singles:
            sw_raw = str(s_it.get("target_word") or s_it.get("word") or "").strip().lower()
            sw_lem = cls.lemma_of(sw_raw) if sw_raw else ""
            anc_raw = str(s_it.get("context_anchor") or s_it.get("anchor") or "").strip().lower()
            anc_lem = cls.lemma_of(anc_raw) if anc_raw else ""

            if sw_lem in selected_anchors or (anc_lem and anc_lem in selected_targets):
                evicted_singles.append(s_it)
                continue

            sel_single.append(s_it)
            if sw_lem:
                selected_targets.add(sw_lem)
            if anc_lem and len(anc_lem) >= 3 and anc_lem not in cls._ANCHOR_STOPWORDS:
                selected_anchors.add(anc_lem)

            if len(sel_single) >= quota_single:
                break

        # Graceful backfill from evicted singles if not enough non-colliding singles exist
        if len(sel_single) < quota_single:
            deficit = quota_single - len(sel_single)
            sel_single.extend(evicted_singles[:deficit])

        # Backfill if one group has fewer items than its allocated quota
        if len(sel_multi) < quota_multi:
            deficit = quota_multi - len(sel_multi)
            rem_singles = [s for s in ordered_singles if s not in sel_single]
            sel_single.extend(rem_singles[:deficit])
        elif len(sel_single) < quota_single and len(sel_multi) < len(multis):
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
        used_stem_signatures: Set[str] = set()

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
            target_tok = None
            half_slice_explanation = None
            is_half_sliced = False

            # Collocation Half-Slice for Multi-word Expressions
            if is_multi_word:
                hs = cls.collocation_half_slice(w, q_clean, pos=it.get("part_of_speech") or "")
                if hs:
                    surface_form = hs["target_slot"]
                    stem = cls.isolate_cloze_clause(hs["stem"])
                    distractors = list(hs["distractors"])
                    half_slice_explanation = hs.get("explanation")
                    is_half_sliced = True
                    matched_start = 0
                    matched_end = 0

            if not is_half_sliced:
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

                if matched_start is None:
                    # Fallback for hyphenated or compound words where spaCy tokenizes punctuation into separate tokens
                    m_surf = re.search(rf"(?<!\w){re.escape(w_lower)}(?!\w)", q_clean, re.IGNORECASE)
                    if not m_surf and "-" in w_lower:
                        w_no_hyphen = w_lower.replace("-", "")
                        w_space = w_lower.replace("-", " ")
                        m_surf = re.search(rf"(?<!\w)(?:{re.escape(w_no_hyphen)}|{re.escape(w_space)})(?!\w)", q_clean, re.IGNORECASE)
                    if m_surf:
                        matched_start = m_surf.start()
                        matched_end = m_surf.end()
                        surface_form = q_clean[matched_start:matched_end]
                        for t in doc:
                            if t.idx >= matched_start:
                                target_tok = t
                                break

                if matched_start is None or matched_end is None or not surface_form:
                    continue

                # Replace target span with exact 4 underscores '____'
                stem = q_clean[:matched_start] + "____" + q_clean[matched_end:]

                # Target Lexical Leakage Gate: ensure target word does not leak into stem outside blank
                stem_no_blank = re.sub(r'_{2,}', ' ', stem)
                stem_words = [tok.lower() for tok in re.findall(r'\b[a-zA-Z]+\b', stem_no_blank)]
                target_lower = w_lower
                target_root = re.sub(r'(?:ed|ing|s|es|ly|tion|ment)$', '', target_lower)
                has_leak = any(
                    tw == target_lower
                    or (len(target_root) >= 4 and re.sub(r'(?:ed|ing|s|es|ly|tion|ment)$', '', tw) == target_root)
                    or (" " not in w and len(tw) >= 4 and cls.shares_derivational_root(tw, target_lower))
                    for tw in stem_words
                )
                if has_leak:
                    # Attempt to isolate clause containing the blank (e.g. across compound coordinate clauses)
                    clauses = re.split(r'(?<=,)\s+(?=and\b|but\b|or\b|so\b)|(?<=;)\s+', stem)
                    isolated_clause = None
                    for cl in clauses:
                        if "____" in cl:
                            cl_no_blank = re.sub(r'_{2,}', ' ', cl)
                            cl_words = [tok.lower() for tok in re.findall(r'\b[a-zA-Z]+\b', cl_no_blank)]
                            if not any(
                                tw == target_lower
                                or (len(target_root) >= 4 and re.sub(r'(?:ed|ing|s|es|ly|tion|ment)$', '', tw) == target_root)
                                or (" " not in w and len(tw) >= 4 and cls.shares_derivational_root(tw, target_lower))
                                for tw in cl_words
                            ):
                                cl_clean = cl.strip().rstrip(",").strip()
                                if len(cl_clean.split()) >= 6:
                                    if not cl_clean.endswith((".", "!", "?")):
                                        cl_clean += "."
                                    isolated_clause = cl_clean
                                    break
                    if isolated_clause:
                        stem = isolated_clause
                    else:
                        continue

                # Isolate clause to create tight, self-contained single-sentence cloze stem
                iso_stem = cls.isolate_cloze_clause(stem)
                if iso_stem and len(re.findall(r'_{2,}', iso_stem)) == 1:
                    stem = iso_stem

            # Ensure exactly one blank exists in stem
            if len(re.findall(r'_{2,}', stem)) != 1:
                continue

            # Ensure stem signature uniqueness across all items in this quiz
            stem_sig = re.sub(r'[^a-z0-9]', '', stem.lower())
            if stem_sig in used_stem_signatures:
                continue
            used_stem_signatures.add(stem_sig)

            # Distractor generation (Sense-Locked WSD & Authentic Phrase Distractors)
            if not is_half_sliced:
                if is_multi_word:
                    distractors = cls.generate_phrase_distractors(
                        phrase=w_lower,
                        pos=it.get("part_of_speech") or "",
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
                            count=6,
                            context_sentence=q_clean
                        )
                        for fd in fallback_d:
                            if fd not in distractors and cls.distractor_occupies_slot(fd, canonical_pos):
                                distractors.append(fd)
                            if len(distractors) >= 3:
                                break
                    distractors = [d for d in distractors if cls.distractor_occupies_slot(d, canonical_pos)]
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
                        if not verb_tag and distractors:
                            base_opts = [cls.lemma_of(surface_form) or surface_form] + [
                                cls.lemma_of(d) or d for d in distractors[:3]
                            ]
                            _, reconciled_opts, _, _ = cls._reconcile_prescribed_inflection(
                                base_options=base_opts,
                                target_index=0,
                                target_word=surface_form,
                                canonical_pos="verb",
                                inflection_desc="base form",
                                authentic_example=q_clean,
                            )
                            target_lemma = (cls.lemma_of(surface_form) or surface_form).lower()
                            distractors = [
                                o for i, o in enumerate(reconciled_opts)
                                if i != 0
                                and o.lower() != surface_form.lower()
                                and (cls.lemma_of(o) or o).lower() != target_lemma
                            ][:3]
                    elif pos_tag == "n" and target_tok is not None and target_tok.tag_ == "NNS":
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
                "is_multi_word": is_multi_word,
                "half_slice_explanation": half_slice_explanation,
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
                target_lemma = (cls.lemma_of(target_w) or target_w).lower()
                seen_d = set()
                dist = []
                for d in it.get("precomputed_distractors", []):
                    if not d:
                        continue
                    d_clean = d.strip()
                    d_lower = d_clean.lower()
                    if d_lower != target_w.lower() and (cls.lemma_of(d_clean) or d_clean).lower() != target_lemma:
                        if d_lower not in seen_d:
                            seen_d.add(d_lower)
                            dist.append(d_clean)
                if len(dist) < 3:
                    fallbacks = cls.generate_zero_collision_distractors(
                        target_word=it.get("base_headword") or target_w,
                        pos=cls._pos_code(it.get("part_of_speech", "")),
                        count=10,
                        context_sentence=it.get("source_sentence", "")
                    )
                    for fb in fallbacks:
                        fb_clean = fb.strip()
                        fb_lower = fb_clean.lower()
                        if fb_lower not in seen_d and fb_lower != target_w.lower() and (cls.lemma_of(fb_clean) or fb_clean).lower() != target_lemma:
                            seen_d.add(fb_lower)
                            dist.append(fb_clean)
                        if len(dist) >= 3:
                            break
                slot = balanced_slots[idx]
                opts = dist[:slot] + [target_w] + dist[slot:]
                it["options"] = opts
                it["precomputed_options"] = opts
                it["correct_answer_index"] = slot
                if it.get("half_slice_explanation"):
                    it["explanation"] = it["half_slice_explanation"]
                else:
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
                if vc.get("pos") and cls._pos_code(vc["pos"]) != "n":
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
                    m_prep = re.search(rf"\b{re.escape(t_clean)}\b\s+([a-z]+)\b", chosen_ex, re.IGNORECASE)
                    found_prep = m_prep.group(1).lower() if m_prep else None
                    if found_prep not in {"of", "for", "on", "in", "to", "with", "from", "into", "against", "about", "towards", "toward", "over", "under", "between"}:
                        found_prep = None
                    out_bundles.append({
                        "pattern": pat_str,
                        "example": chosen_ex,
                        "prep": found_prep,
                        "preps": [found_prep] if found_prep else [],
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
                    m_prep = re.search(rf"\b{re.escape(t_clean)}\b\s+([a-z]+)\b", ex, re.IGNORECASE)
                    found_prep = m_prep.group(1).lower() if m_prep else None
                    if found_prep not in {"of", "for", "on", "in", "to", "with", "from", "into", "against", "about", "towards", "toward", "over", "under", "between"}:
                        found_prep = None
                    out_bundles.append({
                        "pattern": pat_str,
                        "example": ex,
                        "prep": found_prep,
                        "preps": [found_prep] if found_prep else [],
                        "verb_anchor": v_norm,
                        "from_quote_verb": bool(re.search(rf"\b{re.escape(v_norm)}\w*\b", q_low)),
                        "from_sense_example": True,
                        "source": "sense_verb_colloc",
                    })

        return out_bundles
