"""
Deterministic multi-word expression and collocation mining, candidate filtering, and extraction.
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


class ExtractionExpressionsMixin:
    # -------------------------------------------------------------------------
    # 3b. Deterministic Multi-Word Expression & Collocation Mining (ACL + spaCy)
    # -------------------------------------------------------------------------
    @classmethod
    def mine_expression_skeletons(
        cls,
        text: str,
        target_count: int = 8,
        syllabus_expressions: Optional[List[str]] = None
    ) -> List[Dict[str, str]]:
        """
        Deterministically extracts high-value multi-word expressions, phrasal verbs,
        and Academic Collocation List (ACL) items from the indexed sentence pool.
        Standardizes slotted formulas (e.g. [sb], [sth], [one's]) at zero token cost.
        If syllabus_expressions is provided, prioritizes resolving and standardizing
        those curriculum targets first before filling remaining slots with computational candidates.

        Returns list of dicts:
            [{
                "sid": "S-1",
                "quote": "full sentence text",
                "phrase": "have an adverse effect on",
                "type": "collocation",
                "pattern_formula": "have an adverse effect on [sth]"
            }, ...]
        """
        _, pool = cls.tokenize_and_index_sentences(text)
        if not pool:
            return []

        nlp = cls.get_spacy()
        from librarian.expression_patterns import ExpressionPatternEngine
        expr_engine = ExpressionPatternEngine.get_engine(nlp)

        COMMON_PARTICLES = frozenset({"away", "back", "down", "in", "off", "on", "out", "over", "round", "through", "up"})
        DEPENDENT_PREPS = frozenset({"on", "upon", "to", "with", "for", "from", "into", "against", "about", "of", "in"})

        raw_candidates: List[Dict[str, Any]] = []

        for sid, sentence in pool.items():
            sent_clean = sentence.strip()
            if len(sent_clean.split()) < 5:
                continue

            doc = nlp(sent_clean)
            text_lower = sent_clean.lower()

            # 1. Dependency Syntax Phrasal Verbs & Prepositional Combinations via Declarative Expression Pattern Engine
            for match in expr_engine.match_sentence(doc):
                m_type = match["type"]
                if m_type == "phrasal verb":
                    m_words = match["phrase"].lower().split()
                    if len(m_words) == 2 and m_words[1] not in COMMON_PARTICLES and not cls._unit_blocks(match["phrase"].lower()):
                        v_lem_0, p_w_1 = m_words[0], m_words[1]
                        v_ent_0 = cls.get_ldoce_entry(v_lem_0) or {}
                        v_verb_senses_0 = [
                            s for s in (v_ent_0.get("senses") or [])
                            if cls._pos_code(s.get("pos")) == "v" or not s.get("pos")
                        ]
                        primary_governs_prep = False
                        if p_w_1 in DEPENDENT_PREPS and v_verb_senses_0:
                            s0_v = v_verb_senses_0[0]
                            s0_pats = list(s0_v.get("patterns") or []) + [
                                pb.get("pattern") for pb in (s0_v.get("pattern_bundles") or [])
                                if isinstance(pb, dict) and pb.get("pattern")
                            ]
                            if len(v_verb_senses_0) == 1 or any(
                                p_w_1 in re.findall(r"[a-z]+", str(p).lower()) for p in s0_pats
                            ):
                                primary_governs_prep = True
                        if not primary_governs_prep and not cls._phrasal_verb_blocks(match["phrase"].lower(), v_lem_0):
                            m_type = "collocation"
                raw_candidates.append({
                    "sid": sid,
                    "quote": sent_clean,
                    "phrase": match["phrase"],
                    "type": m_type,
                    "pattern_formula": match["pattern_formula"],
                    "pobj": match.get("pobj"),
                    "pobj_proper": match.get("pobj_proper", False),
                    "score": match.get("priority", 8),
                })

            # 2. PedLex Sense-Unit Multi-Word Expression Scan (3..5 tokens)
            # Recovers attested idioms, light-verb phrases, and complex prepositions ('turn a blind eye',
            # 'make a living', 'take advantage of', 'for the sake of', 'a roof over one's head')
            # directly from in-memory ldoce_phrase_index ('sense_unit' tier) in O(1) per n-gram.
            if cls._ensure_in_memory_phrase_index() and cls._phrase_by_key_cache:
                by_key = cls._phrase_by_key_cache
                w_toks = [t for t in doc if not t.is_punct and not t.is_space]
                existing_phrases_in_sent = {
                    c["phrase"].lower() for c in raw_candidates if c.get("sid") == sid
                }
                covered_indices: Set[int] = set()
                _poss_dets = {"my", "your", "his", "her", "its", "our", "their"}
                for n_len in (5, 4, 3):
                    if len(w_toks) < n_len:
                        continue
                    for start_i in range(len(w_toks) - n_len + 1):
                        span_indices = set(range(start_i, start_i + n_len))
                        if span_indices & covered_indices:
                            continue
                        span_t = w_toks[start_i : start_i + n_len]
                        # Must not cross internal punctuation boundaries
                        if any(
                            dt.is_punct and dt.text not in ("-", "'", "\u2019")
                            for dt in doc[span_t[0].i : span_t[-1].i + 1]
                        ):
                            continue
                        # Must contain at least one open-class content word
                        if not any(t.pos_ in ("NOUN", "VERB", "ADJ") for t in span_t):
                            continue
                        # Must not start or end with a determiner/pronoun unless start is 'a'/'an'/'the' in a noun idiom
                        if span_t[-1].pos_ in ("DET", "PRON", "AUX", "CCONJ", "SCONJ"):
                            continue
                        # Must not contain numbers, digits, or numerical tokens
                        if any(t.pos_ == "NUM" or t.like_num or re.search(r"\d", t.text) for t in span_t):
                            continue
                        surf_toks = [t.text.lower() for t in span_t]
                        lem0 = (
                            span_t[0].lemma_.lower()
                            if span_t[0].pos_ in ("VERB", "AUX")
                            else surf_toks[0]
                        )
                        base_toks = [lem0] + surf_toks[1:]
                        # Check for internal possessive determiner ('a roof over their heads' -> 'a roof over your head')
                        has_internal_poss = any(t in _poss_dets for t in base_toks[1:-1])
                        cand_lookup_keys: List[Tuple[str, str, str]] = []
                        b_key = " ".join(base_toks)
                        cand_lookup_keys.append((b_key, b_key, b_key))
                        if has_internal_poss:
                            your_toks = [
                                "your" if (0 < idx_k < n_len - 1 and tk in _poss_dets) else tk
                                for idx_k, tk in enumerate(base_toks)
                            ]
                            ones_toks = [
                                "[one's]" if (0 < idx_k < n_len - 1 and tk in _poss_dets) else tk
                                for idx_k, tk in enumerate(base_toks)
                            ]
                            last_lem = span_t[-1].lemma_.lower()
                            your_sing = your_toks[:-1] + [last_lem]
                            ones_sing = ones_toks[:-1] + [last_lem]
                            cand_lookup_keys.insert(
                                0,
                                (
                                    " ".join(your_sing),
                                    " ".join(your_sing).replace("your", "one's"),
                                    " ".join(ones_sing),
                                ),
                            )
                        for l_key, p_name, p_formula in cand_lookup_keys:
                            rows = by_key.get(l_key)
                            if not rows:
                                continue
                            su_rows = [r for r in rows if r[2] == "sense_unit"]
                            if not su_rows:
                                continue
                            # Avoid matching bare 'turn a blind eye to' when 'turn a blind eye' is the core idiom
                            if n_len == 5 and base_toks[-1] == "to" and by_key.get(" ".join(base_toks[:-1])):
                                continue
                            # Guard B: 'for a change' is only the discourse idiom when 'change' is NOT followed by a post-nominal PP ('in'/'of'/'to')
                            if l_key == "for a change" and start_i + n_len < len(w_toks) and w_toks[start_i + n_len].text.lower() in ("in", "of", "to"):
                                continue
                            # Guard A: 'be made up' followed by 'of' is the compositional phrasal verb 'be made up of', not '[spoken informal] be made up'
                            extra_cov_indices = []
                            if l_key == "be made up" and start_i + n_len < len(w_toks) and w_toks[start_i + n_len].text.lower() == "of":
                                p_name = "be made up of"
                                p_formula = "be made up of [sth]"
                                extra_cov_indices = [start_i + n_len]
                            if p_name in existing_phrases_in_sent:
                                covered_indices.update(span_indices)
                                covered_indices.update(extra_cov_indices)
                                break
                            # Format trailing preposition object slot if the phrase ends with a preposition
                            final_formula = p_formula
                            if span_t[-1].pos_ == "ADP" and not final_formula.endswith("]"):
                                raw_u = str(su_rows[0][3] or "").lower()
                                if "somebody/something" in raw_u or "something/somebody" in raw_u:
                                    final_formula = f"{p_formula} [sth/sb]"
                                elif "somebody" in raw_u:
                                    final_formula = f"{p_formula} [sb]"
                                else:
                                    final_formula = f"{p_formula} [sth]"
                            if not cls.is_pedagogical_expression(p_name) or not cls.is_pedagogical_expression(final_formula):
                                continue
                            m_type_su = cls.classify_expression_type(p_name, sent_clean) or "set phrase"
                            raw_candidates.append({
                                "sid": sid,
                                "quote": sent_clean,
                                "phrase": p_name,
                                "type": m_type_su,
                                "pattern_formula": final_formula,
                                "pobj": None,
                                "pobj_proper": False,
                                "score": 92,
                            })
                            existing_phrases_in_sent.add(p_name)
                            covered_indices.update(span_indices)
                            covered_indices.update(extra_cov_indices)
                            break


        # PedLex attestation gate: a candidate ships only when PedLex itself states the unit.
        # The old gate asked the Oxford Collocations file whether it had a headword for the
        # phrase, which is a weaker test than it sounds - 'keep of' is a preposition lifted out
        # of 'keep somebody out of something' and no dictionary has it as a unit.
        # ldoce_phrase_evidence asks the real question: is this a phrasal-verb block, a PHRASES
        # item, a grammar pattern built on the headword, or a COLLOCATIONS box item?  Evidence
        # that amounts to 'some example sentence contains these words in this order' does not
        # ship, because an example is exactly where a stray preposition looks most like a phrase.
        verified_candidates: List[Dict[str, Any]] = []

        for item in raw_candidates:
            phrase = item.get("phrase", "").lower().strip()
            item_formula = item.get("pattern_formula", phrase)
            if not cls.is_pedagogical_expression(phrase) or not cls.is_pedagogical_expression(item_formula):
                continue
            item_type = item.get("type")
            words = phrase.split()
            if not words or phrase in ("a lot of", "lots of"):
                continue

            # Phrasal verbs and prepositional verbs (e.g. 'rely on', 'long for', 'worry about')
            if item_type == "phrasal verb":
                attested = cls.is_attested_phrase(phrase)
                if not attested and "touch" in words and any(v in words for v in ("keep", "get", "stay", "lose")):
                    # PedLex states it as 'be/keep/stay etc in touch (with something)'
                    attested = cls.is_attested_phrase(" ".join(w for w in words if w != "with"))
                if attested:
                    # 'keep in' is a real phrasal verb, but PedLex's frame for it is 'keep
                    # somebody in', so a parse of 'keep in WeChat Moments' - or of 'keep in
                    # touch' - has read a locative 'in ...' as the particle and is not that
                    # frame.  The check runs for any object, not only a proper noun, because
                    # the frame says where the object sits regardless of what it is.
                    pobj_word = item.get("pobj")
                    if pobj_word and not cls.ldoce_phrase_object_fit(phrase, pobj_word):
                        continue
                    verified_candidates.append(item)
                continue

            # Anything else the dictionary states as a unit (e.g. 'account for', 'participate in')
            if cls.is_attested_phrase(phrase):
                verified_candidates.append(item)
                continue

            # Verbal idioms (e.g. 'take into consideration') that PedLex does not list as a
            # PHRASES item: the head noun's own grammar patterns must show the verb governing it.
            if item_type == "idiom":
                v_lemma, head_noun = words[0], words[-1]
                n_entry = cls.get_ldoce_entry(head_noun)
                if n_entry:
                    idiom_rx = re.compile(
                        r"(?<!\w)" + re.escape(v_lemma) + r"\w*(?:\s+\w+){0,4}\s"
                        + re.escape(head_noun) + r"(?!\w)", re.IGNORECASE)
                    if any(idiom_rx.search(cls._entry_text_normalizer(pat))
                           for pat in (n_entry.get("patterns") or [])):
                        verified_candidates.append(item)
                        continue

            # Collocations (ACL items, possessive frames like 'hear [one\'s] voice')
            elif item_type == "collocation":
                # ACL items or verified possessive collocations
                verified_candidates.append(item)
                continue

            # Default: if none of the specific validation branches match, drop unverified candidate

        raw_candidates = verified_candidates
        if not raw_candidates and not syllabus_expressions:
            return []

        selected: List[Dict[str, str]] = []
        seen_phrases: Set[str] = set()
        seen_quotes: Set[str] = set()

        # Phase 1: If syllabus_expressions is provided, prioritize and standardize those curriculum targets
        if syllabus_expressions:
            syllabus_candidates: List[Dict[str, Any]] = []
            for raw_expr in syllabus_expressions:
                raw_norm = re.sub(r"\s+", " ", raw_expr.strip())
                raw_norm = raw_norm.replace("\u2019", "'").replace("\u2018", "'").replace("…", "...")
                raw_norm = re.sub(r"'\s+s\b", "'s", raw_norm)
                # Strip trailing ', etc.' or 'etc.'
                raw_norm = re.sub(r",?\s*\betc\.?(?=\s+to\b|\s+for\b|\s+in\b|\s+on\b|\s+with\b|$)", "", raw_norm, flags=re.IGNORECASE)
                # Preserve parenthesized metasyntax slots: (sb) -> [sb], (sth) -> [sth], (oneself) -> [oneself]
                raw_norm = re.sub(r"(?:\(\s*|\[\s*)+(?:sb\.?|somebody|someone)(?:\s*\)|\s*\])+", "[sb]", raw_norm, flags=re.IGNORECASE)
                raw_norm = re.sub(r"(?:\(\s*|\[\s*)+(?:sth\.?|something)(?:\s*\)|\s*\])+", "[sth]", raw_norm, flags=re.IGNORECASE)
                raw_norm = re.sub(r"(?:\(\s*|\[\s*)+(?:sb/sth|sth/sb)(?:\s*\)|\s*\])+", "[sth/sb]", raw_norm, flags=re.IGNORECASE)
                raw_norm = re.sub(r"(?:\(\s*|\[\s*)+oneself(?:\s*\)|\s*\])+", "[oneself]", raw_norm, flags=re.IGNORECASE)
                # Strip remaining optional parentheticals (e.g. 'no exaggeration (to say (sth) / that (sth))', 'protect [sth] (from)')
                while re.search(r"\([^()]*\)", raw_norm):
                    raw_norm = re.sub(r"\([^()]*\)", "", raw_norm).strip()
                raw_norm = re.sub(r"\s+", " ", raw_norm).strip()
                # Normalize metasyntax placeholders:
                # 1. Possessive placeholders: one's / ones / sb.'s / sb's / somebody's / someone's -> [one's]
                raw_norm = re.sub(r"\b(?:one's|ones|sb\.?'s|somebody's|someone's)\b", "[one's]", raw_norm, flags=re.IGNORECASE)
                # 2. Reflexive placeholder: oneself -> [oneself]
                raw_norm = re.sub(r"\boneself\b", "[oneself]", raw_norm, flags=re.IGNORECASE)
                # 3. Object placeholders: sb. / sb / somebody / someone -> [sb]; sth. / sth / something / ... -> [sth]
                raw_norm = re.sub(r"\b(?:sb|somebody|someone)\b\.?", "[sb]", raw_norm, flags=re.IGNORECASE)
                raw_norm = re.sub(r"\b(?:sth|something)\b\.?", "[sth]", raw_norm, flags=re.IGNORECASE)
                raw_norm = re.sub(r"\.{2,}", "[sth]", raw_norm)
                raw_norm = re.sub(r"\[+\s*(one's|oneself|sth/sb|sb/sth|sth|sb)\s*\]+", r"[\1]", raw_norm, flags=re.IGNORECASE)
                raw_norm = re.sub(r"\[sb\]\s*/\s*\[sth\]", "[sb/sth]", raw_norm, flags=re.IGNORECASE)
                raw_norm = re.sub(r"\[sth\]\s*/\s*\[sb\]", "[sth/sb]", raw_norm, flags=re.IGNORECASE)
                raw_norm = re.sub(r"\[one's\]\s*/\s*\[sth\]", "[one's]", raw_norm, flags=re.IGNORECASE)
                raw_norm = re.sub(r"\bthe\s*/\s*\[one's\]", "[one's]", raw_norm, flags=re.IGNORECASE)
                raw_norm = re.sub(r"\bitself\s*/\s*themselves\b", "itself/themselves", raw_norm, flags=re.IGNORECASE)
                raw_norm = re.sub(r"\s+", " ", raw_norm).strip()

                # Pre-expand multi-word slash alternatives (e.g. 'have a edge over / an edge on',
                # 'not have the heart to do [sth] / for [sth]', 'open the door to [sth] / to doing [sth]',
                # 'derive from / be derived from')
                raw_norm = re.sub(r"\s*/\s*to\s+doing\s+\[sth\]$", "", raw_norm, flags=re.IGNORECASE)
                multi_slash_seeds = [raw_norm]
                if "/" in raw_norm:
                    m_tail = re.match(r"^(.*?)\s+(to\s+do\s+\[sth\]|for\s+\[sth\]|an?\s+edge\s+(?:over|on))\s*/\s*(.+)$", raw_norm, flags=re.IGNORECASE)
                    if m_tail:
                        prefix_p, alt1, alt2 = m_tail.group(1).strip(), m_tail.group(2).strip(), m_tail.group(3).strip()
                        alt1 = re.sub(r"^a\s+edge\b", "an edge", alt1, flags=re.IGNORECASE)
                        alt2 = re.sub(r"^a\s+edge\b", "an edge", alt2, flags=re.IGNORECASE)
                        multi_slash_seeds = [f"{prefix_p} {alt1}".strip(), f"{prefix_p} {alt2}".strip()]
                    else:
                        slash_parts = [p.strip() for p in re.split(r"\s*/\s*", raw_norm) if p.strip()]
                        if len(slash_parts) >= 2 and all(len(p.split()) >= 2 for p in slash_parts) and not any(p.startswith("[") or p.endswith("]") for p in slash_parts):
                            multi_slash_seeds = slash_parts
                        else:
                            multi_slash_seeds = [re.sub(r"(?<=\S)\s+/\s+(?=\S)", "/", raw_norm)]

                # Expand single-token slash alternatives (e.g. 'put [sth] into the atmosphere/air' ->
                # ['put [sth] into the atmosphere', 'put [sth] into the air'])
                branch_variants: List[str] = []
                for seed in multi_slash_seeds:
                    for bv in cls._slash_variants(seed):
                        if bv not in branch_variants:
                            branch_variants.append(bv)
                matched_branch_expr = None
                matched_sid = None
                matched_sent = None

                for branch_expr in branch_variants:
                    expr_clean = re.sub(r"\s+", " ", branch_expr.strip())
                    words = expr_clean.split()

                    # Filter proper nouns / TV trivia entities (e.g. 'lone ranger' with kind == 'entity')
                    ldoce_ent = cls.get_ldoce_entry(expr_clean)
                    if ldoce_ent and ldoce_ent.get("kind") == "entity":
                        continue

                    # Canonicalize finite forms of 'be' ('is/was/were/are/been') at the start of a syllabus expression to 'be',
                    # preserving 'be' intact per dictionary & syllabus standard ('be exhausted from', 'be obsessed with', 'be featured in').
                    if len(words) > 1 and words[0].lower() in ("is", "was", "were", "are", "been"):
                        words = ["be"] + words[1:]
                        expr_clean = " ".join(words)

                    # Normalize over-extended prepositional prefixes (e.g. 'toward the end of' -> 'the end of')
                    m_prep = re.match(r"^(?:toward|towards|near)\s+(the\s+end\s+of\b.*)", expr_clean, re.IGNORECASE)
                    if m_prep:
                        expr_clean = m_prep.group(1).strip()
                        words = expr_clean.split()

                    # Normalize over-extended determiners/quantifiers for category frames (e.g. 'any type of' -> 'type of')
                    m_det = re.match(r"^(?:any|each|every|some|a|an)\s+((?:type|kind|sort)\s+of\b.*)", expr_clean, re.IGNORECASE)
                    if m_det:
                        expr_clean = m_det.group(1).strip()
                        words = expr_clean.split()

                    # Build regex for placeholder-bearing or standard contiguous expressions
                    has_placeholder = bool(re.search(r"\[(?:one's|oneself|sth|sb|sth/sb)\]", expr_clean, re.IGNORECASE))
                    search_core = re.sub(r"(?:\s+\[(?:sth|sb|sth/sb)\])+$", "", expr_clean, flags=re.IGNORECASE).strip() or expr_clean
                    search_words = search_core.split()
                    # When a syllabus expression starts with copular/passive 'be' (e.g. 'be admitted to', 'be exhausted from'),
                    # the passage sentence may use a reduced participial clause without an explicit 'be' auxiliary ('admitted to the university').
                    search_words_no_be = search_words[1:] if (len(search_words) >= 3 and search_words[0].lower() == "be") else None
                    placeholder_rx = None
                    if has_placeholder:
                        def _build_placeholder_rx(s_words: List[str]):
                            rx_chunks = []
                            for tok in s_words:
                                tl = tok.lower()
                                if tl == "[one's]":
                                    rx_chunks.append(r"(?:my|your|his|her|its|our|their|one's|ones|[a-z]+'s)")
                                elif tl == "[oneself]":
                                    rx_chunks.append(r"(?:myself|yourself|himself|herself|itself|ourselves|yourselves|themselves|oneself)")
                                elif tl in ("[sth]", "[sb]", "[sth/sb]"):
                                    rx_chunks.append(r"(?:[a-z0-9'\-]+(?:\s+[a-z0-9'\-]+){0,5})")
                                elif tl == "not":
                                    rx_chunks.append(r"(?:not|n't|never|hardly|barely|scarcely)")
                                else:
                                    forms = sorted(cls.inflected_forms(tl) | {tl}, key=len, reverse=True)
                                    rx_chunks.append(r"(?:" + "|".join(re.escape(f) for f in forms if f) + r")")
                            return re.compile(r"\b" + r"\s+".join(rx_chunks) + r"\b", re.IGNORECASE)
                        placeholder_rx = _build_placeholder_rx(search_words)
                        placeholder_rx_no_be = _build_placeholder_rx(search_words_no_be) if search_words_no_be else None
                    else:
                        placeholder_rx_no_be = None

                    phrase_rx = None
                    phrase_rx_no_be = None
                    tokens = cls._phrase_tokens(search_core)
                    if len(tokens) >= 2:
                        rx_pat = cls._phrase_regex(tokens, 2, allow_modifier=False)
                        if rx_pat:
                            phrase_rx = re.compile(rx_pat, re.IGNORECASE)
                    if search_words_no_be and len(search_words_no_be) >= 2:
                        tokens_nb = cls._phrase_tokens(" ".join(search_words_no_be))
                        if len(tokens_nb) >= 2:
                            rx_pat_nb = cls._phrase_regex(tokens_nb, 2, allow_modifier=False)
                            if rx_pat_nb:
                                phrase_rx_no_be = re.compile(rx_pat_nb, re.IGNORECASE)

                    content_words = [w for w in search_words if not (w.startswith("[") and w.endswith("]")) and w.lower() not in ("not", "be")]

                    def _verify_verb_placeholder_match(sent_str: str) -> bool:
                        if not (has_placeholder and len(content_words) == 1):
                            return True
                        v_word = content_words[0].lower()
                        v_forms = cls.inflected_forms(v_word) | {v_word}
                        s_doc = nlp(sent_str)
                        for t in s_doc:
                            if t.text.lower() in v_forms or t.lemma_.lower() == v_word:
                                if t.pos_ in ("VERB", "AUX") and not any(c.dep_ in ("det", "poss") for c in t.children):
                                    return True
                        return False

                    # Two-pass sentence search: Pass A requires anchor_evidence_ok(s);
                    # Pass B falls back to all sentences in pool (recovering subheadings like "Hats Off to Volunteers" / "Take Corporate Action")
                    for allow_short_headings in (False, True):
                        for sid, s in pool.items():
                            if not allow_short_headings and not cls.anchor_evidence_ok(s):
                                continue
                            s_lower = s.lower()
                            e_lower = search_core.lower()
                            if placeholder_rx and (placeholder_rx.search(s) or (placeholder_rx_no_be and placeholder_rx_no_be.search(s))) and _verify_verb_placeholder_match(s):
                                matched_sid = sid
                                matched_sent = s
                                break
                            elif not has_placeholder and re.search(r"\b" + re.escape(e_lower.rstrip("?.!")) + r"\b", s_lower):
                                matched_sid = sid
                                matched_sent = s
                                break
                            elif phrase_rx and (phrase_rx.search(cls._entry_text_normalizer(s)) or (phrase_rx_no_be and phrase_rx_no_be.search(cls._entry_text_normalizer(s)))):
                                matched_sid = sid
                                matched_sent = s
                                break

                        if not matched_sid and content_words and not expr_clean.endswith("?"):
                            for sid, s in pool.items():
                                if not allow_short_headings and not cls.anchor_evidence_ok(s):
                                    continue
                                if not _verify_verb_placeholder_match(s):
                                    continue
                                s_lower = s.lower()
                                if all(re.search(r"\b" + re.escape(w.lower().rstrip("?.!")) + r"\b", s_lower) for w in content_words):
                                    matched_sid = sid
                                    matched_sent = s
                                    break
                                else:
                                    s_doc = nlp(s)
                                    s_lemmas = [t.lemma_.lower() for t in s_doc]
                                    s_tokens = [t.text.lower() for t in s_doc]
                                    if all(w.lower().rstrip("?.!") in s_lemmas or w.lower().rstrip("?.!") in s_tokens for w in content_words):
                                        matched_sid = sid
                                        matched_sent = s
                                        break

                        if matched_sid:
                            break

                    if matched_sid and matched_sent:
                        matched_branch_expr = expr_clean
                        break

                if not matched_sid or not matched_sent or not matched_branch_expr:
                    continue

                expr_clean = matched_branch_expr
                words = expr_clean.split()
                has_placeholder = bool(re.search(r"\[(?:one's|oneself|sth|sb|sth/sb)\]", expr_clean, re.IGNORECASE))
                e_lower = expr_clean.lower()
                cand_item = None

                # Family 1B Routing Guard: Open-slot preparatory-subject extraposition ('It is <adj> (for/of sb) to do [sth]')
                # belongs exclusively to Syntactic Grammar ('Information Packaging: dummy_it_extraposition'), never multi-word vocabulary.
                # Only fixed lexicalized idioms ('it is high time', 'it is no wonder', 'it is no use') are permitted as expressions.
                if re.match(
                    r"^it\s+(?:is|was|seems|becomes|gets|'s)\s+(?!high\s+time\b|no\s+wonder\b|no\s+use\b|no\s+good\b|worth\b)[a-z]+(?:\s+(?:for|of)\s+(?:\[?(?:sb|somebody|someone)\]?|[a-z]+))?(?:\s+to(?:\s+do(?:\s+\[?(?:sth|something)\]?)?)?)?$",
                    e_lower,
                ):
                    continue

                # Compositional Direct-Question Copular Clauses Guard:
                # Inverted interrogative clauses built solely from wh-word + copula/auxiliary + pronoun
                # (e.g., 'how was it?', 'what is it?', 'where are they?') lack open-class lexical heads
                # and cannot be embedded into declarative cloze stems without breaking word order.
                if re.match(
                    r"^(?:how|what|where|when|who|why)\s+(?:is|was|are|were|am|did|do|does)\s+(?:it|this|that|he|she|they|you|we|i)\b[?.!]*$",
                    e_lower,
                ):
                    continue

                # Family 1A Preparatory-Object Extraposition & Causative Resultative ('make/find/keep/leave [sth/it] <adj>')
                _fam1a_comp_map = {
                    "easier": "easy",
                    "harder": "hard",
                    "clearer": "clear",
                    "simpler": "simple",
                    "safer": "safe",
                    "surer": "sure",
                    "plainer": "plain",
                }
                _m_fam1a = re.match(
                    r"^(make|makes|made|making|find|finds|found|finding|keep|keeps|kept|keeping|leave|leaves|left|leaving)\s+(?:(it|\[sth\]|\[sb/sth\]|\[sth/sb\]|something)\s+)?(?:more\s+)?([a-z]+)$",
                    e_lower,
                )
                _fam1a_v = None
                _fam1a_adj = None
                if _m_fam1a:
                    _v_raw = _m_fam1a.group(1)
                    _v_lem = {"makes": "make", "made": "make", "making": "make", "finds": "find", "found": "find", "finding": "find", "keeps": "keep", "kept": "keep", "keeping": "keep", "leaves": "leave", "left": "leave", "leaving": "leave"}.get(_v_raw, _v_raw)
                    _adj_raw = _m_fam1a.group(3)
                    _adj_base = _fam1a_comp_map.get(_adj_raw, _adj_raw)
                    if _adj_base in {"possible", "impossible", "easy", "hard", "difficult", "clear", "certain", "plain", "necessary", "safe", "simple"}:
                        _fam1a_v = _v_lem
                        _fam1a_adj = _adj_base

                # 1. Specialized High-Frequency Idioms (figurative, non-compositional units)
                KNOWN_IDIOMS = {
                    "hats off", "hat off", "pat on the back", 
                    "spill the beans", "break the ice", "piece of cake", 
                    "bite the bullet", "call it a day", "under the weather"
                }
                # 2. Specialized Closed-Paradigm Set Phrases (discourse connectors, fixed correlatives/framing units)
                KNOWN_SET_PHRASES = {
                    "as long as", "as well as", "as soon as", "so far as",
                    "in order to", "so as to", "in spite of", "due to", "owing to",
                    "on the other hand", "on the contrary", "in addition to",
                    "as a result", "for instance", "for example", "at least",
                    "at most", "at last", "upside down", "side by side",
                    "peace of mind", "pros and cons", "no exaggeration",
                    "not up to par", "up to par"
                }

                if any(idiom in e_lower for idiom in KNOWN_IDIOMS):
                    formula = "hats off to [sb]" if e_lower in ("hat off", "hats off") and "hats off to" in matched_sent.lower() else expr_clean
                    cand_item = {
                        "sid": matched_sid,
                        "quote": matched_sent,
                        "phrase": expr_clean,
                        "type": "idiom",
                        "pattern_formula": formula,
                        "score": 100,
                    }
                elif any(sp in e_lower for sp in KNOWN_SET_PHRASES):
                    cand_item = {
                        "sid": matched_sid,
                        "quote": matched_sent,
                        "phrase": expr_clean,
                        "type": "set phrase",
                        "pattern_formula": expr_clean,
                        "score": 95,
                    }
                elif "touch" in e_lower and any(v in e_lower for v in ("keep", "stay", "get", "lose", "be")):
                    formula = f"{e_lower} [sb]" if "with" in e_lower else f"{e_lower}"
                    cand_item = {
                        "sid": matched_sid,
                        "quote": matched_sent,
                        "phrase": expr_clean,
                        "type": "idiom",
                        "pattern_formula": formula,
                        "score": 100,
                    }
                elif _fam1a_v and _fam1a_adj:
                    _uses_it = bool(_m_fam1a and _m_fam1a.group(2) == "it")
                    _fam1a_phrase = f"{_fam1a_v} it {_fam1a_adj}" if _uses_it else f"{_fam1a_v} [sth] {_fam1a_adj}"
                    cand_item = {
                        "sid": matched_sid,
                        "quote": matched_sent,
                        "phrase": _fam1a_phrase,
                        "type": "collocation",
                        "pattern_formula": _fam1a_phrase,
                        "score": 85,
                    }
                elif has_placeholder:
                    # Syllabus item already declares an explicit slot formula ('do [one\'s] part', 'put [sth] into the atmosphere', 'make the most of [sth]')
                    lookup_phrase = re.sub(r"\s+", " ", re.sub(r"\[(?:one's|oneself|sth|sb|sth/sb|sb/sth)\]", "", expr_clean, flags=re.IGNORECASE)).strip() or expr_clean
                    cand_type = cls.classify_expression_type(lookup_phrase, matched_sent) or "collocation"
                    if cand_type == "phrasal verb" and re.match(
                        r"^[a-z]+\s+\[(?:sth|sb|sth/sb|sb/sth)\]\s+(?:to|with|into|as|of|from|for|against|upon)\b",
                        expr_clean,
                        flags=re.IGNORECASE,
                    ):
                        v0 = lookup_phrase.split()[0].lower() if lookup_phrase else ""
                        v0_entry = cls.get_ldoce_entry(v0)
                        if v0_entry and any(
                            "transitive" in f"{s.get('grammar') or ''} {s.get('syntax_codes') or ''}".lower()
                            and "intransitive" not in f"{s.get('grammar') or ''} {s.get('syntax_codes') or ''}".lower()
                            for s in (v0_entry.get("senses") or [])
                            if cls._pos_code(s.get("pos")) == "v"
                        ):
                            cand_type = "collocation"
                    cand_item = {
                        "sid": matched_sid,
                        "quote": matched_sent,
                        "phrase": lookup_phrase,
                        "type": cand_type,
                        "pattern_formula": expr_clean,
                        "score": 85,
                    }
                else:
                    doc = nlp(matched_sent)
                    # Find the contiguous token span of expr_clean in matched_sent to avoid grabbing disjoint tokens
                    expr_tokens = []
                    m_span = re.search(r'\b' + re.escape(expr_clean) + r'\b', matched_sent, re.IGNORECASE)
                    if not m_span and len(words) > 1:
                        # Fallback: match words joined by flexible whitespace/punctuation
                        pat = r'\b' + r'\s+'.join(re.escape(w) for w in words) + r'\b'
                        m_span = re.search(pat, matched_sent, re.IGNORECASE)

                    if m_span:
                        sp_span = doc.char_span(m_span.start(), m_span.end())
                        if sp_span:
                            expr_tokens = list(sp_span)

                    if not expr_tokens:
                        # Find best ordered token sequence matching words by text or lemma
                        words_lower = [w.lower() for w in words]
                        best_tokens = None
                        min_dist = 9999
                        for start_t in doc:
                            if start_t.lemma_.lower() == words_lower[0] or start_t.text.lower() == words_lower[0]:
                                curr = [start_t]
                                for w in words_lower[1:]:
                                    cands = [t for t in doc if 0 < (t.i - curr[-1].i) <= 5 and (t.lemma_.lower() == w or t.text.lower() == w)]
                                    if not cands:
                                        break
                                    curr.append(cands[0])
                                if len(curr) == len(words_lower):
                                    dist = curr[-1].i - curr[0].i
                                    if dist < min_dist:
                                        min_dist = dist
                                        best_tokens = curr
                        if best_tokens:
                            expr_tokens = best_tokens
                        else:
                            dedup_tokens = []
                            for w_item in words_lower:
                                matched_t = next(
                                    (
                                        t for t in doc
                                        if (t.text.lower() == w_item or t.lemma_.lower() == w_item)
                                        and t not in dedup_tokens
                                        and (not dedup_tokens or 0 < (t.i - dedup_tokens[-1].i) <= 5)
                                    ),
                                    None
                                )
                                if matched_t is not None:
                                    dedup_tokens.append(matched_t)
                            expr_tokens = dedup_tokens

                    first_tok = expr_tokens[0] if expr_tokens else None
                    is_participial_amod = bool(first_tok and first_tok.dep_ == "amod" and first_tok.head.pos_ in ("NOUN", "PROPN"))
                    is_copular_be_frame = bool(len(words) >= 3 and words[0].lower() in ("be", "feel", "get"))

                    if is_copular_be_frame or (first_tok and first_tok.pos_ in ("VERB", "AUX") and not is_participial_amod):
                        v_lemma = words[0].lower() if is_copular_be_frame else first_tok.lemma_.lower()
                        last_tok = expr_tokens[-1] if expr_tokens else None
                        last_word_low = words[-1].lower()
                        
                        # Syntax-aware detection: check if 'to' is an infinitive marker rather than a preposition
                        is_infinitive_to = False
                        if last_tok is not None and last_tok.text.lower() == "to":
                            if last_tok.pos_ == "PART" or last_tok.dep_ == "aux":
                                is_infinitive_to = True
                            elif last_tok.dep_ != "prep" and last_tok.head.pos_ == "VERB" and last_tok.head.dep_ in ("xcomp", "advcl", "purpcl") and last_tok.i < last_tok.head.i:
                                is_infinitive_to = True

                        if is_infinitive_to and len(words) == 2:
                            # Genuine infinitive construction (e.g. tend to do [sth], fail to do [sth])
                            formula = f"{v_lemma} to do [sth]"
                            cand_type = "collocation"
                        elif (last_tok is not None and (last_tok.pos_ in ("ADP", "PART") or last_tok.dep_ in ("prep", "prt"))) or (is_copular_be_frame and last_word_low in cls._PHRASAL_TAILS):
                            prep_word = last_tok.text.lower() if last_tok is not None else last_word_low
                            # Dynamic Transitivity Audit via Sentence Dependency Tree & Valence
                            # Check whether the verb or particle takes an authentic nominal object in the context sentence
                            verb_tok = expr_tokens[0] if expr_tokens else None
                            has_contextual_obj = False
                            if last_tok is not None:
                                def _is_valid_obj(c_tok, anchor_tok) -> bool:
                                    if c_tok.pos_ not in ("NOUN", "PROPN", "PRON"):
                                        return False
                                    span_start = min(c_tok.i, anchor_tok.i)
                                    span_end = max(c_tok.i, anchor_tok.i)
                                    if any(t.text in (",", ";", ":", "\"", "'", "“", "”", "‘", "’") for t in doc[span_start : span_end + 1]):
                                        return False
                                    return True

                                # 1. Verb direct object (dobj / obj) — only for separable phrasal verb particles (prt)
                                verb_objs = [c for c in verb_tok.children if c.dep_ in ("dobj", "obj") and _is_valid_obj(c, last_tok)] if (verb_tok is not None and last_tok.dep_ == "prt") else []
                                # 2. Particle / prepositional object (pobj / obj)
                                prep_objs = [c for c in last_tok.children if c.dep_ in ("pobj", "obj", "dobj") and _is_valid_obj(c, last_tok)]
                                # 3. Passive voice extraction (e.g. 'the lights were turned off')
                                is_passive = any(c.dep_ in ("auxpass", "nsubjpass") for c in verb_tok.children) if verb_tok is not None else False
                                if verb_objs or prep_objs or is_passive:
                                    has_contextual_obj = True
                            elif is_copular_be_frame:
                                has_contextual_obj = True

                            # If the verb+particle has no object in context (e.g. 'show up', 'go on'),
                            # it is an authentic intransitive phrasal verb: generate clean formula without [sth/sb]
                            is_intransitive = not has_contextual_obj

                            has_internal_noun_or_adj = (
                                is_copular_be_frame
                                or (
                                    len(words) >= 3
                                    and any(t.pos_ in ("NOUN", "PROPN", "ADJ") for t in expr_tokens[1:-1])
                                )
                            )
                            if len(words) == 2:
                                pair_str = f"{v_lemma} {prep_word}"
                                v_ent = cls.get_ldoce_entry(v_lemma) or {}
                                v_verb_senses = [
                                    s for s in (v_ent.get("senses") or [])
                                    if cls._pos_code(s.get("pos")) == "v" or not s.get("pos")
                                ]
                                # A 2-word verb+prep is a prepositional phrasal verb when:
                                # - particle is in COMMON_PARTICLES, OR
                                # - it has a dedicated unit/phrasal_verb block, OR
                                # - prep_word is in DEPENDENT_PREPS and the primary verb sense itself governs prep_word
                                #   (as in 'long for', 'worry about', 'rely on'), unlike secondary/optional prepositional
                                #   collocations (such as 'train for', 'dock at').
                                primary_governs_prep = False
                                if prep_word in DEPENDENT_PREPS and v_verb_senses:
                                    s0 = v_verb_senses[0]
                                    s0_pats = list(s0.get("patterns") or []) + [
                                        pb.get("pattern") for pb in (s0.get("pattern_bundles") or [])
                                        if isinstance(pb, dict) and pb.get("pattern")
                                    ]
                                    if len(v_verb_senses) == 1 or any(
                                        prep_word in re.findall(r"[a-z]+", str(p).lower()) for p in s0_pats
                                    ):
                                        primary_governs_prep = True
                                is_true_pv = (
                                    prep_word in COMMON_PARTICLES
                                    or bool(cls._unit_blocks(pair_str))
                                    or bool(cls._phrasal_verb_blocks(pair_str, v_lemma))
                                    or primary_governs_prep
                                    or cls.classify_expression_type(pair_str, matched_sent) == "phrasal verb"
                                )
                                if v_lemma == "be" and prep_word == "upon":
                                    formula = "be upon [sb/sth]"
                                    cand_type = "set phrase"
                                elif is_intransitive:
                                    formula = f"{v_lemma} {prep_word}"
                                    cand_type = "phrasal verb" if is_true_pv else "collocation"
                                else:
                                    slot_tag = "[sth/sb]"
                                    if prep_word in ("at", "into", "in", "on", "under", "within", "throughout") and prep_objs:
                                        if all(
                                            po.ent_type_ != "PERSON"
                                            and po.lemma_.lower() not in ("person", "people", "man", "woman", "child", "friend", "someone", "somebody", "who", "whom")
                                            for po in prep_objs
                                        ):
                                            slot_tag = "[sth]"
                                    formula = f"{v_lemma} {prep_word} {slot_tag}"
                                    cand_type = "phrasal verb" if is_true_pv else "collocation"
                            else:
                                core = " ".join([v_lemma] + [w.lower() for w in words[1:]])
                                if "importance to" in core:
                                    formula = f"{core} [sth]"
                                elif any(core.startswith(w) for w in ("show thanks", "give thanks", "pay tribute")):
                                    formula = f"{core} [sb]"
                                elif is_intransitive:
                                    formula = core
                                else:
                                    formula = f"{core} [sth/sb]"
                                cand_type = "collocation" if has_internal_noun_or_adj else (cls.classify_expression_type(expr_clean, matched_sent) or "phrasal verb")
                        else:
                            # Verb + object / adj (e.g. make smart choices, keep silent, have a try, take action)
                            # Use expr_clean words with lemmatized head verb so intervening sentence modifiers
                            # (e.g. 'Corporate' in 'Take Corporate Action') are not injected into the canonical formula.
                            core_words = [v_lemma] + [w.lower() for w in words[1:]]
                            core = " ".join(core_words)
                            formula = core
                            cand_type = "collocation"

                        cand_item = {
                            "sid": matched_sid,
                            "quote": matched_sent,
                            "phrase": expr_clean,
                            "type": cand_type,
                            "pattern_formula": formula,
                            "score": 90 if cand_type == "phrasal verb" else 80,
                        }
                    elif words[0].lower() in ("in", "on", "at", "by", "for", "with", "from", "to", "under", "over", "toward", "towards"):
                        # Prepositional expressions (e.g. in the past, toward the end of [sth], in touch with [sb])
                        formula = expr_clean
                        if expr_clean.endswith(" of") or expr_clean.endswith(" for") or expr_clean.endswith(" with"):
                            formula = f"{expr_clean} [sth]"
                        elif expr_clean.endswith(" to"):
                            formula = f"{expr_clean} [sth/sb]"
                        cand_item = {
                            "sid": matched_sid,
                            "quote": matched_sent,
                            "phrase": expr_clean,
                            "type": "set phrase",
                            "pattern_formula": formula,
                            "score": 78 if cls.is_attested_phrase(expr_clean.lower()) else 50,
                        }
                    else:
                        # Nominal/quantifier/partitive/idiomatic expressions (e.g. a piece of [sth], any type of [sth], show thanks to [sb], hats off to [sb])
                        formula = expr_clean
                        if expr_clean.endswith(" of") or expr_clean.endswith(" for") or expr_clean.endswith(" with"):
                            formula = f"{expr_clean} [sth]"
                        elif expr_clean.endswith(" to"):
                            # If expression relates to importance, attention, or priority, default to [sth]
                            if any(k in expr_clean.lower() for k in ("importance", "attention", "priority", "regard")):
                                formula = f"{expr_clean} [sth]"
                            elif any(expr_clean.lower().startswith(w) for w in ("show thanks", "give thanks", "hats off", "hat off", "pay tribute")):
                                formula = f"{expr_clean} [sb]"
                            else:
                                formula = f"{expr_clean} [sth/sb]"
                        elif expr_clean.lower() in ("hat off", "hats off"):
                            formula = "hats off to [sb]"

                        cand_type = "idiom" if expr_clean.lower() in ("hat off", "hats off", "pat on the back") else "collocation"
                        cand_item = {
                            "sid": matched_sid,
                            "quote": matched_sent,
                            "phrase": expr_clean,
                            "type": cand_type,
                            "pattern_formula": formula,
                            "score": 75 if cand_type == "idiom" else 60,
                        }

                if cand_item:
                    syllabus_candidates.append(cand_item)

            # Sort syllabus candidates by pedagogical score (idioms & phrasal verbs first)
            syllabus_candidates.sort(key=lambda x: x["score"], reverse=True)

            # Pass 1: maximize sentence quote diversity among syllabus expressions
            for item in syllabus_candidates:
                p_clean = item["phrase"].lower()
                if p_clean not in seen_phrases and item["quote"] not in seen_quotes:
                    selected.append({
                        "sid": item["sid"],
                        "quote": item["quote"],
                        "phrase": item["phrase"],
                        "type": item["type"],
                        "pattern_formula": item["pattern_formula"],
                    })
                    seen_phrases.add(p_clean)
                    seen_quotes.add(item["quote"])
                if len(selected) >= target_count:
                    break

            # Pass 2: if more syllabus slots needed, allow reusing quote for distinct syllabus phrase
            if len(selected) < target_count:
                for item in syllabus_candidates:
                    p_clean = item["phrase"].lower()
                    if p_clean not in seen_phrases:
                        selected.append({
                            "sid": item["sid"],
                            "quote": item["quote"],
                            "phrase": item["phrase"],
                            "type": item["type"],
                            "pattern_formula": item["pattern_formula"],
                        })
                        seen_phrases.add(p_clean)
                    if len(selected) >= target_count:
                        break

        # Phase 2: If no syllabus provided and still under target_count, fill from computational NLP/ACL/OCD candidates
        if not syllabus_expressions and len(selected) < target_count:
            # Sort raw_candidates by score descending
            raw_candidates.sort(key=lambda x: x["score"], reverse=True)

            # Pass 3: balance across types with sentence quote diversity
            for item in raw_candidates:
                p_clean = item["phrase"].lower()
                if (
                    p_clean not in seen_phrases
                    and item["quote"] not in seen_quotes
                    and not any(cls.are_same_word_family(p_clean, sp) for sp in seen_phrases)
                ):
                    selected.append({
                        "sid": item["sid"],
                        "quote": item["quote"],
                        "phrase": item["phrase"],
                        "type": item["type"],
                        "pattern_formula": item["pattern_formula"],
                    })
                    seen_phrases.add(p_clean)
                    seen_quotes.add(item["quote"])
                if len(selected) >= target_count:
                    break

            # Pass 4: if still under target_count, allow reusing quote if phrase is distinct
            if len(selected) < target_count:
                for item in raw_candidates:
                    p_clean = item["phrase"].lower()
                    if (
                        p_clean not in seen_phrases
                        and not any(cls.are_same_word_family(p_clean, sp) for sp in seen_phrases)
                    ):
                        selected.append({
                            "sid": item["sid"],
                            "quote": item["quote"],
                            "phrase": item["phrase"],
                            "type": item["type"],
                            "pattern_formula": item["pattern_formula"],
                        })
                        seen_phrases.add(p_clean)
                    if len(selected) >= target_count:
                        break

        return selected[:target_count]


    @classmethod
    def build_top_k_expression_candidates(
        cls,
        expr_word: str,
        quote: str,
        k: int = 12,
        expr_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Pure Symbolic Candidate Selection for Multi-Word Expressions (NO Embedding Model):
        1. Phrasal verb blocks & direct multi-word PedLex headword entries
        2. PedLex phrase index candidates (exact, prefix, or slash-expanded unit_key match)
        3. Strictly attested phrase units on constituent headwords (and WordNet/constituent fallbacks ONLY if unattested)
        """
        clean = re.sub(r'\[.*?\]|\(.*?\)', '', expr_word).strip().lower()
        clean = re.sub(r"\b(?:sb's|sb|sth|one's|oneself)\b", "", clean).strip()
        clean = re.sub(r"\s+", " ", clean).strip()
        tokens = clean.split()
        lem_0 = cls.lemmatize_headword(tokens[0]) if tokens else ""
        if tokens and lem_0 == tokens[0] and tokens[0].endswith("ed") and len(tokens[0]) > 4:
            stem_ed = tokens[0][:-2]
            if cls.get_pedlex_entry(stem_ed):
                lem_0 = stem_ed
        lem_tokens = ([lem_0] + tokens[1:]) if tokens else []
        candidates = []
        seen_defs = set()
        q_low = (quote or "").lower()
        locked_expr_type = (
            expr_type
            if (expr_type in cls.EXPRESSION_TYPE_LABELS)
            else (cls.classify_expression_type(expr_word, quote) or "collocation")
        )

        def _add_cand(df_str: str, pos_str: str, src_str: str, ex_list: List[str], raw_s: Optional[Dict[str, Any]], is_attested: bool = True):
            if not df_str:
                return
            df_for_norm = re.sub(r"^(?:\s*\[[^\]]+\]\s*)+", "", df_str.strip())
            m_colloc_pref = re.match(r"^([a-z0-9\s/()'-]{2,40}):\s*(.+)$", df_for_norm, re.I)
            if m_colloc_pref:
                pref_uk = cls._unit_key(m_colloc_pref.group(1))
                if pref_uk and (pref_uk in (u_key, clean) or u_key in pref_uk or clean in pref_uk):
                    df_for_norm = m_colloc_pref.group(2)
            norm_d = re.sub(r"[^a-z0-9]+", " ", df_for_norm.lower()).strip()
            if not norm_d or norm_d in seen_defs:
                return
            # Guard against 'be there (for somebody)' when 'for' is absent from both formula and sentence
            if clean == "be there" and " for " not in f" {q_low} ":
                has_for_sb = "for somebody" in df_str.lower() or "help them" in df_str.lower() or any(
                    "for somebody" in str(u.get("phrase") if isinstance(u, dict) else u).lower()
                    for u in ((raw_s.get("units") if isinstance(raw_s, dict) else None) or [])
                )
                if has_for_sb:
                    return
            # Guard against [spoken] sympathy formula for 'think of' ('I'm thinking of you') when used in narrative recall
            if clean in ("think of", "think about") and "used to say that you care about and feel sympathy" in df_str.lower():
                if not any(p in q_low for p in ("thinking of you", "think of you", "thoughts are with")):
                    return
            # Guard against stative 'to seem similar to someone or something else' for 'remind of' when used in passive voice ('be reminded of')
            if "remind" in clean and "to seem similar to" in df_str.lower():
                if re.search(r"\b(?:am|is|are|was|were|be|been|being|get|gets|got)\s+(?:often\s+|always\s+|suddenly\s+|immediately\s+|constantly\s+)?reminded\b", q_low):
                    return
            # Guard against narrow [technical] chemical reaction sense ('if a chemical substance reacts') in non-chemical contexts
            if "if a chemical substance reacts" in df_str.lower():
                _chem_words = {"chemical", "substance", "acid", "metal", "gas", "oxygen", "water", "element", "compound", "solution", "calcium", "sulphur", "hydrogen", "carbon", "alkali", "molecule", "atom", "chlorine", "iron"}
                if not (set(re.findall(r"[a-z]{3,}", q_low)) & _chem_words):
                    return
            # Guard against literal vehicle backing-up sense ('if a vehicle or its driver reverses') or reciprocal plural-order sense ('the usual things two people do') when 'reverse' takes a singular reflexive pronoun ('reverse oneself/myself')
            if "if a vehicle or its driver reverses" in df_str.lower() or "the usual things two people do" in df_str.lower():
                if "oneself" in expr_word.lower() or re.search(r"\breverse[ds]?\s+(?:myself|yourself|himself|herself|itself|oneself)\b", q_low):
                    return
            seen_defs.add(norm_d)
            cleaned_exs = [str(e).strip().strip('"“”') for e in (ex_list or []) if e and str(e).strip()]
            candidates.append({
                'definition': df_str,
                'pos': locked_expr_type,
                'source': src_str,
                'examples': cleaned_exs,
                'raw_sense': raw_s,
                'attested': is_attested,
                'signpost': (raw_s.get('signpost', '') if isinstance(raw_s, dict) else ''),
                'patterns': list(dict.fromkeys(
                    list((raw_s.get('patterns') if isinstance(raw_s, dict) else None) or [])
                    + list((raw_s.get('units') if isinstance(raw_s, dict) else None) or [])
                )),
            })

        # 1. Phrasal verb blocks & direct multi-word PedLex headword entries (e.g. 'stuffed animal', 'comic book')
        u_key = cls._unit_key(clean)
        lem_u_key = cls._unit_key(" ".join(lem_tokens)) if lem_tokens else u_key
        unit_keys = []
        if re.search(r"\boneself\b", expr_word, re.I) and lem_0:
            for refl_uk in (f"{lem_0} yourself", f"{lem_0} oneself"):
                if refl_uk not in unit_keys:
                    unit_keys.append(refl_uk)
        # If expr_word has an internal modifier slot between a determiner and noun (e.g. 'on a (sth) scale')
        # and the quote realizes that slot with an adjective (e.g. 'on a large scale'), include the realized phrase key
        m_det_mod_slot = re.match(
            r"^([a-z]+\s+(?:a|an|the))\s+(?:\[[^\]]+\]|\([^)]+\))\s+([a-z]+)$",
            expr_word.strip(),
            re.I,
        )
        if m_det_mod_slot and q_low:
            pref_dm = m_det_mod_slot.group(1).lower()
            tail_dm = m_det_mod_slot.group(2).lower()
            m_q_realized = re.search(rf"\b({re.escape(pref_dm)}\s+[a-z]+\s+{re.escape(tail_dm)})\b", q_low)
            if m_q_realized:
                realized_uk = cls._unit_key(m_q_realized.group(1))
                if realized_uk and realized_uk not in unit_keys:
                    unit_keys.append(realized_uk)
        for uk_cand in (u_key, lem_u_key, clean, " ".join(lem_tokens)):
            if uk_cand and uk_cand not in unit_keys:
                unit_keys.append(uk_cand)
            if uk_cand and (" around" in uk_cand or " round" in uk_cand):
                alt_uk = re.sub(r"\baround\b", "round", uk_cand) if " around" in uk_cand else re.sub(r"\bround\b", "around", uk_cand)
                if alt_uk not in unit_keys:
                    unit_keys.append(alt_uk)
            uk_parts = uk_cand.split() if uk_cand else []
            if (
                len(uk_parts) == 3
                and uk_parts[1] in {"away", "back", "down", "in", "off", "on", "out", "over", "round", "through", "up", "around", "aside", "apart", "ahead", "forth", "together"}
                and uk_parts[2] in {"to", "into", "onto", "from", "for", "with", "of", "on", "in", "at", "by"}
            ):
                vp2 = f"{uk_parts[0]} {uk_parts[1]}"
                if vp2 not in unit_keys:
                    unit_keys.append(vp2)
            elif (
                len(uk_parts) == 4
                and uk_parts[0] in {"be", "keep", "stay", "get", "lose", "remain", "fall", "come", "go", "put", "make", "take"}
                and uk_parts[1] in {"in", "on", "at", "out", "off", "under", "beyond", "within"}
                and uk_parts[3] in {"with", "to", "of", "from", "for", "in", "on", "at"}
            ):
                for sub_uk in (f"{uk_parts[0]} {uk_parts[1]} {uk_parts[2]}", f"{uk_parts[1]} {uk_parts[2]} {uk_parts[3]}", f"{uk_parts[1]} {uk_parts[2]}"):
                    if sub_uk not in unit_keys:
                        unit_keys.append(sub_uk)
            elif (
                len(uk_parts) == 3
                and uk_parts[0] in {"make", "take", "have", "do", "give", "get", "keep", "put", "set", "hold", "pay"}
                and uk_parts[1] not in {"in", "on", "at", "out", "off", "up", "down", "away", "back", "over", "through", "to", "for", "with", "of", "from", "a", "an", "the"}
            ):
                for sub_vn in (f"{uk_parts[0]} {uk_parts[2]}", f"{uk_parts[0]} {cls._surface_lemma(uk_parts[2])}"):
                    if sub_vn not in unit_keys:
                        unit_keys.append(sub_vn)

        def _unit_key_matches(cand_raw_or_uk: str, uk_target: str, allow_prefix: bool = False) -> bool:
            cand_uk = cls._unit_key(cand_raw_or_uk)
            if not cand_uk or not uk_target or len(cand_uk.split()) < 2:
                return False
            if cand_uk == uk_target:
                return True
            if allow_prefix and (cand_uk.startswith(uk_target + " ") or uk_target.startswith(cand_uk + " ")):
                return True
            # Handle open modifier lists before 'etc' (e.g. 'on a large/small/grand etc scale' -> 'on a scale')
            if re.search(r"\betc\b", str(cand_raw_or_uk), re.I):
                stripped_etc = re.sub(r"\b(?:[a-z]+(?:/[a-z]+)*\s+)+etc\.?\b", "", str(cand_raw_or_uk), flags=re.I)
                etc_uk = cls._unit_key(stripped_etc)
                if etc_uk and len(etc_uk.split()) >= 2:
                    if etc_uk == uk_target or (allow_prefix and (etc_uk.startswith(uk_target + " ") or uk_target.startswith(etc_uk + " "))):
                        return True
            # Check _declared_unit_spellings (which properly expands token-level slashes like 'open the door/way to something')
            for sp in cls._declared_unit_spellings(cand_raw_or_uk):
                sp_uk = cls._unit_key(re.sub(r"\bjust\b", "", sp).strip())
                if len(sp_uk.split()) < 2:
                    continue
                if sp_uk == uk_target:
                    return True
                if allow_prefix and (sp_uk.startswith(uk_target + " ") or uk_target.startswith(sp_uk + " ")):
                    return True
            # Expand slash alternatives and optional words ('just') in unit phrases like 'may/might/could (just) as well'
            for br in cls.clean_and_expand_pattern_slashes(cand_raw_or_uk):
                br_uk = cls._unit_key(re.sub(r"\bjust\b", "", br).strip())
                if len(br_uk.split()) < 2:
                    continue
                if br_uk == uk_target:
                    return True
                if allow_prefix and (br_uk.startswith(uk_target + " ") or uk_target.startswith(br_uk + " ")):
                    return True
            cand_no_just = re.sub(r"\s+", " ", re.sub(r"\bjust\b", "", cand_uk)).strip()
            if len(cand_no_just.split()) >= 2 and cand_no_just == uk_target:
                return True
            return False

        is_verb_expr = (
            locked_expr_type == "phrasal verb"
            or (tokens and cls.determine_contextual_pos(tokens[0], quote) == "verb")
        )
        _has_internal_obj_slot = bool(
            re.search(
                r"\b[a-z]+\s+[\(\[]\s*(?:sth|sb|sth/sb|sb/sth|something|somebody)\s*[\)\]]\s+(?:to|with|into|from|of|for|on|upon|in|at|by|as|against)\b",
                str(expr_word or "").lower(),
            )
        )
        _is_passive_v_prep = bool(
            quote
            and len(tokens) == 2
            and tokens[1] in {"to", "with", "into", "from", "of", "for", "on", "upon", "as", "against"}
            and re.search(
                rf"\b(?:am|is|are|was|were|be|been|being|get|gets|got|gotten)\s+(?:[a-z]+ly\s+)?{re.escape(tokens[0])}(?:ed|d|en|t)?\s+{re.escape(tokens[1])}\b",
                q_low,
            )
        )
        is_trivalent_prep_frame = _has_internal_obj_slot or _is_passive_v_prep
        if len(tokens) >= 2 or any(" " in uk for uk in unit_keys):
            # Check phrasal verb blocks FIRST so verbal expressions ('wind up') never get hijacked by hyphenated noun entries ('wind-up')
            for uk in unit_keys:
                for block in cls._unit_blocks(uk):
                    b_senses = cls._unit_senses(block, uk)
                    if is_trivalent_prep_frame and b_senses:
                        base_v0 = uk.split()[0]
                        v0_ent = cls.get_pedlex_entry(base_v0)
                        has_trans_base = bool(
                            v0_ent
                            and any(
                                "transitive" in f"{s.get('grammar') or ''} {s.get('syntax_codes') or ''}".lower()
                                and "intransitive" not in f"{s.get('grammar') or ''} {s.get('syntax_codes') or ''}".lower()
                                for s in (v0_ent.get("senses") or [])
                                if cls._pos_code(s.get("pos")) == "v"
                            )
                        )
                        if has_trans_base:
                            blk_hw_l = str(block.get("headword") or "").lower()
                            b_senses = [
                                s for s in b_senses
                                if any(
                                    re.search(
                                        rf"\b{re.escape(base_v0)}\s+(?:somebody|something|someone|sb|sth)\b.*\b{re.escape(uk.split()[-1])}\b",
                                        str(pat_or_hw).lower(),
                                    )
                                    for pat_or_hw in ([blk_hw_l] + list(s.get("patterns") or []))
                                )
                                and not (
                                    ("intransitive" in f"{s.get('grammar') or ''} {s.get('syntax_codes') or ''}".lower()
                                     and "transitive" not in f"{s.get('grammar') or ''} {s.get('syntax_codes') or ''}".lower().replace("intransitive", ""))
                                    or "no passive" in f"{s.get('grammar') or ''} {s.get('syntax_codes') or ''}".lower()
                                )
                            ]
                    if len(b_senses) >= 2 and quote:
                        base_v = uk.split()[0]
                        part_w = uk.split()[-1]
                        # Filter out speech-act retraction senses ('admit that you were wrong', 'something that you said')
                        # when the quote has a separable noun object between verb and particle ('take her child back') or 'back to <NP>'
                        has_sep_or_dir = bool(
                            re.search(rf"\b{re.escape(base_v)}[a-z]*\s+(?:[a-z]+\s+){{1,4}}{re.escape(part_w)}\b", q_low)
                            or re.search(rf"\b{re.escape(part_w)}\s+to\s+(?:the|a|an|his|her|their|our|my|your|[a-z]+)\b", q_low)
                        )
                        if has_sep_or_dir and not any(sw in q_low for sw in ("said", "say", "words", "statement", "remark", "comment", "apolog", "admit", "wrong")):
                            non_speech = [
                                s for s in b_senses
                                if "admit that you were wrong" not in (s.get("definition") or "").lower()
                                and "something that you said" not in (s.get("definition") or "").lower()
                            ]
                            if non_speech:
                                b_senses = non_speech
                        # Filter out fixed-pronoun imperative senses (e.g. 'cut it/that out' = stop doing sth annoying)
                        # when the quote does not contain 'V it/that P'
                        if not re.search(rf"\b{re.escape(base_v)}\s+(?:it|that)\s+{re.escape(part_w)}\b", q_low):
                            non_fixed_imp = [
                                s for s in b_senses
                                if not (
                                    s.get("patterns")
                                    and all(
                                        re.match(rf"^{re.escape(base_v)}\s+(?:it|that)(?:/(?:it|that))?\s+{re.escape(part_w)}$", str(p).strip().lower())
                                        for p in s.get("patterns", [])
                                    )
                                )
                            ]
                            if non_fixed_imp:
                                b_senses = non_fixed_imp
                        prep_b = [dict(s, patterns=list(s.get("patterns") or [uk])) for s in b_senses]
                        res_b = cls._lock_sense({"word": base_v, "senses": prep_b}, quote=quote, target_pos="verb", return_scores=True)
                        if res_b and len(res_b) >= 5 and res_b[4]:
                            has_any_ex = any(s.get("examples") for s in b_senses)
                            ranked_idxs = [
                                s_i for s_i, _ in sorted(
                                    res_b[4],
                                    key=lambda x: (
                                        -(1 if (not has_any_ex or b_senses[x[0]].get("examples")) else 0),
                                        -x[1],
                                        x[0],
                                    ),
                                )
                            ]
                            b_senses = [b_senses[s_i] for s_i in ranked_idxs if 0 <= s_i < len(b_senses)]
                    for s in b_senses:
                        df = cls.format_ldoce_definition(s)
                        _add_cand(df, locked_expr_type, block.get('phrase') or block.get('headword') or uk, s.get('examples', []), s, True)
            for mw_cand in unit_keys:
                _mw_has_trans_base = False
                if is_trivalent_prep_frame and mw_cand:
                    _mw_v0_ent = cls.get_pedlex_entry(mw_cand.split()[0])
                    _mw_has_trans_base = bool(
                        _mw_v0_ent
                        and any(
                            "transitive" in f"{s.get('grammar') or ''} {s.get('syntax_codes') or ''}".lower()
                            and "intransitive" not in f"{s.get('grammar') or ''} {s.get('syntax_codes') or ''}".lower()
                            for s in (_mw_v0_ent.get("senses") or [])
                            if cls._pos_code(s.get("pos")) == "v"
                        )
                    )
                mw_entry = cls.get_pedlex_entry(mw_cand)
                if mw_entry and mw_entry.get("senses"):
                    for s_i, s in enumerate(mw_entry.get("senses") or []):
                        if is_verb_expr and cls._pos_code(s.get("pos")) in ("n", "adj"):
                            continue
                        if _mw_has_trans_base:
                            s_gram_mw = f"{s.get('grammar') or ''} {s.get('syntax_codes') or ''}".lower()
                            mw_hw_pats = [str(mw_entry.get("headword") or "")] + list(s.get("patterns") or [])
                            has_int_obj_mw = any(
                                re.search(
                                    rf"\b{re.escape(mw_cand.split()[0])}\s+(?:somebody|something|someone|sb|sth)\b.*\b{re.escape(mw_cand.split()[-1])}\b",
                                    str(ph).lower(),
                                )
                                for ph in mw_hw_pats
                            )
                            if not has_int_obj_mw or ("intransitive" in s_gram_mw and "transitive" not in s_gram_mw.replace("intransitive", "")) or "no passive" in s_gram_mw:
                                continue
                        df = cls.format_ldoce_definition(s)
                        _add_cand(df, locked_expr_type, f"{mw_cand} #{s_i}", s.get("examples", []), s, True)
                for _host_p, row_p in cls._unit_phrase_rows(mw_cand):
                    if not isinstance(row_p, dict) or row_p.get("resolved_via") == "sense_example":
                        continue
                    res_w = str(row_p.get("resolved_word") or row_p.get("target_word") or "").strip().lower()
                    res_s_idx = row_p.get("resolved_sense_idx")
                    if res_w and isinstance(res_s_idx, int) and res_s_idx >= 0:
                        res_ent = cls.get_pedlex_entry(res_w)
                        res_senses = (res_ent.get("senses") or []) if res_ent else []
                        if 0 <= res_s_idx < len(res_senses):
                            s_res = res_senses[res_s_idx]
                            if _mw_has_trans_base:
                                s_gram_res = f"{s_res.get('grammar') or ''} {s_res.get('syntax_codes') or ''}".lower()
                                if ("intransitive" in s_gram_res and "transitive" not in s_gram_res.replace("intransitive", "")) or "no passive" in s_gram_res:
                                    continue
                            df_res = cls.format_ldoce_definition(s_res)
                            exs_res = list(row_p.get("examples") or s_res.get("examples") or [])
                            _add_cand(df_res, locked_expr_type, f"{res_w} #{res_s_idx}", exs_res, s_res, True)

        is_passive_frame = len(tokens) >= 3 and tokens[0] in ("be", "feel", "get") and tokens[1].endswith("ed")

        # 2. PedLex phrase index candidates (exact or slash-expanded unit_key match first, prefix fallback if empty)
        query_list = []
        for q_s in [clean, u_key, " ".join(lem_tokens), lem_u_key] + list(unit_keys):
            if q_s and q_s not in query_list:
                query_list.append(q_s)
        for allow_pref in (False, True):
            if allow_pref and candidates:
                break
            p_matched_senses = []
            for query_str in query_list:
                phrase_cands = cls.lookup_ldoce_phrase_candidates(query_str)
                for host, s_idx, tier, raw_p in phrase_cands:
                    raw_p_low = str(raw_p).lower()
                    if "hospital" in raw_p_low and quote and not any(hw in q_low for hw in ("hospital", "patient", "ward", "doctor", "ill", "sick", "injured", "emergency")):
                        continue
                    if is_passive_frame and re.search(rf"\b{re.escape(tokens[-1])}\s+(?:doing\s+|somebody\b|sb\b)", re.sub(r"[()\[\]]+", " ", raw_p_low)):
                        continue
                    if not any(_unit_key_matches(raw_p, uk, allow_prefix=allow_pref) for uk in unit_keys):
                        continue
                    h_entry = cls.get_pedlex_entry(host)
                    if h_entry:
                        if tier == "colloc_phrase":
                            matched_colloc_gloss = False
                            for _c_sec, c_items in (h_entry.get("collocations") or {}).items():
                                for c_it in (c_items or []):
                                    if not isinstance(c_it, dict):
                                        continue
                                    c_raw = str(c_it.get("phrase") or c_it.get("collocation") or "").strip()
                                    c_gloss = re.sub(r"^[\(\s=]+|[\)\s]+$", "", str(c_it.get("gloss") or "")).strip()
                                    if c_raw and c_gloss and any(_unit_key_matches(c_raw, uk, allow_prefix=allow_pref) for uk in unit_keys):
                                        c_ex = str(c_it.get("example") or "").strip()
                                        c_exs = [c_ex] if c_ex else []
                                        s_colloc = {"definition": c_gloss, "examples": c_exs, "pos": locked_expr_type}
                                        _add_cand(c_gloss, locked_expr_type, f"{host}:colloc", c_exs, s_colloc, True)
                                        matched_colloc_gloss = True
                            if matched_colloc_gloss:
                                continue
                        if tier == "phrasal_verb":
                            # Phrasal verb blocks are already resolved in Step 1 (`_unit_blocks`);
                            # indexing `h_entry.senses[s_idx]` here would mistakenly index main entry senses (e.g. noun sense 0 of 'factor').
                            continue
                        senses = h_entry.get('senses', [])
                        if s_idx is not None and s_idx < len(senses):
                            s = senses[s_idx]
                            if is_verb_expr and cls._pos_code(s.get("pos")) in ("n", "adj", "adv"):
                                continue
                            if not cls.format_ldoce_definition(s) and s.get("crossrefs"):
                                for xref in (s.get("crossrefs") or []):
                                    if not isinstance(xref, dict):
                                        continue
                                    x_target = str(xref.get("target_word") or "").strip().lower()
                                    x_entry = cls.get_pedlex_entry(x_target) if x_target else None
                                    if x_entry:
                                        for x_i, x_s in enumerate(x_entry.get("senses") or []):
                                            if cls.format_ldoce_definition(x_s) and any(
                                                _unit_key_matches(u.get("phrase", "") if isinstance(u, dict) else str(u), uk, allow_prefix=allow_pref)
                                                for u in (x_s.get("units") or [])
                                                for uk in unit_keys
                                            ):
                                                exact_uk_prio = 1 if any(cls._unit_key(raw_p) == uk for uk in unit_keys) else 0
                                                p_matched_senses.append((exact_uk_prio, x_target, x_i, x_s))
                                continue
                            exact_uk_prio = 1 if any(cls._unit_key(raw_p) == uk for uk in unit_keys) else 0
                            p_matched_senses.append((exact_uk_prio, host, s_idx, s))
            if len(p_matched_senses) >= 2 and quote:
                mock_ps = [item[3] for item in p_matched_senses]
                res_p = cls._lock_sense({"word": p_matched_senses[0][1], "senses": mock_ps}, quote=quote, return_scores=True)
                sc_p = dict(res_p[4]) if (res_p and len(res_p) >= 5 and res_p[4]) else {}
                p_matched_senses.sort(key=lambda x: (-x[0], -sc_p.get(mock_ps.index(x[3]), 0.0), x[2]))
            for _prio, host, s_idx, s in p_matched_senses:
                df = cls._compose_collocation_definition(u_key, locked_expr_type, host, s, cls.format_ldoce_definition(s), quote)
                s_exs = list(s.get('examples', []) or [])
                for u_obj in (s.get('units') or []):
                    if isinstance(u_obj, dict) and u_obj.get('examples'):
                        for u_e in u_obj['examples']:
                            if u_e and u_e not in s_exs:
                                s_exs.append(u_e)
                for pb in (s.get('pattern_bundles') or []):
                    if isinstance(pb, dict) and pb.get('example') and pb['example'] not in s_exs:
                        s_exs.append(pb['example'])
                for arb_e in cls._arbitrated_example_pool(host, s, u_key):
                    if arb_e and arb_e not in s_exs:
                        s_exs.append(arb_e)
                m_ex = (
                    cls._unit_example(u_key, s_exs, allow_modifier=False)
                    or cls._unit_example(u_key, s_exs, allow_modifier=True)
                    or cls._relaxed_unit_example(u_key, s_exs)
                )
                if m_ex:
                    s_exs = [m_ex] + [e for e in s_exs if e != m_ex]
                _add_cand(df, locked_expr_type, f"{host} #{s_idx}", s_exs, s, True)

        # 3. Constituent headword senses:
        # Pass 3A-1 scans all senses of each constituent token strictly for attested multi-word `units` matches.
        stop_func = {
            "a", "an", "the", "to", "of", "in", "on", "at", "by", "for", "with", "from",
            "into", "onto", "upon", "over", "under", "up", "down", "off", "away", "around", "round", "back",
            "not", "no", "be", "do", "have", "one", "ones", "it", "that", "this", "as", "or", "and"
        }
        LIGHT_OR_MODAL_TOKS = {
            "take", "make", "come", "go", "get", "give", "have", "do", "be", "put", "keep",
            "fall", "turn", "set", "run", "hold", "bring", "find", "might", "can", "will",
            "may", "would", "should", "could", "must", "there", "here"
        }
        raw_toks = []
        for t in (lem_tokens + tokens):
            if (len(t) >= 3 and (t not in stop_func or (t == "out" and clean == "out of"))) and t not in raw_toks:
                raw_toks.append(t)
            # When a copular/passive frame has a -ed participle at position 1 ('be obsessed with', 'be featured in', 'be measured in'),
            # also include its base verb lemma so Pass 3A-1/3A-2 can locate units/patterns stored on the base verb entry.
            if t.endswith("ed") and len(t) > 4:
                for v_stem in ([cls._surface_lemma(t)] + cls._headword_candidates(t)):
                    if v_stem and v_stem != t and v_stem not in raw_toks and cls.get_pedlex_entry(v_stem):
                        raw_toks.append(v_stem)
        # Prioritize specific open-class content words ahead of light verbs/modals
        content_toks = [t for t in raw_toks if t not in LIGHT_OR_MODAL_TOKS] + [t for t in raw_toks if t in LIGHT_OR_MODAL_TOKS]

        def _sense_matches_unit_strictly(s_obj: Dict[str, Any], allow_sub: bool = False) -> Tuple[bool, List[str]]:
            s_units = s_obj.get("units") or []
            for u_obj in s_units:
                u_raw = u_obj.get("phrase") if isinstance(u_obj, dict) else str(u_obj or "")
                u_p = cls._unit_key(u_raw or "")
                if not u_p or len(u_p.split()) < 2:
                    continue
                for uk in unit_keys:
                    if _unit_key_matches(u_raw or u_p, uk, allow_prefix=allow_sub):
                        u_ex = (u_obj.get("examples") or []) if isinstance(u_obj, dict) else []
                        return True, u_ex
                    if allow_sub and len(u_p.split()) >= 2 and len(uk.split()) >= 2 and (f" {u_p} " in f" {uk} " or f" {uk} " in f" {u_p} "):
                        u_ex = (u_obj.get("examples") or []) if isinstance(u_obj, dict) else []
                        return True, u_ex
            return False, []

        head_candidates = {tokens[0], lem_0} if tokens else set()
        if len(tokens) >= 2 and tokens[0] in ("be", "feel", "get", "make", "keep", "find", "leave", "hold", "remain", "stay", "become", "turn", "fall"):
            head_candidates.add(tokens[1])
            for v_stem_1 in ([cls._surface_lemma(tokens[1])] + cls._headword_candidates(tokens[1])):
                if v_stem_1:
                    head_candidates.add(v_stem_1)
        if len(tokens) >= 2 and not (len(tokens) >= 3 and tokens[0] in ("be", "feel", "get") and tokens[1].endswith("ed")):
            for ct in content_toks:
                head_candidates.add(ct)

        is_passive_frame = len(tokens) >= 3 and tokens[0] in ("be", "feel", "get") and tokens[1].endswith("ed")

        def _sense_matches_head_pattern(c_tok: str, s_obj: Dict[str, Any]) -> bool:
            if tokens and c_tok in head_candidates:
                if is_passive_frame and cls._pos_code(s_obj.get("pos")) in ("n", "adv"):
                    return False
                if is_verb_expr and c_tok in (tokens[0], lem_0) and cls._pos_code(s_obj.get("pos")) in ("n", "adv"):
                    return False
                if s_obj.get("units") and not _sense_matches_unit_strictly(s_obj, allow_sub=True)[0] and not s_obj.get("patterns") and not s_obj.get("pattern_bundles"):
                    return False
                if is_trivalent_prep_frame:
                    s_gram = f"{s_obj.get('grammar') or ''} {s_obj.get('syntax_codes') or ''}".lower()
                    if "no passive" in s_gram or ("intransitive" in s_gram and "transitive" not in s_gram.replace("intransitive", "")):
                        return False
                excl_toks = {c_tok, "be", "feel", "get"}
                if tokens[0] in ("be", "feel", "get") and len(tokens) >= 2 and c_tok != tokens[1]:
                    excl_toks.add(tokens[1])
                    excl_toks.update(cls._headword_candidates(tokens[1]))
                    excl_toks.add(cls._surface_lemma(tokens[1]))
                other_content = [t for t in (lem_tokens + tokens) if t not in excl_toks and len(t) >= 3 and t not in stop_func]
                other_toks = other_content if other_content else [t for t in (lem_tokens + tokens) if t not in excl_toks and len(t) >= 2]
                if other_toks:
                    pats = []
                    for p in (s_obj.get("patterns") or []):
                        pats.extend(v.lower() for v in cls._slash_variants(str(p)))
                    for pb in (s_obj.get("pattern_bundles") or []):
                        if isinstance(pb, dict) and pb.get("pattern"):
                            pats.extend(v.lower() for v in cls._slash_variants(str(pb["pattern"])))
                    other_tok_set = set(other_toks)
                    for pat_str in pats:
                        if "hospital" in pat_str and quote and not any(hw in q_low for hw in ("hospital", "patient", "ward", "doctor", "ill", "sick", "injured", "emergency")):
                            continue
                        pat_words = re.findall(r"[a-z]+", pat_str)
                        pat_word_set = set(pat_words) | {cls._surface_lemma(w) for w in pat_words}
                        if all(ot in pat_word_set for ot in other_tok_set):
                            # Guard against compound 'out of' pattern matching bare 'of' when 'out' is absent
                            if "of" in other_tok_set and "out" not in tokens and "out" in pat_words:
                                non_out_of = [i for i, w in enumerate(pat_words) if w == "of" and (i == 0 or pat_words[i - 1] != "out")]
                                if not non_out_of:
                                    continue
                            # Enforce token order when the expression starts with a preposition before the head noun/adj ('at first', 'in turn')
                            if len(tokens) >= 2 and tokens[0] in other_tok_set and c_tok == tokens[-1] and tokens[0] != c_tok:
                                p0_pos = [i for i, w in enumerate(pat_words) if w == tokens[0]]
                                hd_pos = [i for i, w in enumerate(pat_words) if w == c_tok or cls._surface_lemma(w) == c_tok]
                                if p0_pos and hd_pos and min(p0_pos) > max(hd_pos):
                                    continue
                            if is_passive_frame and len(other_tok_set) == 1:
                                p_prep = next(iter(other_tok_set))
                                head_forms = {c_tok, tokens[1]} | set(cls._headword_candidates(tokens[1]))
                                head_pos = [i for i, w in enumerate(pat_words) if w in head_forms]
                                prep_pos = [i for i, w in enumerate(pat_words) if w == p_prep]
                                if head_pos and prep_pos and max(prep_pos) < min(head_pos):
                                    continue
                                # Active-only speech/gerund patterns ('admit to somebody', 'admit to doing sth')
                                # do not passivize into 'be <V-ed> to [organization/place]'
                                pat_str_flat = re.sub(r"[()\[\]]+", " ", pat_str)
                                if re.search(rf"\b(?:{'|'.join(re.escape(hf) for hf in head_forms)})\s+{re.escape(p_prep)}\s+(?:doing\s+|somebody\b|sb\b)", pat_str_flat):
                                    continue
                            return True
                        # Directional preposition/particle open-class pattern (e.g. 'move away/out/to/towards etc' covers 'move into', 'turn into/onto/down etc' covers 'turn around')
                        if not other_content and len(other_tok_set) == 1 and next(iter(other_tok_set)) in ("into", "onto", "towards", "toward", "across", "along", "through", "around", "round", "back", "away", "out", "up", "down"):
                            if "etc" in pat_words and any(dp in pat_words for dp in ("to", "towards", "toward", "away", "out", "in", "into", "onto", "down", "up", "around", "round", "left", "right")):
                                return True
                        # Trivalent split-frame verbs ('equate (sth) to (sth)' matching transitive sense 'equate something with something')
                        if is_trivalent_prep_frame and len(other_tok_set) == 1 and next(iter(other_tok_set)) in {"to", "with", "into", "from", "of", "for", "on", "upon", "as", "against"}:
                            s_gram = f"{s_obj.get('grammar') or ''} {s_obj.get('syntax_codes') or ''}".lower()
                            if "transitive" in s_gram and "intransitive" not in s_gram and any(obj_w in pat_words for obj_w in ("something", "somebody", "sth", "sb")):
                                return True
                    if is_trivalent_prep_frame and not pats:
                        s_gram = f"{s_obj.get('grammar') or ''} {s_obj.get('syntax_codes') or ''}".lower()
                        if "transitive" in s_gram and "intransitive" not in s_gram and cls._pos_code(s_obj.get("pos")) == "v":
                            return True
                    if not other_content and s_obj.get("examples"):
                        uk_list = [uk for uk in unit_keys if len(uk.split()) >= 2]
                        for uk in list(uk_list):
                            if uk.startswith(("be ", "feel ", "get ")) and len(uk.split()) >= 3:
                                uk_no_cop = " ".join(uk.split()[1:])
                                if uk_no_cop not in uk_list:
                                    uk_list.append(uk_no_cop)
                                p1 = uk_no_cop.split()[0]
                                if p1.endswith("led") and not p1.endswith("lled"):
                                    uk_br = p1[:-3] + "lled " + " ".join(uk_no_cop.split()[1:])
                                    if uk_br not in uk_list:
                                        uk_list.append(uk_br)
                        if any(cls._unit_example(uk, list(s_obj.get("examples") or [])) for uk in uk_list):
                            return True
            return False

        # Pass 3A-1: Strictly attested multi-word `units` across all senses of each constituent token
        for c_tok in content_toks:
            c_entry = cls.get_pedlex_entry(c_tok)
            if not c_entry:
                continue
            for s_i, s in enumerate(c_entry.get("senses") or []):
                if is_trivalent_prep_frame:
                    s_gram = f"{s.get('grammar') or ''} {s.get('syntax_codes') or ''}".lower()
                    if "no passive" in s_gram or ("intransitive" in s_gram and "transitive" not in s_gram.replace("intransitive", "")):
                        continue
                is_match, matched_unit_ex = _sense_matches_unit_strictly(s)
                if not is_match:
                    continue
                raw_df = cls.format_ldoce_definition(s)
                if not raw_df:
                    continue
                df = cls._compose_collocation_definition(u_key, locked_expr_type, c_tok, s, raw_df, quote) or raw_df
                cand_exs = list(matched_unit_ex or s.get('examples', []) or [])
                for arb_e in cls._arbitrated_example_pool(c_tok, s, u_key):
                    if arb_e and arb_e not in cand_exs:
                        cand_exs.append(arb_e)
                m_ex = (
                    cls._unit_example(u_key, cand_exs, allow_modifier=False)
                    or cls._unit_example(u_key, cand_exs, allow_modifier=True)
                    or cls._relaxed_unit_example(u_key, cand_exs)
                )
                if m_ex:
                    cand_exs = [m_ex] + [e for e in cand_exs if e != m_ex]
                _add_cand(df, locked_expr_type, f"{c_tok} #{s_i}", cand_exs, s, True)

        # Pass 3A-2: Headword `patterns` / `pattern_bundles` match when no multi-word unit/phrase entry exists,
        # or when a verb + preposition frame (e.g. 'remind of') has competing main-entry pattern senses alongside phrasal-verb blocks.
        has_second_content = any(t not in head_candidates for t in content_toks) if tokens else False
        if not candidates or ((is_verb_expr or is_passive_frame) and not has_second_content and not re.search(r"\boneself\b", expr_word, re.I) and len(candidates) < k):
            pat_matched_senses = []
            for c_tok in content_toks:
                c_entry = cls.get_pedlex_entry(c_tok)
                if not c_entry:
                    continue
                for s_i, s in enumerate(c_entry.get("senses") or []):
                    if not _sense_matches_head_pattern(c_tok, s):
                        continue
                    raw_df = cls.format_ldoce_definition(s)
                    if not raw_df:
                        continue
                    # For passive frames ('be admitted to', 'be featured in', 'be measured in'), prioritize senses with
                    # explicit passive patterns ('be <V-ed> <prep>') or transitive object-before-prep patterns ('<V> sb/sth <prep>')
                    rank_prio = 0
                    all_p_strs = [str(p).lower() for p in (s.get("patterns") or [])] + [
                        str(pb.get("pattern") or "").lower() for pb in (s.get("pattern_bundles") or []) if isinstance(pb, dict)
                    ]
                    if is_passive_frame:
                        p_word = tokens[-1]
                        v_stems_1 = [c for c in ([cls._surface_lemma(tokens[1])] + cls._headword_candidates(tokens[1])) if c and c != tokens[1] and cls.get_ldoce_entry(c)]
                        v_stem_1 = v_stems_1[0] if v_stems_1 else cls._surface_lemma(tokens[1])
                        if any(tokens[1] in ps and p_word in ps for ps in all_p_strs):
                            rank_prio = 2
                        elif any(re.search(rf"\b{re.escape(v_stem_1)}\s+(?:somebody|something|sb|sth)\b.*\b{re.escape(p_word)}\b", ps) for ps in all_p_strs):
                            rank_prio = 1
                    elif len(tokens) == 2 and tokens[0] in ("make", "keep", "find", "leave", "remain", "stay", "fall") and c_tok == tokens[1]:
                        rank_prio = 2
                    elif any(
                        cls._unit_key(ps) in unit_keys
                        or any(_unit_key_matches(ps, uk, allow_prefix=True) for uk in unit_keys)
                        for ps in all_p_strs
                    ):
                        rank_prio = 1
                    if not is_passive_frame and any(cls._unit_example(uk, list(s.get("examples") or [])) for uk in unit_keys if len(uk.split()) >= 2):
                        rank_prio = max(rank_prio, 2)
                    pat_matched_senses.append((rank_prio, c_tok, s_i, s, raw_df))
            if not candidates and any(item[0] >= 1 for item in pat_matched_senses):
                pat_matched_senses = [item for item in pat_matched_senses if item[0] >= 1]
            if len(pat_matched_senses) >= 2 and quote:
                mock_s = [item[3] for item in pat_matched_senses]
                res_pm = cls._lock_sense({"word": pat_matched_senses[0][1], "senses": mock_s}, quote=quote, target_pos="verb" if is_passive_frame else None, return_scores=True)
                sc_map = dict(res_pm[4]) if (res_pm and len(res_pm) >= 5 and res_pm[4]) else {}
                pat_matched_senses.sort(key=lambda x: (-x[0], -sc_map.get(mock_s.index(x[3]), 0.0), x[2]))
            else:
                pat_matched_senses.sort(key=lambda x: (-x[0], x[2]))
            for _prio, c_tok, s_i, s, raw_df in pat_matched_senses:
                df = cls._compose_collocation_definition(u_key, locked_expr_type, c_tok, s, raw_df, quote) or raw_df
                cand_exs = list(s.get('examples', []) or [])
                for u_obj in (s.get('units') or []):
                    if isinstance(u_obj, dict) and u_obj.get('examples'):
                        for u_e in u_obj['examples']:
                            if u_e and u_e not in cand_exs:
                                cand_exs.append(u_e)
                for pb in (s.get('pattern_bundles') or []):
                    if isinstance(pb, dict) and pb.get('example') and pb['example'] not in cand_exs:
                        cand_exs.append(pb['example'])
                m_ex = (
                    cls._unit_example(u_key, cand_exs, allow_modifier=False)
                    or cls._unit_example(u_key, cand_exs, allow_modifier=True)
                    or cls._relaxed_unit_example(u_key, cand_exs)
                )
                if m_ex:
                    cand_exs = [m_ex] + [e for e in cand_exs if e != m_ex]
                _add_cand(df, locked_expr_type, f"{c_tok} #{s_i}", cand_exs, s, True)

        # Pass 3B: WordNet (oewn:2025+) Multi-Word Synsets & Unit-Example-Attested Synsets ONLY when PedLex is disabled
        if cls._is_pedlex_disabled() and not candidates and len(tokens) >= 2:
            try:
                wn = cls.get_wordnet()
                if wn:
                    for mw_q in (clean, " ".join(lem_tokens), u_key):
                        if not mw_q or len(mw_q.split()) < 2:
                            continue
                        for syn in (wn.synsets(mw_q) or wn.synsets(mw_q.replace(" ", "_")) or []):
                            s_def = (syn.definition() or "").strip()
                            if not s_def:
                                continue
                            s_pos = getattr(syn, "pos", lambda: "")() if callable(getattr(syn, "pos", None)) else getattr(syn, "pos", "")
                            if s_pos == "v" and not s_def.lower().startswith("to "):
                                s_def = f"to {s_def}"
                            s_exs = [str(e).strip().strip('"“”') for e in (syn.examples() or []) if e]
                            _add_cand(s_def, locked_expr_type, f"wn:{mw_q}", s_exs, {"definition": s_def, "examples": s_exs}, True)
                    if not candidates:
                        for c_tok in [t for t in content_toks if t not in LIGHT_OR_MODAL_TOKS]:
                            for syn in (wn.synsets(c_tok) or []):
                                s_def = (syn.definition() or "").strip()
                                s_exs = [str(e).strip().strip('"“”') for e in (syn.examples() or []) if e]
                                if not s_def or not s_exs:
                                    continue
                                m_wn_ex = cls._unit_example(u_key, s_exs)
                                if m_wn_ex:
                                    s_pos = getattr(syn, "pos", lambda: "")() if callable(getattr(syn, "pos", None)) else getattr(syn, "pos", "")
                                    mock_wn_s = {"definition": s_def, "pos": s_pos, "examples": [m_wn_ex] + [e for e in s_exs if e != m_wn_ex]}
                                    comp_def = cls._compose_collocation_definition(u_key, locked_expr_type, c_tok, mock_wn_s, s_def, quote) or s_def
                                    _add_cand(comp_def, locked_expr_type, f"wn:{c_tok}", mock_wn_s["examples"], mock_wn_s, True)
            except Exception:
                pass

        # Pass 3C: Constituent single-word fallback ONLY if NO phrase-attested candidates were found above
        is_verb_slot_frame = bool(re.search(r"[\[\(](?:sb|sth|oneself|one's)[\]\)]", expr_word, re.IGNORECASE)) and len(tokens) >= 1
        if not candidates and (is_verb_slot_frame or cls.is_attested_phrase(clean)):
            non_light = [t for t in content_toks if t not in LIGHT_OR_MODAL_TOKS] or content_toks
            # For nominal collocations (modifier + head noun, e.g. 'new generation', 'passenger plane'),
            # prioritize the syntactic head noun (rightmost token) over pre-nominal modifiers.
            if not is_verb_slot_frame and locked_expr_type == "collocation" and len(non_light) >= 2:
                non_light = list(reversed(non_light))
            expect_verb = is_verb_slot_frame and tokens and non_light and non_light[0] in (tokens[0], lem_0)
            for c_tok in non_light:
                c_entry = cls.get_pedlex_entry(c_tok)
                if c_entry:
                    for s_i, s in enumerate(c_entry.get("senses") or []):
                        if expect_verb and c_tok in (tokens[0], lem_0) and cls._pos_code(s.get("pos")) != "v":
                            continue
                        # Skip unrelated multi-word unit senses (e.g. 'call somebody names', 'fall victim/prey to')
                        s_units = s.get("units") or []
                        if s_units:
                            is_m, _ = _sense_matches_unit_strictly(s)
                            if not is_m:
                                continue
                        raw_df = cls.format_ldoce_definition(s)
                        if not raw_df:
                            continue
                        df = cls._compose_collocation_definition(u_key, locked_expr_type, c_tok, s, raw_df, quote) or raw_df
                        if (
                            not is_verb_slot_frame
                            and locked_expr_type == "collocation"
                            and len(tokens) == 2
                            and c_tok in (tokens[-1], cls._surface_lemma(tokens[-1]))
                            and cls._pos_code(s.get("pos")) == "n"
                            and df == raw_df
                        ):
                            df = f"{clean}: {raw_df}"
                        _add_cand(df, locked_expr_type, f"{c_tok} #{s_i}", s.get('examples', []), s, False)
                        if len(candidates) >= k - 2:
                            break
                # Also include WordNet verb synsets for verb-slot expressions ONLY when PedLex is disabled
                if cls._is_pedlex_disabled() and expect_verb and c_tok in (tokens[0], lem_0) and len(candidates) < k:
                    try:
                        wn = cls.get_wordnet()
                        if wn:
                            q_words = set(re.findall(r"[a-z]{3,}", q_low))
                            wn_syns = wn.synsets(c_tok, pos="v") or []
                            scored_v_syns = []
                            for syn in wn_syns:
                                s_def = (syn.definition() or "").strip()
                                if not s_def:
                                    continue
                                s_exs = [str(e).strip().strip('"“”') for e in (syn.examples() or []) if e]
                                s_bag = set(re.findall(r"[a-z]{3,}", (s_def + " " + " ".join(s_exs)).lower()))
                                ov = len(q_words & s_bag)
                                if ("out of" in clean or "from" in clean) and any(w in (s_def + " " + " ".join(s_exs)).lower() for w in ("condition", "rank", "poverty", "state", "from")):
                                    ov += 3
                                scored_v_syns.append((ov, s_def, s_exs))
                            scored_v_syns.sort(key=lambda x: -x[0])
                            for _ov, s_def, s_exs in scored_v_syns[:4]:
                                if not s_def.lower().startswith("to "):
                                    s_def = f"to {s_def}"
                                _add_cand(s_def, locked_expr_type, f"wn:{c_tok}", s_exs, {"definition": s_def, "examples": s_exs}, False)
                                if len(candidates) >= k:
                                    break
                    except Exception:
                        pass
                if len(candidates) >= k:
                    break

        # Symbolic Post-Pruning for Reflexive & Pronoun-Bound Expression Frames:
        # 1. Reflexive-bound expressions ('reverse (oneself)', 'devote (oneself) to', 'speak for themselves'):
        #    When the expression or quote explicitly binds a reflexive pronoun, lock to candidate sense(s)
        #    whose `units` or `patterns` explicitly specify 'yourself' / 'oneself' / 'itself' / 'themselves'.
        if len(candidates) > 1 and (
            re.search(r"\b(?:oneself|myself|yourself|himself|herself|itself|ourselves|yourselves|themselves)\b", expr_word, re.I)
            or (lem_0 and re.search(rf"\b{re.escape(lem_0)}[a-z]*\s+(?:myself|yourself|himself|herself|itself|ourselves|yourselves|themselves|oneself)\b", q_low))
        ):
            refl_cands = []
            for c_item in candidates:
                rs = c_item.get("raw_sense") if isinstance(c_item.get("raw_sense"), dict) else {}
                u_strs = [str(u.get("phrase") if isinstance(u, dict) else u).lower() for u in (rs.get("units") or [])]
                p_strs = [str(p).lower() for p in (rs.get("patterns") or [])] + [
                    str(pb.get("pattern") or "").lower() for pb in (rs.get("pattern_bundles") or []) if isinstance(pb, dict)
                ]
                if any(any(rk in up for rk in ("yourself", "oneself", "itself", "themselves")) for up in (u_strs + p_strs)):
                    refl_cands.append(c_item)
            if refl_cands:
                candidates = refl_cands

        # 2. Intransitive/pronoun-bound '<verb> <prep> it' phrasal frame ('kept at it' vs 'keep sb at sth'):
        if len(candidates) > 1 and len(tokens) == 2 and lem_0 and re.search(
            rf"\b(?:{re.escape(tokens[0])}|{re.escape(lem_0)})[a-z]*\s+{re.escape(tokens[1])}\s+it\b", q_low
        ):
            target_it_pat = f"{lem_0} {tokens[1]} it"
            it_cands = []
            for c_item in candidates:
                rs = c_item.get("raw_sense") if isinstance(c_item.get("raw_sense"), dict) else {}
                u_strs = [str(u.get("phrase") if isinstance(u, dict) else u).lower() for u in (rs.get("units") or [])]
                p_strs = [str(p).lower() for p in (rs.get("patterns") or [])] + [
                    str(pb.get("pattern") or "").lower() for pb in (rs.get("pattern_bundles") or []) if isinstance(pb, dict)
                ]
                if any(target_it_pat in up for up in (u_strs + p_strs)):
                    it_cands.append(c_item)
            if it_cands:
                candidates = it_cands

        # 3. Selectional Restriction ([sb] vs [sth]) Post-Pruning:
        #    When the expression formula explicitly declares [sb]/somebody (without [sth]/something),
        #    or declares [sth]/something (without [sb]/somebody), and competing candidates contrast
        #    somebody-only vs something-only patterns/units (e.g. 'in touch (with somebody)' vs
        #    'be/keep/stay etc in touch (with something)'), prune the mismatched selectional slot.
        if len(candidates) > 1:
            has_sb_slot = bool(re.search(r"[\[\(]?\b(?:sb|somebody|someone)\b[\]\)]?", expr_word, re.I))
            has_sth_slot = bool(re.search(r"[\[\(]?\b(?:sth|something)\b[\]\)]?", expr_word, re.I))
            if has_sb_slot != has_sth_slot:
                def _cand_slot_flags(c_item: Dict[str, Any]) -> Tuple[bool, bool]:
                    rs = c_item.get("raw_sense") if isinstance(c_item.get("raw_sense"), dict) else {}
                    up_strs = [str(u.get("phrase") if isinstance(u, dict) else u).lower() for u in (rs.get("units") or [])] + [
                        str(p).lower() for p in (rs.get("patterns") or [])
                    ] + [
                        str(pb.get("pattern") or "").lower() for pb in (rs.get("pattern_bundles") or []) if isinstance(pb, dict)
                    ]
                    c_sb = any(re.search(r"\b(?:sb|somebody|someone)\b", up) for up in up_strs)
                    c_sth = any(re.search(r"\b(?:sth|something)\b", up) for up in up_strs)
                    return c_sb, c_sth

                if has_sb_slot and any(_cand_slot_flags(c)[0] and not _cand_slot_flags(c)[1] for c in candidates):
                    candidates = [c for c in candidates if not (_cand_slot_flags(c)[1] and not _cand_slot_flags(c)[0])]
                elif has_sth_slot and any(_cand_slot_flags(c)[1] and not _cand_slot_flags(c)[0] for c in candidates):
                    candidates = [c for c in candidates if not (_cand_slot_flags(c)[0] and not _cand_slot_flags(c)[1])]

        return candidates[:k]



    @classmethod
    def extract_deterministic_expressions(
        cls,
        text: str,
        target_count: int = 5,
        syllabus_expressions: Optional[List[str]] = None,
        enable_llm_disambiguation: bool = True,
        exclude_words: Optional[Set[str]] = None,
        exclude_definitions: Optional[Set[str]] = None,
        unit_name: str = ""
    ) -> List[Dict[str, Any]]:
        """
        Extracts genuine multi-word academic expressions and collocations from text
        in 100% deterministic fashion (0 Tokens).
        Combines mine_expression_skeletons with get_expression_definition_and_example.
        Returns a list of dicts conforming to VocabularyItem / ExpressionItem:
            {
                "word": canonical formula (e.g. 'shut [sb/sth] down'),
                "part_of_speech": 'collocation' | 'phrasal verb' | ...,
                "definition": authoritative PedLex / WordNet definition,
                "quoted_sentence": verbatim passage sentence,
                "example_usage": authentic PedLex example or passage sentence,
                "design_audit": "AUDIT: [S-ID] -> [Formula] -> [VERBATIM_CONFIRMED]"
            }
        """
        with cls.wsd_log_session(unit_name or "expressions"):
            sent_pool = cls.tokenize_and_index_sentences(text)[1] if text else {}
            skeletons = cls.mine_expression_skeletons(
                text,
                target_count=max(target_count, len(syllabus_expressions)) if syllabus_expressions else target_count + 8,
                syllabus_expressions=syllabus_expressions
            )
            for sk in skeletons:
                sk["_is_syllabus"] = bool(syllabus_expressions)
            if syllabus_expressions:
                seen_ph = {re.sub(r'\[.*?\]|\(.*?\)', '', s.get("phrase", "")).strip().lower() for s in skeletons}
                reserve_exprs = cls.mine_expression_skeletons(
                    text,
                    target_count=target_count + 10,
                    syllabus_expressions=None
                )
                for rs in reserve_exprs:
                    rph = re.sub(r'\[.*?\]|\(.*?\)', '', rs.get("phrase", "")).strip().lower()
                    if rph and rph not in seen_ph:
                        seen_ph.add(rph)
                        rs["_is_syllabus"] = False
                        skeletons.append(rs)
            if not skeletons:
                return []

            excl_w_low = {str(w).strip().lower() for w in (exclude_words or set()) if w}
            excl_d_norm = {re.sub(r"[^a-z0-9]+", " ", str(d).lower()).strip() for d in (exclude_definitions or set()) if d}
            _expr_func_stop = {
                "a", "an", "the", "to", "of", "in", "on", "at", "by", "for", "with", "from", "into", "onto", "upon",
                "over", "under", "up", "down", "off", "away", "out", "back", "about", "around", "through", "along",
                "across", "against", "between", "among", "be", "have", "do", "get", "make", "take", "go", "come"
            }

            attested_sources = (
                "phrase_row", "unit_sense", "headword_entry", "phrasal_verb_block",
                "sense_pattern", "collocation", "crossref", "ldoce_entry", "wordnet"
            )
            results: List[Dict[str, Any]] = []
            seen_formulas: Set[str] = set()
            for s in skeletons:
                if len(results) >= target_count:
                    break
                phrase = s.get("phrase", "")
                formula = s.get("pattern_formula", phrase)
                norm_form = re.sub(r'\[.*?\]|\(.*?\)', '', formula).strip().lower()
                if not norm_form or norm_form in seen_formulas or not cls.is_pedagogical_expression(norm_form) or not cls.is_pedagogical_expression(formula):
                    continue
                is_syllabus_item = bool(s.get("_is_syllabus", False))
                if not is_syllabus_item and any(
                    cls.are_same_word_family(norm_form, sf) or cls.are_same_word_family(formula, r["word"])
                    for sf, r in zip(seen_formulas, results)
                ):
                    continue
                if not is_syllabus_item and any(cls.are_same_word_family(norm_form, sf) for sf in seen_formulas):
                    continue
                # Prevent verb+preposition expressions (e.g. 'conform to [sth/sb]') from duplicating
                # a single-word headword ('conform') already extracted in the same unit.
                if excl_w_low:
                    if not is_syllabus_item and any(
                        cls.are_same_word_family(norm_form, ew) or cls.are_same_word_family(formula, ew)
                        for ew in excl_w_low
                    ):
                        continue
                    f_content_toks = [t for t in norm_form.split() if t not in _expr_func_stop]
                    if len(f_content_toks) == 1 and (
                        f_content_toks[0] in excl_w_low
                        or cls._surface_lemma(f_content_toks[0]) in excl_w_low
                    ):
                        continue
                sid = s.get("sid", "S-1")
                quote = s.get("quote", "")
                expr_type = cls.classify_expression_type(formula, quote) or cls.classify_expression_type(phrase, quote) or s.get("type", "collocation")

                discourse_ctx = ""
                if sid and sent_pool:
                    m_sid_num = re.match(r"S-(\d+)", str(sid))
                    if m_sid_num:
                        s_num = int(m_sid_num.group(1))
                        prev_s = str(sent_pool.get(f"S-{s_num - 1}") or "").strip()
                        next_s = str(sent_pool.get(f"S-{s_num + 1}") or "").strip()
                        if prev_s or next_s:
                            discourse_ctx = " ".join(p for p in (prev_s, f">> {quote} <<", next_s) if p)

                # Primary: Ground multi-word expressions in authoritative PedLex expression evidence tiers
                # (Disable inner LLM disambiguation here since Gate 1 below arbitrates across the full unified pool)
                defn, ex, def_source = cls.expression_definition_evidence(
                    phrase=phrase,
                    expr_type=expr_type,
                    context_sentence=quote,
                    enable_llm_disambiguation=False
                )

                # Gate 1: Neuro-Symbolic Arbitration across Top-K (k=6) Expression Candidates
                locked_expr_sense = None
                if quote:
                    pool = cls.build_top_k_expression_candidates(formula, quote, k=6, expr_type=expr_type)
                    has_attested_pool = any(c.get("attested") for c in pool)
                    is_verb_slot_frame = bool(re.search(r"[\[\(](?:sb|sth|oneself|one's)[\]\)]", formula, re.IGNORECASE))
                    if not has_attested_pool and def_source not in attested_sources and not is_verb_slot_frame:
                        continue
                    if defn and (not has_attested_pool or def_source in attested_sources):
                        norm_defn = re.sub(r'[^a-z0-9]+', ' ', defn.lower()).strip()
                        if not any(re.sub(r'[^a-z0-9]+', ' ', c.get("definition", "").lower()).strip() == norm_defn for c in pool):
                            pool.insert(0, {
                                "definition": defn,
                                "pos": expr_type,
                                "examples": [ex] if ex else [],
                                "score": 30.0,
                                "margin": 0.0,
                                "attested": def_source in attested_sources,
                            })
                    pool = pool[:6]
                    if pool:
                        chosen = None
                        if len(pool) > 1 and enable_llm_disambiguation:
                            chosen_idx = cls.disambiguate_via_llm(formula, quote, pool, discourse_context=discourse_ctx)
                            if 0 <= chosen_idx < len(pool):
                                chosen = pool[chosen_idx]
                            else:
                                logger.info(
                                    f"  [DROP_WSD_REJECTED] Expression '{formula}' rejected by WSD arbitration in [{sid}], advancing to next candidate."
                                )
                                continue
                        elif has_attested_pool or not defn or def_source not in attested_sources:
                            chosen = pool[0]
                        if chosen is not None:
                            prev_defn_norm = re.sub(r"[^a-z0-9]+", " ", (defn or "").lower()).strip()
                            chosen_defn_norm = re.sub(r"[^a-z0-9]+", " ", (chosen.get("definition") or "").lower()).strip()
                            sense_changed = bool(chosen_defn_norm and chosen_defn_norm != prev_defn_norm)
                            if chosen.get("definition"):
                                defn = chosen["definition"]
                                def_source = def_source or "phrase_row"
                            locked_expr_sense = chosen.get("raw_sense")
                            if sense_changed:
                                ex = ""
                            if chosen.get("examples"):
                                ex = (
                                    cls._unit_example(formula, chosen["examples"], allow_modifier=False)
                                    or cls._unit_example(formula, chosen["examples"], allow_modifier=True)
                                    or cls._relaxed_unit_example(formula, chosen["examples"])
                                    or (ex if ex and cls._unit_example(formula, [ex], allow_modifier=True) else chosen["examples"][0])
                                )
                        elif locked_expr_sense is None:
                            locked_expr_sense = pool[0].get("raw_sense")

                low_confidence = not def_source
                ex = cls.ensure_distinct_example(formula, expr_type, ex, quote, sense=locked_expr_sense)

                # Gate 2 (Evidence-Completeness Gate - 宁缺勿滥):
                # Drop any multi-word expression lacking an authoritative definition, distinct corpus example,
                # or mapped to a non-pedagogical cardinal number definition.
                if not defn or cls.is_boilerplate_definition(defn) or not ex or not cls.is_pedagogical_expression(formula, defn):
                    continue
                defn_norm_key = re.sub(r"[^a-z0-9]+", " ", defn.lower()).strip()
                if defn_norm_key in excl_d_norm:
                    continue

                seen_formulas.add(norm_form)
                excl_d_norm.add(defn_norm_key)
                results.append({
                    "word": formula,
                    "part_of_speech": expr_type,
                    "definition": defn,
                    "quoted_sentence": quote,
                    "example_usage": ex,
                    "definition_source": def_source,
                    "low_confidence": low_confidence,
                    "design_audit": f"AUDIT: [{sid}] -> [{formula}] -> [VERBATIM_CONFIRMED]"
                })

            return results

