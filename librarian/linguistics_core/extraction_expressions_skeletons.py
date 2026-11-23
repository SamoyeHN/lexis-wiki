"""
Deterministic multi-word expression and collocation skeleton mining (ACL + spaCy).
"""

import re
from typing import Any, Dict, List, Optional, Set, Tuple


class ExtractionExpressionsSkeletonsMixin:
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

