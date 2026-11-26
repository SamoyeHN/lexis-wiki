"""
Symbolic candidate selection for multi-word expressions and collocations.
"""

import re
from typing import Any, Dict, List, Optional, Tuple


class ExtractionExpressionsCandidatesMixin:
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
        # Realize possessive pronoun slots (e.g. 'do [one's] part', 'take one's time', 'make up one's mind')
        m_poss_slot = re.search(r"\[?(?:one's|someone's|sb's)\]?", expr_word, re.I)
        if m_poss_slot and tokens:
            v_tok = tokens[0]
            n_tok = tokens[-1]
            m_q_poss = re.search(rf"\b({re.escape(v_tok)}\s+(?:my|your|his|her|its|our|their|one's)\s+{re.escape(n_tok)})\b", q_low)
            if m_q_poss:
                realized_poss = cls._unit_key(m_q_poss.group(1))
                if realized_poss and len(realized_poss.split()) >= 2 and realized_poss not in unit_keys:
                    unit_keys.append(realized_poss)
            for gen_poss in (f"{v_tok} one's {n_tok}", f"{v_tok} your {n_tok}", f"{v_tok} our {n_tok}"):
                pk = cls._unit_key(gen_poss)
                if pk and len(pk.split()) >= 2 and pk not in unit_keys:
                    unit_keys.append(pk)
            # If head noun is 'part' with light verb 'do', include standard light-verb / duty alternations
            if n_tok == "part" and v_tok in {"do", "play", "take", "have"}:
                for alt_p in ("play a part", "play part", "take part", "have a part to play", "somebody's part in something"):
                    apk = cls._unit_key(alt_p)
                    if apk and len(apk.split()) >= 2 and apk not in unit_keys:
                        unit_keys.append(apk)
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
            fallback_items = []
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
                                # Retain unit sense if its phrase shares content tokens with quote
                                u_texts = [u.get("phrase") if isinstance(u, dict) else str(u) for u in s_units]
                                if not any(any(cls.text_contains_form(q_low, ut) for ut in re.findall(r"[a-z]{3,}", u_txt.lower()) if ut not in stop_func) for u_txt in u_texts):
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
                        fallback_items.append((c_tok, s_i, s, df))
            if len(fallback_items) >= 2 and quote:
                mock_s = [item[2] for item in fallback_items]
                res_fb = cls._lock_sense({"word": fallback_items[0][0], "senses": mock_s}, quote=quote, return_scores=True)
                sc_fb = dict(res_fb[4]) if (res_fb and len(res_fb) >= 5 and res_fb[4]) else {}
                fallback_items.sort(key=lambda x: (-sc_fb.get(mock_s.index(x[2]), 0.0), x[1]))
            for c_tok, s_i, s, df in fallback_items:
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

