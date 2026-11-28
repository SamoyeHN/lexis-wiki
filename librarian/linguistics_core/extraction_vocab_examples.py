"""
Sense-isolated authentic example selection and validation gate (Gate 2).
"""

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("librarian.linguistics")


class ExtractionVocabExamplesMixin:
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
            if (
                not is_multi
                and len(sense.get("examples") or []) >= 2
                and not exact_pb_exs
                and not matching_pb_exs
                and hasattr(cls, "select_optimal_five_source_example")
            ):
                _ent_opt = cls.get_ldoce_entry(clean_target)
                if _ent_opt:
                    _cpos_opt = "adj" if pos_low.startswith("adj") else ("adv" if pos_low.startswith("adv") else pos_low)
                    _s_idx_opt = next(
                        (i for i, _es in enumerate(_ent_opt.get("senses") or []) if _es is sense or _es.get("definition") == sense.get("definition")),
                        None,
                    )
                    _opt_ex, _, _ = cls.select_optimal_five_source_example(
                        final_target=clean_target,
                        base_word=clean_low,
                        canonical_pos=_cpos_opt,
                        target_entry=_ent_opt,
                        locked_sense=sense,
                        t_s_idx=_s_idx_opt,
                        definition=sense.get("definition"),
                        quote=quoted_sentence,
                    )
                    if _opt_ex and _opt_ex not in cand_pool:
                        cand_pool.append(_opt_ex)
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


