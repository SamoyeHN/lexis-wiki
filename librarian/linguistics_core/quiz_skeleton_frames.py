"""
Authentic example selection across 5 LDOCE compartments and micro-task prompt blueprint assembly.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from .quiz_skeleton_context import CXN_ANCHOR_TYPES, SkeletonContext


class QuizSkeletonFramesMixin:
    @classmethod
    def _assemble_skeleton_micro_task(
        cls,
        ctx: SkeletonContext,
        used_distractors: Set[str],
        batch_target_words: Set[str],
        nlp: Any,
    ) -> Optional[Dict[str, Any]]:
        """
        Stage 4: Authentic Example Selection & Micro-Task Assembly.
        Pre-selects single best authentic example across 5 LDOCE compartments,
        derives cloze slot frames, synthesizes SLA distractors notes, and produces
        the pre-computed generative blueprint.
        """
        w = ctx.w
        w_lower = ctx.w_lower
        raw_w_item = ctx.raw_w_item
        raw_pos = ctx.raw_pos
        canonical_pos = ctx.canonical_pos
        it = ctx.it
        final_target = ctx.final_target
        verb_form_tag = ctx.verb_form_tag
        verb_requires_object = ctx.verb_requires_object
        anchor = ctx.anchor
        anchor_type = ctx.anchor_type
        anchor_source = ctx.anchor_source
        prep_obj = ctx.prep_obj
        quote_governed_prep = ctx.quote_governed_prep
        quote_shown_prep = ctx.quote_shown_prep
        quote_shown_partitive = ctx.quote_shown_partitive
        quote_provenance = ctx.quote_provenance
        quote_wordcount = ctx.quote_wordcount
        quote_adj_is_attributive = ctx.quote_adj_is_attributive
        quote_is_transitive = ctx.quote_is_transitive
        target_inflections_set = ctx.target_inflections_set
        locked_sense = ctx.locked_sense
        t_s_idx = ctx.t_s_idx
        target_entry = ctx.target_entry
        initial_atomic_bundle = ctx.initial_atomic_bundle
        atomic_bundle = ctx.atomic_bundle
        best_pattern = ctx.best_pattern
        ldoce_pattern_example = ctx.ldoce_pattern_example
        ldoce_cloze_frame = ctx.ldoce_cloze_frame
        pattern_anchor = ctx.pattern_anchor
        cxn_override_pat = ctx.cxn_override_pat
        cxn_override_frame = ctx.cxn_override_frame
        causative_extraposition_pat = ctx.causative_extraposition_pat
        causative_extraposition_ex = ctx.causative_extraposition_ex
        is_idiom_slot = ctx.is_idiom_slot
        is_plural_noun = ctx.is_plural_noun
        is_pedlex_plural = ctx.is_pedlex_plural
        is_quantifier_frame = ctx.is_quantifier_frame
        quant_head_noun = ctx.quant_head_noun
        half_slice_target = ctx.half_slice_target
        half_slice_distractors = ctx.half_slice_distractors
        half_slice_frame = ctx.half_slice_frame
        half_slice_meta = ctx.half_slice_meta
        half_slice_explanation = ctx.half_slice_explanation
        distractors = ctx.distractors
        dist_meta = ctx.dist_meta
        distractors3 = ctx.distractors3
        slot_target = ctx.slot_target
        slot = ctx.slot
        raw_options = ctx.raw_options
        inflection_desc = ctx.inflection_desc
        final_options = ctx.final_options
        correct_answer_index = ctx.correct_answer_index
        dist_list = ctx.dist_list
        dist_str = ctx.dist_str
        skeleton_definition = ctx.skeleton_definition
        anchor_downgrade = ctx.anchor_downgrade
        orig_disc_prep = ctx.orig_disc_prep
        attested_adj_preps = ctx.attested_adj_preps
        has_multiple_candidates = ctx.has_multiple_candidates
        _cxn_anchor_types = CXN_ANCHOR_TYPES

        def _clear_and_refill_distractors(
            curr_distractors: List[str],
            curr_meta: Dict[str, str],
            extra_exclude: Optional[Set[str]] = None,
        ) -> Tuple[List[str], Dict[str, str]]:
            return cls._clear_and_refill_distractors(ctx, curr_distractors, curr_meta, used_distractors, extra_exclude=extra_exclude)

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
        # If final_target is a multi-word expression (e.g. 'date back to' / 'date back', 'talk of') sharing a PedLex block
        # with a co-variant ('date from', 'talk ... out of'), ensure ldoce_pattern_example actually instantiates final_target
        # without inserting an unrelated particle/preposition.
        if ldoce_pattern_example and " " in final_target:
            ft_toks = [
                t for t in re.findall(r"[a-z]+", final_target.lower())
                if t not in {"s", "sb", "sth", "somebody", "something", "one", "ones", "oneself", "your", "his", "her", "their", "my", "our", "its"}
            ]
            if len(ft_toks) >= 2:
                def _mw_ex_matches(cand_s: str) -> bool:
                    c_low = (cand_s or "").lower()
                    req_parts = ft_toks
                    if (
                        len(ft_toks) == 3
                        and ft_toks[0] in ("be", "get", "feel", "am", "is", "are", "was", "were", "been", "being")
                        and ft_toks[2] in ("with", "in", "to", "of", "on", "for", "from", "into", "at", "by", "about", "as", "against", "under", "over")
                    ):
                        if not cls.text_contains_form(cand_s, ft_toks[0]):
                            _v_lem_p = cls._surface_lemma(ft_toks[1]) or cls.lemma_of(ft_toks[1]) or ft_toks[1]
                            req_parts = [_v_lem_p, ft_toks[2]]
                    if not all(cls.text_contains_form(cand_s, tk) for tk in req_parts):
                        return False
                    c_toks = re.findall(r"[a-z]+", c_low)
                    f_set = {req_parts[0]} | set(cls.inflected_forms(req_parts[0]))
                    l_set = {req_parts[-1]} | set(cls.inflected_forms(req_parts[-1]))
                    _extra_p = {
                        "out", "in", "up", "down", "off", "on", "away", "back", "over", "into",
                        "from", "with", "for", "to", "of", "at", "by", "about", "against", "through", "around", "aside", "apart"
                    } - set(req_parts)
                    for i_f, tk_f in enumerate(c_toks):
                        if tk_f in f_set:
                            for i_l in range(i_f + 1, min(len(c_toks), i_f + len(req_parts) + 4)):
                                if c_toks[i_l] in l_set:
                                    if not ((set(c_toks[i_f + 1 : i_l]) - set(req_parts)) & _extra_p):
                                        return True
                    return False

                if not _mw_ex_matches(ldoce_pattern_example):
                    replacement_ex = None
                    upstream_ex_cand = (it.get("example_usage") or "").strip().strip('"').strip()
                    if (
                        upstream_ex_cand
                        and upstream_ex_cand.lower() != (it.get("quote") or "").strip().lower()
                        and _mw_ex_matches(upstream_ex_cand)
                    ):
                        replacement_ex = upstream_ex_cand
                    if not replacement_ex:
                        for raw_ex in ((locked_sense or {}).get("examples") or []):
                            ex_txt = (raw_ex.get("text") if isinstance(raw_ex, dict) else str(raw_ex or "")).strip()
                            if ex_txt and ex_txt.lower() != (it.get("quote") or "").strip().lower():
                                if _mw_ex_matches(ex_txt):
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
                                if _mw_ex_matches(ch):
                                    replacement_ex = ch
                                    break
                    ldoce_pattern_example = replacement_ex

        # Pre-compute syntactic slot flags (copular/attributive adjective & noun countability)
        # so Five-Source Optimal Example Pre-Selection and Direct Example-to-Frame Slicing honor them.
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

        # 5.3 Five-Source Optimal Example Pre-Selection (GEMINI.md §3.2):
        # Evaluate pure 5 LDOCE compartments (grammar_boxes, pattern_bundles & sense_examples,
        # LDOCE Thesaurus, Language Activator, curated Collocations/cross_collocations) to select
        # the single best, sense-isolated, complete authentic sentence before deriving cloze_slot_frame.
        # (Oxford Thesaurus and raw BNC fragments permanently excluded).
        winning_example_source = "none"
        _shell_noun_anchors = ("shell_to_do", "shell_of_doing", "shell_that")
        _verb_cxn_anchors = ("sb_to_do", "to_inf", "gerund")
        if not causative_extraposition_pat and (
            (not cxn_override_pat and anchor_type not in _cxn_anchor_types)
            or (canonical_pos == "noun" and anchor_type in _shell_noun_anchors)
            or (canonical_pos == "verb" and anchor_type in _verb_cxn_anchors)
        ):
            opt_ex, opt_pat, winning_example_source = cls.select_optimal_five_source_example(
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
            if not cxn_override_pat and anchor_type not in _cxn_anchor_types:
                best_pattern = opt_pat
            elif opt_pat and not best_pattern:
                best_pattern = opt_pat

        # Synchronize `modified_noun` anchor to the syntactic partner in `ldoce_pattern_example` when `anchor`
        # is absent from the winning example AND came from:
        #   (1) a generic light noun in the quote ('diplomatic move'),
        #   (2) an unattested dictionary fallback absent from the quote ('outdated' -> 'method'),
        #   (3) a non-attributive quote fallback ('analogous' -> 'sponsorship'), or
        #   (4) a sense whose winning example instantiates a bound prepositional frame ('silent on').
        if (
            canonical_pos == "adj"
            and anchor_type == "modified_noun"
            and anchor
            and ldoce_pattern_example
            and not re.search(rf"\b{re.escape(str(anchor).lower())}\w*\b", ldoce_pattern_example.lower())
        ):
            try:
                _sync_doc = nlp(ldoce_pattern_example)
                _, _, _sync_tok = cls._find_target_span_in_doc(
                    _sync_doc,
                    final_target.lower(),
                    base_str=str((target_entry or {}).get("word") or w_lower).strip().lower(),
                    t_forms=target_inflections_set,
                )
                if _sync_tok is not None:
                    _ex_partner, _ex_ptype, _ = cls._example_partner(_sync_tok, final_target, "adj")
                    if _ex_ptype == "prep" and _ex_partner not in attested_adj_preps:
                        _ex_partner, _ex_ptype = None, None
                        if _sync_tok.head and _sync_tok.head.pos_ in ("AUX", "VERB"):
                            for _ch in _sync_tok.head.children:
                                if _ch.dep_ in ("nsubj", "nsubjpass") and _ch.pos_ in ("NOUN", "PROPN"):
                                    _c_lem = (_ch.lemma_ or _ch.text).lower()
                                    if len(_c_lem) >= 3 and _c_lem not in cls._ANCHOR_STOPWORDS and _c_lem != final_target.lower():
                                        _ex_partner, _ex_ptype = _c_lem, "modified_noun"
                                        break
                    _should_sync_adj_anchor = (
                        not quote_adj_is_attributive
                        or _ex_ptype == "prep"
                        or str(anchor).strip().lower() in {"move", "thing", "one", "ones", "way", "part", "place", "time", "point", "case", "type", "kind", "sort"}
                        or not re.search(rf"\b{re.escape(str(anchor).lower())}\w*\b", str(it.get("quote") or "").lower())
                    )
                    if (
                        _should_sync_adj_anchor
                        and _ex_partner
                        and _ex_ptype in ("modified_noun", "prep", "verb_copula")
                        and not cls._in_forbidden_word_set(_ex_partner, batch_target_words)
                    ):
                        anchor = _ex_partner
                        anchor_type = _ex_ptype
                        if _ex_ptype == "prep" and not best_pattern:
                            best_pattern = f"{final_target.lower()} {_ex_partner}"
                        if any(
                            cls.double_key_collision(final_target, d, anchor=anchor, anchor_type=anchor_type, pos="adj")[0]
                            for d in dist_list
                        ):
                            distractors, dist_meta = _clear_and_refill_distractors(distractors, dist_meta)
                            distractors3 = distractors[:3]
                            raw_options = distractors3[:slot] + [slot_target] + distractors3[slot:]
                            used_distractors.update(distractors3)
                            final_options = raw_options
                            dist_list = [opt for opt in final_options if opt.lower() != final_target.lower()]
                            dist_str = ", ".join(dist_list)
            except Exception:
                pass

        # Synchronize adverb `anchor` & `anchor_type` to the syntactic partner in `ldoce_pattern_example`
        # whenever the winning example instantiates a concrete modified verb/adjective different from (or filling in) `anchor`.
        if (
            canonical_pos == "adv"
            and anchor_type != "discourse"
            and ldoce_pattern_example
            and (
                not anchor
                or not re.search(rf"\b{re.escape(str(anchor).lower())}\w*\b", ldoce_pattern_example.lower())
            )
        ):
            try:
                _sync_adv_doc = nlp(ldoce_pattern_example)
                _, _, _sync_adv_tok = cls._find_target_span_in_doc(
                    _sync_adv_doc,
                    final_target.lower(),
                    base_str=final_target.lower(),
                    t_forms={final_target.lower()},
                )
                if _sync_adv_tok is not None:
                    _adv_partner, _adv_ptype, _ = cls._example_partner(_sync_adv_tok, final_target, "adv")
                    if (
                        _adv_partner
                        and _adv_ptype in ("modifies_adj", "modifies_verb")
                        and not cls._in_forbidden_word_set(_adv_partner, batch_target_words)
                    ):
                        anchor = _adv_partner
                        anchor_type = _adv_ptype
            except Exception:
                pass

        # Deep Passive & Transitive Phrasal-Verb Object Anchor Synchronization:
        # When a verbal phrase or transitive verb has no anchor yet (or only a contextual quote fallback absent from the example)
        # and `ldoce_pattern_example` instantiates a concrete direct object (`dobj`) or passive Patient (`nsubjpass`),
        # project that Patient noun as `anchor` (`anchor_type = "object"`) so the LLM prompt locks onto the concrete Patient ('受动者').
        if (
            canonical_pos in ("verb", "phrase")
            and anchor_type not in _cxn_anchor_types
            and ldoce_pattern_example
            and (
                not anchor
                or anchor_type in ("contextual", "collocation")
                or (
                    anchor_type in ("prep", "object")
                    and not re.search(rf"\b{re.escape(str(anchor).lower())}\w*\b", ldoce_pattern_example.lower())
                )
            )
        ):
            try:
                _sync_v_doc = nlp(ldoce_pattern_example)
                _head_v_tok = None
                if " " in final_target.lower():
                    _ft_words = [t for t in re.findall(r"[a-z]+", final_target.lower()) if t not in ("sb", "sth", "somebody", "something", "one", "ones")]
                    if _ft_words:
                        _f_forms = {_ft_words[0]} | set(cls.inflected_forms(_ft_words[0]))
                        _l_w = _ft_words[-1]
                        for _tk in _sync_v_doc:
                            if (_tk.text.lower() in _f_forms or _tk.lemma_.lower() == _ft_words[0]) and _tk.pos_ in ("VERB", "AUX"):
                                if any(
                                    _sync_v_doc[_j].text.lower() == _l_w or _sync_v_doc[_j].lemma_.lower() == _l_w
                                    for _j in range(_tk.i + 1, min(len(_sync_v_doc), _tk.i + len(_ft_words) + 4))
                                ):
                                    _head_v_tok = _tk
                                    break
                else:
                    _, _, _head_v_tok = cls._find_target_span_in_doc(
                        _sync_v_doc,
                        final_target.lower(),
                        base_str=str((target_entry or {}).get("word") or w_lower).strip().lower(),
                        t_forms=target_inflections_set,
                    )
                if _head_v_tok is not None and _head_v_tok.pos_ in ("VERB", "AUX"):
                    _is_v_pass = any(c.dep_ in ("auxpass", "nsubjpass") for c in _head_v_tok.children)
                    _pat_obj_tok = next(
                        (
                            c for c in _head_v_tok.children
                            if (c.dep_ in ("dobj", "obj") or (_is_v_pass and c.dep_ == "nsubjpass"))
                            and c.pos_ in ("NOUN", "PROPN")
                            and c.ent_type_ != "PERSON"
                            and c.lemma_.lower() not in cls._ANCHOR_STOPWORDS
                            and not cls._in_forbidden_word_set(c.lemma_.lower(), batch_target_words)
                        ),
                        None,
                    )
                    if _pat_obj_tok is not None:
                        anchor = _pat_obj_tok.text.lower() if _pat_obj_tok.tag_ == "NNS" else _pat_obj_tok.lemma_.lower()
                        anchor_type = "object"
            except Exception:
                pass

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
                        re_ex = re_bundle.get("example")
                        if re_ex and (
                            cls._is_complete_finite_sentence(re_ex)
                            or not cls._is_complete_finite_sentence(ldoce_pattern_example)
                        ):
                            best_pattern = re_bundle["pattern"]
                            ldoce_pattern_example = re_ex
                        elif ldoce_pattern_example and cls._example_attests_prep_valency(
                            ldoce_pattern_example, final_target, anchor.lower(), prep_dir="any"
                        ):
                            best_pattern = re_bundle["pattern"]
                        else:
                            best_pattern = None
                    else:
                        best_pattern = None
                elif ldoce_pattern_example:
                    if canonical_pos == "noun":
                        _p_dir = "pre" if any(p_for_prep_sync.startswith(pp + " ") for pp in pat_preps_sync) else "post"
                        if not any(
                            cls._example_attests_prep_valency(ldoce_pattern_example, final_target, pp, prep_dir=_p_dir)
                            for pp in pat_preps_sync
                        ):
                            best_pattern = None
                    elif not any(re.search(rf"\b{re.escape(pp)}\b", ldoce_pattern_example.lower()) for pp in pat_preps_sync):
                        best_pattern = None

        # Derive Authentic Cloze Sentence Frame Prototype if an authentic example is available
        if ldoce_pattern_example:
            try:
                ex_doc = nlp(ldoce_pattern_example)
                ex_s_i, ex_e_i, _ = cls._find_target_span_in_doc(
                    ex_doc,
                    final_target.lower(),
                    base_str=str((target_entry or {}).get("word") or w_lower).strip().lower(),
                    t_forms=target_inflections_set,
                )
                if ex_s_i is not None and ex_e_i is not None:
                    tok_texts = (
                        [t.text for t in ex_doc[:ex_s_i]]
                        + ["____"]
                        + [t.text for t in ex_doc[ex_e_i:]]
                    )
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
        eff_pos = "noun" if is_quantifier_frame else canonical_pos
        has_double_key_collision = (
            not half_slice_target
            and canonical_pos != "function_word"
            and any(
                opt.lower() != final_target.lower() and cls.double_key_collision(
                    final_target, opt, anchor, anchor_type, pos=eff_pos, quote=it.get("quote")
                )[0]
                for opt in final_options
            )
        )
        if has_double_key_collision:
            bad_bases = {
                b for b, f in zip(_base_options, final_options)
                if f.lower() != final_target.lower() and cls.double_key_collision(
                    final_target, f, anchor, anchor_type, pos=eff_pos, quote=it.get("quote")
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
        if final_options != _base_options or has_double_key_collision:
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
                if verb_form_tag == "VBN":
                    best_pattern = ""
                else:
                    obj_raw = m_sep_pv.group(2).lower()
                    obj_tag = "[sb/sth]" if ("somebody" in obj_raw and "something" in obj_raw) or "/" in obj_raw else ("[sb]" if obj_raw in ("somebody", "sb") else "[sth]")
                    best_pattern = f"{final_target} {obj_tag}"
            best_pattern = re.sub(r"\s+", " ", best_pattern).strip()

        # Synthesize explicit object-slot pattern for transitive/prepositional phrasal verbs
        # (e.g. 'catch up with' -> 'catch up with [sb/sth]') when PedLex stores the object slot in phrasal_headword or raw_w_item
        if not best_pattern and canonical_pos == "phrase" and verb_form_tag != "VBN":
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
        patient_ontology_info = cls._extract_verb_patient_ontology(
            word=w_lower,
            canonical_pos=canonical_pos,
            target_entry=target_entry,
            locked_sense=locked_sense,
            definition=str(it.get("definition") or ""),
            authentic_example=ldoce_pattern_example or it.get("quote") or it.get("raw_quote") or "",
            quote=str(it.get("quote") or it.get("raw_quote") or ""),
            forbidden_words=batch_target_words,
        )
        patient_ontology_slot = (patient_ontology_info or {}).get("slot_label")
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
        elif half_slice_frame:
            cloze_slot_frame = half_slice_frame
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
            if _is_cxn_family:
                p_clean = re.sub(r"\bdoing\s+something\b", "[doing sth]", p_clean, flags=re.IGNORECASE)
                p_clean = re.sub(r"\bto\s+do\s+something\b", "to do [sth]", p_clean, flags=re.IGNORECASE)
                p_clean = re.sub(r"\bsomebody\b", "[sb]", p_clean, flags=re.IGNORECASE)
                p_clean = re.sub(r"\bsomething\b", "[sth]", p_clean, flags=re.IGNORECASE)
            else:
                p_clean = re.sub(r"\bdoing\s+(?:something|\[sth\])\b", "doing something", p_clean, flags=re.IGNORECASE)
                p_clean = re.sub(r"\bto\s+do\s+(?:something|\[sth\])\b", "to do something", p_clean, flags=re.IGNORECASE)
                p_clean = re.sub(r"\bsomebody\b", "someone", p_clean, flags=re.IGNORECASE)
                p_clean = re.sub(r"\bsomething\b", "something", p_clean, flags=re.IGNORECASE)
                p_clean = re.sub(r"\[sb\]", "someone", p_clean, flags=re.IGNORECASE)
                p_clean = re.sub(r"\[sth\]", "something", p_clean, flags=re.IGNORECASE)
                p_clean = re.sub(r"\[doing\s+sth\]", "doing something", p_clean, flags=re.IGNORECASE)
                p_clean = re.sub(r"\bto\s+\[verb\]", "to do something", p_clean, flags=re.IGNORECASE)
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
                elif (
                    not any(final_target.lower().endswith(" " + prep) for prep in ("on", "in", "at", "for", "with", "about", "under", "from", "to", "by", "of", "into"))
                    and any(p_clean.lower().endswith(" " + prep) for prep in ("on", "in", "at", "for", "with", "about", "under", "from", "to", "by", "of", "into"))
                ):
                    m_end_prep = re.search(r"\b(on|in|at|for|with|about|under|from|to|by|of|into)$", p_clean.lower())
                    if m_end_prep:
                        cloze_slot_frame = f"... ____ {m_end_prep.group(1)}"

        if cloze_slot_frame and len(re.findall(r"_{2,}", cloze_slot_frame)) > 1:
            cloze_slot_frame = None
        if canonical_pos == "noun" and anchor and anchor_type == "prep" and ldoce_pattern_example:
            _anc_in_ex = cls._example_attests_prep_valency(
                ldoce_pattern_example, final_target, str(anchor).strip().lower(), prep_dir="any"
            )
        else:
            _anc_in_ex = bool(
                anchor
                and (
                    not ldoce_pattern_example
                    or re.search(rf"\b{re.escape(str(anchor).strip().lower())}\b", str(ldoce_pattern_example).lower())
                )
            )
        if not cloze_slot_frame and anchor and anchor_type == "prep" and canonical_pos in ("verb", "adj") and _anc_in_ex:
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
        if not _is_cxn_family and not is_quantifier_frame and not half_slice_frame and anchor_type != "partitive":
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
                patient_ontology_slot=patient_ontology_slot,
            )
            if sliced_frame:
                _sliced_is_bare = bool(
                    re.match(
                        r"^\.\.\.\s*(?:a|an|the|each|every|his|her|their|its|our|my|your|to|someone|somebody|he|she|they)?\s*_{2,}\s*$",
                        sliced_frame.strip(),
                        re.IGNORECASE,
                    )
                )
                if not cloze_slot_frame or _is_bare_det_frame:
                    cloze_slot_frame = sliced_frame
                elif not _sliced_is_bare:
                    # Prefer `sliced_frame` unless `cloze_slot_frame` preserves a required `anchor` (or intervening trivalent object) that is attested in the example
                    _anc_low = str(anchor or "").strip().lower()
                    _pat_has_anc = bool(_anc_low and _anc_in_ex and re.search(rf"\b{re.escape(_anc_low)}\b", cloze_slot_frame.lower()))
                    _sliced_has_anc = bool(_anc_low and re.search(rf"\b{re.escape(_anc_low)}\b", sliced_frame.lower()))
                    _pat_has_mid_obj = bool(
                        orig_disc_prep
                        and re.search(r"_{2,}\s+(?:someone|something|oneself|\[sb\]|\[sth\]|\[sb/sth\])\s+[a-z]+", cloze_slot_frame.lower())
                        and not re.search(r"_{2,}\s+(?:someone|something|oneself|\[sb\]|\[sth\]|\[sb/sth\])\s+[a-z]+", sliced_frame.lower())
                    )
                    if not (_pat_has_anc and not _sliced_has_anc) and not _pat_has_mid_obj:
                        cloze_slot_frame = sliced_frame

        # Guaranteed POS-grounded fallback if no example/quote token matched
        if not cloze_slot_frame:
            if canonical_pos == "adj":
                if is_copula_adj:
                    cloze_slot_frame = f"... be ____ {anchor}" if (anchor and anchor_type == "prep" and _anc_in_ex) else "... be ____"
                elif anchor and anchor_type == "modified_noun":
                    cloze_slot_frame = f"... the ____ {anchor}"
                elif anchor and anchor_type == "prep" and _anc_in_ex:
                    cloze_slot_frame = f"... ____ {anchor}"
                else:
                    cloze_slot_frame = "... the ____ one"
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
            elif canonical_pos == "adv":
                if anchor_type == "discourse":
                    cloze_slot_frame = "____, ..."
                elif anchor and anchor_type == "modifies_adj":
                    cloze_slot_frame = f"... ____ {anchor}"
                elif anchor and anchor_type == "modifies_verb":
                    cloze_slot_frame = f"... {anchor} ____"
                else:
                    cloze_slot_frame = "... act ____"
            else:
                cloze_slot_frame = f"... ____ {resolved_obj_anim}"

        _frame_has_copula = bool(
            cloze_slot_frame
            and re.search(
                r"\b(?:be|am|is|are|was|were|been|being|become|became|fall|fell|fallen|remain|remained|stay|stayed|seem|seemed|appear|appeared|look|looked|sound|sounded|feel|felt|get|got|turn|turned|grow|grew|keep|kept)\s+(?:[a-z]+\s+)?_{2,}",
                cloze_slot_frame.lower(),
            )
        )
        _frame_has_post_prep = bool(
            cloze_slot_frame
            and re.search(
                r"_{2,}\s+(?:to|in|of|with|for|from|at|on|about|into|by|as|towards?|against)\b",
                cloze_slot_frame.lower(),
            )
        )
        if canonical_pos == "adj" and (_frame_has_copula or _frame_has_post_prep):
            is_copula_adj = True
            is_attributive_only_adj = False
        elif not is_copula_adj and not is_attributive_only_adj and canonical_pos == "adj" and (cloze_slot_frame or anchor):
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
        _frame_has_vbn_aux = bool(
            cloze_slot_frame
            and re.search(
                r"\b(?:is|are|was|were|been|has|have|had|be|get|got|getting)\s+(?:[a-z]+\s+)?_{2,}",
                cloze_slot_frame.lower(),
            )
        )
        is_base_verb_slot = (
            (inflection_desc or "base form") == "base form"
            and verb_form_tag in (None, "VB")
            and not is_plural_noun
            and not is_be_led_phrase
            and not _frame_has_vbn_aux
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
            if verb_form_tag == "VBN":
                non_base_verb_slot_hint = (
                    f" (Past participle verb slot required for '{final_target}': place 'has/have/had' or "
                    "passive 'is/are/was/were' immediately before '____')."
                )
            elif verb_form_tag == "VBD":
                non_base_verb_slot_hint = (
                    f" (Simple past-tense verb slot required for '{final_target}': place '____' directly after the subject "
                    "as a simple past verb — do not put 'have/has/had' or 'to' before '____')."
                )
            elif verb_form_tag == "VBZ":
                non_base_verb_slot_hint = (
                    f" (Present 3rd-person singular '-s' verb slot required for '{final_target}': place '____' directly after "
                    "a singular subject — do not put any modal or 'to' before '____')."
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

        if anchor_downgrade == "sense_recognition":
            cloze_slot_frame = None
            stem_core = "Write a natural sentence matching the Definition above."
        elif cloze_slot_frame:
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
            if m_adj_noun and m_adj_noun.group(1).lower() in _NON_VERB_FRAME_HEADS:
                m_adj_noun = None
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
                _frame_already_has_vbn_aux = bool(
                    verb_form_tag == "VBN"
                    and re.search(r"\b(?:has|have|had|is|are|was|were|been|be|got|get)\b", cloze_slot_frame.lower())
                )
                _frame_already_has_vbg_gov = bool(
                    verb_form_tag == "VBG"
                    and re.search(
                        r"\b(?:of|in|on|at|for|from|by|with|without|about|after|before|into|upon|is|are|was|were|been|be)\b",
                        cloze_slot_frame.lower(),
                    )
                )
                if is_quantifier_frame and quant_head_noun:
                    stem_core += f" (Quantifier head noun slot: blank only '{quant_head_noun}' inside '{w_lower}')."
                elif half_slice_target:
                    stem_core += f" (Collocation slot: blank only '{half_slice_target}' inside expression '{w_lower}')."
                elif is_plural_noun and not re.search(r"\b(?:several|many|few|fewer|two|three|four|these|those|various)\s+_{2,}", cloze_slot_frame.lower()):
                    stem_core += f" (Plural noun slot required: e.g. '{final_target}')."
                elif is_copula_adj and not _frame_has_copula:
                    stem_core += " (Complete predicate required: stem MUST include the linking verb so the adjective predicate is complete; NEVER place before a noun)."
                elif is_attributive_only_adj and not m_adj_noun:
                    stem_core += " (Attributive pre-noun slot required: place '____' directly before a modified noun; NEVER use after a linking verb)."
                elif is_be_led_phrase and not re.search(r"\b(?:to|will|would|can|could|may|might|should|must)\s+_{2,}", cloze_slot_frame.lower()):
                    stem_core += be_led_slot_hint
                elif is_base_verb_slot:
                    _has_left_aux_in_frame = bool(
                        re.search(
                            r"\b(?:to|will|would|can|could|may|might|should|must|do|does|did|let|make|made|help|helped)\s+(?:[a-z]+\s+)?_{2,}",
                            cloze_slot_frame.lower(),
                        )
                    )
                    if ("one's" in final_target.lower() or "oneself" in final_target.lower()) and not _has_left_aux_in_frame:
                        stem_core += (
                            " (Base-form verbal idiom that already includes 'one's': place '____' after 'to' or a modal verb)."
                        )
                    elif not _has_left_aux_in_frame:
                        stem_core += " (Base-form verb slot: use after a modal verb, infinitive 'to', or a plural/1st-person subject — never after singular 'he/she/it' or 'ended up')."
                elif non_base_verb_slot_hint and not _frame_already_has_vbn_aux and not _frame_already_has_vbg_gov:
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
                    _stripped_base = w_lower[len(_pfx):].lstrip("-")
                    _base_lem = re.sub(r"(?:ed|ing|s|es|ly|able|ible)$", "", _stripped_base).lstrip("-")
                    if len(_base_lem) < 4:
                        _base_lem = _stripped_base
                    _is_hyphenated = w_lower.startswith(f"{_pfx}-")
                    _is_valid_base = bool(cls.get_ldoce_entry(_base_lem)) or _is_hyphenated
                    if _is_valid_base and _base_lem not in {"duce", "ceive", "quire", "late", "peat", "sult", "fuse", "turn", "veal", "member", "lieve"}:
                        if _base_lem == _stripped_base:
                            stem_core += f" (Do NOT use root cognates like '{_stripped_base}' anywhere in the stem)."
                        else:
                            stem_core += f" (Do NOT use root cognates like '{_base_lem}' or '{_stripped_base}' anywhere in the stem)."
                    break

        task_parts = [stem_core]

        # Sense Guard for non-primary polysemous senses when the slot frame lacks strong disambiguating content cues
        if canonical_pos in ("noun", "verb", "adj") and " " not in w_lower and skeleton_definition:
            _sg_ent = cls.get_ldoce_entry(w_lower)
            _sg_senses = (_sg_ent or {}).get("senses") or []
            if len(_sg_senses) > 1:
                _sg_s0 = re.sub(r"^(?:\[[^\]]+\]\s*)+", "", str(_sg_senses[0].get("definition") or "")).strip()
                _sg_locked = re.sub(r"^(?:\[[^\]]+\]\s*)+", "", str(skeleton_definition or "")).strip()
                _sg_s0_low = _sg_s0.lower()
                _sg_locked_low = _sg_locked.lower()
                if _sg_s0_low and _sg_locked_low and _sg_locked_low != _sg_s0_low and _sg_s0_low not in _sg_locked_low and _sg_locked_low not in _sg_s0_low:
                    _sg_stop = {
                        "the", "a", "an", "to", "in", "on", "at", "of", "for", "with",
                        "by", "from", "as", "is", "was", "are", "were", "be", "been", "being",
                        "something", "someone", "somebody", "that", "this", "it", "they", "verb",
                    }
                    _sg_content = [
                        t for t in re.findall(r"[a-zA-Z]+", cloze_slot_frame or "")
                        if t.lower() not in _sg_stop
                    ]
                    if len(_sg_content) <= 1:
                        _s0_short = _sg_s0[:65].rstrip(" ,;.")
                        _lk_short = _sg_locked[:65].rstrip(" ,;.")
                        task_parts.append(
                            f"(Sense Guard: lock to '{_lk_short}' — do NOT use primary sense '{_s0_short}')."
                        )

        # Data-Driven Grammatical & Usage Discriminator: only emit in Micro-Task when it carries
        # a concrete grammatical slot constraint (`[...]`) or a sense-locked `Grammar Box:` rule
        target_base = quant_head_noun if is_quantifier_frame else (half_slice_target if half_slice_target else w_lower)
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
            # Strip generic non-grammatical Contrast labels ('polar opposite', 'orthographic look-alike', 'different degree/scale')
            filtered_disc = re.sub(
                r"Contrast:\s*vs\s*'[^']+'\s*\((?:polar opposite of target meaning|orthographic look-alike / paronym with distinct meaning|different degree/scale or collocational frame|polar opposite)\)\.\s*",
                "",
                discriminator_note,
            ).strip()
            if filtered_disc:
                if verb_form_tag == "VB" and "Grammar Box:" in filtered_disc:
                    _gb_tail = filtered_disc.split("Grammar Box:", 1)[1].lower()
                    for _vb_t in ("VBZ", "VBD", "VBG", "VBN"):
                        _vb_infl = (cls._inflect_verb(w_lower, _vb_t) or "").lower()
                        if _vb_infl and _vb_infl != w_lower and re.search(rf"\b{re.escape(_vb_infl)}\b", _gb_tail):
                            filtered_disc = filtered_disc.replace(
                                "Grammar Box:",
                                f"Grammar Box (Pattern only — keep base form '{w_lower}' in your stem):",
                                1,
                            )
                            break
                task_parts.append(filtered_disc)

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

        # Gate 3 (Evidence & Distractor Completeness Gate): drop unanchored items whose distractors
        # degraded 100% to random corpus/domain fallback when other candidates exist in the unit.
        if (
            anchor_downgrade == "sense_recognition"
            and dist_meta
            and all(str(m or "").endswith("fallback") for m in dist_meta.values())
            and has_multiple_candidates
        ):
            logging.getLogger(__name__).info(
                f"🚫 Gate 3 (DROP_UNANCHORED_FALLBACK_ITEM): '{final_target}' has no collocational anchor "
                f"and 100% fallback distractors ({list(dist_meta.keys())}) — dropped."
            )
            return None


        return {
            "target_word": final_target,
            "base_headword": w,
            "part_of_speech": canonical_pos if not half_slice_target else "phrase_slot",
            "is_multi_word": bool(canonical_pos == "phrase" or " " in w_lower),
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
            "explanation": half_slice_explanation or "",
            # Quote provenance tiering: 'candidate_quote' is the raw curriculum quote and
            # is display-only when 'quote_provenance' == 'weak'; only 'licensed_quote' was
            # strong enough to license sense-locking, anchors and valency evidence.
            "candidate_quote": it.get("raw_quote") or it.get("quote") or "",
            "licensed_quote": it.get("quote") or "",
            "quote_provenance": quote_provenance,
            "quote_wordcount": quote_wordcount,
        }
