"""
Distractor coordination, collision screening, and option assembly.
Coordinates definition-locked distractors, ensures zero double-key collisions,
and enforces parallel inflection concordance across options.
"""

import logging
import re
import zlib
from typing import Any, Dict, List, Optional, Set, Tuple

from .quiz_skeleton_context import CXN_ANCHOR_TYPES, SkeletonContext


class QuizSkeletonDistractorsMixin:
    @classmethod
    def _is_double_key(cls, ctx: SkeletonContext, cand: str) -> bool:
        """Evaluates whether cand forms a double-key collision with ctx.w_lower / ctx.final_target."""
        canonical_pos = ctx.canonical_pos
        w_lower = ctx.w_lower
        anchor = ctx.anchor
        anchor_type = ctx.anchor_type
        it = ctx.it
        if canonical_pos == "function_word":
            return False
        if ctx.is_quantifier_frame and ctx.quant_head_noun:
            c_low_p = cand.strip().lower()
            return cls.double_key_collision(ctx.quant_head_noun, c_low_p, anchor, anchor_type, pos="noun", quote=it.get("quote"))[0]
        if canonical_pos in ("phrase", "idiom_slot"):
            c_low_p = cand.strip().lower()
            return cls.double_key_collision(w_lower, c_low_p, pos="phrase", quote=it.get("quote"))[0]
        if anchor_type == "v_obj_pp" and ctx.v_obj_pp_tail:
            c_low_v = cand.strip().lower()
            return bool(
                cls.lookup_ldoce_phrase(f"{c_low_v} {ctx.v_obj_pp_tail}")
                or cls.lookup_ldoce_phrase(f"{c_low_v} something {ctx.v_obj_pp_tail}")
            )
        flag, _kind = cls.double_key_collision(
            w_lower, cand, anchor, anchor_type, pos=canonical_pos, quote=it.get("quote")
        )
        if flag:
            return True
        if ctx.orig_disc_prep and ctx.orig_disc_prep != anchor:
            flag_orig_p, _ = cls.double_key_collision(
                w_lower, cand, ctx.orig_disc_prep, "prep", pos=canonical_pos, quote=it.get("quote")
            )
            if flag_orig_p:
                return True
        raw_w_low = str(ctx.raw_w_item or "").strip().lower()
        if raw_w_low and raw_w_low != w_lower:
            flag_raw, _ = cls.double_key_collision(
                raw_w_low, cand, anchor, anchor_type, pos=canonical_pos, quote=it.get("quote")
            )
            if flag_raw:
                return True
        ft_low = str(ctx.final_target or "").strip().lower()
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

    @classmethod
    def _is_valid_distractor(cls, ctx: SkeletonContext, cand: str) -> bool:
        """Validates candidate against double-key collision, paronyms, and slot occupants."""
        if not cand or cls._is_double_key(ctx, cand):
            return False
        w_lower = ctx.w_lower
        canonical_pos = ctx.canonical_pos
        c_low = cand.strip().lower()
        if canonical_pos in ("function_word", "phrase", "idiom_slot"):
            if c_low == w_lower:
                return False
            if canonical_pos == "phrase" and not ctx.is_quantifier_frame and (" " in w_lower) != (" " in c_low):
                if ctx.anchor_type != "correl_pair" and w_lower not in cls._CORREL_FIRST_ARMS:
                    return False
            return True
        if len(w_lower) >= 4 and len(c_low) >= 4:
            if w_lower in c_low or c_low in w_lower:
                if cls.is_cross_ontology_derivational_noun_pair(w_lower, c_low):
                    pass
                elif cls.are_same_word_family(w_lower, c_low) or cls.shares_derivational_root(w_lower, c_low):
                    return False
                else:
                    is_rhyme_containment_paronym = (
                        canonical_pos == "noun"
                        and min(len(w_lower), len(c_low)) >= 5
                        and abs(len(w_lower) - len(c_low)) <= 3
                        and (c_low.endswith(w_lower) or w_lower.endswith(c_low))
                        and cls._phonetic_rhyme_tail(w_lower) == cls._phonetic_rhyme_tail(c_low)
                    )
                    if not is_rhyme_containment_paronym and abs(len(w_lower) - len(c_low)) >= 2:
                        return False
        if (
            ctx.is_plural_noun
            and not cls.can_pluralize_noun_distractor(c_low)
            and c_low not in set(cls._NOUN_COUNTABILITY_CONTRAST_MAP.get(w_lower, []))
        ):
            return False

        if not cls._distractor_occupies_slot(
            cand, canonical_pos, requires_object=(ctx.verb_requires_object is True)
        ):
            return False
        return cls.is_cefr_compliant_distractor(cand, w_lower, passage_level=ctx.passage_cefr)

    @classmethod
    def _clear_and_refill_distractors(
        cls,
        ctx: SkeletonContext,
        curr_distractors: List[str],
        curr_meta: Dict[str, str],
        used_distractors: Set[str],
        extra_exclude: Optional[Set[str]] = None,
    ) -> Tuple[List[str], Dict[str, str]]:
        """Filters colliding distractors and refills pool using semantic field contrasts."""
        w_lower = ctx.w_lower
        canonical_pos = ctx.canonical_pos
        anchor = ctx.anchor
        anchor_type = ctx.anchor_type
        it = ctx.it

        near_s, ants_s = cls.semantic_fields(w_lower, pos=canonical_pos)
        near_only_s = near_s - ants_s
        excl_set = set(used_distractors) | set(extra_exclude or set())
        cleared_list: List[str] = []
        for d in curr_distractors:
            if d in excl_set or not cls._is_valid_distractor(ctx, d):
                continue
            has_d_clash, _ = cls.evaluate_syntactic_slot_clash(
                w_lower, d, anchor=anchor, anchor_type=anchor_type, pos=canonical_pos, quote=it.get("quote")
            )
            if (
                not has_d_clash
                and d.lower() in near_only_s
                and sum(1 for x in cleared_list if x.lower() in near_only_s and curr_meta.get(x) != "antonym") >= 1
            ):
                continue
            cleared_list.append(d)
        r_meta: Dict[str, str] = {}
        if len(cleared_list) < 3:
            try:
                if ctx.is_quantifier_frame and ctx.quant_head_noun:
                    q_siblings_all = [
                        "piece", "bit", "slice", "sheet", "block", "drop",
                        "pair", "pack", "stroke", "portion", "set", "share"
                    ]
                    refill_pool = [s for s in q_siblings_all if s != ctx.quant_head_noun and s not in excl_set and s not in cleared_list]
                    r_meta = {d: "quantifier_collocation_contrast" for d in refill_pool}
                elif canonical_pos == "phrase":
                    refill_pool = cls.generate_phrase_distractors(
                        w_lower,
                        pos=ctx.raw_pos or canonical_pos,
                        count=8,
                        exclude_words=excl_set | set(cleared_list),
                        definition=it.get("definition"),
                        quote=it.get("quote"),
                    )
                    r_meta = {}
                else:
                    refill_pool, r_meta = cls.generate_vocab_distractors(
                        target_word=w_lower, pos=canonical_pos,
                        context_anchor=anchor, anchor_type=anchor_type,
                        target_count=8, exclude_words=excl_set,
                        definition=it.get("definition"), quote=it.get("quote"),
                        return_metadata=True,
                        passage_level=ctx.passage_cefr,
                    )
                    r_meta = r_meta or {}
            except Exception:
                refill_pool = []
            for cand in refill_pool:
                if len(cleared_list) >= 3:
                    break
                if cand and cand not in cleared_list and cand not in excl_set and cls._is_valid_distractor(ctx, cand):
                    has_c_clash, _ = cls.evaluate_syntactic_slot_clash(
                        w_lower, cand, anchor=anchor, anchor_type=anchor_type, pos=canonical_pos, quote=it.get("quote")
                    )
                    if (
                        not has_c_clash
                        and cand.lower() in near_only_s
                        and sum(1 for x in cleared_list if x.lower() in near_only_s and (curr_meta.get(x) or r_meta.get(x)) != "antonym") >= 1
                    ):
                        continue
                    cleared_list.append(cand)
            if len(cleared_list) < 3 and canonical_pos != "phrase":
                pos_char = "v" if canonical_pos == "verb" else ("a" if canonical_pos == "adj" else ("r" if canonical_pos == "adv" else "n"))
                try:
                    for cand in cls.generate_zero_collision_distractors(w_lower, pos=pos_char, count=3):
                        if len(cleared_list) >= 3:
                            break
                        if cand and cand not in cleared_list and cand not in excl_set and cls._is_valid_distractor(ctx, cand):
                            if cand.lower() in near_only_s and sum(1 for x in cleared_list if x.lower() in near_only_s) >= 1:
                                continue
                            cleared_list.append(cand)
                except Exception:
                    pass
            if len(cleared_list) < 3:
                pos_char = "v" if canonical_pos == "verb" else ("a" if canonical_pos == "adj" else ("r" if canonical_pos == "adv" else "n"))
                needed = 3 - len(cleared_list)
                pref_vowel: Optional[bool] = None
                if canonical_pos == "noun" and not w_lower.endswith("s") and w_lower[:1].isalpha():
                    pref_vowel = w_lower[0].lower() in "aeiou"
                fb_cands = cls.get_corpus_fallback_candidates(
                    target_word=w_lower,
                    pos=pos_char,
                    target_count=needed,
                    exclude_words=excl_set | set(cleared_list) | {w_lower},
                    prefer_initial_vowel=pref_vowel,
                    passage_level=ctx.passage_cefr,
                )
                for cand in fb_cands:
                    if len(cleared_list) >= 3:
                        break
                    if cand != w_lower and cand not in cleared_list and cand not in excl_set and cls._is_valid_distractor(ctx, cand):
                        cleared_list.append(cand)
            if len(cleared_list) < 3:
                needed = 3 - len(cleared_list)
                fb_cands = cls.get_corpus_fallback_candidates(
                    target_word=w_lower,
                    pos=pos_char,
                    target_count=needed,
                    exclude_words=excl_set | set(cleared_list) | {w_lower},
                    prefer_initial_vowel=None,
                    passage_level=ctx.passage_cefr,
                )
                for cand in fb_cands:
                    if len(cleared_list) >= 3:
                        break
                    if cand != w_lower and cand not in cleared_list and cand not in excl_set and not cls._is_double_key(ctx, cand):
                        cleared_list.append(cand)
        merged_meta = {
            d: (curr_meta.get(d) or r_meta.get(d) or "corpus_fallback")
            for d in cleared_list
        }
        return cleared_list, merged_meta

    @classmethod
    def _coordinate_skeleton_distractors(
        cls,
        ctx: SkeletonContext,
        used_distractors: Set[str],
        batch_target_words: Set[str],
        nlp: Any,
    ) -> None:
        """
        Stage 3: Generate Zero-Collision Distractors with Definition-Locked WSD,
        Sense Antonyms, and Parallel Inflection Concordance.
        """
        w = ctx.w
        w_lower = ctx.w_lower
        raw_w_item = ctx.raw_w_item
        raw_pos = ctx.raw_pos
        canonical_pos = ctx.canonical_pos
        it = ctx.it
        locked_sense = ctx.locked_sense
        t_s_idx = ctx.t_s_idx
        target_entry = ctx.target_entry
        entry_base = ctx.entry_base
        target_p_code = ctx.target_p_code
        anchor = ctx.anchor
        anchor_type = ctx.anchor_type
        prep_obj = ctx.prep_obj
        orig_disc_prep = ctx.orig_disc_prep
        v_obj_pp_tail = ctx.v_obj_pp_tail
        is_idiom_slot = ctx.is_idiom_slot
        quote_governed_prep = ctx.quote_governed_prep
        quote_adj_is_attributive = ctx.quote_adj_is_attributive
        quote_is_transitive = ctx.quote_is_transitive
        causative_extraposition_pat = ctx.causative_extraposition_pat
        causative_extraposition_ex = ctx.causative_extraposition_ex
        cxn_override_pat = ctx.cxn_override_pat
        initial_atomic_bundle = ctx.initial_atomic_bundle
        correlative_frame = ctx.correlative_frame
        m_caus_adj = ctx.m_caus_adj
        sense_native_prep = ctx.sense_native_prep
        ldoce_pattern_example = ctx.ldoce_pattern_example
        pattern_anchor = ctx.pattern_anchor
        ldoce_cloze_frame = ctx.ldoce_cloze_frame
        skeleton_definition = ctx.skeleton_definition
        atomic_bundle = ctx.atomic_bundle
        best_pattern = ctx.best_pattern
        _cxn_anchor_types = CXN_ANCHOR_TYPES

        def _is_double_key(cand: str) -> bool:
            return cls._is_double_key(ctx, cand)

        def _is_valid_distractor(cand: str) -> bool:
            return cls._is_valid_distractor(ctx, cand)

        def _clear_and_refill_distractors(
            curr_distractors: List[str],
            curr_meta: Dict[str, str],
            extra_exclude: Optional[Set[str]] = None,
        ) -> Tuple[List[str], Dict[str, str]]:
            return cls._clear_and_refill_distractors(ctx, curr_distractors, curr_meta, used_distractors, extra_exclude=extra_exclude)

        is_quantifier_frame = False
        quant_head_noun = None
        half_slice_target = None
        half_slice_distractors = None
        half_slice_frame = None
        half_slice_meta = None
        half_slice_explanation = None
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
                cand_sent = (initial_atomic_bundle or {}).get("example") or ""
                if not cand_sent:
                    p_info = cls.lookup_ldoce_phrase(w_lower)
                    if p_info:
                        hw, s_idx = p_info[0], p_info[1]
                        e = cls.get_ldoce_entry(hw)
                        if e and 0 <= s_idx < len(e.get("senses", [])):
                            s_obj = e["senses"][s_idx]
                            cand_sent = (s_obj.get("examples") or [""])[0]
                if not cand_sent:
                    cand_sent = (it.get("example_usage") or "").strip().strip('"').strip()
                is_correl = (anchor_type == "correl_pair" or correlative_frame or w_lower in cls._CORREL_FIRST_ARMS)
                hs = cls.collocation_half_slice(w_lower, cand_sent, pos=raw_pos) if (cand_sent and not is_correl) else None
                if hs:
                    half_slice_target = hs["target_slot"]
                    half_slice_distractors = hs["distractors"]
                    half_slice_frame = hs.get("cloze_slot_frame")
                    half_slice_meta = {d: f"half_slice_{hs['slot_type']}" for d in hs["distractors"]}
                    half_slice_explanation = hs.get("explanation")
                    distractors = half_slice_distractors
                    dist_meta = half_slice_meta
                else:
                    p_dists = cls.generate_phrase_distractors(
                        w_lower,
                        pos=raw_pos or canonical_pos,
                        count=3,
                        exclude_words=used_distractors,
                        definition=it.get("definition"),
                        quote=it.get("quote"),
                    )
                    distractors = p_dists
                    dist_meta = {d: "phrase_contrast" for d in p_dists}
        elif canonical_pos == "idiom_slot" and w_lower == "granted":
            i_cands = ["accepted", "given", "known", "certain", "true"]
            distractors = [c for c in i_cands if c not in used_distractors][:3]
            dist_meta = {d: "idiom_collocation_contrast" for d in distractors}
        elif anchor_type == "v_obj_pp" and v_obj_pp_tail:
            lv_pool = ["take", "bring", "keep", "make", "give", "leave", "set", "call", "turn", "hold", "draw"]
            distractors = [
                c for c in lv_pool
                if c != w_lower
                and c not in used_distractors
                and not cls.lookup_ldoce_phrase(f"{c} {v_obj_pp_tail}")
                and not cls.lookup_ldoce_phrase(f"{c} something {v_obj_pp_tail}")
            ][:3]
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
                return_metadata=True,
                passage_level=ctx.passage_cefr,
            )

        # Ensure we have valid distractors; fallback to zero_collision if needed
        if len(distractors) < 3 and canonical_pos != "phrase":
            pos_char = "v" if canonical_pos == "verb" else ("a" if canonical_pos == "adj" else ("r" if canonical_pos == "adv" else "n"))
            distractors = cls.generate_zero_collision_distractors(w_lower, pos=pos_char, count=3)

        if canonical_pos == "verb":
            if (
                anchor_type == "prep"
                and anchor
                and not orig_disc_prep
                and cls._verb_has_intransitive_prep_pattern(w_lower, str(anchor).lower())
                and not cls._verb_has_transitive_prep_pattern(w_lower, str(anchor).lower())
            ):
                verb_requires_object = False
            else:
                verb_requires_object = cls._verb_takes_object(w_lower)
        else:
            verb_requires_object = None

        # Determine plural-noun target status before distractor screening so _is_valid_distractor
        # rejects uncountable or invariant-plural noun candidates before they enter the final 3 slots.
        is_plural_noun = False
        plural_target = w_lower
        is_pedlex_plural = False
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
            # or if the locked sense's own examples exclusively attest the singular form and a singular pattern bundle exists.
            if quote_plural and initial_atomic_bundle:
                init_pat = (initial_atomic_bundle.get("pattern") or "").lower()
                init_ex = (initial_atomic_bundle.get("example") or "").lower()
                p_cand = cls.pluralize_noun(w_lower)
                if (
                    init_pat
                    and re.search(rf"\b(?:a|an|of|in|on|by|at|for|to|into|under|without)\s+{re.escape(w_lower)}\b", init_pat)
                    and not re.search(rf"\b{re.escape(p_cand)}\b", init_pat)
                ):
                    if not init_ex or not re.search(rf"\b{re.escape(p_cand)}\b", init_ex):
                        quote_plural = False
                elif locked_sense and (not init_pat or not re.search(rf"\b{re.escape(p_cand)}\b", init_pat)):
                    s_exs = [str(e).lower() for e in (locked_sense.get("examples") or []) if str(e).strip()]
                    if (
                        s_exs
                        and any(re.search(rf"\b{re.escape(w_lower)}\b", e) for e in s_exs)
                        and not any(re.search(rf"\b{re.escape(p_cand)}\b", e) for e in s_exs)
                    ):
                        quote_plural = False

            if is_spacy_plural or is_pedlex_plural or w_lower in plural_tantum or quote_plural:
                is_plural_noun = True
                if (is_spacy_plural or is_pedlex_plural or w_lower in plural_tantum) and w_lower.endswith("s"):
                    plural_target = w_lower
                else:
                    plural_target = cls.pluralize_noun(w_lower)

        def _is_double_key(cand: str) -> bool:
            if canonical_pos == "function_word":
                return False
            if is_quantifier_frame and quant_head_noun:
                c_low_p = cand.strip().lower()
                return cls.double_key_collision(quant_head_noun, c_low_p, anchor, anchor_type, pos="noun", quote=it.get("quote"))[0]
            if canonical_pos in ("phrase", "idiom_slot"):
                c_low_p = cand.strip().lower()
                return cls.double_key_collision(w_lower, c_low_p, pos="phrase", quote=it.get("quote"))[0]
            if anchor_type == "v_obj_pp" and v_obj_pp_tail:
                c_low_v = cand.strip().lower()
                return bool(
                    cls.lookup_ldoce_phrase(f"{c_low_v} {v_obj_pp_tail}")
                    or cls.lookup_ldoce_phrase(f"{c_low_v} something {v_obj_pp_tail}")
                )
            flag, _kind = cls.double_key_collision(
                w_lower, cand, anchor, anchor_type, pos=canonical_pos, quote=it.get("quote")
            )
            if flag:
                return True
            if orig_disc_prep and orig_disc_prep != anchor:
                flag_orig_p, _ = cls.double_key_collision(
                    w_lower, cand, orig_disc_prep, "prep", pos=canonical_pos, quote=it.get("quote")
                )
                if flag_orig_p:
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
            if canonical_pos in ("function_word", "phrase", "idiom_slot"):
                if c_low == w_lower:
                    return False
                if canonical_pos == "phrase" and not is_quantifier_frame and (" " in w_lower) != (" " in c_low):
                    if anchor_type != "correl_pair" and w_lower not in cls._CORREL_FIRST_ARMS:
                        return False
                return True
            if len(w_lower) >= 4 and len(c_low) >= 4:
                if w_lower in c_low or c_low in w_lower:
                    if cls.is_cross_ontology_derivational_noun_pair(w_lower, c_low):
                        pass
                    elif cls.are_same_word_family(w_lower, c_low) or cls.shares_derivational_root(w_lower, c_low):
                        return False
                    else:
                        is_rhyme_containment_paronym = (
                            canonical_pos == "noun"
                            and min(len(w_lower), len(c_low)) >= 5
                            and abs(len(w_lower) - len(c_low)) <= 3
                            and (c_low.endswith(w_lower) or w_lower.endswith(c_low))
                            and cls._phonetic_rhyme_tail(w_lower) == cls._phonetic_rhyme_tail(c_low)
                        )
                        if not is_rhyme_containment_paronym and abs(len(w_lower) - len(c_low)) >= 2:
                            return False
            if (
                is_plural_noun
                and not cls.can_pluralize_noun_distractor(c_low)
                and c_low not in set(cls._NOUN_COUNTABILITY_CONTRAST_MAP.get(w_lower, []))
            ):
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
                has_d_clash, _ = cls.evaluate_syntactic_slot_clash(
                    w_lower, d, anchor=anchor, anchor_type=anchor_type, pos=canonical_pos, quote=it.get("quote")
                )
                if (
                    not has_d_clash
                    and d.lower() in near_only_s
                    and sum(1 for x in cleared_list if x.lower() in near_only_s and curr_meta.get(x) != "antonym") >= 1
                ):
                    continue
                cleared_list.append(d)
            r_meta: Dict[str, str] = {}
            if len(cleared_list) < 3:
                try:
                    if is_quantifier_frame and quant_head_noun:
                        refill_pool = [s for s in q_siblings_all if s != quant_head_noun and s not in excl_set and s not in cleared_list]
                        r_meta = {d: "quantifier_collocation_contrast" for d in refill_pool}
                    elif canonical_pos == "phrase":
                        refill_pool = cls.generate_phrase_distractors(
                            w_lower,
                            pos=raw_pos or canonical_pos,
                            count=8,
                            exclude_words=excl_set | set(cleared_list),
                            definition=it.get("definition"),
                            quote=it.get("quote"),
                        )
                        r_meta = {}
                    else:
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
                        has_c_clash, _ = cls.evaluate_syntactic_slot_clash(
                            w_lower, cand, anchor=anchor, anchor_type=anchor_type, pos=canonical_pos, quote=it.get("quote")
                        )
                        if (
                            not has_c_clash
                            and cand.lower() in near_only_s
                            and sum(1 for x in cleared_list if x.lower() in near_only_s and (curr_meta.get(x) or r_meta.get(x)) != "antonym") >= 1
                        ):
                            continue
                        cleared_list.append(cand)
                if len(cleared_list) < 3 and canonical_pos != "phrase":
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
                    needed = 3 - len(cleared_list)
                    pref_vowel: Optional[bool] = None
                    if canonical_pos == "noun" and not w_lower.endswith("s") and w_lower[:1].isalpha():
                        pref_vowel = w_lower[0].lower() in "aeiou"
                    fb_cands = cls.get_corpus_fallback_candidates(
                        target_word=w_lower,
                        pos=pos_char,
                        target_count=needed,
                        exclude_words=excl_set | set(cleared_list) | {w_lower},
                        prefer_initial_vowel=pref_vowel,
                    )
                    for cand in fb_cands:
                        if len(cleared_list) >= 3:
                            break
                        if cand != w_lower and cand not in cleared_list and cand not in excl_set and _is_valid_distractor(cand):
                            cleared_list.append(cand)
                if len(cleared_list) < 3:
                    needed = 3 - len(cleared_list)
                    fb_cands = cls.get_corpus_fallback_candidates(
                        target_word=w_lower,
                        pos=pos_char,
                        target_count=needed,
                        exclude_words=excl_set | set(cleared_list) | {w_lower},
                        prefer_initial_vowel=None,
                    )
                    for cand in fb_cands:
                        if len(cleared_list) >= 3:
                            break
                        if cand != w_lower and cand not in cleared_list and cand not in excl_set and not _is_double_key(cand):
                            cleared_list.append(cand)
            merged_meta = {
                d: (curr_meta.get(d) or r_meta.get(d) or "corpus_fallback")
                for d in cleared_list
            }
            return cleared_list, merged_meta

        if not half_slice_target:
            distractors, dist_meta = _clear_and_refill_distractors(distractors, dist_meta)

        # Assemble full options (target + 3 distractors)
        distractors3 = distractors[:3]
        slot_target = quant_head_noun if is_quantifier_frame else (half_slice_target if half_slice_target else w_lower)
        slot = zlib.crc32(slot_target.encode("utf-8")) % 4
        raw_options = distractors3[:slot] + [slot_target] + distractors3[slot:]
        correct_answer_index = slot
        used_distractors.update(d for d in distractors3 if d)

        def _pluralize_or_keep_uncountable(d_word: str) -> str:
            if (
                dist_meta.get(d_word) == "countability_contrast_uncountable"
                or d_word in set(cls._NOUN_COUNTABILITY_CONTRAST_MAP.get(w_lower, []))
            ):
                return d_word
            return cls.pluralize_noun(d_word)

        # 4. Plan A: All options and target word maintain parallel inflection
        if is_plural_noun:
            inflection_desc = "plural form (NNS)"
            final_target = plural_target
            inflected_distractors3 = [_pluralize_or_keep_uncountable(d) for d in distractors3]
            inflected_distractors3 = cls.screen_inflected_options(
                inflected_distractors3,
                [_pluralize_or_keep_uncountable(d) for d in distractors[3:]],
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
        elif half_slice_target:
            inflection_desc = f"collocation slot ({half_slice_target})"
            final_target = half_slice_target
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
            elif locked_defn and canonical_pos not in ("phrase", "idiom_slot") and not m_caus_adj:
                skeleton_definition = locked_defn
            else:
                skeleton_definition = ext_defn
        else:
            skeleton_definition = locked_defn or ext_defn
        if skeleton_definition and final_target:
            skeleton_definition = re.sub(
                rf"^\s*{re.escape(final_target)}\s*:\s*",
                "",
                skeleton_definition,
                flags=re.IGNORECASE,
            ).strip()

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
            v_anc = atomic_bundle.get("verb_anchor")
            p_anc = atomic_bundle.get("prep")
            if v_anc and anchor_type not in _cxn_anchor_types:
                if str(v_anc).lower() in cls.GENERIC_COLLOCATION_VERBS and p_anc:
                    anchor = p_anc
                    anchor_type = "prep"
                    pattern_anchor = p_anc
                else:
                    anchor = v_anc
                    anchor_type = "verb"
                    pattern_anchor = v_anc
            elif p_anc and anchor_type not in _cxn_anchor_types:
                pat_prep = p_anc
                anchor_in_bundle_ex = bool(
                    anchor
                    and ldoce_pattern_example
                    and re.search(rf"\b{re.escape(str(anchor).lower())}\w*\b", ldoce_pattern_example.lower())
                )
                if (not anchor or anchor_type in ("collocation", "contextual")) and not (
                    canonical_pos == "adj" and quote_adj_is_attributive and pat_prep != quote_governed_prep
                ):
                    if not any(cls.double_key_collision(final_target, d, anchor=pat_prep, anchor_type="prep", pos=canonical_pos)[0] for d in dist_list):
                        anchor = pat_prep
                        anchor_type = "prep"
                        pattern_anchor = pat_prep
                elif (
                    not anchor_in_bundle_ex
                    and anchor_type in ("adj", "verb", "object", "modified_noun", "verb_subject")
                    and not (
                        canonical_pos == "verb"
                        and (quote_is_transitive or anchor_type in ("object", "verb_subject"))
                        and pat_prep != quote_governed_prep
                    )
                    and not (
                        canonical_pos == "adj"
                        and (quote_adj_is_attributive or anchor_type in ("modified_noun", "verb_copula"))
                        and pat_prep != quote_governed_prep
                    )
                ):
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
            if canonical_pos == "adj" and anchor_type == "it_extrapos" and locked_sense and not ldoce_pattern_example:
                _q_norm_ex = (it.get("quote") or "").strip().lower()
                for _pb in (locked_sense.get("pattern_bundles") or []):
                    if not isinstance(_pb, dict):
                        continue
                    _pb_ex = str(_pb.get("example") or "").strip()
                    if _pb_ex and _pb_ex.lower() != _q_norm_ex and re.search(
                        rf"\bit\s*(?:is|was|seems|appears|'s|’s|would\s+be|will\s+be)\b[^.?!;]*?\b{re.escape(final_target)}\b[^.?!;]*?\b{re.escape(str(anchor or 'to'))}\b",
                        _pb_ex.lower(),
                    ):
                        ldoce_pattern_example = _pb_ex
                        break
                if not ldoce_pattern_example:
                    for _raw_ex in (locked_sense.get("examples") or []):
                        _ex_txt = (_raw_ex.get("text") if isinstance(_raw_ex, dict) else str(_raw_ex or "")).strip()
                        if _ex_txt and _ex_txt.lower() != _q_norm_ex and re.search(
                            rf"\bit\s*(?:is|was|seems|appears|'s|’s|would\s+be|will\s+be)\b[^.?!;]*?\b{re.escape(final_target)}\b[^.?!;]*?\b{re.escape(str(anchor or 'to'))}\b",
                            _ex_txt.lower(),
                        ):
                            ldoce_pattern_example = _ex_txt
                            break

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
        has_atomic_valency_ex = bool(
            atomic_bundle
            and (atomic_bundle.get("inf_valency") or atomic_bundle.get("gerund_valency"))
            and ldoce_pattern_example
        )
        if anchor_type not in _cxn_anchor_types and anchor_type not in ("verb_subject", "discourse") and not has_atomic_valency_ex and (
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
            if ld_anchor and not (
                canonical_pos == "verb"
                and (quote_is_transitive or anchor_type == "object")
                and ld_atype == "prep"
                and ld_anchor != quote_governed_prep
            ) and not (
                canonical_pos == "adj"
                and (quote_adj_is_attributive or anchor_type in ("modified_noun", "verb_copula"))
                and ld_atype == "prep"
                and ld_anchor != quote_governed_prep
            ):
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
                        [_pluralize_or_keep_uncountable(d) for d in distractors3],
                        [_pluralize_or_keep_uncountable(d) for d in distractors[3:]],
                        final_target,
                    )
                    used_distractors.update(inflected_distractors3)
                    final_options = inflected_distractors3[:slot] + [final_target] + inflected_distractors3[slot:]
                else:
                    used_distractors.update(distractors3)
                    final_options = raw_options
                dist_list = [opt for opt in final_options if opt.lower() != final_target.lower()]
                dist_str = ", ".join(dist_list)


        ctx.anchor = anchor
        ctx.anchor_type = anchor_type
        ctx.prep_obj = prep_obj
        ctx.quote_shown_prep = quote_shown_prep
        ctx.quote_shown_partitive = quote_shown_partitive
        ctx.is_quantifier_frame = is_quantifier_frame
        ctx.quant_head_noun = quant_head_noun
        ctx.half_slice_target = half_slice_target
        ctx.half_slice_distractors = half_slice_distractors
        ctx.half_slice_frame = half_slice_frame
        ctx.half_slice_meta = half_slice_meta
        ctx.half_slice_explanation = half_slice_explanation
        ctx.verb_requires_object = verb_requires_object
        ctx.is_plural_noun = is_plural_noun
        ctx.is_pedlex_plural = is_pedlex_plural
        ctx.plural_target = plural_target
        ctx.distractors = distractors
        ctx.dist_meta = dist_meta
        ctx.distractors3 = distractors3
        ctx.slot_target = slot_target
        ctx.slot = slot
        ctx.raw_options = raw_options
        ctx.inflection_desc = inflection_desc
        ctx.final_target = final_target
        ctx.final_options = final_options
        ctx.correct_answer_index = correct_answer_index
        ctx.dist_list = dist_list
        ctx.dist_str = dist_str
        ctx.locked_sense = locked_sense
        ctx.target_entry = target_entry
        ctx.t_s_idx = t_s_idx
        ctx.skeleton_definition = skeleton_definition
        ctx.atomic_bundle = atomic_bundle
        ctx.best_pattern = best_pattern
        ctx.ldoce_pattern_example = ldoce_pattern_example
        ctx.ldoce_cloze_frame = ldoce_cloze_frame
        ctx.pattern_anchor = pattern_anchor
