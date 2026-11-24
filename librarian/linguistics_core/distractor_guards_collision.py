"""Universal post-assembly double-key collision and synonym overlap guard."""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger("librarian.linguistics.guards")


class DistractorGuardsCollisionMixin:
    """Universal post-assembly screening for double-key collision and semantic overlap."""

    @classmethod
    def double_key_collision(cls, target: str, distractor: str,
                             anchor: Optional[str] = None,
                             anchor_type: Optional[str] = None,
                             pos: Optional[str] = None,
                             quote: Optional[str] = None) -> Tuple[bool, str]:
        """Deterministic double-key detector. Returns (is_double_key, kind).

        kind is one of:
          "double_key_prep"        : shared bound preposition + semantic neighbor
          "double_key_slot"        : same semantic slot (near-synonym/coordinate/troponym)
          "double_key_collocation" : shared collocation with anchor
          "frame_only"             : shared bound preposition or syntactic frame, semantically distinct
                                     (NOT a double-key; needs a specific exclusion clue)
          "none"                   : clear
        """
        t = (target or "").strip().lower()
        d = (distractor or "").strip().lower()
        if not t or not d or t == d:
            return False, "none"
        if cls.are_same_word_family(t, d) and not cls.is_cross_ontology_derivational_noun_pair(t, d):
            return True, "double_key_family"

        t_lem = cls.lemma_of(t) if " " not in t else t
        d_lem = cls.lemma_of(d) if " " not in d else d

        t_words = t.split()
        d_words = d.split()
        is_phrase_pair = len(t_words) >= 2 or len(d_words) >= 2 or pos in ("phrase", "phrasal_verb", "idiom")

        if is_phrase_pair:
            t_norm_poss = cls.normalize_phrase_possessives(t)
            d_norm_poss = cls.normalize_phrase_possessives(d)
            if t_norm_poss == d_norm_poss:
                return True, "double_key_family"

            t_w = t.split()
            d_w = d.split()
            # Canonicalize inflected head verbs for multi-word phrases (e.g. 'showed up' -> 'show up')
            t_base_head = cls.lemma_of(t_w[0]) if t_w else ""
            d_base_head = cls.lemma_of(d_w[0]) if d_w else ""
            t_base = " ".join([t_base_head] + t_w[1:]) if len(t_w) > 1 else t
            d_base = " ".join([d_base_head] + d_w[1:]) if len(d_w) > 1 else d
            t_base_norm_poss = cls.normalize_phrase_possessives(t_base)
            d_base_norm_poss = cls.normalize_phrase_possessives(d_base)

            if len(t_words) >= 3 and len(d_words) >= 3:
                if (
                    cls.normalize_phrase_determiners(t) == cls.normalize_phrase_determiners(d)
                    or cls.normalize_phrase_determiners(t_base) == cls.normalize_phrase_determiners(d_base)
                ):
                    return True, "double_key_slot"

            phrase_syns = cls._get_phrase_synonyms()
            t_syn_set = (
                phrase_syns.get(t, set())
                | phrase_syns.get(t_norm_poss, set())
                | phrase_syns.get(t_base, set())
                | phrase_syns.get(t_base_norm_poss, set())
            )
            d_syn_set = (
                phrase_syns.get(d, set())
                | phrase_syns.get(d_norm_poss, set())
                | phrase_syns.get(d_base, set())
                | phrase_syns.get(d_base_norm_poss, set())
            )
            if (
                d in t_syn_set or t in d_syn_set
                or d_norm_poss in t_syn_set or t_norm_poss in d_syn_set
                or d_base in t_syn_set or t_base in d_syn_set
                or d_base_norm_poss in t_syn_set or t_base_norm_poss in d_syn_set
            ):
                return True, "double_key_slot"

            for f_grp in cls._FRAME_PREP_SYNONYM_GROUPS:
                t_in_grp = any(x in f_grp for x in (t, t_norm_poss, t_base, t_base_norm_poss))
                d_in_grp = any(x in f_grp for x in (d, d_norm_poss, d_base, d_base_norm_poss))
                if t_in_grp and d_in_grp:
                    return True, "double_key_slot"

            # Multi-word 1-token-difference synonym guard (>= 3 words)
            if len(t_words) >= 3 and len(d_words) == len(t_words):
                for p_t_norm, p_d_norm in [
                    (t_norm_poss, d_norm_poss),
                    (t_base_norm_poss, d_base_norm_poss),
                ]:
                    norm_tw = p_t_norm.split()
                    norm_dw = p_d_norm.split()
                    diff_indices = [idx for idx in range(len(norm_tw)) if norm_tw[idx] != norm_dw[idx]]
                    if len(diff_indices) == 1:
                        d_idx = diff_indices[0]
                        tw_diff = norm_tw[d_idx]
                        dw_diff = norm_dw[d_idx]
                        if d_idx == 0:
                            if any(tw_diff in grp and dw_diff in grp for grp in cls._SYN_VERB_HEAD_CLUSTERS):
                                return True, "double_key_slot"
                            if dw_diff in cls.semantic_fields(tw_diff, pos="verb")[0]:
                                return True, "double_key_slot"
                        elif (
                            dw_diff in cls.semantic_fields(tw_diff, pos="noun")[0]
                            or dw_diff in cls.semantic_fields(tw_diff, pos="adj")[0]
                        ):
                            return True, "double_key_slot"

            # Multi-word WordNet synset overlap check
            wn = cls.get_wordnet()
            wn_t_syns = set()
            wn_d_syns = set()
            for cand_t in (t, t_base):
                wn_t_syns.update(wn.synsets(cand_t))
            for cand_d in (d, d_base):
                wn_d_syns.update(wn.synsets(cand_d))
            if wn_t_syns and wn_d_syns and (wn_t_syns & wn_d_syns):
                return True, "double_key_slot"

            # Same-Head Dual-Preposition Double-Key Guard:
            # Multi-word phrases sharing the exact same content headword and differing only in a relational
            # preposition (e.g. 'equate to' vs 'equate with', 'compare to' vs 'compare with', 'free of' vs 'free from',
            # 'obsessed with' vs 'obsessed by', 'integrate into' vs 'integrate with', 'hinge on' vs 'hinge upon')
            # are interchangeable or co-attested in standard English when no syntactic complement contrast exists.
            if len(t_words) == 2 and len(d_words) == 2 and t_words[0] == d_words[0] and t_words[1] != d_words[1]:
                _rel_preps = {
                    "to", "with", "into", "from", "of", "upon", "on", "for", "at",
                    "by", "about", "as", "against", "between", "among", "towards", "toward", "in", "over",
                }
                if t_words[1] in _rel_preps and d_words[1] in _rel_preps:
                    _adv_particles = {"up", "down", "out", "off", "away", "back", "along", "ahead", "apart", "aside", "around", "through", "together"}
                    if t_words[1] not in _adv_particles and d_words[1] not in _adv_particles:
                        return True, "double_key_prep"

            # Same-Preposition / Same-Tail 2-word phrase double-key guard (e.g. 'care about' vs 'worry about', 'show up' vs 'turn up')
            if len(t_words) == 2 and len(d_words) == 2 and t_words[1] == d_words[1] and t_words[0] != d_words[0]:
                tail_p = t_words[1]
                if tail_p in cls._PARTICLES_AND_PREPS:
                    h_t, h_d = t_words[0], d_words[0]
                    head_pos = "verb" if (cls._wordnet_pos(h_t) & {"v"} or cls._wordnet_pos(h_d) & {"v"}) else (
                        "adj" if (cls._wordnet_pos(h_t) & {"a", "s"} or cls._wordnet_pos(h_d) & {"a", "s"}) else None
                    )
                    is_h_col, _ = cls.double_key_collision(h_t, h_d, anchor=tail_p, anchor_type="prep", pos=head_pos, quote=quote)
                    if is_h_col:
                        return True, "double_key_prep"
                    t_def_text = cls._get_phrase_def_text(t)
                    d_def_text = cls._get_phrase_def_text(d)
                    c_forms = cls.inflected_forms(h_d) | {h_d}
                    t_forms = cls.inflected_forms(h_t) | {h_t}
                    if t_def_text:
                        for cf in c_forms:
                            if re.search(rf"\b{re.escape(cf)}\s+{re.escape(tail_p)}\b", t_def_text):
                                return True, "double_key_slot"
                            if h_d not in cls._DEFINING_GENUS_VERBS and re.search(rf"\b{re.escape(cf)}\b", t_def_text):
                                return True, "double_key_slot"
                    if d_def_text:
                        for tf in t_forms:
                            if re.search(rf"\b{re.escape(tf)}\s+{re.escape(tail_p)}\b", d_def_text):
                                return True, "double_key_slot"
                            if h_t not in cls._DEFINING_GENUS_VERBS and re.search(rf"\b{re.escape(tf)}\b", d_def_text):
                                return True, "double_key_slot"

        # Light-Verb Collocation Double-Key Guard:
        # If target verb occupies a light-verb collocation, dictionary idiom validation is the sole arbiter:
        # Any distractor that ALSO forms an attested collocation with that noun (and prep) in LDOCE
        # is an alternative key (double key). Any candidate that is strictly unattested is a safe foil
        # (NOT a double key), preventing generic synset overlaps from blocking high-discrimination foils.
        lv_slot_ctx = cls.detect_light_verb_collocation_context(
            quote=quote, anchor=anchor, anchor_type=anchor_type, target=t
        )
        if lv_slot_ctx:
            lv_n = lv_slot_ctx["noun"]
            lv_p = lv_slot_ctx.get("prep")
            d_is_attested, _ = cls.is_attested_light_verb_collocation(verb=d, noun=lv_n, prep=lv_p)
            if d_is_attested:
                return True, "double_key_collocation"
            return False, "none"

        # Construction Family Subcategorization Gate (`pedlex_cxn_index`):
        # When `anchor_type` or `quote` locks one of the 8 construction families, any candidate in
        # `ban_groups` (sharing the target's subcategorization frame in the same semantic/syntactic class)
        # is a constructional double-key, whereas verified `contrast_groups` candidates are single-fit safe.
        is_fronted_inv_cxn = (
            anchor_type == "fronted_inv"
            or cls.is_fronted_inversion_context(quote=quote, anchor=anchor, anchor_type=anchor_type, target=t)
        )
        is_lv_verb_cxn = bool(lv_slot_ctx)
        if anchor_type in (
            "it_extrapos", "gerund", "to_inf", "sb_to_do",
            "shell_to_do", "shell_of_doing", "shell_that",
            "correl_pair", "fronted_inv", "npi", "mandative_subj", "lv_triad",
        ) or is_fronted_inv_cxn or is_lv_verb_cxn:
            cxn_cands, active_cxn = cls.get_construction_contrast_distractors(
                t_lem,
                pos=pos or ("adv" if is_fronted_inv_cxn else ("verb" if is_lv_verb_cxn else "noun")),
                quote=quote,
                context_anchor=anchor,
                anchor_type="fronted_inv" if is_fronted_inv_cxn else ("lv_triad" if is_lv_verb_cxn else anchor_type),
                target_count=12,
                require_context_match=True,
            )
            if active_cxn:
                cxn_banned = cls.get_cxn_banned_words(active_cxn.get("ban_groups"))
                if d in cxn_banned or d_lem in cxn_banned:
                    return True, "double_key_pattern"
                if d in cxn_cands or d_lem in cxn_cands:
                    return False, "none"

        # Syntactic Disjointness Exemption Gate (<GRAM> / THESAURUS / grammar_boxes):
        # If distractor `d` has a hard grammatical clash with the syntactic slot of `t` in `quote`/`anchor_type`
        # (e.g., [only before noun] in predicative slot, [not before noun] in attributive slot,
        # pure [uncountable] after 'a/an/many', or no_passive/intransitive in passive/object slot),
        # `d` is 100% syntactically eliminated and cannot be a double key!
        has_slot_clash, clash_reason = cls.evaluate_syntactic_slot_clash(
            t, d, anchor=anchor, anchor_type=anchor_type, pos=pos, quote=quote
        )
        if has_slot_clash:
            if clash_reason.startswith("forbidden_frame_"):
                return False, "none"
            return False, "frame_only"

        near, antonyms = cls.semantic_fields(t, pos=pos)
        if t_lem != t:
            near_lem, ant_lem = cls.semantic_fields(t_lem, pos=pos)
            near |= near_lem
            antonyms |= ant_lem
        ped_ox_syns, ped_ox_ants = cls.get_pedlex_and_oxford_synonyms_antonyms(t, pos=pos)
        for sw in ped_ox_syns:
            if sw not in (t, t_lem):
                near.add(sw)
        for aw in ped_ox_ants:
            if aw not in (t, t_lem):
                antonyms.add(aw)
        # Also include PedLex thesaurus words and grammar_boxes comparison_peers in near semantic field
        for q_t in ({t, t_lem}):
            for item in cls.get_ldoce_thesaurus(q_t):
                raw_w = item.get("word", "").strip().lower()
                if not raw_w or raw_w == t:
                    continue
                # Parse slash-separated tokens (e.g. 'longing/yearning' -> 'longing', 'yearning')
                for chunk in raw_w.split("/"):
                    chunk = chunk.strip()
                    if chunk and " " not in chunk and chunk not in (t, t_lem):
                        near.add(chunk)
        t_prof = cls.get_word_gram_profile(t, pos=pos)
        for peer in t_prof.get("comparison_peers", set()):
            if peer not in (t, t_lem):
                near.add(peer)

        # Orthogonal Collective Noun Exemption:
        # Distinct subgroups in `_COLLECTIVE_NOUN_ONTOLOGY` (e.g. `staff` vs `crew`/`cast`/`jury`,
        # `flock` vs `herd`/`pack`/`swarm`) are mutually exclusive role/species coordinates, not synonyms,
        # even when Oxford Writer's Thesaurus groups them under a broad collective-noun heading.
        if (not pos or str(pos).lower().startswith("n")) and cls._are_orthogonal_collective_nouns(t_lem, d_lem):
            near.discard(d)
            near.discard(d_lem)

        # Compound Head-Sharing Co-Hyponym Gate (e.g., shareholder vs stakeholder, policymaker vs lawmaker):
        # If both nouns share a >= 5-char compound suffix head (excluding general derivational suffixes)
        # and share definition content words, treat them as near-synonyms so they cannot co-occur under generic modifiers.
        _deriv_noun_suffixes = (
            "itude", "ation", "ition", "ession", "ussion", "asion", "ision", "osion", "usion",
            "tion", "sion", "ment", "ness", "ship", "ance", "ence", "ancy", "ency", "ics", "ist", "ism", "ity", "ure",
        )
        if (not pos or str(pos).lower().startswith("n")) and " " not in t_lem and " " not in d_lem:
            if cls._common_suffix_len(t_lem, d_lem) >= 5 and not t_lem.endswith(_deriv_noun_suffixes):
                if cls._get_def_content_words(t_lem, "n") & cls._get_def_content_words(d_lem, "n"):
                    near.add(d)
                    near.add(d_lem)

        # Co-directional Scalar Verb Gate (e.g., swell vs gain / boost / grow / climb):
        # Scalar verbs sharing the identical increase or decrease directional primitive in their definitions
        # act as near-synonyms in change-of-state / magnitude contexts and must not collide as double keys.
        if pos and str(pos).lower().startswith("v") and " " not in t_lem and " " not in d_lem:
            _inc_prims = {"increase", "grow", "expand", "rise", "soar", "climb", "escalate", "enlarge", "amplify", "augment", "swell"}
            _dec_prims = {"decrease", "reduce", "shrink", "drop", "fall", "decline", "diminish", "lessen", "contract"}
            t_def_w = cls._get_def_content_words(t_lem, "v")
            d_def_w = cls._get_def_content_words(d_lem, "v")
            t_inc = bool(t_def_w & _inc_prims) or t_lem in _inc_prims
            d_inc = bool(d_def_w & _inc_prims) or d_lem in _inc_prims
            t_dec = bool(t_def_w & _dec_prims) or t_lem in _dec_prims
            d_dec = bool(d_def_w & _dec_prims) or d_lem in _dec_prims
            if (t_inc and d_inc and not (t_dec or d_dec)) or (t_dec and d_dec and not (t_inc or d_inc)):
                near.add(d)
                near.add(d_lem)

        # Collective / Group Noun Double-Key Gate:
        # If both target and distractor denote collections or groups of people/staff
        # and frame is partitive/of-complement ('a ____ of experts'), any other group noun is a double key!
        if (t in cls.COLLECTIVE_GROUP_NOUNS or t_lem in cls.COLLECTIVE_GROUP_NOUNS) and (d in cls.COLLECTIVE_GROUP_NOUNS or d_lem in cls.COLLECTIVE_GROUP_NOUNS):
            if anchor in ("of", "part of", "member of", "on", "in") or anchor_type in ("prep", "partitive"):
                return True, "double_key_slot"

        # Adverb syntactic modification frame discrimination
        if anchor_type in ("modifies_adj", "modifies_verb"):
            d_entry = cls.get_ldoce_entry(d) or cls.get_ldoce_entry(d_lem)
            if d_entry:
                d_senses = d_entry.get("senses", [])
                has_adj_mod = any(
                    any(k in (s.get("grammar") or "").lower() for k in ("adjective", "adverb", "[+adjective"))
                    for s in d_senses
                )
                has_verb_mod = any(
                    any(k in (s.get("grammar") or "").lower() for k in ("verb", "[+verb"))
                    or not s.get("grammar")  # default manner adverb
                    for s in d_senses
                )
                if anchor_type == "modifies_adj" and not has_adj_mod:
                    return False, "frame_only"
                if anchor_type == "modifies_verb" and not has_verb_mod:
                    return False, "frame_only"

        # Verb Syntactic Pattern Exclusion Gate (Single-Fit Distractor Guarantee):
        # If target governs a specific clausal/non-finite complement pattern ('to_inf', 'gerund', 'that_clause', 'sb_to_do')
        # and distractor is completely incompatible with that complement pattern, the distractor is
        # strictly single-fit discriminated (100% immune to double-key collision).
        # Conversely, if distractor shares the identical complement pattern and is in near semantic field,
        # it is a dangerous double-key collision!
        # NOTE: Only trigger when testing complement patterns (not when an explicit bound preposition or noun collocation anchor is being tested).
        if anchor_type in ("to_inf", "gerund", "that_clause", "sb_to_do") or (not anchor and str(pos or "").strip().lower().startswith("v")):
            t_slots = cls.get_syntactic_pattern_slots(t) | cls.get_syntactic_pattern_slots(t_lem)
            d_slots = cls.get_syntactic_pattern_slots(d) | cls.get_syntactic_pattern_slots(d_lem)
            d_prof = cls.get_word_gram_profile(d, pos=pos)
            d_slots -= d_prof.get("forbidden_frames", set())
            # If anchor_type directly specifies the target complement slot:
            active_slot = anchor_type if anchor_type in ("to_inf", "gerund", "that_clause", "sb_to_do") else None
            if not active_slot:
                # Find intersective slots declared by target
                common_slots = t_slots & {"to_inf", "gerund", "that_clause", "sb_to_do"}
                if len(common_slots) == 1:
                    active_slot = next(iter(common_slots))

            if active_slot and (active_slot in t_slots or anchor_type == active_slot):
                if active_slot not in d_slots or active_slot in d_prof.get("forbidden_frames", set()):
                    # Distractor cannot occupy target's syntactic complement slot -> single-fit discriminated!
                    return False, "none"
                elif d in near or d_lem in near or (cls.semantic_fields(t_lem, pos="v")[0] & cls.semantic_fields(d_lem, pos="v")[0]):
                    # Both share the non-finite/clausal complement slot and are semantic neighbors -> double-key collision!
                    return True, "double_key_pattern"

        t_preps_raw = cls.get_pedlex_collocations(t, pos=pos).get("prep", []) + cls.get_pedlex_collocations(t_lem, pos=pos).get("prep", [])
        t_preps = [w.strip().lower() for p in t_preps_raw for w in p.split("/") if w.strip()]
        for lp in list(cls.get_ldoce_preps(t, pos=pos)) + list(cls.get_ldoce_preps(t_lem, pos=pos)):
            if lp not in t_preps:
                t_preps.append(lp)
        shared = bool(anchor) and anchor.lower() in t_preps

        if anchor_type == "prep" and anchor and shared:
            # Productive Source/Agent Preposition Guard (`from` / `by`):
            # For nouns, `from [source/organization]` and `by [agent]` are productive theta-role adjuncts
            # that attach to any noun denoting support, funding, approval, assistance, or communication
            # (e.g. `sponsorship` vs `endorsement` / `patronage` / `backing` / `funding` / `subsidy` from corporate partners),
            # even when a dictionary only lists `of` as the object-complement preposition for the distractor.
            if (
                pos
                and str(pos).lower().startswith("n")
                and anchor.lower() in ("from", "by")
                and d not in antonyms
                and d_lem not in antonyms
            ):
                _productive_source_domains = {
                    "support", "help", "assistance", "aid", "money", "fund", "funding", "financial",
                    "approval", "permission", "agreement", "payment", "gift", "donation", "advice",
                    "guidance", "message", "statement", "praise", "criticism", "protection", "benefit",
                }
                t_def_w = cls._get_def_content_words(t_lem, "n") | {t_lem}
                d_def_w = cls._get_def_content_words(d_lem, "n") | {d_lem}
                if (t_def_w & _productive_source_domains) and (d_def_w & _productive_source_domains):
                    return True, "double_key_prep"
            d_preps_raw = cls.get_pedlex_collocations(d, pos=pos or "noun").get("prep", []) + cls.get_pedlex_collocations(d_lem, pos=pos or "noun").get("prep", [])
            d_preps = [w.strip().lower() for p in d_preps_raw for w in p.split("/") if w.strip()]
            for lp in list(cls.get_ldoce_preps(d, pos=pos)) + list(cls.get_ldoce_preps(d_lem, pos=pos)):
                if lp not in d_preps:
                    d_preps.append(lp)
            p_idx = cls._get_phrase_index()
            anc_low = anchor.lower()
            if (
                f"{d_lem} {anc_low}" in p_idx
                or f"{d} {anc_low}" in p_idx
                or f"be {d_lem} {anc_low}" in p_idx
                or f"be {d} {anc_low}" in p_idx
            ):
                if anc_low not in d_preps:
                    d_preps.append(anc_low)
            if pos and str(pos).lower().startswith("v"):
                if anchor.lower() in (
                    "in", "on", "at", "into", "onto", "to", "from", "across", "through", "around", "along", "by", "with"
                ):
                    wn = cls.get_wordnet()
                    t_v_syns = (wn.synsets(t_lem, "v") or wn.synsets(t, "v"))[:2]
                    d_v_syns = (wn.synsets(d_lem, "v") or wn.synsets(d, "v"))[:2]
                    t_v_lfs = {s.lexfile() for s in t_v_syns if s.lexfile()}
                    d_v_lfs = {s.lexfile() for s in d_v_syns if s.lexfile()}
                    if "verb.motion" in t_v_lfs and "verb.motion" in d_v_lfs:
                        return True, "double_key_prep"
                    if d not in antonyms and d_lem not in antonyms:
                        t_v_hyps = {h for s in t_v_syns for h in s.hypernyms() if h.lexfile() not in ("verb.stative", "verb.change")}
                        d_v_hyps = {h for s in d_v_syns for h in s.hypernyms() if h.lexfile() not in ("verb.stative", "verb.change")}
                        if (t_v_hyps & d_v_hyps) and anchor.lower() in d_preps:
                            return True, "double_key_prep"
                        if (
                            cls._verb_has_intransitive_prep_pattern(t_lem, anchor.lower())
                            and cls._verb_has_intransitive_prep_pattern(d_lem, anchor.lower())
                        ):
                            t_all_hyps = {h for s in t_v_syns for h in s.hypernyms()}
                            d_all_hyps = {h for s in d_v_syns for h in s.hypernyms()}
                            if t_all_hyps & d_all_hyps:
                                return True, "double_key_prep"
            if anchor.lower() in d_preps:
                if d in near or d_lem in near:
                    return True, "double_key_prep"
                return False, "frame_only"
            if (d in near or d_lem in near) and d not in antonyms and d_lem not in antonyms:
                # For nouns (`pos` starting with 'n'), a near-synonym in `near` is ONLY allowed under a
                # prepositional anchor when it is a curated contrastive peer in LDOCE's `thesaurus` box
                # (e.g. `loan` vs `overdraft` under `of`) and NOT a reciprocal synonym in `oxford_thesaurus`
                # (e.g. `threat` vs `hazard`, `recession` vs `downturn`, `oath` vs `commitment`, `shortage` vs `absence`).
                if pos and str(pos).lower().startswith("n"):
                    if anchor.lower() in ("into", "from", "by"):
                        return True, "double_key_prep"
                    d_strict_n_syns = cls.get_strict_pedlex_and_oxford_synonyms(d, pos="noun") | cls.get_strict_pedlex_and_oxford_synonyms(d_lem, pos="noun")
                    if t in d_strict_n_syns or t_lem in d_strict_n_syns:
                        return True, "double_key_prep"
                    t_ent = cls.get_ldoce_entry(t) or cls.get_ldoce_entry(t_lem) or {}
                    t_thes_words: Set[str] = set()
                    for th_it in (t_ent.get("thesaurus") or []):
                        if isinstance(th_it, dict):
                            cw = str(th_it.get("word") or "").strip().lower()
                            if cw:
                                t_thes_words.add(cw)
                                t_thes_words.add(cw.split()[0])
                    if d not in t_thes_words and d_lem not in t_thes_words:
                        return True, "double_key_prep"
                    # Even if present in `t_thes_words`, if `d` is a direct WordNet synonym/hypernym/hyponym
                    # without its own competing preposition valency (`not d_preps`), it remains a double-key under `prep`.
                    if not d_preps:
                        return True, "double_key_prep"
                if pos and str(pos).lower().startswith(("a", "j")) and not str(pos).lower().startswith("adv"):
                    if anchor.lower() in ("for", "in", "at", "with", "by") and not d_preps:
                        d_strict_a_syns = (
                            cls.get_synonyms(t)
                            | cls.get_synonyms(t_lem)
                            | cls.get_strict_pedlex_and_oxford_synonyms(t, pos="adj")
                        )
                        if d in d_strict_a_syns or d_lem in d_strict_a_syns:
                            return True, "double_key_prep"
                # A near-synonym that either lacks competing prepositional valency in PedLex/OCD
                # or, when the target verb itself can take a direct object (`V + [sth] + prep`, e.g.,
                # `convert/adapt [sth] into [sth]`), is also a transitive verb cannot be safely
                # discriminated by preposition alone. Conversely, when the target verb is strictly
                # intransitive (`listen to`), the slot is `____ + prep` with no intervening NP, so a
                # transitive verb lacking that preposition (`hear`, `watch`, `notice`) is barred (`*notice to`).
                if pos and str(pos).lower().startswith("v"):
                    if anchor.lower() in ("in", "on", "at", "into", "from", "by", "with", "for", "around", "across", "through", "along"):
                        return True, "double_key_prep"
                    if cls._verb_takes_object(t_lem):
                        if not d_preps or cls._verb_takes_object(d_lem):
                            return True, "double_key_prep"
                    elif not d_preps and not cls._verb_takes_object(d_lem):
                        return True, "double_key_prep"
                elif not pos and not d_preps:
                    return True, "double_key_prep"
            # Candidate does NOT govern the anchor preposition -> single-fit discriminated!
            return False, "none"

        if d in near or d_lem in near:
            if d in antonyms or d_lem in antonyms:
                # Deliberate contrast distractor (eliminated by polarity/logic),
                # whitelisted by design in the distractor generator.
                return False, "contrast"
            # If an explicit collocational anchor is present (e.g. adjective modifier or verb)
            # and candidate d does NOT share that collocation in OCD/PedLex, candidate d is ruled out
            # by the anchor slot itself and is therefore a valid, single-fit distractor.
            # EXCEPTION: If anchor is a generic adjective or verb (e.g. huge, massive, new, good, have),
            # OR if `quote` is provided and the purported `anchor` does not actually appear in `quote`,
            # absence in OCD does NOT discriminate near-synonyms! Near-synonyms remain double keys.
            if anchor_type == "v_obj_pp" and anchor:
                a_clean = anchor.lower()
                if (
                    cls.lookup_ldoce_phrase(f"{d} {a_clean}")
                    or cls.lookup_ldoce_phrase(f"{d_lem} {a_clean}")
                    or cls.lookup_ldoce_phrase(f"{d} something {a_clean}")
                    or cls.lookup_ldoce_phrase(f"{d_lem} something {a_clean}")
                ):
                    return True, "double_key_collocation"
                return False, "none"

            if anchor and anchor_type in ("adj", "verb", "object", "modified_noun", "verb_subject", "modifies_adj", "modifies_verb"):
                a_clean = anchor.lower()
                is_generic_anchor = (
                    (anchor_type in ("adj", "modified_noun", "object") and a_clean in cls.GENERIC_COLLOCATION_MODIFIERS) or
                    (anchor_type in ("verb", "verb_subject") and a_clean in cls.GENERIC_COLLOCATION_VERBS)
                )
                if is_generic_anchor:
                    return True, "double_key_slot"
                if quote and len(a_clean) >= 3:
                    q_low = quote.lower()
                    a_lem = cls.lemma_of(a_clean) if " " not in a_clean else a_clean
                    if a_clean not in q_low and a_lem not in q_low:
                        return True, "double_key_slot"

                d_entry = cls.get_pedlex_collocations(d, pos=pos or "noun")
                shared_colloc = False
                if anchor_type == "adj":
                    shared_colloc = a_clean in [a.split()[0].lower() for a in d_entry.get("adj", [])]
                    if not shared_colloc:
                        d_ldoce = cls.get_ldoce_entry(d)
                        if d_ldoce:
                            for c_item in d_ldoce.get("collocations", {}).get("adjectives", []):
                                if c_item.get("pos") and cls._pos_code(c_item["pos"]) != "n":
                                    continue
                                if a_clean in c_item.get("collocation", "").lower():
                                    shared_colloc = True
                                    break
                elif anchor_type == "verb":
                    shared_colloc = a_clean in [v.split()[0].lower() for v in d_entry.get("verb_before", [])]
                    if not shared_colloc:
                        d_ldoce = cls.get_ldoce_entry(d)
                        if d_ldoce:
                            for c_item in d_ldoce.get("collocations", {}).get("verbs", []):
                                if c_item.get("pos") and cls._pos_code(c_item["pos"]) != "n":
                                    continue
                                if a_clean in c_item.get("collocation", "").lower():
                                    shared_colloc = True
                                    break
                elif anchor_type in ("modified_noun", "object"):
                    shared_colloc = a_clean in [n.split()[0].lower() for n in d_entry.get("colloc_nouns", []) + d_entry.get("noun_after", [])]
                    if not shared_colloc:
                        d_ldoce = cls.get_ldoce_entry(d)
                        if d_ldoce:
                            for c_item in d_ldoce.get("collocations", {}).get("nouns", []):
                                if c_item.get("pos") and cls._pos_code(c_item["pos"]) != "n":
                                    continue
                                if a_clean in c_item.get("collocation", "").lower():
                                    shared_colloc = True
                                    break
                    if not shared_colloc and anchor_type == "object":
                        a_entry = cls.get_pedlex_collocations(a_clean, pos="noun")
                        if d in [v.split()[0].lower() for v in a_entry.get("verb_before", [])] or d_lem in [v.split()[0].lower() for v in a_entry.get("verb_before", [])]:
                            shared_colloc = True
                        if not shared_colloc:
                            a_ldoce = cls.get_ldoce_entry(a_clean)
                            if a_ldoce:
                                for v_item in a_ldoce.get("collocations", {}).get("verbs", []):
                                    if v_item.get("pos") and cls._pos_code(v_item["pos"]) != "n":
                                        continue
                                    v_col = v_item.get("collocation", "").lower()
                                    if re.search(rf"\b{re.escape(d)}\b|\b{re.escape(d_lem)}\b", v_col):
                                        shared_colloc = True
                                        break
                    if not shared_colloc and anchor_type == "modified_noun":
                        a_entry = cls.get_pedlex_collocations(a_clean, pos="noun")
                        if d in [adj_w.split()[0].lower() for adj_w in a_entry.get("adj", [])] or d_lem in [adj_w.split()[0].lower() for adj_w in a_entry.get("adj", [])]:
                            shared_colloc = True
                        if not shared_colloc:
                            a_ldoce = cls.get_ldoce_entry(a_clean)
                            if a_ldoce:
                                for adj_item in a_ldoce.get("collocations", {}).get("adjectives", []):
                                    if adj_item.get("pos") and cls._pos_code(adj_item["pos"]) != "n":
                                        continue
                                    adj_col = (adj_item.get("collocation") or adj_item.get("phrase") or "").lower()
                                    if re.search(rf"\b{re.escape(d)}\b|\b{re.escape(d_lem)}\b", adj_col):
                                        shared_colloc = True
                                        break
                        if not shared_colloc:
                            if cls.lookup_ldoce_phrase(f"{d} {a_clean}") or cls.get_ldoce_entry(f"{d} {a_clean}") or (d_lem != d and (cls.lookup_ldoce_phrase(f"{d_lem} {a_clean}") or cls.get_ldoce_entry(f"{d_lem} {a_clean}"))):
                                shared_colloc = True
                        if not shared_colloc and d_ldoce:
                            for s in d_ldoce.get("senses", []):
                                for ex in s.get("examples", []):
                                    if re.search(rf"\b(?:{re.escape(d)}|{re.escape(d_lem)})\s+{re.escape(a_clean)}s?\b", str(ex).lower()):
                                        shared_colloc = True
                                        break
                                if shared_colloc:
                                    break
                elif anchor_type in ("modifies_adj", "modifies_verb"):
                    # Check if distractor adverb explicitly collocates with this anchor in PedLex
                    d_ldoce = cls.get_ldoce_entry(d)
                    if d_ldoce:
                        for s in d_ldoce.get("senses", []):
                            for ex in s.get("examples", []):
                                if re.search(rf"\b{re.escape(a_clean)}\b", ex.lower()):
                                    shared_colloc = True
                                    break
                            if shared_colloc:
                                break
                if not shared_colloc:
                    return False, "none"
            return True, "double_key_slot"

        # Collocational Double-Key Gate: If distractor directly collocates with the anchor in the identical slot
        if anchor:
            a_clean = anchor.lower()
            pos_low_anc = str(pos or "").strip().lower()
            if pos_low_anc.startswith("v") and anchor_type in ("object", "collocation"):
                d_entry = cls.get_pedlex_collocations(d, pos="verb")
                if a_clean in [n.split()[0].lower() for n in d_entry.get("colloc_nouns", [])]:
                    return True, "double_key_collocation"
                a_entry = cls.get_pedlex_collocations(a_clean, pos="noun")
                if d in [v.split()[0].lower() for v in a_entry.get("verb_before", [])]:
                    return True, "double_key_collocation"
                a_ldoce = cls.get_ldoce_entry(a_clean)
                if a_ldoce:
                    for v_item in a_ldoce.get("collocations", {}).get("verbs", []):
                        if v_item.get("pos") and cls._pos_code(v_item["pos"]) != "n":
                            continue
                        if d in v_item.get("collocation", "").lower():
                            return True, "double_key_collocation"
            elif pos_low_anc.startswith(("a", "j")) and not pos_low_anc.startswith("adv") and anchor_type == "modified_noun":
                a_entry = cls.get_pedlex_collocations(a_clean, pos="noun")
                if d in [adj.split()[0].lower() for adj in a_entry.get("adj", [])] and d not in antonyms:
                    return True, "double_key_collocation"
                a_ldoce = cls.get_ldoce_entry(a_clean)
                if a_ldoce:
                    for a_item in a_ldoce.get("collocations", {}).get("adjectives", []):
                        if a_item.get("pos") and cls._pos_code(a_item["pos"]) != "n":
                            continue
                        if d in a_item.get("collocation", "").lower() and d not in antonyms:
                            return True, "double_key_collocation"
            elif pos_low_anc.startswith("n") and anchor_type in ("verb", "verb_subject"):
                d_entry = cls.get_pedlex_collocations(d, pos="noun")
                if a_clean in [v.split()[0].lower() for v in d_entry.get("verb_before", [])] and d not in antonyms:
                    return True, "double_key_collocation"
                d_ldoce = cls.get_ldoce_entry(d)
                if d_ldoce:
                    for v_item in d_ldoce.get("collocations", {}).get("verbs", []):
                        if v_item.get("pos") and cls._pos_code(v_item["pos"]) != "n":
                            continue
                        if a_clean in v_item.get("collocation", "").lower() and d not in antonyms:
                            return True, "double_key_collocation"

        return False, "none"



__all__ = ["DistractorGuardsCollisionMixin"]
