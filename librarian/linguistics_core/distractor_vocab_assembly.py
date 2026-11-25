"""Distractor collision clearance, slot allocation, and multi-tier fallbacks."""

from __future__ import annotations

import logging
import math
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger("librarian.linguistics.vocab_distractors")


class DistractorVocabAssemblyMixin:

    @classmethod
    def _assemble_vocab_distractors(
        cls,
        wsd_ctx: Dict[str, Any],
        harvest_ctx: Dict[str, Any],
        return_metadata: bool = False,
    ) -> Union[List[str], Tuple[List[str], Dict[str, str]]]:
        """Executes Phase 3: Oxford collision clearance, slot allocation, and fallbacks."""
        clean_target = wsd_ctx["clean_target"]
        pos = wsd_ctx["pos"]
        wn_pos = wsd_ctx["wn_pos"]
        context_anchor = wsd_ctx["context_anchor"]
        anchor_type = wsd_ctx["anchor_type"]
        target_count = wsd_ctx["target_count"]
        exclude_words = wsd_ctx["exclude_words"]
        definition = wsd_ctx["definition"]
        quote = wsd_ctx["quote"]
        wn = wsd_ctx["wn"]
        nlp = wsd_ctx["nlp"]
        target_synsets = wsd_ctx["target_synsets"]
        global_synonyms = wsd_ctx["global_synonyms"]
        ped_ox_syns = wsd_ctx["ped_ox_syns"]
        def_tokens = wsd_ctx["def_tokens"]
        target_lfs_set = wsd_ctx["target_lfs_set"]
        target_strictly_countable = wsd_ctx["target_strictly_countable"]
        target_strictly_uncountable = wsd_ctx["target_strictly_uncountable"]
        is_concrete_physical_noun = wsd_ctx["is_concrete_physical_noun"]
        target_is_informal = wsd_ctx["target_is_informal"]
        pedlex_signpost = wsd_ctx["pedlex_signpost"]
        pedlex_units = wsd_ctx["pedlex_units"]

        tier0_ldoce_thesaurus = harvest_ctx["tier0_ldoce_thesaurus"]
        tier1_synonyms = harvest_ctx["tier1_synonyms"]
        tier2_hyponyms = harvest_ctx["tier2_hyponyms"]
        tier2_satellites = harvest_ctx["tier2_satellites"]
        tier3_attributes = harvest_ctx["tier3_attributes"]
        tier3_coordinates = harvest_ctx["tier3_coordinates"]
        candidates = harvest_ctx["candidates"]
        sense_antonyms = list(harvest_ctx["sense_antonyms"])
        _is_collective_noun_target = harvest_ctx["_is_collective_noun_target"]
        _is_aligned_noun_coord = harvest_ctx["_is_aligned_noun_coord"]
        _coord_sort_key = harvest_ctx.get("_coord_sort_key")
        if _coord_sort_key is None:
            del _coord_sort_key

        _pos_ok = lambda lem: cls._pos_ok(lem, wn_pos=wn_pos)
        _is_primarily_adj_or_verb = cls._is_primarily_adj_or_verb
        _is_primarily_noun_or_adj_for_verb = cls._is_primarily_noun_or_adj_for_verb
        _is_informal_or_slang_distractor = lambda w: cls._is_informal_or_slang_distractor(
            w, wn_pos=wn_pos, target_is_informal=target_is_informal, target_lfs_set=target_lfs_set
        )

        # 3. Oxford Collision Clearance (Anti Double-Key Gate - Schemes 3 & 4)
        # Exempt candidates that have a verified syntactic slot clash with target's slot
        # or are anchor-discriminated Tier 0 PedLex Thesaurus peers from global_synonyms ban.
        tier0_set = set(tier0_ldoce_thesaurus)
        forbidden_words = {
            gw for gw in global_synonyms
            if gw not in tier0_set
            and not cls.evaluate_syntactic_slot_clash(
                clean_target, gw, anchor=context_anchor, anchor_type=anchor_type, pos=pos, quote=quote
            )[0]
        } | {clean_target}

        # 3b. Unified Eight Construction Families Subcategorization Gate (`pedlex_cxn_index`):
        # When the target word in context instantiates one of the 8 lexico-syntactic construction families,
        # prioritize subcategorization-contrasting distractors (`contrast_groups \ ban_groups`) and ban
        # same-construction synonyms/antonyms (`ban_groups`) that also fit the grammatical slot.
        _locked_s_idx_cxn: Optional[int] = None
        _is_concrete_physical_noun_sense = bool(
            wn_pos == "n"
            and (
                any(
                    s.lexfile() in (
                        "noun.artifact", "noun.substance", "noun.object",
                        "noun.food", "noun.plant", "noun.animal", "noun.body",
                    )
                    for s in target_synsets
                )
                or (
                    bool(definition)
                    and bool(
                        set(re.findall(r"\b[a-z]{3,}\b", definition.lower()))
                        & {
                            "machine", "tool", "equipment", "device", "instrument",
                            "appliance", "vehicle", "piece", "object", "substance",
                            "clothing", "garment", "container", "hardware",
                        }
                    )
                )
            )
        )
        if definition or quote:
            _t_ent_cxn = cls.get_ldoce_entry(clean_target)
            if _t_ent_cxn:
                _senses_cxn = _t_ent_cxn.get("senses") or []
                if definition:
                    _def_norm_cxn = definition.strip().lower()
                    for _si, _sn in enumerate(_senses_cxn):
                        if str(_sn.get("definition") or "").strip().lower() == _def_norm_cxn:
                            _locked_s_idx_cxn = _si
                            break
                if _locked_s_idx_cxn is None:
                    _locked_s_idx_cxn = cls._lock_sense(
                        _t_ent_cxn, quote=quote or definition, target_pos=pos
                    )
        if _is_concrete_physical_noun_sense:
            cxn_cands, active_cxn = [], None
        else:
            cxn_cands, active_cxn = cls.get_construction_contrast_distractors(
                clean_target,
                pos=pos,
                sense_idx=_locked_s_idx_cxn,
                quote=quote,
                context_anchor=context_anchor,
                anchor_type=anchor_type,
                target_count=max(target_count * 2, 6),
                exclude_words=exclude_words,
                require_context_match=True,
            )
        cxn_cand_set: Set[str] = set()
        if active_cxn:
            cxn_banned = cls.get_cxn_banned_words(active_cxn.get("ban_groups")) | set(active_cxn.get("ban_groups") or [])
            forbidden_words |= cxn_banned
            sense_antonyms = [ant for ant in sense_antonyms if ant not in cxn_banned]
            if cxn_cands:
                cxn_cand_set = set(cxn_cands)
                cxn_excl = cxn_banned | (exclude_words or set()) | {clean_target}
                candidates = [
                    c for c in cxn_cands
                    if c not in cxn_excl and cls.is_cefr_compliant_distractor(c, clean_target)
                ] + [
                    c for c in candidates if c not in cxn_cand_set
                ]
        is_fronted_inv_slot = cls.is_fronted_inversion_context(
            quote=quote, anchor=context_anchor, anchor_type=anchor_type, target=clean_target
        )
        if context_anchor:
            clean_anchor = context_anchor.strip().lower()
            if (wn_pos in ("a", "n") or not wn_pos) and anchor_type == "prep":
                # Preposition Valency Gate - reject candidates that also govern the target preposition
                # (except orthogonal collective/partitive nouns on 'of', e.g. 'lump of' vs 'drop/slice/sheet of',
                # which share 'of' syntactically while remaining mutually exclusive by physical substance).
                for cand in candidates:
                    if clean_anchor == "of" and cls._are_orthogonal_collective_nouns(clean_target, cand):
                        continue
                    cand_preps = cls.get_governed_prepositions(cand, pos=pos)
                    if clean_anchor in cand_preps or cls.lookup_ldoce_phrase(f"{cand} {clean_anchor}"):
                        forbidden_words.add(cand)
            elif wn_pos == "a" and anchor_type == "modified_noun":
                # Modified Noun Collocation Clash - reject legal adjectives that also collocate with the noun
                anchor_entry = cls.get_pedlex_collocations(clean_anchor) or {}
                for a in anchor_entry.get("adj", []):
                    tok = a.split()[0].lower()
                    if tok not in sense_antonyms:
                        forbidden_words.add(tok)
            else:
                anchor_entry = cls.get_pedlex_collocations(clean_anchor) or {}
                # If target is verb, anchor is noun: check verb_before and verb_after
                for v in anchor_entry.get("verb_before", []) + anchor_entry.get("verb_after", []):
                    forbidden_words.add(v.split()[0].lower())
                # If target is adj, anchor is noun: check adj
                for a in anchor_entry.get("adj", []):
                    tok = a.split()[0].lower()
                    if tok not in sense_antonyms:
                        forbidden_words.add(tok)
                # If target is noun, anchor is verb (verb-object frame): ban nouns that also collocate with that verb
                # (When anchor_type == 'adj', e.g. 'economic slump', noun items use definition-grounded sentence stems,
                # so non-synonym coordinates like 'prosperity' must not be banned merely for sharing 'economic'.)
                if not (wn_pos == "n" and anchor_type == "adj"):
                    for n in anchor_entry.get("colloc_nouns", []) + anchor_entry.get("noun_after", []):
                        forbidden_words.add(n.split()[0].lower())

        safe_distractors: List[str] = []
        distractor_metadata: Dict[str, str] = {}

        # Slot 1: Sense-Locked Antonym (if available and not superseded by a Construction Family contrast)
        # Antonyms are whitelisted from context_anchor collision clearance because their
        # collocational plausibility is deliberate, to be eliminated by sentence polarity/logic.
        # However, when a Construction Family subcategorization contrast (`cxn_cand_set`) is active,
        # subcategorization-contrasting candidates take primary slots so syntactic frame contrast is tested.
        max_antonyms = 0 if cxn_cand_set else (1 if (wn_pos != "a" or len(tier2_satellites) >= 2) else 2)
        for ant in sense_antonyms:
            if len([s for s in safe_distractors if distractor_metadata.get(s) == "antonym"]) >= max_antonyms:
                break
            if ant not in safe_distractors and ant not in exclude_words and len(ant) >= 2:
                if wn_pos == "v" and anchor_type == "prep" and cls.double_key_collision(
                    clean_target, ant, anchor=context_anchor, anchor_type=anchor_type, pos=pos, quote=quote
                )[0]:
                    continue
                if cls.is_cefr_compliant_distractor(ant, clean_target):
                    safe_distractors.append(ant)
                    distractor_metadata[ant] = "antonym"

        # Slots 2 & 3: Candidates from PedLex Thesaurus & Activator / Coordinate / Troponym / Synonym trees
        near_set_for_pile, ants_for_pile = cls.semantic_fields(clean_target, pos=pos)
        if _is_collective_noun_target:
            near_set_for_pile = {
                w for w in near_set_for_pile
                if not cls._are_orthogonal_collective_nouns(clean_target, w)
            }
        strict_target_syns = cls.get_strict_pedlex_and_oxford_synonyms(
            clean_target, pos=pos, definition=definition, quote=quote
        )
        wn_same_synset_words: Set[str] = set()
        if target_synsets:
            for s in target_synsets:
                for w in s.words():
                    wn_same_synset_words.add(w.lemma().lower())
        near_only_for_pile = (
            (near_set_for_pile | tier0_set) & (strict_target_syns | wn_same_synset_words)
        ) - ants_for_pile - set(sense_antonyms)
        for cand in candidates:
            if cand in safe_distractors:
                continue
            if not cls.is_cefr_compliant_distractor(cand, clean_target):
                continue
            if _is_informal_or_slang_distractor(cand):
                continue
            has_cand_clash, cand_clash_r = cls.evaluate_syntactic_slot_clash(
                clean_target, cand, anchor=context_anchor, anchor_type=anchor_type, pos=pos, quote=quote
            )
            # Fronted Negative Inversion Arena Gate:
            # In an inverted clause (Aux + Subj), ban all non-inverting adverbs (*often, *steadily, *completely)
            # so multiple-choice distractors test SLA correlative/aspectual nuances rather than syntax giveaways.
            if is_fronted_inv_slot and cand_clash_r == "non_inverting_candidate_in_fronted_inversion_slot":
                continue
            # Synonym-Pile Gate: at most 1 non-antonym distractor may sit inside near_only_for_pile (strict near-synonyms of target)
            if not has_cand_clash and cand in near_only_for_pile and cand not in cxn_cand_set and sum(1 for s in safe_distractors if s in near_only_for_pile and distractor_metadata.get(s) != "antonym") >= 1:
                continue
            # Countability Symmetry Gate for Nouns:
            # - For strictly uncountable noun targets: allow at most 1 countable foil into safe_distractors.
            # - For strictly countable noun targets: allow at most 1 uncountable foil into safe_distractors.
            if wn_pos == "n":
                if target_strictly_uncountable and cls._is_primarily_countable_noun(cand):
                    if sum(1 for s in safe_distractors if cls._is_primarily_countable_noun(s)) >= 1:
                        continue
                elif target_strictly_countable and cls._is_primarily_uncountable_noun(cand):
                    if sum(1 for s in safe_distractors if cls._is_primarily_uncountable_noun(s)) >= 1:
                        continue
            # Attributive position check: if modifying a noun, reject predicative-only adjectives
            if anchor_type == "modified_noun" and cand in cls._PREDICATIVE_ONLY_ADJS:
                continue
            # Allomorph / Word Family Guard: prevent morphological double-keys (e.g. resiliency vs resilience)
            if (
                cls.are_same_word_family(cand, clean_target)
                and not (wn_pos == "n" and cls.is_cross_ontology_derivational_noun_pair(cand, clean_target))
            ) or any(cls.are_same_word_family(cand, s) for s in safe_distractors):
                continue
            # Cross-distractor mutual synonym exclusion: avoid multiple distractors being mutual synonyms
            # (For nouns, if non-antonym coordinates are exhausted and we are filling from `sense_antonyms`,
            # allow at most 1 extra antonym peer so we don't fall back to random orthographic rhymes.)
            cand_syns = cls.get_synonyms(cand)
            is_noun_fallback_ant = (
                wn_pos == "n"
                and cand in sense_antonyms
                and sum(1 for s in safe_distractors if distractor_metadata.get(s) == "antonym") < 2
            )
            if not is_noun_fallback_ant and cand not in cxn_cand_set and any(
                (s in cand_syns or cand in cls.get_synonyms(s))
                and not cls._are_orthogonal_collective_nouns(cand, s)
                for s in safe_distractors
            ):
                continue

            # Deterministic Double-Key & Pattern Exclusion Gate:
            # Rejects candidates that share identical syntactic patterns or collocations (double-key collision)
            if cand not in cxn_cand_set or cand in near_set_for_pile:
                is_dk, dk_reason = cls.double_key_collision(
                    clean_target, cand,
                    anchor=context_anchor,
                    anchor_type=anchor_type,
                    pos=pos,
                    quote=quote
                )
                if is_dk or (dk_reason == "contrast" and cand not in sense_antonyms):
                    continue

            if (cand in cxn_cand_set or cand not in forbidden_words) and cand not in exclude_words and len(cand) >= 2:
                safe_distractors.append(cand)
                if cand in cxn_cand_set:
                    distractor_metadata[cand] = "cxn_subcat_contrast"
                elif cand in sense_antonyms:
                    distractor_metadata[cand] = "antonym"
                elif wn_pos == "n" and target_strictly_uncountable and cls._is_primarily_countable_noun(cand):
                    distractor_metadata[cand] = "countability_contrast_countable"
                elif wn_pos == "n" and target_strictly_countable and cls._is_primarily_uncountable_noun(cand):
                    distractor_metadata[cand] = "countability_contrast_uncountable"
                elif cand in tier0_ldoce_thesaurus:
                    distractor_metadata[cand] = "ldoce_thesaurus"
                elif anchor_type == "prep" and has_cand_clash and cand_clash_r == "unattested_prepositional_government_clash":
                    is_sem_peer = (
                        cand in tier0_ldoce_thesaurus
                        or cand in tier2_hyponyms
                        or cand in tier3_coordinates
                        or (wn_pos == "n" and _is_aligned_noun_coord(cand))
                        or (target_lfs_set and any(cs.lexfile() in target_lfs_set for cs in (wn.synsets(cand, wn_pos) if wn else [])))
                    )
                    if not is_sem_peer and (len(safe_distractors) + len(candidates) > target_count):
                        continue
                    distractor_metadata[cand] = "preposition_valency"
                elif wn_pos == "a":
                    if anchor_type == "prep" and any(cand in adjs for p, adjs in cls._ADJ_PREP_VALENCY_MAP.items() if p != context_anchor):
                        distractor_metadata[cand] = "preposition_valency"
                    elif cand in tier2_satellites:
                        distractor_metadata[cand] = "satellite_synonym"
                    elif cand in tier3_attributes:
                        distractor_metadata[cand] = "attribute_coordinate"
                    else:
                        distractor_metadata[cand] = "synonym"
                else:
                    if cand in tier2_hyponyms:
                        distractor_metadata[cand] = "troponym"
                    elif cand in tier3_coordinates:
                        distractor_metadata[cand] = "coordinate"
                    else:
                        distractor_metadata[cand] = "synonym"
                if len(safe_distractors) >= target_count:
                    break

        # Fallback from Academic Semantic Families if candidates pool is sparse (e.g. specialized adjectives)
        if len(safe_distractors) < target_count and wn_pos == "a":
            for fam in cls._ACADEMIC_ADJ_FAMILIES:
                if clean_target in fam:
                    for a in fam:
                        if a != clean_target and a not in forbidden_words and a not in exclude_words and a not in safe_distractors:
                            if anchor_type == "modified_noun" and a in cls._PREDICATIVE_ONLY_ADJS:
                                continue
                            safe_distractors.append(a)
                            distractor_metadata[a] = "academic_family"
                            if len(safe_distractors) >= target_count:
                                break
                    break

        # Fallback from Academic Semantic Families if candidates pool is sparse (e.g. specialized verbs)
        if len(safe_distractors) < target_count and wn_pos == "v":
            academic_verb_families = [
                ["transmit", "distribute", "circulate", "relay", "deliver", "publish", "release"],
                ["instruct", "direct", "guide", "train", "advise", "command", "brief", "inform"],
                ["acquire", "obtain", "gain", "attain", "secure", "gather", "accumulate", "possess"],
                ["challenge", "assess", "evaluate", "test", "question", "examine", "review", "probe"],
                ["establish", "maintain", "sustain", "create", "develop", "foster", "initiate", "generate"]
            ]
            for fam in academic_verb_families:
                if clean_target in fam:
                    for v in fam:
                        if v != clean_target and v not in forbidden_words and v not in exclude_words and v not in safe_distractors:
                            safe_distractors.append(v)
                            distractor_metadata[v] = "academic_family"
                            if len(safe_distractors) >= target_count:
                                break
                    break

        # For concrete/substance/person nouns, if candidate pool is still sparse (< target_count),
        # harvest domain-aligned same-lexfile peers BEFORE orthographic paronyms so concrete substances (e.g. 'clay')
        # get genuine substance peers ('sand', 'dirt', 'dust') rather than orthographic rhymes ('clue'),
        # and person nouns (e.g. 'scientist') get 2 person coordinates + 1 morphological peer ("2 + 1").
        _is_person_target = wn_pos == "n" and bool(target_synsets) and all(
            s.lexfile() == "noun.person" for s in target_synsets if s.lexfile()
        )
        _pre_fill_cap = (target_count - 1) if (_is_person_target and not is_concrete_physical_noun) else target_count
        if len(safe_distractors) < _pre_fill_cap and wn_pos == "n":
            target_syns_pre = [s for s in target_synsets if s.lexfile()]
            target_lfs_pre = [s.lexfile() for s in target_syns_pre if s.lexfile()]
            if target_lfs_pre:
                if not cls._pedlex_lexfile_cache:
                    for w in cls.get_pedlex_headwords():
                        if (
                            len(w) >= 3
                            and w.isalpha()
                            and not re.fullmatch(r"(.)\1+", w)
                            and cls.is_exact_pedlex_headword(w)
                            and cls.word_zipf(w) >= 3.0
                        ):
                            w_syns = [ws for ws in wn.synsets(w, "n") if ws.lexfile()]
                            if w_syns:
                                lf = w_syns[0].lexfile()
                                if lf not in cls._pedlex_lexfile_cache:
                                    cls._pedlex_lexfile_cache[lf] = []
                                cls._pedlex_lexfile_cache[lf].append(w)
                _ref_syn = target_syns_pre[0]
                _ref_hyp = _ref_syn.hypernyms()[0] if _ref_syn.hypernyms() else _ref_syn
                for t_lf in target_lfs_pre:
                    raw_lf_pool = [
                        fb for fb in cls._pedlex_lexfile_cache.get(t_lf, [])
                        if (
                            fb != clean_target
                            and fb not in forbidden_words
                            and fb not in exclude_words
                            and fb not in global_synonyms
                            and fb not in safe_distractors
                            and not _is_primarily_adj_or_verb(fb)
                            and not _is_informal_or_slang_distractor(fb)
                            and _is_aligned_noun_coord(fb, _ref_syn, _ref_hyp)
                            and cls.is_cefr_compliant_distractor(fb, clean_target)
                        )
                    ]
                    if "_coord_sort_key" in locals():
                        raw_lf_pool.sort(key=_coord_sort_key)
                    for fb in raw_lf_pool:
                        if (
                            not cls.are_same_word_family(fb, clean_target)
                            and not any(cls.are_same_word_family(fb, s) for s in safe_distractors)
                            and not cls.double_key_collision(
                                clean_target, fb, anchor=context_anchor, anchor_type=anchor_type, pos=pos, quote=quote
                            )[0]
                        ):
                            safe_distractors.append(fb)
                            distractor_metadata[fb] = "coordinate"
                            if len(safe_distractors) >= _pre_fill_cap:
                                break
                    if len(safe_distractors) >= _pre_fill_cap:
                        break

        # 2-hop WordNet Semantic-Field Harvest for sparse verbs & adjectives before corpus fallback:
        # (1) Verbs: grandparent hypernym cousins ('admire' -> 'appreciate'/'value'/'consider', 'trap' -> 'release'/'waste'/'remove')
        # (2) Adjectives: similar_to -> head-antonym & its satellites ('renewable' -> 'exhaustible', 'habitable' -> 'unlivable')
        if len(safe_distractors) < target_count and wn_pos in ("v", "a") and target_synsets:
            two_hop_cands: List[Tuple[str, str]] = []
            two_hop_seen = set(safe_distractors) | set(exclude_words) | {clean_target}
            if wn_pos == "v":
                for _s in target_synsets[:2]:
                    for _h in _s.hypernyms():
                        for _gp in _h.hypernyms():
                            for _uncle in _gp.hyponyms():
                                for _lem in _uncle.lemmas():
                                    _lw = _lem.lower()
                                    if (
                                        _lw not in two_hop_seen
                                        and _lw.isalpha()
                                        and cls.is_exact_pedlex_headword(_lw)
                                        and cls.is_cefr_compliant_distractor(_lw, clean_target)
                                    ):
                                        two_hop_seen.add(_lw)
                                        two_hop_cands.append((_lw, "coordinate"))
                                        if len(two_hop_cands) >= 30:
                                            break
                                if len(two_hop_cands) >= 30:
                                    break
                            if len(two_hop_cands) >= 30:
                                break
                        if len(two_hop_cands) >= 30:
                            break
                    if len(two_hop_cands) >= 30:
                        break
            elif wn_pos == "a":
                for _s in target_synsets[:3]:
                    for _sim in _s.relations().get("similar", []):
                        for _sn in _sim.senses():
                            for _ant_sn in _sn.relations().get("antonym", []):
                                _aw = _ant_sn.word().lemma().lower()
                                if (
                                    _aw not in two_hop_seen
                                    and _aw.isalpha()
                                    and cls.is_exact_pedlex_headword(_aw)
                                    and cls.is_cefr_compliant_distractor(_aw, clean_target)
                                ):
                                    two_hop_seen.add(_aw)
                                    two_hop_cands.append((_aw, "antonym"))
                                for _ant_sim in _ant_sn.synset().relations().get("similar", []):
                                    for _as_lem in _ant_sim.lemmas():
                                        _asw = _as_lem.lower()
                                        if (
                                            _asw not in two_hop_seen
                                            and _asw.isalpha()
                                            and cls.is_exact_pedlex_headword(_asw)
                                            and cls.is_cefr_compliant_distractor(_asw, clean_target)
                                        ):
                                            two_hop_seen.add(_asw)
                                            two_hop_cands.append((_asw, "attribute_coordinate"))
            if "_coord_sort_key" in locals() and two_hop_cands:
                two_hop_cands.sort(key=lambda pair: _coord_sort_key(pair[0]))
            for _cw, _cmech in two_hop_cands:
                if len(safe_distractors) >= target_count:
                    break
                if (
                    _cw not in forbidden_words
                    and not cls.are_same_word_family(_cw, clean_target)
                    and not any(cls.are_same_word_family(_cw, s) for s in safe_distractors)
                    and not cls.double_key_collision(
                        clean_target, _cw, anchor=context_anchor, anchor_type=anchor_type, pos=pos, quote=quote
                    )[0]
                    and not cls.evaluate_syntactic_slot_clash(
                        clean_target, _cw, anchor=context_anchor, anchor_type=anchor_type, pos=pos, quote=quote
                    )[0]
                ):
                    safe_distractors.append(_cw)
                    distractor_metadata[_cw] = _cmech

        # Collocational head-noun adjective harvest for sparse adjectives before corpus_fallback:
        # e.g. '... of ____ land' -> pedlex['land']['adj'] ('industrial', 'dry', 'native', 'public')
        #      '... of ____ energy' -> pedlex['energy']['adj'] ('nuclear', 'atomic', 'alternative')
        if len(safe_distractors) < target_count and wn_pos == "a" and context_anchor:
            _anchor_low = str(context_anchor).strip().lower()
            _m_adj_head = re.search(r"_{2,}\s+([a-z]{3,})\b", _anchor_low)
            _head_n = _m_adj_head.group(1) if _m_adj_head else (
                _anchor_low if (anchor_type == "modified_noun" and _anchor_low.isalpha() and len(_anchor_low) >= 3) else ""
            )
            if _head_n:
                _head_pedlex = cls.get_pedlex_collocations(_head_n) or {}
                _raw_adj_cands: List[str] = []
                _seen_adj = set(safe_distractors) | set(exclude_words) | set(forbidden_words) | set(global_synonyms) | {clean_target, _head_n}
                for _entry in _head_pedlex.get("adj", []):
                    for _tok in str(_entry).replace("/", " ").split():
                        _tok = _tok.strip().lower()
                        if (
                            _tok.isalpha()
                            and len(_tok) >= 3
                            and _tok not in _seen_adj
                            and cls.is_exact_pedlex_headword(_tok)
                            and any(ws.pos in ("a", "s") for ws in wn.synsets(_tok))
                            and cls.is_cefr_compliant_distractor(_tok, clean_target)
                        ):
                            _tok_ent = cls.get_ldoce_entry(_tok) or {}
                            _tok_senses = _tok_ent.get("senses") or []
                            if _tok_senses and any(cls._pos_code(str(sn.get("pos") or "")) == "adj" for sn in _tok_senses[:2]):
                                _seen_adj.add(_tok)
                                _raw_adj_cands.append(_tok)
                if "_coord_sort_key" in locals() and _raw_adj_cands:
                    _raw_adj_cands.sort(key=_coord_sort_key)
                for _aw in _raw_adj_cands:
                    if len(safe_distractors) >= target_count:
                        break
                    if (
                        not cls.are_same_word_family(_aw, clean_target)
                        and not any(cls.are_same_word_family(_aw, s) for s in safe_distractors)
                        and not cls.double_key_collision(
                            clean_target, _aw, anchor=context_anchor, anchor_type=anchor_type, pos=pos, quote=quote
                        )[0]
                        and not cls.evaluate_syntactic_slot_clash(
                            clean_target, _aw, anchor=context_anchor, anchor_type=anchor_type, pos=pos, quote=quote
                        )[0]
                    ):
                        safe_distractors.append(_aw)
                        distractor_metadata[_aw] = "collocation"

        # Fallback for Focus / Degree / Approximation / Discourse / Temporal Adverbs before Orthographic Paronym Tier
        # (so focus/degree/discourse/temporal adverbs like 'merely', 'virtually', 'barely', 'similarly', 'nowadays'
        # get semantic/functional contrasts rather than orthographic rhymes or generic manner adverbs)
        if len(safe_distractors) < target_count and wn_pos == "r" and not is_fronted_inv_slot:
            _adv_contrast_map = [
                (
                    {"merely", "solely", "purely", "simply", "exclusively", "just"},
                    ["entirely", "completely", "thoroughly", "utterly", "fully"],
                ),
                (
                    {"virtually", "practically", "nearly", "almost", "essentially", "effectively"},
                    ["barely", "scarcely", "hardly", "rarely", "seldom"],
                ),
                (
                    {"barely", "scarcely", "hardly", "narrowly"},
                    ["entirely", "completely", "thoroughly", "fully", "totally"],
                ),
                (
                    {"similarly", "likewise", "equally", "correspondingly"},
                    ["conversely", "otherwise", "instead", "meanwhile", "nevertheless"],
                ),
                (
                    {"conversely", "contrarily", "alternatively", "otherwise", "instead"},
                    ["similarly", "likewise", "moreover", "furthermore", "meanwhile"],
                ),
                (
                    {"however", "nevertheless", "nonetheless", "still", "yet"},
                    ["therefore", "consequently", "moreover", "furthermore", "similarly"],
                ),
                (
                    {"therefore", "consequently", "thus", "hence", "accordingly"},
                    ["however", "nevertheless", "otherwise", "meanwhile", "instead"],
                ),
                (
                    {"moreover", "furthermore", "besides", "additionally"},
                    ["however", "otherwise", "instead", "conversely", "nevertheless"],
                ),
                (
                    {"nowadays", "currently", "presently", "today"},
                    ["previously", "formerly", "rarely", "seldom", "eventually"],
                ),
                (
                    {"previously", "formerly", "originally", "initially", "earlier"},
                    ["currently", "nowadays", "eventually", "finally", "ultimately"],
                ),
                (
                    {"recently", "lately", "newly"},
                    ["previously", "formerly", "rarely", "eventually", "originally"],
                ),
                (
                    {"eventually", "ultimately", "finally"},
                    ["initially", "originally", "previously", "rarely", "seldom"],
                ),
                (
                    {"thereafter", "subsequently", "afterwards", "afterward"},
                    ["previously", "formerly", "initially", "originally", "meanwhile"],
                ),
                (
                    {"predominantly", "chiefly", "mainly", "primarily", "largely", "mostly"},
                    ["partly", "partially", "barely", "scarcely", "rarely", "equally"],
                ),
                (
                    {"decidedly", "undoubtedly", "unquestionably", "indisputably", "undeniably", "markedly"},
                    ["barely", "scarcely", "slightly", "faintly", "hardly", "marginally"],
                ),
                (
                    {"somehow", "anyhow"},
                    ["nowhere", "never", "seldom", "rarely", "barely", "scarcely"],
                ),
                (
                    {"vaguely", "faintly", "dimly", "obscurely"},
                    ["clearly", "vividly", "distinctly", "precisely", "explicitly", "sharply"],
                ),
                (
                    {"stubbornly", "obstinately", "defiantly", "doggedly"},
                    ["willingly", "readily", "gladly", "eagerly", "calmly", "gently"],
                ),
                (
                    {"conveniently", "smoothly", "effortlessly", "handily"},
                    ["awkwardly", "clumsily", "painfully", "barely", "reluctantly"],
                ),
                (
                    {"responsibly", "carefully", "cautiously", "prudently", "sensibly"},
                    ["recklessly", "carelessly", "rashly", "blindly", "foolishly"],
                ),
                (
                    {"passionately", "fiercely", "fervently", "intensely", "enthusiastically"},
                    ["calmly", "coldly", "indifferently", "mildly", "casually", "reluctantly"],
                ),
                (
                    {"happily", "cheerfully", "joyfully", "gladly", "merrily"},
                    ["sadly", "bitterly", "gloomily", "reluctantly", "anxiously", "angrily"],
                ),
                (
                    {"online", "offline"},
                    ["abroad", "overseas", "indoors", "outdoors", "locally", "remotely"],
                ),
                (
                    {"abroad", "overseas"},
                    ["indoors", "outdoors", "online", "locally", "remotely", "nearby"],
                ),
                (
                    {"indoors", "outdoors", "upstairs", "downstairs", "underground"},
                    ["abroad", "overseas", "online", "locally", "remotely", "nearby"],
                ),
            ]
            for _src_set, _contrast_list in _adv_contrast_map:
                if clean_target in _src_set:
                    for _adv_c in _contrast_list:
                        if (
                            _adv_c != clean_target
                            and _adv_c not in forbidden_words
                            and _adv_c not in exclude_words
                            and _adv_c not in safe_distractors
                            and not cls.are_same_word_family(_adv_c, clean_target)
                            and not any(cls.are_same_word_family(_adv_c, s) for s in safe_distractors)
                        ):
                            safe_distractors.append(_adv_c)
                            distractor_metadata[_adv_c] = "adv_semantic_contrast"
                            if len(safe_distractors) >= target_count:
                                break
                    break

        # Orthographic / Paronym Distractor Tier (with Cognate Near-Synonym Guard):
        # 1. If candidate pool is sparse (len(safe_distractors) < target_count), fill remaining slots with
        #    verified orthographic paronyms / orthogonal-domain root peers.
        # 2. If all 3 slots were filled, blend 1 paronym into Slot 3 when:
        #    - Slot 3 is a generic coordinate/troponym/attribute_coordinate or a 3rd antonym and a sharp
        #      paronym (score >= 45, e.g. adapt/adopt, complement/compliment, simulate/stimulate,
        #      industrial/industrious, considerate/considerable, respectful/respective) exists, OR
        #    - All 3 slots are generic coordinates and a clean orthogonal-domain root peer (score >= 28,
        #      e.g. specimen -> spectrum/spectacle/speculation) exists.
        if wn_pos in ("n", "v", "a", "r") and not is_fronted_inv_slot:
            need_paronym_fill = len(safe_distractors) < target_count
            pure_generic_coords = (
                len(safe_distractors) >= target_count
                and all(
                    distractor_metadata.get(d) in ("coordinate", "troponym", "attribute_coordinate")
                    for d in safe_distractors
                )
            )
            slot3_replaceable = (
                len(safe_distractors) >= target_count
                and (
                    (
                        distractor_metadata.get(safe_distractors[-1]) in (
                            "coordinate", "troponym", "attribute_coordinate", "ldoce_thesaurus",
                            *(("synonym",) if wn_pos == "n" else ()),
                        )
                        and (
                            wn_pos in ("n", "v")
                            or all(
                                distractor_metadata.get(d) not in ("antonym", "ldoce_thesaurus")
                                for d in safe_distractors[:-1]
                            )
                        )
                    )
                    or sum(1 for d in safe_distractors if distractor_metadata.get(d) == "antonym") >= 2
                )
            )
            if need_paronym_fill or pure_generic_coords or slot3_replaceable:
                paronym_cands = cls.get_orthographic_paronym_distractors(
                    clean_target,
                    pos=wn_pos,
                    target_count=target_count,
                    exclude_words=(exclude_words | forbidden_words | set(safe_distractors)),
                    context_anchor=context_anchor,
                    anchor_type=anchor_type,
                    quote=quote,
                    return_scores=True,
                )
                if need_paronym_fill:
                    for p_word, _p_score in paronym_cands:
                        if not (
                            len(clean_target) >= 5
                            and (
                                cls._common_prefix_len(clean_target, p_word) >= 3
                                or cls._common_suffix_len(clean_target, p_word) >= (3 if wn_pos == "n" else 4)
                                or _p_score >= 45
                            )
                        ):
                            continue
                        if wn_pos == "n":
                            if target_strictly_uncountable and cls._is_primarily_countable_noun(p_word):
                                if sum(1 for s in safe_distractors if cls._is_primarily_countable_noun(s)) >= 1:
                                    continue
                            elif target_strictly_countable and cls._is_primarily_uncountable_noun(p_word):
                                if sum(1 for s in safe_distractors if cls._is_primarily_uncountable_noun(s)) >= 1:
                                    continue
                        if p_word not in safe_distractors:
                            safe_distractors.append(p_word)
                            distractor_metadata[p_word] = "orthographic_paronym"
                            if len(safe_distractors) >= target_count:
                                break
                elif paronym_cands:
                    for best_p_word, best_p_score in paronym_cands:
                        # "2 + 1" Standard Distractor Blending:
                        # - For nouns (`wn_pos == 'n'`), Slot 3 is populated by a high-discrimination morphological/phonetic peer when:
                        #   (a) Multi-syllable root-sharing paronym (`len >= 7` and `pref >= 4`, e.g. specimen/spectrum, emergence/emergency, scientist/science),
                        #   (b) Stressed `-ion` (`-tion`/`-sion`) rhyme peer (`recession` -> `concession`/`succession`/`obsession`, `collision` -> `precision`/`coalition`),
                        #   (c) Academic suffix family peer (`-ics`, `-ist`, `-ism`, `-ship`, `-ware`, `-type`, `-ology`, e.g. `electronics`/`physics`, `scientist`/`physicist`, `optimism`/`criticism`, `hardware`/`software`),
                        #   (d) Rhyme-containing noun peer (`min_len >= 5` and suffix containment, e.g. `venue` -> `revenue`/`avenue`),
                        #   while never popping concrete coordinates for short everyday words (shopper, heat, cricket, clue, coat).
                        if wn_pos == "n":
                            _acad_sfx = ("ics", "ist", "ism", "ship", "ware", "type", "ology", "archy", "cracy")
                            is_true_orthographic_peer = (
                                (len(clean_target) >= 7 and cls._common_prefix_len(clean_target, best_p_word) >= 4)
                                or (
                                    clean_target.endswith(("tion", "sion"))
                                    and best_p_word.endswith(("tion", "sion"))
                                    and cls._common_suffix_len(clean_target, best_p_word) >= 4
                                    and cls._phonetic_rhyme_tail(clean_target) == cls._phonetic_rhyme_tail(best_p_word)
                                )
                                or (
                                    clean_target.endswith(("ance", "ence", "ment"))
                                    and best_p_word.endswith(("ance", "ence", "ment"))
                                    and cls._common_suffix_len(clean_target, best_p_word) >= 5
                                    and cls._phonetic_rhyme_tail(clean_target) == cls._phonetic_rhyme_tail(best_p_word)
                                )
                                or any(clean_target.endswith(sf) and best_p_word.endswith(sf) for sf in _acad_sfx)
                                or (
                                    min(len(clean_target), len(best_p_word)) >= 5
                                    and abs(len(clean_target) - len(best_p_word)) <= 3
                                    and (best_p_word.endswith(clean_target) or clean_target.endswith(best_p_word))
                                )
                            )
                        else:
                            is_true_orthographic_peer = (
                                clean_target[0] == best_p_word[0]
                                or cls._common_suffix_len(clean_target, best_p_word) >= 4
                            )
                        can_replace_coord = is_true_orthographic_peer and (
                            best_p_score >= 45
                            or (
                                pure_generic_coords
                                and best_p_score >= 28
                                and cls._common_prefix_len(clean_target, best_p_word) >= 4
                            )
                        )
                        replace_idx = -1
                        if wn_pos == "n" and can_replace_coord:
                            if target_strictly_uncountable and cls._is_primarily_countable_noun(best_p_word):
                                c_indices = [idx for idx, s in enumerate(safe_distractors) if cls._is_primarily_countable_noun(s)]
                                if len(c_indices) >= 1:
                                    replace_idx = c_indices[-1]
                                else:
                                    can_replace_coord = False
                            elif target_strictly_countable and cls._is_primarily_uncountable_noun(best_p_word):
                                u_indices = [idx for idx, s in enumerate(safe_distractors) if cls._is_primarily_uncountable_noun(s)]
                                if len(u_indices) >= 1:
                                    replace_idx = u_indices[-1]
                                else:
                                    can_replace_coord = False
                        if can_replace_coord and best_p_word not in safe_distractors:
                            popped = safe_distractors.pop(replace_idx)
                            distractor_metadata.pop(popped, None)
                            safe_distractors.append(best_p_word)
                            distractor_metadata[best_p_word] = "orthographic_paronym"
                            break

        # Tier 3.9: Noun Countability-Contrast Safe Distractor Slot ("countability_contrast_uncountable")
        # For countable singular/plural or [plural] nouns in `_NOUN_COUNTABILITY_CONTRAST_MAP`, inject at most 1
        # semantically related pure-[uncountable] peer (e.g. device/devices -> equipment/hardware; suggestion -> advice;
        # breakthrough -> progress; belongings -> luggage; finding -> information/evidence) into Slot 3 so the
        # countable/plural determiner frame deterministically rules out the uncountable distractor with 0 double-key risk.
        if wn_pos == "n" and not clean_target.endswith("ics"):
            t_gram_prof_n = cls.get_word_gram_profile(clean_target, pos="noun")
            is_countable_or_plural_noun_target = bool(
                target_strictly_countable
                or t_gram_prof_n.get("pure_countable")
                or t_gram_prof_n.get("pure_plural")
                or (
                    clean_target.endswith("s")
                    and not target_strictly_uncountable
                    and not t_gram_prof_n.get("pure_uncountable")
                )
            )
            if is_countable_or_plural_noun_target:
                t_lem_n = cls.lemma_of(clean_target)
                thes_uncountable_peers: List[str] = []
                t_entry_n = cls.get_ldoce_entry(t_lem_n) or cls.get_ldoce_entry(clean_target)
                if t_entry_n:
                    n_senses = [
                        s for s in (t_entry_n.get("senses") or [])
                        if "noun" in (s.get("pos") or t_entry_n.get("pos") or "").lower()
                    ]
                    is_single_noun_sense = len(n_senses) <= 1
                    d_words_n = {
                        w for w in re.findall(r"\b[a-zA-Z]{3,}\b", (definition or "").lower())
                        if w not in {"the", "and", "for", "with", "from", "that", "this", "something", "someone", "somebody"}
                    }
                    q_words_n = {
                        w for w in re.findall(r"\b[a-zA-Z]{3,}\b", (quote or "").lower())
                        if w not in {"the", "and", "for", "with", "from", "that", "this", "something", "someone", "somebody"}
                    }
                    thes_list_n = t_entry_n.get("thesaurus") or []
                    self_verified_clusters: Set[Any] = set()
                    for th_it in thes_list_n:
                        th_w = str(th_it.get("word") or "").strip().lower()
                        if th_w in (clean_target, t_lem_n):
                            th_dist_toks = set(re.findall(r"\b[a-zA-Z]{3,}\b", str(th_it.get("distinction") or "").lower()))
                            if is_single_noun_sense or not d_words_n or (th_dist_toks & (d_words_n | q_words_n)):
                                self_verified_clusters.add(th_it.get("section_index"))
                    for th_it in thes_list_n:
                        th_w = str(th_it.get("word") or "").strip().lower()
                        th_g = str(th_it.get("grammar") or "").strip().lower()
                        if (
                            th_w
                            and th_w not in (clean_target, t_lem_n)
                            and " " not in th_w
                            and "/" not in th_w
                            and "uncountable" in th_g
                            and not re.search(r"\bcountable\b", th_g)
                            and (is_single_noun_sense or th_it.get("section_index") in self_verified_clusters)
                        ):
                            if th_w not in thes_uncountable_peers:
                                thes_uncountable_peers.append(th_w)

                # If a sense-aligned uncountable peer (or pure_uncountable noun) was already selected in an earlier tier
                # (e.g. Tier 0 ldoce_thesaurus 'credit' for 'loan'), promote its metadata so we don't inject a 2nd uncountable foil.
                for ex_d in safe_distractors:
                    if (
                        ex_d in thes_uncountable_peers
                        or cls.get_word_gram_profile(ex_d, pos="noun").get("pure_uncountable")
                    ):
                        distractor_metadata[ex_d] = "countability_contrast_uncountable"
                        break

            if is_countable_or_plural_noun_target and not any(
                v == "countability_contrast_uncountable" for v in distractor_metadata.values()
            ):
                t_lem_n = cls.lemma_of(clean_target)
                contrast_pool: List[str] = list(thes_uncountable_peers)
                for key_w in (clean_target, t_lem_n):
                    for u_cand in cls._NOUN_COUNTABILITY_CONTRAST_MAP.get(key_w, []):
                        if u_cand not in contrast_pool:
                            contrast_pool.append(u_cand)
                for u_cand in list(t_gram_prof_n.get("comparison_peers") or []):
                    u_clean = str(u_cand or "").strip().lower()
                    if (
                        u_clean
                        and u_clean not in contrast_pool
                        and " " not in u_clean
                        and "_" not in u_clean
                        and "-" not in u_clean
                        and cls.get_word_gram_profile(u_clean, pos="noun").get("pure_uncountable")
                    ):
                        contrast_pool.append(u_clean)
                # For singular countable targets, prefer uncountable foils sharing the target's vowel/consonant initial
                # so indefinite article frames ('... repay a ____') are never forced into mixed-initial rewrites.
                if not clean_target.endswith("s") and clean_target[:1].isalpha():
                    t_is_vowel = clean_target[0].lower() in "aeiou"
                    contrast_pool.sort(key=lambda w: 0 if (bool(w) and (w[0].lower() in "aeiou") == t_is_vowel) else 1)
                picked_uncountable: Optional[str] = None
                for u_word in contrast_pool:
                    if (
                        u_word != clean_target
                        and u_word != t_lem_n
                        and u_word not in safe_distractors
                        and u_word not in exclude_words
                        and not cls.are_same_word_family(u_word, clean_target)
                        and cls.is_exact_pedlex_headword(u_word)
                        and (
                            u_word in thes_uncountable_peers
                            or cls.get_word_gram_profile(u_word, pos="noun").get("pure_uncountable")
                        )
                        and cls.is_cefr_compliant_distractor(u_word, clean_target)
                    ):
                        picked_uncountable = u_word
                        break
                if picked_uncountable:
                    if len(safe_distractors) < target_count:
                        safe_distractors.append(picked_uncountable)
                        distractor_metadata[picked_uncountable] = "countability_contrast_uncountable"
                    elif len(safe_distractors) >= target_count:
                        last_d = safe_distractors[-1]
                        if distractor_metadata.get(last_d) in (
                            "coordinate", "troponym", "attribute_coordinate",
                            "synonym", "ldoce_thesaurus", "dynamic_domain_fallback",
                        ):
                            safe_distractors.pop()
                            distractor_metadata.pop(last_d, None)
                            safe_distractors.append(picked_uncountable)
                            distractor_metadata[picked_uncountable] = "countability_contrast_uncountable"

            # Part B: Symmetrical Countability Contrast for Strictly Uncountable Noun Targets
            # Inject at most 1 semantically related countable foil (e.g. advice -> suggestion/tip;
            # equipment -> tool/device; information -> fact/detail; furniture -> chair/desk;
            # luggage -> belongings/item; progress -> breakthrough/advance) into safe_distractors.
            is_strictly_uncountable_noun_target = bool(
                target_strictly_uncountable
                or t_gram_prof_n.get("pure_uncountable")
            )
            if is_strictly_uncountable_noun_target:
                t_lem_n = cls.lemma_of(clean_target)
                rev_map = cls.get_noun_countability_contrast_map(reverse=True)
                curated_contrast_cands: List[str] = []
                for key_w in (clean_target, t_lem_n):
                    for c_cand in rev_map.get(key_w, []):
                        if c_cand not in curated_contrast_cands:
                            curated_contrast_cands.append(c_cand)

                # 1. Promote or normalize existing countable distractor metadata (protecting paronyms and antonyms)
                curated_in_distractors = [d for d in safe_distractors if d in curated_contrast_cands]
                if curated_in_distractors:
                    distractor_metadata[curated_in_distractors[0]] = "countability_contrast_countable"
                elif not curated_contrast_cands:
                    c_distractors = [
                        d for d in safe_distractors
                        if cls._is_primarily_countable_noun(d)
                        and distractor_metadata.get(d) not in ("orthographic_paronym", "antonym")
                    ]
                    if c_distractors:
                        distractor_metadata[c_distractors[0]] = "countability_contrast_countable"

                # Prune excess countable distractors beyond the 1-foil quota
                c_distractors = [
                    d for d in safe_distractors
                    if cls._is_primarily_countable_noun(d)
                    and distractor_metadata.get(d) not in ("orthographic_paronym", "antonym")
                ]
                if len(c_distractors) > 1:
                    keep_c = curated_in_distractors[0] if curated_in_distractors else c_distractors[0]
                    for extra_c in list(c_distractors):
                        if extra_c != keep_c and extra_c in safe_distractors:
                            safe_distractors.remove(extra_c)
                            distractor_metadata.pop(extra_c, None)

                # 2. If no countable foil is present yet, or a curated foil is available and missing, inject at most 1 from reverse countability contrast map or thesaurus
                has_curated_contrast = any(d in curated_contrast_cands for d in safe_distractors)
                need_curated_injection = bool(curated_contrast_cands and not has_curated_contrast)
                if not any(v == "countability_contrast_countable" for v in distractor_metadata.values()) or need_curated_injection:
                    contrast_pool_c: List[str] = list(curated_contrast_cands)

                    # Also inspect LDOCE thesaurus for countable peers
                    t_entry_n = cls.get_ldoce_entry(t_lem_n) or cls.get_ldoce_entry(clean_target)
                    if t_entry_n:
                        thes_list_n = t_entry_n.get("thesaurus") or []
                        for th_it in thes_list_n:
                            th_w = str(th_it.get("word") or "").strip().lower()
                            th_g = str(th_it.get("grammar") or "").strip().lower()
                            if (
                                th_w
                                and th_w not in (clean_target, t_lem_n)
                                and " " not in th_w
                                and "/" not in th_w
                                and "countable" in th_g
                                and not re.search(r"\buncountable\b", th_g)
                                and th_w not in contrast_pool_c
                            ):
                                contrast_pool_c.append(th_w)

                    picked_countable: Optional[str] = None
                    for c_word in contrast_pool_c:
                        if (
                            c_word != clean_target
                            and c_word != t_lem_n
                            and c_word not in safe_distractors
                            and c_word not in exclude_words
                            and not cls.are_same_word_family(c_word, clean_target)
                            and cls.is_exact_pedlex_headword(c_word)
                            and cls._is_primarily_countable_noun(c_word)
                            and cls.is_cefr_compliant_distractor(c_word, clean_target)
                        ):
                            picked_countable = c_word
                            break

                    if picked_countable:
                        # If safe_distractors has an un-curated countable distractor, replace it
                        uncurated_c = [
                            d for d in safe_distractors
                            if cls._is_primarily_countable_noun(d)
                            and d not in curated_contrast_cands
                            and distractor_metadata.get(d) not in ("orthographic_paronym", "antonym")
                        ]
                        if need_curated_injection and picked_countable in curated_contrast_cands:
                            if uncurated_c:
                                rep_d = uncurated_c[0]
                                rep_idx = safe_distractors.index(rep_d)
                                safe_distractors[rep_idx] = picked_countable
                                distractor_metadata.pop(rep_d, None)
                                distractor_metadata[picked_countable] = "countability_contrast_countable"
                            elif len(safe_distractors) >= target_count:
                                last_d = safe_distractors[-1]
                                if distractor_metadata.get(last_d) in (
                                    "coordinate", "troponym", "attribute_coordinate",
                                    "synonym", "ldoce_thesaurus", "dynamic_domain_fallback",
                                ):
                                    safe_distractors.pop()
                                    distractor_metadata.pop(last_d, None)
                                    safe_distractors.append(picked_countable)
                                    distractor_metadata[picked_countable] = "countability_contrast_countable"
                        elif len(safe_distractors) < target_count:
                            safe_distractors.append(picked_countable)
                            distractor_metadata[picked_countable] = "countability_contrast_countable"

        # Second Defense: Dynamic domain-aligned fallback using target WordNet lexfile
        # When candidates pool is still sparse, extract homogeneous fallback words from the target's
        # sense-locked WordNet lexfile (e.g. noun.cognition, noun.group) to prevent Domain/Category Mismatch.
        if len(safe_distractors) < target_count and wn_pos == "n":
            target_syns = [s for s in target_synsets if s.lexfile()] or [s for s in wn.synsets(clean_target, "n") if s.lexfile()]
            target_lfs = [s.lexfile() for s in target_syns if s.lexfile()]
            if target_lfs:
                # Lazy-build or retrieve lexfile-indexed PedLex nouns
                if not cls._pedlex_lexfile_cache:
                    for w in cls.get_pedlex_headwords():
                        if (
                            len(w) >= 3
                            and w.isalpha()
                            and not re.fullmatch(r"(.)\1+", w)
                            and cls.is_exact_pedlex_headword(w)
                            and cls.word_zipf(w) >= 3.0
                        ):
                            w_syns = [ws for ws in wn.synsets(w, "n") if ws.lexfile()]
                            if w_syns:
                                lf = w_syns[0].lexfile()
                                if lf not in cls._pedlex_lexfile_cache:
                                    cls._pedlex_lexfile_cache[lf] = []
                                cls._pedlex_lexfile_cache[lf].append(w)
                for t_lf in target_lfs:
                    raw_fb_pool = [
                        fb for fb in cls._pedlex_lexfile_cache.get(t_lf, [])
                        if (
                            fb != clean_target
                            and fb not in forbidden_words
                            and fb not in exclude_words
                            and fb not in global_synonyms
                            and fb not in safe_distractors
                            and not _is_primarily_adj_or_verb(fb)
                            and not _is_informal_or_slang_distractor(fb)
                            and cls.is_cefr_compliant_distractor(fb, clean_target)
                        )
                    ]
                    if "_coord_sort_key" in locals():
                        raw_fb_pool.sort(key=_coord_sort_key)
                    else:
                        raw_fb_pool.sort(key=lambda x: -cls._zipf_freq(x))
                    for fb in raw_fb_pool:
                        if target_strictly_uncountable:
                            if not cls._is_primarily_uncountable_noun(fb):
                                _fb_ent = cls.get_ldoce_entry(fb) or {}
                                _fb_sns = [
                                    sn for sn in (_fb_ent.get("senses") or [])
                                    if not sn.get("pos") or cls._pos_code(sn.get("pos")) == "n"
                                ] or (_fb_ent.get("senses") or [])
                                if not _fb_sns or "uncountable" not in " ".join(str(sn.get("grammar") or "").lower() for sn in _fb_sns[:2]):
                                    continue
                        if (
                            not cls.are_same_word_family(fb, clean_target)
                            and not any(cls.are_same_word_family(fb, s) for s in safe_distractors)
                            and not cls.double_key_collision(
                                clean_target, fb, anchor=context_anchor, anchor_type=anchor_type, pos=pos, quote=quote
                            )[0]
                        ):
                            safe_distractors.append(fb)
                            distractor_metadata[fb] = "dynamic_domain_fallback"
                            if len(safe_distractors) >= target_count:
                                break
                    if len(safe_distractors) >= target_count:
                        break

            if len(safe_distractors) < target_count and target_strictly_uncountable:
                core_unc_pool = [
                    "guidance", "counsel", "feedback", "information", "equipment", "hardware",
                    "machinery", "progress", "research", "evidence", "assistance", "funding",
                    "legislation", "pollution", "luggage", "baggage", "furniture", "property",
                ]
                for unc_cand in (list(near_set_for_pile) + core_unc_pool):
                    if (
                        unc_cand != clean_target
                        and unc_cand not in safe_distractors
                        and unc_cand not in forbidden_words
                        and unc_cand not in exclude_words
                        and cls._is_primarily_uncountable_noun(unc_cand)
                        and not cls.are_same_word_family(unc_cand, clean_target)
                        and not any(cls.are_same_word_family(unc_cand, s) for s in safe_distractors)
                        and cls.is_cefr_compliant_distractor(unc_cand, clean_target)
                        and not cls.double_key_collision(
                            clean_target, unc_cand, anchor=context_anchor, anchor_type=anchor_type, pos=pos, quote=quote
                        )[0]
                    ):
                        safe_distractors.append(unc_cand)
                        distractor_metadata[unc_cand] = "uncountable_domain_peer"
                        if len(safe_distractors) >= target_count:
                            break

        # Second Defense for Adjectives (`wn_pos == "a"`): Morphological & Syntactic-Slot Aligned Peers
        # Prevents participial `-ing` adjectives (e.g. `rewarding`) or relational `adj.pert` adjectives (e.g. `doctoral`)
        # from falling back to generic `['simple', 'common', 'similar']`.
        if len(safe_distractors) < target_count and wn_pos == "a":
            _is_pert_adj = any(s.lexfile() == "adj.pert" for s in target_synsets if s.lexfile()) or bool(
                cls.get_word_gram_profile(clean_target, pos="adj").get("only_before_noun")
            )
            if clean_target.endswith("ing"):
                _adj_morph_pool = [
                    "demanding", "challenging", "exhausting", "frustrating", "promising",
                    "convincing", "appealing", "astonishing", "overwhelming", "confusing",
                    "alarming", "disturbing", "encouraging", "refreshing",
                ]
            elif _is_pert_adj or clean_target.endswith(("al", "ic", "ar", "ory", "ary", "ive")):
                _ory_ive_pool = [
                    "advisory", "compulsory", "mandatory", "voluntary", "legislative", "executive",
                    "administrative", "judicial", "disciplinary", "consultative", "investigative",
                    "corrective", "preventive", "selective",
                ]
                _al_ic_pool = [
                    "academic", "medical", "surgical", "clinical", "legal", "musical",
                    "vocational", "clerical", "dental", "naval", "royal", "rural",
                    "urban", "civil", "regional", "domestic", "internal", "external",
                ]
                if clean_target.endswith(("ory", "ive")):
                    _adj_morph_pool = _ory_ive_pool + [w for w in _al_ic_pool if w not in _ory_ive_pool]
                else:
                    _adj_morph_pool = _al_ic_pool + [w for w in _ory_ive_pool if w not in _al_ic_pool]
            elif clean_target.endswith("ed"):
                _adj_morph_pool = [
                    "Exhausted", "confused", "frustrated", "convinced", "ashamed",
                    "relieved", "pleased", "disappointed", "crowded", "isolated",
                ]
                _adj_morph_pool = [w.lower() for w in _adj_morph_pool]
            else:
                _adj_morph_pool = []
            for _m_adj in _adj_morph_pool:
                if (
                    _m_adj != clean_target
                    and _m_adj not in forbidden_words
                    and _m_adj not in exclude_words
                    and _m_adj not in global_synonyms
                    and _m_adj not in safe_distractors
                    and not cls.are_same_word_family(_m_adj, clean_target)
                    and not any(cls.are_same_word_family(_m_adj, s) for s in safe_distractors)
                    and cls.is_cefr_compliant_distractor(_m_adj, clean_target)
                    and not cls.evaluate_syntactic_slot_clash(
                        clean_target, _m_adj, anchor=context_anchor, anchor_type=anchor_type, pos="adj", quote=quote
                    )[0]
                    and not cls.double_key_collision(
                        clean_target, _m_adj, anchor=context_anchor, anchor_type=anchor_type, pos="adj", quote=quote
                    )[0]
                ):
                    safe_distractors.append(_m_adj)
                    distractor_metadata[_m_adj] = "morphological_peer"
                    if len(safe_distractors) >= target_count:
                        break

        # Negative Inversion Fallback: strictly select peer inversion triggers
        if len(safe_distractors) < target_count and is_fronted_inv_slot and clean_target in cls._NEGATIVE_INVERSION_ADVERBS:
            for inv_peer in ("seldom", "hardly", "scarcely", "barely", "little", "no sooner", "rarely"):
                if (
                    inv_peer != clean_target
                    and inv_peer not in safe_distractors
                    and inv_peer not in forbidden_words
                    and inv_peer not in exclude_words
                    and not cls.double_key_collision(
                        clean_target, inv_peer, anchor=context_anchor, anchor_type=anchor_type, pos=pos, quote=quote
                    )[0]
                ):
                    clash, clash_reason = cls.evaluate_syntactic_slot_clash(
                        clean_target, inv_peer, anchor=context_anchor, anchor_type=anchor_type, pos=pos, quote=quote
                    )
                    if clash and clash_reason == "non_inverting_candidate_in_fronted_inversion_slot":
                        continue
                    safe_distractors.append(inv_peer)
                    distractor_metadata[inv_peer] = "fronted_inv_peer"
                    if len(safe_distractors) >= target_count:
                        break

        # Universal Fallback if candidates pool is still sparse: select stratified core corpus items
        if len(safe_distractors) < target_count and not is_fronted_inv_slot:
            needed = target_count - len(safe_distractors)
            excl = set(forbidden_words) | set(exclude_words) | set(safe_distractors) | {clean_target}
            pref_vowel: Optional[bool] = None
            if wn_pos == "n" and not clean_target.endswith("s") and clean_target[:1].isalpha():
                pref_vowel = clean_target[0].lower() in "aeiou"
            fb_cands = cls.get_corpus_fallback_candidates(
                target_word=clean_target,
                pos=wn_pos or pos or "noun",
                target_count=needed,
                exclude_words=excl,
                prefer_initial_vowel=pref_vowel,
            )
            for fb in fb_cands:
                safe_distractors.append(fb)
                distractor_metadata[fb] = "corpus_fallback"
                if len(safe_distractors) >= target_count:
                    break

        final_distractors = safe_distractors[:target_count]
        if return_metadata:
            return final_distractors, {k: distractor_metadata.get(k, "distractor") for k in final_distractors}
        return final_distractors

