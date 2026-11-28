"""Multi-tier candidate harvesting for vocabulary distractors."""

from __future__ import annotations

import logging
import math
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger("librarian.linguistics.vocab_distractors")


class DistractorVocabHarvestMixin:

    _ABSTRACT_ONTOLOGY_HYPERNYMS = {
        "oewn-00024900-n",  # state
        "oewn-13943868-n",  # condition, status
        "oewn-05207437-n",  # state, province
        "oewn-00023100-n",  # quality
        "oewn-00024444-n",  # attribute
        "oewn-00002137-n",  # abstraction, abstract entity
        "oewn-00001740-n",  # entity
        "oewn-00002684-n",  # physical object
        "oewn-00003553-n",  # whole, unit
        "oewn-00007347-n",  # attribute
        "oewn-00030761-n",  # psychological feature
    }

    @classmethod
    def _harvest_vocab_candidates(
        cls,
        wsd_ctx: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Executes Phase 2: Multi-tier candidate harvesting and coordinate ranking."""
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
        sense_antonyms = list(wsd_ctx["sense_antonyms"])

        _pos_ok = lambda lem: cls._pos_ok(lem, wn_pos=wn_pos)
        _is_primarily_adj_or_verb = cls._is_primarily_adj_or_verb
        _is_primarily_noun_or_adj_for_verb = cls._is_primarily_noun_or_adj_for_verb
        _is_informal_or_slang_distractor = lambda w: cls._is_informal_or_slang_distractor(
            w, wn_pos=wn_pos, target_is_informal=target_is_informal, target_lfs_set=target_lfs_set
        )
        _coord_sort_key = None
        _is_aligned_noun_coord = lambda cand_lem, sis_syn=None, hyper_syn=None: True

        # PedLex 6th Edition Thesaurus, Grammar Box Comparison Peers & Activator Integration (Tier 0: Primary Pedagogical Lexical Source)
        tier0_ldoce_thesaurus: List[str] = []
        seen = {clean_target}
        seen.update(sense_antonyms)

        thes_items = list(cls.get_ldoce_thesaurus(clean_target))
        t_gram_prof = cls.get_word_gram_profile(clean_target, pos=pos)
        for peer_w in t_gram_prof.get("comparison_peers", set()):
            thes_items.append({
                "word": peer_w,
                "form": "word",
                "distinction": definition or "",
                "grammar": "",
                "register": "",
                "concept": "",
                "is_grambox_peer": True,
            })

        if thes_items:
            # Rank items by content-word definition and quote overlap + syntactic slot clash bonus
            _thes_stopwords = {
                "the", "and", "for", "that", "with", "from", "into", "about", "over", "under", "between", "through",
                "something", "someone", "somebody", "anything", "anyone", "anybody", "everything", "everyone",
                "person", "people", "thing", "things", "way", "time", "part", "place", "kind", "type", "sort", "form",
                "act", "action", "state", "quality", "fact", "process", "example", "used", "using", "use", "make",
                "making", "made", "have", "having", "had", "give", "giving", "take", "taking", "get", "getting",
                "want", "wanting", "feel", "feeling", "show", "showing", "cause", "causing", "become", "becoming",
                "able", "very", "more", "most", "much", "many", "some", "other", "another", "such", "same", "different",
                "good", "bad", "high", "low", "large", "small", "long", "short", "new", "old", "young", "particular",
                "especially", "usually", "often", "always", "never", "when", "where", "which", "what", "who", "whom",
                "whose", "while", "because", "although", "though", "whether", "without", "within", "during", "before",
                "after", "above", "below", "around", "among", "against", "towards", "toward", "upon", "onto", "they",
                "them", "their", "theirs", "themselves", "you", "your", "yours", "yourself", "our", "ours", "his",
                "her", "hers", "its", "this", "that", "these", "those", "there", "here", "will", "would", "could",
                "should", "might", "must", "can", "may", "not", "only", "also", "even", "still", "just", "well",
                "than", "then", "too", "out", "off", "down", "back", "away", "one", "ones", "two", "all", "both",
                "each", "every", "either", "neither", "none", "own", "being", "been", "were", "was", "are", "does", "did",
            }
            q_words = {w for w in re.findall(r"\b[a-zA-Z]{3,}\b", (quote or "").lower()) if w not in _thes_stopwords}
            d_words = {w for w in re.findall(r"\b[a-zA-Z]{3,}\b", (definition or "").lower()) if w not in _thes_stopwords}
            content_words = d_words | q_words

            # Map each concept cluster to the target word's own distinction in that cluster (if present)
            # so polysemous targets (e.g. 'compulsion' sense 1 'strong desire' vs sense 2 'forcing/obligation')
            # do not pull Activator/Thesaurus clusters belonging to a different sense.
            cluster_self_distinction: Dict[str, Set[str]] = {}
            target_ortho_set = set(cls.orthographic_variants(clean_target)) if hasattr(cls, "orthographic_variants") else {clean_target}
            for it in thes_items:
                it_w = str(it.get("word", "")).strip().lower()
                it_concept = str(it.get("concept", "")).strip().lower()
                it_cluster_key = it_concept or (f"sec_{it.get('section_index')}" if it.get("section_index") is not None else "")
                if it_w in target_ortho_set and it_cluster_key:
                    self_toks = {
                        w for w in re.findall(r"\b[a-zA-Z]{3,}\b", (str(it.get("distinction", "")) + " " + it_concept).lower())
                        if w not in _thes_stopwords and w not in target_ortho_set
                    }
                    cluster_self_distinction[it_cluster_key] = self_toks

            scored_thes_items = []
            for it in thes_items:
                it_concept = str(it.get("concept", "")).strip().lower()
                it_cluster_key = it_concept or (f"sec_{it.get('section_index')}" if it.get("section_index") is not None else "")
                it_dist = str(it.get("distinction", "")).strip().lower()
                it_reg = str(it.get("register") or "").lower()
                if not target_is_informal and any(
                    r in it_reg for r in ("informal", "slang", "taboo", "old-fashioned", "old use", "biblical")
                ):
                    continue
                # POS-frame guard on Language Activator clusters:
                # Reject verb concept clusters ('to put something somewhere') when target is a noun/adjective,
                # and reject noun concept clusters ('a layer of...') when target is a verb.
                if wn_pos != "v" and (it_concept.startswith("to ") or it_dist.startswith("to ")):
                    continue
                if wn_pos == "v" and it_concept and not it_concept.startswith("to ") and it_dist.startswith(("a ", "an ", "the ")):
                    continue
                if d_words and it_cluster_key in cluster_self_distinction and not it.get("is_grambox_peer"):
                    if not (cluster_self_distinction[it_cluster_key] & (d_words | q_words | ped_ox_syns)):
                        continue
                t_text = (it.get("word", "") + " " + it_dist + " " + it_concept).lower()
                t_toks = {w for w in re.findall(r"\b[a-zA-Z]{3,}\b", t_text) if w not in _thes_stopwords}
                overlap_score = len(t_toks & d_words) * 3 + len(t_toks & q_words)

                raw_cand_head = str(it.get("word", "")).strip().lower().split("/")[0].split()[0] if it.get("word") else ""
                has_clash = False
                if raw_cand_head:
                    has_clash, _ = cls.evaluate_syntactic_slot_clash(
                        clean_target, raw_cand_head,
                        anchor=context_anchor, anchor_type=anchor_type, pos=pos, quote=quote
                    )
                    if has_clash:
                        overlap_score += 15

                # If definition/quote provided, filter out Activator/Thesaurus clusters with zero content-word overlap against the target sense
                # UNLESS the candidate has a verified syntactic slot clash or is a comparison peer
                if content_words and overlap_score == 0 and not it.get("is_grambox_peer") and not has_clash:
                    continue

                # Boost true Thesaurus distinction records slightly over broad Activator clusters
                if not it.get("concept"):
                    overlap_score += 1
                if it.get("is_grambox_peer"):
                    overlap_score += 5
                scored_thes_items.append((overlap_score, it))
            scored_thes_items.sort(key=lambda x: x[0], reverse=True)

            for _, it in scored_thes_items:
                raw_w = it.get("word", "").strip().lower()
                # Skip instructional template phrases containing 'etc'
                if "etc" in raw_w:
                    continue
                # If target is a single word and the entry is a concrete multi-word phrase
                # (e.g. 'breach of protocol', 'defy convention', 'a matter of convention'),
                # skip it to prevent dismantling collocations into false synonyms or action heads.
                if " " not in clean_target and it.get("form") == "phrase":
                    if not any(ph in raw_w.split() for ph in ("sb", "sth", "someone", "something", "oneself")):
                        continue
                # If phrase contains spaces and slashes (e.g. 'the Department of Science/English/Trade'),
                # it is a template frame when the first alternative itself contains a space (unless verb phrase);
                # if the first alternative is a single word (e.g. 'compete/take part'), keep single-word alternatives.
                if "/" in raw_w and wn_pos != "v":
                    parts_check = [p.strip() for p in raw_w.split("/") if p.strip()]
                    if parts_check and " " in parts_check[0]:
                        continue
                chunks = [ch.strip() for ch in raw_w.split("/")] if "/" in raw_w else [raw_w]
                for chunk in chunks:
                    if not chunk:
                        continue
                    parts = chunk.split()
                    cand = None
                    if len(parts) == 1:
                        cand = parts[0]
                    elif len(parts) >= 2:
                        # Extract verb head from verb phrases ONLY if followed by standard placeholders (e.g. 'perform sth')
                        # or bound prepositions (e.g. 'derive from', 'stem from', 'result from')
                        # Do NOT extract verb head from phrasal verbs or idiomatic phrases (e.g. 'come forward', 'let me')
                        if wn_pos == "v":
                            _light_verbs = {"make", "get", "have", "take", "give", "do", "put", "set", "keep", "let", "go", "come", "be"}
                            if (
                                len(parts) == 2
                                and parts[1] in ("sb", "sth", "someone", "something", "oneself")
                                and parts[0].lower() not in _light_verbs
                            ):
                                cand = parts[0]
                            elif (
                                len(parts) == 2
                                and parts[1] in cls._STANDARD_PREPOSITIONS
                                and parts[0].lower() not in _light_verbs
                            ):
                                cand = parts[0]
                            else:
                                continue
                        # Extract noun head from noun phrases (e.g. 'branch of', 'facilities for', 'member of')
                        elif wn_pos == "n" and parts[1] in ("of", "for", "to", "in", "on", "with", "as", "sb", "sth"):
                            if len(parts) >= 3 and parts[-1] not in ("sb", "sth", "someone", "something", "oneself", "etc"):
                                continue
                            p_doc = nlp(parts[0])
                            if p_doc and p_doc[0].pos_ in ("NOUN", "PROPN"):
                                cand = parts[0]
                    if not cand or not cand.isalpha() or len(cand) < 3:
                        continue

                    if cls.is_exact_pedlex_headword(cand):
                        picked = cand
                    else:
                        cand_doc = nlp(cand)
                        cand_lemma = cand_doc[0].lemma_.lower()
                        picked = cand_lemma if cls.is_exact_pedlex_headword(cand_lemma) else cand
                    has_picked_clash, _ = cls.evaluate_syntactic_slot_clash(
                        clean_target, picked,
                        anchor=context_anchor, anchor_type=anchor_type, pos=pos, quote=quote
                    )
                    # Differentiated PedLex Thesaurus peers (e.g. 'loan' vs 'mortgage'/'overdraft' under anchor 'of')
                    # are curated contrastive peers: allow direct synset synonyms ONLY when a bound prepositional
                    # anchor or syntactic slot clash strictly rules them out, and never when the candidate literally
                    # defines the target word in its dictionary definition (e.g. 'interested' in 'intrigued').
                    is_anchor_discriminated_peer = bool(
                        context_anchor
                        and anchor_type == "prep"
                        and picked not in d_words
                        and cls.lemma_of(picked) not in d_words
                        and (
                            wn_pos != "n"
                            or (
                                it.get("section_index") is not None
                                and context_anchor.lower() not in ("into", "between")
                                and clean_target not in cls.get_strict_pedlex_and_oxford_synonyms(picked, pos="noun")
                                and cls.lemma_of(clean_target) not in cls.get_strict_pedlex_and_oxford_synonyms(picked, pos="noun")
                            )
                        )
                        and not cls.double_key_collision(
                            clean_target, picked,
                            anchor=context_anchor, anchor_type=anchor_type, pos=pos, quote=quote
                        )[0]
                    )

                    if (
                        picked
                        and picked != clean_target
                        and picked not in seen
                        and (picked not in global_synonyms or has_picked_clash or is_anchor_discriminated_peer)
                        and not cls.are_same_word_family(picked, clean_target)
                        and not any(cls.are_same_word_family(picked, existing) for existing in tier0_ldoce_thesaurus)
                        and not cls._is_collocational_partner(clean_target, picked)
                        and _pos_ok(picked)
                        and cls.is_cefr_compliant_distractor(picked, clean_target)
                        and not _is_informal_or_slang_distractor(picked)
                    ):
                        cand_entry = cls.get_ldoce_entry(picked)
                        if cand_entry:
                            c_primary = (cand_entry.get("pos") or "").lower()
                            c_poses = [p.lower() for p in cand_entry.get("all_poses", [c_primary])]
                            c_senses = cand_entry.get("senses") or []
                            if wn_pos == "n":
                                if not any("noun" in p for p in c_poses):
                                    continue
                                # Reject if primary PedLex POS is verb or adjective unless it has an attested noun sense
                                if c_primary in ("verb", "adjective", "adverb") and not any(s.get("pos") == "noun" for s in c_senses):
                                    continue
                            elif wn_pos == "v":
                                if not any("verb" in p for p in c_poses):
                                    continue
                                if c_primary in ("noun", "adjective", "adverb") and not any(s.get("pos") == "verb" for s in c_senses):
                                    continue
                            elif wn_pos == "a":
                                if not any("adj" in p for p in c_poses):
                                    continue
                                if not cls.distractor_occupies_slot(picked, "adj"):
                                    continue
                                # Reject words whose adjective senses are strictly [only before noun] when target is not attributive-only
                                adj_senses = [s for s in c_senses if s.get("pos") == "adjective"]
                                if adj_senses and all("[only before noun]" in str(s.get("grammar") or "") for s in adj_senses):
                                    t_entry = cls.get_ldoce_entry(clean_target)
                                    t_grams = [str(s.get("grammar") or "") for s in ((t_entry or {}).get("senses") or [])]
                                    if not (t_grams and all("[only before noun]" in g for g in t_grams)):
                                        continue
                        seen.add(picked)
                        tier0_ldoce_thesaurus.append(picked)
                        # Cluster Throttling: pick at most 1 distinct distractor per Thesaurus/Activator entry
                        break
                if len(tier0_ldoce_thesaurus) >= target_count * 2:
                    break

        tier1_synonyms: List[str] = []
        tier2_satellites: List[str] = []
        tier3_attributes: List[str] = []
        tier2_hyponyms: List[str] = []
        tier3_coordinates: List[str] = []
        coord_hyper_overlap: Dict[str, int] = {}
        coord_domain_cues: Set[str] = set()

        for s in target_synsets:
            # Tier 1: Synonyms in synset (Only for nouns/adjectives; strictly banned for verbs)
            if wn_pos != "v":
                for sw in s.words():
                    lemma = sw.lemma().lower()
                    if (
                        lemma not in seen
                        and " " not in lemma
                        and "_" not in lemma
                        and "-" not in lemma
                        and cls.is_exact_pedlex_headword(lemma)
                        and cls.is_cefr_compliant_distractor(lemma, clean_target)
                        and not _is_informal_or_slang_distractor(lemma)
                    ):
                        if wn_pos == "n" and (_is_primarily_adj_or_verb(lemma) or not cls.is_exact_pedlex_headword(lemma)):
                            continue
                        if lemma in global_synonyms:
                            continue
                        seen.add(lemma)
                        tier1_synonyms.append(lemma)

            if wn_pos == "a":
                # For Adjectives: WordNet organizes concepts into Bipolar Clusters (Satellite Synset Harvester)
                # Tier 2: Satellite synsets (fine-grained descriptive neighbors in the bipolar cluster)
                sat_synsets = []
                if s.pos == "a":
                    sat_synsets.extend(s.get_related("also") + s.get_related("similar"))
                elif s.pos == "s":
                    for head in s.get_related("similar"):
                        sat_synsets.extend(head.get_related("also") + head.get_related("similar"))

                for sim in sat_synsets:
                    for sw in sim.words():
                        lemma = sw.lemma().lower()
                        if (
                            lemma not in seen
                            and (
                                lemma not in global_synonyms
                                or cls.evaluate_syntactic_slot_clash(
                                    clean_target, lemma, anchor=context_anchor, anchor_type=anchor_type, pos=pos, quote=quote
                                )[0]
                            )
                            and " " not in lemma
                            and "_" not in lemma
                            and "-" not in lemma
                            and cls.is_exact_pedlex_headword(lemma)
                            and _pos_ok(lemma)
                            and cls.is_cefr_compliant_distractor(lemma, clean_target)
                            and not _is_informal_or_slang_distractor(lemma)
                        ):
                            seen.add(lemma)
                            tier2_satellites.append(lemma)

                # Tier 3: Attribute coordinates (adjectives sharing the same abstract attribute, e.g. speed, scope, magnitude)
                attrs = s.get_related("attribute")
                if s.pos == "s":
                    for head in s.get_related("similar"):
                        attrs.extend(head.get_related("attribute"))
                for attr in attrs:
                    for att_adj in attr.get_related("attribute"):
                        for aw in att_adj.words():
                            lemma = aw.lemma().lower()
                            if (
                                lemma not in seen
                                and lemma not in global_synonyms
                                and " " not in lemma
                                and "_" not in lemma
                                and "-" not in lemma
                                and cls.is_exact_pedlex_headword(lemma)
                                and _pos_ok(lemma)
                                and cls.is_cefr_compliant_distractor(lemma, clean_target)
                                and not _is_informal_or_slang_distractor(lemma)
                            ):
                                seen.add(lemma)
                                tier3_attributes.append(lemma)
                        for sim in att_adj.get_related("similar"):
                            for sw in sim.words():
                                lemma = sw.lemma().lower()
                                if (
                                    lemma not in seen
                                    and lemma not in global_synonyms
                                    and " " not in lemma
                                    and "_" not in lemma
                                    and "-" not in lemma
                                    and cls.is_exact_pedlex_headword(lemma)
                                    and _pos_ok(lemma)
                                    and cls.is_cefr_compliant_distractor(lemma, clean_target)
                                    and not _is_informal_or_slang_distractor(lemma)
                                ):
                                    seen.add(lemma)
                                    tier3_attributes.append(lemma)
                # Tier 3b: Relational Adjective Pertainym / Derivation Graph Traversal
                # Covers both denominal pertainyms (`adj.pert` like `doctoral`, `medical`, `musical`)
                # and deverbal/institutional relational adjectives (`regulatory`, `compensatory`, `descriptive`, `advisory`)
                # whose WordNet `derivation` links point to verbs (`pos == "v"`) or LDOCE `word_family` nouns/verbs.
                if len(tier2_satellites) + len(tier3_attributes) < target_count * 2:
                    pert_base_syns = []
                    pert_base_lemmas = []
                    for sn in s.senses():
                        for rel_sn in sn.get_related("pertainym") + sn.get_related("derivation"):
                            try:
                                rs = rel_sn.synset()
                                if rs and rs.pos in ("n", "v") and rs not in pert_base_syns:
                                    pert_base_syns.append(rs)
                                r_lem = rel_sn.word().lemma().lower()
                                r_pos = rs.pos if rs else None
                                if r_lem and r_pos in ("n", "v") and (r_lem, r_pos) not in pert_base_lemmas:
                                    pert_base_lemmas.append((r_lem, r_pos))
                            except Exception:
                                pass
                    for _b_lem, _b_pos in pert_base_lemmas:
                        if _b_pos == "v":
                            for _bs in wn.synsets(_b_lem, "v")[:3]:
                                if _bs not in pert_base_syns:
                                    pert_base_syns.append(_bs)
                    if not pert_base_syns:
                        _t_ent_wf = cls.get_ldoce_entry(clean_target) or {}
                        _wf_map = _t_ent_wf.get("word_family") or {}
                        for _wf_w in list(_wf_map.get("noun") or []) + list(_wf_map.get("verb") or []):
                            _wf_clean = str(_wf_w or "").strip().lower()
                            if _wf_clean and _wf_clean != clean_target:
                                for _rs in wn.synsets(_wf_clean, "n")[:2] + wn.synsets(_wf_clean, "v")[:2]:
                                    if _rs not in pert_base_syns:
                                        pert_base_syns.append(_rs)
                    sister_base_syns = []
                    for pns in pert_base_syns:
                        for hyp in pns.hypernyms():
                            if hyp.id in ("oewn-05844071-n", "oewn-00002137-n"):
                                continue
                            if hyp not in sister_base_syns:
                                sister_base_syns.append(hyp)
                            for coh in hyp.hyponyms()[:25]:
                                if coh not in sister_base_syns:
                                    sister_base_syns.append(coh)
                    _tier3b_cands: List[str] = []
                    for sns in sister_base_syns:
                        for n_sn in sns.senses():
                            for drv_sn in n_sn.get_related("derivation") + n_sn.get_related("pertainym"):
                                try:
                                    ds = drv_sn.synset()
                                    if not ds or ds.pos not in ("a", "s"):
                                        continue
                                    lemma = drv_sn.word().lemma().lower()
                                    if (
                                        lemma not in seen
                                        and lemma not in global_synonyms
                                        and not cls.are_same_word_family(lemma, clean_target)
                                        and " " not in lemma
                                        and "_" not in lemma
                                        and "-" not in lemma
                                        and cls.is_exact_pedlex_headword(lemma)
                                        and _pos_ok(lemma)
                                        and cls.is_cefr_compliant_distractor(lemma, clean_target)
                                        and not _is_informal_or_slang_distractor(lemma)
                                    ):
                                        _c_ent_3b = cls.get_ldoce_entry(lemma) or {}
                                        if (_c_ent_3b.get("pos") or "").lower() in ("noun", "verb", "adverb"):
                                            continue
                                        seen.add(lemma)
                                        _tier3b_cands.append(lemma)
                                except Exception:
                                    pass
                    if _tier3b_cands:
                        _rel_sfx = ("ory", "ary", "ive", "al", "ic", "ar", "ular", "ental")
                        _deverbal_sfx = ("ory", "ary", "ive")
                        _t_has_rel_sfx = clean_target.endswith(_rel_sfx)
                        _t_is_deverbal_sfx = clean_target.endswith(_deverbal_sfx)
                        _tier3b_cands.sort(
                            key=lambda w: (
                                0 if cls._common_suffix_len(clean_target, w) >= 3 else 1,
                                0 if (_t_is_deverbal_sfx and w.endswith(_deverbal_sfx)) else 1,
                                0 if (_t_has_rel_sfx and w.endswith(_rel_sfx)) else 1,
                                0
                                if (
                                    cls.is_exact_pedlex_headword(w)
                                    or int((cls.get_ldoce_entry(w) or {}).get("core_level") or 0) > 0
                                    or cls._lexicon_profile(w)[2] in ("A2", "B1", "B2")
                                )
                                else 1,
                            )
                        )
                        tier3_attributes.extend(_tier3b_cands)
            else:
                # Tier 2b/3r: Adverb Pertainym & Base-Adjective Bridge (`wn_pos == "r"`)
                # WordNet adverb synsets have 0 hypernyms/hyponyms; they connect to their base adjective
                # via `pertainym`/`derivation` (or regular `-ly` morphology). Traverse the base adjective's
                # antonyms and attribute/similar coordinates and project them back into attested `-ly` adverbs.
                if wn_pos == "r":
                    def _adj_to_adv_forms(adj_lem: str) -> List[str]:
                        a_l = adj_lem.strip().lower()
                        if not a_l or not a_l.isalpha():
                            return []
                        cands = [a_l + "ly"]
                        if a_l.endswith("y") and len(a_l) > 2:
                            cands.append(a_l[:-1] + "ily")
                        if a_l.endswith("le") and len(a_l) > 3:
                            cands.append(a_l[:-2] + "ly")
                        if a_l.endswith("ic"):
                            cands.append(a_l + "ally")
                        if a_l.endswith("ll"):
                            cands.append(a_l + "y")
                        return list(dict.fromkeys(cands))

                    base_adj_syns = []
                    for sn in s.senses():
                        for rel_sn in sn.get_related("pertainym") + sn.get_related("derivation"):
                            try:
                                rs = rel_sn.synset()
                                if rs and rs.pos in ("a", "s") and rs not in base_adj_syns:
                                    base_adj_syns.append(rs)
                            except Exception:
                                pass
                    if not base_adj_syns and clean_target.endswith("ly") and len(clean_target) > 4:
                        stem_cands = [clean_target[:-2], clean_target[:-2] + "e"]
                        if clean_target.endswith("ily"):
                            stem_cands.append(clean_target[:-3] + "y")
                        if clean_target.endswith("ally"):
                            stem_cands.append(clean_target[:-4])
                        for st_c in stem_cands:
                            for as_c in wn.synsets(st_c):
                                if as_c.pos in ("a", "s") and as_c not in base_adj_syns:
                                    base_adj_syns.append(as_c)
                            if base_adj_syns:
                                break
                    for b_adj in base_adj_syns[:3]:
                        # 1. Base-adjective antonyms projected to `-ly` adverbs -> `tier1_antonyms`
                        adj_ant_syns = list(b_adj.get_related("antonym"))
                        for b_sn in b_adj.senses():
                            for a_sn in b_sn.get_related("antonym"):
                                try:
                                    if a_sn.synset() and a_sn.synset() not in adj_ant_syns:
                                        adj_ant_syns.append(a_sn.synset())
                                except Exception:
                                    pass
                        if b_adj.pos == "s":
                            for h_adj in b_adj.get_related("similar"):
                                for h_sn in h_adj.senses():
                                    for a_sn in h_sn.get_related("antonym"):
                                        try:
                                            if a_sn.synset() and a_sn.synset() not in adj_ant_syns:
                                                adj_ant_syns.append(a_sn.synset())
                                        except Exception:
                                            pass
                        for ant_s in adj_ant_syns:
                            for aw in ant_s.words():
                                for adv_f in _adj_to_adv_forms(aw.lemma()):
                                    if (
                                        adv_f not in seen
                                        and adv_f not in global_synonyms
                                        and not cls.are_same_word_family(adv_f, clean_target)
                                        and (cls.is_exact_pedlex_headword(adv_f) or bool(wn.synsets(adv_f, "r")))
                                        and _pos_ok(adv_f)
                                        and cls.is_cefr_compliant_distractor(adv_f, clean_target)
                                        and not _is_informal_or_slang_distractor(adv_f)
                                    ):
                                        seen.add(adv_f)
                                        tier3_coordinates.append(adv_f)
                            for ant_sim in ant_s.get_related("similar")[:6]:
                                for asw in ant_sim.words():
                                    for adv_f in _adj_to_adv_forms(asw.lemma()):
                                        if (
                                            adv_f not in seen
                                            and adv_f not in global_synonyms
                                            and not cls.are_same_word_family(adv_f, clean_target)
                                            and (cls.is_exact_pedlex_headword(adv_f) or bool(wn.synsets(adv_f, "r")))
                                            and _pos_ok(adv_f)
                                            and cls.is_cefr_compliant_distractor(adv_f, clean_target)
                                            and not _is_informal_or_slang_distractor(adv_f)
                                        ):
                                            seen.add(adv_f)
                                            tier3_coordinates.append(adv_f)
                        # 2. Attribute coordinates projected to `-ly` adverbs -> `tier3_coordinates`
                        adj_attrs = list(b_adj.get_related("attribute"))
                        if b_adj.pos == "s":
                            for h_adj in b_adj.get_related("similar"):
                                adj_attrs.extend(h_adj.get_related("attribute"))
                        for attr_n in adj_attrs:
                            for att_adj in attr_n.get_related("attribute"):
                                if att_adj.id == b_adj.id:
                                    continue
                                for aw in att_adj.words():
                                    for adv_f in _adj_to_adv_forms(aw.lemma()):
                                        if (
                                            adv_f not in seen
                                            and adv_f not in global_synonyms
                                            and not cls.are_same_word_family(adv_f, clean_target)
                                            and (cls.is_exact_pedlex_headword(adv_f) or bool(wn.synsets(adv_f, "r")))
                                            and _pos_ok(adv_f)
                                            and cls.is_cefr_compliant_distractor(adv_f, clean_target)
                                            and not _is_informal_or_slang_distractor(adv_f)
                                        ):
                                            seen.add(adv_f)
                                            tier3_coordinates.append(adv_f)

                # Tier 2: Hyponyms (more specific concepts / troponyms)
                # Only harvest hyponyms from the locked target_synsets (already represented by `s in target_synsets`),
                # and for nouns skip hyponyms to prevent synonym-pile double-keys with semantic_fields.
                if wn_pos == "v":
                    for tok in re.findall(r"\b[a-z]{4,}\b", (s.definition() or "").lower()):
                        if tok not in cls._PARONYM_STOP_DEF_TOKS:
                            coord_domain_cues.add(tok)
                    for hypo in s.hyponyms():
                        for hw in hypo.words():
                            lemma = hw.lemma().lower()
                            if (
                                lemma not in seen
                                and lemma not in global_synonyms
                                and " " not in lemma
                                and "_" not in lemma
                                and "-" not in lemma
                                and cls.is_exact_pedlex_headword(lemma)
                                and not _is_primarily_noun_or_adj_for_verb(lemma)
                                and _pos_ok(lemma)
                                and cls.is_cefr_compliant_distractor(lemma, clean_target)
                                and not _is_informal_or_slang_distractor(lemma)
                            ):
                                seen.add(lemma)
                                tier2_hyponyms.append(lemma)
                    # SnappyWords/WordVis Verb Script Relations: entails & causes (same-event-script actions, never synonyms)
                    for rel_syn in s.get_related("entails") + s.get_related("causes"):
                        for rw in rel_syn.words():
                            lemma = rw.lemma().lower()
                            if (
                                lemma not in seen
                                and lemma not in global_synonyms
                                and " " not in lemma
                                and "_" not in lemma
                                and "-" not in lemma
                                and cls.is_exact_pedlex_headword(lemma)
                                and not _is_primarily_noun_or_adj_for_verb(lemma)
                                and _pos_ok(lemma)
                                and cls.is_cefr_compliant_distractor(lemma, clean_target)
                                and not _is_informal_or_slang_distractor(lemma)
                                and lemma not in ("say", "tell", "speak", "talk", "whisper", "shout", "reply", "state", "declare", "be", "have", "do", "make", "get")
                            ):
                                seen.add(lemma)
                                coord_hyper_overlap[lemma] = max(coord_hyper_overlap.get(lemma, 0), 2)
                                tier3_coordinates.append(lemma)
                # Helper to verify that a polysemous noun coordinate's top PedLex senses align with the WordNet sister branch & countability
                def _is_aligned_noun_coord(cand_lem: str, sis_syn: Optional[Any] = None, hyper_syn: Optional[Any] = None) -> bool:
                    if cls._is_collocational_partner(clean_target, cand_lem):
                        return False
                    c_ent_n = cls.get_ldoce_entry(cand_lem) or {}
                    c_senses_n = [
                        sn for sn in (c_ent_n.get("senses") or [])
                        if not sn.get("pos") or cls._pos_code(sn.get("pos")) == "n"
                    ] or (c_ent_n.get("senses") or [])
                    if not c_senses_n:
                        return True
                    # Countability & Number Symmetry Gate for noun coordinates:
                    # 1. Reject countable artifact homonyms (e.g. 'diaper', 'sheet', 'till', 'sliver', 'nap', 'duck')
                    #    whose top 2 PedLex noun senses do not attest [uncountable] when target is strictly uncountable.
                    # 2. Symmetric: Reject strictly [uncountable] mass nouns (e.g. 'adrenalin', 'afterbirth')
                    #    when target is strictly [countable].
                    # 3. Reject pluralia tantum [plural] nouns (e.g. 'banns') when clean_target is singular.
                    if target_strictly_uncountable:
                        top2_grams = " ".join(str(sn.get("grammar") or "").lower() for sn in c_senses_n[:2])
                        if "uncountable" not in top2_grams:
                            return False
                    elif target_strictly_countable:
                        top2_grams = " ".join(str(sn.get("grammar") or "").lower() for sn in c_senses_n[:2])
                        if "uncountable" in top2_grams and "countable" not in re.sub(r"\buncountable\b", "", top2_grams):
                            return False
                    if not clean_target.endswith("s") and cand_lem.endswith("s"):
                        g0_low = str(c_senses_n[0].get("grammar") or "").lower()
                        if "plural" in g0_low and "countable" not in re.sub(r"\buncountable\b", "", g0_low):
                            return False
                    c_wn_n = [cs for cs in wn.synsets(cand_lem, "n") if cs.lexfile()] or wn.synsets(cand_lem, "n")
                    cand_target_syn = sis_syn if (sis_syn and sis_syn.lexfile()) else (c_wn_n[0] if c_wn_n else None)
                    if c_wn_n:
                        cand_lf = cand_target_syn.lexfile() if cand_target_syn else c_wn_n[0].lexfile()
                        if s.lexfile() == "noun.person" and cand_lf != "noun.person":
                            return False
                        if s.lexfile() and s.lexfile() != "noun.person" and cand_lf == "noun.person":
                            return False
                        # Reject collective group nouns (e.g. 'staff') when target is not a group noun
                        if s.lexfile() and s.lexfile() != "noun.group" and (cand_lf == "noun.group" or (c_wn_n and c_wn_n[0].lexfile() == "noun.group")):
                            return False
                        # Auditory / Speech alignment gate: When target synset is sound/communication and context has auditory cues,
                        # coordinate must also have primary lexfile in sound/communication/event, blocking abstract attributes like 'capacity', 'diameter', 'lifestyle'.
                        if s.lexfile() in ("noun.communication", "noun.sound"):
                            if wsd_ctx.get("has_auditory_context") or (context_anchor and context_anchor.lower() in getattr(cls, "_AUDITORY_PREDICATES", set())):
                                if cand_lf not in ("noun.communication", "noun.sound", "noun.event"):
                                    return False
                                if cand_lf == "noun.event":
                                    cand_noun_defs = " ".join(str(sn.get("definition") or "").lower() for sn in c_senses_n)
                                    if not any(snd_w in cand_noun_defs for snd_w in ("sound", "noise", "audible", "loud", "faint", "listen", "ear")):
                                        return False
                        # Physical Token / Document / Label Affordance Gate:
                        # When target is a small paper token, label, sticker, ticket, or postal stamp,
                        # reject large shipping containers ('parcel', 'box', 'crate'), postal routing codes ('postcode', 'zipcode'),
                        # monetary fee amounts ('cost', 'price', 'fee', 'fare', 'charge', 'rate', 'dollar', 'cent'),
                        # and mathematical diagrams/charts ('graph', 'chart', 'diagram').
                        is_token_target = (
                            bool(locals().get("_material_genus_words", set()) & {"paper", "card", "cardboard"})
                            or (definition and any(tk in definition.lower() for tk in ("piece of paper", "small piece", "postage stamp", "label", "sticker", "ticket", "token")))
                        )
                        if is_token_target:
                            cand_noun_defs = " ".join(str(sn.get("definition") or "").lower() for sn in c_senses_n)
                            if any(c_box in cand_noun_defs for c_box in ("package", "parcel", "box", "container", "wrapped")):
                                return False
                            if any(c_code in cand_noun_defs or c_code in cand_lem for c_code in ("postcode", "zipcode", "zip code", "code")):
                                return False
                            if any(c_fee in cand_noun_defs for c_fee in ("unit of money", "currency", "amount of money", "price", "fee", "cost", "sum of money", "charge")):
                                return False
                            if any(c_diag in cand_noun_defs for c_diag in ("diagram", "drawing that uses", "sets of measurements", "mathematical", "quantity")):
                                return False
                            cand_all_lfs = {cs.lexfile() for cs in c_wn_n if cs.lexfile()}
                            if cand_all_lfs and not (cand_all_lfs & {"noun.artifact", "noun.object", "noun.communication"}):
                                return False
                        # Reject biological/anatomical coordinates (e.g. 'palate' under 'surface') when target is non-biological
                        _bio_lfs = ("noun.body", "noun.animal", "noun.plant")
                        if (
                            s.lexfile()
                            and s.lexfile() not in _bio_lfs
                            and not locals().get("is_bio_sample_def")
                            and (
                                (sis_syn is not None and sis_syn.lexfile() in _bio_lfs)
                                or (sis_syn is None and c_wn_n and c_wn_n[0].lexfile() in _bio_lfs)
                            )
                        ):
                            return False
                        # Reject metonymic building/container/artifact coordinates (e.g. 'room', 'suite', 'house', 'hall', 'floor')
                        # whose primary WordNet sense is concrete physical/location when target is a `noun.group` collective noun.
                        if s.lexfile() == "noun.group" and cand_lf in (
                            "noun.artifact", "noun.location", "noun.body",
                            "noun.substance", "noun.object", "noun.food", "noun.plant",
                        ):
                            return False
                    # Monosemous in PedLex and sharing primary WordNet lexfile -> aligned
                    # (Exclude broad `noun.person` and abstract ontology lexfiles `noun.state`, `noun.attribute`, `noun.cognition`, `noun.act`
                    # so abstract nouns must share domain cues/tokens with the target).
                    if (
                        len(c_senses_n) == 1
                        and s.lexfile()
                        and s.lexfile() not in ("noun.person", "noun.state", "noun.attribute", "noun.cognition", "noun.act")
                        and cand_target_syn
                        and cand_target_syn.lexfile() == s.lexfile()
                    ):
                        return True
                    if sis_syn is None or hyper_syn is None:
                        return True
                    _meta_stop = {
                        "someone", "somebody", "anyone", "anybody", "something", "anything",
                        "person", "people", "thing", "things", "place", "time", "part", "type",
                        "kind", "form", "amount", "number", "small", "large", "used", "make",
                        "made", "making", "taken", "order", "particular", "usually", "often",
                        "point", "period", "state", "situation", "condition", "fact", "case", "way", "level",
                    }
                    # Check if top-2 PedLex senses share domain tokens (with plural -s normalization) with sister synset / hypernym / target definition
                    prim_txt = re.sub(
                        r"\bthe\s+earth\b",
                        "",
                        " ".join(
                            " ".join([
                                str(sn.get("definition") or ""),
                                str(sn.get("signpost") or ""),
                                " ".join(sn.get("units") or []),
                            ])
                            for sn in c_senses_n[:2]
                        ).lower(),
                    )
                    raw_prim_toks = {
                        tok for tok in re.findall(r"\b[a-z]{3,}\b", prim_txt)
                        if tok not in cls._PARONYM_STOP_DEF_TOKS and tok not in cls._ANCHOR_STOPWORDS and tok not in _meta_stop and tok != cand_lem
                    }
                    def _stem_root(tok: str) -> str:
                        t = tok.lower()
                        for sfx in ("fulness", "lessness", "ability", "iveness", "ment", "ness", "ful", "less", "able", "ible", "ive", "ion", "ity", "ic", "al", "ly", "ing", "ed", "er", "or", "s"):
                            if t.endswith(sfx) and len(t) - len(sfx) >= 3:
                                return t[:-len(sfx)]
                        return t

                    prim_toks = set(raw_prim_toks) | {
                        tok[:-1] for tok in raw_prim_toks if tok.endswith("s") and len(tok) > 3
                    } | {_stem_root(tok) for tok in raw_prim_toks}
                    branch_txt = " ".join([
                        str(sis_syn.definition() or ""),
                        " ".join(sis_syn.examples() or []),
                        str(hyper_syn.definition() or ""),
                        " ".join(hw.lemma() for hw in hyper_syn.words()),
                        " ".join(ghw.lemma() for gh in hyper_syn.hypernyms() for ghw in gh.words()),
                    ]).lower()
                    raw_branch_toks = {
                        tok for tok in re.findall(r"\b[a-z]{3,}\b", branch_txt)
                        if tok not in cls._PARONYM_STOP_DEF_TOKS and tok not in cls._ANCHOR_STOPWORDS and tok not in _meta_stop and tok != cand_lem
                    } | ((coord_domain_cues - _meta_stop) - {cand_lem})
                    if definition:
                        raw_branch_toks |= ((def_tokens - _meta_stop) - {cand_lem})
                    branch_toks = set(raw_branch_toks) | {
                        tok[:-1] for tok in raw_branch_toks if tok.endswith("s") and len(tok) > 3
                    } | {_stem_root(tok) for tok in raw_branch_toks}
                    return bool(prim_toks & branch_toks)

                # Tier 3: Coordinate terms (sisters under same hypernym - Troponyms for verbs)
                for hyper in s.hypernyms():
                    if hasattr(hyper, "id") and hyper.id in cls._ABSTRACT_ONTOLOGY_HYPERNYMS:
                        continue
                    h_depth = hyper.min_depth() if hasattr(hyper, "min_depth") else 0
                    if wn_pos == "n":
                        if h_depth < 4:
                            continue
                        if h_depth == 4 and len(hyper.hyponyms()) > 50 and getattr(hyper, "lexfile", lambda: "")() in ("noun.state", "noun.Tops", "noun.act", "noun.attribute"):
                            continue
                    # Allow up to 200 hyponyms for specific mid/deep-level noun hypernyms (min_depth >= 4, e.g. container=166, fabric=176, metallic element=78)
                    if wn_pos == "n" and h_depth >= 4:
                        max_hypos = 200
                    else:
                        max_hypos = 85 if (len(s.hypernyms()) == 1 and not tier3_coordinates) else 50
                    if len(hyper.hyponyms()) > max_hypos:
                        continue
                    for hw in hyper.words():
                        for tok in re.findall(r"\b[a-z]{3,}\b", hw.lemma().lower()):
                            if tok not in cls._PARONYM_STOP_DEF_TOKS:
                                coord_domain_cues.add(tok)
                    for tok in re.findall(r"\b[a-z]{4,}\b", (hyper.definition() or "").lower()):
                        if tok not in cls._PARONYM_STOP_DEF_TOKS:
                            coord_domain_cues.add(tok)
                    for sis_c in hyper.hyponyms():
                        for sw_c in sis_c.words():
                            lem_c = sw_c.lemma().lower()
                            if lem_c.isalpha() and len(lem_c) >= 4:
                                coord_domain_cues.add(lem_c)
                    h_ov = 0
                    if definition:
                        h_txt = (hyper.definition() + " " + " ".join(hw.lemma() for hw in hyper.words())).lower()
                        h_ov = sum(1 for dt in def_tokens if dt in h_txt) * 2 + 1
                    sis_pool = list(hyper.hyponyms())
                    for sis in list(sis_pool):
                        # Unwrap multi-word or category intermediate nodes (e.g. 'auditory communication' -> 'utterance'/'speech', 'natural fiber' -> 'silk')
                        if (
                            wn_pos == "n"
                            and sis.id != s.id
                            and 0 < len(sis.hyponyms()) <= 20
                            and (
                                not any(" " not in sw.lemma() and "_" not in sw.lemma() and "-" not in sw.lemma() for sw in sis.words())
                                or any(sw.lemma() in ("marker", "label", "fastener", "covering", "document", "sheet", "ticket", "token", "speech", "utterance", "vocalization", "sound", "noise") for sw in sis.words())
                            )
                        ):
                            for sub in sis.hyponyms():
                                sis_pool.append(sub)
                                if (
                                    0 < len(sub.hyponyms()) <= 15
                                    and (
                                        not any(" " not in sw.lemma() and "_" not in sw.lemma() and "-" not in sw.lemma() for sw in sub.words())
                                        or any(sw.lemma() in ("label", "tag", "ticket", "speech", "utterance", "sound", "noise", "vocalization") for sw in sub.words())
                                    )
                                ):
                                    sis_pool.extend(sub.hyponyms())
                    for sis in sis_pool:
                        if sis.id == s.id:
                            continue
                        for sw in sis.words():
                            raw_lem = sw.lemma()
                            if wn_pos == "n" and (not raw_lem.islower() or len(raw_lem) < 3):
                                continue
                            lemma = raw_lem.lower()
                            if (
                                lemma not in seen
                                and lemma not in global_synonyms
                                and " " not in lemma
                                and "_" not in lemma
                                and "-" not in lemma
                                and cls.is_exact_pedlex_headword(lemma)
                                and _pos_ok(lemma)
                                and cls.is_cefr_compliant_distractor(lemma, clean_target)
                                and not _is_informal_or_slang_distractor(lemma)
                            ):
                                if wn_pos == "n" and (_is_primarily_adj_or_verb(lemma) or not cls.is_exact_pedlex_headword(lemma)):
                                    continue
                                if wn_pos == "n" and s.lexfile() == "noun.person":
                                    c_syns_p = wn.synsets(lemma, "n")
                                    if not c_syns_p or c_syns_p[0].lexfile() != "noun.person":
                                        continue
                                if wn_pos == "n" and not _is_aligned_noun_coord(lemma, sis, hyper):
                                    continue
                                if wn_pos == "v" and (_is_primarily_noun_or_adj_for_verb(lemma) or not cls.is_exact_pedlex_headword(lemma)):
                                    continue
                                # Verbs: Block speech/reporting verbs with clausal or double-object valency that break transitive substitution
                                if wn_pos == "v" and lemma in ("say", "tell", "speak", "talk", "whisper", "shout", "reply", "state", "declare"):
                                    continue
                                seen.add(lemma)
                                coord_hyper_overlap[lemma] = max(coord_hyper_overlap.get(lemma, 0), h_ov)
                                tier3_coordinates.append(lemma)
                # Tier 3.2: Co-Meronyms (fellow parts/members under the same holonym, e.g. petal -> sepal/stamen/pistil, roof -> wall/floor)
                if wn_pos == "n" and len(tier0_ldoce_thesaurus) + len(tier3_coordinates) < target_count * 2:
                    holo_pool = list(s.holonyms()) if hasattr(s, "holonyms") else (s.get_related("holonym_part") + s.get_related("holonym_member"))
                    for holo in holo_pool:
                        mero_list = list(holo.meronyms()) if hasattr(holo, "meronyms") else (holo.get_related("meronym_part") + holo.get_related("meronym_member"))
                        if not (1 < len(mero_list) <= 35):
                            continue
                        for hw in holo.words():
                            for tok in re.findall(r"\b[a-z]{3,}\b", hw.lemma().lower()):
                                if tok not in cls._PARONYM_STOP_DEF_TOKS:
                                    coord_domain_cues.add(tok)
                        for tok in re.findall(r"\b[a-z]{4,}\b", (holo.definition() or "").lower()):
                            if tok not in cls._PARONYM_STOP_DEF_TOKS:
                                coord_domain_cues.add(tok)
                        for co_mero in mero_list:
                            if co_mero.id == s.id:
                                continue
                            for mw in co_mero.words():
                                lemma = mw.lemma().lower()
                                if (
                                    lemma not in seen
                                    and lemma not in global_synonyms
                                    and " " not in lemma
                                    and "_" not in lemma
                                    and "-" not in lemma
                                    and cls.is_exact_pedlex_headword(lemma)
                                    and not _is_primarily_adj_or_verb(lemma)
                                    and _pos_ok(lemma)
                                    and cls.is_cefr_compliant_distractor(lemma, clean_target)
                                    and not _is_informal_or_slang_distractor(lemma)
                                ):
                                    seen.add(lemma)
                                    coord_hyper_overlap[lemma] = max(coord_hyper_overlap.get(lemma, 0), 1)
                                    tier3_coordinates.append(lemma)
                # Tier 3.5: Grand-hypernym coordinate terms (Grand-sisters under same grandparent)
                # First Defense: Extends candidate pool for nouns when immediate coordinates are sparse,
                # strictly constrained to the same WordNet lexfile and non-mega-bucket grandparents.
                if wn_pos == "n" and len(tier0_ldoce_thesaurus) + len(tier3_coordinates) < target_count * 2:
                    target_lf = s.lexfile()
                    for hyper in s.hypernyms():
                        # If direct coordinates are sparse (< target_count) and the direct hypernym is a specific
                        # concrete domain node (depth >= 5, e.g. ['soil', 'dirt'] for 'clay'), include non-synonym hypernym lemmas
                        if (
                            len(tier0_ldoce_thesaurus) + len(tier3_coordinates) < target_count
                            and hasattr(hyper, "min_depth")
                            and hyper.min_depth() >= 5
                        ):
                            for hw in hyper.words():
                                h_lem = hw.lemma().lower()
                                if (
                                    h_lem.isalpha()
                                    and len(h_lem) >= 3
                                    and h_lem not in seen
                                    and h_lem not in global_synonyms
                                    and cls.is_exact_pedlex_headword(h_lem)
                                    and not _is_primarily_adj_or_verb(h_lem)
                                    and _pos_ok(h_lem)
                                    and cls.is_cefr_compliant_distractor(h_lem, clean_target)
                                    and not _is_informal_or_slang_distractor(h_lem)
                                    and _is_aligned_noun_coord(h_lem, hyper, hyper)
                                ):
                                    seen.add(h_lem)
                                    coord_hyper_overlap[h_lem] = max(coord_hyper_overlap.get(h_lem, 0), 1)
                                    tier3_coordinates.append(h_lem)
                        for g_hyper in hyper.hypernyms():
                            if hasattr(g_hyper, "id") and g_hyper.id in cls._ABSTRACT_ONTOLOGY_HYPERNYMS:
                                continue
                            # Block upper-ontology root grand-hypernyms (depth <= 3 for substances/communications/artifacts so 'fuel' or 'symbol' at depth 4 is kept; depth <= 4 otherwise)
                            min_g_depth = 3 if target_lf in ("noun.substance", "noun.communication", "noun.artifact") else 4
                            if hasattr(g_hyper, "min_depth") and g_hyper.min_depth() <= min_g_depth:
                                continue
                            max_g_hypos = 65 if not tier3_coordinates else 35
                            if len(g_hyper.hyponyms()) > max_g_hypos:
                                continue
                            if (
                                len(tier0_ldoce_thesaurus) + len(tier3_coordinates) < target_count
                                and hasattr(g_hyper, "min_depth")
                                and g_hyper.min_depth() >= 5
                                and (not target_lf or g_hyper.lexfile() == target_lf)
                            ):
                                for ghw in g_hyper.words():
                                    gh_lem = ghw.lemma().lower()
                                    if (
                                        gh_lem.isalpha()
                                        and len(gh_lem) >= 3
                                        and gh_lem not in seen
                                        and gh_lem not in global_synonyms
                                        and cls.is_exact_pedlex_headword(gh_lem)
                                        and not _is_primarily_adj_or_verb(gh_lem)
                                        and _pos_ok(gh_lem)
                                        and cls.is_cefr_compliant_distractor(gh_lem, clean_target)
                                        and not _is_informal_or_slang_distractor(gh_lem)
                                        and _is_aligned_noun_coord(gh_lem, g_hyper, g_hyper)
                                    ):
                                        seen.add(gh_lem)
                                        coord_hyper_overlap[gh_lem] = max(coord_hyper_overlap.get(gh_lem, 0), 1)
                                        tier3_coordinates.append(gh_lem)
                            g_sis_pool = list(g_hyper.hyponyms())
                            unwrapped_g_sis = []
                            for g_sis in g_sis_pool:
                                if g_sis.id == hyper.id:
                                    continue
                                unwrapped_g_sis.append(g_sis)
                                # Unwrap intermediate taxonomic grouping nodes (e.g. 'marker' -> 'label' -> 'sticker'/'ticket', 'natural fiber' -> 'silk')
                                if (
                                    0 < len(g_sis.hyponyms()) <= 20
                                    and (
                                        not any(" " not in sw.lemma() and "_" not in sw.lemma() and "-" not in sw.lemma() for sw in g_sis.words())
                                        or any(sw.lemma() in ("marker", "label", "fastener", "covering", "document", "sheet", "ticket", "token", "speech", "utterance", "vocalization", "sound", "noise") for sw in g_sis.words())
                                    )
                                ):
                                    for sub in g_sis.hyponyms():
                                        unwrapped_g_sis.append(sub)
                                        if (
                                            0 < len(sub.hyponyms()) <= 15
                                            and (
                                                not any(" " not in sw.lemma() and "_" not in sw.lemma() and "-" not in sw.lemma() for sw in sub.words())
                                                or any(sw.lemma() in ("label", "tag", "ticket", "speech", "utterance", "sound", "noise", "vocalization") for sw in sub.words())
                                            )
                                        ):
                                            unwrapped_g_sis.extend(sub.hyponyms())
                            for sis in unwrapped_g_sis:
                                if target_lf and sis.lexfile() != target_lf:
                                    continue
                                for sw in sis.words():
                                    raw_lem = sw.lemma()
                                    if not raw_lem.islower() or len(raw_lem) < 3:
                                        continue
                                    lemma = raw_lem.lower()
                                    if (
                                        lemma not in seen
                                        and lemma not in global_synonyms
                                        and " " not in lemma
                                        and "_" not in lemma
                                        and "-" not in lemma
                                        and cls.is_exact_pedlex_headword(lemma)
                                        and _pos_ok(lemma)
                                        and cls.is_cefr_compliant_distractor(lemma, clean_target)
                                        and not _is_informal_or_slang_distractor(lemma)
                                    ):
                                        if _is_primarily_adj_or_verb(lemma) or not cls.is_exact_pedlex_headword(lemma):
                                            continue
                                        if s.lexfile() == "noun.person":
                                            c_syns_p = wn.synsets(lemma, "n")
                                            if not c_syns_p or c_syns_p[0].lexfile() != "noun.person":
                                                continue
                                        if not _is_aligned_noun_coord(lemma, sis, g_hyper):
                                            continue
                                        seen.add(lemma)
                                        coord_hyper_overlap.setdefault(lemma, 0)
                                        tier3_coordinates.append(lemma)
                # Tier 3.2b: Aristotelian Genus-Noun Taxonomic Bridge
                # When a noun's WordNet hypernym branch is sparse (< target_count) and its dictionary definition
                # follows the Aristotelian "a/an [modifier]* <genus_noun> in/of/about/..." structure (e.g., obituary ->
                # "an article in a newspaper..."), harvest direct single-word co-hyponyms of the genus noun in the same lexfile.
                if wn_pos == "n" and definition and len(tier0_ldoce_thesaurus) + len(tier3_coordinates) < target_count:
                    _m_genus = re.match(
                        r"^(?:(?:\[[^\]]+\]\s*)+)?(?:a|an|the)\s+"
                        r"(?:(?:small|large|short|long|brief|formal|official|public|special|particular|printed|written|spoken|musical|medical|scientific|legal|political|religious|traditional|modern|simple|basic)\s+)*"
                        r"([a-z]{4,})\s+(?:in|of|about|for|that|which|who|where|on|from|to|with|by|used)\b",
                        definition.strip().lower(),
                    )
                    if _m_genus:
                        _g_noun = _m_genus.group(1)
                        if _g_noun not in cls._PARONYM_STOP_DEF_TOKS and _g_noun not in cls._ANCHOR_STOPWORDS and _g_noun != clean_target:
                            _g_cands = []
                            for _g_idx, _g_syn in enumerate(wn.synsets(_g_noun, "n")):
                                if not _g_syn.lexfile():
                                    continue
                                if target_lfs_set and _g_syn.lexfile() not in target_lfs_set:
                                    continue
                                if hasattr(_g_syn, "min_depth") and _g_syn.min_depth() < 4:
                                    continue
                                _g_hypos = _g_syn.hyponyms()
                                if not (0 < len(_g_hypos) <= 50):
                                    continue
                                _g_def_toks = set(re.findall(r"\b[a-z]{4,}\b", (_g_syn.definition() or "").lower()))
                                _g_ov = len(_g_def_toks & def_tokens)
                                _g_cands.append((-_g_ov, _g_idx, _g_syn))
                            _g_cands.sort(key=lambda x: (x[0], x[1]))
                            for _, _, _g_syn in _g_cands[:1]:
                                for _gsis in _g_syn.hyponyms():
                                    if _gsis.id == s.id or (target_lfs_set and _gsis.lexfile() not in target_lfs_set):
                                        continue
                                    for _gsw in _gsis.words():
                                        _raw_glem = _gsw.lemma()
                                        if not _raw_glem.islower() or len(_raw_glem) < 3:
                                            continue
                                        _glem = _raw_glem.lower()
                                        if (
                                            _glem not in seen
                                            and _glem not in global_synonyms
                                            and " " not in _glem
                                            and "_" not in _glem
                                            and "-" not in _glem
                                            and cls.is_exact_pedlex_headword(_glem)
                                            and not _is_primarily_adj_or_verb(_glem)
                                            and _pos_ok(_glem)
                                            and cls.is_cefr_compliant_distractor(_glem, clean_target)
                                            and not _is_informal_or_slang_distractor(_glem)
                                            and _is_aligned_noun_coord(_glem, _gsis, _g_syn)
                                        ):
                                            seen.add(_glem)
                                            coord_hyper_overlap[_glem] = max(coord_hyper_overlap.get(_glem, 0), 1)
                                            tier3_coordinates.append(_glem)
                # Tier 3.3: PedLex `ldoce_sense_cue_index` Scenario/Material Co-Hyponym Tier
                # When a concrete/scenario noun sits under a singleton or sparse WordNet branch (e.g. 'stamp' under 'postage stamp',
                # 'ticket' under 'commercial document'), query `ldoce_sense_cue_index` for primary-sense nouns sharing >= 2 domain cues.
                _is_concrete_t = wn_pos == "n" and s.lexfile() in (
                    "noun.artifact", "noun.substance", "noun.communication",
                    "noun.possession", "noun.plant", "noun.food", "noun.object",
                )
                if _is_concrete_t and len(tier0_ldoce_thesaurus) + len(tier3_coordinates) < target_count * 2:
                    conn_cue = cls._ldoce_conn if cls._ensure_ldoce_conn() else None
                    if conn_cue is not None:
                        _cue_stop = {
                            "piece", "small", "large", "something", "someone", "used", "make", "made", "making",
                            "type", "kind", "form", "part", "place", "time", "person", "people", "thing",
                            "things", "usually", "often", "especially", "particular", "official", "amount",
                            "number", "group", "system", "area", "side", "front", "back", "surface", "edge",
                            "keep", "put", "take", "give", "show", "shows", "have", "having", "with", "from",
                            "into", "onto", "over", "under", "between", "through", "before", "after", "when",
                            "where", "which", "that", "their", "your", "other", "another", "many", "more",
                            "most", "some", "such", "very", "quite", "rather", "able", "good", "well", "high",
                            "long", "short", "thin", "thick", "flat", "round", "hard", "soft", "dark", "light",
                            "white", "black", "brown", "yellow", "green", "blue", "colour", "color", "shaped",
                            "you", "and", "the", "for", "out", "can", "has", "been", "are", "was", "were",
                            "his", "her", "its", "our", "who", "what", "how", "all", "any", "one", "two",
                            "not", "but", "second", "first", "last", "week", "year", "day", "open", "like", "got",
                            "about", "just", "recently", "already", "still", "even", "also", "only", "always", "never",
                            "taken", "tested", "examined", "order",
                        }
                        _t_cues: List[str] = []
                        if definition:
                            for _dt in def_tokens:
                                if _dt not in _cue_stop and len(_dt) >= 3 and _dt != clean_target:
                                    _t_cues.append(_dt)
                            if pedlex_signpost and pedlex_signpost not in _cue_stop:
                                _t_cues.append(pedlex_signpost)
                            for _pu in pedlex_units:
                                for _ptok in re.findall(r"\b[a-z]{3,}\b", _pu):
                                    if _ptok not in _cue_stop and _ptok != clean_target:
                                        _t_cues.append(_ptok)
                        for _wt in re.findall(r"\b[a-z]{4,}\b", (s.definition() or "").lower()):
                            if _wt not in _cue_stop and _wt != clean_target:
                                _t_cues.append(_wt)
                                coord_domain_cues.add(_wt)
                        try:
                            for ( _cw, ) in conn_cue.execute(
                                "SELECT cue_word FROM ldoce_sense_cue_index WHERE host_word = ? AND sense_idx = 0",
                                (clean_target,),
                            ).fetchall():
                                if _cw not in _cue_stop and len(_cw) >= 3 and _cw != clean_target:
                                    _t_cues.append(_cw)
                                    coord_domain_cues.add(_cw)
                        except Exception:
                            pass
                        _t_cues = list(dict.fromkeys(_t_cues))
                        if len(_t_cues) >= 2:
                            try:
                                _q_sql = (
                                    f"SELECT host_word, COUNT(DISTINCT cue_word) as cnt, SUM(weight) as wsum "
                                    f"FROM ldoce_sense_cue_index "
                                    f"WHERE sense_idx = 0 AND cue_word IN ({','.join(['?'] * len(_t_cues))}) "
                                    f"AND host_word != ? AND host_word NOT LIKE '% %' AND host_word NOT LIKE '%-%' "
                                    f"GROUP BY host_word HAVING cnt >= 2 "
                                    f"ORDER BY cnt DESC, wsum DESC LIMIT 35"
                                )
                                _cue_rows = conn_cue.execute(_q_sql, _t_cues + [clean_target]).fetchall()
                            except Exception:
                                _cue_rows = []
                            _allowed_cue_lfs = set(target_lfs_set)
                            if _allowed_cue_lfs & {"noun.communication", "noun.artifact", "noun.possession", "noun.group"}:
                                _allowed_cue_lfs |= {"noun.communication", "noun.artifact", "noun.possession"}
                            if _allowed_cue_lfs & {"noun.state", "noun.feeling", "noun.attribute", "noun.act", "noun.event"}:
                                _allowed_cue_lfs |= {"noun.state", "noun.feeling", "noun.attribute", "noun.act", "noun.event", "noun.cognition"}
                            _wn_coords_enough = len(tier3_coordinates) >= target_count
                            for _hw, _cnt, _wsum in _cue_rows:
                                _lem = str(_hw or "").strip().lower()
                                if (
                                    not _lem.isalpha()
                                    or len(_lem) < 3
                                    or _lem in seen
                                    or _lem in global_synonyms
                                    or (_is_concrete_t and _lem.endswith(("ing", "tion", "sion", "ment", "ness", "ity")))
                                    or not cls.is_exact_pedlex_headword(_lem)
                                    or _is_primarily_adj_or_verb(_lem)
                                    or not _pos_ok(_lem)
                                    or not cls.is_cefr_compliant_distractor(_lem, clean_target)
                                    or _is_informal_or_slang_distractor(_lem)
                                    or not _is_aligned_noun_coord(_lem)
                                ):
                                    continue
                                _h_ent = cls.get_ldoce_entry(_lem) or {}
                                if cls._pos_code(_h_ent.get("pos") or "") != "n":
                                    continue
                                if (
                                    not cls.is_exact_pedlex_headword(_lem)
                                    and int(_h_ent.get("core_level") or 0) == 0
                                    and not any(sn.get("examples") for sn in (_h_ent.get("senses") or []))
                                ):
                                    continue
                                _c_syns_cue = wn.synsets(_lem, "n")
                                if s.lexfile() == "noun.person":
                                    if not _c_syns_cue or _c_syns_cue[0].lexfile() != "noun.person":
                                        continue
                                elif _c_syns_cue and _c_syns_cue[0].lexfile() == "noun.person":
                                    continue
                                if _allowed_cue_lfs and (
                                    not _c_syns_cue or _c_syns_cue[0].lexfile() not in _allowed_cue_lfs
                                ):
                                    continue
                                seen.add(_lem)
                                if _wn_coords_enough:
                                    coord_hyper_overlap.setdefault(_lem, 0)
                                else:
                                    coord_hyper_overlap[_lem] = max(coord_hyper_overlap.get(_lem, 0), int(_cnt))
                                tier3_coordinates.append(_lem)

        # Tier 3.35: Pustejovsky Qualia Scenario Lifecycle Verb Bridge (for sparse WordNet verb branches like `enroll`)
        if (
            wn_pos == "v"
            and definition
            and len(tier0_ldoce_thesaurus) + len(tier2_hyponyms) + len(tier3_coordinates) < target_count * 2
        ):
            _qualia_peers = cls._get_verb_qualia_scenario_peers(
                clean_target,
                definition=definition,
                quote=quote,
                context_anchor=context_anchor,
                anchor_type=anchor_type,
            )
            for _qv, _qinfo in sorted(_qualia_peers.items(), key=lambda kv: (-kv[1]["score"], kv[0])):
                if (
                    _qv in seen
                    or not _qv.isalpha()
                    or not cls.is_exact_pedlex_headword(_qv)
                    or _is_primarily_noun_or_adj_for_verb(_qv)
                    or not _pos_ok(_qv)
                    or not cls.is_cefr_compliant_distractor(_qv, clean_target)
                    or _is_informal_or_slang_distractor(_qv)
                ):
                    continue
                if _qv in global_synonyms:
                    _has_slot_clash, _ = cls.evaluate_syntactic_slot_clash(
                        clean_target,
                        _qv,
                        anchor=context_anchor,
                        anchor_type=anchor_type,
                        pos=pos,
                        quote=quote,
                    )
                    if not _has_slot_clash:
                        continue
                seen.add(_qv)
                coord_hyper_overlap[_qv] = max(coord_hyper_overlap.get(_qv, 0), int(_qinfo["score"]))
                tier3_coordinates.append(_qv)

        # Tier 3.4: Orthogonal-Domain Collective Noun Peer Pool (`_COLLECTIVE_NOUN_ONTOLOGY`)
        # For collective nouns (human role collectives, animal collectives, vessel/batch collectives),
        # ban co-synonyms in the same subgroup and inject orthogonal subgroup heads from the same macro class.
        _is_collective_noun_target = False
        if wn_pos == "n":
            _coll_macro: Optional[str] = None
            _coll_own_sub: Tuple[str, ...] = ()
            for _m_cat, _sub_list in cls._COLLECTIVE_NOUN_ONTOLOGY.items():
                for _s_grp in _sub_list:
                    if clean_target in _s_grp:
                        _coll_macro = _m_cat
                        _coll_own_sub = _s_grp
                        break
                if _coll_macro:
                    break
            if _coll_macro:
                _def_low_coll = (definition or "").strip().lower()
                if _coll_macro == "partitive_measure":
                    _part_def_cues = {
                        "piece", "mass", "amount", "quantity", "line", "drop",
                        "grain", "slice", "sheet", "strip", "strand", "butter",
                        "coal", "smoke", "water", "paper", "hair", "salt",
                    }
                    _is_collective_noun_target = (
                        not _def_low_coll
                        or bool(set(re.findall(r"\b[a-z]{3,}\b", _def_low_coll)) & _part_def_cues)
                    )
                else:
                    _non_coll_prefixes = (
                        "a natural ability", "a long thick stick", "a hard cover", "a flat piece",
                        "a small paper", "a large bag", "a thin flat", "a narrow piece", "a mould", "a mold",
                    )
                    _coll_def_cues = {
                        "people", "person", "workers", "employees", "teachers", "actors", "performers",
                        "musicians", "singers", "sailors", "officers", "soldiers", "customers", "members",
                        "priests", "leaders", "group", "crowd", "organization", "department", "university",
                        "college", "school", "ship", "ships", "plane", "aircraft", "play", "film", "movie",
                        "theatre", "theater", "court", "judge", "decide", "birds", "sheep", "goats", "animals",
                        "cows", "cattle", "wolves", "dogs", "insects", "bees", "ants", "fish", "chickens",
                        "hens", "vehicles", "boats", "things", "items", "objects", "flowers", "keys",
                    }
                    if _def_low_coll:
                        _def_coll_toks = set(re.findall(r"\b[a-z]{3,}\b", _def_low_coll))
                        _is_collective_noun_target = bool(_def_coll_toks & _coll_def_cues) and not _def_low_coll.startswith(_non_coll_prefixes)
                    else:
                        _is_collective_noun_target = bool(not target_synsets or target_synsets[0].lexfile() == "noun.group")
                if _is_collective_noun_target:
                    global_synonyms.update(_coll_own_sub)
                    # Remove orthogonal collective-noun subgroup members that Oxford Writer's Thesaurus
                    # or broad secondary WordNet synsets (e.g. `staff`/`faculty`) lumped into `global_synonyms`.
                    for _other_grp in cls._COLLECTIVE_NOUN_ONTOLOGY[_coll_macro]:
                        if _other_grp == _coll_own_sub:
                            continue
                        for _orth_w in _other_grp:
                            global_synonyms.discard(_orth_w)
                    tier1_synonyms = [w for w in tier1_synonyms if w not in global_synonyms]
                    tier3_coordinates = [w for w in tier3_coordinates if w not in global_synonyms]
                    for _grp_rank, _other_grp in enumerate(cls._COLLECTIVE_NOUN_ONTOLOGY[_coll_macro]):
                        if _other_grp == _coll_own_sub:
                            continue
                        _sorted_grp = sorted(
                            _other_grp,
                            key=lambda _w: cls.distractor_familiarity_key(_w, clean_target)[0],
                        )
                        for _peer_w in _sorted_grp:
                            if (
                                _peer_w != clean_target
                                and _peer_w not in global_synonyms
                                and _peer_w not in exclude_words
                                and cls.is_cefr_compliant_distractor(_peer_w, clean_target)
                            ):
                                if _peer_w not in seen:
                                    seen.add(_peer_w)
                                    tier3_coordinates.append(_peer_w)
                                coord_hyper_overlap[_peer_w] = max(coord_hyper_overlap.get(_peer_w, 0), 35 - _grp_rank)
                                break

        # Sort tier3_coordinates by hypernym domain overlap, primary-sense domain alignment, PedLex richness, and core B1/B2 CEFR proximity
        # (cap target reference rank at B2=4 so C2-rated targets like 'slump' or 'motel' do not prefer obscure C2 words over B1/B2 peers)
        if tier2_hyponyms or tier3_coordinates or wn_pos == "n":
            t_lvl_str = cls._lexicon_profile(clean_target)[2] or "B1"
            t_rank = min(cls.CEFR_ORDER.get(t_lvl_str, 3), 4)
            analyzer = cls.get_cefr_analyzer()
            t_float = analyzer.get_average_word_level_float(clean_target) if hasattr(analyzer, "get_average_word_level_float") else float(t_rank)
            if t_float is None:
                t_float = float(t_rank)
            p_code_coord = {"n": "n", "v": "v", "a": "adj", "r": "adv"}.get(wn_pos or "n", "n")

            def _coord_sort_key(cand_w: str) -> Tuple[int, int, int, int, int, int, int, int, int, int]:
                c_ent = cls.get_ldoce_entry(cand_w) or {}
                c_senses = c_ent.get("senses") or []
                pos_senses = [sn for sn in c_senses if not sn.get("pos") or cls._pos_code(sn.get("pos")) == p_code_coord] or c_senses
                has_ex = any(sn.get("examples") for sn in pos_senses)
                countability_mismatch = 0
                if wn_pos == "n" and pos_senses:
                    c_g0 = str(pos_senses[0].get("grammar") or "").lower()
                    if target_strictly_countable and "uncountable" in c_g0 and "countable" not in re.sub(r"\buncountable\b", "", c_g0):
                        countability_mismatch = 1
                    elif target_strictly_uncountable and "uncountable" not in c_g0:
                        countability_mismatch = 1
                prim_aligned = False
                if pos_senses and coord_domain_cues:
                    prim_def = str(pos_senses[0].get("definition") or "").lower()
                    prim_toks = set(re.findall(r"\b[a-z]{3,}\b", prim_def)) - {cand_w}
                    if prim_toks & (coord_domain_cues - {cand_w}) and len(prim_toks) >= 2:
                        prim_aligned = True
                if not prim_aligned and target_lfs_set:
                    c_wn = wn.synsets(cand_w, wn_pos)
                    if c_wn and c_wn[0].lexfile() in target_lfs_set:
                        prim_aligned = True
                if not prim_aligned and _is_collective_noun_target and coord_hyper_overlap.get(cand_w, 0) >= 30:
                    prim_aligned = True
                c_lvl_raw = cls._lexicon_profile(cand_w)[2]
                c_lvl = c_lvl_raw or "C1"
                c_rank = cls.CEFR_ORDER.get(c_lvl, 5)
                c_float = analyzer.get_average_word_level_float(cand_w) if hasattr(analyzer, "get_average_word_level_float") else float(c_rank)
                if c_float is None:
                    c_float = float(c_rank)
                diff_overshoot = 1 if (c_float > t_float + 0.6) else 0
                has_ldoce_core = 0 if int(c_ent.get("core_level") or 0) > 0 else 1
                is_curated_peer = 0 if coord_hyper_overlap.get(cand_w, 0) >= 20 else 1
                is_non_core = 0 if (cls.is_exact_pedlex_headword(cand_w) or has_ldoce_core == 0 or (c_lvl_raw in ("A1", "A2", "B1", "B2") and diff_overshoot == 0)) else 1
                gerund_mismatch = 1 if (wn_pos == "n" and cand_w.endswith("ing") and not clean_target.endswith("ing")) else 0
                return (
                    gerund_mismatch,
                    countability_mismatch,
                    0 if prim_aligned else 1,
                    is_curated_peer,
                    diff_overshoot,
                    has_ldoce_core,
                    is_non_core,
                    -coord_hyper_overlap.get(cand_w, 0),
                    0 if has_ex else 1,
                    abs(c_rank - t_rank),
                    1 if c_rank >= 6 else 0,
                )
            if tier2_hyponyms:
                tier2_hyponyms.sort(key=_coord_sort_key)
            if tier3_coordinates:
                tier3_coordinates.sort(key=_coord_sort_key)

        # Candidate prioritization: PedLex Thesaurus & Activators are primary (Tier 0); WordNet is secondary/fallback
        if wn_pos == "v":
            antonym_candidates = [
                ant for ant in sense_antonyms
                if ant != clean_target and cls.is_cefr_compliant_distractor(ant, clean_target)
            ]
            candidates = tier0_ldoce_thesaurus + tier2_hyponyms + tier3_coordinates + antonym_candidates
        elif wn_pos == "a":
            # For adjectives: retain satellite_synonyms first to satisfy existing fine-grained satellite tests,
            # then inject PedLex thesaurus peers, compliant antonyms, followed by attribute coordinates.
            antonym_candidates = [
                ant for ant in sense_antonyms
                if ant != clean_target and cls.is_cefr_compliant_distractor(ant, clean_target)
            ]
            candidates = tier2_satellites + tier0_ldoce_thesaurus + antonym_candidates + tier3_attributes + tier1_synonyms
            # Preposition valency cloze (): prioritize academic adjectives governing DIFFERENT prepositions
            if anchor_type == "prep" and context_anchor and not (
                clean_target.endswith("ed") and context_anchor.strip().lower() in ("by", "with")
            ):
                clean_anchor = context_anchor.strip().lower()
                valency_candidates = []
                for p, adjs in cls._ADJ_PREP_VALENCY_MAP.items():
                    if p != clean_anchor:
                        for a in adjs:
                            if a != clean_target and a not in exclude_words and cls.is_cefr_compliant_distractor(a, clean_target):
                                valency_candidates.append(a)
                # Prioritize local semantic peers (satellites, PedLex thesaurus) for domain relevance,
                # then complement with academic adjectives governing distinct prepositions
                candidates = tier2_satellites + tier0_ldoce_thesaurus + antonym_candidates + valency_candidates + tier3_attributes + tier1_synonyms
        elif wn_pos == "n":
            extra_noun_ants = [
                a for a in sense_antonyms[1:]
                if cls.is_cefr_compliant_distractor(a, clean_target) and not _is_informal_or_slang_distractor(a)
            ]
            if _is_collective_noun_target:
                candidates = tier3_coordinates + tier0_ldoce_thesaurus + tier2_hyponyms + tier1_synonyms + extra_noun_ants
            else:
                candidates = tier0_ldoce_thesaurus + tier1_synonyms + tier2_hyponyms + tier3_coordinates + extra_noun_ants
        else:
            candidates = tier0_ldoce_thesaurus + tier1_synonyms + tier2_hyponyms + tier3_coordinates

        return {
            "tier0_ldoce_thesaurus": tier0_ldoce_thesaurus,
            "tier1_synonyms": tier1_synonyms,
            "tier2_hyponyms": tier2_hyponyms,
            "tier2_satellites": tier2_satellites,
            "tier3_attributes": tier3_attributes,
            "tier3_coordinates": tier3_coordinates,
            "candidates": candidates,
            "sense_antonyms": sense_antonyms,
            "_is_collective_noun_target": _is_collective_noun_target,
            "_is_aligned_noun_coord": _is_aligned_noun_coord,
            "_coord_sort_key": _coord_sort_key,
        }
