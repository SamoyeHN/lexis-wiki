"""Noun countability contrast, collective noun ontology, and grammar profiles."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple


class DistractorGuardsCountabilityMixin:
    """Guards for noun countability symmetry, collective nouns, taboo terms, and grammar profiles."""

    # Orthogonal-Domain Collective Noun Peer Pool:
    # Collective nouns (WordNet `noun.group`) have no antonyms and often sit under fragmented or
    # metonymic WordNet hypernyms (`force/personnel`, `social unit`, `gathering` with `room`/`suite`).
    # Grouping them into three macro ontological classes (`human_role`, `animal`, `vessel_batch`)
    # where each inner tuple represents a synonym subgroup guarantees:
    # 1. Intra-subgroup synonyms (e.g. `staff`/`personnel`/`workforce`, `jury`/`panel`) are mutually banned.
    # 2. Orthogonal subgroup heads within the same macro class serve as high-discrimination Tier 3 coordinates.
    _COLLECTIVE_NOUN_ONTOLOGY: Dict[str, List[Tuple[str, ...]]] = {
        "human_role": [
            ("staff", "personnel", "workforce"),
            ("crew",),
            ("cast", "troupe"),
            ("faculty",),
            ("jury", "panel", "tribunal"),
            ("audience", "crowd", "congregation", "throng", "mob"),
            ("committee", "council", "board", "commission", "assembly", "delegation"),
            ("choir", "chorus", "orchestra", "band"),
            ("clientele",),
            ("clergy",),
            ("police", "squad", "troop", "regiment", "contingent"),
            ("gentry", "aristocracy", "elite"),
        ],
        "animal": [
            ("flock",),
            ("herd", "cattle", "livestock"),
            ("pack",),
            ("swarm", "colony"),
            ("shoal",),
            ("poultry", "brood"),
        ],
        "vessel_batch": [
            ("fleet", "convoy", "armada"),
            ("batch", "consignment"),
            ("bundle", "bunch", "cluster", "array"),
            ("cargo", "freight"),
        ],
    }


    _PREDICATIVE_ONLY_ADJS = {
        "asleep", "afraid", "alive", "alone", "ashamed", "aware", "awake", "content", "unable", "prone", "glad"
    }


    @classmethod
    def _are_orthogonal_collective_nouns(cls, w1: str, w2: str) -> bool:
        """True when `w1` and `w2` belong to DIFFERENT synonym subgroups in `_COLLECTIVE_NOUN_ONTOLOGY`
        (e.g., `staff` vs `crew`/`cast`/`jury`/`faculty`, or `flock` vs `herd`/`pack`/`swarm`).
        Oxford Writer's Thesaurus groups co-hyponym collective nouns in the same thesaurus bucket,
        whereas different subgroups denote mutually exclusive roles/species and are prime coordinates."""
        a = (w1 or "").strip().lower()
        b = (w2 or "").strip().lower()
        if not a or not b or a == b:
            return False
        for sub_list in cls._COLLECTIVE_NOUN_ONTOLOGY.values():
            grp_a = next((i for i, grp in enumerate(sub_list) if a in grp), None)
            grp_b = next((i for i, grp in enumerate(sub_list) if b in grp), None)
            if grp_a is not None and grp_b is not None:
                return grp_a != grp_b
        return False

    @classmethod
    def _is_collocational_partner(cls, target: str, cand: str) -> bool:
        """Anti-Collocation Distractor Gate: Ensures a candidate that is a bound collocational
        partner, predicate head, or dependent object of the target word (e.g. 'breach' with 'convention')
        never serves as a distractor for that target word, preventing distractor-to-stem leakage."""
        t_low = (target or "").strip().lower()
        c_low = (cand or "").strip().lower()
        if not t_low or not c_low or t_low == c_low:
            return False
        # 1. Oxford collocations of target
        t_coll = cls.get_oxford_collocations(t_low) or {}
        for cat in ("colloc_nouns", "verb_before", "noun_after", "verb"):
            for cw in t_coll.get(cat) or []:
                if re.search(rf"\b{re.escape(c_low)}\b", cw.lower()):
                    return True
        # 2. Oxford collocations of candidate
        c_coll = cls.get_oxford_collocations(c_low) or {}
        for cat in ("colloc_nouns", "verb_before", "noun_after", "verb"):
            for cw in c_coll.get(cat) or []:
                if re.search(rf"\b{re.escape(t_low)}\b", cw.lower()):
                    return True
        # 3. Direct relational phrases in PedLex phrase index
        for p_cand in (f"{c_low} of {t_low}", f"{t_low} of {c_low}"):
            if cls.lookup_ldoce_phrase(p_cand):
                return True
        return False

    @classmethod
    def _is_taboo_or_proper_noun(cls, cand_w: str, wn_pos: Optional[str] = None) -> bool:
        """Unified safety gate blocking capitalized proper nouns/demonyms (e.g. Polack, Islam)
        and taboo/offensive/vulgar/derogatory/trademark entries across PedLex and WordNet."""
        cw = (cand_w or "").strip().lower()
        if not cw:
            return True
        _classroom_sensitive = {
            "nudity", "mistress", "prostitute", "prostitution", "harlot",
            "concubine", "brothel", "adultery", "erotic", "porn", "pornography",
            "rape", "rapist", "incest", "orgasm", "molest", "molester",
            "pedophile", "paedophile", "pimp", "whore", "slut", "hooker",
            "fetish", "penis", "vagina", "anus", "masturbate", "masturbation",
            "sodomy", "bestiality", "fornication", "genitals", "genitalia",
            "testicle", "scrotum", "clitoris", "erection", "ejaculate", "ejaculation",
        }
        if cw in _classroom_sensitive:
            return True
        p_code = {"n": "n", "v": "v", "a": "adj", "r": "adv", "s": "adj"}.get(wn_pos or "", "")
        e_cand = cls.get_ldoce_entry(cw)
        if e_cand:
            homographs = e_cand.get("homographs") or []
            if homographs and all(
                isinstance(h.get("headword"), str) and h["headword"] and h["headword"][0].isupper()
                for h in homographs
            ):
                return True
            senses = e_cand.get("senses") or []
            pos_senses = [s for s in senses if not p_code or not s.get("pos") or cls._pos_code(s.get("pos")) == p_code] or senses
            _hard_bad_regs = ("taboo", "offensive", "vulgar", "derogatory", "trademark")
            _hard_bad_defs = (
                "offensive word", "do not use this word", "very offensive",
                "used as an insult", "disparaging", "ethnic slur",
            )
            for s in pos_senses:
                reg_str = str(s.get("register") or "").lower()
                defn_str = str(s.get("definition") or "").lower()
                if any(tok in reg_str for tok in _hard_bad_regs):
                    return True
                if any(cue in defn_str for cue in _hard_bad_defs):
                    return True
        try:
            wn = cls.get_wordnet()
            if wn_pos in ("a", "s"):
                c_words = wn.words(cw, pos="a") + wn.words(cw, pos="s")
            elif wn_pos in ("n", "v", "r"):
                c_words = wn.words(cw, pos=wn_pos)
            else:
                c_words = wn.words(cw)
            c_syns = [s for w in c_words for s in w.synsets()]
            if c_syns and all(
                all(lemma and lemma[0].isupper() for lemma in s.lemmas())
                for s in c_syns if s.lemmas()
            ):
                return True
        except Exception:
            pass
        return False

    _NOUN_COUNTABILITY_CONTRAST_MAP: Dict[str, List[str]] = {
        "device": ["equipment", "hardware", "machinery"],
        "devices": ["equipment", "hardware", "machinery"],
        "appliance": ["equipment", "hardware", "machinery"],
        "appliances": ["equipment", "hardware", "machinery"],
        "instrument": ["equipment", "apparatus", "machinery"],
        "gadget": ["equipment", "hardware"],
        "tool": ["equipment", "hardware", "machinery"],
        "machine": ["machinery", "equipment", "hardware"],
        "facility": ["equipment", "infrastructure", "funding"],
        "facilities": ["equipment", "infrastructure", "funding"],
        "suggestion": ["advice", "guidance", "counsel"],
        "proposal": ["advice", "guidance", "legislation"],
        "recommendation": ["advice", "guidance", "feedback"],
        "tip": ["advice", "guidance", "information"],
        "breakthrough": ["progress", "headway", "research"],
        "advance": ["progress", "headway"],
        "stride": ["progress", "headway"],
        "achievement": ["progress", "success"],
        "belongings": ["luggage", "baggage", "property"],
        "valuables": ["wealth", "jewelry", "luggage"],
        "possession": ["property", "wealth", "luggage"],
        "possessions": ["property", "wealth", "luggage"],
        "clothes": ["clothing", "apparel", "footwear"],
        "garment": ["clothing", "apparel"],
        "outfit": ["clothing", "apparel"],
        "costume": ["clothing", "apparel"],
        "finding": ["information", "evidence", "research"],
        "findings": ["information", "evidence", "research"],
        "fact": ["information", "evidence", "knowledge"],
        "detail": ["information", "evidence"],
        "clue": ["evidence", "information"],
        "statistic": ["information", "data", "research"],
        "study": ["research", "evidence", "literature"],
        "experiment": ["research", "testing"],
        "law": ["legislation", "regulation"],
        "statute": ["legislation", "law"],
        "rule": ["legislation", "guidance"],
        "loan": ["credit", "funding", "finance"],
        "grant": ["funding", "finance", "aid"],
        "subsidy": ["funding", "finance", "assistance"],
        "donation": ["funding", "charity", "aid"],
        "resource": ["funding", "equipment", "wealth"],
        "resources": ["funding", "equipment", "wealth"],
        "comment": ["feedback", "criticism", "advice"],
        "remark": ["feedback", "commentary"],
        "remedy": ["assistance", "relief", "treatment"],
        "measure": ["legislation", "action", "assistance"],
        "emission": ["pollution", "contamination"],
        "emissions": ["pollution", "contamination"],
        "pollutant": ["pollution", "contamination", "waste"],
        "contaminant": ["pollution", "contamination", "waste"],
        "chair": ["furniture", "seating"],
        "table": ["furniture", "equipment"],
        "desk": ["furniture", "equipment"],
        "cabinet": ["furniture", "storage"],
        "vehicle": ["traffic", "transportation", "transport"],
        "poem": ["poetry", "literature", "verse"],
        "view": ["scenery", "landscape"],
        "task": ["homework", "work", "labor"],
        "assignment": ["homework", "work", "research"],
        "chore": ["housework", "work", "labor"],
        "job": ["employment", "work", "labor"],
        "product": ["merchandise", "produce", "goods"],
        "item": ["merchandise", "equipment", "baggage"],
        "commodity": ["merchandise", "commerce", "trade"],
        "permit": ["permission", "authorization", "consent"],
        "license": ["permission", "authorization"],
        "program": ["software", "programming", "funding"],
        "application": ["software", "technology"],
        "suitcase": ["baggage", "luggage"],
        "bag": ["baggage", "luggage"],
        "report": ["news", "information"],
        "story": ["news", "information"],
        "article": ["news", "information"],
        "bulletin": ["news", "information"],
        "hotel": ["accommodation", "housing"],
        "room": ["accommodation", "housing"],
        "apartment": ["accommodation", "housing"],
        "house": ["accommodation", "housing"],
        "coin": ["cash", "money", "currency"],
        "bill": ["cash", "money", "currency"],
        "note": ["cash", "money", "currency"],
        "ring": ["jewelry", "jewellery"],
        "necklace": ["jewelry", "jewellery"],
        "bracelet": ["jewelry", "jewellery"],
        "knife": ["cutlery"],
        "fork": ["cutlery"],
        "spoon": ["cutlery"],
        "pen": ["stationery"],
        "pencil": ["stationery"],
        "notebook": ["stationery"],
        "envelope": ["stationery"],
        "action": ["behavior", "conduct"],
        "deed": ["behavior", "conduct"],
        "gesture": ["behavior", "conduct"],
        "habit": ["behavior", "conduct"],
        "cheer": ["applause"],
        "ovation": ["applause"],
        "car": ["traffic", "transportation", "transport"],
        "bus": ["traffic", "transportation", "transport"],
        "train": ["traffic", "transportation", "transport"],
        "instruction": ["guidance", "advice", "training"],
        "direction": ["guidance", "advice"],
        "guideline": ["guidance", "advice"],
    }

    _NOUN_COUNTABILITY_REVERSE_MAP: Optional[Dict[str, List[str]]] = None

    @classmethod
    def get_noun_countability_contrast_map(cls, reverse: bool = False) -> Dict[str, List[str]]:
        """Returns the noun countability contrast map (countable -> uncountable foils, or reversed)."""
        if not reverse:
            return cls._NOUN_COUNTABILITY_CONTRAST_MAP
        if cls._NOUN_COUNTABILITY_REVERSE_MAP is None:
            rev: Dict[str, List[str]] = {}
            for c_word, u_list in cls._NOUN_COUNTABILITY_CONTRAST_MAP.items():
                for u_word in u_list:
                    if u_word not in rev:
                        rev[u_word] = []
                    if c_word not in rev[u_word]:
                        rev[u_word].append(c_word)
            cls._NOUN_COUNTABILITY_REVERSE_MAP = rev
        return cls._NOUN_COUNTABILITY_REVERSE_MAP

    @classmethod
    def _is_primarily_uncountable_noun(cls, word: str) -> bool:
        """Determines if a noun is strictly or primarily uncountable in PedLex."""
        w = (word or "").strip().lower()
        if not w:
            return False
        prof = cls.get_word_gram_profile(w, pos="noun")
        if prof.get("pure_uncountable"):
            return True
        if prof.get("pure_countable") or prof.get("pure_plural"):
            return False
        ent = cls.get_ldoce_entry(w)
        if not ent:
            return False
        senses = [s for s in (ent.get("senses") or []) if not s.get("pos") or cls._pos_code(s.get("pos")) == "n"] or ent.get("senses") or []
        if not senses:
            return False
        g0 = str(senses[0].get("grammar") or "").lower()
        m_u = re.search(r"\buncountable\b", g0)
        m_c = re.search(r"\bcountable\b", g0)
        if m_u and not m_c:
            return True
        if m_u and m_c:
            return m_u.start() < m_c.start()
        return False

    @classmethod
    def _is_primarily_countable_noun(cls, word: str) -> bool:
        """Determines if a noun is strictly or primarily countable (or plural) in PedLex."""
        w = (word or "").strip().lower()
        if not w or w.endswith("ics"):
            return False
        prof = cls.get_word_gram_profile(w, pos="noun")
        if prof.get("pure_countable") or prof.get("pure_plural"):
            return True
        if prof.get("pure_uncountable"):
            return False
        ent = cls.get_ldoce_entry(w)
        if not ent:
            return False
        senses = [s for s in (ent.get("senses") or []) if not s.get("pos") or cls._pos_code(s.get("pos")) == "n"] or ent.get("senses") or []
        if not senses:
            return False
        g0 = str(senses[0].get("grammar") or "").lower()
        m_c = re.search(r"\bcountable\b", g0)
        m_u = re.search(r"\buncountable\b", g0)
        if m_c and not m_u:
            return True
        if m_c and m_u:
            return m_c.start() < m_u.start()
        return False

    @classmethod
    def get_word_gram_profile(cls, word: str, pos: Optional[str] = None) -> Dict[str, Any]:
        """Extracts a unified grammatical constraint profile for `word` from `pedlex.db` (`senses`, `thesaurus`, and `grammar_boxes`)."""
        w_clean = (word or "").strip().lower()
        profile: Dict[str, Any] = {
            "only_before_noun": False,
            "not_before_noun": w_clean in cls._PREDICATIVE_ONLY_ADJS,
            "pure_uncountable": False,
            "pure_countable": False,
            "pure_plural": False,
            "pure_intransitive": False,
            "no_passive": False,
            "no_progressive": False,
            "forbidden_frames": set(),
            "allowed_frames": set(),
            "comparison_peers": set(),
        }
        if not w_clean:
            return profile
        entry = cls.get_ldoce_entry(w_clean)
        if not entry:
            return profile

        pos_norm = (pos or "").strip().lower()
        wn_pos = "a" if pos_norm.startswith("a") or pos_norm == "j" else (
            "n" if pos_norm.startswith("n") else (
                "v" if pos_norm.startswith("v") else ""
            )
        )

        senses = entry.get("senses") or []
        matching_senses = []
        for s in senses:
            sp = (s.get("pos") or entry.get("pos") or "").lower()
            if wn_pos == "a" and "adj" in sp:
                matching_senses.append(s)
            elif wn_pos == "n" and "noun" in sp:
                matching_senses.append(s)
            elif wn_pos == "v" and "verb" in sp:
                matching_senses.append(s)
            elif not wn_pos:
                matching_senses.append(s)
        if not matching_senses:
            if wn_pos:
                return profile
            matching_senses = senses

        raw_sense_grams = [str(s.get("grammar") or "").strip().lower() for s in matching_senses]
        non_empty_sense_grams = [g for g in raw_sense_grams if g]
        # A word-level constraint holds across the POS when EVERY matching sense has a grammar tag
        # satisfying the constraint (e.g., entryhead <gram> propagated to all senses), OR when no
        # sense carries a <gram> tag and the self-referential THESAURUS exponent supplies it
        # (e.g., 'added', 'additional'). If a polysemous word has 1 restricted sense and other
        # unrestricted senses (grammar == ""), it is NOT globally restricted across all senses.
        if matching_senses and len(non_empty_sense_grams) == len(matching_senses):
            grams = list(non_empty_sense_grams)
        elif not non_empty_sense_grams:
            w_ortho_gram = set(cls.orthographic_variants(w_clean)) if hasattr(cls, "orthographic_variants") else {w_clean}
            grams = [
                str(th["grammar"]).strip().lower()
                for th in (entry.get("thesaurus") or [])
                if (th.get("word") or "").strip().lower() in w_ortho_gram and th.get("grammar")
            ]
        else:
            grams = []

        if grams:
            if (not wn_pos or wn_pos == "a") and all("only before noun" in g for g in grams):
                profile["only_before_noun"] = True
            if (not wn_pos or wn_pos == "a") and all("not before noun" in g or "never before noun" in g for g in grams):
                # Guard against PedLex sense grammar pollution by embedded sub-phrases
                # (e.g., `out of this world [not before noun]` under `incredible`, which has attested noun collocations)
                has_noun_collocs = bool((entry.get("collocations") or {}).get("nouns"))
                if not has_noun_collocs or w_clean in cls._PREDICATIVE_ONLY_ADJS:
                    profile["not_before_noun"] = True
            if (not wn_pos or wn_pos == "n") and all(
                "uncountable" in g and not re.search(r"\bcountable\b", g)
                for g in grams
            ):
                profile["pure_uncountable"] = True
            if (not wn_pos or wn_pos == "v") and all(
                "intransitive" in g and not re.search(r"\btransitive\b", g)
                for g in grams
            ):
                profile["pure_intransitive"] = True

        if (
            (not wn_pos or wn_pos == "v")
            and non_empty_sense_grams
            and any("intransitive" in g for g in non_empty_sense_grams)
            and not any(re.search(r"\btransitive\b", g) for g in non_empty_sense_grams)
            and not cls._verb_takes_object(w_clean)
        ):
            profile["pure_intransitive"] = True

        if not wn_pos or wn_pos == "n":
            if (
                not w_clean.endswith("s")
                and raw_sense_grams
                and "uncountable" in raw_sense_grams[0]
                and not any(re.search(r"\bcountable\b", g) for g in raw_sense_grams)
                and not any("plural" in g for g in raw_sense_grams)
            ):
                profile["pure_uncountable"] = True
            if (
                raw_sense_grams
                and bool(re.search(r"\bcountable\b", raw_sense_grams[0]))
                and not any("uncountable" in g for g in raw_sense_grams)
            ):
                profile["pure_countable"] = True
            if (
                raw_sense_grams
                and "plural" in raw_sense_grams[0]
                and not any("uncountable" in g for g in raw_sense_grams)
            ):
                profile["pure_plural"] = True

        for gb in entry.get("grammar_boxes") or []:
            for p in gb.get("comparison_peers") or []:
                p_clean = str(p).strip().lower()
                if p_clean and p_clean != w_clean and " " not in p_clean:
                    profile["comparison_peers"].add(p_clean)
            syn_c = gb.get("syntactic_constraints") or {}
            # Only apply no_passive / no_progressive globally when box is not restricted to a subset of senses
            gb_senses = gb.get("sense_indices") or []
            if not gb_senses or len(gb_senses) >= len(matching_senses):
                if syn_c.get("no_passive"):
                    profile["no_passive"] = True
                if syn_c.get("no_progressive"):
                    profile["no_progressive"] = True
            for ff in syn_c.get("forbidden_frames") or []:
                profile["forbidden_frames"].add(ff)
            for af in syn_c.get("allowed_frames") or []:
                profile["allowed_frames"].add(af)

        return profile


    @classmethod
    def get_word_to_complement_profile(cls, word: str, pos: Optional[str] = None) -> Tuple[bool, bool]:
        """Returns `(has_to_inf, has_to_prep)` indicating whether `word` accepts an infinitive
        complement (`to + do`) and/or prepositional complement (`to + noun/gerund`).
        
        Sourced deterministically from:
          1. LDOCE cxn_specs & pedlex_cxn_index (e.g. N_SHELL_TO_DO, V_INF_ONLY, ADJ_ANIMATE_CONTROL)
          2. LDOCE pattern bundles & patterns (e.g. 'reluctance to do sth' vs 'aversion to sth')
          3. Oxford Collocations Dictionary (`prep` == 'to') & _ADJ_PREP_VALENCY_MAP['to']
        """
        w = (word or "").strip().lower()
        if not w:
            return False, False
        has_to_inf = False
        has_to_prep = False

        entry = cls.get_ldoce_entry(w)
        if entry:
            for s in entry.get("senses", []):
                for cs in s.get("cxn_specs", []):
                    sg = cs.get("subcat_group", "")
                    if sg in (
                        "N_SHELL_TO_DO", "V_INF_ONLY", "V_BOTH_GERUND_INF",
                        "ADJ_ANIMATE_CONTROL", "ADJ_IT_EXTRAPOS", "ADJ_TOUGH_BOTH"
                    ):
                        has_to_inf = True

            for s in entry.get("senses", []):
                all_p = s.get("patterns", []) + [
                    b.get("pattern", "") for b in s.get("pattern_bundles", []) if isinstance(b, dict)
                ]
                for p in all_p:
                    if not p or not isinstance(p, str):
                        continue
                    p_low = p.lower()
                    if any(cue in p_low for cue in [
                        "to do", "to be", "to have", "(to do", "to + inf", "to inf", "to [do", "to act"
                    ]):
                        has_to_inf = True
                    elif any(cue in p_low for cue in [
                        "to something", "to somebody", "to sth", "to sb", "to (doing)", "to doing", "to noun"
                    ]) or p_low.endswith(" to"):
                        has_to_prep = True

        ocd = cls.get_oxford_collocations()
        if "to" in ocd.get(w, {}).get("prep", []):
            has_to_prep = True

        if w in cls._ADJ_PREP_VALENCY_MAP.get("to", []):
            has_to_prep = True

        return has_to_inf, has_to_prep



__all__ = ["DistractorGuardsCountabilityMixin"]
