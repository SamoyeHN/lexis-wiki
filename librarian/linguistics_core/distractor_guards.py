"""Syntactic slot clash, double-key collision, countability contrast, and valency guards."""

from __future__ import annotations

import logging
import os
import re
import sqlite3
from typing import Any, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger("librarian.linguistics.guards")


class DistractorGuardsMixin:

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

    # Family 7 Fronted Negative / Restrictive Inversion Arena:
    # High-order negative or restrictive triggers that trigger Subject-Auxiliary Inversion (SAI)
    # when fronted at the beginning of a clause/sentence (e.g. 'Hardly had the meeting begun when...').
    _NEGATIVE_INVERSION_ADVERBS: Set[str] = {
        "hardly", "scarcely", "barely",
        "seldom", "rarely",
        "little", "no sooner",
    }
    _NEGATIVE_INVERSION_PHRASES: Set[str] = {
        "no sooner", "not only", "not until",
        "under no circumstances", "on no account", "in no way",
        "at no time", "by no means", "only then", "only later",
        "only when", "only after", "only by", "only if",
    }
    _ALL_NEGATIVE_INVERSION_TRIGGERS: Set[str] = _NEGATIVE_INVERSION_ADVERBS | _NEGATIVE_INVERSION_PHRASES

    _AUX_INV_RE: str = r"(?:had|have|has|did|do|does|was|were|is|are|can|could|will|would|should|must|shall|might|may)"
    _SUBJ_INV_RE: str = r"(?:i|you|he|she|it|we|they|one|the|a|an|passengers?|students?|people|[a-z]+)"

    @classmethod
    def is_fronted_inversion_context(
        cls,
        quote: Optional[str] = None,
        anchor: Optional[str] = None,
        anchor_type: Optional[str] = None,
        target: Optional[str] = None,
    ) -> bool:
        """True when the syntactic context represents a fronted negative/restrictive inversion slot (Aux + Subj inversion)."""
        if anchor_type == "fronted_inv" or anchor == "inversion":
            return True
        if not quote:
            return False
        q = quote.strip()
        inv_triggers_pat = "|".join(re.escape(w) for w in sorted(cls._NEGATIVE_INVERSION_ADVERBS, key=len, reverse=True))
        target_esc = (
            rf"(?:{re.escape(target)}|{inv_triggers_pat})"
            if target
            else r"[a-zA-Z -]+"
        )
        slot_pattern = (
            rf"(?:^|[.;!?]\s*|(?:^|[,;!?])\s*\b(?:and|but)\s+)[\"\'“”‘’\s]*"
            rf"(?:_{{2,}}|{target_esc})\s+{cls._AUX_INV_RE}\s+{cls._SUBJ_INV_RE}\b"
        )
        return bool(re.search(slot_pattern, q, re.IGNORECASE))

    @classmethod
    def is_light_verb_collocation_context(
        cls,
        quote: Optional[str] = None,
        anchor: Optional[str] = None,
        anchor_type: Optional[str] = None,
        target: Optional[str] = None,
    ) -> bool:
        """Returns True if the context represents the verb slot of an attested light-verb collocation."""
        return cls.detect_light_verb_collocation_context(
            quote=quote, anchor=anchor, anchor_type=anchor_type, target=target
        ) is not None

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

    _STANDARD_PREPOSITIONS = {
        "into", "to", "of", "for", "with", "in", "on", "from", "at",
        "about", "towards", "against", "over", "between", "through", "under", "upon"
    }

    @classmethod
    def get_governed_prepositions(cls, word: str, pos: Optional[str] = None) -> Set[str]:
        """Returns the set of bound/governed prepositions for `word` (noun, adjective, or verb).
        
        Sourced deterministically from:
          1. Oxford Collocations Dictionary (`prep` section)
          2. _ADJ_PREP_VALENCY_MAP
          3. LDOCE pattern bundles & patterns (e.g. 'insight into', 'overview of', 'allergic to')
        """
        w = (word or "").strip().lower()
        if not w:
            return set()
        preps: Set[str] = set()

        # 1. Oxford Collocations Dictionary
        ocd = cls.get_oxford_collocations()
        if w in ocd:
            for p in ocd[w].get("prep", []):
                p_clean = p.strip().lower()
                if p_clean in cls._STANDARD_PREPOSITIONS:
                    preps.add(p_clean)

        # 2. _ADJ_PREP_VALENCY_MAP
        for p, adjs in cls._ADJ_PREP_VALENCY_MAP.items():
            if w in adjs:
                preps.add(p)

        # 3. LDOCE patterns and pattern_bundles
        entry = cls.get_ldoce_entry(w)
        if entry:
            for s in entry.get("senses", []):
                all_pats = s.get("patterns", []) + [
                    b.get("pattern", "") for b in s.get("pattern_bundles", []) if isinstance(b, dict)
                ]
                for pat in all_pats:
                    if not pat or not isinstance(pat, str):
                        continue
                    tokens = pat.lower().split()
                    for tok in tokens:
                        if tok in cls._STANDARD_PREPOSITIONS:
                            preps.add(tok)

        return preps

    @classmethod
    def _verb_has_intransitive_prep_pattern(cls, word: str, prep: str) -> bool:
        """Checks whether `word` (or its orthographic variants) has an attested LDOCE verb pattern
        where the verb is immediately followed by `prep` without an intervening object (`sb`/`sth`/`somebody`/`something`),
        e.g., `enrol in`, `participate in`, `graduate from`, `apply for`, `listen to`."""
        w = (word or "").strip().lower()
        p = (prep or "").strip().lower()
        if not w or not p:
            return False
        entry = cls.get_ldoce_entry(w) or cls.get_ldoce_entry(cls.lemma_of(w))
        if not entry:
            return False
        variants = set(cls.orthographic_variants(w)) | set(cls.orthographic_variants(cls.lemma_of(w)))
        for s in entry.get("senses", []):
            if s.get("pos") and cls._pos_code(s["pos"]) != "v":
                continue
            all_pats = list(s.get("patterns") or []) + [
                b.get("pattern", "") for b in (s.get("pattern_bundles") or []) if isinstance(b, dict)
            ]
            for pat in all_pats:
                if not pat or not isinstance(pat, str):
                    continue
                p_low = pat.strip().lower()
                if p == "to" and re.search(r"\bto\s+(?:do|be|have)\b", p_low):
                    continue
                for v_var in variants:
                    m = re.match(rf"^{re.escape(v_var)}\s+([a-z/]+)\b", p_low)
                    if m:
                        prep_cluster = [tok.strip() for tok in m.group(1).split("/") if tok.strip()]
                        if p in prep_cluster:
                            return True
        return False

    @classmethod
    def _verb_has_transitive_prep_pattern(cls, word: str, prep: str) -> bool:
        """Checks whether `word` (or its orthographic variants) has an attested LDOCE verb pattern
        where the verb takes an intervening object before `prep`, e.g. `charge sb (money) for sth`,
        `remind sb of sth`, `deprive sb of sth`, `rob sb of sth`, `congratulate sb on sth`."""
        w = (word or "").strip().lower()
        p = (prep or "").strip().lower()
        if not w or not p:
            return False
        entry = cls.get_ldoce_entry(w) or cls.get_ldoce_entry(cls.lemma_of(w))
        if not entry:
            return False
        variants = set(cls.orthographic_variants(w)) | set(cls.orthographic_variants(cls.lemma_of(w)))
        obj_pat = r"(?:sb|sth|somebody|something|someone|oneself)"
        for s in entry.get("senses", []):
            if s.get("pos") and cls._pos_code(s["pos"]) != "v":
                continue
            all_pats = list(s.get("patterns") or []) + [
                b.get("pattern", "") for b in (s.get("pattern_bundles") or []) if isinstance(b, dict)
            ]
            for pat in all_pats:
                if not pat or not isinstance(pat, str):
                    continue
                p_clean = re.sub(r"[\(\)]", " ", pat).strip().lower()
                if p == "to" and re.search(r"\bto\s+(?:do|be|have)\b", p_clean):
                    continue
                for v_var in variants:
                    if re.search(rf"\b{re.escape(v_var)}\s+(?:[^\n]+\s+)?{obj_pat}\b[^\n]*?\b{re.escape(p)}\b", p_clean):
                        return True
        return False

    @classmethod
    def _get_verb_qualia_scenario_peers(
        cls,
        target_word: str,
        definition: Optional[str] = None,
        quote: Optional[str] = None,
        context_anchor: Optional[str] = None,
        anchor_type: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Pustejovsky Generative Lexicon Qualia Scenario Verb Bridge:
        Extracts concrete scenario/institution/activity nouns from the target verb's definition
        and harvests co-scenario lifecycle verbs from `ocd[noun]['verb_before']` that share the same
        domain scenario while exhibiting contrasting syntactic valency (e.g., `apply for`, `attend [obj]`,
        `graduate from`, `withdraw from` for `enroll in [school/university/course]`)."""
        t = (target_word or "").strip().lower()
        if not t:
            return {}
        ocd = cls.get_oxford_collocations()
        if ocd is None:
            return {}
        def_text = (definition or "").strip()
        if not def_text:
            t_ent = cls.get_ldoce_entry(t) or cls.get_ldoce_entry(cls.lemma_of(t))
            if t_ent and t_ent.get("senses"):
                def_text = str(t_ent["senses"][0].get("definition") or "")
        if not def_text:
            return {}

        _meta_stop_nouns = {
            "something", "somebody", "someone", "person", "people", "thing", "things",
            "way", "time", "part", "place", "kind", "type", "form", "amount", "number",
            "state", "action", "process", "result", "effect", "use", "order", "group",
            "system", "area", "point", "level", "fact", "case", "situation", "problem",
            "information", "work", "life", "world", "day", "year", "man", "woman", "child",
        }
        _delexical_or_generic_verbs = {
            "have", "get", "make", "take", "give", "do", "be", "go", "come", "put", "set",
            "keep", "let", "start", "begin", "stop", "stay", "send", "offer", "provide",
            "run", "follow", "change", "alter", "shape", "decide", "determine", "influence",
            "steer", "pass", "fail", "teach", "hold", "miss", "train", "find", "use", "show",
            "bring", "move", "turn", "call", "try", "ask", "need", "feel", "become", "mean",
        }
        t_variants = set(cls.orthographic_variants(t)) | set(cls.orthographic_variants(cls.lemma_of(t)))
        scenario_nouns: List[str] = []
        for raw_tok in re.findall(r"[a-z]{4,}", def_text.lower()):
            lem_n = cls.lemma_of(raw_tok)
            for cand_n in (raw_tok, lem_n):
                if (
                    cand_n
                    and cand_n not in _meta_stop_nouns
                    and cand_n not in t_variants
                    and cand_n in ocd
                    and cand_n not in scenario_nouns
                ):
                    n_ent = cls.get_ldoce_entry(cand_n)
                    if n_ent and cls._pos_code(n_ent.get("pos") or "") == "n":
                        scenario_nouns.append(cand_n)
        if not scenario_nouns:
            return {}

        anc_prep = (context_anchor or "").strip().lower() if anchor_type == "prep" else ""
        peers: Dict[str, Dict[str, Any]] = {}
        for n in scenario_nouns[:5]:
            vb_list = ocd.get(n, {}).get("verb_before", [])
            for raw_item in vb_list:
                if not isinstance(raw_item, str):
                    continue
                for chunk in [c.strip().lower() for c in raw_item.split("/") if c.strip()]:
                    if chunk.startswith(("something ", "somebody ", "someone ")):
                        continue
                    # 1. Prepositional frame: `V + prep + (det) + N` (e.g. `apply for a course`, `graduate from university`)
                    m_prep = re.match(
                        rf"^([a-z]{{4,}})\s+(for|from|to|with|at|on|in|into|of)\s+(?:a\s+|an\s+|the\s+|\(a\)\s+|\(your\)\s+)?{re.escape(n)}\b",
                        chunk,
                    )
                    if m_prep:
                        vh, vp = m_prep.group(1), m_prep.group(2)
                        if vh in t_variants or vh in _delexical_or_generic_verbs:
                            continue
                        if anc_prep and vp == anc_prep:
                            # Same preposition on the same scenario noun would risk a double-key collision
                            continue
                        info = peers.setdefault(
                            vh,
                            {"score": 0, "nouns": set(), "preps": set(), "transitive": False, "contrasting_prep": False},
                        )
                        if (n, "prep", vp) not in info["nouns"]:
                            info["nouns"].add((n, "prep", vp))
                            info["preps"].add(vp)
                            info["contrasting_prep"] = True
                            info["score"] += 4
                        continue

                    # 2. Direct transitive frame: `V + (det) + N` (e.g. `attend (a) school`, `attend a course`, `leave school`)
                    m_trans = re.match(
                        rf"^([a-z]{{4,}})\s+(?:a\s+|an\s+|the\s+|\(a\)\s+|\(your\)\s+)?{re.escape(n)}\b",
                        chunk,
                    )
                    if m_trans:
                        vh = m_trans.group(1)
                        if vh in t_variants or vh in _delexical_or_generic_verbs:
                            continue
                        if anc_prep and cls._verb_has_intransitive_prep_pattern(vh, anc_prep):
                            continue
                        info = peers.setdefault(
                            vh,
                            {"score": 0, "nouns": set(), "preps": set(), "transitive": False, "contrasting_prep": False},
                        )
                        if (n, "trans", "") not in info["nouns"]:
                            info["nouns"].add((n, "trans", ""))
                            info["transitive"] = True
                            info["score"] += 3
        return peers

    @classmethod
    def _is_attested_verb(cls, word: str) -> bool:
        """Determines whether `word` is attested as a genuine verb in WordNet or LDOCE."""
        w = (word or "").strip().lower()
        if not w:
            return False
        wn = cls.get_wordnet()
        if wn.words(w, pos="v"):
            return True
        w_lem = cls.lemma_of(w)
        if w_lem != w and wn.words(w_lem, pos="v"):
            return True
        entry = cls.get_ldoce_entry(w) or cls.get_ldoce_entry(w_lem)
        if entry and entry.get("pos") in ("verb", "v"):
            return True
        return False

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

    @classmethod
    def evaluate_syntactic_slot_clash(
        cls,
        target: str,
        candidate: str,
        anchor: Optional[str] = None,
        anchor_type: Optional[str] = None,
        pos: Optional[str] = None,
        quote: Optional[str] = None
    ) -> Tuple[bool, str]:
        """Determines whether `candidate` is grammatically barred from occupying `target`'s syntactic slot in `quote`/`anchor_type`.
        
        Returns `(has_clash, reason)` where `has_clash == True` guarantees that `candidate` is syntactically ruled out
        (making it a high-discrimination single-fit distractor and immune to synonym double-key collisions).
        """
        t = (target or "").strip().lower()
        c = (candidate or "").strip().lower()
        if not t or not c or t == c:
            return False, ""

        c_prof = cls.get_word_gram_profile(c, pos=pos)
        t_prof = cls.get_word_gram_profile(t, pos=pos)
        # Also check if `t`'s own THESAURUS box explicitly constrains `c` under the shared synonym sense
        # (e.g. `extra` THESAURUS box labels `spare [only before noun]`, `complete` THESAURUS box labels `done [not before noun]`)
        t_entry = cls.get_ldoce_entry(t)
        if t_entry:
            t_self_countable = any(
                (th.get("word") or "").strip().lower() == t
                and "countable" in str(th.get("grammar") or "").lower()
                and "uncountable" not in str(th.get("grammar") or "").lower()
                for th in (t_entry.get("thesaurus") or [])
            )
            for th in t_entry.get("thesaurus") or []:
                if (th.get("word") or "").strip().lower() == c and th.get("grammar"):
                    th_g = str(th["grammar"]).lower()
                    if "only before noun" in th_g:
                        c_prof["only_before_noun"] = True
                    if "not before noun" in th_g or "never before noun" in th_g:
                        c_prof["not_before_noun"] = True
                    if "uncountable" in th_g and not re.search(r"\bcountable\b", th_g):
                        c_prof["pure_uncountable"] = True
                        if t_self_countable:
                            t_prof["pure_countable"] = True
        pos_norm = (pos or "").strip().lower()
        wn_pos = "a" if pos_norm.startswith("a") or pos_norm == "j" else (
            "n" if pos_norm.startswith("n") else (
                "v" if pos_norm.startswith("v") else ""
            )
        )

        # Inspect quote via spaCy if provided
        is_attr_adj_slot = anchor_type == "modified_noun"
        is_pred_adj_slot = bool(wn_pos == "a" and (anchor_type in ("prep", "to_inf", "that_clause") or (anchor and not is_attr_adj_slot)))
        has_count_det_or_plural = False
        is_passive_verb_slot = False
        has_direct_obj_slot = anchor_type in ("object", "sb_to_do")
        active_verb_frame = anchor_type if anchor_type in ("to_inf", "gerund", "that_clause", "sb_to_do") else None
        extracted_gov_prep: Optional[str] = None
        is_to_inf_slot = anchor_type in ("to_inf", "shell_to_do")
        is_to_prep_slot = bool(anchor_type == "prep" and (anchor or "").strip().lower() == "to")

        if not is_to_inf_slot and not is_to_prep_slot and anchor:
            anc_str = anchor.strip().lower()
            if anc_str.startswith("to "):
                anc_doc = cls.get_spacy()(anc_str)
                if len(anc_doc) >= 2:
                    anc_nxt = anc_doc[1]
                    if anc_nxt.pos_ == "ADV" and len(anc_doc) >= 3:
                        anc_nxt = anc_doc[2]
                    if anc_nxt.tag_ == "VBG" or anc_nxt.pos_ in ("NOUN", "PROPN", "PRON", "DET") or anc_nxt.text.lower() in ("the", "a", "an", "his", "her", "their", "our", "its", "this", "that", "these", "those"):
                        is_to_prep_slot = True
                    elif anc_nxt.tag_ == "VB" or (anc_nxt.pos_ in ("VERB", "AUX") and anc_nxt.tag_ not in ("VBG", "VBN", "VBD", "VBZ")):
                        if cls._is_attested_verb(anc_nxt.lemma_) or cls._is_attested_verb(anc_nxt.text):
                            is_to_inf_slot = True
                        else:
                            is_to_prep_slot = True
                    elif not cls._is_attested_verb(anc_nxt.lemma_) and not cls._is_attested_verb(anc_nxt.text):
                        is_to_prep_slot = True

        if quote and anchor and anchor_type in ("prep", "to_inf", "that_clause", "sb_to_do"):
            if not re.search(rf"\b{re.escape(anchor.lower())}\b", quote.lower()):
                quote = None

        if quote:
            # If quote is a cloze stem with a blank (e.g. '____'), restore target word so spaCy dependency tree parses authentically
            parse_quote = re.sub(r'_{2,}', t, quote) if re.search(r'_{2,}', quote) else quote
            nlp = cls.get_spacy()
            q_doc = nlp(parse_quote)
            for tok in q_doc:
                if cls.form_matches(tok.lemma_.lower(), t) or cls.form_matches(tok.text.lower(), t) or re.match(r"^_{2,}$", tok.text):
                    # Check for to + inf or to + gerund/noun
                    if tok.i + 1 < len(q_doc) and q_doc[tok.i + 1].text.lower() == "to":
                        if tok.i + 2 < len(q_doc):
                            nxt = q_doc[tok.i + 2]
                            if nxt.pos_ == "ADV" and tok.i + 3 < len(q_doc):
                                nxt = q_doc[tok.i + 3]
                            if nxt.tag_ == "VBG" or nxt.pos_ in ("NOUN", "PROPN", "PRON", "DET") or nxt.text.lower() in ("the", "a", "an", "his", "her", "their", "our", "its", "this", "that", "these", "those"):
                                is_to_prep_slot = True
                            elif nxt.tag_ == "VB" or (nxt.pos_ in ("VERB", "AUX") and nxt.tag_ not in ("VBG", "VBN", "VBD", "VBZ")):
                                if cls._is_attested_verb(nxt.lemma_) or cls._is_attested_verb(nxt.text):
                                    is_to_inf_slot = True
                                else:
                                    is_to_prep_slot = True
                            elif not cls._is_attested_verb(nxt.lemma_) and not cls._is_attested_verb(nxt.text):
                                is_to_prep_slot = True
                    # Check for directly right-adjacent preposition or prep child
                    if tok.i + 1 < len(q_doc):
                        next_tok = q_doc[tok.i + 1]
                        if next_tok.text.lower() in cls._STANDARD_PREPOSITIONS and next_tok.pos_ in ("ADP", "SCONJ"):
                            extracted_gov_prep = next_tok.text.lower()
                    if not extracted_gov_prep:
                        for ch in tok.children:
                            if ch.dep_ == "prep" and ch.text.lower() in cls._STANDARD_PREPOSITIONS and ch.i > tok.i:
                                extracted_gov_prep = ch.text.lower()
                                break
                    if wn_pos == "a" or tok.pos_ == "ADJ":
                        if tok.dep_ in ("amod", "compound", "nmod") or (tok.head and tok.head.pos_ in ("NOUN", "PROPN") and tok.i < tok.head.i):
                            is_attr_adj_slot = True
                        elif tok.dep_ in ("acomp", "oprd", "attr", "advcl") or (
                            tok.dep_ == "ROOT" and any(ch.dep_ in ("cop", "aux", "nsubj", "nsubjpass") for ch in tok.children)
                        ):
                            is_pred_adj_slot = True
                    elif wn_pos == "n" or tok.pos_ == "NOUN":
                        if tok.tag_ == "NNS" or any(
                            ch.dep_ in ("det", "nummod")
                            and ch.lemma_.lower() in ("a", "an", "one", "two", "three", "many", "few", "several", "each", "every", "another", "both")
                            for ch in tok.children
                        ):
                            has_count_det_or_plural = True
                    elif wn_pos == "v" or tok.pos_ == "VERB":
                        if any(ch.dep_ in ("auxpass", "aux:pass", "nsubjpass", "nsubj:pass") for ch in tok.children):
                            is_passive_verb_slot = True
                        if any(ch.dep_ in ("dobj", "obj") for ch in tok.children):
                            # When the quiz slot is an intransitive prepositional frame (`... ____ <prep>`)
                            # derived from an ambitransitive verb whose original passage quote had an object
                            # (e.g., passage: "Sue enrolled her kids in a nearby public school." vs quiz slot: "... ____ in"),
                            # do not mark the slot as having a direct object.
                            if not (
                                anchor_type == "prep"
                                and anchor
                                and not re.search(r"_{2,}", quote or "")
                                and cls._verb_has_intransitive_prep_pattern(t, anchor.strip().lower())
                            ):
                                has_direct_obj_slot = True
                        if not active_verb_frame:
                            if any(ch.dep_ in ("dobj", "obj") for ch in tok.children) and any(
                                ch.dep_ == "xcomp" and any(gc.lemma_.lower() == "to" for gc in ch.children)
                                for ch in tok.children
                            ):
                                active_verb_frame = "sb_to_do"
                            elif any(
                                ch.dep_ in ("ccomp", "xcomp")
                                and any(gc.dep_ in ("nsubj", "dobj", "obj") for gc in ch.children)
                                and any(gc.lemma_.lower() == "to" for gc in ch.children)
                                for ch in tok.children
                            ):
                                active_verb_frame = "sb_to_do"
                    break

        # 1. Adjective positional slot clash ([only before noun] vs predicative; [not before noun] vs attributive)
        if wn_pos == "a" or (not wn_pos and (c_prof["only_before_noun"] or c_prof["not_before_noun"])):
            if is_pred_adj_slot and c_prof["only_before_noun"] and not t_prof["only_before_noun"]:
                return True, "adj_only_before_noun_in_predicative_slot"
            if is_attr_adj_slot and c_prof["not_before_noun"] and not t_prof["not_before_noun"]:
                return True, "adj_not_before_noun_in_attributive_slot"

        # 2. Noun countability clash (pure [uncountable] in countable determiner/plural slot, or pure [countable] singular in uncountable slot)
        if (wn_pos == "n" or not wn_pos) and (
            has_count_det_or_plural
            or (t.endswith("s") and not t.endswith("ics") and not t_prof.get("pure_uncountable"))
        ):
            if c_prof["pure_uncountable"] and not t_prof["pure_uncountable"]:
                return True, "uncountable_noun_in_countable_slot"
        if (
            (wn_pos == "n" or not wn_pos)
            and t_prof.get("pure_uncountable")
            and not has_count_det_or_plural
            and c_prof.get("pure_countable")
            and not c_prof.get("pure_uncountable")
            and not c.endswith("s")
        ):
            return True, "countable_singular_noun_in_uncountable_slot"

        # 3. Verb transitivity / passive / forbidden complement frame clash
        if wn_pos == "v" or not wn_pos:
            if is_passive_verb_slot and (c_prof["no_passive"] or c_prof["pure_intransitive"]):
                return True, "no_passive_or_intransitive_in_passive_slot"
            if has_direct_obj_slot and c_prof["pure_intransitive"] and not t_prof["pure_intransitive"]:
                return True, "intransitive_verb_in_transitive_object_slot"
            if active_verb_frame and (
                active_verb_frame in c_prof["forbidden_frames"]
                or (active_verb_frame == "sb_to_do" and "V_sb_to_do" in c_prof["forbidden_frames"])
                or (active_verb_frame == "to_inf" and "V_to_do" in c_prof["forbidden_frames"])
            ):
                return True, f"forbidden_frame_{active_verb_frame}"

        # 4. Aspectual Duration / Modifier Clash (e.g. quick/brief vs stare/gaze; prolonged/steady vs glance)
        _INSTANTANEOUS_MODS = {"quick", "brief", "fleeting", "hasty", "momentary", "swift", "passing", "instant", "rapid"}
        _DURATIVE_MODS = {"prolonged", "steady", "fixed", "continuous", "sustained", "unwavering", "unblinking", "endless", "long"}

        is_instant_slot = bool(anchor and anchor.lower() in _INSTANTANEOUS_MODS)
        is_durative_slot = bool(anchor and anchor.lower() in _DURATIVE_MODS)

        if quote and not (is_instant_slot or is_durative_slot):
            nlp = cls.get_spacy()
            q_doc = nlp(quote)
            for tok in q_doc:
                if cls.form_matches(tok.lemma_.lower(), t) or cls.form_matches(tok.text.lower(), t):
                    for ch in tok.children:
                        ch_lem = ch.lemma_.lower()
                        if ch_lem in _INSTANTANEOUS_MODS:
                            is_instant_slot = True
                        elif ch_lem in _DURATIVE_MODS:
                            is_durative_slot = True
                    if tok.head and not cls.form_matches(tok.head.lemma_.lower(), t):
                        head_lem = tok.head.lemma_.lower()
                        if head_lem in _INSTANTANEOUS_MODS:
                            is_instant_slot = True
                        elif head_lem in _DURATIVE_MODS:
                            is_durative_slot = True
                    break

        c_ent = cls.get_ldoce_entry(c) or {}
        c_senses = c_ent.get("senses", [])
        c_defs = " ".join(str(s.get("definition", "")).lower() for s in c_senses)

        if is_instant_slot:
            has_durative_def = any(
                cue in c_defs
                for cue in [
                    "for a long time", "fixedly", "steadily", "without moving your eyes",
                    "long-lasting", "giving it all your attention"
                ]
            )
            has_quick_def = any(
                cue in c_defs
                for cue in ["quick", "quickly", "a short time", "brief", "sudden", "suddenly", "for a moment"]
            )
            if has_durative_def and not has_quick_def:
                return True, "durative_candidate_in_instantaneous_aspect_slot"

        if is_durative_slot:
            has_quick_def = any(
                cue in c_defs
                for cue in ["quick", "quickly", "a short time", "brief", "sudden", "suddenly", "for a moment"]
            )
            has_durative_def = any(
                cue in c_defs
                for cue in [
                    "for a long time", "fixedly", "steadily", "without moving your eyes",
                    "long-lasting"
                ]
            )
            if has_quick_def and not has_durative_def:
                return True, "instantaneous_candidate_in_durative_aspect_slot"

        # 5. Bound Degree / Intensifier Collocation Clash (e.g. sound asleep vs sound *sleepy/*awake; wide awake vs wide *asleep)
        _BOUND_INTENSIFIERS = {"sound", "wide", "fast", "bitterly", "highly", "gravely", "utterly", "dead"}
        norm_anc = (anchor or "").strip().lower()
        if not norm_anc and quote:
            q_low = quote.lower()
            for bi in _BOUND_INTENSIFIERS:
                if re.search(rf"\b{re.escape(bi)}\s+{re.escape(t)}\b", q_low):
                    norm_anc = bi
                    break

        if norm_anc in _BOUND_INTENSIFIERS:
            t_bound = cls.lookup_ldoce_phrase(f"{norm_anc} {t}")
            c_bound = cls.lookup_ldoce_phrase(f"{norm_anc} {c}")
            if t_bound and not c_bound:
                return True, "unattested_bound_degree_collocation_clash"

        # 6. Bound Prepositional Government Clash (e.g. insight into vs overview of / description of; allergic to vs aware of)
        gov_prep = None
        if anchor_type == "prep" and anchor:
            clean_anc_p = anchor.strip().lower()
            if clean_anc_p in cls._STANDARD_PREPOSITIONS:
                gov_prep = clean_anc_p
        if not gov_prep and extracted_gov_prep:
            gov_prep = extracted_gov_prep

        if gov_prep and (wn_pos in ("n", "a") or pos_norm in ("n", "noun", "a", "adj", "adjective")):
            # Quirk/Biber Invariant: 'of' postmodification is universally permissible for English nouns
            # (genitive/partitive/appositive; e.g. 'increase of 10%', 'destruction of the city');
            # lack of an explicit LDOCE [+of] code on a noun never constitutes a prepositional government clash.
            if gov_prep == "of" and (wn_pos == "n" or pos_norm in ("n", "noun")):
                pass
            else:
                c_preps = cls.get_governed_prepositions(c, pos=pos)
                if c_preps and gov_prep not in c_preps:
                    if not cls.lookup_ldoce_phrase(f"{c} {gov_prep}"):
                        return True, "unattested_prepositional_government_clash"
        elif (
            gov_prep
            and (wn_pos == "v" or pos_norm in ("v", "verb"))
            and anchor_type == "prep"
            and not has_direct_obj_slot
            and cls._verb_has_intransitive_prep_pattern(t, gov_prep)
        ):
            c_preps = cls.get_governed_prepositions(c, pos="verb")
            if gov_prep not in c_preps and not cls.lookup_ldoce_phrase(f"{c} {gov_prep}") and not cls._verb_has_intransitive_prep_pattern(c, gov_prep):
                qualia_peers = cls._get_verb_qualia_scenario_peers(
                    t,
                    quote=quote,
                    context_anchor=gov_prep,
                    anchor_type="prep",
                )
                if c_preps or (
                    c in qualia_peers and (
                        qualia_peers[c].get("contrasting_prep") or qualia_peers[c].get("transitive")
                    )
                ):
                    return True, "unattested_prepositional_government_clash"

        # 7. Subcategorization Frame Clash: to + infinitive vs to + gerund/preposition
        if is_to_inf_slot or is_to_prep_slot:
            c_inf, c_prep = cls.get_word_to_complement_profile(c, pos=pos)
            if is_to_inf_slot and c_prep and not c_inf:
                return True, "incompatible_to_infinitive_subcat_clash"
            if is_to_prep_slot and c_inf and not c_prep:
                return True, "incompatible_to_gerund_subcat_clash"

        # 8. Fronted Negative Inversion Slot & Correlative Clash
        # In a fronted inversion slot (Aux + Subj inversion), candidates must be valid negative/restrictive
        # inversion triggers. Non-inverting adverbs (*often, *steadily, *completely) are hard-clashing ungrammatical giveaways.
        # Furthermore, correlative pairs must match:
        # - 'no sooner' requires 'than' (clashes with 'when' / 'before')
        # - 'hardly', 'scarcely', 'barely' require 'when' / 'before' (clash with 'than')
        # - 'seldom', 'rarely', 'little' are aspectually incompatible with point-in-time 'when' / 'before'
        is_fronted_inv_slot = cls.is_fronted_inversion_context(
            quote=quote, anchor=anchor, anchor_type=anchor_type, target=t
        )
        if is_fronted_inv_slot:
            if c not in cls._ALL_NEGATIVE_INVERSION_TRIGGERS:
                return True, "non_inverting_candidate_in_fronted_inversion_slot"
            if quote:
                q_has_than = bool(re.search(r"\bthan\b", quote, re.IGNORECASE))
                q_has_when_before = bool(re.search(r"\b(?:when|before)\b", quote, re.IGNORECASE))
                if q_has_than:
                    if c in ("hardly", "scarcely", "barely", "seldom", "rarely", "little"):
                        return True, "correlative_than_mismatch_in_inversion_slot"
                if q_has_when_before:
                    if c == "no sooner":
                        return True, "correlative_when_mismatch_in_inversion_slot"
                    if c in ("seldom", "rarely", "little"):
                        return True, "aspectual_when_mismatch_in_inversion_slot"

        return False, ""

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

        t_preps_raw = cls.get_oxford_collocations(t, pos=pos).get("prep", []) + cls.get_oxford_collocations(t_lem, pos=pos).get("prep", [])
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
            d_preps_raw = cls.get_oxford_collocations(d, pos=pos or "noun").get("prep", []) + cls.get_oxford_collocations(d_lem, pos=pos or "noun").get("prep", [])
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

                d_entry = cls.get_oxford_collocations(d, pos=pos or "noun")
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
                        a_entry = cls.get_oxford_collocations(a_clean, pos="noun")
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
                        a_entry = cls.get_oxford_collocations(a_clean, pos="noun")
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
                d_entry = cls.get_oxford_collocations(d, pos="verb")
                if a_clean in [n.split()[0].lower() for n in d_entry.get("colloc_nouns", [])]:
                    return True, "double_key_collocation"
                a_entry = cls.get_oxford_collocations(a_clean, pos="noun")
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
                a_entry = cls.get_oxford_collocations(a_clean, pos="noun")
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
                d_entry = cls.get_oxford_collocations(d, pos="noun")
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

    _PREDICATIVE_ONLY_ADJS = {
        "asleep", "afraid", "alive", "alone", "ashamed", "aware", "awake", "content", "unable", "prone", "glad"
    }

    _ADJ_PREP_VALENCY_MAP = {
        "to": ["vulnerable", "immune", "conducive", "susceptible", "allergic", "attentive", "adjacent", "subordinate", "accessible"],
        "of": ["aware", "capable", "characteristic", "deprived", "indicative", "reminiscent", "composed", "devoid", "conscious"],
        "for": ["eligible", "notorious", "responsible", "suitable", "renowned", "adequate", "prepared", "qualified"],
        "with": ["consistent", "compatible", "acquainted", "associated", "compliant", "congruent", "comparable"],
        "from": ["distinct", "exempt", "absent", "derived", "detached", "remote"],
        "in": ["proficient", "inherent", "rich", "experienced", "versed", "lacking"],
        "on": ["reliant", "dependent", "contingent", "intent", "keen"]
    }

    _ACADEMIC_ADJ_FAMILIES = [
        ["comprehensive", "extensive", "thorough", "limited", "exhaustive", "restricted", "fragmented"],
        ["available", "accessible", "obtainable", "unavailable", "inaccessible", "restricted", "exclusive"],
        ["stable", "volatile", "constant", "variable", "consistent", "fluctuating", "fragile"],
        ["crucial", "fundamental", "marginal", "negligible", "vital", "trivial", "secondary"],
        ["vulnerable", "susceptible", "immune", "resistant", "resilient", "defenseless"],
        ["feasible", "viable", "impractical", "attainable", "unrealistic", "plausible"],
        ["explicit", "ambiguous", "apparent", "obscure", "evident", "vague", "distinct"],
        ["substantial", "considerable", "minimal", "negligible", "significant", "modest"],
        ["coherent", "consistent", "incompatible", "contradictory", "harmonious", "conflicting"],
    ]

    # Academic Register Gate: Filter out slang, informal, archaic, or fantasy/fairy-tale words
    # that WordNet taxonomy occasionally harbors (e.g. 'dwarf', 'beast' for 'individual', or 'swell', 'cracking' for 'positive').
    _REGISTER_BANNED_WORDS = {
        # Archaic / fantasy / fairy-tale nouns
        "dwarf", "beast", "fauna", "flora", "benthos", "heterotroph", "amphidiploid", "diploid",
        "ogre", "giant", "goblin", "witch", "wizard", "fairy", "elf", "gnome", "troll",
        # Colloquial slang / archaic positive adjectives
        "swell", "cracking", "bully", "smashing", "peachy", "dandy", "nifty", "corking", "groovy", "slap-up", "bang-up",
        # Taboo / Inappropriate words for educational assessment
        "sex", "sexy", "sexual", "erotic", "porn", "pornography", "nude", "naked", "suicide", "murder"
    }

    _ANCHOR_STOPWORDS = {
        "a", "an", "the", "of", "to", "in", "on", "for", "with", "and", "or", "at",
        "by", "from", "into", "onto", "out", "off", "up", "down", "as", "be", "is",
        "are", "was", "were", "been", "being", "have", "has", "had", "do", "does",
        "did", "will", "would", "shall", "should", "can", "could", "may", "might",
        "must", "that", "this", "these", "those", "it", "its", "he", "she", "we",
        "they", "you", "i", "me", "my", "your", "his", "her", "their", "our",
        "each", "every", "some", "any", "all", "both", "few", "sth", "sb",
        # Degree adverbs, quantifiers and determiners can never serve as
        # collocational anchors (they carry no lexical collocation signal;
        # e.g. 'more' in 'more hours' must not be harvested for 'autonomy').
        "more", "most", "less", "least", "many", "much", "very", "quite",
        "rather", "such", "own", "other", "another", "same", "different",
        "particular", "certain", "various", "several", "enough", "only",
        "just", "even", "also", "well", "still", "yet", "already",
        # Empty / generic head nouns carry virtually zero lexical collocational value
        # (e.g. 'thing', 'stuff', 'item' in 'positive thing' or 'important matter').
        # Banning them prevents trivial or unacademic frames like 'positive thing'.
        "thing", "things", "stuff", "person", "people", "way", "ways", "item", "items",
        # Indefinite pronouns and quantifier nouns are placeholders, not collocations:
        # 'offer someone something' and 'lots of goodies' say nothing about how the
        # target actually collocates, so they must never become anchors.
        "something", "anything", "everything", "nothing", "someone", "somebody",
        "anyone", "anybody", "everyone", "everybody", "nobody", "one", "ones",
        "lot", "lots", "bit", "bits", "kind", "kinds", "sort", "sorts",
        "piece", "pieces", "pair", "pairs", "number", "amount", "rest"
    }

    # Fixed prepositional adjuncts / set phrases that are interjections or
    # discourse markers, NEVER valency complements of a governing headword.
    # When a (prep, pobj) dependency pair hits this blacklist it is skipped,
    # so 'correlate for example' can no longer be mistaken for a frame.
    _ADJUNCT_PREP_PAIRS = {
        ("for", "example"), ("for", "instance"), ("in", "fact"), ("in", "general"),
        ("at", "first"), ("on", "average"), ("in", "particular"), ("in", "summary"),
        ("in", "total"), ("in", "brief"), ("in", "short"), ("in", "part")
    }

    # When a verb has NO Oxford Collocations entry (or an empty 'prep' valency
    # list), only these typical academic complement prepositions may be
    # trusted from syntactic guessing; anything else degrades to anchor=None
    # (neutral template) instead of fabricating a frame (e.g. 'degrade in a
    # society' from a fronted adjunct).
    _TYPICAL_COMPLEMENT_PREPS = {"into", "to", "from"}

    _LV_TRIAD_NOUN_CONTRASTS: Dict[str, Tuple[List[str], Set[str]]] = {
        "attention": (["impact", "role", "advantage", "use", "place", "charge"], {"attention", "heed", "tribute", "homage", "respect"}),
        "impact": (["attention", "role", "advantage", "use", "place", "charge"], {"impact", "effect", "influence", "bearing"}),
        "effect": (["attention", "role", "advantage", "use", "place", "charge"], {"impact", "effect", "influence", "bearing"}),
        "role": (["attention", "impact", "advantage", "use", "charge", "care"], {"role", "part"}),
        "part": (["attention", "impact", "advantage", "use", "charge", "care"], {"role", "part"}),
        "advantage": (["attention", "impact", "role", "place", "effort", "decision"], {"advantage", "use", "care", "charge", "notice", "account"}),
        "use": (["attention", "impact", "role", "place", "effort", "decision"], {"advantage", "use", "sense", "fun", "room", "way"}),
    }

    _LIGHT_VERB_SLOT_PEERS: Dict[str, List[str]] = {
        "cast": ["throw", "shed", "put", "drop", "lay", "send"],
        "shed": ["cast", "throw", "drop", "shine", "pour", "spread"],
        "strike": ["hit", "beat", "keep", "reach", "make", "hold", "catch"],
        "pay": ["give", "offer", "bring", "make", "render", "present"],
        "pose": ["present", "cause", "create", "raise", "set", "make"],
        "wreak": ["cause", "bring", "make", "create", "play"],
        "bear": ["have", "hold", "keep", "carry", "show", "bring"],
        "exert": ["put", "apply", "impose", "exercise", "bring"],
        "draw": ["make", "reach", "pull", "take", "write"],
        "run": ["take", "bear", "have", "face", "meet"],
        "take": ["make", "have", "get", "give", "bring", "hold"],
        "make": ["take", "do", "have", "give", "create", "bring"],
        "give": ["make", "bring", "provide", "offer", "take", "grant"],
        "catch": ["see", "get", "take", "find", "hold"],
        "keep": ["hold", "take", "make", "have", "save"],
        "lose": ["miss", "drop", "waste", "forget"],
        "put": ["set", "place", "lay", "bring", "make"],
        "set": ["put", "place", "lay", "make", "give"],
        "lay": ["put", "set", "place", "make", "give"],
        "play": ["act", "take", "make", "perform", "do"],
        "hold": ["keep", "take", "have", "bear", "carry"],
        "bring": ["take", "give", "make", "carry", "put"],
    }

    _STANDARD_COLLOC_PREPS: Set[str] = {
        "on", "in", "to", "at", "for", "from", "with", "between",
        "over", "against", "into", "upon", "under", "about", "beyond",
        "through", "toward", "towards",
    }

    @classmethod
    def is_attested_light_verb_collocation(
        cls,
        verb: str,
        noun: str,
        prep: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """Validates whether `verb` forms an attested idiom or established collocation with `noun` (and `prep`).
        
        Returns `(is_attested, reason)` where:
        - `is_attested == True` indicates that `verb` is co-attested with `noun` in standard English (LDOCE),
          meaning that using `verb` as a distractor for a light-verb collocation with `noun` would create a DOUBLE KEY.
        - `is_attested == False` guarantees that `verb` is strictly unattested with `noun`, making it a
          high-discrimination, single-fit safe foil.
        """
        v_clean = (verb or "").strip().lower()
        n_clean = (noun or "").strip().lower()
        if not v_clean or not n_clean:
            return False, ""
        v_base = cls.lemma_of(v_clean).lower() if " " not in v_clean else v_clean
        n_base = cls.lemma_of(n_clean).lower() if " " not in n_clean else n_clean
        p_base = (prep or "").strip().lower()

        # 1. lookup_ldoce_phrase (gold standard curated idioms and multi-word phrases)
        phrs = [
            f"{v_base} {n_base}",
            f"{v_base} a {n_base}",
            f"{v_base} an {n_base}",
            f"{v_base} the {n_base}",
        ]
        if p_base:
            phrs.extend([
                f"{v_base} {n_base} {p_base}",
                f"{v_base} a {n_base} {p_base}",
                f"{v_base} an {n_base} {p_base}",
                f"{v_base} the {n_base} {p_base}",
            ])
        for p in phrs:
            if cls.lookup_ldoce_phrase(p):
                return True, f"phrase_index:{p}"

        # 2. LDOCE noun entry curated collocations (source != 'corpus')
        n_ent = cls.get_ldoce_entry(n_base)
        if n_ent:
            colls = n_ent.get("collocations") or {}
            if isinstance(colls, dict):
                candidates = []
                for v_item in colls.get("verbs", []):
                    if v_item.get("pos") and cls._pos_code(v_item["pos"]) != "n":
                        continue
                    if v_item.get("source") != "corpus":
                        candidates.append((v_item.get("collocation") or "").lower())
                for p_item in colls.get("phrases", []):
                    if p_item.get("pos") and cls._pos_code(p_item["pos"]) != "n":
                        continue
                    if p_item.get("source") != "corpus":
                        candidates.append((p_item.get("collocation") or "").lower())

                for c_text in candidates:
                    if not re.search(rf"\b{re.escape(v_base)}\b", c_text):
                        continue

                    # If n_base is immediately preceded by certain prepositions (e.g. 'into doubt', 'beyond doubt'),
                    # the verb does not take n_base as direct object
                    m_prep_before = re.search(
                        rf"\b([a-z]+)\s+(?:a\s+|an\s+|the\s+)?{re.escape(n_base)}\b", c_text
                    )
                    if m_prep_before:
                        pb = m_prep_before.group(1).lower()
                        if pb in ("into", "beyond", "without", "out of", "under"):
                            continue

                    # Check governed preposition AFTER noun
                    m_prep_after = re.search(rf"\b{re.escape(n_base)}(?:s)?\s+([a-z]+)", c_text)
                    gov_p = (
                        m_prep_after.group(1).lower()
                        if m_prep_after and m_prep_after.group(1).lower() in cls._STANDARD_COLLOC_PREPS
                        else None
                    )
                    if gov_p and p_base and gov_p != p_base:
                        # Collocation specifies a specific post-noun preposition (e.g. 'on') that clashes with p_base
                        continue

                    return True, f"curated_colloc:{c_text}"

        return False, "unattested"

    @classmethod
    def detect_light_verb_collocation_context(
        cls,
        quote: Optional[str] = None,
        anchor: Optional[str] = None,
        anchor_type: Optional[str] = None,
        target: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Detects whether `target` (or the blank in `quote`) represents the verb slot of a light-verb collocation.
        
        Returns a dict `{'verb': base_verb, 'noun': noun_lemma, 'prep': prep_or_none, 'is_light_verb': True}`
        if verified as an attested light-verb collocation, else `None`."""
        t_clean = (target or "").strip().lower()
        t_lem = cls.lemma_of(t_clean) if t_clean else ""
        base_v = None
        if t_clean in cls._LIGHT_VERB_SLOT_PEERS:
            base_v = t_clean
        elif t_lem in cls._LIGHT_VERB_SLOT_PEERS:
            base_v = t_lem

        if not base_v:
            return None

        noun: Optional[str] = None
        prep: Optional[str] = None

        if quote:
            parse_quote = re.sub(r'_{2,}', t_clean or base_v, quote)
            try:
                doc = cls.get_spacy()(parse_quote)
                for tok in doc:
                    if cls.form_matches(tok.text.lower(), t_clean) or cls.form_matches(tok.lemma_.lower(), base_v):
                        objs = [ch for ch in tok.children if ch.dep_ in ("dobj", "obj", "nsubjpass", "nsubj:pass")]
                        if objs:
                            d = objs[0]
                            noun = d.lemma_.lower()
                            preps = [ch for ch in d.children if ch.dep_ in ("prep", "dative")]
                            if not preps:
                                preps = [ch for ch in tok.children if ch.dep_ in ("prep", "dative") and ch.i > d.i]
                            if preps:
                                prep = preps[0].text.lower()
                            break
            except Exception:
                pass

        if not noun and quote:
            v_pat = rf"(?:{re.escape(t_clean)}|{re.escape(base_v)}|_{{2,}})"
            m = re.search(
                rf"\b{v_pat}\s+(?:(?:a|an|the|his|her|their|our|its|my|your|no|any|some|much|more|little|less)\s+)?(?:[a-z]+\s+)?([a-z]+)(?:\s+([a-z]+))?",
                quote,
                re.IGNORECASE,
            )
            if m:
                cand_n = m.group(1).lower()
                cand_p = (m.group(2) or "").lower()
                if cand_p in cls._STANDARD_COLLOC_PREPS:
                    noun = cls.lemma_of(cand_n)
                    prep = cand_p
                else:
                    noun = cls.lemma_of(cand_n)
                    prep = None

        if not noun and anchor and anchor_type in ("lv_triad", "colloc", "noun"):
            noun = cls.lemma_of(anchor.strip().lower())

        if noun:
            is_attested, _ = cls.is_attested_light_verb_collocation(verb=base_v, noun=noun, prep=prep)
            if is_attested:
                return {"verb": base_v, "noun": noun, "prep": prep, "is_light_verb": True}

        return None
