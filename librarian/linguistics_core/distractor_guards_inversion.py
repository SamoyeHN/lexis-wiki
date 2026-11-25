"""Fronted negative inversion and light-verb collocation context guards."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

from .linguistic_rules_loader import (
    get_all_negative_inversion_triggers,
    get_negative_inversion_adverbs,
    get_negative_inversion_phrases,
)


class DistractorGuardsInversionMixin:
    """Guards for fronted negative/restrictive inversion slots and light-verb collocations."""

    # Family 7 Fronted Negative / Restrictive Inversion Arena:
    # High-order negative or restrictive triggers that trigger Subject-Auxiliary Inversion (SAI)
    # when fronted at the beginning of a clause/sentence (e.g. 'Hardly had the meeting begun when...').
    _NEGATIVE_INVERSION_ADVERBS: Set[str] = get_negative_inversion_adverbs()
    _NEGATIVE_INVERSION_PHRASES: Set[str] = get_negative_inversion_phrases()
    _ALL_NEGATIVE_INVERSION_TRIGGERS: Set[str] = get_all_negative_inversion_triggers()

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
        "exert": ["deliver", "offer", "yield", "maintain", "implement"],
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



__all__ = ["DistractorGuardsInversionMixin"]
