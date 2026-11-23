"""
Zero-collision collocational anchor mining from OCD and LDOCE pattern bundles/examples.
"""

import datetime
import hashlib
import json
import logging
import os
import re
import sqlite3
import threading
import zlib
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

from .linguistic_rules_loader import (
    get_delexical_quantifier_nouns,
    get_delexical_verbs,
)

logger = logging.getLogger("librarian.linguistics")


class ZeroCollisionAnchorsMixin:
    @classmethod
    def find_ocd_zero_collision_anchor(
        cls,
        target_word: str,
        pos: str = "noun",
        distractors: Optional[List[str]] = None,
        definition: Optional[str] = None,
        quote: Optional[str] = None
    ) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
        """
        [DEPRECATED in favor of PedLex 6th Edition]
        Delegates directly to find_ldoce_zero_collision_anchor to ensure 100% PedLex semantic ordering
        and pedagogical grounding across all callers.
        """
        return cls.find_ldoce_zero_collision_anchor(
            target_word=target_word,
            pos=pos,
            distractors=distractors,
            definition=definition,
            quote=quote
        )


    # Every cascade in find_ldoce_zero_collision_anchor needs a collocation item or a
    # grammar pattern to hang the anchor on. 18,884 PedLex entries have neither - only
    # plain sense examples - so the lookup returned nothing and the item shipped with
    # no structural model at all. Read the partner straight off the dependency tree of
    # the example that illustrates the sense this item teaches.
    _EXAMPLE_ANCHOR_TYPES = {
        ("adj", "NOUN"): ("modified_noun", "modifying noun {}"),
        ("adj", "PROPN"): ("modified_noun", "modifying noun {}"),
        ("noun", "VERB"): ("verb", "direct object of verb {}"),
        ("noun", "AUX"): ("verb", "direct object of verb {}"),
        ("noun", "ADP"): ("prep", "followed by preposition {}"),
        ("adv", "VERB"): ("modifies_verb", "modifying verb {}"),
        ("adv", "AUX"): ("modifies_verb", "modifying verb {}"),
        ("adv", "ADJ"): ("modifies_adj", "modifying adjective {}"),
        ("verb", "NOUN"): ("object", "direct object noun {}"),
        ("verb", "PROPN"): ("object", "direct object noun {}"),
    }

    # Light verbs a noun can pair with almost anything: 'like a chocolate',
    # 'need a penalty'. The pairing is grammatically real but says nothing about how
    # this noun actually collocates, so the sense-example cascade rejects it and keeps
    # searching for a partner with genuine collocational content.
    _DELEXICAL_VERBS = get_delexical_verbs()

    @classmethod
    def _is_delexical_anchor(cls, anchor: Optional[str], anchor_type: Optional[str],
                             pos: str) -> bool:
        """True when a noun target was anchored by an empty light verb."""
        if not anchor or anchor_type not in ("verb", "verb_subject"):
            return False
        p_low = (pos or "").strip().lower()
        if not (p_low.startswith("n") or p_low == "s"):
            return False
        return anchor in cls._DELEXICAL_VERBS

    # B2: 'a lot of', 'sorts of', 'plenty of' look like a frame but carry no collocational
    # content - literally any noun fits them. A partitive compound is only worth shipping
    # when the quantifier noun says something about what may fill the slot, which is what
    # makes 'baskets of goodies' a teachable frame and 'lots of goodies' a fake one.
    _DELEXICAL_QUANTIFIER_NOUNS = get_delexical_quantifier_nouns()

    @classmethod
    def _partitive_compound_anchor(
        cls,
        target_word: str,
        quote: Optional[str],
        forbidden: Optional[Set[str]] = None
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        B2: the frame a noun has inside a partitive compound in the passage itself.

        'various baskets of goodies' gives 'goody' no anchor under any existing priority:
        spaCy hangs the target under 'of' and 'of' under the quantifier noun, so the walker -
        which only ever inspects the target's children or a verbal head - sees nothing, and
        the item shipped with anchor=None while the micro-task still implied a frame. Walk
        the target's own path upward instead and hand back the compound the student reads:
        ('baskets of', 'partitive', 'basket').

        Returns (anchor, anchor_type, quantifier_lemma), or (None, None, None) when the
        target is not sitting in a contentful 'N of ____' frame.
        """
        tw = (target_word or "").strip().lower()
        text = (quote or "").strip()
        if not tw or not text:
            return None, None, None
        nlp = cls.get_spacy()
        if nlp is None:
            return None, None, None
        try:
            doc = nlp(text)
        except Exception:
            return None, None, None
        banned = {w.strip().lower() for w in (forbidden or set()) if w and w.strip()}

        for tok in doc:
            if tok.pos_ not in ("NOUN", "PROPN"):
                continue
            if tok.text.lower() != tw and tok.lemma_.lower() != tw:
                continue
            if tok.dep_ != "pobj":
                continue
            prep = tok.head
            if prep is None or prep.dep_ != "prep" or prep.text.lower() != "of":
                continue
            quant = prep.head
            if quant is None or quant.pos_ not in ("NOUN", "PROPN"):
                continue
            quant_lem = (quant.lemma_ or quant.text).lower()
            quant_surf = quant.text.lower()
            if len(quant_lem) < 3 or quant_lem == tw or quant_lem in cls._ANCHOR_STOPWORDS:
                continue
            if quant_lem in cls._DELEXICAL_QUANTIFIER_NOUNS:
                continue
            if cls.are_same_word_family(quant_lem, tw):
                continue
            if quant_lem in banned or quant_surf in banned:
                continue
            return f"{quant_surf} of", "partitive", quant_lem
        return None, None, None

    @classmethod
    def _example_partner(cls, target_tok, target_word: str, pos: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """The collocational partner of the target token inside one authentic example.
        The relation logic follows the token's own POS, because the example may carry an
        inflected or derived form of the target ('depiction' -> 'depicting')."""
        tw = target_word.strip().lower()
        p_low = (pos or "").lower()
        tok_pos = target_tok.pos_
        p_key = ("adj" if p_low.startswith("adj") else
                 "verb" if tok_pos in ("VERB", "AUX") else
                 "adj" if tok_pos == "ADJ" else
                 "adv" if tok_pos == "ADV" else
                 "noun" if tok_pos in ("NOUN", "PROPN") else
                 "verb" if p_low.startswith("v") else
                 "adv" if p_low.startswith("adv") else "noun")

        def usable(tok) -> Optional[str]:
            lem = (tok.lemma_ or tok.text).lower()
            if len(lem) < 3 or lem == tw or lem in cls._ANCHOR_STOPWORDS:
                return None
            if cls.are_same_word_family(lem, tw):
                return None
            return lem

        head = target_tok.head

        # A verb is anchored by what it acts upon, never by its own head.
        if p_key == "verb":
            for dep in ("dobj", "obj", "attr", "oprd"):
                for child in target_tok.children:
                    if child.dep_ == dep:
                        lem = usable(child)
                        if lem:
                            return lem, "object", f"direct object noun {lem}"
            for child in target_tok.children:
                if child.dep_ in ("prep", "obl", "nmod"):
                    lem = usable(child)
                    if lem:
                        return lem, "object", f"prepositional complement {lem}"
                    if child.dep_ == "prep" and child.text.lower() in (
                        "for", "to", "on", "at", "about", "with", "over", "into", "of", "from", "in"
                    ):
                        p_str = child.text.lower()
                        return p_str, "prep", f"governing preposition {p_str}"

        if p_key == "adv":
            if target_tok.i + 1 < len(target_tok.doc):
                next_adj = target_tok.doc[target_tok.i + 1]
                if next_adj.pos_ == "ADJ" and not next_adj.text[:1].isupper():
                    lem_adj = usable(next_adj)
                    if lem_adj:
                        return lem_adj, "modifies_adj", f"modifying adj {lem_adj}"
            if head is not None and head.pos_ in ("VERB", "ADJ", "ADV"):
                if not (head.pos_ == "ADJ" and head.text[:1].isupper() and head.i > 0):
                    lem = usable(head)
                    if lem:
                        atype = "modifies_adj" if head.pos_ in ("ADJ", "ADV") else "modifies_verb"
                        return lem, atype, f"modifying {head.pos_.lower()} {lem}"
            if target_tok.i + 1 < len(target_tok.doc):
                next_tok = target_tok.doc[target_tok.i + 1]
                if next_tok.pos_ in ("ADJ", "VERB") and not (next_tok.pos_ == "ADJ" and next_tok.text[:1].isupper()):
                    lem = usable(next_tok)
                    if lem:
                        atype = "modifies_adj" if next_tok.pos_ == "ADJ" else "modifies_verb"
                        return lem, atype, f"modifying {next_tok.pos_.lower()} {lem}"
            if target_tok.i > 0:
                prev_tok = target_tok.doc[target_tok.i - 1]
                if prev_tok.pos_ in ("ADJ", "VERB", "ADV") and not (prev_tok.pos_ == "ADJ" and prev_tok.text[:1].isupper() and prev_tok.i > 0):
                    lem = usable(prev_tok)
                    if lem:
                        atype = "modifies_adj" if prev_tok.pos_ in ("ADJ", "ADV") else "modifies_verb"
                        return lem, atype, f"modifying {prev_tok.pos_.lower()} {lem}"
            return None, None, None

        if p_key == "adj":
            # Climb past coordinated adjectives ('cold, formal, and impersonal' -> head of 'cold')
            eff_head = head
            while eff_head is not None and eff_head.dep_ == "conj" and eff_head.head.pos_ == "ADJ" and eff_head.head != eff_head:
                eff_head = eff_head.head
            if eff_head is not None and eff_head.pos_ == "ADJ" and eff_head.head != eff_head:
                eff_head = eff_head.head
            # 1. Attributive noun modification (`amod`, `compound`, or coordinated `conj` modifying a noun)
            if eff_head is not None and eff_head.pos_ in ("NOUN", "PROPN") and target_tok.dep_ in ("amod", "compound", "conj"):
                lem = usable(eff_head)
                if lem:
                    return lem, "modified_noun", f"modifying noun {lem}"
            # 2. Post-adjective bound preposition valency (`prep` child)
            for child in target_tok.children:
                if child.dep_ == "prep" and child.i > target_tok.i and child.text.lower() in (
                    "of", "to", "for", "with", "about", "from", "in", "on", "at", "into", "against", "over", "towards", "under", "upon"
                ):
                    p_str = child.text.lower()
                    return p_str, "prep", f"followed by bound preposition {p_str}"
            # 3. Predicative / Causative verb complement or subject noun through copula
            if (target_tok.dep_ in ("acomp", "attr", "oprd", "agent", "rcmod", "conj")
                    and eff_head is not None and eff_head.pos_ in ("AUX", "VERB")):
                v_lem_a = (eff_head.lemma_ or eff_head.text).lower()
                if v_lem_a in (
                    "make", "render", "find", "deem", "keep", "consider",
                    "feel", "sound", "seem", "remain", "stay", "become", "grow", "turn", "prove",
                ):
                    return v_lem_a, "verb_copula", f"predicate complement of verb {v_lem_a}"
                for child in eff_head.children:
                    if child.dep_ in ("nsubj", "nsubjpass", "expl") and child.pos_ in ("NOUN", "PROPN"):
                        lem = usable(child)
                        if lem:
                            return lem, "modified_noun", f"predicative complement of {lem}"
                if eff_head.head is not None and eff_head.head.pos_ in ("NOUN", "PROPN"):
                    lem = usable(eff_head.head)
                    if lem:
                        return lem, "modified_noun", f"predicative complement of {lem}"
            return None, None, None

        # 'parents who are too permissive': the adjective reaches its noun through a copula.
        if (target_tok.dep_ in ("acomp", "attr", "oprd", "agent", "rcmod")
                and head is not None and head.pos_ in ("AUX", "VERB")):
            for child in head.children:
                if child.dep_ in ("nsubj", "nsubjpass", "expl") and child.pos_ in ("NOUN", "PROPN"):
                    lem = usable(child)
                    if lem:
                        return lem, "modified_noun", f"predicative complement of {lem}"
            if head.head is not None and head.head.pos_ in ("NOUN", "PROPN"):
                lem = usable(head.head)
                if lem:
                    return lem, "modified_noun", f"predicative complement of {lem}"

        if head is not None:
            lem = usable(head)
            mapped = cls._EXAMPLE_ANCHOR_TYPES.get((p_key, head.pos_))
            if lem and mapped:
                atype, frame = mapped
                return lem, atype, frame.format(lem)
            # 'a life of luxury': the preposition is noise, the noun it points at is the anchor.
            if (not lem or head.pos_ == "ADP") and head.head is not None and head.head.pos_ in ("NOUN", "PROPN"):
                grand = usable(head.head)
                if grand:
                    return grand, "object", f"paired with {grand}"
            if lem and head.pos_ not in ("PUNCT", "DET", "ADP", "AUX", "CCONJ",
                                         "SCONJ", "PRON", "NUM", "PART", "INTJ"):
                return lem, "object", f"paired with {lem}"

        for child in target_tok.children:
            if child.dep_ in ("nsubj", "dobj", "obj", "attr", "oprd", "nmod", "compound", "amod"):
                lem = usable(child)
                if lem:
                    return lem, "object", f"paired with {lem}"
        return None, None, None

    @classmethod
    def _is_target_form(cls, tok, target_word: str) -> bool:
        """True when the token is the target itself or an inflected / derived form of it
        ('posts' -> 'post', 'chocolates' -> 'chocolate', 'depiction' -> 'depicting').
        Deliberately stricter than are_same_word_family, which would happily pair
        'hand' with 'handful'."""
        tw = target_word.strip().lower()
        lem = (tok.lemma_ or tok.text).lower()
        if lem == tw or tok.text.lower() == tw:
            return True
        shorter, diff = min(len(lem), len(tw)), abs(len(lem) - len(tw))
        if not (lem.startswith(tw) or tw.startswith(lem)):
            return False
        if shorter >= 6 and diff <= 4:
            return True
        return 4 <= shorter < 6 and diff <= 2

    @classmethod
    def _anchor_from_sense_example(
        cls,
        target_word: str,
        entry: Dict[str, Any],
        pos: str = "noun",
        definition: Optional[str] = None,
        quote: Optional[str] = None,
        distractors: Optional[List[str]] = None,
        locked_sense: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
        """
        Last-resort anchor and authentic example, taken strictly from the locked sense's
        own dictionary examples so polysemous entries never leak cross-sense or cross-POS
        examples (e.g. 'recognized the symptoms' for the 'publicly honour' sense of 'recognize',
        or 'will be online' [adj] for the adverb 'online').
        Returns (anchor_word, anchor_type, frame_description, authentic_example).
        """
        tw = target_word.strip().lower()
        if not tw or not entry:
            return None, None, None, None
        nlp = cls.get_spacy()
        if nlp is None:
            return None, None, None, None

        dists = [d.strip().lower() for d in (distractors or []) if d.strip()]
        senses = entry.get("senses") or []
        ordered: List[Dict[str, Any]] = []
        if locked_sense is not None:
            ordered = [locked_sense]
        else:
            locked = cls._lock_sense(entry, definition=definition, quote=quote, target_pos=pos)
            if locked is not None and 0 <= locked < len(senses):
                ordered = [senses[locked]]
            elif len(senses) == 1:
                ordered = list(senses)

        pool: List[str] = []
        for sense in ordered:
            pool.extend(sense.get("examples") or [])
        if not pool and len(senses) <= 1:
            pool.extend(entry.get("corpus_examples") or [])
            pool.extend(entry.get("other_dict_examples") or [])

        for ex in pool[:10]:
            if not ex or len(ex.split()) < 3:
                continue
            try:
                doc = nlp(ex)
            except Exception:
                continue
            for tok in doc:
                if not cls._is_target_form(tok, tw):
                    continue
                partner, atype, frame = cls._example_partner(tok, tw, pos)
                if not partner:
                    continue
                if cls._is_delexical_anchor(partner, atype, pos):
                    continue
                if any(cls.double_key_collision(tw, d, anchor=partner, anchor_type=atype, pos=pos)[0]
                       for d in dists):
                    continue
                return partner, atype, frame, ex
        return None, None, None, None

    @classmethod
    def find_ldoce_zero_collision_anchor(
        cls,
        target_word: str,
        pos: str = "noun",
        distractors: Optional[List[str]] = None,
        definition: Optional[str] = None,
        quote: Optional[str] = None,
        locked_sense: Optional[Dict[str, Any]] = None
    ) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
        """
        Retrieves an authoritative, natural collocational anchor and authentic corpus example
        from PedLex Dictionary of Contemporary English (PedLex 6th Edition).
        Guarantees that the chosen anchor collocates with the target word in PedLex with authentic
        native-speaker example backing, while preventing double-key collisions with distractors.
        Strictly enforces Sense-First Primacy (GEMINI.md §1.10): when a sense is locked, sense
        patterns and examples are drawn exclusively from the locked sense, never roaming across
        unrelated senses of the entry.
        Returns:
            (anchor_word, anchor_type, frame_description, authentic_example)
        """
        if not target_word:
            return None, None, None, None

        tw_clean = target_word.strip().lower()
        entry = cls.get_ldoce_entry(tw_clean)
        if not entry:
            return None, None, None, None

        dists = [d.strip().lower() for d in (distractors or []) if d.strip()]
        p_low = pos.strip().lower()

        anchor, atype, frame_desc, authentic_ex = None, None, None, None

        # Resolve locked_sense once if not passed in (only when definition or quote provides context)
        senses = entry.get("senses", [])
        locked_s_idx: Optional[int] = None
        if locked_sense is not None and senses:
            for idx_s, s_cand in enumerate(senses):
                if s_cand is locked_sense or (
                    s_cand.get("definition") == locked_sense.get("definition")
                    and s_cand.get("pos") == locked_sense.get("pos")
                ):
                    locked_s_idx = idx_s
                    break
        if locked_sense is None and senses and (definition or quote):
            target_pos_norm = (
                "adj" if (p_low.startswith("adj") or p_low == "a") else
                "verb" if p_low.startswith("v") else
                "adv" if (p_low.startswith("adv") or p_low == "r") else
                "noun"
            )
            frame_ev = cls._adverb_frame_from_quote(tw_clean, quote) if target_pos_norm == "adv" else None
            locked_s_idx = cls._lock_sense(
                entry,
                definition=definition,
                quote=quote,
                frame_evidence=frame_ev,
                target_pos=target_pos_norm
            )
            if locked_s_idx is not None and 0 <= locked_s_idx < len(senses):
                locked_sense = senses[locked_s_idx]

        # Strict Sense-First scope: never roam into other senses when locked_sense is resolved;
        # if unresolved, filter senses strictly by target POS to prevent cross-homograph leakage.
        p_code = cls._pos_code(pos) if pos else ""
        if locked_sense is not None:
            candidate_senses = [locked_sense]
        elif p_code:
            candidate_senses = [
                s for s in senses
                if not (s.get("pos") and cls._pos_code(s["pos"]) != p_code)
            ] or senses
        else:
            candidate_senses = senses

        # Precompute sense lemma set and other-sense lemma set for filtering entry-level collocations against the locked sense
        sense_lemmas: Set[str] = set()
        other_sense_lemmas: Set[str] = set()
        is_polysemous = len(senses) > 1 and locked_sense is not None
        if locked_sense is not None:
            try:
                nlp = cls.get_spacy()
                self_forms = {tw_clean} | {t.lemma_.lower() for t in nlp(tw_clean)}
                s_def_text = (
                    (locked_sense.get("definition") or "")
                    + " "
                    + (locked_sense.get("signpost") or "")
                    + " "
                    + " ".join(str(x) for x in (locked_sense.get("synonyms") or []))
                    + " "
                    + " ".join(str(x) for x in (locked_sense.get("patterns") or []))
                ).lower()
                s_text = (
                    s_def_text
                    + (
                        ""
                        if is_polysemous
                        else (" " + " ".join(str(x) for x in (locked_sense.get("examples") or []))).lower()
                    )
                )
                sense_lemmas = {
                    tok.lemma_.lower()
                    for tok in nlp(s_text)
                    if tok.is_alpha and not tok.is_stop and len(tok.lemma_) >= 3
                } - self_forms - cls._ANCHOR_STOPWORDS - {
                    "member", "members", "group", "groups", "organization", "organizations",
                    "system", "systems", "part", "parts", "number", "amount", "kind", "type", "form"
                }
                if is_polysemous:
                    other_chunks = []
                    for idx_o, s_other in enumerate(senses):
                        if s_other is locked_sense or idx_o == locked_s_idx:
                            continue
                        other_chunks.append(
                            (s_other.get("definition") or "")
                            + " "
                            + " ".join(str(x) for x in (s_other.get("examples") or [])[:4])
                            + " "
                            + (s_other.get("signpost") or "")
                            + " "
                            + " ".join(str(x) for x in (s_other.get("synonyms") or []))
                        )
                    if other_chunks:
                        o_text = " ".join(other_chunks).lower()
                        raw_other_lemmas = (
                            {
                                tok.lemma_.lower()
                                for tok in nlp(o_text)
                                if tok.is_alpha and not tok.is_stop and len(tok.lemma_) >= 3
                            }
                            - self_forms
                            - cls._ANCHOR_STOPWORDS
                        )
                        disc_sense_lemmas = sense_lemmas - raw_other_lemmas
                        if disc_sense_lemmas:
                            sense_lemmas = disc_sense_lemmas
                        other_sense_lemmas = raw_other_lemmas - sense_lemmas
            except Exception:
                sense_lemmas = set()
                other_sense_lemmas = set()

        def _score_colloc_item(it_c: Dict[str, Any]) -> int:
            it_s_indices = it_c.get("sense_indices")
            if isinstance(it_s_indices, list) and it_s_indices and locked_s_idx is not None:
                return 100 if locked_s_idx in it_s_indices else -100
            if not sense_lemmas:
                return 0
            try:
                nlp = cls.get_spacy()
                self_forms = {tw_clean} | {t.lemma_.lower() for t in nlp(tw_clean)}
                c_text = (
                    (it_c.get("collocation") or "")
                    + " "
                    + (it_c.get("gloss") or "")
                    + " "
                    + (it_c.get("example") or "")
                    + " "
                    + " ".join(str(x) for x in (it_c.get("examples") or []))
                ).lower()
                c_lems = {
                    t.lemma_.lower()
                    for t in nlp(c_text)
                    if t.is_alpha and not t.is_stop and len(t.lemma_) >= 3
                } - self_forms - cls._ANCHOR_STOPWORDS
                pos_hits = len(c_lems & sense_lemmas)
                neg_hits = len(c_lems & other_sense_lemmas) if other_sense_lemmas else 0
                if is_polysemous and neg_hits > pos_hits:
                    return -50
                return pos_hits - neg_hits
            except Exception:
                return 0

        # 1. Adjective Cascades (Instance-Aware: modified_noun >> prep when quote is attributive; prep >> modified_noun when predicative)
        if p_low.startswith("adj") or p_low == "a":
            quote_adj_is_attributive = any(
                "only before noun" in str(s.get("grammar") or "").lower()
                for s in candidate_senses
                if isinstance(s, dict)
            )
            if not quote_adj_is_attributive and quote:
                try:
                    nlp = cls.get_spacy()
                    if nlp:
                        doc_q_adj = nlp(quote)
                        for tok_q in doc_q_adj:
                            if cls._is_target_form(tok_q, tw_clean):
                                has_q_prep = any(ch.dep_ == "prep" and ch.i > tok_q.i for ch in tok_q.children)
                                if tok_q.dep_ == "amod" and tok_q.head.pos_ in ("NOUN", "PROPN") and not has_q_prep:
                                    quote_adj_is_attributive = True
                                break
                except Exception:
                    pass

            def _try_adj_prep_anchor() -> None:
                nonlocal anchor, atype, frame_desc, authentic_ex
                for s in candidate_senses:
                    for pat in s.get("patterns", []):
                        pat_l_a = re.sub(r"\[.*?\]|\(.*?\)", "", pat).strip().lower()
                        if re.search(rf"\bit\s+is\s+{re.escape(tw_clean)}\b.*\bto\b", pat_l_a):
                            continue
                        for p in ("of", "to", "for", "with", "about", "from", "in", "on", "at"):
                            if re.search(rf"\b{re.escape(tw_clean)}(?:\s*/\s*[a-z]+)*\s+(?:[a-z]+\s*/\s*)*{p}\b", pat_l_a):
                                if p == "to":
                                    m_inf = re.search(r"\bto\s+([a-z]+)", pat_l_a)
                                    if m_inf and m_inf.group(1) not in (
                                        "sb", "sth", "someone", "something", "oneself", "himself", "herself", "them", "the", "a", "an"
                                    ):
                                        continue  # Infinitive pattern (e.g. 'simple to use'), NOT a bound preposition
                                # Select example that directly demonstrates the bound preposition following the adjective
                                matching_exs = [
                                    e for e in s.get("examples", [])
                                    if re.search(rf"\b{re.escape(tw_clean)}\w*(?:\s+[a-z]+){{0,2}}\s+{p}\b", e.lower())
                                ]
                                ex = matching_exs[0] if matching_exs else (s.get("examples", [""])[0] if s.get("examples") else "")
                                if not any(cls.double_key_collision(tw_clean, d, anchor=p, anchor_type="prep", pos="adj")[0] for d in dists):
                                    anchor = p
                                    atype = "prep"
                                    frame_desc = f"followed by bound preposition {p}"
                                    authentic_ex = ex
                                    break
                        if anchor:
                            break
                    if anchor:
                        break

            def _try_adj_modified_noun_anchor() -> None:
                nonlocal anchor, atype, frame_desc, authentic_ex
                if not anchor and candidate_senses:
                    try:
                        nlp = cls.get_spacy()
                        quote_lower = (quote or "").lower()
                        cand_nouns = []
                        for s in candidate_senses:
                            for ex in s.get("examples", []):
                                doc_ex = nlp(ex)
                                for tok in doc_ex:
                                    if cls._is_target_form(tok, tw_clean):
                                        eff_h = tok.head
                                        while eff_h is not None and eff_h.dep_ == "conj" and eff_h.head.pos_ == "ADJ" and eff_h.head != eff_h:
                                            eff_h = eff_h.head
                                        if eff_h is not None and eff_h.pos_ == "ADJ" and eff_h.head != eff_h:
                                            eff_h = eff_h.head
                                        if tok.dep_ in ("amod", "conj") and eff_h is not None and eff_h.pos_ in ("NOUN", "PROPN") and tok.i < eff_h.i:
                                            cand_n = eff_h.lemma_.lower()
                                            if cand_n not in cls._ANCHOR_STOPWORDS and len(cand_n) >= 3:
                                                cand_nouns.append((cand_n, ex))
                        cand_nouns.sort(key=lambda x: (1 if quote_lower and x[0] in quote_lower else 0), reverse=True)
                        for cand_n, ex in cand_nouns:
                            if not any(cls.double_key_collision(tw_clean, d, anchor=cand_n, anchor_type="modified_noun", pos="adj")[0] for d in dists):
                                anchor = cand_n
                                atype = "modified_noun"
                                frame_desc = f"modifying noun {cand_n}"
                                authentic_ex = ex
                                break
                    except Exception:
                        pass

                if not anchor and entry.get("collocations", {}).get("nouns"):
                    colloc_nouns = [
                        it_c for it_c in entry["collocations"]["nouns"]
                        if not it_c.get("pos") or cls._pos_code(it_c.get("pos")) == "adj"
                    ]
                    if sense_lemmas:
                        if is_polysemous:
                            colloc_nouns = [it_c for it_c in colloc_nouns if _score_colloc_item(it_c) > 0]
                        colloc_nouns.sort(key=_score_colloc_item, reverse=True)
                    for it in colloc_nouns:
                        c_str = it.get("collocation", "").lower()
                        ex = it.get("example", "")
                        tok_n = cls._pick_anchor_token([c_str], target_word=tw_clean, prefer_pos="noun")
                        if tok_n and len(tok_n) >= 3 and tok_n not in cls._ANCHOR_STOPWORDS:
                            if not any(cls.double_key_collision(tw_clean, d, anchor=tok_n, anchor_type="modified_noun", pos="adj")[0] for d in dists):
                                anchor = tok_n
                                atype = "modified_noun"
                                frame_desc = f"modifying noun {tok_n}"
                                authentic_ex = ex
                                break

            if quote_adj_is_attributive:
                _try_adj_modified_noun_anchor()
                if not anchor:
                    _try_adj_prep_anchor()
            else:
                _try_adj_prep_anchor()
                if not anchor:
                    _try_adj_modified_noun_anchor()

            # Level 2.5: Causative / Linking Predicative Verb Frames (e.g. 'make it possible', 'keep silent', 'feel surprised')
            if not anchor and entry.get("collocations", {}).get("phrases"):
                colloc_phrases = [
                    it_c for it_c in entry["collocations"]["phrases"]
                    if not it_c.get("pos") or cls._pos_code(it_c.get("pos")) == "adj"
                ]
                if is_polysemous and sense_lemmas:
                    colloc_phrases = [it_c for it_c in colloc_phrases if _score_colloc_item(it_c) > 0]
                for it in colloc_phrases:
                    c_str = it.get("collocation", "").lower()
                    ex = it.get("example", "")
                    for cv in (
                        "make", "render", "find", "deem", "keep", "consider",
                        "feel", "sound", "seem", "remain", "stay", "become", "grow", "turn", "prove",
                    ):
                        if re.search(rf"\b{cv}\b", c_str):
                            if not any(cls.double_key_collision(tw_clean, d, anchor=cv, anchor_type="verb_copula", pos="adj")[0] for d in dists):
                                anchor = cv
                                atype = "verb_copula"
                                frame_desc = f"predicate complement of causative verb {cv}"
                                authentic_ex = ex
                                break
                    if anchor:
                        break

            # Level 3: Evaluative Adverb Modifiers (e.g. 'deceptively simple', 'perfectly possible')
            if not anchor:
                verb_collocs = [
                    it_c for it_c in (entry.get("collocations", {}).get("verbs", []) + entry.get("collocations", {}).get("adverbs", []))
                    if not it_c.get("pos") or cls._pos_code(it_c.get("pos")) == "adj"
                ]
                if is_polysemous and sense_lemmas:
                    verb_collocs = [it_c for it_c in verb_collocs if _score_colloc_item(it_c) > 0]
                for it in verb_collocs:
                    c_str = it.get("collocation", "").lower()
                    ex = it.get("example", "")
                    tok_adv = cls._pick_anchor_token([c_str], target_word=tw_clean)
                    if tok_adv and (tok_adv.endswith("ly") or tok_adv in ("quite", "fairly", "pretty")):
                        if not any(cls.double_key_collision(tw_clean, d, anchor=tok_adv, anchor_type="adv_mod", pos="adj")[0] for d in dists):
                            anchor = tok_adv
                            atype = "adv_mod"
                            frame_desc = f"modified by adverb {tok_adv}"
                            authentic_ex = ex
                            break
                    if anchor:
                        break

        # 2. Verb Cascades (direct object patient >> governing prep)
        elif p_low.startswith("v"):
            is_strictly_intransitive = (
                bool(candidate_senses)
                and any(s.get("grammar") for s in candidate_senses)
                and all(
                    "intransitive" in [str(g).lower().strip("[] ") for g in (s.get("grammar") or [])]
                    and "transitive" not in [str(g).lower().strip("[] ") for g in (s.get("grammar") or [])]
                    for s in candidate_senses
                    if s.get("grammar")
                )
            )
            # Step A: Prioritize direct object patient strictly from the locked target sense
            if not is_strictly_intransitive:
                try:
                    nlp = cls.get_spacy()
                    for s in candidate_senses:
                        s_grams = [str(g).lower().strip("[] ") for g in (s.get("grammar") or [])]
                        if "intransitive" in s_grams and "transitive" not in s_grams:
                            continue
                        for ex in s.get("examples", []):
                            doc_ex = nlp(ex)
                            for tok in doc_ex:
                                if tok.lemma_.lower() == tw_clean and tok.pos_ in ("VERB", "AUX"):
                                    for child in tok.children:
                                        if child.dep_ in ("dobj", "obj") and child.is_alpha and len(child.lemma_) >= 3 and child.lemma_.lower() not in cls._ANCHOR_STOPWORDS and child.pos_ in ("NOUN", "PROPN"):
                                            cand_obj = child.lemma_.lower()
                                            if not any(cls.double_key_collision(tw_clean, d, anchor=cand_obj, anchor_type="object", pos="verb")[0] for d in dists):
                                                anchor = cand_obj
                                                atype = "object"
                                                frame_desc = f"direct object noun {cand_obj}"
                                                authentic_ex = ex
                                                break
                                    if anchor:
                                        break
                            if anchor:
                                break
                        if anchor:
                            break
                except Exception:
                    pass

            # Step B: Sense-aligned direct object nouns in collocations
            if not anchor and not is_strictly_intransitive and entry.get("collocations", {}).get("nouns"):
                try:
                    nlp = cls.get_spacy()
                except Exception:
                    nlp = None
                colloc_nouns = [
                    it for it in entry["collocations"]["nouns"]
                    if not it.get("pos") or cls._pos_code(it.get("pos")) == "v"
                ]
                if sense_lemmas and nlp is not None:
                    if is_polysemous:
                        colloc_nouns = [it_c for it_c in colloc_nouns if _score_colloc_item(it_c) > 0]
                    colloc_nouns.sort(key=_score_colloc_item, reverse=True)
                for it in colloc_nouns:
                    ex = it.get("example", "")
                    c_str = it.get("collocation", "").lower()
                    if re.search(rf"\b{re.escape(tw_clean)}\w*\s+(?:to|on|at|about|for|with|over|into|of|from|in)\b", c_str):
                        continue
                    tok_clean = cls._pick_anchor_token([c_str], target_word=tw_clean, prefer_pos="noun")
                    if not tok_clean:
                        continue
                    if ex and nlp is not None:
                        try:
                            doc_ex = nlp(ex)
                            is_noun_compound = any(
                                t.pos_ in ("NOUN", "PROPN") and t.dep_ == "compound" and (t.head.lemma_.lower() == tok_clean or t.head.text.lower() == tok_clean)
                                for t in doc_ex if t.lemma_.lower() == tw_clean or t.text.lower().startswith(tw_clean)
                            )
                            has_verb_usage = any(
                                t.pos_ in ("VERB", "AUX")
                                for t in doc_ex if t.lemma_.lower() == tw_clean or t.text.lower().startswith(tw_clean)
                            )
                            is_prep_obj = any(
                                n_tok.dep_ == "pobj" and n_tok.head.dep_ == "prep" and (
                                    n_tok.head.head.lemma_.lower() == tw_clean or n_tok.head.head.text.lower().startswith(tw_clean)
                                )
                                for n_tok in doc_ex
                                if n_tok.lemma_.lower() == tok_clean or n_tok.text.lower() == tok_clean
                            )
                            if is_noun_compound or not has_verb_usage or is_prep_obj:
                                continue
                        except Exception:
                            pass
                    if not any(cls.double_key_collision(tw_clean, d, anchor=tok_clean, anchor_type="object", pos="verb")[0] for d in dists):
                        anchor = tok_clean
                        atype = "object"
                        frame_desc = f"direct object noun {tok_clean}"
                        authentic_ex = ex
                        break
                    if anchor:
                        break

            # Step C: Preposition patterns strictly within the locked sense
            if not anchor:
                for s in candidate_senses:
                    for pat in s.get("patterns", []):
                        # Strip infinitival `to do` so patterns like 'offer to do something'
                        # are never mistaken for governing preposition 'to'.
                        pat_l = re.sub(r"\bto\s+do\b.*", "", pat.lower())
                        for p in ("to", "on", "at", "about", "for", "with", "over", "into", "of", "from", "in"):
                            if re.search(rf"\b{p}\b", pat_l):
                                matching_exs = [e for e in s.get("examples", []) if re.search(rf"\b{p}\b", e.lower())]
                                ex = matching_exs[0] if matching_exs else (s.get("examples", [""])[0] if s.get("examples") else "")
                                if not any(cls.double_key_collision(tw_clean, d, anchor=p, anchor_type="prep", pos="verb")[0] for d in dists):
                                    anchor = p
                                    atype = "prep"
                                    frame_desc = f"governing preposition {p}"
                                    authentic_ex = ex
                                    break
                        if anchor:
                            break
                    if anchor:
                        break

        # 3. Noun Cascades (agentive subject verbs >> governing verbs >> locked-sense preposition valency >> descriptive adjectives)
        elif p_low.startswith("n"):
            # Level 1: Agentive / Characteristic Subject Verbs (e.g. 'telephone rings', 'clock chimes', 'heart beats')
            subj_verb_collocs = [
                it for it in entry.get("collocations", {}).get("verbs", [])
                if not it.get("pos") or cls._pos_code(it.get("pos")) == "n"
            ]
            if is_polysemous and sense_lemmas:
                subj_verb_collocs = [it_c for it_c in subj_verb_collocs if _score_colloc_item(it_c) > 0]
            for it in subj_verb_collocs:
                c_str = it.get("collocation", "").lower()
                ex = it.get("example", "")
                is_subj = bool(re.search(
                    r"\b(?:ring|rings|rang|rung|chime|chimes|chimed|beat|beats|beating"
                    r"|erupt|erupts|erupted|start|starts|started|occur|occurs|occurred"
                    r"|exist|exists|existed|change|changes|changed|hardens|hardened)\b",
                    c_str,
                ))
                if is_subj:
                    tok_partner = cls._pick_anchor_token([c_str], target_word=tw_clean)
                    if tok_partner:
                        v_lem = "ring" if tok_partner in ("rings", "rang", "rung") else (
                            "chime" if tok_partner in ("chimes", "chimed") else (
                                "beat" if tok_partner in ("beats", "beating") else (
                                    "erupt" if tok_partner in ("erupts", "erupted") else tok_partner
                                )
                            )
                        )
                        if not any(cls.double_key_collision(tw_clean, d, anchor=v_lem, anchor_type="verb_subject", pos="noun")[0] for d in dists):
                            anchor = v_lem
                            atype = "verb_subject"
                            frame_desc = f"subject of verb {v_lem}"
                            authentic_ex = ex
                            break

            # Level 2: Governing Verbs scored against locked sense
            if not anchor:
                c_verbs = [
                    it for it in entry.get("collocations", {}).get("verbs", [])
                    if not it.get("pos") or cls._pos_code(it.get("pos")) == "n"
                ]
                if sense_lemmas and c_verbs:
                    if is_polysemous:
                        c_verbs = [it_c for it_c in c_verbs if _score_colloc_item(it_c) > 0]
                    c_verbs.sort(key=_score_colloc_item, reverse=True)
                for it in c_verbs:
                    c_str = it.get("collocation", "").lower()
                    ex = it.get("example", "")
                    tok_partner = cls._pick_anchor_token([c_str], target_word=tw_clean, prefer_pos="verb")
                    if tok_partner and not cls._is_delexical_anchor(tok_partner, "verb", "noun"):
                        if not any(cls.double_key_collision(tw_clean, d, anchor=tok_partner, anchor_type="verb", pos="noun")[0] for d in dists):
                            anchor = tok_partner
                            atype = "verb"
                            frame_desc = f"direct object of verb {tok_partner}"
                            authentic_ex = ex
                            break

            # Level 3: Noun preposition patterns strictly from locked sense (e.g. 'control of/over', 'access to', 'interest in')
            if not anchor:
                quote_preps = []
                if quote:
                    for p_cand in ("over", "of", "to", "for", "with", "about", "from", "in", "on", "into", "against", "towards", "toward"):
                        if re.search(rf"\b{p_cand}\b", quote.lower()):
                            quote_preps.append(p_cand)

                search_order = quote_preps + [p for p in ("over", "of", "to", "for", "with", "about", "from", "in", "on", "into", "against", "towards", "toward") if p not in quote_preps]

                for s in candidate_senses:
                    for pat in s.get("patterns", []):
                        pat_l = pat.lower()
                        for p in search_order:
                            if re.search(rf"\b{p}\b", pat_l):
                                matching_exs = [e for e in s.get("examples", []) if re.search(rf"\b{p}\b", e.lower())]
                                ex = matching_exs[0] if matching_exs else (s.get("examples", [""])[0] if s.get("examples") else "")
                                if not any(cls.double_key_collision(tw_clean, d, anchor=p, anchor_type="prep", pos="noun", quote=quote)[0] for d in dists):
                                    anchor = p
                                    atype = "prep"
                                    frame_desc = f"followed by preposition {p}"
                                    authentic_ex = ex
                                    break
                        if anchor:
                            break
                    if anchor:
                        break

            # Level 4: Descriptive Adjectives scored against locked sense
            if not anchor:
                c_adjs = [
                    it for it in entry.get("collocations", {}).get("adjectives", [])
                    if not it.get("pos") or cls._pos_code(it.get("pos")) == "n"
                ]
                if sense_lemmas and c_adjs:
                    if is_polysemous:
                        c_adjs = [it_c for it_c in c_adjs if _score_colloc_item(it_c) > 0]
                    c_adjs.sort(key=_score_colloc_item, reverse=True)
                for it in c_adjs:
                    c_str = it.get("collocation", "").lower()
                    ex = it.get("example", "")
                    tok_partner = cls._pick_anchor_token([c_str], target_word=tw_clean, prefer_pos="adj")
                    if tok_partner and tok_partner not in ("good", "bad", "great"):
                        if not any(cls.double_key_collision(tw_clean, d, anchor=tok_partner, anchor_type="adj", pos="noun")[0] for d in dists):
                            if locked_sense is not None:
                                sense_match_exs = [
                                    s_ex for s_ex in (locked_sense.get("examples") or [])
                                    if re.search(rf"\b{re.escape(tok_partner)}\b", s_ex.lower())
                                ]
                                if sense_match_exs:
                                    ex = sense_match_exs[0]
                            anchor = tok_partner
                            atype = "adj"
                            frame_desc = f"modified by adjective {tok_partner}"
                            authentic_ex = ex
                            break

        # 4. Adverb Cascades (modifies_verb >> modifies_adj or modifies_adj >> modifies_verb based on adverb semantics & quote evidence)
        elif p_low.startswith("adv") or p_low == "r":
            quote_frame = cls._adverb_frame_from_quote(tw_clean, quote)

            # First priority: Anchor token directly present in authentic quote
            if quote_frame and quote:
                try:
                    nlp = cls.get_spacy()
                    doc = nlp(quote)
                    for tok in doc:
                        if tok.lemma_.lower() == tw_clean or tok.text.lower() == tw_clean:
                            head = tok.head
                            if head and head.lemma_.lower() not in cls._ANCHOR_STOPWORDS and head.lemma_.lower() != tw_clean:
                                h_lem = head.lemma_.lower()
                                if not any(cls.double_key_collision(tw_clean, d, anchor=h_lem, anchor_type=quote_frame, pos="adv", quote=quote)[0] for d in dists):
                                    anchor = h_lem
                                    atype = quote_frame
                                    frame_desc = f"modifying {head.pos_.lower()} {h_lem}"
                                    authentic_ex = quote
                                    break
                except Exception:
                    pass

            # Second priority: Syntactic extraction strictly from locked PedLex sense examples
            if not anchor:
                try:
                    nlp = cls.get_spacy()
                    is_degree_adv = tw_clean in (
                        "extremely", "completely", "totally", "utterly", "highly", "deeply",
                        "fairly", "quite", "rather", "pretty", "slightly", "barely", "scarcely",
                        "hardly", "vastly", "immensely", "hugely", "terribly", "awfully", "decidedly"
                    ) or (locked_sense and any(k in (locked_sense.get("grammar") or "").lower() for k in ("adjective", "adverb", "[+adjective")))

                    target_pos_order = [("ADJ", "modifies_adj"), ("VERB", "modifies_verb")] if is_degree_adv else [("VERB", "modifies_verb"), ("ADJ", "modifies_adj")]
                    for req_pos, req_atype in target_pos_order:
                        for s in candidate_senses:
                            for ex in s.get("examples", []):
                                doc = nlp(ex)
                                for tok in doc:
                                    if tok.lemma_.lower() == tw_clean or tok.text.lower() == tw_clean:
                                        head = tok.head
                                        cand_tok = None
                                        if head and head.pos_ == req_pos and head.lemma_.lower() not in cls._ANCHOR_STOPWORDS and head.lemma_.lower() != tw_clean:
                                            cand_tok = head
                                        elif tok.i > 0 and doc[tok.i - 1].pos_ == req_pos and doc[tok.i - 1].lemma_.lower() not in cls._ANCHOR_STOPWORDS and doc[tok.i - 1].lemma_.lower() != tw_clean:
                                            cand_tok = doc[tok.i - 1]
                                        elif tok.i + 1 < len(doc) and doc[tok.i + 1].pos_ == req_pos and doc[tok.i + 1].lemma_.lower() not in cls._ANCHOR_STOPWORDS and doc[tok.i + 1].lemma_.lower() != tw_clean:
                                            cand_tok = doc[tok.i + 1]
                                        if cand_tok is not None:
                                            cand_head = cand_tok.lemma_.lower()
                                            if not any(cls.double_key_collision(tw_clean, d, anchor=cand_head, anchor_type=req_atype, pos="adv", quote=quote)[0] for d in dists):
                                                anchor = cand_head
                                                atype = req_atype
                                                frame_desc = f"modifying {cand_tok.pos_.lower()} {cand_head}"
                                                authentic_ex = ex
                                                break
                                if anchor:
                                    break
                            if anchor:
                                break
                        if anchor:
                            break
                except Exception:
                    pass

        # 5. Last resort: the dictionary's own locked sense example.
        if not anchor:
            anchor, atype, frame_desc, authentic_ex = cls._anchor_from_sense_example(
                tw_clean, entry, pos=p_low, definition=definition,
                quote=quote, distractors=dists, locked_sense=locked_sense,
            )

        if anchor:
            return anchor, atype, frame_desc, authentic_ex

        return None, None, None, None



    @classmethod
    def ldoce_example_for_anchor(
        cls,
        word: str,
        anchor: Optional[str],
        locked_sense: Optional[Dict[str, Any]] = None,
        definition: Optional[str] = None,
        quote: Optional[str] = None,
        pos: Optional[str] = None
    ) -> Optional[str]:
        """
        The authentic PedLex sentence that demonstrates the (target, anchor) pair.
        Returning None is meaningful: it means the dictionary never puts these two words
        together in the locked sense, so a quote-derived anchor is an accident of one sentence
        ('further penalty', 'maintain contact') and cannot serve as a structural model.
        Strictly enforces Sense-First Primacy (GEMINI.md §1.10): when locked_sense is resolved,
        never pulls examples from other senses of the entry.
        """
        if not word or not anchor:
            return None
        tw_l = word.strip().lower()
        anc_l = re.sub(r"[^a-z]", "", anchor.strip().lower())
        if len(anc_l) < 2:
            return None
        anc_re = re.compile(rf"\b{re.escape(anc_l)}\b")
        tw_re = re.compile(rf"\b{re.escape(tw_l)}\b")

        t_entry = cls.get_ldoce_entry(tw_l)
        if t_entry:
            if locked_sense is None and (definition or quote or pos) and t_entry.get("senses"):
                s_idx = cls._lock_sense(t_entry, definition=definition, quote=quote, target_pos=pos)
                if s_idx is not None and 0 <= s_idx < len(t_entry["senses"]):
                    locked_sense = t_entry["senses"][s_idx]

            # 0. Sense-First Priority: Check locked_sense's own pattern_bundles, patterns, and examples
            if locked_sense:
                for pb in locked_sense.get("pattern_bundles") or []:
                    if isinstance(pb, dict) and pb.get("example"):
                        pat_str = (pb.get("pattern") or "").lower()
                        ex_str = pb["example"].lower()
                        if anc_re.search(pat_str) or anc_re.search(ex_str):
                            return pb["example"]
                for pat in locked_sense.get("patterns") or []:
                    if anc_re.search(pat.lower()) and locked_sense.get("examples"):
                        m_exs = [e for e in locked_sense["examples"] if anc_re.search(e.lower())]
                        return m_exs[0] if m_exs else locked_sense["examples"][0]
                for ex in locked_sense.get("examples") or []:
                    if anc_re.search(ex.lower()):
                        return ex

            # 1. Curated collocations of the target (only if anchor is NOT a bare preposition from another sense,
            # and strictly sense-isolated when locked_sense is resolved on a polysemous word)
            is_bare_prep = anc_l in ("of", "to", "for", "with", "about", "from", "in", "on", "at", "into", "over", "under", "against", "towards", "toward", "by", "upon")
            if not is_bare_prep:
                senses_list = t_entry.get("senses") or []
                is_poly = len(senses_list) > 1 and locked_sense is not None
                locked_idx = None
                if is_poly:
                    for idx_s, s_c in enumerate(senses_list):
                        if s_c is locked_sense or (
                            s_c.get("definition") == locked_sense.get("definition")
                            and s_c.get("pos") == locked_sense.get("pos")
                        ):
                            locked_idx = idx_s
                            break
                s_kws: Set[str] = set()
                o_kws: Set[str] = set()
                if is_poly and locked_sense is not None:
                    l_str = f"{locked_sense.get('signpost') or ''} {locked_sense.get('definition') or ''} {' '.join(str(x) for x in (locked_sense.get('examples') or []))} {' '.join(str(x) for x in (locked_sense.get('synonyms') or []))}".lower()
                    s_kws = {w for w in re.findall(r"[a-z]{4,}", l_str) if w != tw_l and w not in cls._ANCHOR_STOPWORDS}
                    for idx_o, s_o in enumerate(senses_list):
                        if s_o is locked_sense or idx_o == locked_idx:
                            continue
                        o_str = f"{s_o.get('signpost') or ''} {s_o.get('definition') or ''} {' '.join(str(x) for x in (s_o.get('examples') or [])[:4])}".lower()
                        o_kws |= {w for w in re.findall(r"[a-z]{4,}", o_str) if w != tw_l and w not in cls._ANCHOR_STOPWORDS and w not in s_kws}

                def _colloc_matches_locked_sense(item_obj: Dict[str, Any]) -> bool:
                    if item_obj.get("pos") and locked_sense is not None:
                        t_pos = locked_sense.get("pos") or t_entry.get("pos")
                        if t_pos and cls._pos_code(item_obj["pos"]) != cls._pos_code(t_pos):
                            return False
                    if not is_poly:
                        return True
                    s_inds = item_obj.get("sense_indices")
                    if isinstance(s_inds, list) and s_inds and locked_idx is not None:
                        return locked_idx in s_inds
                    c_probe = f"{item_obj.get('collocation') or ''} {item_obj.get('gloss') or ''} {item_obj.get('example') or ''}".lower()
                    c_words = {w for w in re.findall(r"[a-z]{4,}", c_probe) if w != tw_l and w not in cls._ANCHOR_STOPWORDS}
                    if o_kws and len(c_words & o_kws) >= len(c_words & s_kws) and len(c_words & o_kws) > 0:
                        return False
                    if s_kws and len(c_words & s_kws) == 0:
                        return False
                    return True

                for items in (t_entry.get("collocations") or {}).values():
                    for item in items or []:
                        if anc_re.search((item.get("collocation") or "").lower()) and item.get("example") and _colloc_matches_locked_sense(item):
                            return item["example"]
                for item in t_entry.get("cross_collocations") or []:
                    if anc_re.search((item.get("collocation") or "").lower()) and item.get("example") and _colloc_matches_locked_sense(item):
                        return item["example"]

            # 2. Only scan all senses if no locked_sense was resolved
            if locked_sense is None:
                for s in t_entry.get("senses") or []:
                    for pat in s.get("patterns") or []:
                        if anc_re.search(pat.lower()) and s.get("examples"):
                            m_exs = [e for e in s["examples"] if anc_re.search(e.lower())]
                            return m_exs[0] if m_exs else s["examples"][0]
                for s in t_entry.get("senses") or []:
                    for ex in s.get("examples") or []:
                        if anc_re.search(ex.lower()):
                            return ex

        # 3. The anchor's own entry may be the one that records the pairing (for content-word anchors)
        if anc_l not in ("of", "to", "for", "with", "about", "from", "in", "on", "at", "into", "over", "under", "against", "towards", "toward", "by", "upon"):
            a_entry = cls.get_ldoce_entry(anc_l)
            if a_entry:
                for items in (a_entry.get("collocations") or {}).values():
                    for item in items or []:
                        if tw_re.search((item.get("collocation") or "").lower()) and item.get("example"):
                            return item["example"]

        return None

    @classmethod
    def ldoce_sense_example(
        cls,
        word: str,
        pos: Optional[str] = None,
        definition: Optional[str] = None,
        quote: Optional[str] = None,
        locked_sense: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        The dictionary's own sentence for the sense this item teaches, with no anchor
        requirement attached. Used as the structural model when no collocational anchor
        survived the gates - an item with no anchor still needs a real pattern to model.
        """
        chosen_sense = locked_sense
        if chosen_sense is None:
            entry = cls.get_ldoce_entry(word)
            if not entry:
                return None
            senses = entry.get("senses") or []
            locked = cls._lock_sense(entry, definition=definition, quote=quote, target_pos=pos)
            if locked is not None and 0 <= locked < len(senses):
                chosen_sense = senses[locked]
        if chosen_sense is not None:
            for ex in chosen_sense.get("examples") or []:
                if ex and len(ex.split()) >= 3:
                    return ex
            for pb in chosen_sense.get("pattern_bundles", []):
                if isinstance(pb, dict) and pb.get("example") and len(pb["example"].split()) >= 3:
                    return pb["example"]
        return None

