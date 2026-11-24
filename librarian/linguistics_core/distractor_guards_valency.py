"""Preposition government, valency frames, and syntactic slot clash evaluation."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from .linguistic_rules_loader import (
    get_academic_adj_families,
    get_adj_prep_valency_map,
    get_adjunct_prep_pairs,
    get_anchor_stopwords,
    get_register_banned_words,
    get_typical_complement_preps,
)


class DistractorGuardsValencyMixin:
    """Guards for preposition government, valency complementation, and syntactic slot clashes."""

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

        # 1. PedLex Collocations
        ped_coll = cls.get_pedlex_collocations(w)
        if ped_coll:
            for p in ped_coll.get("prep", []):
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
        and harvests co-scenario lifecycle verbs from `pedlex[noun]['verb_before']` that share the same
        domain scenario while exhibiting contrasting syntactic valency (e.g., `apply for`, `attend [obj]`,
        `graduate from`, `withdraw from` for `enroll in [school/university/course]`)."""
        t = (target_word or "").strip().lower()
        if not t:
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
                    and cls.is_exact_pedlex_headword(cand_n)
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
            vb_list = (cls.get_pedlex_collocations(n) or {}).get("verb_before", [])
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


    _ADJ_PREP_VALENCY_MAP: Dict[str, List[str]] = get_adj_prep_valency_map()
    _ACADEMIC_ADJ_FAMILIES: List[List[str]] = get_academic_adj_families()
    _REGISTER_BANNED_WORDS: Set[str] = get_register_banned_words()
    _ANCHOR_STOPWORDS: Set[str] = get_anchor_stopwords()
    _ADJUNCT_PREP_PAIRS: Set[Tuple[str, str]] = get_adjunct_prep_pairs()
    _TYPICAL_COMPLEMENT_PREPS: Set[str] = get_typical_complement_preps()


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
        if not gov_prep and extracted_gov_prep and anchor_type in (None, "", "prep"):
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



__all__ = ["DistractorGuardsValencyMixin"]
