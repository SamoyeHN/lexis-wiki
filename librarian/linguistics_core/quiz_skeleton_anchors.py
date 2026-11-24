"""
Collocational anchor resolution and syntactic valency binding.
Resolves POS-specific anchors (verb, noun, adjective, adverb) from authentic quotes,
PedLex atomic pattern bundles, and valency structures.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from .quiz_skeleton_context import CXN_ANCHOR_TYPES, SkeletonContext


class QuizSkeletonAnchorsMixin:
    @classmethod
    def _resolve_skeleton_anchors(
        cls,
        ctx: SkeletonContext,
        batch_target_words: Set[str],
        nlp: Any,
    ) -> None:
        """
        Stage 2: Retrieve Authentic Collocational Anchor (Sense-Guided)
        across verb, noun, adjective, and adverb constructions.
        """
        w = ctx.w
        w_lower = ctx.w_lower
        raw_w_item = ctx.raw_w_item
        raw_pos = ctx.raw_pos
        canonical_pos = ctx.canonical_pos
        it = ctx.it
        anchor = ctx.anchor
        anchor_type = ctx.anchor_type
        anchor_source = ctx.anchor_source
        prep_obj = ctx.prep_obj
        target_inflections_set = ctx.target_inflections_set
        locked_sense = ctx.locked_sense
        t_s_idx = ctx.t_s_idx
        target_entry = ctx.target_entry
        sense_native_prep = ctx.sense_native_prep
        initial_atomic_bundle = ctx.initial_atomic_bundle
        word_subcats = ctx.word_subcats
        is_idiom_slot = ctx.is_idiom_slot
        quote_adj_is_attributive = ctx.quote_adj_is_attributive
        quote_is_transitive = ctx.quote_is_transitive
        quote_governed_prep = ctx.quote_governed_prep
        cxn_override_pat = ctx.cxn_override_pat
        cxn_override_frame = ctx.cxn_override_frame
        pattern_anchor = ctx.pattern_anchor
        _cxn_anchor_types = CXN_ANCHOR_TYPES


        def _tok_matches_target(tok, doc_obj) -> bool:
            if "-" not in w_lower:
                if (tok.i > 0 and doc_obj[tok.i - 1].text == "-" and not doc_obj[tok.i - 1].whitespace_) or (
                    tok.i + 1 < len(doc_obj) and doc_obj[tok.i + 1].text == "-" and not tok.whitespace_
                ):
                    return False
            t_l = tok.text.lower()
            return t_l == w_lower or tok.lemma_.lower() == w_lower or t_l in target_inflections_set

        quote_adj_is_attributive = False

        if canonical_pos == "verb":
            q_low_v = (it.get("quote") or "").lower()
            # Family 8: Mandative Subjunctive Verb ('insist/suggest/demand/recommend that [sb] (should) [V-base]')
            if not anchor and "V_MANDATIVE_SUBJ" in word_subcats and q_low_v:
                if re.search(
                    rf"\b{re.escape(w_lower)}(?:s|es|ed|d|ing)?\s+that\s+(?:[a-z]+\s+){{1,4}}(?:should\s+[a-z]+|be\b|not\s+[a-z]+|[a-z]+\b)",
                    q_low_v,
                ) and not re.search(rf"\b{re.escape(w_lower)}(?:s|es|ed|d|ing)?\s+(?:on|upon|to|with)\b", q_low_v):
                    anchor = "that"
                    anchor_type = "mandative_subj"
                    cxn_override_pat = f"{w_lower} that [sb] (should) [V-base]"
                    cxn_override_frame = "... ____ that [sb] (should) [V-base]"
            # Family 7: Negative Polarity Verb ('budge' under negation)
            if not anchor and "NPI_ITEM" in word_subcats and q_low_v:
                if re.search(r"\b(?:not|n't|never|hardly|scarcely|barely|refused?\s+to|would\s+not|could\s+not|will\s+not|cannot|can't|won't|wouldn't|couldn't)\b", q_low_v):
                    anchor = "not"
                    anchor_type = "npi"
                    cxn_override_pat = f"not/never {w_lower}"
                    cxn_override_frame = "... not/never ____"
            # Family 3: Gerund-Only vs Infinitive-Only Verb Complementation
            if not anchor and "V_GERUND_ONLY" in word_subcats and q_low_v:
                if re.search(rf"\b{re.escape(w_lower)}(?:s|es|ed|d|ing)?\s+(?:not\s+)?[a-z]+ing\b", q_low_v):
                    anchor = "doing"
                    anchor_type = "gerund"
                    cxn_override_pat = f"{w_lower} [doing sth]"
                    cxn_override_frame = "... ____ [doing sth]"
            if not anchor and "V_INF_ONLY" in word_subcats:
                sense_pats_low = [str(p).lower() for p in ((locked_sense or {}).get("patterns") or [])]
                sense_is_inf_only = bool(
                    sense_pats_low
                    and all("to do" in p and not re.search(r"\b(?:sb|somebody|someone)\s+to\s+do\b", p) for p in sense_pats_low)
                )
                if sense_is_inf_only or (
                    q_low_v and re.search(rf"\b{re.escape(w_lower)}(?:s|es|ed|d|ing)?\s+(?:not\s+)?to\s+[a-z]+\b", q_low_v)
                ):
                    anchor = "to"
                    anchor_type = "to_inf"
                    cxn_override_pat = f"{w_lower} to do [sth]"
                    cxn_override_frame = "... ____ to do [sth]"
            # Family 4: Verb + [sb] + to do [sth]
            _has_sb_to_do_pat = "V_OBJ_TO_DO" in word_subcats or any(
                re.search(r"\b(?:sb|somebody|someone)\s+to\s+do\b", str(p).lower())
                for p in ((locked_sense or {}).get("patterns") or [])
            )
            if not anchor and _has_sb_to_do_pat and q_low_v:
                if re.search(rf"\b{re.escape(w_lower)}(?:s|es|ed|d|ing)?\s+(?:[a-z]+\s+){{1,3}}to\s+[a-z]+\b", q_low_v):
                    anchor = "to"
                    anchor_type = "sb_to_do"
                    cxn_override_pat = f"{w_lower} [sb] to do [sth]"
                    cxn_override_frame = "... ____ [sb] to do [sth]"

            # Instance-Level Syntactic Frame Resolution (F1..F5):
            # Inspect the quote's syntactic dependency frame around the target verb BEFORE selecting an anchor:
            #   F1_TRIVALENT_NP_PP / F2_MONOTRANS_NP: Verb governs a direct object (`dobj`/`obj` or quoted-speech object) -> Transitive!
            #   F4_INTRANS_GOV_PREP: Verb has NO direct object and governs a preposition (`prep` outside quoted speech) -> Intransitive + Prep!
            #   F5_INTRANS_BARE_OR_ERGATIVE: Verb has neither `dobj` nor `prep` -> Ergative Inchoative (`S_patient + V`) or Max-PMI Fallback!
            if not anchor and it["quote"]:
                try:
                    doc = nlp(it["quote"])
                    v_collocs = cls.get_pedlex_collocations(w_lower, pos="verb")
                    consensus_preps = set(v_collocs.get("prep", []))
                    if sense_native_prep:
                        consensus_preps.add(sense_native_prep)
                    for tok in doc:
                        if not _tok_matches_target(tok, doc):
                            continue
                        # Detect quotation marks opened after `tok.i` (`print "Here's a pat on the back for You..."`)
                        # Tokens inside a quoted utterance after `tok` are NOT prepositional dependents of `tok`.
                        quote_open_i = next(
                            (t_q.i for t_q in doc[tok.i + 1 :] if t_q.text in ('"', "“", "”", "‘", "’") or (t_q.text == "'" and t_q.i == tok.i + 1)),
                            None,
                        )
                        quote_has_quoted_obj = bool(quote_open_i is not None and quote_open_i <= tok.i + 2)
                        dobj_children = [
                            c for c in tok.children
                            if c.dep_ in ("dobj", "obj") and (quote_open_i is None or c.i < quote_open_i)
                        ]
                        has_quote_dobj = bool(dobj_children) or quote_has_quoted_obj

                        # Collect post-verbal prepositional children outside quoted speech
                        verified_prep = None   # (token_index, prep, pobj) confirmed by OCD/sense valency
                        unverified_prep = None  # (token_index, prep, pobj) bare syntactic complement
                        verified_prep_has_gerund = False
                        _cand_prep_children = [ch for ch in tok.children if ch.dep_ == "prep"]
                        for _dc in dobj_children:
                            for _d_prep in _dc.children:
                                if _d_prep.dep_ == "prep" and _d_prep not in _cand_prep_children:
                                    _cand_prep_children.append(_d_prep)
                        for child in _cand_prep_children:
                            if quote_open_i is not None and child.i > quote_open_i:
                                continue
                            prep_tok = child.text.lower()
                            if prep_tok not in ("to", "with", "from", "on", "for", "in", "into", "of", "against", "at", "upon", "towards", "as", "under", "about", "over", "between", "by"):
                                continue
                            if child.i <= tok.i:
                                continue  # Post-head linear constraint
                            child_pobj = None
                            has_pcomp_vbg = any(pc.dep_ == "pcomp" or pc.tag_ == "VBG" for pc in child.children)
                            for pchild in child.children:
                                if pchild.dep_ in ("pobj", "pcomp") and pchild.lemma_.lower() not in cls._ANCHOR_STOPWORDS:
                                    child_pobj = pchild.lemma_.lower()
                                    break
                            if child_pobj is not None and (prep_tok, child_pobj) in cls._ADJUNCT_PREP_PAIRS:
                                continue  # Fixed adjunct / interjection (e.g. 'for example')
                            if (
                                child_pobj in (
                                    "morning", "afternoon", "evening", "night", "midnight", "noon", "day", "week",
                                    "weekend", "month", "year", "decade", "century", "hour", "minute", "moment",
                                    "sunset", "sunrise", "dawn", "dusk", "spring", "summer", "autumn", "fall", "winter",
                                    "o'clock", "oclock",
                                )
                                or (prep_tok == "at" and any(pc.pos_ == "NUM" for pc in child.children))
                            ) and prep_tok != sense_native_prep:
                                continue  # Temporal adjunct PP (e.g. 'on such a freezing morning', 'at six o'clock')
                            if prep_tok in consensus_preps and (child.head == tok or has_pcomp_vbg or prep_tok == sense_native_prep or prep_tok != "of"):
                                if verified_prep is None:
                                    verified_prep = (child.i, prep_tok, child_pobj)
                                    verified_prep_has_gerund = has_pcomp_vbg
                            elif child.head == tok and unverified_prep is None:
                                unverified_prep = (child.i, prep_tok, child_pobj)

                        if has_quote_dobj:
                            # F1 (`V + NP + PP`) or F2 (`V + NP`): Instance is TRANSITIVE.
                            quote_is_transitive = True
                            if verified_prep is not None and (verified_prep_has_gerund or verified_prep[1] == sense_native_prep):
                                anchor, prep_obj = verified_prep[1], verified_prep[2]
                                anchor_type = "prep"
                                quote_governed_prep = verified_prep[1]
                            else:
                                for child in dobj_children:
                                    if (
                                        child.pos_ in ("NOUN", "PROPN")
                                        and child.lemma_.lower() not in cls._ANCHOR_STOPWORDS
                                        and not cls._in_forbidden_word_set(child.lemma_.lower(), batch_target_words)
                                    ):
                                        obj_form = child.text.lower() if child.tag_ == "NNS" else child.lemma_.lower()
                                        anchor = obj_form
                                        anchor_type = "object"
                                        break
                                if verified_prep is not None:
                                    quote_governed_prep = verified_prep[1]
                                    # If `dobj` was a human pronoun (`warned him against`) and the sense explicitly has a `V + sb + prep` frame,
                                    # allow the verified preposition if no lexical object was extracted from the quote.
                                    if not anchor and any(
                                        c.pos_ == "PRON" and c.lemma_.lower() in ("he", "him", "she", "her", "me", "us", "you", "they", "them", "someone", "somebody")
                                        for c in dobj_children
                                    ):
                                        anchor, prep_obj = verified_prep[1], verified_prep[2]
                                        anchor_type = "prep"
                        else:
                            # F4 (`V + PP`) vs F5 (`S_patient + V` ergative inchoative):
                            chosen = None
                            if verified_prep is not None and unverified_prep is not None:
                                chosen = verified_prep if verified_prep[0] < unverified_prep[0] else unverified_prep
                            elif verified_prep is not None:
                                chosen = verified_prep
                            elif unverified_prep is not None:
                                chosen = unverified_prep

                            _generic_human_subjs = {
                                "boy", "girl", "man", "woman", "person", "people", "child", "children",
                                "student", "teacher", "volunteer", "worker", "soldier", "player", "friend",
                                "doctor", "parent", "father", "mother", "son", "daughter", "guy", "kid",
                                "member", "leader", "author", "writer", "reader", "speaker", "listener",
                                "user", "customer", "villager", "passenger", "driver", "guest", "visitor",
                            }
                            subj_candidates = [c for c in tok.children if c.dep_ in ("nsubj", "nsubjpass") and c.i < tok.i]
                            if not subj_candidates and tok.dep_ == "xcomp" and tok.head.pos_ in ("VERB", "AUX") and tok.head.lemma_.lower() in (
                                "begin", "start", "continue", "cease", "tend", "fail", "seem", "appear",
                                "refuse", "manage", "threaten", "used", "going", "supposed", "expected",
                            ):
                                subj_candidates = [c for c in tok.head.children if c.dep_ in ("nsubj", "nsubjpass") and c.i < tok.head.i]
                            patient_subj_form: Optional[str] = None
                            for child in subj_candidates:
                                c_lem = child.lemma_.lower()
                                if (
                                    child.pos_ in ("NOUN", "PROPN")
                                    and child.ent_type_ != "PERSON"
                                    and c_lem not in cls._ANCHOR_STOPWORDS
                                    and c_lem not in _generic_human_subjs
                                    and not cls._in_forbidden_word_set(c_lem, batch_target_words)
                                ):
                                    c_supers = cls.get_wordnet_supersenses(c_lem, pos="n")
                                    if c_supers and all(ss == "noun.person" for ss in c_supers):
                                        continue
                                    patient_subj_form = child.text.lower() if child.tag_ == "NNS" else c_lem
                                    break

                            s_def_low_erg = str((locked_sense or {}).get("definition") or it.get("definition") or "").lower().strip()
                            s_gram_low_v = str((locked_sense or {}).get("grammar") or "").lower()
                            _has_have_aux = any(c.dep_ == "aux" and c.lemma_.lower() == "have" for c in tok.children)
                            is_quote_passive = (
                                any(c.dep_ in ("auxpass", "nsubjpass") for c in tok.children)
                                or (tok.tag_ == "VBN" and not _has_have_aux)
                            )
                            is_strictly_trans_sense = (
                                "transitive" in s_gram_low_v
                                and "intransitive" not in s_gram_low_v.replace("transitive", "")
                            )
                            is_ergative_inchoative_sense = bool(
                                not is_strictly_trans_sense
                                and not is_quote_passive
                                and (
                                    re.match(
                                        r"^(?:\[[^\]]+\]\s*)*(?:if\s+(?:a|an|the|something|somebody's|someone's|your)\b|to\s+(?:become|grow|fall|rise|sink|shrink|expand|disappear|appear|cease|stop|begin\s+to\s+work)\b)",
                                        s_def_low_erg,
                                    )
                                    or re.search(
                                        r"\bor\s+if\s+(?:it|they|something|a|an)\s+[a-z]+s\b|\bor\s+to\s+become\s+[a-z]+\b",
                                        s_def_low_erg,
                                    )
                                    or (
                                        "intransitive" in s_gram_low_v
                                        and chosen is None
                                    )
                                    or w_lower in (
                                        "drop", "rise", "fall", "melt", "freeze", "boil", "evaporate", "condense",
                                        "shrink", "expand", "collapse", "explode", "burst", "crack", "shatter",
                                        "fade", "vanish", "disappear", "emerge", "subside", "open", "close",
                                    )
                                )
                            )
                            if patient_subj_form and is_ergative_inchoative_sense and (
                                chosen is None or chosen[1] != sense_native_prep
                            ):
                                anchor = patient_subj_form
                                anchor_type = "verb_subject"
                            elif patient_subj_form and is_quote_passive and (
                                chosen is None or chosen[1] != sense_native_prep
                            ):
                                # Deep Passive Projection: in passive voice (`[Patient] be V-ed`),
                                # `patient_subj_form` (`nsubjpass`) is the underlying direct object (`受动者`), NOT `verb_subject`.
                                anchor = patient_subj_form
                                anchor_type = "object"
                            elif chosen is not None and not (
                                is_quote_passive
                                and is_strictly_trans_sense
                                and verified_prep is None
                                and str(raw_w_item or w_lower).strip().lower() == w_lower
                            ):
                                anchor, prep_obj = chosen[1], chosen[2]
                                anchor_type = "prep"
                                quote_governed_prep = anchor
                        break
                except Exception:
                    pass

            # Verb Step B3: Fallback to sense-aligned direct noun object (from locked sense examples + OCD) or sense-native prep
            if not anchor:
                has_sense_inf_or_gerund = bool(
                    not quote_is_transitive
                    and initial_atomic_bundle
                    and (initial_atomic_bundle.get("inf_valency") or initial_atomic_bundle.get("gerund_valency"))
                    and initial_atomic_bundle.get("example")
                )
                if not has_sense_inf_or_gerund:
                    sense_ex_dobj_nouns: List[str] = []
                    if locked_sense and locked_sense.get("examples"):
                        try:
                            for s_ex_str in (locked_sense.get("examples") or []):
                                doc_sex = nlp(str(s_ex_str))
                                for t_sex in doc_sex:
                                    if _tok_matches_target(t_sex, doc_sex):
                                        is_sex_passive = any(c.dep_ in ("auxpass", "nsubjpass") for c in t_sex.children)
                                        for ch_sex in t_sex.children:
                                            # Deep Passive Projection: include `nsubjpass` when the dictionary example is passive
                                            # so the underlying Patient ('受动者') is extracted as a direct object candidate.
                                            if (
                                                (ch_sex.dep_ in ("dobj", "obj") or (is_sex_passive and ch_sex.dep_ == "nsubjpass"))
                                                and ch_sex.pos_ in ("NOUN", "PROPN")
                                                and ch_sex.ent_type_ != "PERSON"
                                                and ch_sex.lemma_.lower() not in cls._ANCHOR_STOPWORDS
                                                and not cls._in_forbidden_word_set(ch_sex.lemma_.lower(), batch_target_words)
                                            ):
                                                n_f = ch_sex.text.lower() if ch_sex.tag_ == "NNS" else ch_sex.lemma_.lower()
                                                if n_f not in sense_ex_dobj_nouns:
                                                    sense_ex_dobj_nouns.append(n_f)
                        except Exception:
                            pass
                    v_entry = cls.get_pedlex_collocations(w_lower, pos="verb")
                    ped_nouns = v_entry.get("colloc_nouns", []) + v_entry.get("noun_after", [])
                    candidates = sense_ex_dobj_nouns + [n for n in ped_nouns if n not in sense_ex_dobj_nouns]
                    if candidates:
                        anchor = cls._select_sense_aligned_anchor(candidates, definition=it.get("definition"), quote=it.get("quote"), target_word=w_lower, forbidden=batch_target_words)
                        if not anchor and sense_ex_dobj_nouns:
                            anchor = sense_ex_dobj_nouns[0]
                        if anchor:
                            anchor_type = "object"
                if not anchor and sense_native_prep and not cls._in_forbidden_word_set(sense_native_prep, batch_target_words):
                    anchor = sense_native_prep
                    anchor_type = "prep"

        elif canonical_pos == "noun":
            n_entry = cls.get_pedlex_collocations(w_lower, pos="noun")
            q_low_n = (it.get("quote") or "").lower()
            # Family 5: Shell Nouns & Abstract Nouns Taking Post-Nominal Complements
            if not anchor and q_low_n and word_subcats & {"N_SHELL_TO_DO", "N_SHELL_OF_DOING", "N_SHELL_THAT_ONLY"}:
                _is_shell_to_inf = False
                if "N_SHELL_TO_DO" in word_subcats and re.search(rf"\b{re.escape(w_lower)}(?:s|es|ies)?\s+to\s+[a-z]+\b", q_low_n):
                    try:
                        doc_n_shell = nlp(it["quote"])
                        for tok_n in doc_n_shell:
                            if _tok_matches_target(tok_n, doc_n_shell) and tok_n.i + 2 < len(doc_n_shell):
                                nxt1 = doc_n_shell[tok_n.i + 1]
                                nxt2 = doc_n_shell[tok_n.i + 2]
                                if nxt1.text.lower() == "to" and (
                                    nxt1.tag_ == "TO"
                                    or nxt2.tag_ == "VB"
                                    or nxt2.pos_ in ("VERB", "AUX")
                                ) and nxt2.pos_ not in ("DET", "PRON", "PROPN", "NOUN", "NUM"):
                                    _is_shell_to_inf = True
                                    break
                    except Exception:
                        pass
                if _is_shell_to_inf:
                    anchor = "to"
                    anchor_type = "shell_to_do"
                    cxn_override_pat = f"{w_lower} to do [sth]"
                    cxn_override_frame = "... ____ to do [sth]"
                elif "N_SHELL_OF_DOING" in word_subcats and re.search(rf"\b{re.escape(w_lower)}(?:s|es|ies)?\s+of\s+(?:[a-z]+\s+)?[a-z]+ing\b", q_low_n):
                    anchor = "of"
                    anchor_type = "shell_of_doing"
                    cxn_override_pat = f"{w_lower} of [doing sth]"
                    cxn_override_frame = "... ____ of [doing sth]"
                elif "N_SHELL_THAT_ONLY" in word_subcats and re.search(rf"\b{re.escape(w_lower)}(?:s|es|ies)?\s+that\s+[a-z]+\b", q_low_n):
                    anchor = "that"
                    anchor_type = "shell_that"
                    cxn_override_pat = f"the {w_lower} that ..."
                    cxn_override_frame = "... the ____ that ..."
            if not anchor and it["quote"]:
                try:
                    doc = nlp(it["quote"])
                    ocd_prep_tokens = set()
                    for p in n_entry.get("prep", []):
                        cleaned = re.sub(r'[\(\)]', '', p).lower()
                        for prep_tok in cleaned.replace("/", " ").split():
                            if prep_tok in ("to", "with", "from", "on", "for", "in", "into", "of", "against", "at", "upon", "towards", "toward", "over", "under", "about"):
                                ocd_prep_tokens.add(prep_tok)
                    if locked_sense is not None:
                        for eb in cls.extract_sense_pattern_bundles(locked_sense, w_lower):
                            if eb.get("prep"):
                                ocd_prep_tokens.add(eb["prep"])
                    else:
                        for lp in cls.get_ldoce_preps(w_lower):
                            ocd_prep_tokens.add(lp)

                    for tok in doc:
                        if _tok_matches_target(tok, doc):
                            for child in tok.children:
                                if child.dep_ == "prep" and child.text.lower() in ("to", "with", "from", "on", "for", "in", "into", "of", "against", "at", "upon", "towards", "toward", "over", "under", "about"):
                                    prep_word = child.text.lower()
                                    if ocd_prep_tokens and prep_word not in ocd_prep_tokens:
                                        continue
                                    pobj_tok = None
                                    for pchild in child.children:
                                        if pchild.dep_ == "pobj" and pchild.pos_ != "PRON" and pchild.lemma_.lower() not in cls._ANCHOR_STOPWORDS and pchild.lemma_.lower() != w_lower:
                                            pobj_tok = pchild.lemma_.lower()
                                            break
                                    if pobj_tok is not None and (prep_word, pobj_tok) in cls._ADJUNCT_PREP_PAIRS:
                                        continue
                                    anchor = prep_word
                                    anchor_type = "prep"
                                    prep_obj = pobj_tok
                                    quote_governed_prep = anchor
                                    break
                            if not anchor:
                                has_prep_or_bundle_ev = bool(
                                    sense_native_prep
                                    or (locked_sense and locked_sense.get("pattern_bundles"))
                                    or (n_entry and n_entry.get("prep"))
                                )
                                attested_adj_text = " ".join([
                                    " ".join(str(a) for a in (n_entry.get("adj") or [])),
                                    " ".join(str(u) for u in ((locked_sense or {}).get("units") or [])),
                                    " ".join(str(p) for p in ((locked_sense or {}).get("patterns") or [])),
                                    " ".join(str(ex) for ex in ((locked_sense or {}).get("examples") or [])),
                                ]).lower()
                                for child in tok.children:
                                    if child.dep_ in ("amod", "compound"):
                                        mod_tok = (
                                            child.text.lower()
                                            if child.dep_ == "compound" and child.tag_ in ("NNS", "NNPS")
                                            else child.lemma_.lower()
                                        )
                                        if (
                                            child.pos_ != "PRON"
                                            and mod_tok not in cls._ANCHOR_STOPWORDS
                                            and mod_tok not in cls.GENERIC_COLLOCATION_MODIFIERS
                                            and not cls._is_delexical_anchor(mod_tok, "adj", "noun")
                                            and mod_tok != w_lower
                                            and not cls._in_forbidden_word_set(mod_tok, batch_target_words)
                                            and (
                                                not has_prep_or_bundle_ev
                                                or bool(re.search(rf"\b{re.escape(mod_tok)}\b", attested_adj_text))
                                            )
                                        ):
                                            anchor = mod_tok
                                            anchor_type = "adj"
                                            break
                            if not anchor and tok.dep_ in ("dobj", "pobj", "attr"):
                                head_tok = tok.head
                                if head_tok.pos_ == "VERB":
                                    head_tok_lemma = head_tok.lemma_.lower()
                                    if (head_tok.pos_ != "PRON"
                                            and head_tok_lemma not in cls._ANCHOR_STOPWORDS
                                            and head_tok_lemma != w_lower
                                            and head_tok_lemma not in cls._DELEXICAL_VERBS
                                            and not cls._in_forbidden_word_set(head_tok_lemma, batch_target_words)):
                                        verb_before_set = {v.lower().split()[0] for v in n_entry.get("verb_before", [])}
                                        if not verb_before_set or head_tok_lemma in verb_before_set:
                                            anchor = head_tok_lemma
                                            anchor_type = "verb"
                            break
                except Exception:
                    pass

            # Priority D: dictionary candidates, strictly gated by quote/definition evidence
            if not anchor and n_entry:
                filtered_adj = [
                    a for a in (n_entry.get("adj") or [])
                    if str(a).strip().lower() not in cls.GENERIC_COLLOCATION_MODIFIERS
                ]
                sense_mod_cands = []
                if isinstance(locked_sense, dict):
                    for pat_s in (locked_sense.get("patterns") or []):
                        pat_toks = [t.lower() for t in re.findall(r"[a-z]+", str(pat_s)) if t.lower() not in ("a", "an", "the", "some", "any", "one")]
                        if len(pat_toks) == 2 and cls.form_matches(pat_toks[1], w_lower) and pat_toks[0] not in cls.GENERIC_COLLOCATION_MODIFIERS:
                            sense_mod_cands.append(pat_toks[0])
                if filtered_adj or sense_mod_cands or n_entry.get("verb_before"):
                    candidates = sense_mod_cands + filtered_adj + n_entry.get("verb_before", [])
                    anchor = cls._select_sense_aligned_anchor(candidates, definition=it.get("definition"), quote=it.get("quote"), target_word=w_lower, forbidden=batch_target_words)
                    if anchor:
                        anchor_type = "adj" if cls._token_in_phrases(anchor, sense_mod_cands + filtered_adj) else "verb"
                        if cls._is_delexical_anchor(anchor, anchor_type, "noun"):
                            anchor, anchor_type = None, "collocation"
                if not anchor and sense_native_prep and not cls._in_forbidden_word_set(sense_native_prep, batch_target_words):
                    anchor = sense_native_prep
                    anchor_type = "prep"
                if not anchor and n_entry.get("prep"):
                    p_cand = cls._pick_anchor_token(n_entry["prep"], target_word=w_lower, forbidden=batch_target_words)
                    if p_cand:
                        anchor = p_cand
                        anchor_type = "prep"

            # Priority E (B2): the partitive compound the passage itself shows.
            if not anchor and it["quote"]:
                pt_anchor, pt_type, pt_head = cls._partitive_compound_anchor(
                    w_lower,
                    it["quote"],
                    forbidden=batch_target_words
                )
                if pt_anchor:
                    anchor, anchor_type, prep_obj = pt_anchor, pt_type, pt_head

        elif canonical_pos == "adj":
            adj_locked_gram = str((locked_sense or {}).get("grammar") or "").lower()
            adj_prof = cls.get_word_gram_profile(w_lower, pos="adj")
            is_not_before_noun_adj = (
                "not before noun" in adj_locked_gram
                or "never before noun" in adj_locked_gram
                or bool(adj_prof.get("not_before_noun"))
            )
            q_low_adj = (it.get("quote") or "").lower()
            # Family 8: Mandative Subjunctive Adjective ('It is essential/vital/imperative that [sb] (should) [V-base]')
            if not anchor and "ADJ_MANDATIVE_SUBJ" in word_subcats and q_low_adj:
                if re.search(rf"\bit\s+(?:is|was)\s+(?:not\s+|very\s+|quite\s+)?{re.escape(w_lower)}\s+that\b", q_low_adj):
                    anchor = "that"
                    anchor_type = "mandative_subj"
                    cxn_override_pat = f"it is {w_lower} that [sb] (should) [V-base]"
                    cxn_override_frame = "It is ____ that [sb] (should) [V-base]"
            # Family 1B: Subject Extraposition 'It is <adj> (for/of [sb]) to do [sth]' & For-To Infinitival Guard
            # Never extract 'for' or 'of' inside 'It is <adj> for/of sb to do' as a lexical preposition anchor ('*necessary for [sth]').
            if not anchor and q_low_adj:
                m_it_adj_inf = re.search(
                    rf"\bit\s+(?:is|was|seems|appears|becomes|gets|'s|’s|would\s+be|will\s+be|can\s+be|could\s+be)\s+(?:not\s+|very\s+|quite\s+|extremely\s+|particularly\s+|especially\s+|too\s+|so\s+)?{re.escape(w_lower)}\b(?:[^.?!;]*?\b(for|of)\s+[^.?!;]+?)?\s+to\s+[a-z]+\b",
                    q_low_adj,
                )
                if m_it_adj_inf:
                    anchor = "to"
                    anchor_type = "it_extrapos"
                    for_or_of = (m_it_adj_inf.group(1) or "").lower()
                    if for_or_of in ("for", "of"):
                        cxn_override_pat = f"it is {w_lower} {for_or_of} [sb] to do [sth]"
                        cxn_override_frame = f"It is ____ {for_or_of} [sb] to do [sth]"
                    else:
                        cxn_override_pat = f"it is {w_lower} to do [sth]"
                        cxn_override_frame = "It is ____ to do [sth]"
            # Family 1C: Finite `that`-clause Extraposition ('It is <adj> that ...' or coordinated 'It is ... and/but <adj> that ...')
            _it_that_extrapos_adjs = {
                "likely", "unlikely", "clear", "obvious", "possible", "impossible",
                "probable", "improbable", "certain", "uncertain", "true", "evident",
                "apparent", "natural", "strange", "odd", "surprising", "fortunate", "unfortunate",
            }
            if not anchor and q_low_adj and ("ADJ_IT_EXTRAPOS" in word_subcats or w_lower in _it_that_extrapos_adjs):
                if re.search(
                    rf"\bit\s*(?:is|was|seems|appears|'s|’s|would\s+be|will\s+be)\b[^.?!;]*?\b{re.escape(w_lower)}\s+that\b",
                    q_low_adj,
                ):
                    anchor = "that"
                    anchor_type = "it_extrapos"
                    cxn_override_pat = f"it is {w_lower} that ..."
                    cxn_override_frame = "It is ____ that ..."
            # Pre-fetch PedLex entry & attested prepositions so Adjective Step A1 never mistakes a clause-level
            # temporal/circumstantial adjunct (e.g. 'remained unsolved for 1,400 years') for a bound adjective preposition.
            adj_entry = cls.get_pedlex_collocations(w_lower, pos="adj")
            attested_adj_preps = set()
            if sense_native_prep:
                attested_adj_preps.add(sense_native_prep.lower())
            if adj_entry and adj_entry.get("prep"):
                for raw_p in adj_entry["prep"]:
                    raw_p_clean = re.sub(r"\[.*?\]|\(.*?\)", " ", str(raw_p).lower())
                    for m_ap in re.finditer(
                        rf"\b{re.escape(w_lower)}\s+(to|for|with|of|in|from|at|about|on|by|into|towards|against|over)\b",
                        raw_p_clean,
                    ):
                        attested_adj_preps.add(m_ap.group(1))
            if locked_sense:
                for b in cls.extract_sense_pattern_bundles(locked_sense, w_lower):
                    for p_tok in (b.get("preps") or []):
                        attested_adj_preps.add(p_tok.lower())
            else:
                for p_tok in cls.get_ldoce_preps(w_lower, pos="adj"):
                    attested_adj_preps.add(p_tok.lower())
            for p_key, adj_set in cls._ADJ_PREP_VALENCY_MAP.items():
                if w_lower in adj_set:
                    attested_adj_preps.add(p_key)
            if w_lower in {
                "inaudible", "audible", "invisible", "visible", "imperceptible",
                "immune", "allergic", "susceptible", "proportional", "akin",
                "analogous", "adjacent", "prior", "subsequent", "preferable",
                "superior", "inferior", "identical", "relevant", "loyal",
                "secondary", "accustomed",
            }:
                attested_adj_preps.add("to")

            # Adjective Step A: Syntactic Dependency Analysis on Authentic Quote
            if not anchor and it["quote"]:
                try:
                    doc = nlp(it["quote"])
                    _temporal_pobj_lemmas = {
                        "year", "month", "week", "day", "hour", "minute", "second",
                        "decade", "century", "millennium", "period", "time", "while", "moment", "age", "era"
                    }
                    q_start_i, q_end_i, tok = cls._find_target_span_in_doc(
                        doc, w_lower, base_str=w_lower, t_forms=target_inflections_set
                    )
                    if tok is not None and q_start_i is not None and q_end_i is not None:
                        eff_head = tok
                        while eff_head.dep_ == "conj" and eff_head.head and eff_head.head.pos_ == "ADJ" and eff_head.head.i < eff_head.i:
                            eff_head = eff_head.head
                        # A1. Predicative Valency: Check if adjective governs a lexicographically attested bound preposition
                        for child in tok.children:
                            c_low = child.text.lower()
                            if child.dep_ == "prep" and child.i >= q_end_i and c_low in ("to", "for", "with", "of", "in", "from", "at", "about", "on"):
                                if c_low not in attested_adj_preps:
                                    continue
                                pobj_tokens = [pc for pc in child.children if pc.dep_ == "pobj"]
                                if pobj_tokens and any(
                                    pc.ent_type_ in ("DATE", "TIME") or pc.lemma_.lower() in _temporal_pobj_lemmas
                                    for pc in pobj_tokens
                                ):
                                    continue
                                anchor = c_low
                                anchor_type = "prep"
                                quote_governed_prep = anchor
                                for pchild in pobj_tokens:
                                    if pchild.pos_ != "PRON" and pchild.text.lower() not in cls._ANCHOR_STOPWORDS and pchild.lemma_.lower() != w_lower:
                                        prep_obj = pchild.lemma_.lower()
                                        break
                                break
                        # A2. Attributive Modification: Check if adjective modifies a head noun (dep_ == "amod" / "compound")
                        if not anchor and not is_not_before_noun_adj:
                            head_n = None
                            if eff_head.dep_ in ("amod", "compound") and eff_head.head and eff_head.head.pos_ in ("NOUN", "PROPN") and q_end_i <= eff_head.head.i:
                                head_n = eff_head.head
                            elif q_end_i < len(doc) and doc[q_end_i].pos_ in ("NOUN", "PROPN"):
                                head_n = doc[q_end_i]
                            elif eff_head.dep_ in ("amod", "compound") and q_start_i > 0 and doc[q_start_i - 1].pos_ == "DET":
                                # Quoted nominalized expression preceded by determiner (e.g. 'a simple "Hello"')
                                quote_adj_is_attributive = True
                            if head_n is not None:
                                quote_adj_is_attributive = True
                                if (
                                    head_n.pos_ != "PRON"
                                    and head_n.text.lower() not in cls._ANCHOR_STOPWORDS
                                    and head_n.lemma_.lower() != w_lower
                                    and not cls._in_forbidden_word_set(head_n.lemma_.lower(), batch_target_words)
                                ):
                                    anchor = head_n.lemma_.lower()
                                    anchor_type = "modified_noun"
                        # A3. Predicative / Causative Complement: Check if adjective is complement of a causative/linking verb
                        if not anchor and eff_head.dep_ in ("ccomp", "xcomp", "acomp", "oprd") and eff_head.head and eff_head.head.pos_ in ("VERB", "AUX"):
                            head_v = eff_head.head.lemma_.lower()
                            if head_v in (
                                "make", "render", "find", "deem", "keep", "consider",
                                "feel", "sound", "seem", "remain", "stay", "become", "grow", "turn", "prove", "look", "appear",
                            ):
                                anchor = head_v
                                anchor_type = "verb_copula"
                except Exception:
                    pass

            # Adjective Step B: Sense-Native Preposition Valency & Collocation Lexicon Lookup (OCD) if no anchor found in quote
            if not anchor and not quote_adj_is_attributive and sense_native_prep and not cls._in_forbidden_word_set(sense_native_prep, batch_target_words):
                anchor = sense_native_prep
                anchor_type = "prep"
            if not anchor and adj_entry:
                if not is_not_before_noun_adj:
                    # Use `colloc_nouns` (NEVER `noun_after`, which stores `colls["phrases"]` like "keep silent")
                    candidates = list(adj_entry.get("colloc_nouns", []))
                    if candidates:
                        anchor = cls._select_sense_aligned_anchor(candidates, definition=it.get("definition"), quote=it.get("quote"), target_word=w_lower, forbidden=batch_target_words)
                        if anchor:
                            anchor_type = "modified_noun"
                if not anchor and not quote_adj_is_attributive and sense_native_prep and not cls._in_forbidden_word_set(sense_native_prep, batch_target_words):
                    anchor = sense_native_prep
                    anchor_type = "prep"
                if not anchor and not quote_adj_is_attributive and adj_entry.get("prep"):
                    for raw_p in adj_entry["prep"]:
                        raw_p_clean = re.sub(r"\[.*?\]|\(.*?\)", " ", str(raw_p).lower())
                        m_ap = re.search(
                            rf"\b{re.escape(w_lower)}\s+(to|for|with|of|in|from|at|about|on|by|into|towards|against|over)\b",
                            raw_p_clean,
                        )
                        if m_ap and not cls._in_forbidden_word_set(m_ap.group(1), batch_target_words):
                            anchor = m_ap.group(1)
                            anchor_type = "prep"
                            break

            # Adjective Step C: Fall back to locked sense pattern first, then curated academic adjective preposition valency map
            if not anchor and not quote_adj_is_attributive:
                ld_entry = target_entry or cls.get_ldoce_entry(w_lower)
                if ld_entry:
                    candidate_senses = [locked_sense] if locked_sense else ld_entry.get("senses", [])
                    for s in candidate_senses:
                        if not isinstance(s, dict):
                            continue
                        for pat in s.get("patterns", []):
                            pat_low_c = re.sub(r"\[.*?\]|\(.*?\)", " ", str(pat).lower())
                            # For-To Infinitival Guard: do not extract 'for'/'of'/'to' as a lexical prep from 'it is <adj> (for/of sb) to do'
                            if re.search(rf"\bit\s+is\s+{re.escape(w_lower)}\b.*\bto\b", pat_low_c):
                                continue
                            m_adj_prep = re.search(
                                rf"\b{re.escape(w_lower)}\s+(of|to|for|with|about|from|in|on|at)\b",
                                pat_low_c,
                            )
                            if m_adj_prep:
                                p = m_adj_prep.group(1)
                                if p == "to":
                                    m_inf = re.search(r"\bto\s+([a-z]+)", pat_low_c)
                                    if m_inf and m_inf.group(1) not in (
                                        "sb", "sth", "someone", "something", "oneself", "himself", "herself", "them", "the", "a", "an"
                                    ):
                                        continue
                                anchor = p
                                anchor_type = "prep"
                                break
                        if anchor:
                            break
                if not anchor:
                    for p, adjs in cls._ADJ_PREP_VALENCY_MAP.items():
                        if w_lower in adjs:
                            anchor = p
                            anchor_type = "prep"
                            break

        elif canonical_pos in ("adv", "function_word", "phrase"):
            q_low_adv = (it.get("quote") or "").lower()
            # Family 7: Fronted Negative Inversion ('Seldom/Rarely/Never/Under no circumstances + Aux + Subj + V')
            if not anchor and word_subcats & {"ADV_FRONTED_INV", "PHR_FRONTED_INV"}:
                if canonical_pos == "phrase" or (
                    q_low_adv
                    and re.search(
                        rf"(?:^|[.;!?]\s*|\b(?:and|but)\s+){re.escape(w_lower)}\s+(?:do|does|did|have|has|had|am|is|are|was|were|can|could|will|would|shall|should|may|might|must)\b",
                        q_low_adv,
                    )
                ):
                    anchor = "inversion"
                    anchor_type = "fronted_inv"
                    cxn_override_pat = f"{w_lower} + Aux + Subj + V"
                    cxn_override_frame = "____ + Aux + Subj + V"
            # Family 7: Negative Polarity Adverb/Phrase ('remotely', 'whatsoever', 'at all', 'lift a finger', etc.)
            if not anchor and "NPI_ITEM" in word_subcats:
                if canonical_pos == "phrase" or (
                    q_low_adv
                    and re.search(
                        r"\b(?:not|n't|never|hardly|scarcely|barely|seldom|rarely|little|few|no|neither|nor|without|doubt|deny|refused?|failed?|unlikely|impossible)\b",
                        q_low_adv,
                    )
                ):
                    anchor = "not"
                    anchor_type = "npi"
                    cxn_override_pat = f"not/never ... {w_lower}"
                    cxn_override_frame = "... not/never ... ____"
            # Adverb Syntactic Dependency Walker & Sense-Lock Gate
            if not anchor and canonical_pos == "adv" and it.get("quote"):
                _discourse_conjunct_advs = {
                    "similarly", "likewise", "conversely", "otherwise", "instead", "meanwhile",
                    "moreover", "furthermore", "however", "nevertheless", "nonetheless", "therefore",
                    "consequently", "thus", "hence", "accordingly",
                }
                _temporal_stance_advs = {
                    "nowadays", "currently", "recently", "lately", "previously", "formerly",
                    "originally", "initially", "eventually", "ultimately", "finally", "traditionally",
                    "thereafter",
                }
                if w_lower in _discourse_conjunct_advs:
                    anchor_type = "discourse"
                elif w_lower not in _temporal_stance_advs:
                    try:
                        doc = nlp(it["quote"])
                        for tok in doc:
                            if _tok_matches_target(tok, doc):
                                if tok.i == 0 and tok.i + 1 < len(doc) and doc[tok.i + 1].text == ",":
                                    anchor_type = "discourse"
                                    break
                                if (
                                    tok.i + 1 < len(doc)
                                    and doc[tok.i + 1].pos_ == "ADJ"
                                    and doc[tok.i + 1].lemma_.lower() not in cls._ANCHOR_STOPWORDS
                                    and doc[tok.i + 1].lemma_.lower() != w_lower
                                ):
                                    anchor_type = "modifies_adj"
                                    if not doc[tok.i + 1].text[:1].isupper():
                                        anchor = doc[tok.i + 1].lemma_.lower()
                                    break
                                head = tok.head
                                if (
                                    head
                                    and abs(tok.i - head.i) <= 4
                                    and head.lemma_.lower() not in cls._ANCHOR_STOPWORDS
                                    and head.lemma_.lower() != w_lower
                                ):
                                    h_lem = head.lemma_.lower()
                                    if head.pos_ in ("ADJ", "ADV"):
                                        anchor_type = "modifies_adj"
                                        if not (head.text[:1].isupper() and head.i > 0):
                                            anchor = h_lem
                                        break
                                    elif head.pos_ == "VERB":
                                        anchor = h_lem
                                        anchor_type = "modifies_verb"
                                        break
                    except Exception:
                        pass

            # Priority B: Dictionary lookup from PedLex zero collision anchor
            if not anchor and canonical_pos == "adv" and anchor_type != "discourse":
                ld_anc, ld_atype, ld_frame, _ = cls.find_ldoce_zero_collision_anchor(
                    target_word=w_lower,
                    pos="adv",
                    definition=it.get("definition"),
                    quote=it.get("quote"),
                    locked_sense=locked_sense
                )
                if ld_anc and not (anchor_type == "modifies_adj" and ld_atype == "modifies_verb"):
                    aligned = cls._select_sense_aligned_anchor(
                        [ld_anc],
                        definition=it.get("definition"),
                        quote=it.get("quote"),
                        target_word=w_lower,
                        forbidden=batch_target_words
                    )
                    if aligned:
                        anchor = aligned
                        anchor_type = anchor_type or ld_atype or "modifies_verb"

        _cxn_anchor_types = {
            "it_extrapos", "gerund", "to_inf", "sb_to_do",
            "shell_to_do", "shell_of_doing", "shell_that",
            "correl_pair", "fronted_inv", "npi", "mandative_subj", "lv_triad", "v_obj_pp"
        }

        # 1.5 Sense-Native Atomic Pattern Anchor Inheritance (GEMINI.md §1.10):
        # If Step 1 found no anchor (or only a weak fallback) and the locked sense's
        # atomic pattern bundle specifies a bound preposition, inherit it BEFORE Step 3
        # distractor generation so double-key clearance runs against the true anchor.
        if not is_idiom_slot and sense_native_prep and anchor_type not in _cxn_anchor_types and (not anchor or anchor_type == "collocation"):
            if not cls._in_forbidden_word_set(sense_native_prep, batch_target_words):
                anchor = sense_native_prep
                anchor_type = "prep"
                pattern_anchor = sense_native_prep

        # 2. Syntactic Fallback: Extract anchor from quoted sentence if OCD / prep lookup was empty
        if not anchor and anchor_type != "discourse" and it["quote"]:
            q_clean = it["quote"].strip().strip('"').strip("'").strip("“").strip("”")
            try:
                doc = nlp(q_clean)
                for tok in doc:
                    if _tok_matches_target(tok, doc):
                        for child in tok.children:
                            if child.dep_ in ("dobj", "pobj") and child.pos_ != "PRON" and child.lemma_.lower() not in cls._ANCHOR_STOPWORDS and child.lemma_.lower() != w_lower:
                                anchor = child.lemma_.lower()
                                anchor_type = "contextual"
                                break
                            elif child.dep_ == "prep":
                                for pchild in child.children:
                                    if pchild.dep_ == "pobj" and pchild.pos_ != "PRON" and pchild.lemma_.lower() not in cls._ANCHOR_STOPWORDS and pchild.lemma_.lower() != w_lower:
                                        anchor = pchild.lemma_.lower()
                                        anchor_type = "contextual"
                                        break
                        if not anchor and tok.head and tok.head.pos_ in ("NOUN", "VERB") and tok.dep_ not in ("conj", "parataxis", "advcl", "ccomp"):
                            h_lem = tok.head.lemma_.lower()
                            if tok.head.pos_ != "PRON" and h_lem not in cls._ANCHOR_STOPWORDS and h_lem not in cls._DELEXICAL_VERBS and h_lem != w_lower:
                                anchor = h_lem
                                anchor_type = "contextual"
                        break
            except Exception:
                pass


        ctx.anchor = anchor
        ctx.anchor_type = anchor_type
        ctx.anchor_source = anchor_source
        ctx.prep_obj = prep_obj
        ctx.quote_adj_is_attributive = quote_adj_is_attributive
        ctx.quote_is_transitive = quote_is_transitive
        ctx.quote_governed_prep = quote_governed_prep
        ctx.cxn_override_pat = cxn_override_pat
        ctx.cxn_override_frame = cxn_override_frame
        ctx.pattern_anchor = pattern_anchor
        if "attested_adj_preps" in locals():
            ctx.attested_adj_preps = attested_adj_preps
