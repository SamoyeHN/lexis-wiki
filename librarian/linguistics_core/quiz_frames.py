"""
Verb patient ontology classification and cloze slot frame derivation.
"""

import datetime
import hashlib
import json
import os
import re
import sqlite3
import threading
import zlib
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union


class QuizFramesMixin:
    @classmethod
    def _resolve_object_animacy_from_example(
        cls,
        example_text: Optional[str],
        target_word: str,
        quote: Optional[str] = None,
        default_tag: str = "[sb/sth]",
    ) -> str:
        """
        Resolves object animacy (`someone`, `something`, or `oneself`) directly from the winning example sentence
        (falling back to `quote`) so `cloze_slot_frame` never emits slash choices like `someone/something`,
        `him/her/them`, or `it/them`.
        """
        tag_l = (default_tag or "").strip().lower()
        if tag_l in ("[sb]", "sb", "somebody", "someone"):
            return "someone"
        if tag_l in ("[sth]", "sth", "something"):
            return "something"

        reflexive_pronouns = {
            "himself", "herself", "themselves", "ourselves", "yourself", "yourselves", "oneself", "myself",
        }
        human_pronouns = {
            "him", "her", "me", "us", "you", "whom", "who",
            "someone", "somebody", "anyone", "anybody", "everyone", "everybody", "nobody",
        }
        human_nouns = {
            "person", "people", "man", "men", "woman", "women", "child", "children", "kid", "boy", "girl",
            "student", "teacher", "professor", "worker", "employee", "employer", "staff", "colleague",
            "customer", "client", "patient", "voter", "reader", "viewer", "listener", "audience", "guest",
            "friend", "family", "parent", "mother", "father", "son", "daughter", "brother", "sister",
            "citizen", "resident", "victim", "suspect", "witness", "prisoner", "leader", "manager",
            "director", "president", "governor", "minister", "official", "politician", "candidate",
            "soldier", "troop", "army", "enemy", "opponent", "rival", "partner", "ally", "neighbor",
            "neighbour", "stranger", "visitor", "tourist", "passenger", "driver", "pilot", "doctor",
            "nurse", "lawyer", "judge", "police", "officer", "author", "writer", "poet", "artist",
            "actor", "actress", "singer", "musician", "player", "athlete", "fan", "crowd", "public",
            "community", "population", "generation", "youth", "adult", "teenager", "baby", "infant",
        }

        nlp = cls.get_spacy()
        tw_l = (target_word or "").strip().lower()
        tw_last = tw_l.split()[-1] if tw_l else ""

        for src_txt in (example_text, quote):
            if not src_txt or not nlp:
                continue
            try:
                doc = nlp(src_txt)
                for tok in doc:
                    if tok.text.lower() == tw_last or tok.lemma_.lower() == tw_last:
                        # Check fronted human wh/relative pronoun before the verb/preposition (e.g. 'Guess who I ran into in town today!')
                        if any(pt.text.lower() in ("who", "whom") for pt in doc[:tok.i]):
                            return "someone"
                        # Inspect the 1..4 tokens following the target word/particle
                        window = doc[tok.i + 1 : min(len(doc), tok.i + 5)]
                        for w_tok in window:
                            if w_tok.is_punct:
                                break
                            w_low = w_tok.text.lower()
                            w_lem = w_tok.lemma_.lower()
                            if w_low in reflexive_pronouns:
                                return "oneself"
                            if w_low in human_pronouns or w_lem in human_nouns or w_tok.ent_type_ == "PERSON":
                                return "someone"
                            if w_low in ("it", "this", "that", "which", "what") or (
                                w_tok.pos_ == "NOUN" and w_lem not in human_nouns
                            ):
                                return "something"
            except Exception:
                pass
        return "something"

    PHYSICAL_SUPERSENSES = {
        "noun.artifact", "noun.object", "noun.substance", "noun.body",
        "noun.animal", "noun.plant", "noun.food",
    }
    PERSON_SUPERSENSES = {"noun.person", "noun.group"}
    ABSTRACT_EVENT_SUPERSENSES = {
        "noun.state", "noun.event", "noun.act", "noun.feeling",
        "noun.cognition", "noun.phenomenon", "noun.process",
        "noun.communication", "noun.attribute", "noun.relation",
        "noun.time", "noun.possession",
    }

    CRISIS_KEYWORDS = {
        "conflict", "crisis", "tension", "war", "violence", "fighting",
        "protest", "clash", "rebellion", "strike", "dispute", "quarrel",
        "scandal", "threat", "danger", "hostility", "aggression", "revolt",
        "riot", "rioting",
    }
    PROBLEM_KEYWORDS = {
        "problem", "difficulty", "issue", "challenge", "trouble", "dilemma",
        "obstacle", "hardship", "barrier", "disagreement", "dispute", "controversy",
    }
    PROCESS_KEYWORDS = {
        "process", "activity", "negotiation", "negotiations", "discussion",
        "discussions", "campaign", "investigation", "reform", "development",
        "change", "procedure", "learning", "trade", "cooperation", "talks",
    }
    FEELING_KEYWORDS = {
        "pain", "symptom", "symptoms", "anxiety", "fear", "anger", "stress",
        "grief", "suffering", "loneliness", "panic", "burden", "feeling",
        "feelings", "illness", "depression", "guilt", "distress",
    }
    ASSET_KEYWORDS = {
        "money", "fund", "funds", "capital", "investment", "budget", "tax",
        "taxes", "grant", "grants", "resource", "resources", "wealth", "revenue",
    }
    OPINION_KEYWORDS = {
        "opinion", "view", "views", "idea", "concern", "concerns", "support",
        "criticism", "doubt", "doubts", "dissatisfaction", "objection", "objections", "intention",
    }

    @classmethod
    def _classify_noun_patient_ontology(cls, lemma: str) -> str:
        if lemma in ("person", "people", "someone", "somebody", "human", "individual"):
            return "person"
        if lemma in (
            "day", "days", "time", "times", "week", "weeks", "month", "months", "year", "years",
            "hour", "hours", "minute", "minutes", "second", "seconds", "moment", "moments",
            "morning", "mornings", "afternoon", "afternoons", "evening", "evenings", "night", "nights",
            "period", "periods", "date", "dates", "season", "seasons",
        ):
            return "temporal"
        if lemma in (
            "item", "items", "belonging", "belongings", "good", "goods",
            "clothes", "clothing", "equipment", "furniture", "luggage",
            "baggage", "fuel", "trash", "garbage", "waste",
        ):
            return "physical"
        wn = cls.get_wordnet()
        if not wn:
            return "unknown"
        syns = wn.synsets(lemma, pos="n")
        if not syns:
            return "unknown"
        all_lfs = {s.lexfile() for s in syns}
        if all_lfs & {"noun.time"}:
            return "temporal"
        s0_lf = syns[0].lexfile()
        if s0_lf in cls.PERSON_SUPERSENSES:
            return "person"
        if s0_lf in cls.PHYSICAL_SUPERSENSES:
            return "physical"
        if s0_lf in cls.ABSTRACT_EVENT_SUPERSENSES:
            return "abstract_event"
        if all_lfs & cls.PERSON_SUPERSENSES and not (all_lfs & cls.PHYSICAL_SUPERSENSES):
            return "person"
        if all_lfs & cls.PHYSICAL_SUPERSENSES:
            return "physical"
        if all_lfs & cls.ABSTRACT_EVENT_SUPERSENSES:
            return "abstract_event"
        return "unknown"

    _GOV_PREPS = {
        "with", "on", "upon", "of", "to", "for", "from", "into",
        "at", "about", "after", "against", "over", "around", "through", "between", "in",
    }
    _PURE_PARTICLES = {
        "up", "down", "out", "off", "away", "back", "aside", "apart", "along", "ahead", "forward",
    }
    _PHYSICAL_DEF_VERBS = {
        "put", "place", "move", "store", "fill", "carry", "lift", "drop", "cut", "break",
        "clean", "throw", "burn", "wash", "build", "repair", "pack", "tidy", "fix", "hold",
        "wear", "shut", "open", "fasten", "nail", "tie", "wrap", "pour", "load", "unload",
        "wipe", "rip", "tear", "rub", "push", "pull", "feed", "destroy", "remove",
    }
    _COGNITIVE_DEF_VERBS = {
        "understand", "think", "know", "believe", "remember", "consider", "decide",
        "judge", "learn", "realize", "recognize", "accept", "mention", "speak", "talk",
        "explain", "define", "study", "ask", "expect", "depend", "relate",
    }

    @classmethod
    def _extract_verb_patient_ontology(
        cls,
        word: str,
        canonical_pos: str,
        target_entry: Optional[Dict[str, Any]],
        locked_sense: Optional[Dict[str, Any]],
        definition: Optional[str],
        authentic_example: Optional[str],
        quote: Optional[str] = None,
        forbidden_words: Optional[Set[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Determines whether a verb or verbal phrase is locked to:
        1. Concrete Physical Action Sense (e.g., `[concrete physical object, e.g. the lid / the coat]`)
        2. Abstract Event / Situation / Conflict / Problem / Process Sense (e.g., `[situation/crisis, e.g. the war / the conflict]`, `[problem/issue, e.g. the problem / the dispute]`)
        3. Human Recipient / Experiencer Sense (e.g., `[person/group, e.g. the parents / the students]`)
        Zero hardcoded word-to-patient maps or negative trap lists.
        """
        if canonical_pos not in ("verb", "phrase"):
            return None
        defn_low = str((locked_sense or {}).get("definition") or definition or "").strip().lower()
        if not defn_low:
            return None

        nlp = cls.get_spacy()
        wn = cls.get_wordnet()
        raw_w = (word or "").strip().lower()
        w_toks = [
            t for t in re.findall(r"[a-z]+", raw_w)
            if t not in ("sb", "sth", "somebody", "something", "ones", "one", "oneself")
        ]
        if not w_toks:
            return None
        head_v = w_toks[0]
        last_w = w_toks[-1]
        head_forms = ({head_v} | {f.lower() for f in cls.inflected_forms(head_v)}) if head_v else set()

        # Build predicate-internal lexical shield (`pred_words`) to prevent internal nouns
        # (e.g. 'care' in 'take care of', 'pleasure' in 'take pleasure in', 'threat' in 'pose a threat to',
        # 'difference' in 'make a difference', 'aid' in "come to one's aid") from being cannibalized as patients.
        pred_words: Set[str] = set(w_toks) | set(head_forms)
        if nlp and len(w_toks) >= 2:
            try:
                for p_tok in nlp(" ".join(w_toks)):
                    if p_tok.is_alpha:
                        pred_words.add(p_tok.lemma_.lower())
                        pred_words.add(p_tok.text.lower())
            except Exception:
                pass

        pat_list = [str(p) for p in ((locked_sense or {}).get("patterns") or []) if p]
        pat_str = " ".join(pat_list).lower()
        gram_str = str((locked_sense or {}).get("grammar") or "").lower()
        has_any_slot_marker = bool(re.search(r"\b(?:sb|sth|somebody|something|someone)\b", raw_w + " " + pat_str))
        has_explicit_sb_only = (
            bool(re.search(r"\((?:sb|somebody|someone)\)", raw_w))
            or (
                bool(re.search(r"\b(?:sb|somebody|someone)\b", raw_w + " " + pat_str))
                and not bool(re.search(r"\b(?:sth|something)\b", raw_w + " " + pat_str))
            )
        )
        has_explicit_sth_only = (
            bool(re.search(r"\((?:sth|something)\)", raw_w))
            or (
                bool(re.search(r"\b(?:sth|something)\b", raw_w + " " + pat_str))
                and not bool(re.search(r"\b(?:sb|somebody|someone)\b", raw_w + " " + pat_str))
            )
        )
        is_split_trivalent = bool(
            re.search(
                r"\((?:sb|sth|somebody|something|sb/sth|sth/sb|oneself)\)\s+(?:into|with|to|of|for|from|on|as)\b",
                raw_w,
            )
        )

        # Early Gate for Closed / Non-Verbal Phrases:
        # Multi-word expressions that do not end with a governing preposition or particle and carry no (sb)/(sth)
        # slot marker (e.g. 'peace of mind', 'make a difference', 'take place', 'play a role', "come to one's aid",
        # 'on sale', 'in charge', "a roof over one's head") take no external post-phrase object.
        if len(w_toks) >= 2 and not has_any_slot_marker:
            if head_v in (
                "in", "on", "at", "off", "by", "for", "with", "from", "under",
                "out", "beyond", "within", "without", "a", "an", "the", "no", "all",
            ) and last_w not in cls._GOV_PREPS:
                return None
            if len(w_toks) == 3 and w_toks[1] in ("and", "or", "of") and last_w not in cls._GOV_PREPS:
                return None
            if last_w not in cls._GOV_PREPS and last_w not in cls._PURE_PARTICLES:
                return None

        def _is_forb(w_str: str) -> bool:
            if not w_str or not forbidden_words:
                return False
            return cls._in_forbidden_word_set(w_str.lower(), forbidden_words)

        sample_texts: List[str] = []
        for s_txt in [authentic_example, quote]:
            if s_txt and str(s_txt).strip():
                sample_texts.append(str(s_txt).strip())
        for ex_s in (locked_sense or {}).get("examples") or []:
            if ex_s and str(ex_s).strip() not in sample_texts:
                sample_texts.append(str(ex_s).strip())
        for pb in (locked_sense or {}).get("pattern_bundles") or []:
            if isinstance(pb, dict) and pb.get("example"):
                pb_ex = str(pb["example"]).strip()
                if pb_ex and pb_ex not in sample_texts:
                    sample_texts.append(pb_ex)

        cands: List[Tuple[str, str]] = []  # (np_text, lemma)
        seen_lem: Set[str] = set()
        has_physical_def = False

        # Determine whether the phrase ends with a governing preposition (Paradigms II, III, IV)
        # vs an adverbial particle (Paradigm I: 'pack up', 'clean up', 'nail down', 'wipe out')
        # vs a split trivalent verb (Paradigm V: 'convert (sth) into (sth)', 'endow (sb/sth) with (sth)').
        is_prep_ending_phrase = (
            len(w_toks) >= 2
            and last_w in cls._GOV_PREPS
            and last_w not in cls._PURE_PARTICLES
            and not is_split_trivalent
        )

        # Gate for intransitive 2-word verb+particle phrasal verbs ('wind up', 'set off', 'slave away', 'turn up', 'show up', 'lie down', 'pay off', 'get by', 'stay on'):
        # If the locked sense is intransitive (no (sb)/(sth) slot marker, not marked [transitive], and defined by an intransitive/copular predicate),
        # it takes no external patient object in this sense.
        is_particle_pv = len(w_toks) == 2 and last_w in cls._PURE_PARTICLES
        if is_particle_pv and not has_any_slot_marker and "transitive" not in gram_str.replace("intransitive", ""):
            if re.match(
                r"^\s*(?:\[[^\]]+\]\s*)?(?:to\s+(?:be|arrive|start|go|work|happen|improve|continue|hide|stay|live|have\s+enough|put\s+yourself)\b|if\s+)",
                defn_low,
            ):
                return None

        # 1. Parse sample texts for valency-governed direct objects, passive subjects, and terminal-prep objects
        if nlp and head_v:
            for txt in sample_texts:
                try:
                    doc = nlp(re.sub(r"\s*\(=[^)]*\)", "", txt))
                except Exception:
                    continue
                for tok in doc:
                    tok_low = tok.text.lower()
                    tok_lem = tok.lemma_.lower()
                    is_match_verb = (
                        tok_low in head_forms
                        or tok_lem == head_v
                        or (len(w_toks) >= 2 and tok.pos_ in ("VERB", "ADJ") and tok_lem in pred_words and tok_lem not in cls._GOV_PREPS)
                    )
                    if not is_match_verb:
                        continue
                    # External direct objects / passive subjects (strictly excluding predicate-internal nouns)
                    direct_cands = [
                        c for c in tok.children
                        if c.dep_ in ("dobj", "obj", "nsubjpass", "nsubj:pass")
                        and c.pos_ in ("NOUN", "PROPN", "PRON")
                        and c.is_alpha
                        and c.lemma_.lower() not in pred_words
                        and c.text.lower() not in pred_words
                    ]
                    # Coordinated verbs sharing object (e.g. 'peel and dice the potatoes')
                    for conj_c in tok.children:
                        if conj_c.dep_ == "conj" and conj_c.pos_ == "VERB" and not any(c.dep_ in ("nsubj", "nsubjpass") for c in conj_c.children):
                            direct_cands.extend([
                                c for c in conj_c.children
                                if c.dep_ in ("dobj", "obj")
                                and c.pos_ in ("NOUN", "PROPN", "PRON")
                                and c.is_alpha
                                and c.lemma_.lower() not in pred_words
                                and c.text.lower() not in pred_words
                            ])
                    # Governed prepositional objects ONLY when the multi-word phrase ends with `last_w` as a governing
                    # preposition (`p_c.text.lower() == last_w`) and is not a split trivalent verb with an external dobj.
                    # Never harvest circumstantial adjunct prepositions ('in a suitcase', 'at the station', 'before dinner').
                    if is_prep_ending_phrase and not (len(w_toks) == 2 and direct_cands):
                        for p_c in tok.children:
                            if p_c.dep_ == "prep" and p_c.text.lower() == last_w:
                                for gc in p_c.children:
                                    if gc.dep_ == "pobj" and gc.is_alpha and gc.lemma_.lower() not in pred_words:
                                        direct_cands.append(gc)
                            elif p_c.dep_ in ("prep", "prt", "advmod") and p_c.text.lower() in pred_words:
                                for gc in p_c.children:
                                    if gc.dep_ == "prep" and gc.text.lower() == last_w:
                                        for ggc in gc.children:
                                            if ggc.dep_ == "pobj" and ggc.is_alpha and ggc.lemma_.lower() not in pred_words:
                                                direct_cands.append(ggc)
                        # Paradigm IV (`V + N + prep`, e.g. 'make sense of', 'take pleasure in', 'pose a threat to', 'lay siege to'):
                        # traverse the predicate-internal noun's children for the terminal governing preposition `last_w`.
                        for int_n in tok.children:
                            if int_n.dep_ in ("dobj", "obj", "attr", "oprd") and (int_n.lemma_.lower() in pred_words or int_n.text.lower() in pred_words):
                                for n_prep in int_n.children:
                                    if n_prep.dep_ == "prep" and n_prep.text.lower() == last_w:
                                        for pobj in n_prep.children:
                                            if pobj.dep_ == "pobj" and pobj.is_alpha and pobj.lemma_.lower() not in pred_words:
                                                direct_cands.append(pobj)
                            elif int_n.dep_ == "prep" and int_n.text.lower() in pred_words:
                                for p_n in int_n.children:
                                    if p_n.dep_ == "pobj" and (p_n.lemma_.lower() in pred_words or p_n.text.lower() in pred_words):
                                        for n_prep in p_n.children:
                                            if n_prep.dep_ == "prep" and n_prep.text.lower() == last_w:
                                                for pobj in n_prep.children:
                                                    if pobj.dep_ == "pobj" and pobj.is_alpha and pobj.lemma_.lower() not in pred_words:
                                                        direct_cands.append(pobj)

                    if not direct_cands and tok.dep_ in ("acl", "amod", "relcl") and tok.head.pos_ in ("NOUN", "PROPN") and tok.head.is_alpha and tok.head.lemma_.lower() not in pred_words:
                        if len(w_toks) == 1 or is_prep_ending_phrase or has_any_slot_marker:
                            direct_cands = [tok.head]

                    for c_tok in direct_cands:
                        c_text = c_tok.text.lower()
                        c_lem = c_tok.lemma_.lower()
                        if c_lem in pred_words or c_text in pred_words:
                            continue
                        if (
                            c_text in ("him", "her", "them", "us", "you", "whom", "someone", "somebody")
                            or c_tok.ent_type_ == "PERSON"
                            or c_lem in ("person", "people", "man", "woman", "child", "kid", "student", "victim", "suspect", "worker", "parent", "volunteer")
                        ) and not has_explicit_sth_only:
                            if "person" not in seen_lem:
                                seen_lem.add("person")
                                cands.append(("someone", "person"))
                            continue
                        if c_tok.pos_ not in ("NOUN", "PROPN"):
                            continue
                        if c_lem in (
                            "thing", "something", "someone", "somebody", "one", "way",
                            "time", "day", "days", "week", "month", "year", "place", "part", "side", "rapist", "murderer", "corpse",
                        ) or _is_forb(c_lem):
                            continue
                        # Quantifier / partitive of-phrase resolution (e.g. 'wave of rioting', 'forms of illness')
                        real_c_tok = c_tok
                        real_lem = c_lem
                        for pch in c_tok.children:
                            if pch.dep_ == "prep" and pch.text.lower() == "of":
                                pobj = next(
                                    (gc for gc in pch.children if gc.dep_ == "pobj" and gc.pos_ in ("NOUN", "PROPN") and gc.is_alpha),
                                    None,
                                )
                                if pobj and not _is_forb(pobj.lemma_.lower()) and pobj.lemma_.lower() not in ("thing", "something", "someone") and pobj.lemma_.lower() not in pred_words:
                                    real_c_tok = pobj
                                    real_lem = pobj.lemma_.lower()
                                    break
                        if real_lem not in seen_lem and real_lem not in pred_words:
                            seen_lem.add(real_lem)
                            cands.append((f"the {real_c_tok.text.lower()}", real_lem))

        # 2. Dependency-Constrained Definition Parsing (Step 2)
        if nlp and len(cands) < 2:
            try:
                def_clean = re.sub(r"^\s*\[[^\]]+\]\s*", "", defn_low).strip()
                def_doc = nlp(def_clean)
                # Identify primary definition predicate verbs (ROOT or leading infinitive 'to <verb>')
                def_verbs = [
                    t for t in def_doc
                    if t.pos_ == "VERB"
                    and (
                        t.dep_ in ("ROOT", "xcomp", "ccomp", "conj")
                        or (t.i > 0 and def_doc[t.i - 1].text.lower() == "to" and t.i <= 4)
                    )
                ]
                if not def_verbs:
                    def_verbs = [t for t in def_doc if t.dep_ == "ROOT"]

                for dv in def_verbs:
                    dv_lem = dv.lemma_.lower()
                    dv_dobjs = [c for c in dv.children if c.dep_ in ("dobj", "obj", "attr", "oprd")]
                    # Also check xcomp child when dv is a light/aspectual verb ('start using a plan', 'try to find someone', 'manage to get past something')
                    for xc in dv.children:
                        if xc.dep_ in ("xcomp", "acomp") and xc.pos_ == "VERB":
                            dv_dobjs.extend([c for c in xc.children if c.dep_ in ("dobj", "obj")])
                    has_external_dobj = False
                    for dobj in dv_dobjs:
                        d_text = dobj.text.lower()
                        d_lem = dobj.lemma_.lower()
                        if d_lem in pred_words or d_text in pred_words:
                            continue
                        if d_text in ("yourself", "oneself", "itself", "themselves"):
                            has_external_dobj = True
                            continue
                        if d_text in ("you", "someone", "somebody", "people", "person") and not has_explicit_sth_only:
                            has_external_dobj = True
                            if "person" not in seen_lem:
                                seen_lem.add("person")
                                cands.append(("someone", "person"))
                            continue
                        # Generic physical object marker ('to put things into cases', 'to throw something away', 'to make a place clean')
                        if d_lem in ("thing", "things", "something", "item", "items", "good", "goods", "object", "objects", "belonging", "belongings", "clothes", "equipment", "place", "material", "container", "machine", "vehicle"):
                            has_external_dobj = True
                            if dv_lem in cls._PHYSICAL_DEF_VERBS or (
                                dv_lem == "make"
                                and any(w in def_clean for w in ("clean", "tidy", "damaged", "dirty", "flat", "warm", "hot", "cold", "dry", "wet"))
                            ) or (
                                dv_lem == "become"
                                and any(w in def_clean for w in ("damaged", "worn", "broken"))
                            ):
                                has_physical_def = True
                                if any(w in def_clean for w in ("case", "cases", "bag", "bags", "clothes", "clothing", "wear", "suit")):
                                    for ex_np, ex_l in [("the clothes", "clothes"), ("the equipment", "equipment")]:
                                        if ex_l not in seen_lem and not _is_forb(ex_l):
                                            seen_lem.add(ex_l)
                                            cands.append((ex_np, ex_l))
                                elif d_lem == "place":
                                    for ex_np, ex_l in [("the room", "room"), ("the area", "area")]:
                                        if ex_l not in seen_lem and not _is_forb(ex_l):
                                            seen_lem.add(ex_l)
                                            cands.append((ex_np, ex_l))
                                else:
                                    for ex_np, ex_l in [("the equipment", "equipment"), ("the items", "item")]:
                                        if ex_l not in seen_lem and not _is_forb(ex_l):
                                            seen_lem.add(ex_l)
                                            cands.append((ex_np, ex_l))
                            elif dv_lem in ("write", "record", "list"):
                                for ex_np, ex_l in [("the details", "detail"), ("the points", "point")]:
                                    if ex_l not in seen_lem and not _is_forb(ex_l):
                                        seen_lem.add(ex_l)
                                        cands.append((ex_np, ex_l))
                            elif dv_lem in ("delay", "postpone", "cancel", "arrange"):
                                for ex_np, ex_l in [("the meeting", "meeting"), ("the event", "event")]:
                                    if ex_l not in seen_lem and not _is_forb(ex_l):
                                        seen_lem.add(ex_l)
                                        cands.append((ex_np, ex_l))
                            continue
                        # Quantifier 'all of something' / 'amount of something' / 'shape from a piece of paper'
                        if d_lem in ("all", "amount", "number", "lot", "part", "piece", "shape", "form"):
                            has_external_dobj = True
                            if dv_lem in cls._PHYSICAL_DEF_VERBS or dv_lem == "use":
                                has_physical_def = True
                                for ex_np, ex_l in [("the supplies", "supply"), ("the equipment", "equipment")]:
                                    if ex_l not in seen_lem and not _is_forb(ex_l):
                                        seen_lem.add(ex_l)
                                        cands.append((ex_np, ex_l))
                            continue
                        if dobj.pos_ == "NOUN" and dobj.is_alpha:
                            has_external_dobj = True
                            if d_lem not in (
                                "one", "way", "time", "day", "days", "week", "month", "year",
                                "side", "example", "order", "fact", "action",
                            ) and not _is_forb(d_lem) and d_lem not in seen_lem:
                                seen_lem.add(d_lem)
                                cands.append((f"the {d_text}", d_lem))
                            for conj_n in dobj.children:
                                if conj_n.dep_ == "conj" and conj_n.pos_ == "NOUN" and conj_n.is_alpha:
                                    cn_lem = conj_n.lemma_.lower()
                                    if cn_lem not in pred_words and cn_lem not in seen_lem and not _is_forb(cn_lem) and cn_lem not in ("thing", "something", "someone", "somebody", "etc"):
                                        seen_lem.add(cn_lem)
                                        cands.append((f"the {conj_n.text.lower()}", cn_lem))

                    # Rule of Preposition Degradation:
                    # Only inspect a preposition in the definition if `dv` did NOT already take an external direct object
                    # (preventing locative/purpose/manner adjuncts like 'put things [into cases] [for a trip]' or
                    # 'meet someone [by chance]' from hijacking the Patient ontology).
                    if not has_external_dobj:
                        gov_verbs = [dv] + [
                            xc for xc in dv.children
                            if xc.dep_ in ("xcomp", "acomp", "advcl") and xc.pos_ in ("VERB", "ADJ")
                        ]
                        for gv in gov_verbs:
                            # Also allow light-noun constructions inside the definition ('to take care of someone', 'to ask for information from someone')
                            prep_heads = [gv] + [
                                c for c in gv.children
                                if (c.dep_ in ("dobj", "obj", "prep") and (c.lemma_.lower() in pred_words or c.lemma_.lower() in ("care", "action", "effort", "request", "information", "advice", "in")))
                            ]
                            for ph in prep_heads:
                                for p_c in ph.children:
                                    if p_c.dep_ == "prep" and p_c.lemma_.lower() in ("of", "with", "to", "for", "about", "after", "upon", "on", "into", "from", "through", "around", "between"):
                                        for gc in p_c.children:
                                            if gc.dep_ != "pobj" or not gc.is_alpha:
                                                continue
                                            gc_text = gc.text.lower()
                                            gc_lem = gc.lemma_.lower()
                                            if gc_lem in pred_words or gc_text in pred_words:
                                                continue
                                            if gc_text in ("you", "someone", "somebody", "people", "person") and not has_explicit_sth_only:
                                                if "person" not in seen_lem:
                                                    seen_lem.add("person")
                                                    cands.append(("someone", "person"))
                                                continue
                                            if gc.pos_ == "NOUN" and gc_lem not in (
                                                "thing", "something", "someone", "somebody", "one", "way",
                                                "time", "day", "days", "week", "month", "year", "place", "part", "side", "example", "order", "fact", "chance",
                                            ) and not _is_forb(gc_lem) and gc_lem not in seen_lem:
                                                seen_lem.add(gc_lem)
                                                cands.append((f"the {gc_text}", gc_lem))
                # Conditional-subject patient definitions for intransitive verbs ('if a crisis/swelling subsides...', 'if something bad catches up with you...')
                if len(cands) < 2 and def_clean.startswith("if "):
                    for d_tok in def_doc:
                        if d_tok.dep_ in ("dobj", "obj", "pobj") and d_tok.text.lower() in ("you", "someone", "somebody") and not has_explicit_sth_only:
                            if "person" not in seen_lem:
                                seen_lem.add("person")
                                cands.append(("someone", "person"))
                        elif len(w_toks) == 1 and d_tok.dep_ == "nsubj" and d_tok.pos_ == "NOUN" and d_tok.is_alpha:
                            s_lem = d_tok.lemma_.lower()
                            if s_lem not in pred_words and s_lem not in ("thing", "something", "someone", "somebody", "person", "people") and not _is_forb(s_lem) and s_lem not in seen_lem:
                                seen_lem.add(s_lem)
                                cands.append((f"the {d_tok.text.lower()}", s_lem))
            except Exception:
                pass

        # Explicit `(sb)` syllabus/pattern marker guarantee
        if has_explicit_sb_only and "person" not in seen_lem:
            seen_lem.add("person")
            cands.insert(0, ("someone", "person"))

        # 3. Fallback to Oxford collocations if still sparse (STRICTLY for single-word base verbs)
        if len(cands) < 2 and head_v and len(w_toks) == 1:
            collocs = cls.get_oxford_collocations(head_v)
            for cn in (collocs or {}).get("colloc_nouns") or []:
                cn_clean = re.sub(r"[^a-z]", "", str(cn).lower())
                if (
                    cn_clean
                    and cn_clean not in seen_lem
                    and cn_clean not in pred_words
                    and not _is_forb(cn_clean)
                    and cn_clean not in (
                        "day", "days", "time", "times", "week", "weeks", "month", "months", "year", "years",
                        "hour", "hours", "minute", "minutes", "second", "seconds", "moment", "moments",
                        "morning", "mornings", "afternoon", "afternoons", "evening", "evenings", "night", "nights",
                        "period", "periods", "date", "dates", "season", "seasons",
                    )
                ):
                    seen_lem.add(cn_clean)
                    cands.append((f"the {cn_clean}", cn_clean))
                    if len(cands) >= 4:
                        break

        if not cands:
            return None

        # Classify candidates into physical, abstract_event, person
        phys_cands: List[str] = []
        abs_cands: List[str] = []
        person_cands: List[str] = []
        for np_text, lem in cands:
            cat = cls._classify_noun_patient_ontology(lem)
            if cat == "person":
                if not has_explicit_sth_only:
                    person_cands.append(np_text)
            elif cat == "physical":
                phys_cands.append(np_text)
            elif cat == "abstract_event":
                abs_cands.append(np_text)

        # Lexical clues in definition
        has_animal = any("animal" in (wn.synsets(lem, pos="n")[0].lexfile() if wn and wn.synsets(lem, pos="n") else "") for _, lem in cands) or bool(re.search(r"\b(?:animals?|dogs?|horses?|elephants?|cattle)\b", defn_low))
        has_person_def = not has_explicit_sth_only and (
            has_explicit_sb_only
            or bool(
                re.search(
                    r"^\s*(?:\[[^\]]+\]\s*)?to\s+(?:make|tell|persuade|advise|help|ask|allow|force|encourage|cause|give|show|teach|remind|warn|comfort|treat|meet|admire|punish)\s+(?:someone|somebody|people)\b",
                    defn_low,
                )
            )
            or bool(
                re.search(r"\b(?:someone|somebody)\s+(?:feel|think|believe|understand|do)\b", defn_low)
            )
            or bool(
                re.search(r"\b(?:with|to|for|about|against|upon|after|at|of|from)\s+(?:you|someone|somebody|a person|people)\b", defn_low)
            )
            or bool(
                re.search(r"\b(?:affect|affects|punish|punishes|harm|harms|catch up with|befall|befalls)\s+(?:you|someone|somebody|a person|people)\b", defn_low)
            )
        )

        if has_explicit_sb_only and person_cands:
            exemplars = person_cands[:2]
            ex_inline = " / ".join(exemplars)
            slot_label = f"[person/group, e.g. {ex_inline}]"
            ont_type = "person_group"
        elif has_physical_def and phys_cands:
            exemplars = phys_cands[:2]
            ex_inline = " / ".join(exemplars)
            if has_animal:
                slot_label = f"[concrete physical object/animal, e.g. {ex_inline}]"
            else:
                slot_label = f"[concrete physical object, e.g. {ex_inline}]"
            ont_type = "concrete_physical"
        elif (has_person_def or (person_cands and not abs_cands and not phys_cands)) and person_cands:
            exemplars = person_cands[:2]
            ex_inline = " / ".join(exemplars)
            slot_label = f"[person/group, e.g. {ex_inline}]"
            ont_type = "person_group"
        elif abs_cands and (
            len(abs_cands) >= len(phys_cands)
            or any(w in defn_low for w in ("situation", "problem", "crisis", "conflict", "process", "activity", "tension", "event"))
        ):
            exemplars = abs_cands[:2]
            ex_inline = " / ".join(exemplars)
            all_abs_text = " ".join(exemplars).lower() + " " + defn_low
            if any(w in all_abs_text for w in cls.CRISIS_KEYWORDS):
                slot_label = f"[situation/crisis, e.g. {ex_inline}]"
                ont_type = "situation_crisis"
            elif any(w in all_abs_text for w in cls.PROBLEM_KEYWORDS):
                slot_label = f"[problem/issue, e.g. {ex_inline}]"
                ont_type = "problem_issue"
            elif any(w in all_abs_text for w in cls.PROCESS_KEYWORDS):
                slot_label = f"[process/activity, e.g. {ex_inline}]"
                ont_type = "process_activity"
            elif any(w in all_abs_text for w in cls.FEELING_KEYWORDS):
                slot_label = f"[feeling/state, e.g. {ex_inline}]"
                ont_type = "feeling_state"
            elif any(w in all_abs_text for w in cls.ASSET_KEYWORDS):
                slot_label = f"[funds/resources, e.g. {ex_inline}]"
                ont_type = "funds_resources"
            elif any(w in all_abs_text for w in cls.OPINION_KEYWORDS):
                slot_label = f"[opinion/concern, e.g. {ex_inline}]"
                ont_type = "opinion_concern"
            else:
                slot_label = f"[situation/event, e.g. {ex_inline}]"
                ont_type = "situation_event"
        elif person_cands and (len(person_cands) > len(phys_cands) or has_person_def):
            exemplars = person_cands[:2]
            ex_inline = " / ".join(exemplars)
            slot_label = f"[person/group, e.g. {ex_inline}]"
            ont_type = "person_group"
        elif phys_cands:
            exemplars = phys_cands[:2]
            ex_inline = " / ".join(exemplars)
            if has_animal:
                slot_label = f"[concrete physical object/animal, e.g. {ex_inline}]"
            elif person_cands:
                slot_label = f"[physical person or concrete object, e.g. {ex_inline}]"
            else:
                slot_label = f"[concrete physical object, e.g. {ex_inline}]"
            ont_type = "concrete_physical"
        elif abs_cands:
            exemplars = abs_cands[:2]
            ex_inline = " / ".join(exemplars)
            slot_label = f"[situation/event, e.g. {ex_inline}]"
            ont_type = "situation_event"
        elif person_cands:
            exemplars = person_cands[:2]
            ex_inline = " / ".join(exemplars)
            slot_label = f"[person/group, e.g. {ex_inline}]"
            ont_type = "person_group"
        else:
            return None

        return {
            "slot_label": slot_label,
            "exemplars": exemplars,
            "ontology_type": ont_type,
        }

    # Backward-compatible alias
    _extract_physical_patient_ontology = _extract_verb_patient_ontology

    @classmethod
    def derive_cloze_slot_frame_from_example(
        cls,
        example_text: Optional[str],
        final_target: str,
        base_word: str,
        canonical_pos: str,
        anchor: Optional[str] = None,
        anchor_type: Optional[str] = None,
        is_copula_adj: bool = False,
        is_attributive_only_adj: bool = False,
        is_plural_noun: bool = False,
        is_pure_uncountable_target: bool = False,
        is_target_strictly_countable: bool = False,
        has_mixed_vc_initials: bool = False,
        has_uncountable_contrast_foil: bool = False,
        verb_form_tag: Optional[str] = None,
        forbidden_words: Optional[Set[str]] = None,
        patient_ontology_slot: Optional[str] = None,
    ) -> Optional[str]:
        """
        Slices a compact, authentic 2-to-5-word `cloze_slot_frame` directly from the winning example's
        syntactic neighborhood around the target word, guaranteeing at least one concrete left or right
        context word/POS anchor (`... their ____ achievement`, `... take out a ____`, `... ____ to [verb]`,
        `... ____ for`, `... is ____ in`) and zero bare `None` frames.
        """
        if not example_text or not final_target:
            return None
        nlp = cls.get_spacy()
        if not nlp:
            return None

        ex_clean = re.sub(r"\s*\(=[^)]*\)", "", str(example_text)).strip()
        ft_low = final_target.strip().lower()
        bw_low = (base_word or ft_low).strip().lower()
        t_forms = {ft_low, bw_low} | set(cls.inflected_forms(bw_low)) | set(cls.inflected_forms(ft_low))

        def _is_forbidden(word_str: str) -> bool:
            if not word_str or not forbidden_words:
                return False
            return cls._in_forbidden_word_set(word_str.lower(), forbidden_words)

        def _normalize_det(det_str: str, det_tok_obj=None) -> str:
            if det_tok_obj is not None and det_tok_obj.dep_ == "poss" and det_tok_obj.pos_ in ("NOUN", "PROPN", "ADJ"):
                return "their"
            d_l = (det_str or "").strip().lower()
            if not d_l:
                return ""
            if d_l.endswith("'s") or d_l.endswith("’s"):
                return "their"
            if canonical_pos == "noun":
                if is_pure_uncountable_target or is_plural_noun:
                    if d_l in ("a", "an", "each", "every", "one"):
                        if is_plural_noun:
                            return "several" if has_uncountable_contrast_foil else "the"
                        return ""
                    if is_plural_noun and has_uncountable_contrast_foil and d_l in ("the", "some", "any"):
                        return "several"
                elif has_uncountable_contrast_foil and not is_plural_noun:
                    if d_l in ("a", "an", "the"):
                        return "each" if has_mixed_vc_initials else "a"
                elif has_mixed_vc_initials and d_l in ("a", "an"):
                    return "each" if has_uncountable_contrast_foil else "the"
            elif canonical_pos == "adj":
                if has_mixed_vc_initials and d_l in ("a", "an"):
                    return "the"
            return d_l

        try:
            doc = nlp(ex_clean)
        except Exception:
            return None

        _cop_contraction_map = {
            "'s": "is", "’s": "is",
            "'re": "are", "’re": "are",
            "'m": "am", "’m": "am",
            "'ll": "will", "’ll": "will",
            "'d": "would", "’d": "would",
        }

        # 1. Multi-word phrase / expression
        if " " in ft_low or canonical_pos == "phrase":
            ft_clean_for_toks = re.sub(
                r"\b(?:one's|oneself|someone's|somebody's|my|your|his|her|its|our|their)\b",
                " ",
                ft_low,
            )
            ft_toks = [t for t in re.findall(r"[a-z]+", ft_clean_for_toks) if t not in ("sb", "sth", "somebody", "something", "one", "ones")]
            if ft_toks:
                # 1.0 Passive 3-Word Verbal Phrase ('be equipped with', 'be associated with', 'be based on')
                is_passive_3w_phrase = (
                    len(ft_toks) >= 3
                    and ft_toks[0] in ("be", "get", "feel")
                    and (ft_toks[1].endswith("ed") or ft_toks[1].endswith("en") or cls.lemma_of(ft_toks[1]) != ft_toks[1])
                )
                if is_passive_3w_phrase:
                    base_v = cls.lemma_of(ft_toks[1])
                    prep_w = ft_toks[-1]
                    v_tok = None
                    for tok in doc:
                        if (tok.lemma_.lower() == base_v or tok.text.lower() == ft_toks[1]) and tok.pos_ in ("VERB", "AUX"):
                            v_tok = tok
                            break
                    p_tok = None
                    if v_tok:
                        for tok in doc:
                            if tok.i > v_tok.i and tok.text.lower() == prep_w and tok.pos_ in ("ADP", "SCONJ"):
                                p_tok = tok
                                break
                    if v_tok and p_tok:
                        is_ex_passive = (
                            any(c.dep_ in ("auxpass", "nsubjpass") for c in v_tok.children)
                            or (v_tok.i > 0 and doc[v_tok.i - 1].text.lower() in ("is", "are", "was", "were", "been", "be", "get", "got"))
                        )
                        prep_anim = cls._resolve_object_animacy_from_example(ex_clean, prep_w, default_tag="something")
                        if is_ex_passive:
                            pass_subj = next((c for c in v_tok.children if c.dep_ in ("nsubjpass", "nsubj") and c.i < v_tok.i), None)
                            s_text = pass_subj.text.lower() if pass_subj else "something"
                            is_human = (
                                cls._resolve_object_animacy_from_example(ex_clean, base_v) == "someone"
                                or s_text in ("students", "teachers", "people", "student", "teacher", "children", "workers")
                            )
                            slot_label = f"[someone, e.g. {s_text}]" if is_human else f"[something, e.g. {s_text}]"
                            aux_toks = [c.text.lower() for c in v_tok.children if (c.dep_ in ("aux", "auxpass") or c.pos_ == "AUX") and c.i < v_tok.i]
                            aux_str = (" ".join(aux_toks) + " ") if aux_toks else ("will " if ft_toks[0] == "be" else "were/are ")
                            if ft_toks[0] == "be" and any(w in aux_str for w in ("is", "are", "was", "were", "been")):
                                aux_str = "will "
                            return f"... {slot_label} {aux_str}____ {prep_anim}"
                        else:
                            # Active diathesis: Direct object of base_v is the Patient (becomes passive subject)
                            dobj = next((c for c in v_tok.children if c.dep_ in ("dobj", "obj")), None)
                            if not dobj:
                                for tok in doc[v_tok.i + 1 : p_tok.i]:
                                    if tok.pos_ in ("NOUN", "PROPN", "PRON"):
                                        dobj = tok
                                        break
                            d_text = dobj.text.lower() if dobj else "someone"
                            is_human = (
                                cls._resolve_object_animacy_from_example(ex_clean, base_v) == "someone"
                                or d_text in ("students", "teachers", "people", "student", "teacher", "children", "workers")
                            )
                            slot_label = f"[someone, e.g. {d_text}]" if is_human else f"[something, e.g. {d_text}]"
                            aux_str = "will " if ft_toks[0] == "be" else "were "
                            return f"... {slot_label} {aux_str}____ {prep_anim}"

                first_lem = nlp(ft_toks[0])[0].lemma_.lower() if nlp else ft_toks[0]
                first_forms = {ft_toks[0], first_lem} | set(cls.inflected_forms(ft_toks[0])) | set(cls.inflected_forms(first_lem))
                last_w = ft_toks[-1]
                start_i, end_i = None, None
                for tok in doc:
                    if tok.text.lower() in first_forms or tok.lemma_.lower() in (ft_toks[0], first_lem):
                        if ft_toks[0] in ("a", "an", "the") and len(ft_toks) >= 2:
                            if tok.i + 1 >= len(doc) or (
                                doc[tok.i + 1].text.lower() != ft_toks[1]
                                and doc[tok.i + 1].lemma_.lower() != ft_toks[1]
                            ):
                                continue
                        for j in range(tok.i + len(ft_toks) - 1, min(len(doc), tok.i + len(ft_toks) + 4)):
                            if doc[j].text.lower() == last_w or doc[j].lemma_.lower() == last_w:
                                start_i, end_i = tok.i, j + 1
                                break
                        if start_i is not None:
                            break
                if start_i is not None and end_i is not None:
                    _head_ent = cls.get_ldoce_entry(ft_toks[0])
                    _head_primary_pos = str((_head_ent or {}).get("pos") or "").strip().lower()
                    _is_binomial_or_of_np = (
                        (
                            len(ft_toks) == 3
                            and ft_toks[1] in ("and", "or", "by", "to")
                            and (
                                ft_toks[0] == ft_toks[2]
                                or ft_toks[0] in ("back", "up", "down", "in", "out", "on", "off", "more", "sooner", "now", "first", "day", "time", "side", "face", "hand", "step", "word", "pros", "ups", "odds", "ins")
                            )
                        )
                        or (len(ft_toks) >= 3 and ft_toks[1] == "of")
                    )
                    is_verbal_phrase = (
                        not _is_binomial_or_of_np
                        and (
                            doc[start_i].pos_ == "VERB"
                            or (
                                bool(cls._wordnet_pos(ft_toks[0]) & {"v"})
                                and _head_primary_pos not in ("noun", "adjective", "adverb", "preposition", "determiner")
                            )
                        )
                    )
                    left_aux = ""
                    if is_verbal_phrase and start_i > 0 and verb_form_tag not in ("VBD", "VBZ", "VBG", "VBN"):
                        prev_raw = doc[start_i - 1].text.lower()
                        if prev_raw == "'ll":
                            left_aux = "will "
                        elif prev_raw == "'d":
                            left_aux = "would "
                        elif prev_raw in ("to", "will", "would", "could", "can", "might", "may", "should", "must", "did", "do", "does"):
                            left_aux = f"{prev_raw} "
                    # Copular Adjective Phrase: when phrase head is an adjective followed by a preposition
                    # (e.g. 'full of', 'short of', 'proud of', 'fond of'), recover the finite copula from the example
                    _head_is_adj = not (ft_toks[0].endswith("ed") or ft_toks[0].endswith("en")) and (
                        _head_primary_pos == "adjective" or bool(cls._wordnet_pos(ft_toks[0]) & {"a", "s"})
                    )
                    if _head_is_adj and not left_aux:
                        target_tok = doc[start_i]
                        cop_word = None
                        if target_tok.head.pos_ in ("AUX", "VERB") and target_tok.head.lemma_.lower() in ("be", "seem", "feel", "become", "look", "sound", "taste", "smell", "remain", "stay") and target_tok.head.i < start_i:
                            cop_word = target_tok.head.text.lower()
                        if not cop_word:
                            for c in target_tok.children:
                                if (c.dep_ in ("cop", "aux") or c.lemma_.lower() == "be") and c.i < start_i:
                                    cop_word = c.text.lower()
                                    break
                        if not cop_word:
                            for anc in target_tok.ancestors:
                                if anc.pos_ in ("AUX", "VERB") and anc.lemma_.lower() in ("be", "seem", "feel", "become", "look", "sound", "taste", "smell", "remain", "stay") and anc.i < start_i:
                                    cop_word = anc.text.lower()
                                    break
                        if not cop_word:
                            cop_word = "was"
                        cop_word = _cop_contraction_map.get(cop_word, cop_word)
                        left_aux = f"{cop_word} "

                    # Check if a direct object sits inside a separable phrasal verb ('turned the whole house over', 'wake you up')
                    mid_toks = doc[start_i + 1 : end_i - 1]
                    if len(ft_toks) == 2 and last_w in ("up", "down", "out", "in", "off", "on", "away", "back", "over", "through", "around", "along", "aside", "apart") and any(t.pos_ in ("PRON", "NOUN", "PROPN") for t in mid_toks):
                        anim = patient_ontology_slot or cls._resolve_object_animacy_from_example(ex_clean, ft_toks[0], default_tag="[sb/sth]")
                        return f"... {left_aux}____ {anim}"
                    # If the phrase ends with a transitive preposition ('set out on', 'catch up with', 'listen to', 'take pleasure in')
                    if last_w in ("with", "to", "for", "of", "into", "from", "on", "at", "about", "against", "upon", "over", "after") or (last_w == "in" and len(ft_toks) >= 3):
                        anim = patient_ontology_slot or cls._resolve_object_animacy_from_example(ex_clean, ft_low, default_tag="[sb/sth]")
                        return f"... {left_aux}____ {anim}"
                    # Check if the phrase starts with a verb and takes a direct object right after `end_i`
                    if is_verbal_phrase and end_i < len(doc):
                        nxt_t = doc[end_i]
                        _is_2w_adv_pv = (
                            len(ft_toks) == 2
                            and last_w in ("up", "down", "out", "in", "off", "on", "away", "back", "over", "through", "around", "along", "aside", "apart")
                        )
                        if (
                            nxt_t.pos_ == "ADP"
                            and nxt_t.text.lower() not in ("between", "during", "since", "until", "before", "after", "throughout")
                            and (not _is_2w_adv_pv or nxt_t.text.lower() == str(anchor or "").strip().lower())
                        ):
                            return f"... {left_aux}____ {nxt_t.text.lower()}"
                        has_dobj_after = (
                            nxt_t.dep_ in ("dobj", "obj")
                            or (
                                nxt_t.pos_ in ("DET", "PRON", "NOUN", "PROPN")
                                and nxt_t.text.lower() not in ("all", "every", "each", "both", "today", "yesterday", "tomorrow")
                                and any(t.dep_ in ("dobj", "obj") for t in doc[end_i : min(len(doc), end_i + 4)])
                            )
                        )
                        if has_dobj_after:
                            anim = patient_ontology_slot or cls._resolve_object_animacy_from_example(ex_clean, ft_low, default_tag="[sb/sth]")
                            return f"... {left_aux}____ {anim}"
                    if is_verbal_phrase:
                        is_pv_passive = any(c.dep_ in ("auxpass", "nsubjpass") for c in doc[start_i].children)
                        if not is_pv_passive and start_i > 0 and doc[start_i - 1].text.lower() in ("was", "were", "is", "are", "been", "be", "got", "get"):
                            is_pv_passive = True
                        pv_pass_subj = next(
                            (
                                c for c in doc[start_i].children
                                if c.dep_ in ("nsubjpass", "nsubj") and c.i < start_i and c.pos_ in ("NOUN", "PROPN") and c.is_alpha and not _is_forbidden(c.lemma_.lower())
                            ),
                            None,
                        )
                        # Passive-Preserving Frame for VBN verbal phrases ('The lid was firmly nailed down.', 'The elephants were chained up by their legs.'):
                        if is_pv_passive and verb_form_tag == "VBN":
                            aux_start = pv_pass_subj.i + 1 if pv_pass_subj is not None else max(0, start_i - 3)
                            aux_toks = [
                                t.text.lower()
                                for t in doc[aux_start : start_i]
                                if t.pos_ == "AUX"
                                or t.dep_ in ("aux", "auxpass", "neg")
                                or t.text.lower() in ("was", "were", "is", "are", "been", "be", "has", "have", "had", "got", "get")
                            ]
                            mid_pv_aux = (" ".join(aux_toks) + " ") if aux_toks else "was/were "
                            if patient_ontology_slot:
                                mid_pv_aux_norm = re.sub(r"\b(?:was|were)\b", "was/were", mid_pv_aux)
                                mid_pv_aux_norm = re.sub(r"\b(?:is|are)\b", "is/are", mid_pv_aux_norm)
                                mid_pv_aux_norm = re.sub(r"\b(?:has|have)\b", "has/have", mid_pv_aux_norm)
                                return f"... {patient_ontology_slot} {mid_pv_aux_norm}____"
                            if pv_pass_subj is not None:
                                s_low = pv_pass_subj.text.lower()
                                det_cands = [d for d in pv_pass_subj.children if d.dep_ in ("det", "poss") and d.i < pv_pass_subj.i and d.is_alpha]
                                if det_cands:
                                    d_low = det_cands[-1].text.lower()
                                    if d_low in ("my", "your", "our"):
                                        d_low = "a" if pv_pass_subj.tag_ == "NN" else "the"
                                    return f"... {d_low} {s_low} {mid_pv_aux}____"
                                if pv_pass_subj.tag_ == "NN":
                                    return f"... a {s_low} {mid_pv_aux}____"
                                return f"... {s_low} {mid_pv_aux}____"
                        # Deep Passive Projection fallback when options are in base form ('nail down'):
                        if is_pv_passive and verb_form_tag not in ("VBD", "VBZ", "VBG", "VBN"):
                            if patient_ontology_slot:
                                return f"... {left_aux or 'to '}____ {patient_ontology_slot}"
                            if pv_pass_subj is not None:
                                s_low = pv_pass_subj.text.lower()
                                det_cands = [d for d in pv_pass_subj.children if d.dep_ in ("det", "poss") and d.i < pv_pass_subj.i and d.is_alpha]
                                if det_cands:
                                    d_low = det_cands[-1].text.lower()
                                    if d_low in ("my", "your", "our"):
                                        d_low = "a" if pv_pass_subj.tag_ == "NN" else "the"
                                    obj_np = f"{d_low} {s_low}"
                                elif pv_pass_subj.tag_ == "NN":
                                    obj_np = f"the {s_low}"
                                else:
                                    obj_np = s_low
                                return f"... {left_aux or 'to '}____ {obj_np}"
                        if left_aux:
                            return f"... {left_aux}____"
                        if verb_form_tag in ("VBD", "VBZ", "VBG", "VBN"):
                            if end_i < len(doc) and doc[end_i].is_alpha and not _is_forbidden(doc[end_i].lemma_.lower()):
                                return f"... ____ {doc[end_i].text.lower()}"
                            subj_cands = [
                                c for c in doc[start_i].children
                                if c.dep_ in (("nsubj", "nsubjpass") if verb_form_tag == "VBN" else ("nsubj",)) and c.i < start_i and c.is_alpha
                            ]
                            subj_tok = subj_cands[-1] if subj_cands else (
                                doc[start_i - 1]
                                if not is_pv_passive and start_i > 0 and doc[start_i - 1].is_alpha and doc[start_i - 1].pos_ in ("NOUN", "PROPN", "PRON")
                                else None
                            )
                            if subj_tok and subj_tok.pos_ in ("NOUN", "PROPN") and not _is_forbidden(subj_tok.lemma_.lower()):
                                s_low = subj_tok.text.lower()
                                mid_pv_aux = ""
                                if is_pv_passive and verb_form_tag == "VBN" and subj_tok.i + 1 < start_i:
                                    aux_toks = [
                                        t.text.lower()
                                        for t in doc[subj_tok.i + 1 : start_i]
                                        if t.pos_ in ("AUX", "PART", "ADV") or t.dep_ in ("aux", "auxpass", "neg", "advmod")
                                    ]
                                    if aux_toks:
                                        mid_pv_aux = " ".join(aux_toks) + " "
                                det_cands = [d for d in subj_tok.children if d.dep_ == "det" and d.i < subj_tok.i and d.is_alpha]
                                if det_cands:
                                    d_low = det_cands[-1].text.lower()
                                    if d_low in ("my", "your", "our"):
                                        d_low = "a" if subj_tok.tag_ == "NN" else "the"
                                    return f"... {d_low} {s_low} {mid_pv_aux}____"
                                if subj_tok.tag_ == "NN":
                                    return f"... a {s_low} {mid_pv_aux}____"
                                return f"... {s_low} {mid_pv_aux}____"
                            if re.search(r"\bher\b", ft_low):
                                return "... she ____"
                            if re.search(r"\bhis\b", ft_low):
                                return "... he ____"
                            if re.search(r"\btheir\b", ft_low):
                                return "... they ____"
                            return "... someone ____"
                        return "... to ____"
                    is_clausal_connector_phrase = (
                        (len(ft_toks) == 3 and ft_toks[0] == "as" and ft_toks[2] == "as")
                        or ft_low in ("even though", "so that", "in order that", "now that", "provided that", "providing that", "as if", "as though")
                    )
                    if is_clausal_connector_phrase and end_i < len(doc) and doc[end_i].is_alpha and not _is_forbidden(doc[end_i].lemma_.lower()):
                        return f"... ____ {doc[end_i].text.lower()}"
                    is_self_contained_pp = (
                        ft_toks[0] in ("in", "on", "at", "by", "for", "under", "over", "from", "within", "without", "beyond", "behind", "between", "among", "through", "throughout", "around", "across")
                        or (len(ft_toks) >= 2 and (ft_toks[0], ft_toks[1]) in (("all", "over"), ("out", "of"), ("up", "to"), ("apart", "from"), ("instead", "of"), ("according", "to")))
                    )
                    if start_i > 0 and doc[start_i - 1].is_alpha and not _is_forbidden(doc[start_i - 1].lemma_.lower()):
                        prev_tok = doc[start_i - 1]
                        prev_w = prev_tok.text.lower()
                        if prev_w in ("and", "or", "got"):
                            return "... have ____"
                        if not (is_self_contained_pp and prev_tok.pos_ in ("ADP", "ADJ", "ADV", "DET", "CCONJ", "SCONJ")):
                            return f"... {prev_w} ____"
                    if end_i < len(doc) and doc[end_i].is_alpha and not _is_forbidden(doc[end_i].lemma_.lower()):
                        return f"... ____ {doc[end_i].text.lower()}"
                    if is_self_contained_pp:
                        for v_tok in doc[ : start_i]:
                            if v_tok.pos_ in ("VERB", "AUX") and v_tok.is_alpha and not _is_forbidden(v_tok.lemma_.lower()):
                                return f"... {v_tok.text.lower()} ... ____"
                        for v_tok in doc[end_i : ]:
                            if v_tok.pos_ in ("VERB", "AUX") and v_tok.is_alpha and not _is_forbidden(v_tok.lemma_.lower()):
                                return f"... ____ ... {v_tok.text.lower()}"
            return None

        # Locate target token / span in `doc` (supporting hyphenated multi-token compounds like `brand-new`)
        t_start_i, t_end_i, t_tok = cls._find_target_span_in_doc(
            doc, ft_low, base_str=bw_low, t_forms=t_forms
        )
        if t_tok is None or t_start_i is None or t_end_i is None:
            return None

        # 2. Adjective (`canonical_pos == "adj"`)
        if canonical_pos == "adj":
            eff_head = t_tok
            while eff_head.dep_ == "conj" and eff_head.head and eff_head.head.pos_ == "ADJ" and eff_head.head.i < eff_head.i:
                eff_head = eff_head.head
            _cop_contraction_map = {
                "'s": "is", "’s": "is",
                "'re": "are", "’re": "are",
                "'m": "am", "’m": "am",
                "'ll": "will", "’ll": "will",
                "'d": "would", "’d": "would",
            }
            # Check for bound preposition immediately following target span (excluding 'as')
            prep_child = next(
                (
                    c for c in list(t_tok.children) + (list(eff_head.children) if eff_head != t_tok else [])
                    if c.dep_ == "prep"
                    and c.i >= t_end_i
                    and c.text.lower() in ("in", "of", "to", "for", "with", "about", "at", "on", "from", "by", "into", "over", "towards", "against")
                ),
                None,
            )
            if prep_child is None and t_end_i < len(doc) and doc[t_end_i].pos_ == "ADP" and doc[t_end_i].text.lower() in (
                "in", "of", "to", "for", "with", "about", "at", "on", "from", "by", "into", "over", "towards", "against"
            ) and eff_head.dep_ != "amod":
                prep_child = doc[t_end_i]
            if (
                is_copula_adj
                or anchor_type == "verb_copula"
                or (prep_child is not None and eff_head.dep_ != "amod")
                or eff_head.dep_ in ("acomp", "oprd", "xcomp", "ccomp")
            ):
                cop_word = anchor.lower() if (anchor_type == "verb_copula" and anchor) else "be"
                gov_v = None
                if eff_head.head and eff_head.head.pos_ in ("AUX", "VERB") and eff_head.head.i < t_start_i:
                    raw_cop = eff_head.head.text.lower()
                    cop_word = _cop_contraction_map.get(raw_cop, raw_cop)
                    gov_v = eff_head.head
                else:
                    for c in list(eff_head.children) + list(t_tok.children):
                        if c.dep_ in ("cop", "aux", "auxpass") and c.i < t_start_i:
                            raw_cop = c.text.lower()
                            cop_word = _cop_contraction_map.get(raw_cop, raw_cop)
                            gov_v = c
                            break

                obj_slot = ""
                if gov_v is not None and gov_v.pos_ == "VERB":
                    interv_obj = next(
                        (
                            c for c in gov_v.children
                            if c.dep_ in ("dobj", "obj") and gov_v.i <= c.i < t_start_i
                        ),
                        None,
                    )
                    if interv_obj is None:
                        interv_obj = next(
                            (
                                c for c in eff_head.children
                                if c.dep_ in ("nsubj", "dobj") and gov_v.i <= c.i < t_start_i
                            ),
                            None,
                        )
                    if interv_obj is not None:
                        is_human_obj = (
                            interv_obj.text.lower() in ("someone", "somebody", "sb", "her", "him", "them", "us", "me")
                            or interv_obj.ent_type_ == "PERSON"
                            or interv_obj.lemma_.lower() in {
                                "person", "people", "man", "men", "woman", "women", "child", "children", "kid", "boy", "girl",
                                "student", "teacher", "professor", "worker", "employee", "staff", "colleague", "customer", "client"
                            }
                        )
                        obj_slot = "[someone] " if is_human_obj else "[something] "

                has_prev_as = (t_start_i > 0 and doc[t_start_i - 1].text.lower() == "as")
                has_next_as = (t_end_i < len(doc) and doc[t_end_i].text.lower() == "as")
                if has_prev_as and has_next_as:
                    correl_tail = "as"
                    if t_end_i + 1 < len(doc) and doc[t_end_i + 1].is_alpha:
                        correl_tail = f"as {doc[t_end_i + 1].text.lower()}"
                    return f"... {cop_word} {obj_slot}as ____ {correl_tail}"

                if prep_child is not None and anchor_type != "verb_copula":
                    return f"... {cop_word} {obj_slot}____ {prep_child.text.lower()}"
                if anchor_type == "prep" and anchor and anchor.lower() != "as":
                    return f"... {cop_word} {obj_slot}____ {anchor.lower()}"
                return f"... {cop_word} {obj_slot}____"

            # Attributive pre-nominal `Adj + Noun` (`eff_head.dep_ == "amod"`)
            if eff_head.dep_ in ("amod", "compound") and eff_head.head and eff_head.head.pos_ in ("NOUN", "PROPN") and t_end_i <= eff_head.head.i:
                head_n = eff_head.head
                head_txt = head_n.text.lower()
                if _is_forbidden(head_n.lemma_.lower()):
                    head_txt = "[noun]"
                det_tok = next(
                    (c for c in head_n.children if c.dep_ in ("det", "poss") and c.i < t_start_i),
                    None,
                )
                det_str = _normalize_det(det_tok.text if det_tok else "", det_tok_obj=det_tok)
                if det_str:
                    return f"... {det_str} ____ {head_txt}"
                if t_start_i > 0 and doc[t_start_i - 1].pos_ == "ADP" and doc[t_start_i - 1].is_alpha:
                    return f"... {doc[t_start_i - 1].text.lower()} ____ {head_txt}"
                return f"... ____ {head_txt}"

            # Fallback if immediately followed by a noun in surface order
            if t_end_i < len(doc) and doc[t_end_i].pos_ in ("NOUN", "PROPN"):
                nxt_n = doc[t_end_i]
                head_txt = "[noun]" if _is_forbidden(nxt_n.lemma_.lower()) else nxt_n.text.lower()
                prev_det = doc[t_start_i - 1].text if (t_start_i > 0 and doc[t_start_i - 1].pos_ in ("DET", "PRON")) else ""
                det_str = _normalize_det(prev_det)
                if det_str:
                    return f"... {det_str} ____ {head_txt}"
                if t_start_i > 0 and doc[t_start_i - 1].pos_ == "ADP" and doc[t_start_i - 1].is_alpha:
                    return f"... {doc[t_start_i - 1].text.lower()} ____ {head_txt}"
                return f"... ____ {head_txt}"

            if prep_child is not None:
                return f"... ____ {prep_child.text.lower()}"
            if t_start_i > 0 and doc[t_start_i - 1].is_alpha and doc[t_start_i - 1].pos_ not in ("NOUN", "PROPN"):
                prev_w = _cop_contraction_map.get(doc[t_start_i - 1].text.lower(), doc[t_start_i - 1].text.lower())
                if prev_w in ("and", "or") and eff_head.head and eff_head.head.pos_ in ("AUX", "VERB"):
                    prev_w = _cop_contraction_map.get(eff_head.head.text.lower(), eff_head.head.text.lower())
                return f"... {prev_w} ____"
            return "... the ____ [noun]"

        # 3. Verb (`canonical_pos == "verb"`)
        if canonical_pos == "verb":
            if t_tok.pos_ in ("NOUN", "PROPN") and t_tok.dep_ in ("compound", "pobj", "dobj", "attr"):
                return f"... ____ {patient_ontology_slot or '[object]'}"
            left_aux = ""
            _aux_link_verbs = (
                "has", "have", "had", "is", "are", "was", "were", "been", "being", "be",
                "get", "got", "seem", "seemed", "seems", "become", "became", "becomes",
                "appear", "appeared", "appears", "feel", "felt", "look", "looked", "remain", "remained",
            )
            if t_tok.i > 0:
                prev_t = doc[t_tok.i - 1]
                if (
                    verb_form_tag in ("VBN", "VBG")
                    and (prev_t.pos_ in ("AUX", "ADP") or prev_t.text.lower() in _aux_link_verbs)
                ) or (
                    verb_form_tag == "VBD"
                    and prev_t.text.lower() in _aux_link_verbs
                ):
                    if prev_t.text.lower() in ("been", "being") and t_tok.i > 1 and doc[t_tok.i - 2].text.lower() in ("has", "have", "had", "is", "are", "was", "were"):
                        left_aux = f"{doc[t_tok.i - 2].text.lower()} {prev_t.text.lower()} "
                    elif (
                        verb_form_tag == "VBG"
                        and prev_t.pos_ == "ADP"
                        and t_tok.i > 1
                        and doc[t_tok.i - 2].pos_ in ("VERB", "ADJ", "NOUN", "ADV")
                        and doc[t_tok.i - 2].is_alpha
                        and not _is_forbidden(doc[t_tok.i - 2].lemma_.lower())
                    ):
                        left_aux = f"{doc[t_tok.i - 2].text.lower()} {prev_t.text.lower()} "
                    else:
                        left_aux = prev_t.text.lower() + " "
                elif (
                    verb_form_tag in ("VBN", "VBG")
                    and prev_t.pos_ == "ADV"
                    and t_tok.i > 1
                    and (doc[t_tok.i - 2].pos_ == "AUX" or doc[t_tok.i - 2].text.lower() in _aux_link_verbs)
                    and not _is_forbidden(prev_t.lemma_.lower())
                ):
                    left_aux = f"{doc[t_tok.i - 2].text.lower()} {prev_t.text.lower()} "

            # Intransitive / Ergative Inchoative / Passive verb flag
            is_tok_passive = any(c.dep_ in ("auxpass", "nsubjpass") for c in t_tok.children)

            # 3a. Infinitive complement (`to + V-base`)
            if (
                t_tok.i + 2 < len(doc)
                and doc[t_tok.i + 1].text.lower() == "to"
                and (doc[t_tok.i + 1].tag_ == "TO" or doc[t_tok.i + 2].pos_ in ("VERB", "AUX"))
            ):
                return f"... {left_aux}____ to [verb]"

            # 3b. Bound preposition complement (`pose for`, `correlate with`, `originate in`, `remind sb of`, `endow sb with`)
            prep_c = next(
                (
                    c for c in t_tok.children
                    if (c.dep_ == "prep" or (c.dep_ == "agent" and anchor_type == "prep"))
                    and not (is_tok_passive and c.text.lower() == "by" and anchor_type != "prep")
                    and c.i > t_tok.i
                    and c.text.lower() in ("for", "with", "in", "into", "to", "from", "on", "of", "at", "about", "against", "over", "under", "between", "as", "upon", "towards", "by")
                ),
                None,
            )
            if (
                prep_c is not None
                and anchor_type != "verb_subject"
                and (anchor_type == "prep" or not any(c.dep_ in ("dobj", "obj") and c.i < prep_c.i for c in t_tok.children))
            ):
                mid_dobj = next(
                    (c for c in t_tok.children if c.dep_ in ("dobj", "obj") and t_tok.i < c.i < prep_c.i),
                    None,
                )
                if mid_dobj is not None:
                    _md_low = mid_dobj.text.lower()
                    if patient_ontology_slot:
                        _mid_slot = patient_ontology_slot
                    elif _md_low.endswith(("self", "selves")):
                        _mid_slot = "oneself"
                    elif _md_low in ("me", "you", "him", "her", "us", "whom", "someone", "somebody", "everyone", "anyone", "nobody") or mid_dobj.ent_type_ == "PERSON":
                        _mid_slot = "someone"
                    else:
                        _mid_slot = cls._resolve_object_animacy_from_example(ex_clean, ft_low, default_tag="something")
                    return f"... {left_aux}____ {_mid_slot} {prep_c.text.lower()}"
                return f"... {left_aux}____ {prep_c.text.lower()}"

            # 3c. Direct object (`dobj` / `obj` / `oprd`)
            _temporal_np_nouns = {
                "winter", "summer", "spring", "autumn", "fall", "morning", "afternoon",
                "evening", "night", "day", "week", "month", "year", "time", "hour", "minute",
            }
            dobj_c = (
                next(
                    (
                        c for c in t_tok.children
                        if c.dep_ in ("dobj", "obj", "oprd")
                        and c.i > t_tok.i
                        and c.lemma_.lower() not in _temporal_np_nouns
                    ),
                    None,
                )
                or next(
                    (
                        c for c in t_tok.children
                        if c.dep_ in ("dobj", "obj", "oprd")
                        and c.i > t_tok.i
                    ),
                    None,
                )
                if anchor_type != "verb_subject"
                else None
            )
            if dobj_c is not None:
                _partitive_quant_lemmas = {
                    "ton", "pile", "load", "heap", "stack", "hundred", "thousand", "million",
                    "lot", "plenty", "piece", "bit", "drop", "slice", "amount", "quantity",
                    "number", "type", "kind", "sort", "series", "range", "variety",
                }
                partitive_of_tail = ""
                if dobj_c.lemma_.lower() in _partitive_quant_lemmas:
                    prep_of_c = next(
                        (
                            ch for ch in dobj_c.children
                            if ch.dep_ == "prep" and ch.text.lower() == "of" and ch.i == dobj_c.i + 1
                        ),
                        None,
                    )
                    if prep_of_c is not None:
                        pobj_of_c = next(
                            (
                                pch for pch in prep_of_c.children
                                if pch.dep_ == "pobj" and pch.pos_ in ("NOUN", "PROPN") and not _is_forbidden(pch.lemma_.lower())
                            ),
                            None,
                        )
                        if pobj_of_c is not None:
                            partitive_of_tail = f" of {pobj_of_c.text.lower()}"
                # Check for a trailing particle (`tip his chair back`) or governed trivalent preposition (`attach importance to`)
                trail_prt_or_prep = ""
                if not partitive_of_tail and dobj_c.i + 1 < len(doc):
                    nxt_after_obj = doc[dobj_c.i + 1]
                    if nxt_after_obj.dep_ == "prt" and nxt_after_obj.head == t_tok:
                        trail_prt_or_prep = f" {nxt_after_obj.text.lower()}"
                    elif (
                        nxt_after_obj.dep_ == "prep"
                        and (nxt_after_obj.head == t_tok or nxt_after_obj.head == dobj_c)
                        and nxt_after_obj.text.lower() in ("to", "into", "onto", "on", "from", "with", "for", "against", "in", "at", "upon", "towards", "as")
                    ):
                        trail_prt_or_prep = f" {nxt_after_obj.text.lower()}"
                if patient_ontology_slot and not partitive_of_tail:
                    return f"... {left_aux}____ {patient_ontology_slot}{trail_prt_or_prep}"
                if dobj_c.pos_ == "PRON" or dobj_c.ent_type_ == "PERSON" or _is_forbidden(dobj_c.lemma_.lower()):
                    anim = cls._resolve_object_animacy_from_example(ex_clean, ft_low, default_tag="[sb/sth]")
                    return f"... {left_aux}____ {anim}{trail_prt_or_prep}"
                if dobj_c.pos_ in ("NOUN", "PROPN"):
                    # Collect determiners/quantifiers/numbers between `t_tok` and `dobj_c`
                    raw_span = list(doc[t_tok.i + 1 : dobj_c.i + 1])
                    poss_idx = next((idx for idx, t in enumerate(raw_span[:-1]) if t.dep_ == "poss" and t.pos_ in ("NOUN", "PROPN")), None)
                    if poss_idx is not None:
                        after_poss = [t.text.lower() for t in raw_span[poss_idx + 1 :] if (t.is_alpha or t.like_num) and not _is_forbidden(t.lemma_.lower())]
                        span_toks = ["the"] + after_poss if after_poss else [dobj_c.text.lower()]
                    else:
                        span_toks = [
                            t.text.lower()
                            for t in raw_span
                            if (t.is_alpha or t.like_num) and not _is_forbidden(t.lemma_.lower())
                        ]
                    if 1 <= len(span_toks) <= 4:
                        return f"... {left_aux}____ {' '.join(span_toks)}{partitive_of_tail}{trail_prt_or_prep}"
                    return f"... {left_aux}____ {dobj_c.text.lower()}{partitive_of_tail}{trail_prt_or_prep}"

            # 3d. Intransitive / Ergative Inchoative / Passive verb with subject (`the custom ____`, `the lake had ____ overnight`)
            is_tok_passive = any(c.dep_ in ("auxpass", "nsubjpass") for c in t_tok.children)
            subj_c = next(
                (c for c in t_tok.children if c.dep_ in ("nsubj", "nsubjpass") and c.i < t_tok.i),
                None,
            )
            if (
                subj_c is None
                and t_tok.dep_ == "amod"
                and t_tok.head.dep_ == "ROOT"
                and t_tok.head.pos_ in ("NOUN", "PROPN")
                and t_tok.i > t_tok.head.i
            ):
                subj_c = t_tok.head
            xcomp_bridge = ""
            if subj_c is None and t_tok.dep_ == "xcomp" and t_tok.head.pos_ in ("VERB", "AUX") and t_tok.head.i < t_tok.i:
                subj_c = next(
                    (c for c in t_tok.head.children if c.dep_ in ("nsubj", "nsubjpass") and c.i < t_tok.head.i),
                    None,
                )
                if subj_c is not None:
                    bridge_toks = [t.text.lower() for t in doc[t_tok.head.i : t_tok.i] if (t.is_alpha or "'" in t.text) and not _is_forbidden(t.lemma_.lower())]
                    if 1 <= len(bridge_toks) <= 3:
                        xcomp_bridge = " ".join(bridge_toks) + " "
            if subj_c is not None and not _is_forbidden(subj_c.lemma_.lower()):
                mid_aux = left_aux
                if not mid_aux and not xcomp_bridge and subj_c.i + 1 < t_tok.i:
                    aux_span = [
                        t.text.lower()
                        for t in doc[subj_c.i + 1 : t_tok.i]
                        if (t.pos_ in ("AUX", "PART") or t.dep_ in ("aux", "auxpass", "neg"))
                    ]
                    if aux_span:
                        mid_aux = "".join(
                            (tok_s if tok_s.startswith("'") or tok_s == "n't" else f" {tok_s}")
                            for tok_s in aux_span
                        ).strip() + " "
                post_adv = ""
                if t_tok.i + 1 < len(doc):
                    nxt_t = doc[t_tok.i + 1]
                    if (
                        nxt_t.dep_ in ("prt", "advmod")
                        and nxt_t.is_alpha
                        and not _is_forbidden(nxt_t.lemma_.lower())
                    ):
                        post_adv = f" {nxt_t.text.lower()}"
                # Deep Passive Projection: if the verb in the example is passive (`subj_c` is Patient / 受动者)
                # but the quiz options are in active base form (`verb_form_tag == "VB"`), never place the Patient
                # before a base-form verb (which would invert 受动者 into 施动者); project Patient into the object slot.
                if is_tok_passive and verb_form_tag not in ("VBD", "VBZ", "VBG", "VBN"):
                    if patient_ontology_slot:
                        return f"... {left_aux}____ {patient_ontology_slot}{post_adv}"
                    if subj_c.pos_ in ("NOUN", "PROPN"):
                        s_det = next((c.text.lower() for c in subj_c.children if c.dep_ in ("det", "poss") and c.i < subj_c.i), "")
                        if not s_det and subj_c.tag_ == "NN":
                            s_det = "the"
                        s_prefix = f"{s_det} {subj_c.text.lower()}".strip()
                        return f"... {left_aux}____ {s_prefix}{post_adv}"
                # Active Transitive Relative-Clause Guard: if `anchor_type == "object"` in active voice
                # (meaning the direct object was fronted in a relative/wh clause and `subj_c` is the active Agent / 施动者),
                # do not slice a dangling `... [Agent] ____` frame without its object.
                elif not is_tok_passive and anchor_type == "object":
                    if patient_ontology_slot:
                        return f"... {left_aux}____ {patient_ontology_slot}{post_adv}"
                    if (
                        t_tok.dep_ == "relcl"
                        and t_tok.head.pos_ in ("NOUN", "PROPN")
                        and not _is_forbidden(t_tok.head.lemma_.lower())
                    ):
                        rel_det = "the " if t_tok.head.tag_ == "NN" else ""
                        return f"... {left_aux}____ {rel_det}{t_tok.head.text.lower()}{post_adv}"
                    if anchor and not _is_forbidden(str(anchor).lower()):
                        return f"... {left_aux}____ the {str(anchor).lower()}{post_adv}"
                else:
                    if is_tok_passive and verb_form_tag == "VBN" and patient_ontology_slot:
                        _raw_p_aux = xcomp_bridge or mid_aux or "was/were "
                        _raw_p_aux = re.sub(r"\b(?:was|were)\b", "was/were", _raw_p_aux)
                        _raw_p_aux = re.sub(r"\b(?:is|are)\b", "is/are", _raw_p_aux)
                        _raw_p_aux = re.sub(r"\b(?:has|have)\b", "has/have", _raw_p_aux)
                        return f"... {patient_ontology_slot} {_raw_p_aux}____{post_adv}"
                    if subj_c.pos_ == "PRON":
                        if patient_ontology_slot:
                            _aux_norm = xcomp_bridge or mid_aux or ""
                            if is_tok_passive:
                                _aux_norm = _aux_norm or "was/were "
                                _aux_norm = re.sub(r"\b(?:was|were)\b", "was/were", _aux_norm)
                                _aux_norm = re.sub(r"\b(?:is|are)\b", "is/are", _aux_norm)
                                _aux_norm = re.sub(r"\b(?:has|have)\b", "has/have", _aux_norm)
                            return f"... {patient_ontology_slot} {_aux_norm}____{post_adv}".replace("  ", " ")
                        return f"... {subj_c.text.lower()} {xcomp_bridge or mid_aux}____"
                    s_det = next((c.text.lower() for c in subj_c.children if c.dep_ in ("det", "poss") and c.i < subj_c.i), "")
                    s_prefix = f"{s_det} {subj_c.text.lower()}".strip()
                    return f"... {s_prefix} {xcomp_bridge or mid_aux}____{post_adv}"

            if left_aux:
                return f"... {left_aux}____"

            if prep_c is not None:
                return f"... {left_aux}____ {prep_c.text.lower()}"
            if (
                t_tok.i + 1 < len(doc)
                and doc[t_tok.i + 1].is_alpha
                and doc[t_tok.i + 1].pos_ in ("ADP", "PART", "ADV")
                and doc[t_tok.i + 1].text.lower() not in ("next", "last", "this", "that", "every", "each", "all", "both", "today", "yesterday", "tomorrow")
                and not _is_forbidden(doc[t_tok.i + 1].lemma_.lower())
            ):
                return f"... {left_aux}____ {doc[t_tok.i + 1].text.lower()}"
            if anchor and anchor_type == "prep" and not _is_forbidden(str(anchor).lower()):
                return f"... {left_aux}____ {str(anchor).lower()}"
            if t_tok.i > 0 and doc[t_tok.i - 1].is_alpha and doc[t_tok.i - 1].pos_ not in ("DET", "ADP", "CCONJ", "SCONJ") and not _is_forbidden(doc[t_tok.i - 1].lemma_.lower()):
                return f"... {doc[t_tok.i - 1].text.lower()} ____"
            return f"... {left_aux}____ {patient_ontology_slot or '[object]'}"

        # 4. Noun (`canonical_pos == "noun"`)
        if canonical_pos == "noun":
            det_tok = next(
                (c for c in t_tok.children if c.dep_ in ("det", "poss") and c.i < t_tok.i),
                None,
            )
            amod_tok = next(
                (c for c in t_tok.children if c.dep_ in ("amod", "compound") and c.i < t_tok.i and c.is_alpha and not _is_forbidden(c.lemma_.lower())),
                None,
            )
            if amod_tok is None and t_tok.i > 0 and doc[t_tok.i - 1].pos_ == "ADJ" and doc[t_tok.i - 1].is_alpha and not _is_forbidden(doc[t_tok.i - 1].lemma_.lower()):
                amod_tok = doc[t_tok.i - 1]
            amod_txt = ""
            amod_first_i = amod_tok.i if amod_tok is not None else t_tok.i
            if amod_tok is not None:
                if (
                    amod_tok.i >= 2
                    and doc[amod_tok.i - 1].text in ("-", "–")
                    and (doc[amod_tok.i - 2].is_alpha or doc[amod_tok.i - 2].like_num)
                    and not _is_forbidden(doc[amod_tok.i - 2].lemma_.lower())
                ):
                    amod_txt = f"{doc[amod_tok.i - 2].text.lower()}-{amod_tok.text.lower()}"
                    amod_first_i = amod_tok.i - 2
                else:
                    amod_txt = amod_tok.text.lower()
            if det_tok is None:
                prev_det_i = (amod_first_i - 1) if amod_tok is not None else (t_tok.i - 1)
                if prev_det_i >= 0 and doc[prev_det_i].text.lower() in ("a", "an", "the", "each", "every", "his", "her", "their", "its", "our", "my", "your", "no"):
                    det_tok = doc[prev_det_i]
            default_noun_det = (
                ("several" if is_plural_noun else ("each" if has_mixed_vc_initials else "a"))
                if has_uncountable_contrast_foil
                else ("a" if is_target_strictly_countable and not is_plural_noun and not has_mixed_vc_initials else ("the" if not is_pure_uncountable_target else ""))
            )
            det_str = _normalize_det(
                det_tok.text if det_tok else default_noun_det,
                det_tok_obj=det_tok,
            )
            det_for_amod = ""
            if amod_txt:
                if det_tok is not None:
                    raw_d = det_tok.text.strip().lower()
                    if raw_d.endswith("'s") or raw_d.endswith("’s") or (det_tok.dep_ == "poss" and det_tok.pos_ in ("NOUN", "PROPN", "ADJ")):
                        det_for_amod = "their"
                    elif raw_d in ("a", "an"):
                        if is_pure_uncountable_target or is_plural_noun:
                            det_for_amod = ("several" if has_uncountable_contrast_foil else "the") if is_plural_noun else ""
                        else:
                            det_for_amod = "an" if amod_txt[0] in "aeiou" else "a"
                    else:
                        det_for_amod = _normalize_det(raw_d, det_tok_obj=det_tok)
                elif not is_pure_uncountable_target and not is_plural_noun:
                    det_for_amod = ("an" if amod_txt[0] in "aeiou" else "a") if (is_target_strictly_countable or has_uncountable_contrast_foil) else "the"
            prep_after = next(
                (c for c in t_tok.children if c.dep_ == "prep" and c.i == t_tok.i + 1 and c.text.lower() in ("of", "for", "in", "to", "on", "with", "over", "between", "from", "into", "against", "towards")),
                None,
            )
            if prep_after is None and t_tok.i + 1 < len(doc) and doc[t_tok.i + 1].pos_ == "ADP" and doc[t_tok.i + 1].text.lower() in ("of", "for", "in", "to", "on", "with", "over", "between", "from"):
                prep_after = doc[t_tok.i + 1]

            # 4.0 Noun + infinitive complement (`a ____ to [verb]`, `several ____ to detect`)
            if (
                t_tok.i + 2 < len(doc)
                and doc[t_tok.i + 1].text.lower() == "to"
                and (doc[t_tok.i + 1].tag_ == "TO" or doc[t_tok.i + 2].pos_ in ("VERB", "AUX"))
            ):
                d_shell = det_str or ("a" if not is_pure_uncountable_target and not is_plural_noun and not has_mixed_vc_initials else "the")
                v_inf_tok = doc[t_tok.i + 2]
                v_inf_lem = v_inf_tok.lemma_.lower()
                if anchor_type != "shell_to_do" and v_inf_lem not in cls._DELEXICAL_VERBS and not _is_forbidden(v_inf_lem):
                    return f"... {d_shell + ' ' if d_shell else ''}____ to {v_inf_tok.text.lower()}"
                return f"... {d_shell + ' ' if d_shell else ''}____ to [verb]"

            # 4a. Direct object of a lexical or support verb (`take out a ____`, `make a large ____ to`, `take a ten-minute ____`)
            _SUPPORT_VERBS_NOUN = {
                "make", "take", "give", "pay", "hold", "lose", "gain", "keep",
                "put", "set", "bear", "lay", "draw", "reach", "strike", "pose", "stand",
            }
            if t_tok.dep_ in ("dobj", "obj") and t_tok.head and t_tok.head.pos_ == "VERB":
                v_head = t_tok.head
                v_lem = v_head.lemma_.lower()
                if (
                    v_lem not in cls._DELEXICAL_VERBS
                    or v_lem in _SUPPORT_VERBS_NOUN
                    or (v_lem in ("have", "get") and (amod_txt or prep_after is not None))
                ) and not _is_forbidden(v_lem):
                    prt_tok = next((c.text.lower() for c in v_head.children if c.dep_ == "prt" and c.i < t_tok.i), "")
                    v_surface = v_head.text.lower() if v_head.tag_ in ("VBD", "VBN", "VB", "VBP") else v_lem
                    v_phrase = f"{v_surface} {prt_tok}".strip()
                    if amod_txt:
                        np_mid_str = f"{det_for_amod} {amod_txt}".strip()
                        np_mid = f"{np_mid_str} " if np_mid_str else ""
                    else:
                        np_mid = f"{det_str} " if det_str else ""
                    if prep_after is not None:
                        return f"... {v_phrase} {np_mid}____ {prep_after.text.lower()}"
                    return f"... {v_phrase} {np_mid}____"

            # 4b. Pre-nominal adjective modifier + noun (+ optional prep) (`a personal ____ for`, `an economic ____`)
            if amod_tok is not None and amod_txt:
                np_mid = f"{det_for_amod} {amod_txt}".strip()
                if prep_after is not None:
                    return f"... {np_mid} ____ {prep_after.text.lower()}"
                return f"... {np_mid} ____"

            # 4c. Noun + bound preposition (`the ____ of`, `a ____ for`)
            if prep_after is not None:
                d_pref = det_str or (default_noun_det if not is_pure_uncountable_target else "")
                return f"... {d_pref + ' ' if d_pref else ''}____ {prep_after.text.lower()}"

            # 4d. Subject of a lexical verb (`the ____ unfolds`)
            if t_tok.dep_ in ("nsubj", "nsubjpass") and t_tok.head and t_tok.head.pos_ in ("VERB", "AUX") and t_tok.head.i > t_tok.i:
                v_head = t_tok.head
                if v_head.lemma_.lower() not in cls._DELEXICAL_VERBS and not _is_forbidden(v_head.lemma_.lower()):
                    d_pref = det_str or "the"
                    return f"... {d_pref} ____ {v_head.text.lower()}"

            # 4e. Object of a preposition (`in ____`, `on the ____`, `under ____`)
            if t_tok.dep_ == "pobj" and t_tok.head and t_tok.head.dep_ == "prep":
                p_left = t_tok.head.text.lower()
                det_in_ex = _normalize_det(det_tok.text, det_tok_obj=det_tok) if det_tok is not None else ""
                # Sentence-initial prepositional adjunct followed by comma: "By convention, ...", "In theory, ..."
                is_initial_prep = (
                    t_tok.head.i == 0
                    or getattr(t_tok.head, "is_sent_start", False)
                    or (t_tok.head.i > 0 and doc[t_tok.head.i - 1].text in (".", "!", "?", ";"))
                )
                if is_initial_prep and t_end_i < len(doc) and doc[t_end_i].text in (",", ";"):
                    return f"{p_left.capitalize()} {det_in_ex + ' ' if det_in_ex else ''}____, ..."
                return f"... {p_left} {det_in_ex + ' ' if det_in_ex else ''}____"

            d_pref = det_str or "the"
            return f"... {d_pref} ____"

        # 5. Adverb (`canonical_pos == "adv"`)
        if canonical_pos == "adv":
            # 5a. Sentence-initial discourse/conjunctive adverb followed by comma ('Similarly, we find...')
            if (
                (
                    t_tok.i == 0
                    or getattr(t_tok, "is_sent_start", False)
                    or (t_tok.i > 0 and doc[t_tok.i - 1].text in (".", "!", "?", ";"))
                )
                and t_tok.i + 1 < len(doc)
                and doc[t_tok.i + 1].text in (",", ";")
            ):
                return "____, ..."
            # 5b. `hence + the/poss + NP` ('hence the tight security' -> '... ____ the tight security')
            if (
                ft_low in ("hence", "thus", "notably", "especially", "particularly", "mainly", "mostly", "chiefly", "even", "only")
                and t_tok.i + 2 < len(doc)
                and doc[t_tok.i + 1].text.lower() in ("the", "his", "her", "their", "our", "its", "this", "that", "these", "those")
            ):
                _det_t = doc[t_tok.i + 1].text.lower()
                if doc[t_tok.i + 2].pos_ in ("NOUN", "PROPN") and not _is_forbidden(doc[t_tok.i + 2].lemma_.lower()):
                    return f"... ____ {_det_t} {doc[t_tok.i + 2].text.lower()}"
                if (
                    t_tok.i + 3 < len(doc)
                    and doc[t_tok.i + 2].pos_ == "ADJ"
                    and doc[t_tok.i + 3].pos_ in ("NOUN", "PROPN")
                    and not _is_forbidden(doc[t_tok.i + 3].lemma_.lower())
                ):
                    return f"... ____ {_det_t} {doc[t_tok.i + 2].text.lower()} {doc[t_tok.i + 3].text.lower()}"
            # 5c. Immediately following adjective (`looked vaguely familiar` -> `... ____ familiar`) or adjacent modified head (`Adv + Adj/Verb` or `Verb + Adv`)
            if (
                t_tok.i + 1 < len(doc)
                and doc[t_tok.i + 1].is_alpha
                and doc[t_tok.i + 1].pos_ == "ADJ"
                and not doc[t_tok.i + 1].text[:1].isupper()
                and doc[t_tok.i + 1].lemma_.lower() not in cls._ANCHOR_STOPWORDS
                and not _is_forbidden(doc[t_tok.i + 1].lemma_.lower())
            ):
                return f"... ____ {doc[t_tok.i + 1].text.lower()}"
            if (
                t_tok.head
                and t_tok.head != t_tok
                and t_tok.head.is_alpha
                and t_tok.head.pos_ in ("VERB", "ADJ")
                and t_tok.head.lemma_.lower() not in cls._ANCHOR_STOPWORDS
                and not _is_forbidden(t_tok.head.lemma_.lower())
                and abs(t_tok.i - t_tok.head.i) == 1
            ):
                if t_tok.i < t_tok.head.i:
                    return f"... ____ {t_tok.head.text.lower()}"
                return f"... {t_tok.head.text.lower()} ____"
            # 5d. Degree/temporal modifier immediately preceding and modifying adverb ('shortly thereafter' -> '... shortly ____')
            if (
                t_tok.i > 0
                and doc[t_tok.i - 1].is_alpha
                and doc[t_tok.i - 1].pos_ == "ADV"
                and (doc[t_tok.i - 1].head == t_tok or t_tok.head == doc[t_tok.i - 1])
                and not _is_forbidden(doc[t_tok.i - 1].lemma_.lower())
                and doc[t_tok.i - 1].text.lower() not in ("not", "never", "also", "even", "just", "only")
            ):
                return f"... {doc[t_tok.i - 1].text.lower()} ____"
            # 5e. `Verb + Object (Pronoun/NP) + Clause-Final Adverb` ('We must stop him somehow.' -> '... stop someone ____', 'handle the chemical responsibly' -> '... handle something ____')
            if (
                t_tok.head
                and t_tok.head != t_tok
                and t_tok.head.is_alpha
                and t_tok.head.pos_ == "VERB"
                and t_tok.head.lemma_.lower() not in cls._ANCHOR_STOPWORDS
                and not _is_forbidden(t_tok.head.lemma_.lower())
                and t_tok.i > t_tok.head.i
            ):
                _mid_obj = next(
                    (
                        c for c in t_tok.head.children
                        if c.dep_ in ("dobj", "obj", "attr") and t_tok.head.i < c.i < t_tok.i
                    ),
                    None,
                )
                if _mid_obj is not None:
                    _obj_slot = (
                        "someone"
                        if (
                            _mid_obj.text.lower() in ("him", "her", "me", "us", "you", "someone", "somebody", "anyone", "everyone")
                            or _mid_obj.ent_type_ == "PERSON"
                        )
                        else "something"
                    )
                    return f"... {t_tok.head.text.lower()} {_obj_slot} ____"
                return f"... {t_tok.head.text.lower()} ____"
            # 5f. Adjective modifier immediately preceding adverb
            if (
                t_tok.i > 0
                and doc[t_tok.i - 1].is_alpha
                and doc[t_tok.i - 1].pos_ in ("ADV", "ADJ")
                and not _is_forbidden(doc[t_tok.i - 1].lemma_.lower())
                and doc[t_tok.i - 1].text.lower() not in ("not", "never", "also", "even", "just", "only")
            ):
                return f"... {doc[t_tok.i - 1].text.lower()} ____"
            if (
                t_tok.i + 1 < len(doc)
                and doc[t_tok.i + 1].is_alpha
                and doc[t_tok.i + 1].pos_ in ("VERB", "ADJ")
                and doc[t_tok.i + 1].lemma_.lower() not in cls._ANCHOR_STOPWORDS
                and not _is_forbidden(doc[t_tok.i + 1].lemma_.lower())
            ):
                return f"... ____ {doc[t_tok.i + 1].text.lower()}"
            if (
                t_tok.i > 0
                and doc[t_tok.i - 1].is_alpha
                and doc[t_tok.i - 1].pos_ in ("VERB", "ADJ")
                and doc[t_tok.i - 1].lemma_.lower() not in cls._ANCHOR_STOPWORDS
                and not _is_forbidden(doc[t_tok.i - 1].lemma_.lower())
            ):
                return f"... {doc[t_tok.i - 1].text.lower()} ____"
            if t_tok.i + 1 < len(doc) and doc[t_tok.i + 1].is_alpha and doc[t_tok.i + 1].lemma_.lower() not in cls._ANCHOR_STOPWORDS and not _is_forbidden(doc[t_tok.i + 1].lemma_.lower()):
                return f"... ____ {doc[t_tok.i + 1].text.lower()}"
            if t_tok.i > 0 and doc[t_tok.i - 1].is_alpha and doc[t_tok.i - 1].lemma_.lower() not in cls._ANCHOR_STOPWORDS and not _is_forbidden(doc[t_tok.i - 1].lemma_.lower()):
                return f"... {doc[t_tok.i - 1].text.lower()} ____"

        return None

    @classmethod
    def collocation_half_slice(
        cls,
        phrase: str,
        sentence: str,
        pos: str = "",
    ) -> Optional[Dict[str, Any]]:
        """
        SLA Lexical Approach: Collocation-Slot Half-Slice Cloze (搭配槽位半切片完形).
        Decomposes multi-word collocations, phrasal verbs, and prepositional frames into:
          - A tested single-word slot (adjective collocator, light verb, phrasal particle, noun slot)
          - Preserved collocational anchor in the sentence stem (e.g. '____ praise', 'is ____ with')
          - Pure single-word distractors targeting learner L1 negative transfer & collocational clash.
        Guarantees zero whole-chunk masking, zero length disparity, and maximum psychometric discrimination.
        """
        if not phrase or not sentence:
            return None

        clean_p = re.sub(r"[\(\[][^\)\]]+[\)\]]", "", phrase).strip()
        p_words = [w.lower() for w in clean_p.split() if w]
        if len(p_words) < 2:
            return None

        nlp = cls.get_spacy()
        doc = nlp(sentence)
        s_lower = sentence.lower()

        _DEGREE_ADJ_DISTRACTORS = {
            "high": ["tall", "deep", "heavy"],
            "heavy": ["strong", "thick", "deep"],
            "deep": ["heavy", "high", "thick"],
            "strong": ["heavy", "deep", "powerful"],
            "sharp": ["steep", "high", "bitter"],
            "broad": ["wide", "large", "thick"],
            "profound": ["intense", "severe", "deep"],
            "great": ["tall", "high", "heavy"],
        }
        _LIGHT_VERB_DISTRACTORS = {
            "make": ["take", "give", "do"],
            "made": ["taken", "given", "done"],
            "makes": ["takes", "gives", "does"],
            "making": ["taking", "giving", "doing"],
            "take": ["make", "play", "have"],
            "took": ["made", "played", "had"],
            "takes": ["makes", "plays", "has"],
            "taking": ["making", "playing", "having"],
            "give": ["make", "take", "do"],
            "gave": ["made", "took", "did"],
            "gives": ["makes", "takes", "does"],
            "giving": ["making", "taking", "doing"],
            "pay": ["give", "make", "lend"],
            "paid": ["given", "made", "lent"],
            "cast": ["throw", "shed", "put"],
            "shed": ["cast", "throw", "shine"],
            "strike": ["keep", "make", "hold"],
            "struck": ["kept", "made", "held"],
        }
        _PREP_NOUN_DISTRACTORS = {
            "ease": ["rest", "peace", "comfort"],
            "advance": ["progress", "forward", "ahead"],
            "purpose": ["reason", "intent", "aim"],
            "risk": ["danger", "threat", "hazard"],
            "hand": ["reach", "touch", "arm"],
            "vain": ["waste", "loss", "void"],
            "fact": ["truth", "reality", "deed"],
            "peace": ["ease", "rest", "calm"],
            "turn": ["time", "round", "shift"],
            "accident": ["mishap", "misfortune", "casualty"],
        }
        _PARTICIPLE_DISTRACTORS = {
            "filled": ["loaded", "crowded", "packed"],
            "based": ["founded", "grounded", "modeled"],
            "equipped": ["furnished", "fitted", "armed"],
            "associated": ["connected", "linked", "related"],
            "proud": ["fond", "aware", "jealous"],
            "short": ["free", "devoid", "clear"],
            "full": ["short", "empty", "bare"],
        }

        # 1. Copular / Passive + Participle/Adjective + Preposition (e.g. 'be filled with', 'be new to', 'be proud of')
        m_cop = re.search(
            r"(?:\b(?:is|was|are|were|be|been|being)|'(?:re|s|m))\s+([a-zA-Z'-]+)\s+([a-zA-Z]+)\b",
            sentence,
            re.IGNORECASE,
        )
        if m_cop and len(p_words) >= 2:
            cand_adj = m_cop.group(1).lower()
            cand_prep = m_cop.group(2).lower()
            adj_ent = cls.get_ldoce_entry(cand_adj)
            all_p = [p.lower() for p in ((adj_ent or {}).get("all_poses", []) or [])]
            is_valid_adj_participle = (
                cand_adj in _PARTICIPLE_DISTRACTORS
                or (adj_ent and ("adjective" in (adj_ent.get("pos") or "").lower() or "adjective" in all_p))
                or bool(cls._wordnet_pos(cand_adj) & {"a", "s"})
                or (cand_adj.endswith(("ed", "ing", "en")) and len(cand_adj) > 4)
            ) and cand_adj not in ("worry", "care", "think", "talk", "hear", "know", "speak", "tell", "ask", "look")
            if is_valid_adj_participle and cand_adj in p_words and (cand_prep in p_words or cand_prep in ("with", "of", "on", "in", "by", "for", "about", "to")):
                v_tok_str = m_cop.group(1)
                v_start = s_lower.find(v_tok_str.lower(), m_cop.start())
                v_end = v_start + len(v_tok_str)
                stem = sentence[:v_start] + "____" + sentence[v_end:]
                target_tok = sentence[v_start:v_end]
                dists = _PARTICIPLE_DISTRACTORS.get(v_tok_str.lower())
                if not dists:
                    dists = cls.generate_zero_collision_distractors(target_tok, pos="a", count=6, context_sentence=sentence)
                    dists = [d for d in dists if d.lower() != target_tok.lower()][:3]
                if len(dists) < 3:
                    dists = [d for d in ["old", "experienced", "early"] if d != target_tok.lower()][:3]
                return {
                    "target_slot": target_tok,
                    "base_headword": phrase,
                    "stem": stem,
                    "slot_type": "participle_collocator",
                    "cloze_slot_frame": f"... is/was/are ____ {m_cop.group(2)}",
                    "distractors": dists,
                    "explanation": f"In English, \"{clean_p}\" is the fixed collocation. \"{dists[0]}\", \"{dists[1]}\", or \"{dists[2]}\" cannot collocate with \"{m_cop.group(2)}\" in this context.",
                }

        # 2. Adjective + Noun Collocation (e.g. 'high praise', 'heavy rain', 'deep trouble')
        m_adj = re.search(
            r"\b(high|tall|deep|heavy|strong|big|sharp|broad|wide|steep|rich|profound|great)\s+([a-z]+)\b",
            sentence,
            re.IGNORECASE,
        )
        if m_adj and (p_words[0] in _DEGREE_ADJ_DISTRACTORS or p_words[-1] == m_adj.group(2).lower()):
            adj_tok_str = m_adj.group(1)
            adj_start = s_lower.find(adj_tok_str.lower(), m_adj.start())
            adj_end = adj_start + len(adj_tok_str)
            stem = sentence[:adj_start] + "____" + sentence[adj_end:]
            target_tok = sentence[adj_start:adj_end]
            dists = _DEGREE_ADJ_DISTRACTORS.get(adj_tok_str.lower(), ["tall", "deep", "heavy"])
            noun_str = m_adj.group(2)
            return {
                "target_slot": target_tok,
                "base_headword": phrase,
                "stem": stem,
                "slot_type": "collocator_adj",
                "cloze_slot_frame": f"... ____ {noun_str}",
                "distractors": dists,
                "explanation": f"In English, \"{target_tok.lower()} {noun_str}\" is the standard idiomatic collocation. Adjectives like \"{dists[0]}\", \"{dists[1]}\", or \"{dists[2]}\" cannot collocate with \"{noun_str}\".",
            }

        # 3. Prepositional Phrase / Set Phrase (e.g. 'at ease', 'in advance', 'on purpose', 'by accident')
        _PREP_SET = {"at", "in", "on", "by", "for", "under", "with"}
        if p_words[0] in _PREP_SET and len(p_words) == 2:
            p0, p1 = p_words[0], p_words[1]
            m_pp = re.search(rf"\b{re.escape(p0)}\s+{re.escape(p1)}\b", sentence, re.IGNORECASE)
            if m_pp:
                noun_start = s_lower.find(p1, m_pp.start())
                noun_end = noun_start + len(p1)
                stem = sentence[:noun_start] + "____" + sentence[noun_end:]
                target_tok = sentence[noun_start:noun_end]
                dists = list(_PREP_NOUN_DISTRACTORS.get(p1, []))
                if len(dists) < 3:
                    # Dynamically harvest Oxford near-synonyms with PedLex zero-collocation check
                    ox_entries = cls.get_oxford_thesaurus_entries(p1)
                    ox_syns = []
                    phrase_index = cls._get_phrase_index()
                    for ent in ox_entries:
                        if ent.get("pos") in ("noun", ""):
                            for s in ent.get("senses", []):
                                ox_syns.extend(s.get("core_synonyms", []))
                                ox_syns.extend(s.get("synonyms", []))
                    for syn in ox_syns:
                        s_low = syn.lower().strip()
                        if (
                            s_low != p1
                            and len(s_low.split()) == 1
                            and s_low.isalpha()
                            and s_low not in dists
                            and s_low not in ("chance", "mistake")
                            and f"{p0} {s_low}" not in phrase_index
                            and not cls.double_key_collision(p1, s_low, pos="n")[0]
                        ):
                            dists.append(s_low)
                            if len(dists) >= 3:
                                break
                if len(dists) < 3:
                    dists = _PREP_NOUN_DISTRACTORS.get(p1, ["rest", "peace", "comfort"])
                return {
                    "target_slot": target_tok,
                    "base_headword": phrase,
                    "stem": stem,
                    "slot_type": "fixed_noun_slot",
                    "cloze_slot_frame": f"... {p0} ____",
                    "distractors": dists[:3],
                    "explanation": f"\"{clean_p}\" is a fixed prepositional set phrase. Nouns like \"{dists[0]}\", \"{dists[1]}\", or \"{dists[2]}\" do not form this standard idiom.",
                }

        # 4. Nominal Genitive Idioms (e.g. 'peace of mind', 'state of mind', 'matter of course', 'line of duty')
        m_ng = re.search(
            r"\b([a-zA-Z]+)\s+of\s+([a-zA-Z]+)\b",
            sentence,
            re.IGNORECASE,
        )
        if m_ng and (" of " in clean_p.lower()):
            ng_n1 = m_ng.group(1).lower()
            ng_n2 = m_ng.group(2).lower()
            if ng_n1 in p_words and ng_n2 in p_words:
                n1_start = s_lower.find(ng_n1, m_ng.start())
                n1_end = n1_start + len(ng_n1)
                stem = sentence[:n1_start] + "____" + sentence[n1_end:]
                target_tok = sentence[n1_start:n1_end]
                phrase_index = cls._get_phrase_index()

                # Dynamic Oxford near-synonym extraction with PedLex zero-collocation filter
                ox_syns: List[str] = []
                for ent in cls.get_oxford_thesaurus_entries(ng_n1):
                    if ent.get("pos") in ("noun", ""):
                        for s in ent.get("senses", []):
                            ox_syns.extend(s.get("core_synonyms", []))
                            ox_syns.extend(s.get("synonyms", []))

                ng_dists: List[str] = []
                for syn in ox_syns:
                    s_low = syn.lower().strip()
                    if (
                        s_low != ng_n1
                        and len(s_low.split()) == 1
                        and s_low.isalpha()
                        and s_low not in ng_dists
                        and f"{s_low} of {ng_n2}" not in phrase_index
                        and not cls.is_attested_phrase(f"{s_low} of {ng_n2}")
                    ):
                        ng_dists.append(s_low)
                        if len(ng_dists) >= 3:
                            break

                if len(ng_dists) < 3:
                    fb_n = cls.generate_zero_collision_distractors(target_tok.lower(), pos="n", count=6, context_sentence=sentence)
                    for fn in fb_n:
                        if (
                            fn != ng_n1
                            and fn not in ng_dists
                            and f"{fn} of {ng_n2}" not in phrase_index
                        ):
                            ng_dists.append(fn)
                            if len(ng_dists) >= 3:
                                break

                if len(ng_dists) >= 3:
                    return {
                        "target_slot": target_tok,
                        "base_headword": phrase,
                        "stem": stem,
                        "slot_type": "nominal_genitive_noun",
                        "cloze_slot_frame": f"... ____ of {ng_n2}",
                        "distractors": ng_dists[:3],
                        "explanation": f"\"{clean_p}\" is a fixed idiomatic collocation. Nouns like \"{ng_dists[0]}\", \"{ng_dists[1]}\", or \"{ng_dists[2]}\" do not form this standard idiom with \"of {ng_n2}\".",
                    }

        # 5. Complex Prepositions (e.g. 'in terms of', 'on behalf of', 'with regard to', 'in light of', 'in view of')
        m_cp = re.search(
            r"\b(in|on|at|by|with|under)\s+([a-zA-Z]+)\s+(of|with|to|for|from|as)\b",
            sentence,
            re.IGNORECASE,
        )
        if m_cp and len(p_words) >= 3 and p_words[0] in ("in", "on", "at", "by", "with", "under"):
            cp_p1 = m_cp.group(1).lower()
            cp_noun = m_cp.group(2).lower()
            cp_p2 = m_cp.group(3).lower()
            if cp_p1 == p_words[0] and cp_noun in p_words and cp_p2 == p_words[-1]:
                n_start = s_lower.find(cp_noun, m_cp.start())
                n_end = n_start + len(cp_noun)
                stem = sentence[:n_start] + "____" + sentence[n_end:]
                target_tok = sentence[n_start:n_end]
                phrase_index = cls._get_phrase_index()
                peer_medial = [
                    k.split()[1] for k in phrase_index
                    if len(k.split()) == 3
                    and k.split()[0] == cp_p1
                    and k.split()[2] == cp_p2
                    and k.split()[1] != cp_noun
                    and k.split()[1].isalpha()
                ]
                valid_cp_dists: List[str] = []
                for pm in peer_medial:
                    cand_full = f"{cp_p1} {pm} {cp_p2}"
                    if not cls.double_key_collision(clean_p.lower(), cand_full, pos="phrase")[0]:
                        if pm not in valid_cp_dists:
                            valid_cp_dists.append(pm)
                            if len(valid_cp_dists) >= 3:
                                break
                if len(valid_cp_dists) >= 3:
                    return {
                        "target_slot": target_tok,
                        "base_headword": phrase,
                        "stem": stem,
                        "slot_type": "complex_prep_noun",
                        "cloze_slot_frame": f"... {cp_p1} ____ {cp_p2}",
                        "distractors": valid_cp_dists[:3],
                        "explanation": f"\"{clean_p}\" is the fixed complex preposition. While \"{valid_cp_dists[0]}\", \"{valid_cp_dists[1]}\", or \"{valid_cp_dists[2]}\" appear in other fixed phrases, only \"{target_tok.lower()}\" fits the meaning required here.",
                    }

        # 6. Bare Verbal Idioms (e.g. 'give rise to', 'catch sight of', 'shed light on', 'cast doubt on', 'pay tribute to')
        m_bvi = re.search(
            r"\b(give|gave|gives|giving|catch|caught|catches|catching|shed|cast|lose|lost|loses|losing|pay|paid|pays|paying|take|took|takes|taking|set|keep|kept|keeps|keeping)\s+(rise|birth|way|place|sight|glimpse|light|doubt|track|attention|tribute|heed|touch|ear|vent|voice)\s+(to|of|on|in|with)\b",
            sentence,
            re.IGNORECASE,
        )
        if m_bvi and len(p_words) >= 3:
            v_tok = m_bvi.group(1).lower()
            bare_n = m_bvi.group(2).lower()
            p_prep = m_bvi.group(3).lower()
            if bare_n in p_words and p_prep == p_words[-1]:
                bn_start = s_lower.find(bare_n, m_bvi.start())
                bn_end = bn_start + len(bare_n)
                stem = sentence[:bn_start] + "____" + sentence[bn_end:]
                target_tok = sentence[bn_start:bn_end]

                doc_v = nlp(v_tok)
                v_lemma = doc_v[0].lemma_.lower() if doc_v else v_tok
                phrase_index = cls._get_phrase_index()
                peer_bare = [
                    k.split()[1] for k in phrase_index
                    if len(k.split()) == 3
                    and k.split()[0] == v_lemma
                    and k.split()[2] == p_prep
                    and k.split()[1] != bare_n
                    and k.split()[1].isalpha()
                    and k.split()[1] not in ("in", "on", "up", "out", "off", "away", "over", "down", "it", "so", "as")
                ]
                curated_bare_map = {
                    "rise": ["birth", "way", "place"],
                    "sight": ["glimpse", "wind", "track"],
                    "light": ["doubt", "shadow", "gloom"],
                    "doubt": ["light", "blame", "aspersions"],
                    "tribute": ["attention", "heed", "respect"],
                    "track": ["sight", "count", "hold"],
                    "touch": ["contact", "reach", "track"],
                }
                bvi_dists = list(curated_bare_map.get(bare_n, []))
                for pb in peer_bare:
                    if pb not in bvi_dists and pb != bare_n:
                        bvi_dists.append(pb)
                        if len(bvi_dists) >= 3:
                            break
                if len(bvi_dists) >= 3:
                    return {
                        "target_slot": target_tok,
                        "base_headword": phrase,
                        "stem": stem,
                        "slot_type": "bare_verbal_idiom_noun",
                        "cloze_slot_frame": f"... {v_tok} ____ {p_prep}",
                        "distractors": bvi_dists[:3],
                        "explanation": f"\"{clean_p}\" is a fixed verbal idiom. Nouns like \"{bvi_dists[0]}\", \"{bvi_dists[1]}\", or \"{bvi_dists[2]}\" form different idioms that do not fit the meaning here.",
                    }

        # 6b. Tripartite Phrasal-Prepositional Verbs (e.g. 'keep in touch with', 'stay in touch with')
        m_tpp = re.search(
            r"\b(keep|kept|keeps|keeping|stay|stayed|stays|staying|get|got|gets|getting|be|was|were|been|being)\s+in\s+touch\s+with\b",
            sentence,
            re.IGNORECASE,
        )
        if m_tpp and "in touch with" in clean_p.lower():
            v_tok = m_tpp.group(1)
            t_start = s_lower.find("touch", m_tpp.start())
            t_end = t_start + len("touch")
            stem = sentence[:t_start] + "____" + sentence[t_end:]
            target_tok = sentence[t_start:t_end]
            tpp_dists = ["contact", "reach", "communication"]
            return {
                "target_slot": target_tok,
                "base_headword": phrase,
                "stem": stem,
                "slot_type": "phrasal_prep_verb_noun",
                "cloze_slot_frame": f"... {v_tok.lower()} in ____ with",
                "distractors": tpp_dists,
                "explanation": f"\"{clean_p}\" is a fixed phrasal-prepositional idiom. Near-synonyms like \"{tpp_dists[0]}\", \"{tpp_dists[1]}\", or \"{tpp_dists[2]}\" do not complete this fixed construction.",
            }

        # Determine any trailing preposition from the phrase
        trailing_prep = ""
        for p_tok in reversed(p_words):
            if p_tok.lower() in ("to", "on", "in", "of", "with", "for", "between", "against", "into"):
                trailing_prep = f" {p_tok.lower()}"
                break

        # 7. Light-Verb Collocations (e.g. 'make a commitment to', 'commitment made to')
        m_lv_act = re.search(
            r"\b(make|made|makes|making|take|took|takes|taking|give|gave|gives|giving|do|did|does|doing|pay|paid|cast|strike|struck)\s+(?:a|an|the)?\s+(?:[a-zA-Z]+\s+)?(commitment|promise|decision|effort|contribution|tribute|balance|doubt)\b",
            sentence,
            re.IGNORECASE,
        )
        if m_lv_act and (m_lv_act.group(1).lower() in p_words or m_lv_act.group(2).lower() in p_words):
            v_tok_str = m_lv_act.group(1)
            v_start = s_lower.find(v_tok_str.lower(), m_lv_act.start())
            v_end = v_start + len(v_tok_str)
            stem = sentence[:v_start] + "____" + sentence[v_end:]
            target_tok = sentence[v_start:v_end]
            dists = _LIGHT_VERB_DISTRACTORS.get(v_tok_str.lower(), ["take", "give", "do"])
            noun_word = m_lv_act.group(2)
            det_str = "a/the " if noun_word.lower() in ("commitment", "promise", "decision", "effort", "contribution", "balance") else ""
            frame_str = f"... ____ {det_str}{noun_word}{trailing_prep}".strip()
            return {
                "target_slot": target_tok,
                "base_headword": phrase,
                "stem": stem,
                "slot_type": "light_verb",
                "cloze_slot_frame": frame_str,
                "distractors": dists,
                "explanation": f"In English, \"{v_tok_str.lower()}\" collocates with \"{noun_word}\". Verbs like \"{dists[0]}\", \"{dists[1]}\", or \"{dists[2]}\" are unidiomatic in this collocation.",
            }

        m_lv_pass = re.search(
            r"\b(commitment|promise|decision|effort|contribution|tribute|balance|doubt|praise)\s+(made|taken|given|done|paid|cast|struck)\s+(to|for|with|on)\b",
            sentence,
            re.IGNORECASE,
        )
        if m_lv_pass and (m_lv_pass.group(2).lower() in p_words or m_lv_pass.group(1).lower() in p_words):
            v_tok_str = m_lv_pass.group(2)
            v_start = s_lower.find(v_tok_str.lower(), m_lv_pass.start())
            v_end = v_start + len(v_tok_str)
            stem = sentence[:v_start] + "____" + sentence[v_end:]
            target_tok = sentence[v_start:v_end]
            dists = _LIGHT_VERB_DISTRACTORS.get(v_tok_str.lower(), ["taken", "given", "done"])
            noun_word = m_lv_pass.group(1)
            p_prep = f" {m_lv_pass.group(3)}" if m_lv_pass.group(3) else trailing_prep
            det_str = "a/the " if noun_word.lower() in ("commitment", "promise", "decision", "effort", "contribution", "balance") else ""
            # Canonical collocation order puts verb before noun (e.g. '... ____ a/the commitment to')
            frame_str = f"... ____ {det_str}{noun_word}{p_prep}".strip()
            return {
                "target_slot": target_tok,
                "base_headword": phrase,
                "stem": stem,
                "slot_type": "light_verb",
                "cloze_slot_frame": frame_str,
                "distractors": dists,
                "explanation": f"In English, \"{v_tok_str.lower()}\" collocates with \"{noun_word}\". Verbs like \"{dists[0]}\", \"{dists[1]}\", or \"{dists[2]}\" cannot collocate with this noun.",
            }

        # 8. Two-Word Phrasal Verbs (e.g. 'shut down', 'get ahead', 'break up', 'call off')
        _PARTICLES_AND_PREPS = {
            "in", "off", "up", "down", "away", "back", "over", "into", "through",
            "on", "to", "for", "out", "with", "from", "around", "about",
            "by", "at", "under", "between", "against", "after", "before",
            "along", "across", "behind", "beyond", "aside", "apart", "ahead"
        }
        if len(p_words) == 2 and p_words[1] in _PARTICLES_AND_PREPS:
            v_base, part_word = p_words[0], p_words[1]
            m_pv = re.search(
                rf"\b([a-zA-Z]+)\s+({re.escape(part_word)})\b",
                sentence,
                re.IGNORECASE,
            )
            if m_pv:
                cand_v = m_pv.group(1).lower()
                doc_cand = nlp(cand_v)
                cand_lemma = doc_cand[0].lemma_.lower() if doc_cand else cand_v
                if cand_lemma == v_base or cand_v == v_base:
                    p_start = s_lower.find(part_word, m_pv.start())
                    p_end = p_start + len(part_word)
                    stem = sentence[:p_start] + "____" + sentence[p_end:]
                    target_tok = sentence[p_start:p_end]
                    _CONTRAST_PARTICLES = ["down", "up", "off", "away", "out", "in", "back", "over", "ahead", "behind"]
                    pv_dists = [p for p in _CONTRAST_PARTICLES if p != part_word][:3]
                    return {
                        "target_slot": target_tok,
                        "base_headword": phrase,
                        "stem": stem,
                        "slot_type": "phrasal_verb_particle",
                        "cloze_slot_frame": f"... {m_pv.group(1)} ____",
                        "distractors": pv_dists,
                        "explanation": f"In English, \"{m_pv.group(1).lower()} {part_word}\" is the correct phrasal verb here. Particles like \"{pv_dists[0]}\", \"{pv_dists[1]}\", or \"{pv_dists[2]}\" create different meanings or unidiomatic combinations.",
                    }

        # 9. Guarded 2-Word Compound Noun / Modifier Half-Slice
        if len(p_words) == 2:
            w0, w1 = p_words[0], p_words[1]
            m_2w = re.search(rf"\b({re.escape(w0)})\s+({re.escape(w1)})\b", sentence, re.IGNORECASE)
            if m_2w:
                phrase_index = cls._get_phrase_index()
                # Check if w1 is a head noun with attested peer modifiers in PedLex (e.g. 'fighter plane', 'cargo plane')
                attested_mod_peers = [
                    k.split()[0] for k in phrase_index
                    if len(k.split()) == 2
                    and k.split()[1] == w1
                    and k.split()[0] != w0
                    and k.split()[0].isalpha()
                    and k.split()[0] not in _PARTICLES_AND_PREPS
                ]
                if len(attested_mod_peers) >= 3:
                    w0_tok = m_2w.group(1)
                    w0_start = s_lower.find(w0_tok.lower(), m_2w.start())
                    w0_end = w0_start + len(w0_tok)
                    stem = sentence[:w0_start] + "____" + sentence[w0_end:]
                    target_tok = sentence[w0_start:w0_end]
                    return {
                        "target_slot": target_tok,
                        "base_headword": phrase,
                        "stem": stem,
                        "slot_type": "compound_noun_modifier",
                        "cloze_slot_frame": f"... ____ {m_2w.group(2)}",
                        "distractors": attested_mod_peers[:3],
                        "explanation": f"In English, \"{target_tok.lower()} {m_2w.group(2)}\" is the standard compound noun. Modifiers like \"{attested_mod_peers[0]}\", \"{attested_mod_peers[1]}\", or \"{attested_mod_peers[2]}\" form different terms.",
                    }
                # If no attested modifiers exist, DO NOT force an ungrammatical half-slice!
                # Returning None delegates whole-chunk assessment to generate_phrase_distractors,
                # ensuring high-quality symmetrical peer phrases (e.g. passenger plane -> passenger train, passenger ship).
                return None

        return None

    @classmethod
    def isolate_cloze_clause(cls, stem: str) -> str:
        """
        SLA Discourse Unit Isolation Gate:
        Splits compound/complex sentences across coordinate and subordinate clause boundaries
        to isolate the self-contained clause containing the blank '____'.
        Eliminates cross-item leakage (e.g. other vocabulary targets appearing elsewhere in the same sentence)
        and guarantees tight, natural single-sentence cloze test stems.
        """
        if not stem or "____" not in stem:
            return stem

        # Major coordinating & subordinating boundaries (lookbehind must be fixed-width in Python 3.14)
        clauses = re.split(
            r"(?<=,)\s+(?=(?:and|but|or|so|saying|standing|the\s+[A-Z]\w+|when|while|after|before|as|although|because|since|if)\b)|(?<=;)\s+",
            stem
        )
        if len(clauses) <= 1:
            return stem

        for cl in clauses:
            if "____" in cl:
                cl_clean = cl.strip().rstrip(",").strip()
                # Clean leading coordinating conjunctions
                cl_clean = re.sub(r"^(?:and|but|so|or)\s+", "", cl_clean, flags=re.IGNORECASE).strip()
                # A clause starting with '____' lacks an overt subject (coordinate predicate); do not isolate
                if cl_clean.startswith("____"):
                    return stem
                # Prepositional/adverbial fragment guard: introductory adjuncts must not sever the main clause
                first_word = cl_clean.split()[0].lower() if cl_clean.split() else ""
                if first_word in ("by", "in", "at", "on", "with", "without", "from", "to", "for", "through", "despite"):
                    return stem
                if len(cl_clean.split()) >= 4:
                    nlp = cls.get_spacy()
                    doc = nlp(cl_clean)
                    has_finite_verb = any(t.pos_ in ("VERB", "AUX") and t.tag_ in ("VBD", "VBZ", "VBP", "MD", "VB") for t in doc)
                    if not has_finite_verb:
                        return stem
                    if cl_clean and cl_clean[0].islower():
                        cl_clean = cl_clean[0].upper() + cl_clean[1:]
                    if not cl_clean.endswith((".", "!", "?")):
                        cl_clean += "."
                    return cl_clean

        return stem

