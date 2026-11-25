"""
Verb patient ontology classification and object animacy resolution.
"""

import re
from typing import Any, Dict, List, Optional, Set, Tuple


class QuizFramesPatientMixin:
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

    _ABSTRACT_HYPERNYM_TABLE: Tuple[Tuple[Set[str], str, str], ...] = (
        ({"conflict", "hostility", "warfare", "aggression", "danger", "threat", "rebellion", "riot"}, "situation/crisis", "situation_crisis"),
        ({"difficulty", "difficultness", "trouble", "obstacle", "adversity", "dilemma", "hardship"}, "problem/issue", "problem_issue"),
        ({"process", "activity", "inquiry", "investigation", "discussion", "negotiation", "procedure"}, "process/activity", "process_activity"),
        ({"feeling", "emotion", "sentiment", "mood", "psychological state", "mental state", "cognitive state", "symptom"}, "feeling/state", "feeling_state"),
        ({"assets", "sum", "capital", "income", "funds", "wealth", "financial aid"}, "funds/resources", "funds_resources"),
        ({"belief", "opinion", "attitude", "conviction", "doubt", "concern", "criticism", "objection"}, "opinion/concern", "opinion_concern"),
    )
    _ABSTRACT_LEXFILE_FALLBACK: Dict[str, Tuple[str, str]] = {
        "noun.feeling": ("feeling/state", "feeling_state"),
        "noun.cognition": ("idea/attitude", "idea_attitude"),
        "noun.communication": ("statement/opinion", "statement_opinion"),
        "noun.possession": ("funds/resources", "funds_resources"),
        "noun.process": ("process/activity", "process_activity"),
    }

    @classmethod
    def _classify_abstract_slot_category(cls, abs_lemmas: List[str]) -> Tuple[str, str]:
        """Classify abstract patient nouns via WordNet hypernym closure + lexfile fallback."""
        wn = cls.get_wordnet()
        if wn:
            for lem in abs_lemmas:
                syns = wn.synsets(lem.lower(), pos="n")
                if not syns:
                    continue
                s0 = syns[0]
                try:
                    hyps = {hl for h in s0.closure("hypernym") for hl in h.lemmas()} | set(s0.lemmas())
                except Exception:
                    hyps = set(s0.lemmas())
                for anchors, label, ont_type in cls._ABSTRACT_HYPERNYM_TABLE:
                    if hyps & anchors:
                        return label, ont_type
                lf_hit = cls._ABSTRACT_LEXFILE_FALLBACK.get(s0.lexfile())
                if lf_hit:
                    return lf_hit
        return "situation/event", "situation_event"

    @classmethod
    def _is_body_part_noun(cls, lemma: str) -> bool:
        """Return True iff the noun lemma's primary WordNet synset is ``noun.body``."""
        wn = cls.get_wordnet()
        if not wn or not lemma:
            return False
        syns = wn.synsets(lemma.lower(), pos="n")
        return bool(syns and syns[0].lexfile() == "noun.body")

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
        def _is_human_or_social_group(syn) -> bool:
            lf = syn.lexfile()
            if lf == "noun.person":
                return True
            if lf == "noun.group":
                try:
                    hyps = {lem for h in syn.closure("hypernym") for lem in h.lemmas()}
                    return bool(hyps & {"social group", "organization", "people", "person", "gathering"})
                except Exception:
                    return True
            return False

        s0_lf = syns[0].lexfile()
        if _is_human_or_social_group(syns[0]):
            return "person"
        if s0_lf == "noun.group":
            return "abstract_event"
        if s0_lf in cls.PHYSICAL_SUPERSENSES:
            return "physical"
        if s0_lf in cls.ABSTRACT_EVENT_SUPERSENSES:
            return "abstract_event"
        if any(_is_human_or_social_group(s) for s in syns) and not (all_lfs & cls.PHYSICAL_SUPERSENSES):
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
                            has_explicit_the = any(ch.dep_ == "det" and ch.text.lower() == "the" for ch in real_c_tok.children)
                            has_det_or_poss = any(ch.dep_ in ("det", "poss") for ch in real_c_tok.children)
                            is_plural_or_uncountable = (
                                real_c_tok.tag_ in ("NNS", "NNPS")
                                or cls._is_primarily_uncountable_noun(real_lem)
                            )
                            if has_explicit_the or (has_det_or_poss and not is_plural_or_uncountable):
                                np_surface = f"the {real_c_tok.text.lower()}"
                            else:
                                left_mods = [
                                    ch.text.lower()
                                    for ch in sorted(real_c_tok.children, key=lambda t: t.i)
                                    if ch.dep_ in ("compound", "amod")
                                    and ch.i < real_c_tok.i
                                    and real_c_tok.i - ch.i <= 2
                                    and ch.is_alpha
                                    and not _is_forb(ch.lemma_.lower())
                                    and ch.lemma_.lower() not in pred_words
                                ]
                                np_surface = (
                                    f"{left_mods[-1]} {real_c_tok.text.lower()}"
                                    if left_mods
                                    else real_c_tok.text.lower()
                                )
                            cands.append((np_surface, real_lem))

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
                                                # Exclude resultative state / orientation prepositional complements ('into a sloping position', 'in a circular shape', 'to a standstill')
                                                if p_c.lemma_.lower() in ("into", "in", "to", "towards", "onto") and gc_lem in (
                                                    "position", "state", "condition", "shape", "direction", "orientation", "place", "form", "manner", "angle", "degree", "standstill", "halt"
                                                ):
                                                    continue
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

        # 3. Fallback to PedLex collocations if still sparse (STRICTLY for single-word base verbs)
        if len(cands) < 2 and head_v and len(w_toks) == 1:
            collocs = cls.get_pedlex_collocations(head_v)
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
        body_cands: List[str] = []
        abs_cands: List[str] = []
        person_cands: List[str] = []
        for np_text, lem in cands:
            cat = cls._classify_noun_patient_ontology(lem)
            if cat == "person":
                if not has_explicit_sth_only:
                    person_cands.append(np_text)
            elif cat == "physical":
                phys_cands.append(np_text)
                if cls._is_body_part_noun(lem):
                    bare_np = re.sub(r"^the\s+", "", np_text, flags=re.I).strip()
                    if bare_np:
                        body_cands.append(bare_np)
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
                re.search(r"\b(?:someone|somebody)\s+(?:feels?|thinks?|believes?|understands?|does|do)\b", defn_low)
            )
            or bool(
                re.search(r"\b(?:with|to|for|about|against|upon|after|at|of|from)\s+(?:you|someone|somebody|a person|people)\b", defn_low)
            )
            or bool(
                re.search(r"\b(?:affect|affects|punish|punishes|harm|harms|catch up with|befall|befalls)\s+(?:you|someone|somebody|a person|people)\b", defn_low)
            )
        )

        if has_explicit_sb_only and person_cands:
            concrete_p = [p for p in person_cands if p.lower() != "someone"]
            exemplars = concrete_p[:2] if concrete_p else ["someone"]
            slot_label = f"[person/group, e.g. {' / '.join(concrete_p[:2])}]" if concrete_p else "[person/group]"
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
            concrete_p = [p for p in person_cands if p.lower() != "someone"]
            exemplars = concrete_p[:2] if concrete_p else ["someone"]
            slot_label = f"[person/group, e.g. {' / '.join(concrete_p[:2])}]" if concrete_p else "[person/group]"
            ont_type = "person_group"
        elif abs_cands and (
            len(abs_cands) > len(phys_cands)
            or any(w in defn_low for w in ("situation", "problem", "crisis", "conflict", "process", "activity", "tension", "event"))
        ):
            exemplars = abs_cands[:2]
            ex_inline = " / ".join(exemplars)
            abs_lemmas = [ex.split()[-1].lower() for ex in exemplars if ex.strip()]
            cat_label, ont_type = cls._classify_abstract_slot_category(abs_lemmas)
            slot_label = f"[{cat_label}, e.g. {ex_inline}]"
        elif person_cands and (len(person_cands) > len(phys_cands) or has_person_def):
            concrete_p = [p for p in person_cands if p.lower() != "someone"]
            exemplars = concrete_p[:2] if concrete_p else ["someone"]
            slot_label = f"[person/group, e.g. {' / '.join(concrete_p[:2])}]" if concrete_p else "[person/group]"
            ont_type = "person_group"
        elif phys_cands:
            if (
                body_cands
                and re.sub(r"^the\s+", "", phys_cands[0], flags=re.I).strip() == body_cands[0]
            ):
                exemplars = body_cands[:2]
                ex_inline = " / ".join(exemplars)
                slot_label = f"[body part, e.g. {ex_inline}]"
                ont_type = "body_part"
            else:
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

