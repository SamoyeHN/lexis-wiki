"""
Deterministic computational morphology, lemmatization, verb/noun inflections, and word families.
"""

import logging
import math
import os
import re
import sqlite3
import threading
from functools import lru_cache
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union

logger = logging.getLogger("librarian.linguistics")


class MorphologyEngineMixin:
    _spacy_nlp = None
    _wn = None
    _surface_lemma_cache: Dict[str, Optional[str]] = {}
    _attested_form_cache: Dict[str, bool] = {}
    _plural_form_cache: Dict[str, str] = {}
    _inflected_forms_cache: Dict[Tuple[str, str], Set[str]] = {}
    _derivational_base_cache: Dict[str, Set[str]] = {}
    @classmethod
    def get_spacy(cls):
        """Lazy-loads the spaCy English model (~12MB) with a bounded LRU Doc cache."""
        if cls._spacy_nlp is None:
            import spacy
            try:
                nlp = spacy.load("en_core_web_sm")
            except OSError:
                from spacy.cli import download as _spacy_download
                _spacy_download("en_core_web_sm")
                nlp = spacy.load("en_core_web_sm")
            orig_call = nlp.__class__.__call__

            @lru_cache(maxsize=4096)
            def _cached_doc(text: str):
                return orig_call(nlp, text)

            class _CachedLanguage(nlp.__class__):
                def __call__(self, text, *args, **kwargs):
                    if isinstance(text, str) and not args and not kwargs:
                        return _cached_doc(text)
                    return orig_call(self, text, *args, **kwargs)

            nlp.__class__ = _CachedLanguage
            cls._spacy_nlp = nlp
        return cls._spacy_nlp

    @classmethod
    def get_wordnet(cls):
        """Lazy-loads the Open English WordNet ('oewn:2025+') client with multithreading enabled."""
        if cls._wn is None:
            try:
                import wn
                # Web dashboard / background worker threads share WordNet SQLite connection.
                # SQLite check_same_thread must be disabled for read-only multi-threaded queries.
                wn.config.allow_multithreading = True
                cls._wn = wn.Wordnet("oewn:2025+")
            except Exception:
                cls._wn = False
        if cls._wn is False or cls._wn is None:
            return None
        return cls._wn

    @staticmethod
    def _normalize_bre_ame(word: str) -> str:
        """Normalizes systematic British/American and classical orthographic alternations to a canonical form."""
        w = (word or "").lower().strip()
        if len(w) < 4:
            return w
        w = re.sub(r"^artefact", "artifact", w)
        w = re.sub(r"^sceptic", "skeptic", w)
        w = re.sub(r"^grey", "gray", w)
        w = re.sub(r"^enquir", "inquir", w)
        w = re.sub(r"^esthetic", "aesthetic", w)
        w = re.sub(r"^archeolog", "archaeolog", w)
        w = re.sub(r"^medieval", "mediaeval", w)
        w = re.sub(r"^programmes\b", "programs", w)
        w = re.sub(r"^programme\b", "program", w)
        w = re.sub(r"^(enrol|unrol|fulfil|instil|distil|appal|enthral)$", r"\1l", w)
        w = re.sub(r"^(skil|wil)ful", r"\1lful", w)
        if not w.startswith(("four", "hour", "pour", "tour", "sour", "flour", "your", "devour")):
            w = re.sub(r"([a-z]{2,})our(s|ed|ing|able|ite|ites|less|ful)?$", r"\1or\2", w)
        w = re.sub(r"([a-z]{2,})is(e|es|ed|ing|ation|ations)$", r"\1iz\2", w)
        w = re.sub(r"([a-z]{2,})ys(e|es|ed|ing)$", r"\1yz\2", w)
        if len(w) >= 6:
            w = re.sub(r"([bcfglmnprstv])re(s)?$", r"\1er\2", w)
            w = re.sub(r"([a-z]{2,})ogue(s)?$", r"\1og\2", w)
            w = re.sub(r"([dfglprt])ence(s)?$", r"\1ense\2", w)
        if len(w) >= 7:
            _native_ll_stems = {
                "fill", "will", "bill", "kill", "drill", "chill", "skill", "spill", "thrill",
                "grill", "mill", "pill", "till", "still", "distill", "instill", "fulfill",
                "spell", "shell", "smell", "swell", "dwell", "tell", "sell", "bell", "well", "yell",
                "call", "fall", "hall", "mall", "stall", "tall", "wall", "install", "recall", "appall", "enthrall",
                "poll", "roll", "enroll", "unroll", "scroll", "stroll", "toll", "troll",
                "dull", "full", "hull", "lull", "mull", "null", "pull", "skull",
            }
            m_ll = re.match(r"^([a-z]+l)l(ed|ing|er|ers|or|ors|ous)$", w)
            if m_ll and (m_ll.group(1) + "l") not in _native_ll_stems:
                w = m_ll.group(1) + m_ll.group(2)
        return w

    @staticmethod
    def _edit_distance_le2(s1: str, s2: str) -> bool:
        """Fast check for whether Levenshtein edit distance between s1 and s2 is <= 2."""
        if s1 == s2:
            return True
        if abs(len(s1) - len(s2)) > 2:
            return False
        prev = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1, 1):
            curr = [i]
            min_row = i
            for j, c2 in enumerate(s2, 1):
                cost = 0 if c1 == c2 else 1
                val = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
                curr.append(val)
                if val < min_row:
                    min_row = val
            if min_row > 2:
                return False
            prev = curr
        return prev[-1] <= 2

    @classmethod
    @lru_cache(maxsize=4096)
    def are_spelling_variants(cls, w1: str, w2: str) -> bool:
        """
        Determines whether two words are regional (BrE <-> AmE) or orthographic spelling
        variants of the exact same lexical item (e.g. artefact/artifact, colour/color,
        analyse/analyze, sceptical/skeptical).
        """
        c1 = (w1 or "").lower().strip()
        c2 = (w2 or "").lower().strip()
        if not c1 or not c2:
            return False
        if c1 == c2:
            return True
        if cls._normalize_bre_ame(c1) == cls._normalize_bre_ame(c2):
            return True

        l1 = cls.lemma_of(c1) or c1
        l2 = cls.lemma_of(c2) or c2
        if c1.replace("-", "") == c2.replace("-", ""):
            return True
        if (l1 == l2 or cls._normalize_bre_ame(l1) == cls._normalize_bre_ame(l2)) and (
            c1.replace("-", "") == c2.replace("-", "")
            or cls._edit_distance_le2(c1, c2)
            or cls._normalize_bre_ame(c1) == cls._normalize_bre_ame(c2)
        ):
            return True

        # Check PedLex entry variants / alias_of
        try:
            for probe_a, probe_b in ((l1, l2), (c1, c2)):
                e1 = cls.get_ldoce_entry(probe_a)
                if e1:
                    vars1 = {str(v).lower().strip() for v in (e1.get("variants") or []) if v}
                    if e1.get("alias_of"):
                        vars1.add(str(e1["alias_of"]).lower().strip())
                    if probe_b in vars1:
                        return True
        except Exception:
            pass

        # Check WordNet co-synset identity with orthographic proximity (edit distance <= 2)
        if min(len(l1), len(l2)) >= 5 and cls._edit_distance_le2(l1, l2):
            if l1[:2] == l2[:2] or {l1[:2], l2[:2]} in ({"ae", "es"}, {"ar", "ar"}, {"sc", "sk"}, {"en", "in"}, {"oe", "es"}):
                try:
                    wn = cls.get_wordnet()
                    if wn:
                        s1 = {ss.id for ss in wn.synsets(l1)}
                        s2 = {ss.id for ss in wn.synsets(l2)}
                        if s1 and (s1 & s2):
                            return True
                except Exception:
                    pass
        return False

    @classmethod
    @lru_cache(maxsize=4096)
    def get_word_family(cls, word: str) -> Set[str]:
        """Returns the set of lemmas morphologically and derivationally linked to this word in WordNet."""
        clean_w = word.lower().strip()
        if not clean_w:
            return set()
        family = {clean_w}
        probes = {clean_w}
        # Include underlying verb/base lemma for participial (-ed/-ing) or inflected forms (e.g. moved/moving -> move)
        if clean_w.endswith(("ed", "ing", "s")) and len(clean_w) >= 4:
            lem = cls.lemma_of(clean_w)
            base_cands = {lem} if lem else set()
            if clean_w.endswith("ed") and len(clean_w) >= 4:
                base_cands.update({clean_w[:-2], clean_w[:-1]})
                if clean_w.endswith("ied") and len(clean_w) >= 5:
                    base_cands.add(clean_w[:-3] + "y")
            elif clean_w.endswith("ing") and len(clean_w) >= 5:
                base_cands.update({clean_w[:-3], clean_w[:-3] + "e"})
            for bc in base_cands:
                if bc and bc != clean_w and len(bc) >= 3 and clean_w[:3] == bc[:3]:
                    if bc == lem or ("v" in cls._wordnet_pos(bc)):
                        family.add(bc)
                        probes.add(bc)
        try:
            wn = cls.get_wordnet()
            for p_w in list(probes):
                for s in wn.synsets(p_w):
                    for sense in s.senses():
                        if sense.word().lemma().lower() == p_w:
                            for rel in sense.get_related('derivation'):
                                rel_w = rel.word().lemma().lower()
                                if rel_w.isalpha() and '_' not in rel_w and ' ' not in rel_w:
                                    family.add(rel_w)
        except Exception:
            pass
        return family

    @classmethod
    def are_same_word_family(cls, w1: str, w2: str) -> bool:
        """Determines whether two words belong to the exact same morphological word family or spelling variant set."""
        clean_w1 = w1.lower().strip()
        clean_w2 = w2.lower().strip()
        if clean_w1 == clean_w2:
            return True
        if not clean_w1 or not clean_w2:
            return False
        if cls.are_spelling_variants(clean_w1, clean_w2):
            return True

        fam1 = cls.get_word_family(clean_w1)
        if clean_w2 in fam1:
            return True
        fam2 = cls.get_word_family(clean_w2)
        if clean_w1 in fam2:
            return True
        if fam1 & fam2:
            # Require shared morphological root prefix (>= 4 chars, or >= 3 chars for -ed/-ing participial pairs like moved/moving)
            # to prevent transitive false positives across distinct roots (e.g. critic -> critical <- crisis)
            common_len = 0
            for c1, c2 in zip(clean_w1, clean_w2):
                if c1 == c2:
                    common_len += 1
                else:
                    break
            is_participial_pair = (
                (clean_w1.endswith("ed") and clean_w2.endswith("ing"))
                or (clean_w1.endswith("ing") and clean_w2.endswith("ed"))
            )
            if common_len >= 4 or (is_participial_pair and common_len >= 3):
                return True

        # Morphological stem prefix fallback: if both words length >= 5 and share stem prefix + valid derivational/inflectional suffix
        if " " in clean_w1 or " " in clean_w2:
            return False
        _valid_family_suffixes = {
            "s", "es", "ed", "d", "ing", "er", "ers", "or", "ors", "est", "ly", "ily", "ally",
            "ness", "ment", "ments", "tion", "tions", "ation", "ations", "ition", "itions",
            "able", "ible", "al", "ial", "ic", "ical", "ically", "ive", "ively", "ous", "ously",
            "ful", "fully", "less", "lessly", "ize", "izes", "ized", "izing", "ise", "ises", "ised", "ising",
            "ify", "ifies", "ified", "ifying", "ity", "ities", "ty", "ties", "ist", "ists", "ism", "isms",
            "ance", "ances", "ence", "ences", "ant", "ants", "ent", "ents", "ery", "ry", "cy",
            "dom", "ship", "hood", "ward", "wards", "wise", "en", "ens", "ened", "ening",
        }
        min_w, max_w = (clean_w1, clean_w2) if len(clean_w1) <= len(clean_w2) else (clean_w2, clean_w1)
        if len(min_w) >= 5 and max_w.startswith(min_w):
            rem = max_w[len(min_w):]
            if rem in _valid_family_suffixes or (len(rem) >= 2 and rem[0] == min_w[-1] and rem[1:] in _valid_family_suffixes):
                return True
        # Handle silent -e drop before vowel-initial suffixes (-ing / -ed / -ation / -able / -ive, etc.)
        if len(min_w) >= 5 and min_w.endswith("e") and max_w.startswith(min_w[:-1]):
            rem_e = max_w[len(min_w) - 1:]
            if rem_e and rem_e[0] in "aeiou" and rem_e in _valid_family_suffixes:
                return True
        return False

    @classmethod
    def shares_derivational_root(cls, w1: str, w2: str) -> bool:
        """
        Checks whether w1 and w2 share a morphological or prefix-stripped derivational root
        (e.g. 'unsolved' vs 'solve'/'solved'/'solving'/'solution', 'impossible' vs 'possible').
        Used by Target Stem Leakage gates so a stem testing a prefixed word never leaks its base root.
        """
        c1 = (w1 or "").lower().strip()
        c2 = (w2 or "").lower().strip()
        if not c1 or not c2:
            return False
        if cls.are_same_word_family(c1, c2):
            return True
        _prefixes = ("un", "in", "im", "ir", "il", "dis", "non", "mis", "re", "over", "under", "out", "pre", "post", "sub", "inter", "counter", "de")
        variants_1 = {c1}
        variants_2 = {c2}
        for p in _prefixes:
            if c1.startswith(p) and len(c1) - len(p) >= 4:
                variants_1.add(c1[len(p):])
            if c2.startswith(p) and len(c2) - len(p) >= 4:
                variants_2.add(c2[len(p):])
        for v1 in variants_1:
            for v2 in variants_2:
                if v1 == v2 or cls.are_same_word_family(v1, v2):
                    return True
                s1 = re.sub(r"(?:ed|ing|s|es|ly|tion|ment|able|ible|ive|al)$", "", v1)
                s2 = re.sub(r"(?:ed|ing|s|es|ly|tion|ment|able|ible|ive|al)$", "", v2)
                if len(s1) >= 4 and s1 == s2:
                    return True
        return False

    # ------------------------------------------------------------------
    # Distractor Slot Legality & Prescribed-Inflection Concordance
    # (distractor POS/slot legality, inflection/example agreement)
    # ------------------------------------------------------------------

    _VERB_FORM_LABELS = {
        "VBD": "past tense (VBD)",
        "VBN": "past participle (VBN)",
        "VBG": "present participle (VBG)",
        "VBZ": "third-person singular (VBZ)",
        "VBP": "present tense (VBP)",
        "VB": "base form",
    }
    _VERB_FORM_MICRO_LABELS = {
        "VBD": "past-tense",
        "VBN": "past-participle",
        "VBG": "present-participle",
        "VBZ": "third-person-singular",
        "VBP": "present-tense",
        "VB": "base-form",
    }
    # base -> (past, past participle, -ing form, 3rd person singular)
    _IRREGULAR_VERB_FORMS = {
        "be": ("was", "been", "being", "is"),
        "have": ("had", "had", "having", "has"),
        "do": ("did", "done", "doing", "does"),
        "say": ("said", "said", "saying", "says"),
        "make": ("made", "made", "making", "makes"),
        "take": ("took", "taken", "taking", "takes"),
        "give": ("gave", "given", "giving", "gives"),
        "get": ("got", "gotten", "getting", "gets"),
        "go": ("went", "gone", "going", "goes"),
        "come": ("came", "come", "coming", "comes"),
        "see": ("saw", "seen", "seeing", "sees"),
        "know": ("knew", "known", "knowing", "knows"),
        "think": ("thought", "thought", "thinking", "thinks"),
        "keep": ("kept", "kept", "keeping", "keeps"),
        "let": ("let", "let", "letting", "lets"),
        "put": ("put", "put", "putting", "puts"),
        "set": ("set", "set", "setting", "sets"),
        "pay": ("paid", "paid", "paying", "pays"),
        "send": ("sent", "sent", "sending", "sends"),
        "build": ("built", "built", "building", "builds"),
        "find": ("found", "found", "finding", "finds"),
        "hold": ("held", "held", "holding", "holds"),
        "lead": ("led", "led", "leading", "leads"),
        "mean": ("meant", "meant", "meaning", "means"),
        "win": ("won", "won", "winning", "wins"),
        "begin": ("began", "begun", "beginning", "begins"),
        "crossbreed": ("crossbred", "crossbred", "crossbreeding", "crossbreeds"),
        "outrun": ("outran", "outrun", "outrunning", "outruns"),
        "overcome": ("overcame", "overcome", "overcoming", "overcomes"),
        "undergo": ("underwent", "undergone", "undergoing", "undergoes"),
        "undo": ("undid", "undone", "undoing", "undoes"),
        "redo": ("redid", "redone", "redoing", "redoes"),
        "outdo": ("outdid", "outdone", "outdoing", "outdoes"),
        "overdo": ("overdid", "overdone", "overdoing", "overdoes"),
        "forego": ("forewent", "foregone", "foregoing", "foregoes"),
        "undertake": ("undertook", "undertaken", "undertaking", "undertakes"),
        "withhold": ("withheld", "withheld", "withholding", "withholds"),
        "withstand": ("withstood", "withstood", "withstanding", "withstands"),
        "prefer": ("preferred", "preferred", "preferring", "prefers"),
        "refer": ("referred", "referred", "referring", "refers"),
        "occur": ("occurred", "occurred", "occurring", "occurs"),
        # Without these the mechanical inflector invents 'choosed', 'writed',
        # 'spaked' - forms no gate downstream can tell apart from real ones.
        "choose": ("chose", "chosen", "choosing", "chooses"),
        "write": ("wrote", "written", "writing", "writes"),
        "speak": ("spoke", "spoken", "speaking", "speaks"),
        "run": ("ran", "run", "running", "runs"),
        "break": ("broke", "broken", "breaking", "breaks"),
        "bring": ("brought", "brought", "bringing", "brings"),
        "lay": ("laid", "laid", "laying", "lays"),
        "tell": ("told", "told", "telling", "tells"),
        "sell": ("sold", "sold", "selling", "sells"),
        "spend": ("spent", "spent", "spending", "spends"),
        "meet": ("met", "met", "meeting", "meets"),
        "read": ("read", "read", "reading", "reads"),
        "drive": ("drove", "driven", "driving", "drives"),
        "draw": ("drew", "drawn", "drawing", "draws"),
        "fly": ("flew", "flown", "flying", "flies"),
        "forget": ("forgot", "forgotten", "forgetting", "forgets"),
        "steal": ("stole", "stolen", "stealing", "steals"),
        "catch": ("caught", "caught", "catching", "catches"),
        "throw": ("threw", "thrown", "throwing", "throws"),
        "fight": ("fought", "fought", "fighting", "fights"),
        "wear": ("wore", "worn", "wearing", "wears"),
        "sing": ("sang", "sung", "singing", "sings"),
        "swim": ("swam", "swum", "swimming", "swims"),
        "grow": ("grew", "grown", "growing", "grows"),
        "show": ("showed", "shown", "showing", "shows"),
        "hear": ("heard", "heard", "hearing", "hears"),
        "leave": ("left", "left", "leaving", "leaves"),
        "understand": ("understood", "understood", "understanding", "understands"),
        "stand": ("stood", "stood", "standing", "stands"),
        "sit": ("sat", "sat", "sitting", "sits"),
        "fall": ("fell", "fallen", "falling", "falls"),
        "rise": ("rose", "risen", "rising", "rises"),
        "shake": ("shook", "shaken", "shaking", "shakes"),
        "ride": ("rode", "ridden", "riding", "rides"),
        "freeze": ("froze", "frozen", "freezing", "freezes"),
        "buy": ("bought", "bought", "buying", "buys"),
        "teach": ("taught", "taught", "teaching", "teaches"),
        "feel": ("felt", "felt", "feeling", "feels"),
        "lose": ("lost", "lost", "losing", "loses"),
        "sleep": ("slept", "slept", "sleeping", "sleeps"),
        "shine": ("shone", "shone", "shining", "shines"),
        "light": ("lit", "lit", "lighting", "lights"),
        "bend": ("bent", "bent", "bending", "bends"),
        "cut": ("cut", "cut", "cutting", "cuts"),
        "hit": ("hit", "hit", "hitting", "hits"),
        "hurt": ("hurt", "hurt", "hurting", "hurts"),
        "shut": ("shut", "shut", "shutting", "shuts"),
        "spread": ("spread", "spread", "spreading", "spreads"),
        "cost": ("cost", "cost", "costing", "costs"),
        "deal": ("dealt", "dealt", "dealing", "deals"),
        "seek": ("sought", "sought", "seeking", "seeks"),
        "creep": ("crept", "crept", "creeping", "creeps"),
        "weep": ("wept", "wept", "weeping", "weeps"),
        "sweep": ("swept", "swept", "sweeping", "sweeps"),
        "leap": ("leapt", "leapt", "leaping", "leaps"),
        "kneel": ("knelt", "knelt", "kneeling", "kneels"),
        "lend": ("lent", "lent", "lending", "lends"),
        "feed": ("fed", "fed", "feeding", "feeds"),
        "bleed": ("bled", "bled", "bleeding", "bleeds"),
        "breed": ("bred", "bred", "breeding", "breeds"),
        "speed": ("sped", "sped", "speeding", "speeds"),
        "flee": ("fled", "fled", "fleeing", "flees"),
        "sink": ("sank", "sunk", "sinking", "sinks"),
        "drink": ("drank", "drunk", "drinking", "drinks"),
        "shrink": ("shrank", "shrunk", "shrinking", "shrinks"),
        "ring": ("rang", "rung", "ringing", "rings"),
        "spring": ("sprang", "sprung", "springing", "springs"),
        "arise": ("arose", "arisen", "arising", "arises"),
        "awake": ("awoke", "awoken", "awaking", "awakes"),
        "bear": ("bore", "borne", "bearing", "bears"),
        "beat": ("beat", "beaten", "beating", "beats"),
        "bite": ("bit", "bitten", "biting", "bites"),
        "blow": ("blew", "blown", "blowing", "blows"),
        "dig": ("dug", "dug", "digging", "digs"),
        "hang": ("hung", "hung", "hanging", "hangs"),
        "hide": ("hid", "hidden", "hiding", "hides"),
        "forbid": ("forbade", "forbidden", "forbidding", "forbids"),
        "forgive": ("forgave", "forgiven", "forgiving", "forgives"),
        "shoot": ("shot", "shot", "shooting", "shoots"),
        "slide": ("slid", "slid", "sliding", "slides"),
        "stick": ("stuck", "stuck", "sticking", "sticks"),
        "strike": ("struck", "struck", "striking", "strikes"),
        "swear": ("swore", "sworn", "swearing", "swears"),
        "tear": ("tore", "torn", "tearing", "tears"),
        "wake": ("woke", "woken", "waking", "wakes"),
        "weave": ("wove", "woven", "weaving", "weaves"),
        "wind": ("wound", "wound", "winding", "winds"),
        "burst": ("burst", "burst", "bursting", "bursts"),
        "split": ("split", "split", "splitting", "splits"),
        "quit": ("quit", "quit", "quitting", "quits"),
        "rid": ("rid", "rid", "ridding", "rids"),
        "bet": ("bet", "bet", "betting", "bets"),
        "cast": ("cast", "cast", "casting", "casts"),
        "broadcast": ("broadcast", "broadcast", "broadcasting", "broadcasts"),
        "forecast": ("forecast", "forecast", "forecasting", "forecasts"),
        "swing": ("swung", "swung", "swinging", "swings"),
        "bind": ("bound", "bound", "binding", "binds"),
        "grind": ("ground", "ground", "grinding", "grinds"),
    }

    _wordnet_pos_cache: Dict[str, Set[str]] = {}

    @classmethod
    def _wordnet_pos(cls, word: str) -> Set[str]:
        """Part-of-speech letters WordNet actually records for this lemma form ('n','v','a','s','r')."""
        w_key = (word or "").strip().lower()
        if not w_key:
            return set()
        cached = cls._wordnet_pos_cache.get(w_key)
        if cached is not None:
            return cached
        wn = cls.get_wordnet()
        if not wn:
            return set()
        out: Set[str] = set()
        try:
            for ss in wn.synsets(w_key):
                pos = getattr(ss, "pos", None)
                if pos:
                    out.add(pos)
        except Exception:
            pass
        cls._wordnet_pos_cache[w_key] = out
        return out

    @classmethod
    def _verb_takes_object(cls, word: str) -> Optional[bool]:
        """
        True/False from PedLex valency evidence for this verb; None when the dictionary is
        silent. Evidence is read first from authoritative entryhead/sense <gram> tags
        ('[transitive]' vs pure '[intransitive]'), then from grammar patterns and sense
        examples ('She logged the details of the meeting' -> direct object). A verb whose
        senses publish neither grammar tags, patterns, nor examples is unknown, never
        assumed intransitive.
        """
        entry = cls.get_ldoce_entry((word or "").strip().lower())
        if not entry:
            return None
        verb_senses = [
            s for s in (entry.get("senses", []) or [])
            if not (s.get("pos") or "") or "verb" in (s.get("pos") or "").lower()
        ]
        base_verb_senses = [s for s in verb_senses if not s.get("from_phrasal_verb")]
        target_senses = base_verb_senses or verb_senses
        verb_grams = [
            str(s.get("grammar") or "").strip().lower()
            for s in target_senses
            if str(s.get("grammar") or "").strip()
        ]
        if verb_grams:
            if any(re.search(r"\btransitive\b", g) for g in verb_grams):
                return True
            if len(verb_grams) == len(target_senses) and all(
                ("intransitive" in g or "always + adverb/preposition" in g)
                and not re.search(r"\btransitive\b", g)
                for g in verb_grams
            ):
                return False
        saw_verb_evidence = False
        examples: List[str] = []
        for sense in target_senses:
            for pat in sense.get("patterns", []) or []:
                pl = (pat or "").lower()
                if not pl:
                    continue
                saw_verb_evidence = True
                # Strip prepositional complements ('to somebody', 'with something', 'of something') before checking direct object
                pl_no_prep = re.sub(
                    r"\b(?:to|for|with|of|in|on|at|from|about|into|by|upon|over|against|between)\s+(?:somebody|someone|something|sb|sth)(?:/(?:somebody|someone|something|sb|sth))?\b",
                    "",
                    pl,
                )
                if any(m in pl_no_prep for m in ("somebody", "someone", "something", "sb ", "sb$", "sth")):
                    return True
            for ex in sense.get("examples", []) or []:
                if ex and len(ex.split()) >= 3:
                    examples.append(ex)
        if examples:
            saw_verb_evidence = True
            nlp = cls.get_spacy()
            if nlp:
                v_low = (word or "").strip().lower()
                for ex in examples[:6]:
                    ex_clean = re.sub(r"\(.*?\)", "", ex).strip()
                    if not ex_clean:
                        continue
                    try:
                        doc = nlp(ex_clean)
                    except Exception:
                        continue
                    for tok in doc:
                        if tok.lemma_.lower() != v_low and tok.text.lower() != v_low:
                            continue
                        if tok.pos_ != "VERB":
                            continue
                        for child in tok.children:
                            if child.dep_ in ("dobj", "obj") and child.is_alpha and child.pos_ in ("NOUN", "PROPN"):
                                return True
        return False if saw_verb_evidence else None

    @classmethod
    def _distractor_occupies_slot(
        cls,
        distractor: str,
        canonical_pos: Optional[str],
        requires_object: bool = False
    ) -> bool:
        """
        Slot Legality Gate .

        A distractor must be able to occupy the SAME syntactic slot as the target, or the
        item is solvable by morphology alone instead of by meaning: 'log in to a ____'
        offered ['account', 'somehow', 'autonomy', 'message'], and a transitive verb frame
        offered the intransitive 'flourish'. The gate consults only positive evidence —
        WordNet lemma POS records and PedLex valency patterns — and never vetoes a word the
        dictionaries are silent about, so rare but legitimate distractors survive.
        """
        word = (distractor or "").strip().lower()
        if not word:
            return False
        cp = (canonical_pos or "").strip().lower()
        required = {
            "noun": {"n"}, "verb": {"v"}, "adj": {"a", "s"}, "adjective": {"a", "s"},
            "adv": {"r"}, "adverb": {"r"},
        }.get(cp)
        if not required:
            return True  # phrase / idiom_slot / function_word: no single-POS slot to violate

        pos_set = cls._wordnet_pos(word)
        if not pos_set:
            return True  # no lexical record → absence of evidence is not evidence of illegality
        if not (pos_set & required):
            return False  # WordNet records this lemma, and never in the target's slot

        if cp == "verb" and requires_object and cls._verb_takes_object(word) is False:
            return False  # intransitive-only verb cannot fill a transitive frame
        return True

    @classmethod
    def distractor_occupies_slot(
        cls,
        distractor: str,
        canonical_pos: Optional[str],
        requires_object: bool = False
    ) -> bool:
        """
        Public Slot Legality Gate  for downstream consumers such as the evaluator,
        which only ever sees the inflected option strings a quiz actually printed.

        An inflected surface form usually has no dictionary record of its own, so the verdict
        is re-run on its lemma; a word is still only vetoed on positive evidence, never because
        a dictionary happens to be silent about it.
        """
        word = (distractor or "").strip().lower()
        if not word:
            return False
        lemma = cls.lemma_of(word)
        if cls._distractor_occupies_slot(word, canonical_pos, requires_object):
            # An inflected surface form has no entry of its own, so it is silent about valency:
            # a transitive frame is re-checked on the lemma before the option is accepted.
            if requires_object and lemma and lemma != word:
                if cls._verb_takes_object(lemma) is False:
                    return False
            return True
        # An inflected surface form can carry a dictionary record of its own that belongs to a
        # different slot ('disturbed' and 'irritated' are recorded as adjectives), so the verdict
        # is re-run on the lemma before anything is vetoed.
        if lemma and lemma != word:
            return cls._distractor_occupies_slot(lemma, canonical_pos, requires_object)
        return False

    @classmethod
    def verb_takes_object(cls, verb: Optional[str]) -> Optional[bool]:
        """Public valency probe: True (transitive), False (intransitive-only), None (silent)."""
        return cls._verb_takes_object(verb)

    @classmethod
    def inflection_tag_from_label(cls, inflection_label: Optional[str]) -> Optional[str]:
        """Maps a blueprint inflection declaration ('past tense (VBD)') back to its Penn tag."""
        label = (inflection_label or "").strip().lower()
        if not label:
            return None
        if "base form" in label:
            return "VB"
        if "plural" in label:
            return "NNS"
        m = re.search(r"\((vbd|vbn|vbg|vbz|vbp|vb|nns|nn)\)", label)
        if m:
            return m.group(1).upper()
        for tag, micro in cls._VERB_FORM_MICRO_LABELS.items():
            if micro.replace("-", " ") in label.replace("-", " "):
                return tag
        return None

    @classmethod
    def verb_form_for_tag(cls, base_verb: Optional[str], tag: Optional[str]) -> Optional[str]:
        """Public wrapper: the surface form a base verb must take for a Penn tag."""
        if not base_verb or not tag:
            return None
        return cls._inflect_verb(base_verb, tag)

    @classmethod
    def lemma_of(cls, word: Optional[str]) -> str:
        """
        Lemma for a single word form, preferring a verbal parse and falling back to the input.
        """
        return cls._analyse_word_form(word)[1]

    _LEMMA_CARRIERS = (
        "{w}",
        "she {w} it",
        "they {w} it",
        "she has {w} it",
        "she is {w} it",
        "{w} it every day",
    )

    _INVARIANT_PLURAL_NOUNS = frozenset({
        "species", "series", "means", "headquarters", "aircraft", "spacecraft",
        "offspring", "corps", "chassis", "rendezvous", "apparatus", "crossroads",
        "barracks", "gallows", "innings",
    })

    @classmethod
    def _is_base_verb_lemma(cls, w: str) -> bool:
        """Returns True if `w` is an attested base/infinitive verb lemma in WordNet or PedLex."""
        if not w:
            return False
        if "-" in w and cls._is_base_verb_lemma(w.replace("-", "")):
            return True
        if w in cls._IRREGULAR_VERB_FORMS:
            return True
        try:
            wn = cls.get_wordnet()
            if wn and any(wo.lemma().lower() == w for wo in wn.words(w, pos="v")):
                return True
        except Exception:
            pass
        if any(w in forms for base, forms in cls._IRREGULAR_VERB_FORMS.items() if base != w):
            return False
        ent = cls.get_ldoce_entry(w)
        if ent and (ent.get("word") or "").strip().lower() == w:
            all_p = [cls._pos_code(p) for p in (ent.get("all_poses") or [ent.get("pos") or ""])]
            if "v" in all_p:
                v_defs = [
                    str(s.get("definition") or "").strip().lower()
                    for s in (ent.get("senses") or [])
                    if "verb" in str(s.get("pos") or ent.get("pos") or "").lower()
                ]
                if v_defs and all(
                    d.startswith(("the past ", "past tense", "past participle", "the present participle", "3rd person"))
                    for d in v_defs
                ):
                    return False
                return True
        # Fallback when PedLex and WordNet are not available:
        # Check via spaCy POS tagging
        if cls.get_wordnet() is None and (cls._is_pedlex_disabled() or not cls._ensure_ldoce_conn()):
            nlp = cls.get_spacy()
            if nlp:
                try:
                    doc = nlp(f"They {w} it.")
                    if doc and len(doc) >= 2 and doc[1].pos_ == "VERB" and doc[1].tag_ in ("VB", "VBP"):
                        return True
                except Exception:
                    pass
        return False

    @classmethod
    @lru_cache(maxsize=32768)
    def _analyse_word_form_cached(cls, w: str) -> Tuple[Optional[str], str]:
        if not w or " " in w:
            return (None, w)
        if "-" in w:
            closed_w = w.replace("-", "")
            if cls._is_base_verb_lemma(closed_w):
                return ("VB", closed_w)
            parts = w.split("-")
            if len(parts) == 2:
                nlp = cls.get_spacy()
                if nlp:
                    d_tail = nlp(parts[1])
                    if d_tail and len(d_tail) > 0:
                        tail_lem = d_tail[0].lemma_.lower().strip() or parts[1]
                        return (d_tail[0].tag_, f"{parts[0]}-{tail_lem}")
            return (None, w)
        if w in cls._INVARIANT_PLURAL_NOUNS or (len(w) > 4 and w.endswith("ics")):
            return (None, w)
        nlp = cls.get_spacy()
        if not nlp:
            return (None, w)
        bare_tag, bare_lemma = None, w
        try:
            doc = nlp(w)
            if doc:
                tok = doc[0]
                bare_tag = tok.tag_ if tok.pos_ == "VERB" else None
                bare_lemma = (tok.lemma_ or w).lower()
        except Exception:
            pass
        if bare_lemma != w:
            dict_available = bool(cls.get_wordnet()) or (not cls._is_pedlex_disabled() and cls._ensure_ldoce_conn())
            if (
                bare_lemma in cls._INVARIANT_PLURAL_NOUNS
                or (dict_available and not cls._wordnet_pos(bare_lemma) and not cls.is_exact_pedlex_headword(bare_lemma))
            ):
                bare_lemma = w
                bare_tag = None
            elif bare_tag and not cls._is_base_verb_lemma(bare_lemma):
                bare_tag = None
                bare_lemma = w
            elif (
                bare_tag
                and "v" not in cls._wordnet_pos(w)
                and cls._wordnet_pos(w)
                and cls._inflect_verb(bare_lemma, bare_tag) != w
            ):
                bare_tag = None
                bare_lemma = w
        elif bare_tag in ("VBD", "VBN", "VBG", "VBZ") and w not in cls._IRREGULAR_VERB_FORMS.get(w, ()):
            bare_tag = None
        elif bare_tag in ("VB", "VBP") and not cls._is_base_verb_lemma(w):
            bare_tag = None
        if bare_tag:
            return (bare_tag, bare_lemma)
        if bare_lemma != w:
            return (None, bare_lemma)
        # Only consult the 5 verbal carrier frames when `w` has a plausible verbal inflection suffix
        # ('-ed', '-ing', '-s', '-en') or is a short word that could be an irregular verb form.
        if not (w.endswith(("ed", "ing", "s", "en")) or len(w) <= 6):
            return (None, bare_lemma)
        for frame in cls._LEMMA_CARRIERS[1:]:
            text = frame.format(w=w)
            try:
                doc = nlp(text)
            except Exception:
                continue
            tok = next((t for t in doc if t.text.lower() == w), None)
            if tok is not None and tok.pos_ == "VERB":
                cand_lemma = (tok.lemma_ or w).lower()
                cand_tag = tok.tag_
                if not cls._is_base_verb_lemma(cand_lemma):
                    continue
                if cand_lemma != w:
                    if (
                        cls._inflect_verb(cand_lemma, cand_tag) != w
                        and w not in cls._IRREGULAR_VERB_FORMS.get(cand_lemma, ())
                    ):
                        continue
                else:
                    if cand_tag in ("VBD", "VBN", "VBG", "VBZ") and w not in cls._IRREGULAR_VERB_FORMS.get(w, ()):
                        continue
                    if (
                        cand_tag in ("VB", "VBP")
                        and w.endswith(("ed", "ing"))
                        and "a" in cls._wordnet_pos(w)
                        and not cls._is_base_verb_lemma(w)
                    ):
                        continue
                return (cand_tag, cand_lemma)
        return (None, bare_lemma)

    @classmethod
    def _analyse_word_form(cls, word: Optional[str]) -> Tuple[Optional[str], str]:
        """
        Reads spaCy's Penn tag and lemma for a single word form.

        A bare '-ed' token is routinely tagged as an adjective ('annoyed', 'irritated'), which
        made both the lemma and the verb-form tag unusable. The word is therefore re-read inside
        minimal form-carrier frames that force a verbal parse, and the verbal reading wins.
        Returns (penn_tag_or_None, lemma).
        """
        w = (word or "").strip().lower()
        return cls._analyse_word_form_cached(w)

    @classmethod
    def verb_form_tag_of(cls, word: Optional[str]) -> Optional[str]:
        """
        Penn-Treebank tag this word form carries as a verb, or None when no frame produced a
        verbal parse. Used to detect option sets that mix verb forms.
        """
        return cls._analyse_word_form(word)[0]

    _FORM_CARRIERS = {
        "VBD": "she {form} it",
        "VBN": "she has {form} it",
        "VBG": "she is {form} it",
        "VBZ": "she {form} it",
        "VBP": "they {form} it",
        "VB": "they {form} it",
    }

    # Verbs that double the final consonant in the past and the progressive even though they
    # have two syllables, because the stress falls on the last one: emBED -> embedded, reFER
    # -> referred, conTROL -> controlled, ocCUR -> occurred.  The shape test below only
    # recognises monosyllables, so without this list PedLex's pattern 'be embedded in
    # something' can never be reached from 'embed'.  Verbs that end in -er or -or but stress
    # the first syllable ('differ', 'offer', 'visit', 'profit') are correctly left out.
    _DOUBLING_VERBS = frozenset({
        "embed", "refer", "prefer", "transfer", "confer", "defer", "concur", "demur",
        "occur", "recur", "incur", "deter", "abhor", "control", "patrol", "regret", "offset", "upset",
        "submit", "admit", "omit", "commit", "permit", "transmit", "remit", "repel",
        "expel", "compel", "dispel", "recap", "equip",
    })

    @classmethod
    def _doubles_final_consonant(cls, verb: str) -> bool:
        """CVC doubling test - monosyllables by shape ('stop'->'stopped'), stressed-final
        disyllables by list ('embed'->'embedded')."""
        w = (verb or "").strip().lower()
        if w in cls._DOUBLING_VERBS:
            return True
        if len(w) < 3:
            return False
        # Do not double vowels, semivowels, or consonants that do not double in English
        if w[-1] in "aeiouwxyjqzvh":
            return False
        if w[-2] not in "aeiou":
            return False
        # Do not double if preceded by another vowel (e.g. 'feel', 'sleep', 'spread', 'rain')
        if len(w) >= 3 and w[-3] in "aeiou":
            return False
        return len(re.findall(r"[aeiou]+", w)) == 1 and len(w) <= 8

    @classmethod
    def _inflect_verb(cls, verb: str, tag: str) -> Optional[str]:
        """Deterministically inflects a base-form verb to the requested Penn-Treebank form."""
        w = (verb or "").strip().lower()
        tag = (tag or "").upper()
        if not w:
            return None
        # Normalize already-inflected verb forms (e.g. 'led' -> 'lead', 'broken' -> 'break', 'walked' -> 'walk')
        # unless `w` is itself an attested base verb lemma (e.g. 'lay', 'found', 'fell', 'wind', 'saw').
        if not cls._is_base_verb_lemma(w):
            for _irr_base, _irr_forms in cls._IRREGULAR_VERB_FORMS.items():
                if w in _irr_forms:
                    w = _irr_base
                    break
            else:
                try:
                    _b_tag, _b_lem = cls._analyse_word_form_cached(w)
                    if _b_lem and _b_lem != w and _b_tag in ("VBD", "VBN", "VBG", "VBZ") and cls._is_base_verb_lemma(_b_lem):
                        w = _b_lem
                except Exception:
                    pass
        if tag in ("VB", "VBP"):
            return w
        irregular = cls._IRREGULAR_VERB_FORMS.get(w)
        if not irregular:
            for _pref in ("with", "over", "under", "out", "up", "re", "un", "mis", "fore", "for", "be"):
                if len(w) > len(_pref) + 2 and w.startswith(_pref):
                    _stem = w[len(_pref):]
                    _stem_irr = cls._IRREGULAR_VERB_FORMS.get(_stem)
                    if _stem_irr:
                        irregular = tuple(_pref + f for f in _stem_irr)
                        break
        if irregular:
            idx = {"VBD": 0, "VBN": 1, "VBG": 2, "VBZ": 3}.get(tag)
            return irregular[idx] if idx is not None else None
        # Consult authoritative PedLex verb_table if available (using raw row / cache to avoid derived-entry recursion)
        try:
            _vt_entry = cls._ldoce_cache.get(w)
            if _vt_entry is None and cls._ensure_ldoce_conn():
                _raw_row = cls._ldoce_raw_row(w)
                _vt_entry = _raw_row[0] if _raw_row else None
            _vt = (_vt_entry or {}).get("verb_table") if isinstance(_vt_entry, dict) else None
            if isinstance(_vt, dict) and (_vt.get("lemma") or "").strip().lower() == w:
                for _row in _vt.get("forms") or []:
                    _grp = str(_row.get("group") or "")
                    _tns = str(_row.get("tense") or "")
                    _per = str(_row.get("person") or "")
                    _fms = [str(x).strip().lower() for x in (_row.get("forms") or []) if str(x).strip() and " " not in str(x).strip()]
                    if not _fms:
                        continue
                    if tag == "VBD" and _grp == "Simple Form" and _tns == "Past":
                        return _fms[0]
                    if tag == "VBN" and _grp == "Simple Form" and _tns == "Present perfect":
                        return _fms[0]
                    if tag == "VBG" and _grp == "Continuous Form":
                        return _fms[0]
                    if tag == "VBZ" and _grp == "Simple Form" and "he, she, it" in _per and _tns in ("", "Present"):
                        return _fms[0]
        except Exception:
            pass
        if tag == "VBD":
            if w.endswith("ie"):
                return w[:-2] + "ied"
            if len(w) > 2 and w.endswith("y") and w[-2] not in "aeiou":
                return w[:-1] + "ied"
            if w.endswith("e"):
                return w + "d"
            if cls._doubles_final_consonant(w):
                return w + w[-1] + "ed"
            return w + "ed"
        if tag == "VBN":
            if w.endswith("ie"):
                return w[:-2] + "ied"
            if len(w) > 2 and w.endswith("y") and w[-2] not in "aeiou":
                return w[:-1] + "ied"
            if w.endswith("e"):
                return w + "d"
            if cls._doubles_final_consonant(w):
                return w + w[-1] + "ed"
            return w + "ed"
        if tag == "VBG":
            if w.endswith("ie"):
                return w[:-2] + "ying"
            if w.endswith(("ee", "oe", "ye")):
                return w + "ing"
            if len(w) > 2 and w.endswith("e"):
                return w[:-1] + "ing"
            if cls._doubles_final_consonant(w):
                return w + w[-1] + "ing"
            return w + "ing"
        if tag == "VBZ":
            if w.endswith(("s", "x", "z", "ch", "sh")):
                return w + "es"
            if w in ("echo", "go", "do", "veto", "torpedo", "embargo", "undergo", "forego", "outdo"):
                return w + "es"
            if len(w) > 2 and w.endswith("y") and w[-2] not in "aeiou":
                return w[:-1] + "ies"
            return w + "s"
        return None

    @classmethod
    def _morph_form_valid(cls, base: str, form: str, tag: str) -> bool:
        """
        Confirms with spaCy that `form` is a real English word whose lemma is `base` and
        that it can be tagged `tag`. Guards against the mechanical inflector inventing
        forms such as 'prefered' before they are welded into the options.
        """
        b = (base or "").strip().lower()
        f = (form or "").strip().lower()
        tag = (tag or "").upper()
        if not b or not f:
            return False
        if tag.startswith("VB") or tag in ("VBD", "VBN", "VBG", "VBZ", "VBP"):
            if not cls._is_base_verb_lemma(b) and not (cls._wordnet_pos(b) & {"v"}):
                return False
        if b in cls._IRREGULAR_VERB_FORMS and b not in {"found", "wind", "ground", "fell", "saw", "bound", "lie"}:
            expected_irr = cls._inflect_verb(b, tag)
            if expected_irr and f != expected_irr:
                return False
        if not cls.is_attested_form(f):
            return False
        nlp = cls.get_spacy()
        if not nlp:
            return f == b
        carrier = cls._FORM_CARRIERS.get(tag, "they {form} it").format(form=f)
        try:
            doc = nlp(carrier)
        except Exception:
            return False
        for tok in doc:
            if tok.text.lower() == f:
                return tok.lemma_.lower() == b
        return False

    @classmethod
    def inflected_forms(cls, word: str, pos: Optional[str] = None) -> Set[str]:
        """
        the single legal-form generator.

        Every gate that compares a word against a surface form (pattern evidence in
        _lock_sense, 'does the example contain the headword', the CEFR ceiling) needs
        the same question answered: which surface forms may this lemma legitimately
        take? Until now each call site wrote its own '\\b{word}\\b' regex, so 'attaches'
        silently failed to match the pattern 'attach something to something'.
        """
        w = (word or "").strip().lower()
        if not w:
            return set()
        key = (w, (pos or "").strip().lower())
        cached = cls._inflected_forms_cache.get(key)
        if cached is not None:
            return set(cached)
        forms: Set[str] = {w}

        irregular = cls._IRREGULAR_VERB_FORMS.get(w)
        if irregular:
            forms.update(f for f in irregular if f)
        if w == "be":
            forms.update({"am", "is", "are", "was", "were", "been", "being"})

        _irregular_adj_forms = {
            "good": ("better", "best"),
            "bad": ("worse", "worst"),
            "far": ("farther", "further", "farthest", "furthest"),
            "little": ("less", "least"),
            "many": ("more", "most"),
            "much": ("more", "most"),
        }
        if w in _irregular_adj_forms:
            forms.update(_irregular_adj_forms[w])

        p = (pos or "").strip().lower()
        _pure_adjectives = {
            "good", "bad", "happy", "sad", "big", "small", "old", "young", "new",
            "high", "low", "long", "short", "large", "great", "little", "different",
            "important", "available", "possible", "impossible", "likely", "unlikely",
            "able", "unable", "sure", "certain", "aware", "afraid", "alive", "alone",
            "asleep", "awake",
        }
        if w not in ("a", "an", "the", "as", "at", "to", "in", "on", "of", "by", "or", "so", "if"):
            # Multi-word non-verb phrases (e.g. 'climate change' as noun) and pure adjectives must not undergo verb inflections
            is_verb_eligible = (
                ((" " not in w) or (p in ("verb", "v", "phrasal verb")))
                and p not in ("adjective", "adj", "a", "s", "adverb", "adv", "r")
                and w not in _pure_adjectives
            )
            # If multi-word and no POS explicitly provided, check if the head word is a verb
            if " " in w and not is_verb_eligible and p not in ("noun", "n", "adjective", "adj", "a", "s", "adverb", "adv", "r"):
                try:
                    head_tok = w.split()[0]
                    if head_tok in cls._IRREGULAR_VERB_FORMS or head_tok == "be" or cls.get_spacy()(w)[0].pos_ == "VERB":
                        is_verb_eligible = True
                except Exception:
                    pass

            if is_verb_eligible:
                if " " in w:
                    w_parts = w.split()
                    head_v = w_parts[0]
                    v_tail = " " + " ".join(w_parts[1:])
                    for tag in ("VB", "VBZ", "VBD", "VBN", "VBG"):
                        inf_head = cls._inflect_verb(head_v, tag)
                        if inf_head:
                            forms.add(inf_head + v_tail)
                    if head_v == "be" or head_v in ("am", "is", "are", "was", "were", "been", "being"):
                        for be_form in ("am", "is", "are", "was", "were", "been", "being"):
                            forms.add(be_form + v_tail)
                else:
                    for tag in ("VBZ", "VBD", "VBN", "VBG"):
                        inflected = cls._inflect_verb(w, tag)
                        if inflected:
                            forms.add(inflected)

            if p not in ("adjective", "adj", "a", "s", "adverb", "adv", "r", "verb", "v") and w not in _pure_adjectives:
                if " " not in w:
                    plural = cls.pluralize_noun(w)
                    if plural:
                        forms.add(plural)

            # Hyphenated prefix/compound spelling variant (e.g. 're-schedule' <-> 'reschedule', 'in-person' <-> 'in person')
            if "-" in w and " " not in w:
                forms.add(w.replace("-", " "))
                closed_w = w.replace("-", "")
                if len(closed_w) >= 4:
                    forms.add(closed_w)
                    if is_verb_eligible:
                        for tag in ("VBZ", "VBD", "VBN", "VBG"):
                            inf_c = cls._inflect_verb(closed_w, tag)
                            if inf_c:
                                forms.add(inf_c)
                    pl_c = cls.pluralize_noun(closed_w)
                    if pl_c:
                        forms.add(pl_c)

        if p in ("", "adjective", "adj", "a", "s") and " " not in w:
            if len(w) >= 3:
                if w.endswith("e"):
                    forms.update({w + "r", w + "st"})
                elif w.endswith("y") and w[-2] not in "aeiou":
                    forms.update({w[:-1] + "ier", w[:-1] + "iest"})
                else:
                    if len(w) >= 3 and w[-1] in "bdfgklmnprstz" and w[-2] in "aeiou" and (len(w) == 3 or w[-3] not in "aeiou"):
                        forms.update({w + w[-1] + "er", w + w[-1] + "est"})
                    forms.update({w + "er", w + "est"})

        # US/UK orthographic equivalence closure across all inflected forms
        for f_cand in list(forms):
            if hasattr(cls, "_pure_orthographic_variants"):
                for ov in cls._pure_orthographic_variants(f_cand):
                    forms.add(ov)
            if f_cand.endswith(("enroll", "unroll", "fulfill", "instill", "distill", "appall", "enthrall")):
                uk_l = f_cand[:-1]
                forms.update({uk_l, uk_l + "s", uk_l + "led", uk_l + "ling"})
            elif f_cand.endswith(("enrol", "unrol", "fulfil", "instil", "distil", "appal", "enthral")):
                us_ll = f_cand + "l"
                forms.update({us_ll, us_ll + "s", us_ll + "ed", us_ll + "ing"})

        forms = {f for f in forms if f}
        cls._inflected_forms_cache[key] = set(forms)
        return forms

    @classmethod
    def _surface_lemma(cls, form: str) -> Optional[str]:
        """
        Lemma of a bare surface form, resolved through carrier phrases.

        spaCy mis-tags an isolated token - 'attaches' on its own is read as a proper
        noun and its lemma comes back unchanged - so the form is disambiguated inside
        'the x', 'they x it' and 'very x here' before a lemma is trusted.
        """
        f = (form or "").strip().lower()
        if not f:
            return None
        if f in cls._surface_lemma_cache:
            return cls._surface_lemma_cache[f]
        nlp = cls.get_spacy()
        if not nlp:
            return None
        lemma: Optional[str] = None
        for text in (f, f"the {f}", f"they {f} it", f"very {f} here"):
            try:
                doc = nlp(text)
            except Exception:
                continue
            for tok in doc:
                if tok.text.lower() == f:
                    candidate = tok.lemma_.lower()
                    if candidate and candidate != f:
                        lemma = candidate
                        break
            if lemma:
                break
        cls._surface_lemma_cache[f] = lemma
        return lemma

    @classmethod
    def deinflect_comparative_superlative(cls, word: str) -> Optional[str]:
        """
        De-inflects an inflected comparative (-er, -ier, -r) or superlative (-est, -iest, -st)
        adjective form back to its dictionary base lemma (e.g. 'warmer' -> 'warm', 'colder' -> 'cold',
        'hotter' -> 'hot', 'happier' -> 'happy', 'largest' -> 'large', 'better' -> 'good').
        Returns the base adjective lemma if attested in PedLex/LDOCE/WordNet, or None.
        """
        w = (word or "").strip().lower()
        if not w or len(w) < 4:
            return None
        irregular = {
            "better": "good", "best": "good",
            "worse": "bad", "worst": "bad",
            "farther": "far", "further": "far", "farthest": "far", "furthest": "far",
            "less": "little", "least": "little",
            "more": "much", "most": "much",
        }
        if w in irregular:
            return irregular[w]
        stems: List[str] = []
        if w.endswith("iest") and len(w) >= 5:
            stems.append(w[:-4] + "y")
        if w.endswith("ier") and len(w) >= 4:
            stems.append(w[:-3] + "y")
        if w.endswith("est") and len(w) >= 4:
            if len(w) >= 5 and w[-4] == w[-5] and w[-4] in "bdfgklmnprstz":
                stems.append(w[:-4])
            stems.append(w[:-3])
            stems.append(w[:-2])
        if w.endswith("er") and len(w) >= 4:
            if len(w) >= 4 and w[-3] == w[-4] and w[-3] in "bdfgklmnprstz":
                stems.append(w[:-3])
            stems.append(w[:-2])
            stems.append(w[:-1])

        for stem in stems:
            if len(stem) < 3:
                continue
            ld = cls.get_ldoce_entry(stem)
            if ld and ld.get("kind") != "derived":
                all_p = [str(ld.get("pos") or "")] + [str(s.get("pos") or "") for s in (ld.get("senses") or [])]
                if any("adj" in p.lower() for p in all_p):
                    if w in cls.inflected_forms(stem, pos="adjective"):
                        return stem
            wn = cls.get_wordnet()
            if wn and (wn.synsets(stem, pos="a") or wn.synsets(stem, pos="s")):
                if w in cls.inflected_forms(stem, pos="adjective"):
                    return stem
        return None

    @classmethod
    def is_causative_object_complement(cls, tok: Any, doc: Any) -> bool:
        """
        Determines whether `tok` functions as the predicative adjective Object Complement
        in a complex-transitive causative/resultative construction:
            Verb_causative + Object_NP + Complement_Adj
        e.g. 'make the Earth warmer', 'keep the water cleaner', 'render the system faster',
             'leave the door open', 'turn the liquid darker'.
        """
        CAUSATIVE_RESULTATIVE_VERBS = frozenset({
            "make", "keep", "render", "leave", "turn", "get", "drive", "set", "paint", "strike"
        })
        head_v = tok.head
        while head_v and head_v.pos_ not in ("VERB", "AUX") and head_v.head != head_v:
            head_v = head_v.head
        if not head_v or head_v.pos_ not in ("VERB", "AUX"):
            return False
        if head_v.lemma_.lower() not in CAUSATIVE_RESULTATIVE_VERBS or head_v.i >= tok.i:
            return False
        # Intervening direct object NP between head verb and complement token
        intervening = doc[head_v.i + 1 : tok.i]
        has_obj = any(t.pos_ in ("NOUN", "PROPN", "PRON") for t in intervening)
        has_absorbed_compound = any(c.dep_ in ("compound", "nmod") and c.pos_ in ("NOUN", "PROPN") for c in tok.children)
        return has_obj or has_absorbed_compound

    @classmethod
    def form_matches(cls, surface: str, target: str) -> bool:
        """
        True when `surface` IS `target` or a legal inflection of it ('attaches' ~ 'attach',
        'attaching' ~ 'attach', 'children' ~ 'child'). spaCy's lemmatizer is the fallback
        for forms the deterministic generator cannot derive ('chose' ~ 'choose').
        """
        s = (surface or "").strip().lower()
        t = (target or "").strip().lower()
        if not s or not t:
            return False
        if s == t:
            return True
        if s in cls.inflected_forms(t) or t in cls.inflected_forms(s):
            return True
        # Lemma fallback only for forms the generator cannot derive. It requires the
        # surface to be a real word first, otherwise a misspelling that happens to
        # lemmatize back to the target ('choosed' -> 'choose') counts as a match.
        return cls.is_attested_form(s) and cls._surface_lemma(s) == t

    @classmethod
    def text_contains_form(cls, text: Optional[str], target: str, standalone: bool = True) -> bool:
        """
        Does `text` contain `target` as a word or as any legal inflection of it?
        
        When `standalone=True` (default), strictly requires non-hyphen word boundaries
        for non-hyphenated targets so severed sub-tokens of hyphenated compounds
        (e.g. 'prone' in 'waste-prone', 'indulgent' in 'self-indulgent', 'son' in 'son-in-law')
        never falsely count as standalone occurrences.
        If `target` itself contains a hyphen (e.g. 'in-person'), also matches its open
        space-separated form ('in person') in dictionary examples.
        """
        body = (text or "").lower()
        t = (target or "").strip().lower()
        if not body or not t:
            return False

        forms = [t]
        try:
            forms = list(set([t] + list(cls.inflected_forms(t))))
        except Exception:
            pass

        for f in forms:
            if "-" in f:
                pat = rf"(?<![a-zA-Z0-9\-]){re.escape(f)}(?![a-zA-Z0-9\-])"
                if re.search(pat, body):
                    return True
                open_f = f.replace("-", " ")
                pat_open = rf"(?<![a-zA-Z0-9\-]){re.escape(open_f)}(?![a-zA-Z0-9\-])"
                if re.search(pat_open, body):
                    return True
                if " " not in f:
                    closed_f = f.replace("-", "")
                    if len(closed_f) >= 4:
                        pat_closed = rf"(?<![a-zA-Z0-9\-])\b{re.escape(closed_f)}\b(?![a-zA-Z0-9\-])"
                        if re.search(pat_closed, body):
                            return True
            elif standalone:
                pat = rf"(?<![a-zA-Z0-9\-])\b{re.escape(f)}\b(?![a-zA-Z0-9\-])"
                if re.search(pat, body):
                    return True
            else:
                pat = rf"\b{re.escape(f)}\b"
                if re.search(pat, body):
                    return True
        return False

    @classmethod
    def is_attested_form(cls, form: str) -> bool:
        """
        Is this surface form an actual English word form?

        Grounds on WordNet and PedLex 6th Edition (SQLite `pedlex` table). If the form
        is attested in WordNet or directly in the PedLex database, it is accepted immediately.
        Otherwise, resolves lemma via spaCy and verifies that the lemma is attested in WordNet
        or PedLex and that the form legitimately derives from that lemma (e.g. regular noun plural
        or verb inflection).
        """
        f = (form or "").strip().lower()
        if not f:
            return False
        if f in cls._attested_form_cache:
            return cls._attested_form_cache[f]
        if "-" in f and cls.is_attested_form(f.replace("-", "")):
            cls._attested_form_cache[f] = True
            return True
        verdict = False
        try:
            wn = cls.get_wordnet()
            if wn and wn.words(f):
                verdict = True
            elif cls._ensure_ldoce_conn():
                cur = cls._ldoce_conn.cursor()
                cur.execute("SELECT 1 FROM ldoce WHERE word = ? LIMIT 1", (f,))
                if cur.fetchone() is not None:
                    verdict = True

            if not verdict:
                lemma = cls._surface_lemma(f)
                if lemma:
                    # Lemma must be a real attested word in WordNet or PedLex
                    lemma_attested = False
                    if wn and wn.words(lemma):
                        lemma_attested = True
                    elif cls._ensure_ldoce_conn():
                        cur = cls._ldoce_conn.cursor()
                        cur.execute("SELECT 1 FROM ldoce WHERE word = ? LIMIT 1", (lemma,))
                        if cur.fetchone() is not None:
                            lemma_attested = True

                    if lemma_attested:
                        # Guard against lemma being an inflected form alias in PedLex (e.g. 'spake' -> 'a past tense of speak', 'chose' -> 'past tense of choose')
                        entry = cls.get_ldoce_entry(lemma) if cls._ensure_ldoce_conn() else None
                        is_inflected_alias = False
                        if entry:
                            for s in entry.get("senses", []):
                                d = (s.get("definition") or "").lower()
                                if re.search(r"\b(?:past tense|past participle|plural of)\b", d):
                                    is_inflected_alias = True
                                    break
                        if not is_inflected_alias:
                            lemma_poses = cls._wordnet_pos(lemma)
                            lemma_ldoce_pos = str((entry or {}).get("pos") or "").lower()
                            can_be_noun = bool(lemma_poses & {"n"}) or "noun" in lemma_ldoce_pos
                            can_be_verb = (
                                cls._is_base_verb_lemma(lemma)
                                or bool(lemma_poses & {"v"})
                                or "verb" in lemma_ldoce_pos
                            )
                            # Check if f is the legitimate plural or inflected verb form of this attested lemma
                            if can_be_noun and cls.pluralize_noun(lemma) == f:
                                verdict = True
                            elif can_be_verb and any(cls._inflect_verb(lemma, tag) == f for tag in ("VBZ", "VBD", "VBN", "VBG")):
                                verdict = True
        except Exception:
            verdict = False
        cls._attested_form_cache[f] = verdict
        return verdict


    @classmethod
    def is_known_english_word(cls, word: str) -> Optional[bool]:
        """
        Tri-state lexical validation against the local Open English WordNet client.

        Returns:
            True  -> recognized English word (e.g. 'embed', 'hop')
            False -> confirmed non-word, e.g. a lemmatizer fragment like 'embe'
            None  -> not verifiable (multi-word, non-alphabetic, or WordNet unavailable)

        Backed by the local SQLite WordNet client, so no network access is required.
        """
        w = (word or "").strip().lower()
        if not re.fullmatch(r"[a-z]+(?:['\-][a-z]+)*", w):
            return None  # multi-word / slots / non-alphabetic: out of scope
        try:
            alt = w.replace("'", "").replace("-", "")
            wn_client = cls.get_wordnet()
            if wn_client is not None:
                if len(wn_client.lemmas(w)) > 0:
                    return True
                # Retry with punctuation flattened (e.g. 'ice-cream' -> 'icecream')
                if alt != w and len(wn_client.lemmas(alt)) > 0:
                    return True

            # Fallback 1: PedLex 6th Edition dictionary check (covers contemporary words like 'selfie' and function words like 'whether', 'until', 'what')
            try:
                ldoce_entry = cls.get_ldoce_entry(w)
                if ldoce_entry:
                    return True
                if alt != w and cls.get_ldoce_entry(alt):
                    return True
            except Exception:
                pass

            # Fallback 2: cefrpy Lexical Database (covers advanced derived terms like 'entrepreneurship', 'multidiscipline')
            try:
                from cefrpy import CEFRAnalyzer
                if CEFRAnalyzer().is_word_in_database(w):
                    return True
                if alt != w and CEFRAnalyzer().is_word_in_database(alt):
                    return True
            except Exception:
                pass

            if wn_client is None and cls._is_pedlex_disabled():
                return None
            return False
        except Exception:
            return None  # WordNet unavailable: let callers decide the safe default

    @classmethod
    def _guard_lemma(cls, surface: str, lemma: str) -> str:
        """
        Accept a spaCy lemma only when it is a recognized English word.

        spaCy's rule-based lemmatizer can emit out-of-vocabulary guesses
        (e.g. 'embed' -> 'embe') when the tagger mislabels a base form as
        inflected. In that case the surface form — which comes from the real
        source text — is the safer headword.
        """
        lemma = lemma.lower()
        if lemma == surface.lower():
            return lemma
        if cls.is_known_english_word(lemma) is True:
            return lemma
        # Lemma is a non-word (or unverifiable): keep the original surface form.
        return surface.lower()

    @classmethod
    def lemmatize_headword(cls, word_or_phrase: str, context_sentence: Optional[str] = None) -> str:
        """
        Derives canonical base dictionary headwords from sentence context,
        eliminating inflected headwords (-ed, -ing, 3sg).
        Lemmas are validated against the dictionary so the lemmatizer's
        OOV fallback can never corrupt a headword (e.g. 'embed' -> 'embe').
        """
        nlp = cls.get_spacy()
        target = word_or_phrase.strip()

        # If a single word or hyphenated compound
        if "-" in target:
            # Hyphenated compound: do not take doc[0] which chops to prefix!
            wn = cls.get_wordnet()
            if wn and (wn.synsets(target) or wn.synsets(target.replace("-", "")) or wn.synsets(target.replace("-", "_"))):
                return target.lower()
            parts = target.split("-")
            last_lemma = cls.lemmatize_headword(parts[-1], context_sentence)
            return "-".join(parts[:-1] + [last_lemma])

        if " " not in target:
            doc = nlp(target)
            if len(doc) > 0 and doc[0].pos_ in ("VERB", "NOUN", "ADJ"):
                return cls._guard_lemma(target, doc[0].lemma_)
            return target.lower()

        # If a multi-word expression (e.g. phrasal verb or idiom)
        doc = nlp(target)
        lemmatized_tokens = []
        for token in doc:
            if token.pos_ == "VERB":
                lemmatized_tokens.append(cls._guard_lemma(token.text, token.lemma_))
            else:
                lemmatized_tokens.append(token.text.lower())
        return " ".join(lemmatized_tokens)

    @classmethod
    def pluralize_noun(cls, word: str) -> str:
        """
        Converts a singular noun to its plural form with support for standard morphology
        and common irregulars. Used to enforce parallel inflection across options.

        Memoized: the already-plural probe below is a spaCy call, and this function sits
        inside every inflection-aware match, so uncached it dominates the engine's runtime.
        """
        w = (word or "").strip().lower()
        if not w:
            return word
        cached = cls._plural_form_cache.get(w)
        if cached is not None:
            return cached
        result = cls._pluralize_noun_uncached(w)
        cls._plural_form_cache[w] = result
        return result

    @classmethod
    def _is_genuine_plural_noun(cls, word: str) -> bool:
        """
        Determines whether `word` is genuinely an already-plural noun form (e.g. 'things',
        'assets', 'criteria', 'men', 'goods') while rejecting spaCy tagger suffix hallucinations
        on singular Latinate `-men` nouns (`specimen` -> `speciman`, `omen` -> `oman`,
        `regimen` -> `regiman`, `acumen` -> `acuman`).
        """
        w = (word or "").strip().lower()
        if not w:
            return False
        plural_tantum = {
            "goods", "customs", "clothes", "belongings", "surroundings",
            "fireworks", "premises", "congratulations", "people", "men",
            "women", "children", "feet", "teeth", "mice", "lice", "geese", "oxen",
            "criteria", "phenomena", "data", "media", "bacteria", "curricula",
            "strata", "memoranda", "analyses", "crises", "theses", "bases",
            "foci", "alumni", "stimuli", "fungi", "cacti", "nuclei", "syllabi",
            "radii", "larvae", "antennae", "formulae", "vertebrae", "algae",
        }
        if w in plural_tantum:
            return True
        if w.endswith("ics") and len(w) >= 5:
            return True
        if not w.endswith("s") or w.endswith(("ss", "us", "is")):
            return False
        ld_w = cls.get_ldoce_entry(w)
        if ld_w:
            for s in ld_w.get("senses") or []:
                g_low = str(s.get("grammar") or "").lower()
                if re.search(r"\bplural\b", g_low):
                    return True
        nlp = cls.get_spacy()
        doc = nlp(f"these {w}")
        tok = doc[1] if len(doc) > 1 else doc[0]
        if tok.tag_ != "NNS":
            return False
        lem = tok.lemma_.lower()
        if lem == w:
            return False
        # Verify that the lemmatized singular `lem` is a real attested lowercase noun in WordNet or PedLex
        # (rejecting fake -man lemmas like 'speciman'/'regiman' and proper-noun collisions like 'Oman').
        try:
            wn = cls.get_wordnet()
            if any(x.lemma() == lem for x in wn.words(lem, pos="n")):
                return True
        except Exception:
            pass
        ld_ent = cls.get_ldoce_entry(lem)
        if ld_ent and any("noun" in (p or "").lower() for p in ld_ent.get("all_poses", [ld_ent.get("pos") or ""])):
            return True
        return False

    @classmethod
    def _pluralize_noun_uncached(cls, w: str) -> str:
        irregulars = {
            'person': 'people', 'man': 'men', 'woman': 'women', 'child': 'children',
            'foot': 'feet', 'tooth': 'teeth', 'mouse': 'mice', 'criterion': 'criteria',
            'phenomenon': 'phenomena', 'datum': 'data', 'analysis': 'analyses',
            'crisis': 'crises', 'thesis': 'theses', 'basis': 'bases', 'focus': 'foci',
            'goods': 'goods', 'customs': 'customs', 'clothes': 'clothes',
            'belongings': 'belongings', 'surroundings': 'surroundings',
            'fireworks': 'fireworks', 'premises': 'premises', 'congratulations': 'congratulations',
            'species': 'species', 'subspecies': 'subspecies', 'series': 'series',
            'means': 'means', 'offspring': 'offspring', 'aircraft': 'aircraft',
            'spacecraft': 'spacecraft', 'headquarters': 'headquarters', 'crossroads': 'crossroads',
            'corps': 'corps', 'sheep': 'sheep', 'deer': 'deer', 'fish': 'fish',
            'salmon': 'salmon', 'trout': 'trout', 'swine': 'swine', 'bison': 'bison', 'moose': 'moose'
        }
        if w in irregulars:
            return irregulars[w]
        if w in irregulars.values():
            return w
        if w == 'lens':
            return 'lenses'

        # Check if word is already recognized as a genuine plural noun (e.g. 'things', 'assets', 'possessions', 'dreams')
        # to prevent corrupt pseudo-plurals like 'thingses' or 'dreamses' while properly pluralizing 'specimen' -> 'specimens'.
        if cls._is_genuine_plural_noun(w):
            return w

        if w.endswith(('s', 'sh', 'ch', 'x', 'z')):
            return w + 'es'
        if w in ('echo', 'hero', 'potato', 'tomato', 'veto', 'torpedo', 'embargo', 'volcano'):
            return w + 'es'
        if w.endswith('y') and len(w) > 1 and w[-2] not in 'aeiou':
            return w[:-1] + 'ies'
        if w.endswith('f'):
            if w in ('leaf', 'half', 'wolf', 'shelf', 'thief', 'calf', 'loaf'):
                return w[:-1] + 'ves'
        if w.endswith('fe'):
            if w in ('life', 'knife', 'wife'):
                return w[:-2] + 'ves'
        return w + 's'

    @classmethod
    def can_pluralize_noun_distractor(cls, word: str) -> bool:
        """
        Determines whether a singular noun candidate `word` can be naturally pluralized into
        an attested plural form (`NNS`) for a plural-noun quiz slot.
        Rejects:
          1. Invariant nouns whose plural equals the singular (`subspecies`, `series`, `aircraft`),
          2. Purely uncountable nouns (`pharma`, `subsistence`, `celibacy`),
          3. Predominantly uncountable `[countable, uncountable]` nouns whose mechanical `-s` plural
             (e.g. `subsidences` from `subsidence`) is completely unattested in WordNet, `cefrpy`,
             and PedLex sense examples/collocations.
        """
        w = (word or "").strip().lower()
        if not w or " " in w:
            return False
        pl = cls.pluralize_noun(w)
        if not pl:
            return False
        if pl == w:
            if w.endswith("s") and not w.endswith(("ss", "us", "is")) and cls._is_genuine_plural_noun(w):
                return True
            return False
        gram_prof = cls.get_word_gram_profile(w, pos="noun")
        if gram_prof.get("pure_uncountable"):
            return False
        if not cls.is_attested_form(pl):
            return False
        # Check if the noun has any uncountable tag in PedLex; if so, require explicit plural attestation
        entry = cls.get_ldoce_entry(w)
        has_uncountable_tag = False
        if entry:
            for s in entry.get("senses", []):
                g = str(s.get("grammar") or "").lower()
                if "uncountable" in g:
                    has_uncountable_tag = True
                    break
        if has_uncountable_tag:
            wn = cls.get_wordnet()
            if wn and wn.words(pl):
                return True
            try:
                cefr = cls.get_cefr_analyzer()
                if cefr and cefr.get_average_word_level_CEFR(pl) is not None:
                    return True
            except Exception:
                pass
            if entry:
                for s in entry.get("senses", []):
                    for ex in s.get("examples", []):
                        if re.search(rf"\b{re.escape(pl)}\b", str(ex).lower()):
                            return True
                    for col in s.get("collocations", []):
                        c_txt = col.get("pattern", "") if isinstance(col, dict) else str(col)
                        if re.search(rf"\b{re.escape(pl)}\b", c_txt.lower()):
                            return True
            return False
        return True

