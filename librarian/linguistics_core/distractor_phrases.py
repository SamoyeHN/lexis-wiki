"""Multi-word phrase distractors, idiom mapping, and phrase normalization."""

from __future__ import annotations

import logging
import re
import sqlite3
from typing import Any, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger("librarian.linguistics.phrases")


class DistractorPhrasesMixin:

    _PHRASE_INDEX_CACHE: Optional[Dict[str, str]] = None
    _PHRASE_SYNONYMS_CACHE: Optional[Dict[str, Set[str]]] = None

    _TABOO_PHRASE_TOKENS: Set[str] = {
        "arse", "ass", "asshole", "bastard", "bitch", "bloody", "bollocks",
        "bugger", "bum", "crap", "cunt", "damn", "dick", "fag", "fart", "fuck", "fucking",
        "motherfucker", "nigger", "piss", "prick", "pussy", "shit", "shitty",
        "slut", "tit", "twat", "wanker", "whore", "doss"
    }

    _TABOO_MULTIWORD_PHRASES: Set[str] = {
        "go down on", "get it on", "get laid", "knock up", "jerk off", "jack off",
        "toss off", "make out with", "sleep with", "sleep around", "fool around with",
        "come on to", "have it off", "do it with", "turn on", "screw around",
    }

    _FRAME_PREP_SYNONYM_GROUPS: List[Set[str]] = [
        {"in light of", "in view of", "in the light of"},
        {"for the sake of", "for the good of", "for the benefit of"},
        {"on the brink of", "on the verge of", "on the edge of"},
        {"by means of", "by way of", "by dint of"},
        {"with regard to", "with respect to", "in regard to", "as regards"},
        {"in terms of", "in respect of"},
        {"at the expense of", "at the cost of"},
        {"in the event of", "in case of"},
        {"in accordance with", "in compliance with", "in conformity with"},
        {"on behalf of", "in behalf of"},
        {"all at once", "all of a sudden"},
        {"first and foremost", "first of all"},
        {"more or less", "approximately"},
    ]

    _SYN_VERB_HEAD_CLUSTERS: List[Set[str]] = [
        {"bear", "keep", "have", "hold"},
        {"hold", "catch", "take", "draw"},
        {"make", "take", "have", "give", "do"},
        {"come", "go", "get", "rush", "hurry"},
        {"turn", "direct", "shift", "focus", "draw", "call", "pay", "give"},
        {"lose", "drop", "miss"},
    ]

    _PARTICLES_AND_PREPS: Set[str] = {
        "in", "off", "up", "down", "away", "back", "over", "into", "through",
        "on", "to", "for", "out", "with", "from", "around", "about", "of",
        "by", "at", "under", "between", "against", "after", "before",
        "without", "within", "along", "across", "behind", "beyond", "upon",
        "onto", "towards", "toward", "aside", "apart", "ahead", "aback", "forth"
    }

    _DEFINING_GENUS_VERBS: Set[str] = {
        "be", "have", "do", "make", "get", "give", "take", "use", "say", "think",
        "feel", "know", "show", "cause", "keep", "put", "go", "come", "move",
        "turn", "start", "stop", "try", "want", "need", "help", "allow", "let",
        "mean", "seem", "become", "act", "happen"
    }

    @staticmethod
    def normalize_phrase_possessives(s: str) -> str:
        return re.sub(
            r"\b(?:one's|your|my|his|her|its|our|their|sb's|somebody's|someone's)\b",
            "<POSS>",
            (s or "").lower(),
        )

    @staticmethod
    def normalize_phrase_determiners(s: str) -> str:
        return re.sub(
            r"\b(?:a|an|the|one's|your|my|his|her|its|our|their|sb's|somebody's|someone's)\b",
            "<DET>",
            (s or "").lower(),
        )

    @classmethod
    def _get_phrase_def_text(cls, p_str: str) -> str:
        chunks: List[str] = []
        e_direct = cls.get_ldoce_entry(p_str)
        if e_direct:
            for s in e_direct.get("senses", []):
                chunks.append(str(s.get("definition") or ""))
        p_tokens = p_str.split()
        if p_tokens:
            e_head = cls.get_ldoce_entry(p_tokens[0])
            if e_head:
                for s in e_head.get("senses", []):
                    phw = str(s.get("phrasal_headword") or "").lower()
                    pats = " ".join(str(pt) for pt in (s.get("patterns") or [])).lower()
                    if p_str in phw or p_str in pats:
                        chunks.append(str(s.get("definition") or ""))
                for pv in e_head.get("phrasal_verbs", []) or []:
                    if isinstance(pv, dict):
                        pv_name = str(pv.get("phrase") or pv.get("headword") or "").lower()
                        if p_str in pv_name:
                            for pvs in pv.get("senses", []) or []:
                                if isinstance(pvs, dict):
                                    chunks.append(str(pvs.get("definition") or ""))
                for ph in e_head.get("phrases", []) or []:
                    if isinstance(ph, dict):
                        ph_name = str(ph.get("phrase") or ph.get("headword") or "").lower()
                        if p_str in ph_name:
                            for phs in ph.get("senses", []) or []:
                                if isinstance(phs, dict):
                                    chunks.append(str(phs.get("definition") or ""))
        return " ; ".join(chunks).lower()

    @classmethod
    def is_inappropriate_phrase(cls, phrase: str) -> bool:
        """
        Determines whether a candidate phrase contains taboo, vulgar, offensive,
        or coarse slang tokens unsuited for pedagogical curriculum assessments.
        """
        tokens = [t.lower() for t in re.findall(r'[a-zA-Z]+', phrase)]
        if not tokens:
            return False
        if any(t in cls._TABOO_PHRASE_TOKENS for t in tokens):
            return True
        norm_p = " ".join(tokens)
        if norm_p in cls._TABOO_MULTIWORD_PHRASES:
            return True
        return False

    @classmethod
    def _get_phrase_synonyms(cls) -> Dict[str, Set[str]]:
        """
        Loads and memoizes synonym groups generated by slash-alternations in PedLex raw phrases
        (e.g., 'be/keep/stay etc in touch (with something)' -> {'be in touch with', 'keep in touch with', 'stay in touch with'}).
        """
        if cls._PHRASE_SYNONYMS_CACHE is not None:
            return cls._PHRASE_SYNONYMS_CACHE
        if cls._ensure_in_memory_phrase_index() and cls._PHRASE_SYNONYMS_CACHE is not None:
            return cls._PHRASE_SYNONYMS_CACHE
        cls._PHRASE_SYNONYMS_CACHE = {}
        return cls._PHRASE_SYNONYMS_CACHE

    @classmethod
    def _get_phrase_index(cls) -> Dict[str, str]:
        """Loads and memoizes {phrase_key: tier} from ldoce_phrase_index in the PedLex database."""
        if cls._PHRASE_INDEX_CACHE is not None:
            return cls._PHRASE_INDEX_CACHE
        if cls._ensure_in_memory_phrase_index() and cls._PHRASE_INDEX_CACHE is not None:
            return cls._PHRASE_INDEX_CACHE
        cls._PHRASE_INDEX_CACHE = {}
        return cls._PHRASE_INDEX_CACHE

    @classmethod
    def generate_phrase_distractors(
        cls,
        phrase: str,
        pos: str = "",
        count: int = 3,
        exclude_words: Optional[Set[str]] = None,
        definition: Optional[str] = None,
        quote: Optional[str] = None,
    ) -> List[str]:
        """
        Synthesizes high-discrimination, authentic distractors for multi-word phrasal verbs, idioms, and collocations
        strictly grounded in PedLex 6th Edition's ldoce_phrase_index and lexical evidence.
        
        Symmetric structural families:
          1. Head/Tail Sharing (e.g. 'peace of mind' -> 'state of mind', 'frame of mind', 'turn of mind';
                                    'keep in touch with' -> 'stay in touch with', 'get in touch with')
          2. Light-Verb & Lexical Frames (e.g. 'have a try' -> 'have a go', 'have a heart', 'have a care')
          3. Binomial / Parallel Match (e.g. 'pros and cons' -> 'bits and pieces', 'back and forth', 'give and take')
          4. 2-Word Phrasal Verbs (Particle contrast & Core verb contrast via phrasal_verb index)
        """
        phrase_clean = phrase.strip().lower()
        words = phrase_clean.split()
        exclude = set(exclude_words or [])
        distractors: List[str] = []
        n_words = len(words)

        phrase_index = cls._get_phrase_index()
        phrase_synonyms = cls._get_phrase_synonyms()
        synonym_set = phrase_synonyms.get(phrase_clean, set())

        _NON_VERB_PHRASE_HEADS = {
            "a", "an", "the", "no", "not", "all", "any", "some", "every", "each", "both",
            "either", "neither", "much", "many", "more", "most", "less", "least", "few",
            "little", "other", "another", "such", "what", "which", "whose", "this", "that",
            "these", "those", "my", "your", "his", "her", "its", "our", "their", "one",
            "in", "on", "at", "by", "for", "with", "from", "to", "of", "out", "up", "down",
            "over", "under", "into", "onto", "upon", "about", "above", "below", "between",
            "among", "through", "during", "before", "after", "since", "until", "without",
            "within", "against", "along", "around", "behind", "beside", "beyond", "near",
            "off", "oward", "towards", "as", "if", "so", "than", "though", "while", "when"
        }
        head_w = words[0] if words else ""
        _PARTICLES_AND_PREPS = {
            "in", "off", "up", "down", "away", "back", "over", "into", "through",
            "on", "to", "for", "out", "with", "from", "around", "about", "of",
            "by", "at", "under", "between", "against", "after", "before",
            "without", "within", "along", "across", "behind", "beyond", "upon",
            "onto", "towards", "toward", "aside", "apart", "ahead", "aback", "forth"
        }
        _VERB_NP_HEADS = {
            "take", "make", "have", "do", "give", "get", "keep", "lose", "catch",
            "pay", "save", "bear", "hold", "set", "play", "run", "put", "bring",
            "turn", "go", "come", "fall", "break", "cut", "draw", "drop", "find",
            "form", "gain", "lay", "lead", "leave", "meet", "pass", "pull", "push",
            "raise", "reach", "see", "send", "shake", "show", "spend", "spread",
            "stand", "start", "stay", "strike", "tell", "throw", "touch", "use",
            "waste", "win", "work", "be", "become", "feel", "seem", "look", "sound",
            "remain", "grow", "prove", "smell", "taste", "change", "join", "cross"
        }
        t_entry_top = cls.get_ldoce_entry(phrase_clean) if n_words == 2 else None
        t_pos_top = str((t_entry_top or {}).get("pos", "")).strip().lower()
        pos_norm = str(pos or "").strip().lower()
        if not pos_norm or pos_norm in ("phrase", "collocation", "set phrase", "expression"):
            if phrase_index.get(phrase_clean) == "phrasal_verb":
                pos_norm = "phrasal_verb"
            elif n_words == 2 and words[0] in _VERB_NP_HEADS and words[1] in _PARTICLES_AND_PREPS:
                pos_norm = "phrasal_verb"
            elif words and words[0] in _PARTICLES_AND_PREPS:
                pos_norm = "preposition"
            elif n_words == 2 and (
                bool(cls._wordnet_pos(words[1]) & {"n"})
                or str((cls.get_ldoce_entry(words[1]) or {}).get("pos") or "").strip().lower() == "noun"
            ):
                pos_norm = "noun"
            else:
                pos_norm = "collocation"
        _ADV_OR_ADJ_COMPLEMENTS = {
            "better", "worse", "best", "worst", "well", "badly", "hard", "fast",
            "far", "more", "less", "long", "short", "high", "low", "early", "late",
            "alone", "wrong", "right", "free", "clear", "open", "close", "sure", "certain",
        }
        is_compound_noun = (
            n_words == 2
            and words[0] not in _PARTICLES_AND_PREPS
            and words[0] not in _NON_VERB_PHRASE_HEADS
            and words[1] not in _PARTICLES_AND_PREPS
            and words[1] not in _ADV_OR_ADJ_COMPLEMENTS
            and str((cls.get_ldoce_entry(words[1]) or {}).get("pos") or "").strip().lower() not in ("adverb", "adjective")
            and pos_norm not in ("verb", "v", "phrasal verb", "phrasal_verb", "adj", "adjective", "adv", "adverb")
            and (
                t_pos_top == "noun"
                or pos_norm in ("noun", "n")
                or (
                    words[0] not in _VERB_NP_HEADS
                    and bool(cls._wordnet_pos(words[1]) & {"n"})
                    and (
                        not bool(cls._wordnet_pos(words[0]) & {"v"})
                        or bool(cls._wordnet_pos(words[0]) & {"n", "a", "s"})
                    )
                )
            )
        )
        head_ent_top = cls.get_ldoce_entry(head_w) if head_w else None
        head_ldoce_pos = str((head_ent_top or {}).get("pos") or "").strip().lower()
        target_head_is_verb = (
            bool(head_w)
            and not is_compound_noun
            and head_w not in _NON_VERB_PHRASE_HEADS
            and not head_w.endswith("ed")
            and pos_norm not in ("adj", "adjective")
            and head_ldoce_pos != "adjective"
            and (
                bool(cls._wordnet_pos(head_w) & {"v"})
                or phrase_index.get(phrase_clean) == "phrasal_verb"
                or pos_norm in ("verb", "v", "phrasal verb", "phrasal_verb")
            )
        )
        target_head_is_prep = head_w in {
            "in", "on", "at", "by", "for", "with", "from", "to", "of", "out", "up", "down",
            "over", "under", "into", "onto", "upon", "about", "above", "below", "between",
            "among", "through", "during", "before", "after", "since", "until", "without",
            "within", "against", "along", "around", "behind", "beside", "beyond"
        }

        _PARTICLE_OR_DET_MIDDLES = {
            "a", "an", "the", "my", "your", "his", "her", "its", "our", "their", "some", "any", "no",
            "it", "them", "him", "me", "us", "you", "this", "that", "these", "those", "all", "what",
            "up", "down", "in", "out", "on", "off", "away", "back", "over", "under", "around", "along",
            "through", "into", "onto", "to", "for", "with", "from", "by", "at", "of", "about", "as",
            "got", "been", "done", "finished", "heard", "taken", "caught", "lost", "stuck", "lying", "too"
        }
        target_is_v_noun_prep = (
            n_words == 3
            and target_head_is_verb
            and words[1] not in _PARTICLE_OR_DET_MIDDLES
            and bool(cls._wordnet_pos(words[1]) & {"n"})
        )
        target_head_is_det = head_w in ("a", "an", "the")

        _norm_poss = cls.normalize_phrase_possessives
        _norm_det_poss = cls.normalize_phrase_determiners
        target_norm_poss = _norm_poss(phrase_clean)
        target_norm_det_poss = _norm_det_poss(phrase_clean)
        _BASE_VERBS_ENDING_IN_ING = {
            "bring", "sing", "ring", "swing", "spring", "cling",
            "string", "fling", "sting", "wring", "sling", "wing",
        }

        def _is_vbg_participle(w_tok: str) -> bool:
            wl = (w_tok or "").lower()
            return wl.endswith("ing") and len(wl) > 4 and wl not in _BASE_VERBS_ENDING_IN_ING

        _target_frame_syn_set: Set[str] = set()
        for _f_grp in cls._FRAME_PREP_SYNONYM_GROUPS:
            if phrase_clean in _f_grp or target_norm_poss in _f_grp:
                _target_frame_syn_set |= _f_grp

        target_phrase_def_text = cls._get_phrase_def_text(phrase_clean) if n_words == 2 else ""

        def ships(cand: str, allow_tail_overlap: bool = False, tail_len: int = 0) -> bool:
            if (cand == phrase_clean or _norm_poss(cand) == target_norm_poss
                    or (n_words >= 3 and _norm_det_poss(cand) == target_norm_det_poss)
                    or cand in _target_frame_syn_set
                    or cand in exclude or cand in synonym_set
                    or "." in cand or "[" in cand or "]" in cand or "(" in cand or ")" in cand or "/" in cand
                    or cls.is_inappropriate_phrase(cand)
                    or not (cand in phrase_index or cls.is_attested_phrase(cand))):
                return False
            # Front-loaded unified multi-word phrase double key collision gate
            if cls.double_key_collision(phrase_clean, cand, pos="phrase", quote=quote)[0]:
                return False
            # Pedagogical alphanumeric purity gate: reject numbers/digits unless target contains digits
            if any(ch.isdigit() for ch in cand) and not any(ch.isdigit() for ch in phrase_clean):
                return False
            # Reject malformed or uncontracted colloquial contractions
            if any(tok in {"cant", "wont", "dont", "didnt", "isnt", "arent", "wasnt", "werent"} for tok in cand.split()):
                return False
            c_words = cand.split()
            if not c_words:
                return False
            if any("'" in cw and cw != "one's" for cw in c_words) and not any("'" in tw and tw != "one's" for tw in words):
                return False
            _CLAUSE_SUBJ_HEADS = {
                "i", "you", "he", "she", "it", "we", "they", "there", "what", "whatever",
                "who", "whoever", "anything", "something", "nothing", "everything",
                "this", "that", "these", "those",
            }
            if c_head := c_words[0]:
                if c_head in _CLAUSE_SUBJ_HEADS and head_w not in _CLAUSE_SUBJ_HEADS:
                    return False
                if not target_head_is_det and c_head in ("a", "an", "the"):
                    return False
            if "one's" in words and not any(cw in ("your", "one's") for cw in c_words):
                return False
            c_head = c_words[0]
            if is_compound_noun:
                if len(c_words) == 2 and (c_words[1] in _PARTICLES_AND_PREPS or c_head in _VERB_NP_HEADS):
                    return False
                if not (cls._wordnet_pos(c_words[-1]) & {"n"}):
                    return False
            elif target_head_is_verb:
                # Active lexical/light verb phrases (e.g. 'make the most of', 'take pleasure in', 'play a part', 'fare better')
                # must only pair with active verb-headed phrases, never determiner/noun/prep fragments
                # ('no to speak of', 'a bad judge of'), nominal compounds ('walk-on part', 'fare dodger'),
                # or mismatched inflectional forms ('getting better' vs base 'fare better').
                if c_head in _NON_VERB_PHRASE_HEADS or not (cls._wordnet_pos(c_head) & {"v"}):
                    return False
                if _is_vbg_participle(c_head) != _is_vbg_participle(head_w):
                    return False
                if phrase_index.get(cand) != "phrasal_verb":
                    c_ent_full = cls.get_ldoce_entry(cand) or cls.get_ldoce_entry(cand.replace(" ", "-"))
                    if c_ent_full and str(c_ent_full.get("pos") or "").strip().lower() == "noun":
                        return False
                    if len(c_words) >= 3:
                        c_ent_pre = cls.get_ldoce_entry(f"{c_words[0]}-{c_words[1]}")
                        if c_ent_pre and str(c_ent_pre.get("pos") or "").strip().lower() in ("adjective", "noun"):
                            return False
                ch_entry = cls.get_ldoce_entry(c_head)
                ch_primary_pos = str((ch_entry or {}).get("pos") or "").lower()
                if any(bad_p in ch_primary_pos for bad_p in ("interjection", "exclamation", "adjective", "adverb", "preposition", "determiner")):
                    return False
                if n_words == 2 and len(c_words) == 2:
                    if words[1] in _PARTICLES_AND_PREPS and c_words[1] not in _PARTICLES_AND_PREPS:
                        return False
                    if c_words[1] in _PARTICLES_AND_PREPS:
                        _ditrans_only_heads = {
                            "attribute", "equip", "provide", "confine", "devote", "dedicate",
                            "expose", "subject", "owe", "accuse", "deprive", "remind",
                            "convince", "inform", "warn", "assure", "rob", "rid", "cure",
                        }
                        if c_head in _ditrans_only_heads and head_w not in _ditrans_only_heads:
                            return False
                _BE_FORMS = {"be", "am", "is", "are", "was", "were", "been", "being"}
                if c_head in _BE_FORMS and head_w not in _BE_FORMS:
                    return False
                if n_words == 3 and words[1:] == ["to", "do"] and c_words[1:] == ["to", "do"]:
                    if c_head in {
                        "connive", "deign", "depute", "detail", "discipline", "disdain",
                        "omit", "do", "make", "give", "take", "get", "put",
                    }:
                        return False
                if target_is_v_noun_prep and len(c_words) == 3:
                    # Preserve Verb + Bare-Noun + Prep structural symmetry: reject phrasal verb + prep
                    # ('bring out in', 'come on in', 'take up with') or Verb + Det + Noun ('take a chance')
                    if c_words[1] in _PARTICLE_OR_DET_MIDDLES or not (cls._wordnet_pos(c_words[1]) & {"n"}):
                        return False
                if n_words == 3 and words[1] in ("a", "an", "the"):
                    # Preserve Verb + Det + Noun structural symmetry (e.g. 'play a part'):
                    # reject phrasal-verb nominalizations ('walk on part') and synonymous light-verb variants ('have a part').
                    if len(c_words) != 3 or c_words[1] not in ("a", "an", "the", "one's", "your", "his", "her", "their", "some", "no"):
                        return False
                    if c_words[1:] == words[1:] and head_w in ("play", "have", "take", "make", "do", "give") and c_head in ("play", "have", "take", "make", "do", "give"):
                        return False
            elif target_head_is_det:
                if c_head not in ("a", "an", "the"):
                    return False
            elif target_head_is_prep:
                if c_head not in _PARTICLES_AND_PREPS:
                    return False
                if n_words == 3 and words[1] in ("a", "an", "the"):
                    if len(c_words) != 3 or c_words[1] not in ("a", "an", "the"):
                        return False
                    c2_ent = cls.get_ldoce_entry(c_words[2])
                    c2_pos = str((c2_ent or {}).get("pos") or "").strip().lower()
                    if c2_pos in ("adjective", "adverb", "determiner", "preposition", "conjunction") or not (cls._wordnet_pos(c_words[2]) & {"n"} or c2_pos == "noun"):
                        return False
                if n_words == 4 and words[1] in ("a", "an", "the"):
                    if len(c_words) != 4 or c_words[1] not in ("a", "an", "the"):
                        return False
                    if words[-1] not in _PARTICLES_AND_PREPS and c_words[-1] == words[-1]:
                        return False
                    # Preserve singular vs plural internal noun agreement in 'Prep + the + N + of' frames
                    # (e.g. block 'for the likes of' [plural] for 'for the sake of' [singular])
                    if words[2].endswith("s") != c_words[2].endswith("s"):
                        return False
            else:
                # Non-verbal, non-prep, non-det phrase (e.g. 'as long as', 'all over the world', 'no longer', 'hats off to'):
                # never pair with phrasal verbs, verb-headed phrases, or preposition-headed fragments.
                _is_binomial_target = n_words == 3 and words[1] in ("and", "or")
                if phrase_index.get(cand) == "phrasal_verb" or c_head in _VERB_NP_HEADS or (c_head in _PARTICLES_AND_PREPS and not _is_binomial_target):
                    return False
                if not head_w.endswith("ed") and (
                    any(w.endswith("ed") or w.endswith("ing") for w in c_words[:2])
                    or c_head in (
                        "paid", "held", "made", "kept", "left", "built", "born", "done", "gone", "seen", "known", "given", "taken"
                    )
                ):
                    return False
                _FUNCTION_STOP = _NON_VERB_PHRASE_HEADS | _PARTICLES_AND_PREPS | {"and", "or"}
                tw_check = words[:-tail_len] if (allow_tail_overlap and tail_len > 0 and len(words) > tail_len) else words
                cw_check = c_words[:-tail_len] if (allow_tail_overlap and tail_len > 0 and len(c_words) > tail_len) else c_words
                if any(
                    cls.are_same_word_family(tw, cw)
                    for tw in tw_check if len(tw) >= 3 and tw not in _FUNCTION_STOP
                    for cw in cw_check if len(cw) >= 3 and cw not in _FUNCTION_STOP
                ):
                    return False
                if c_head not in _NON_VERB_PHRASE_HEADS and not is_compound_noun and not _is_binomial_target:
                    ch_wn_pos = cls._wordnet_pos(c_head)
                    ch_ent = cls.get_ldoce_entry(c_head)
                    ch_ldoce_pos = str((ch_ent or {}).get("pos") or "").strip().lower()
                    if ("v" in ch_wn_pos and not (ch_wn_pos & {"n", "a", "s", "r"})) or ch_ldoce_pos == "verb":
                        return False

            return True

        tier_rank = {tier: index for index, tier in enumerate(cls._PHRASE_TIERS)}

        def strength(cand: str) -> int:
            """PedLex's own phrasal-verb block beats a PHRASES item, which beats a pattern."""
            idx_tier = phrase_index.get(cand)
            if idx_tier == "phrasal_verb":
                return 0
            hits = cls.ldoce_phrase_hits(cand)
            if hits:
                return min((tier_rank.get(tier, len(tier_rank)) for tier, _text in hits),
                           default=len(tier_rank))
            tier = "phrase" if idx_tier == "sense_unit" else (idx_tier or "example")
            return tier_rank.get(tier, len(tier_rank))

        # -------------------------------------------------------------
        # 0. Pronominal Separated Phrasal Verbs (e.g. 'shut it down', 'bring them in')
        # -------------------------------------------------------------
        if (
            n_words == 3
            and words[1] in ("it", "them", "him", "her", "me", "us")
            and words[2] in _PARTICLES_AND_PREPS
            and (cls._wordnet_pos(words[0]) & {"v"} or phrase_index.get(f"{words[0]} {words[2]}") == "phrasal_verb")
        ):
            base_pv = f"{words[0]} {words[2]}"
            base_dists = cls.generate_phrase_distractors(
                base_pv, pos="phrasal_verb", count=count * 2, exclude_words=exclude, definition=definition
            )
            pron_cands = [
                f"{d.split()[0]} {words[1]} {d.split()[1]}"
                for d in base_dists
                if len(d.split()) == 2 and d.split()[1] in _PARTICLES_AND_PREPS
            ]
            if len(pron_cands) >= count:
                return pron_cands[:count]

        # -------------------------------------------------------------
        # 0a. Unified Construction Families Phrase Contrast (`pedlex_cxn_index`)
        # (Article Minimal-Pair PPs, Fronted Negative Inversion Prepositional Phrases, Correlative Triggers)
        # -------------------------------------------------------------
        if phrase_clean in cls._CORREL_FIRST_ARMS:
            correl_cands, _ = cls.get_construction_contrast_distractors(
                phrase_clean, pos="phrase", target_count=count, exclude_words=exclude
            )
            if len(correl_cands) >= count:
                return correl_cands[:count]
            distractors.extend(correl_cands)
        else:
            cxn_p_cands, cxn_p_spec = cls.get_construction_contrast_distractors(
                phrase_clean, pos="phrase", target_count=count * 2, exclude_words=exclude
            )
            if cxn_p_spec and cxn_p_spec.get("subcat_group") in (
                "PP_BARE_ARTICLE", "PP_DEF_ARTICLE", "PHR_FRONTED_INV", "CORREL_PAIR"
            ):
                subcat_g = cxn_p_spec.get("subcat_group")
                filtered_cxn_p: List[str] = []
                # For PP_BARE_ARTICLE <-> PP_DEF_ARTICLE, place the exact article minimal-pair counterpart first
                if subcat_g == "PP_BARE_ARTICLE" and n_words >= 3:
                    counterpart = f"{words[0]} the {' '.join(words[1:])}"
                    if counterpart not in exclude and counterpart != phrase_clean:
                        filtered_cxn_p.append(counterpart)
                elif subcat_g == "PP_DEF_ARTICLE" and n_words >= 4 and words[1] == "the":
                    counterpart = f"{words[0]} {' '.join(words[2:])}"
                    if counterpart not in exclude and counterpart != phrase_clean:
                        filtered_cxn_p.append(counterpart)
                for cp in cxn_p_cands:
                    if cp in filtered_cxn_p or cp in exclude or cp == phrase_clean:
                        continue
                    if subcat_g in ("PP_BARE_ARTICLE", "PP_DEF_ARTICLE") and words[0] in ("in", "out", "on", "at", "by", "under", "for"):
                        if cp.split()[0] not in ("in", "out", "on", "at", "by", "under", "for"):
                            continue
                    filtered_cxn_p.append(cp)
                if len(filtered_cxn_p) >= count:
                    return filtered_cxn_p[:count]
                distractors.extend(filtered_cxn_p)

        # -------------------------------------------------------------
        # 0. Declarative Collocation & Prepositional Frames
        # (Verb + Prep, Adjective + Prep, Noun + Prep, 3/4-Part Prepositional Frames)
        # -------------------------------------------------------------
        try:
            from librarian.expression_patterns import (
                COLLOCATION_VERB_PREPS,
                ADJ_DEPENDENT_PREPS,
                NOUN_DEPENDENT_PREPS,
                FRAME_PREPOSITION_NOUNS,
            )
        except ImportError:
            COLLOCATION_VERB_PREPS = {}
            ADJ_DEPENDENT_PREPS = {}
            NOUN_DEPENDENT_PREPS = {}
            FRAME_PREPOSITION_NOUNS = set()

        if n_words == 2 and words[0] in COLLOCATION_VERB_PREPS and words[1] == COLLOCATION_VERB_PREPS[words[0]]:
            verb, prep = words[0], words[1]
            verb_cands: List[str] = []
            # Priority 1: Same preposition contrast (e.g. 'worry about' -> 'care about', 'complain about'; 'derive from' -> 'benefit from', 'suffer from', 'differ from')
            for v, p in COLLOCATION_VERB_PREPS.items():
                if p == prep and v != verb:
                    c = f"{v} {prep}"
                    if ships(c) and c not in verb_cands:
                        verb_cands.append(c)
            if len(verb_cands) >= count:
                return verb_cands[:count]
            distractors.extend(verb_cands)

        elif n_words == 2 and (
            words[0] in ADJ_DEPENDENT_PREPS
            or (
                words[1] in ("of", "with", "to", "for", "from", "in", "on", "at", "about")
                and not target_head_is_verb
                and not target_head_is_prep
                and not is_compound_noun
            )
        ):
            adj, prep = words[0], words[1]
            adj_cands: List[str] = []
            if adj in ("regardless", "instead", "because", "ahead", "outside", "inside", "irrespective", "devoid"):
                for cp_cand in ("instead of", "because of", "ahead of", "short of", "capable of", "aware of", "proud of"):
                    if cp_cand != phrase_clean and ships(cp_cand) and cp_cand not in adj_cands:
                        adj_cands.append(cp_cand)
            elif adj.endswith("ed") and prep == "with":
                for ed_cand in ("faced with", "equipped with", "blessed with", "obsessed with", "pleased with", "satisfied with", "bored with"):
                    if ed_cand != phrase_clean and ships(ed_cand) and ed_cand not in adj_cands:
                        adj_cands.append(ed_cand)
            # Priority 1: Same preposition contrast (e.g. 'aware of' -> 'conscious of', 'capable of', 'tired of')
            for a, p in ADJ_DEPENDENT_PREPS.items():
                if p == prep and a != adj:
                    c = f"{a} {prep}"
                    if ships(c) and c not in adj_cands:
                        adj_cands.append(c)
            # Priority 2: Cross-preposition adjective collocations
            for a, p in ADJ_DEPENDENT_PREPS.items():
                if a != adj and p != prep:
                    c = f"{a} {p}"
                    if ships(c) and c not in adj_cands:
                        adj_cands.append(c)
            if len(adj_cands) >= count:
                return adj_cands[:count]
            distractors.extend(adj_cands)

        elif n_words == 2 and words[0] in NOUN_DEPENDENT_PREPS:
            noun, prep = words[0], words[1]
            noun_cands: List[str] = []
            # Priority 1: Same preposition noun collocations (e.g. 'access to' -> 'approach to', 'attitude to', 'solution to')
            for n, p in NOUN_DEPENDENT_PREPS.items():
                if p == prep and n != noun:
                    c = f"{n} {prep}"
                    if ships(c) and c not in noun_cands:
                        noun_cands.append(c)
            # Priority 2: Cross-preposition noun collocations
            for n, p in NOUN_DEPENDENT_PREPS.items():
                if n != noun and p != prep:
                    c = f"{n} {p}"
                    if ships(c) and c not in noun_cands:
                        noun_cands.append(c)
            if len(noun_cands) >= count:
                return noun_cands[:count]
            distractors.extend(noun_cands)

        elif (n_words in (3, 4) and any(w in FRAME_PREPOSITION_NOUNS for w in words)
              and words[0] in ("in", "at", "on", "by", "for", "with", "under", "out")):
            frame_cands: List[str] = []
            canonical_frames = [
                "in terms of", "in front of", "in light of", "in view of", "in spite of",
                "in addition to", "in response to", "on behalf of", "in charge of",
                "in need of", "in search of", "in favor of", "in order to",
                # 4-word prepositional frames (e.g. 'at the expense of')
                "at the expense of", "for the sake of", "in the event of", "at the mercy of",
                "at the hands of", "for the good of", "by means of"
            ]
            # Prioritize frames sharing the initial preposition or final preposition
            for f in canonical_frames:
                if f != phrase_clean and len(f.split()) == n_words:
                    fw = f.split()
                    if fw[0] == words[0] or fw[-1] == words[-1]:
                        if ships(f) and f not in frame_cands:
                            frame_cands.append(f)
            for f in canonical_frames:
                if f != phrase_clean and len(f.split()) == n_words and ships(f) and f not in frame_cands:
                    frame_cands.append(f)
            if len(frame_cands) >= count:
                return frame_cands[:count]
            distractors.extend(frame_cands)

        # -------------------------------------------------------------
        # Quantifier / Partitive frame (e.g. 'a piece of', 'a bit of', 'a slice of')
        # -------------------------------------------------------------
        elif n_words == 3 and words[0] in ("a", "an") and words[2] == "of":
            head_n = words[1]
            q_siblings = [
                "piece", "bit", "slice", "sheet", "block", "drop",
                "pair", "pack", "stroke", "portion", "set", "share"
            ]
            quant_cands: List[str] = []
            for qs in q_siblings:
                if qs != head_n:
                    art = "an" if qs[0] in "aeiou" else "a"
                    cand_frame = f"{art} {qs} of"
                    if ships(cand_frame) and cand_frame not in quant_cands:
                        quant_cands.append(cand_frame)
            if len(quant_cands) >= count:
                return quant_cands[:count]
            distractors.extend(quant_cands)

        # -------------------------------------------------------------
        # 0b. Two-word Compound Nouns & Nominal Collocations (e.g. 'jet lag', 'culture shock', 'rush hour', 'high praise', 'passenger plane')
        # -------------------------------------------------------------
        if n_words == 2 and not distractors and is_compound_noun:
            t_entry = t_entry_top
            if not t_entry and len(words) >= 2:
                t_entry = cls.get_ldoce_entry(words[1])
            t_senses = (t_entry or {}).get("senses", [])
            if t_entry and (t_entry.get("word") == words[1] or t_entry.get("payload_headword") == words[1]):
                t_senses = [s for s in t_senses if s.get("pos") == "noun"] or t_senses
            t_gram = " ".join(str(s.get("grammar", "")) for s in t_senses).lower()
            t_is_uncountable = "uncountable" in t_gram and "countable" not in t_gram.replace("uncountable", "")
            comp_cands: List[str] = []

            # Priority 0: Antonym / Opposite Collocations (e.g. 'high praise' -> opposite noun 'criticism' -> 'strong criticism')
            if len(words) >= 2 and words[1]:
                opp_nouns = set()
                w1_entry = cls.get_ldoce_entry(words[1])
                for s in (w1_entry or {}).get("senses", []):
                    opp_w = s.get("opposite")
                    if opp_w and isinstance(opp_w, str):
                        clean_opp = opp_w.strip().lower()
                        if clean_opp in phrase_index or cls.get_ldoce_entry(clean_opp):
                            opp_nouns.add(clean_opp)
                for opp in opp_nouns:
                    opp_entry = cls.get_ldoce_entry(opp)
                    for s in (opp_entry or {}).get("senses", []):
                        for pb in s.get("pattern_bundles", []):
                            pat = pb.get("pattern", "").strip().lower()
                            pat_w = pat.split()
                            if len(pat_w) == 2 and pat not in exclude and pat not in synonym_set:
                                if pat in phrase_index or cls.is_attested_phrase(pat):
                                    if pat not in comp_cands and ships(pat):
                                        comp_cands.append(pat)
                                        if len(comp_cands) >= count:
                                            break
                        if len(comp_cands) >= count:
                            break
                    if len(comp_cands) >= count:
                        break
            if len(comp_cands) >= count:
                return comp_cands[:count]

            # Priority 1: Slash-expanded and head/modifier-sharing 2-word nominal collocations in phrase_index
            # (e.g., 'passenger train/plane/ship' -> 'passenger train', 'passenger ship'; '* plane' -> 'fighter plane')
            # Guard: When the modifier `words[0]` is a generic high-frequency adjective ('new', 'old', 'good', 'big', etc.),
            # pairing `new + <random noun>` ('new day', 'new life', 'new man') or `<adj> + generation` ('younger generation',
            # 'future generation') produces non-compound phrases or context-overlapping double keys; fall through to
            # canonical 2-word compound nouns instead.
            _GENERIC_COMPOUND_ADJ_MODS = {
                "new", "old", "good", "bad", "big", "small", "great", "high", "low",
                "long", "short", "young", "early", "late", "modern", "ancient", "different",
                "special", "common", "general", "whole", "main", "major", "minor",
            }
            _PARTICIPLE_SUFFIX_STOP = {"formed", "laid", "made", "improved", "born", "built", "grown", "found", "known"}
            same_mod_cands: List[str] = []
            same_head_cands: List[str] = []
            if words[0] not in _GENERIC_COMPOUND_ADJ_MODS:
                for k in phrase_index:
                    kw = k.split()
                    if len(kw) != 2:
                        continue
                    # Expand slash-grouped 2-word collocations (e.g. 'passenger train/plane/ship')
                    if "/" in kw[1] and "/" not in kw[0] and kw[0] == words[0]:
                        w1_syns = cls.get_synonyms(words[1])
                        for alt_n in kw[1].split("/"):
                            alt_n = alt_n.strip()
                            if alt_n and alt_n != words[1] and alt_n.isalpha() and alt_n not in w1_syns:
                                cand_alt = f"{kw[0]} {alt_n}"
                                if cand_alt not in exclude and cand_alt not in synonym_set and not cls.is_inappropriate_phrase(cand_alt):
                                    if cand_alt not in same_mod_cands:
                                        same_mod_cands.append(cand_alt)
                        continue
                    if "/" in k:
                        continue
                    if kw[1] in _PARTICLES_AND_PREPS or kw[0] in _PARTICLES_AND_PREPS or kw[0] in _NON_VERB_PHRASE_HEADS or kw[0] in _VERB_NP_HEADS:
                        continue
                    if kw[0] == words[0] and kw[1] != words[1]:
                        if kw[1] in _PARTICIPLE_SUFFIX_STOP or kw[1].endswith("ed"):
                            continue
                        w1_pos = cls._wordnet_pos(kw[1])
                        if ("n" in w1_pos) and not cls.double_key_collision(words[1], kw[1], pos="noun")[0]:
                            if k not in exclude and k not in synonym_set and not cls.is_inappropriate_phrase(k) and k not in same_mod_cands:
                                same_mod_cands.append(k)
                    elif kw[1] == words[1] and kw[0] != words[0]:
                        w0_pos = cls._wordnet_pos(kw[0])
                        if (w0_pos & {"n", "a", "s"}) and not cls.double_key_collision(words[0], kw[0])[0]:
                            if k not in exclude and k not in synonym_set and not cls.is_inappropriate_phrase(k) and k not in same_head_cands:
                                same_head_cands.append(k)

            same_mod_cands.sort(
                key=lambda c: (
                    0 if c not in phrase_index else 1,
                    cls.CEFR_ORDER.get(cls.get_word_cefr(c.split()[1]) or "B2", 4),
                    strength(c),
                    c,
                )
            )
            same_head_cands.sort(key=lambda c: (strength(c), c))
            for sc in same_mod_cands + same_head_cands:
                if sc not in comp_cands:
                    comp_cands.append(sc)
                if len(comp_cands) >= count:
                    return comp_cands[:count]

            # Priority 2: Canonical countable/uncountable 2-word compound nouns
            if t_is_uncountable:
                canonical_compounds = [
                    "strong criticism", "common sense", "peer pressure", "culture shock",
                    "stage fright", "food poisoning", "blood pressure", "junk food",
                    "global warming", "climate change", "heart failure", "sleep deprivation"
                ]
            else:
                canonical_compounds = [
                    "rush hour", "heart attack", "side effect", "turning point",
                    "traffic jam", "brain drain", "wake-up call", "blind spot",
                    "comfort zone", "safety net", "generation gap", "role model"
                ]
            for cc in canonical_compounds:
                if cc != phrase_clean and cc not in exclude and cc not in synonym_set and not cls.is_inappropriate_phrase(cc):
                    if cc in phrase_index or cls.get_ldoce_entry(cc) is not None or cls.is_attested_phrase(cc):
                        if cc not in comp_cands:
                            comp_cands.append(cc)
            if len(comp_cands) >= count:
                return comp_cands[:count]
            distractors.extend(comp_cands)

        # -------------------------------------------------------------
        # 0c. Two-word Prepositional State Phrases (e.g. 'at ease', 'at risk', 'on guard', 'in danger')
        # -------------------------------------------------------------
        if n_words == 2 and not distractors and target_head_is_prep and words[1] not in _PARTICLES_AND_PREPS:
            canonical_prep_states = [
                "on guard", "at risk", "in danger", "on duty", "under pressure",
                "by chance", "on purpose", "in vain", "at fault", "in charge",
                "on board", "at peace"
            ]
            prep_state_cands: List[str] = []
            for ps in canonical_prep_states:
                ps_w = ps.split()
                if ps != phrase_clean and ps not in exclude and ps not in synonym_set and len(ps_w) == 2:
                    if not cls.double_key_collision(words[1], ps_w[1], pos="noun")[0]:
                        if ps in phrase_index or cls.is_attested_phrase(ps):
                            prep_state_cands.append(ps)
            if len(prep_state_cands) >= count:
                return prep_state_cands[:count]
            distractors.extend(prep_state_cands)

        # -------------------------------------------------------------
        # 1. Two-word phrase (e.g. 'send out', 'tend to', 'cut out', 'work for')
        # -------------------------------------------------------------
        if n_words == 2 and len(distractors) < count:
            v, p = words[0], words[1]
            diff_part: List[str] = []
            _adverbial_particles = ["in", "off", "up", "down", "away", "back", "over", "through", "on", "out", "around", "along", "ahead", "apart", "aside", "together"]
            _relational_preps = {"to", "with", "into", "from", "of", "upon", "for", "at", "by", "about", "as", "against", "between", "among", "towards", "toward"}
            particles = ["in", "off", "up", "down", "away", "back", "over", "into", "through", "on", "to", "for", "out", "with", "from", "around", "about", "of"]
            p_is_particle = p in particles
            p_is_adverbial_particle = p in _adverbial_particles and p not in _relational_preps
            v_is_verb = target_head_is_verb or bool(cls._wordnet_pos(v) & {"v"})

            if p_is_particle:
                # Only vary the particle on the same verb when `p` and `part` are genuine adverbial particles
                # ('break up' -> 'break down'), NEVER when `p` or `part` is a relational preposition
                # ('equate to' -x-> 'equate with', 'compare to' -x-> 'compare with', 'integrate into' -x-> 'integrate with').
                if p_is_adverbial_particle:
                    for part in _adverbial_particles:
                        if part != p:
                            cand = f"{v} {part}"
                            if ships(cand) and cand not in diff_part:
                                diff_part.append(cand)

                    # PedLex's own phrasal-verb list for this verb is the strongest pool there is
                    for cand in cls.get_ldoce_phrasal_verbs(v):
                        parts = cand.split()
                        if len(parts) == 2 and parts[1] != p and parts[1] in _adverbial_particles and ships(cand) and cand not in diff_part:
                            diff_part.insert(0, cand)

                same_part: List[str] = []
                is_transitive_pv = bool(
                    re.search(r"\[(?:sb|sth|sb/sth|sth/sb|somebody|something)\]|\b(?:somebody|something|someone)\b", (phrase or "").lower())
                    or (
                        definition
                        and (
                            re.search(
                                r"\bto\s+[a-z]+\s+(?:something|someone|somebody|a\s+|an\s+|the\s+|your\s+|all\s+|people|animals|buildings|doors|windows|space|room|house|shop)\b",
                                definition.lower(),
                            )
                            or re.search(r"\bif\s+[^,]+?\s+(?:is|are|was|were)\s+[a-z]+ed\b", definition.lower())
                        )
                    )
                )
                _pure_intrans_pv_heads = {
                    "come", "go", "fall", "sit", "stand", "lie", "sleep", "arrive", "die",
                    "fade", "grow", "look", "listen", "wait", "stay", "remain", "live",
                    "work", "run", "walk", "jump", "step", "show", "wake", "get",
                }
                _obscure_pv_heads = {"ante", "fess", "pony", "cough", "chat", " shack", "dish", "jazz", "ham", "man"}
                common_verbs = (
                    [
                        "turn", "set", "take", "put", "pick", "hold", "keep", "give",
                        "make", "bring", "break", "cut", "shut", "pull", "clear", "open",
                        "build", "write", "hand", "pass", "carry", "find", "leave", "call", "move",
                    ]
                    if is_transitive_pv
                    else [
                        "point", "carry", "turn", "figure", "bring", "stand", "make", "find",
                        "set", "take", "come", "go", "get", "give", "hold", "look", "run",
                        "keep", "leave", "work", "fall", "call", "pass", "lead", "move", "break",
                    ]
                )
                common_verbs_rank = {cv_w: idx_cv for idx_cv, cv_w in enumerate(common_verbs)}
                for cv in common_verbs:
                    if cv != v:
                        cand = f"{cv} {p}"
                        if ships(cand) and cand not in diff_part and cand not in same_part:
                            same_part.append(cand)

                # Query phrase_index for 2-word phrasal verbs ending with particle `p`
                p_cands = [
                    k for k in phrase_index
                    if " " in k and len(k.split()) == 2 and k.split()[1] == p and k.split()[0] != v
                ]
                p_cands.sort(key=lambda k: (0 if phrase_index.get(k) == "phrasal_verb" else 1, k))
                tier0_count = sum(1 for c in same_part if strength(c) == 0)
                for k in p_cands:
                    if tier0_count >= count:
                        break
                    cand_head = k.split()[0]
                    if cand_head in _obscure_pv_heads or (is_transitive_pv and cand_head in _pure_intrans_pv_heads):
                        continue
                    # Enforce head syntactic concord: if target head is a verb, cand head must be a primary verb
                    # (not an exclamation/interjection/adjective greeting frame like 'welcome to')
                    if v_is_verb:
                        if not (cls._wordnet_pos(cand_head) & {"v"}):
                            continue
                        ch_entry = cls.get_ldoce_entry(cand_head)
                        ch_primary_pos = str((ch_entry or {}).get("pos") or "").lower()
                        if any(bad_p in ch_primary_pos for bad_p in ("interjection", "exclamation", "adjective", "adverb", "preposition", "determiner")):
                            continue
                    if k not in diff_part and k not in same_part and ships(k):
                        same_part.append(k)
                        if strength(k) == 0:
                            tier0_count += 1
                            if tier0_count >= count:
                                break

                diff_part.sort(key=lambda c: (strength(c), c))
                same_part.sort(
                    key=lambda c: (
                        strength(c),
                        0 if c.split()[0] in common_verbs_rank else 1,
                        common_verbs_rank.get(c.split()[0], 999),
                        c,
                    )
                )

                # Literal Physical Verb Sense Guard:
                # When the phrasal verb's definition explicitly names the base physical instrument/action `v`
                # (e.g. 'nail down' -> 'to fasten something to something else with nails'), same-verb particle
                # siblings ('nail in', 'nail on', 'nail up') share the identical physical action and risk physical
                # double-keys; prioritize verb variation (`same_part`: 'hold down', 'bring down', 'turn down').
                is_literal_base_verb_def = bool(
                    definition and re.search(rf"\b{re.escape(v)}(?:s|es|ed|ing)?\b", definition.lower())
                )
                if is_literal_base_verb_def and len(same_part) >= count:
                    new_cands = same_part
                elif diff_part and same_part:
                    if is_literal_base_verb_def:
                        combined = list(same_part)
                        for item in diff_part:
                            if item not in combined:
                                combined.append(item)
                    else:
                        combined = [diff_part[0], same_part[0]]
                        if len(diff_part) > 1:
                            combined.append(diff_part[1])
                        elif len(same_part) > 1:
                            combined.append(same_part[1])
                        for item in diff_part[2:] + same_part[2:]:
                            if item not in combined:
                                combined.append(item)
                    new_cands = combined
                elif diff_part:
                    new_cands = diff_part
                elif same_part:
                    new_cands = same_part
                else:
                    new_cands = []
                for item in new_cands:
                    if item not in distractors:
                        distractors.append(item)
                    if len(distractors) >= count:
                        break
            else:
                # Verb + Complement/Adjective/Adverb Collocation (e.g. 'make possible', 'fare better', 'go wrong')
                p_is_adj = bool(cls._wordnet_pos(p) & {"a", "s"})
                p_is_adv = p in _ADV_OR_ADJ_COMPLEMENTS or bool(cls._wordnet_pos(p) & {"r"})
                diff_comp: List[str] = []
                for k in phrase_index:
                    kw = k.split()
                    if len(kw) == 2 and kw[0] == v and kw[1] != p and kw[1] not in particles:
                        if p_is_adv and str((cls.get_ldoce_entry(kw[1]) or {}).get("pos") or "").strip().lower() == "noun":
                            continue
                        if p_is_adj:
                            if not (cls._wordnet_pos(kw[1]) & {"a", "s"}):
                                continue
                        if ships(k) and k not in diff_comp:
                            diff_comp.append(k)
                if p_is_adv and len(diff_comp) < count:
                    for k in phrase_index:
                        kw = k.split()
                        if len(kw) == 2 and kw[1] == p and kw[0] != v:
                            if v == "fare" and kw[0] in ("do", "get"):
                                continue
                            if ships(k) and k not in diff_comp:
                                diff_comp.append(k)
                    for _adv_col in ("know better", "think better", "go wrong", "fall short", "come first", "work hard", "hold fast", "stand firm"):
                        if _adv_col != phrase_clean and ships(_adv_col) and _adv_col not in diff_comp:
                            diff_comp.append(_adv_col)
                if diff_comp:
                    diff_comp.sort(key=lambda c: (strength(c), c))
                    for item in diff_comp:
                        if item not in distractors:
                            distractors.append(item)
                        if len(distractors) >= count:
                            break

        # -------------------------------------------------------------
        # 2. Multi-word collocations & phrases (3+ words)
        # -------------------------------------------------------------
        elif n_words >= 3 and len(distractors) < count:
            cands_found: List[str] = []

            # Strategy A0: Correlative / Equative `as <adj/adv> as` Phrases (e.g. 'as long as', 'as far as', 'as well as')
            if n_words == 3 and words[0] == "as" and words[2] == "as":
                popular_as_as = [
                    "as far as", "as well as", "as good as", "as much as",
                    "as fast as", "as soon as", "as early as", "as late as",
                ]
                for k in popular_as_as:
                    if k != phrase_clean and ships(k) and k not in cands_found:
                        # Avoid pairing 'as long as' with 'as soon as' first when other `as ... as` connectors exist,
                        # since both can introduce future-time/conditional subordinate clauses.
                        if phrase_clean == "as long as" and k == "as soon as" and len(cands_found) >= count:
                            continue
                        cands_found.append(k)
                for k in phrase_index:
                    kw = k.split()
                    if len(kw) == 3 and kw[0] == "as" and kw[2] == "as" and ships(k) and k not in cands_found:
                        cands_found.append(k)

            # Strategy A0b: 3-word Verbless Evaluative / Interjection Idioms ending in 'to'
            # (e.g. 'hats off to', 'good luck to', 'good riddance to', 'no thanks to')
            if (
                n_words == 3
                and not target_head_is_verb
                and not target_head_is_prep
                and not target_head_is_det
                and words[2] == "to"
            ):
                for _ev_to in ("good luck to", "good riddance to", "no thanks to", "hats off to"):
                    if _ev_to != phrase_clean and ships(_ev_to) and _ev_to not in cands_found:
                        cands_found.append(_ev_to)

            # Strategy A: Binomial Symmetric Match (e.g. 'pros and cons', 'up and down')
            if n_words == 3 and words[1] in ("and", "or"):
                op = words[1]
                popular_binomials = [
                    "bits and pieces", "back and forth", "give and take",
                    "checks and balances", "ups and downs", "black and white",
                    "born and bred", "by and large", "again and again"
                ]
                for k in popular_binomials:
                    if ships(k) and k not in cands_found:
                        cands_found.append(k)
                for k in phrase_index:
                    kw = k.split()
                    if len(kw) == 3 and kw[1] == op and ships(k) and k not in cands_found:
                        cands_found.append(k)

            # Strategy A2: Possessive Idiom Structural Symmetry (`one's` / `your` at matching index)
            if "one's" in words and len(cands_found) < count:
                poss_idx = words.index("one's")
                _RELATIONAL_PREPS = {
                    "to", "on", "of", "from", "into", "for", "with", "at", "against", "about", "upon", "over"
                }
                _ADV_PARTICLES = {
                    "up", "down", "out", "off", "away", "back", "in", "along", "around", "through", "aside", "apart"
                }
                _UNDIGNIFIED_POSS_IDIOMS = {
                    _norm_poss(s)
                    for s in (
                        "pick your nose", "pick your teeth", "blow your nose", "get off your ass",
                        "get off your butt", "contemplate your navel", "lose your virginity",
                        "be your woman", "cry into your beer", "wet your pants", "soil your hands",
                        "have to your credit", "look to your laurels", "do your damnedest",
                    )
                }
                _POSS_SYNONYM_CLUSTERS = [
                    {_norm_poss(s) for s in ("do your best", "do your utmost", "do your part", "do your bit", "try your best")},
                    {_norm_poss(s) for s in ("turn your attention to", "turn your mind to", "direct your attention to", "give your attention to")},
                    {_norm_poss(s) for s in ("come to your aid", "come to your rescue", "come to your assistance", "go to your aid")},
                    {_norm_poss(s) for s in ("come into your mind", "cross your mind", "enter your mind", "pop into your head", "come into your head")},
                    {_norm_poss(s) for s in ("catch your eye", "catch your attention", "attract your attention", "get your attention")},
                    {_norm_poss(s) for s in ("keep your word", "keep your promise")},
                ]
                _target_poss_syns: Set[str] = set()
                for _p_grp in _POSS_SYNONYM_CLUSTERS:
                    if target_norm_poss in _p_grp:
                        _target_poss_syns |= _p_grp
                poss_cands: List[str] = []
                for k in phrase_index:
                    kw = k.split()
                    if len(kw) != n_words or kw[poss_idx] not in ("your", "one's"):
                        continue
                    if target_head_is_det and kw[0] not in ("a", "an"):
                        continue
                    if not target_head_is_det and kw[0] in ("a", "an", "the"):
                        continue
                    norm_k = _norm_poss(k)
                    if norm_k in _UNDIGNIFIED_POSS_IDIOMS or norm_k in _target_poss_syns:
                        continue
                    if words[-1] in _RELATIONAL_PREPS:
                        if kw[-1] not in _RELATIONAL_PREPS:
                            continue
                    elif words[-1] in _ADV_PARTICLES:
                        if kw[-1] not in _ADV_PARTICLES:
                            continue
                    else:
                        if kw[-1] in (_RELATIONAL_PREPS | _ADV_PARTICLES):
                            continue
                        # Reject truncated 3-word prefixes of 4-word phrasal idioms (e.g. 'keep your hands [off]', 'let your guard [down]')
                        if phrase_index.get(k) == "phrasal_verb" or (
                            phrase_index.get(k) not in ("sense_unit", "phrase_row")
                            and any(f"{k} {p}" in phrase_index for p in (_RELATIONAL_PREPS | _ADV_PARTICLES))
                        ):
                            continue
                    if ships(k):
                        poss_cands.append(k)
                if poss_cands:
                    poss_cands.sort(
                        key=lambda c: (
                            0 if phrase_index.get(c) in ("sense_unit", "phrase_row") else 1,
                            0 if c.split()[0] == words[0] else (1 if (poss_idx >= 1 and c.split()[poss_idx - 1] == words[poss_idx - 1]) else 2),
                            strength(c),
                            c,
                        )
                    )
                    seen_poss_pref = {_norm_poss(" ".join(words[:-1]))}
                    seen_poss_tail = {_norm_poss(" ".join(words[1:]))}
                    for pc in poss_cands:
                        pc_w = pc.split()
                        p_pref = _norm_poss(" ".join(pc_w[:-1]))
                        p_tail = _norm_poss(" ".join(pc_w[1:]))
                        if p_pref in seen_poss_pref or p_tail in seen_poss_tail:
                            continue
                        seen_poss_pref.add(p_pref)
                        seen_poss_tail.add(p_tail)
                        cands_found.append(pc)
                        if len(cands_found) >= count:
                            break
                    if len(cands_found) < count:
                        for pc in poss_cands:
                            if pc not in cands_found:
                                p_tail = _norm_poss(" ".join(pc.split()[1:]))
                                if p_tail not in seen_poss_tail:
                                    seen_poss_tail.add(p_tail)
                                    cands_found.append(pc)
                                    if len(cands_found) >= count:
                                        break

            # Strategy B: Common Suffix / Tail Match (e.g. 'peace of mind' -> '... of mind')
            # Guard: When words[0] is a light/support verb (take, find, have, get, make, give, keep, feel, etc.)
            # and tl == n_words - 1 (matching the exact noun+prep tail, e.g. '... pleasure in' for 'take pleasure in'
            # -> 'find pleasure in'), swapping only the light verb creates synonymous collocation double-keys!
            light_support_verbs = {
                "take", "find", "have", "get", "make", "give", "keep", "hold", "bear",
                "pay", "show", "feel", "put", "set", "bring", "come", "go", "lose", "catch"
            }
            is_v_it_adj_target = (
                n_words == 3
                and words[0] in ("make", "find", "keep", "leave")
                and words[1] == "it"
                and bool(cls._wordnet_pos(words[2]) & {"a", "s"})
            )
            if (not cands_found or len(cands_found) < count) and not is_v_it_adj_target:
                tail_lengths = [2] if target_is_v_noun_prep else ([2, 1] if n_words >= 3 else [1])
                for tl in tail_lengths:
                    if words[0] in light_support_verbs and tl >= n_words - 1:
                        continue
                    tail_phrase = " " + " ".join(words[-tl:])
                    norm_tail_phrase = " " + _norm_poss(" ".join(words[-tl:]))
                    # Look for exact length matches with the same suffix (including possessive-normalized suffix)
                    tail_cands = []
                    for k in phrase_index:
                        kw = k.split()
                        if (
                            len(kw) == n_words
                            and (k.endswith(tail_phrase) or _norm_poss(k).endswith(norm_tail_phrase))
                            and ships(k, allow_tail_overlap=True, tail_len=tl)
                            and k not in cands_found
                        ):
                            tail_cands.append(k)
                    if tail_cands:
                        # Prioritize by authentic lexical strength
                        tail_cands.sort(key=lambda c: (strength(c), c))
                        cands_found.extend(tail_cands)
                        if len(cands_found) >= count:
                            break

            # Strategy C: Head-Bound Prepositional Transitive Match
            # (e.g. 'keep in touch with' -> 'keep a lid on', 'keep close tabs on', 'keep an eye on')
            if len(cands_found) < count:
                particles_set = {"with", "to", "for", "from", "on", "in", "at", "into", "against", "out of", "about", "of"}
                if words[-1] in particles_set:
                    head_cands = []
                    for k in phrase_index:
                        kw = k.split()
                        if len(kw) == n_words and kw[0] == words[0] and kw[-1] in particles_set and ships(k) and k not in cands_found:
                            head_cands.append(k)
                    if head_cands:
                        head_cands.sort(key=lambda c: (strength(c), c))
                        cands_found.extend(head_cands)

            # Strategy D: Light-Verb & Construct Prefix Match (e.g. 'have a try', 'take a look', 'make a move', 'make it possible')
            if len(cands_found) < count:
                prefix_2 = " ".join(words[:2]) + " "
                prefix_cands = []
                is_v_it_adj = (
                    n_words == 3
                    and words[0] in ("make", "find", "keep", "leave")
                    and words[1] == "it"
                    and bool(cls._wordnet_pos(words[2]) & {"a", "s"})
                )
                inf_taking_adjs = {
                    "possible", "impossible", "easy", "easier", "hard", "harder",
                    "difficult", "simple", "simpler", "safe", "safer", "necessary", "essential",
                }
                if is_v_it_adj:
                    # For extraposition '(make|find|keep|leave) it <adj>' (e.g. 'make it possible ... to do [sth]', 'find it hard ... to do [sth]'),
                    # prioritize attested '<verb> it <adj/participle>' constructions that take that-clauses or NP complements
                    # rather than '(for sb) to do sth' to guarantee zero double-key collision.
                    if words[0] == "make":
                        curated_v_it = ["make it clear", "make it known", "make it certain", "make it plain", "make it good"]
                    elif words[0] == "find":
                        curated_v_it = ["find it strange", "find it odd", "find it clear", "find it funny", "find it weird"]
                    else:
                        curated_v_it = [f"{words[0]} it clear", f"{words[0]} it plain", f"{words[0]} it secret", f"{words[0]} it quiet"]
                    for ph in curated_v_it:
                        if ph != phrase_clean and ph not in exclude and ph not in synonym_set and ph not in cands_found and ph not in prefix_cands:
                            if words[0] != "make" or ships(ph):
                                prefix_cands.append(ph)
                popular_have_a = ["have a go", "have a care", "have a heart", "have a think", "have a word", "have a ball", "have a breather"]
                for ph in popular_have_a:
                    if prefix_2.strip() == "have a" and ships(ph) and ph not in cands_found and ph not in prefix_cands:
                        prefix_cands.append(ph)
                popular_in_the = ["in the end", "in the dark", "in the way", "in the mood", "in the open", "in the wild", "in the shade", "in the lead"]
                for ph in popular_in_the:
                    if prefix_2.strip() == "in the" and n_words == 3 and ships(ph) and ph not in cands_found and ph not in prefix_cands:
                        prefix_cands.append(ph)
                popular_on_the = ["on the whole", "on the spot", "on the surface", "on the rise", "on the way", "on the move", "on the side"]
                for ph in popular_on_the:
                    if prefix_2.strip() == "on the" and n_words == 3 and ships(ph) and ph not in cands_found and ph not in prefix_cands:
                        prefix_cands.append(ph)
                popular_at_a = ["at a loss", "at a glance", "at a time", "at a distance", "at a standstill", "at a price", "in a hurry", "on a diet"]
                for ph in popular_at_a:
                    if prefix_2.strip() in ("at a", "at an") and n_words == 3 and ships(ph) and ph not in cands_found and ph not in prefix_cands:
                        prefix_cands.append(ph)
                for k in phrase_index:
                    kw = k.split()
                    if len(kw) == n_words and k.startswith(prefix_2) and ships(k) and k not in cands_found and k not in prefix_cands:
                        if prefix_2.strip() in ("at a", "at an") and kw[2] in ("meeting", "party", "conference", "hotel", "table", "school", "hospital", "station", "restaurant"):
                            continue
                        if is_v_it_adj:
                            if kw[2] in cls._PHRASE_PARTICLE_WORDS or kw[2] in {"snappy"}:
                                continue
                            if words[2] in inf_taking_adjs and kw[2] in inf_taking_adjs:
                                continue
                        prefix_cands.append(k)
                if prefix_cands:
                    if not is_v_it_adj and prefix_2.strip() not in ("in the", "on the", "at a", "at an"):
                        prefix_cands.sort(key=lambda c: (cls.CEFR_ORDER.get(cls.get_word_cefr(c.split()[-1]) or "B2", 4), strength(c), c))
                    cands_found.extend(prefix_cands)

            # Strategy E: Structural Sister Frames (Prepositional / Nominal frames)
            if len(cands_found) < count:
                if "end of" in phrase_clean:
                    for c in ["at the beginning of", "in the middle of", "for the rest of", "at the start of"]:
                        if ships(c) and c not in cands_found:
                            cands_found.append(c)
                elif "piece of" in phrase_clean:
                    for c in ["a series of", "a range of", "a couple of", "a matter of", "a variety of"]:
                        if ships(c) and c not in cands_found:
                            cands_found.append(c)
                elif "thanks to" in phrase_clean:
                    for c in ["show respect to", "pay tribute to", "give credit to", "express gratitude to"]:
                        if ships(c) and c not in cands_found:
                            cands_found.append(c)
                elif "type of" in phrase_clean:
                    for c in ["any form of", "any sort of", "any kind of", "any part of"]:
                        if ships(c) and c not in cands_found:
                            cands_found.append(c)
                elif phrase_clean == "all over the world" or (n_words == 4 and words[:2] == ["all", "over"]):
                    for c in ["all over the place", "in the long run", "from time to time", "for the time being"]:
                        if c != phrase_clean and (ships(c) or c in phrase_index) and c not in cands_found:
                            cands_found.append(c)
                elif target_head_is_prep and n_words == 3 and words[1] in ("a", "an"):
                    for c in ["at a time", "in a hurry", "on a diet", "at a loss", "in a row", "on a roll", "at a glance", "in a mess", "on a whim"]:
                        if ships(c) and c not in cands_found:
                            cands_found.append(c)
                elif target_head_is_prep and n_words == 4 and words[1] in ("a", "an"):
                    for c in ["on a regular basis", "at a rapid pace", "on a tight budget", "in a big way", "on a daily basis", "to a certain extent", "at a steady rate"]:
                        if ships(c) and c not in cands_found:
                            cands_found.append(c)

            # Strategy F: Same-Length Syntactic Frame Matching (Same connector/prep at same index)
            if len(cands_found) < count:
                seen_f_prefixes = {" ".join(c.split()[:-1]) for c in cands_found if len(c.split()) >= 3}
                for idx, w in enumerate(words):
                    if w in ("of", "in", "to", "for", "with", "on", "at", "from", "over", "under", "into", "through"):
                        for k in phrase_index:
                            kw = k.split()
                            if len(kw) == n_words and kw[idx] == w and ships(k) and k not in cands_found:
                                k_pref = " ".join(kw[:-1]) if len(kw) >= 3 else k
                                if k_pref in seen_f_prefixes:
                                    continue
                                seen_f_prefixes.add(k_pref)
                                cands_found.append(k)
                                if len(cands_found) >= count + 6:
                                    break
                        if len(cands_found) >= count + 3:
                            break

            for item in cands_found:
                if item not in distractors:
                    distractors.append(item)
                if len(distractors) >= count:
                    break

        # -------------------------------------------------------------
        # 3. General Fallback if pool is still sparse
        # -------------------------------------------------------------
        if len(distractors) < count:
            if n_words == 2:
                if is_compound_noun:
                    core_noun_fallbacks = [
                        "rush hour", "heart attack", "side effect", "turning point",
                        "traffic jam", "common sense", "peer pressure", "culture shock",
                        "comfort zone", "safety net", "role model", "wake-up call"
                    ]
                    for cnf in core_noun_fallbacks:
                        if cnf != phrase_clean and cnf not in distractors and ships(cnf):
                            distractors.append(cnf)
                            if len(distractors) >= count:
                                break
                elif target_head_is_verb:
                    core_pv_fallbacks = [
                        "carry out", "point out", "turn down", "bring up", "set up",
                        "take on", "give up", "look into", "put off", "call off",
                        "hold on", "break down", "come across", "figure out", "work out"
                    ]
                    for cpv in core_pv_fallbacks:
                        if cpv != phrase_clean and cpv not in distractors and ships(cpv):
                            distractors.append(cpv)
                            if len(distractors) >= count:
                                break
                elif target_head_is_prep:
                    core_prep_fallbacks = [
                        "at risk", "on guard", "in danger", "under pressure", "by chance",
                        "on duty", "at peace", "in vain", "at fault", "on purpose"
                    ]
                    for cpp in core_prep_fallbacks:
                        if cpp != phrase_clean and cpp not in distractors and ships(cpp):
                            distractors.append(cpp)
                            if len(distractors) >= count:
                                break

                if len(distractors) < count:
                    for k in phrase_index:
                        if len(k.split()) == 2 and ships(k) and k not in distractors:
                            if target_head_is_verb and phrase_index.get(k) != "phrasal_verb":
                                continue
                            distractors.append(k)
                            if len(distractors) >= count:
                                break
            elif n_words >= 3:
                seen_fb_prefixes = {" ".join(c.split()[:-1]) for c in distractors if len(c.split()) >= 3}
                # Fallback to authentic multi-word phrases of matching length from phrase_index
                for k in phrase_index:
                    kw = k.split()
                    if len(kw) == n_words and ships(k) and k not in distractors:
                        k_pref = " ".join(kw[:-1]) if len(kw) >= 3 else k
                        if k_pref in seen_fb_prefixes:
                            continue
                        seen_fb_prefixes.add(k_pref)
                        distractors.append(k)
                    if len(distractors) >= count:
                        break

        res = distractors[:count]
        if "one's" in words:
            res = [re.sub(r"\byour\b", "one's", d) for d in res]
        res = [
            re.sub(r"\ba\s+(?!(?:uni|use|usu|urol|utens|eul|eup|eur|one)\w*)([aeiou][a-z]+)\b", r"an \1", d)
            for d in res
        ]
        return res
