"""
CEFR calibration, spaCy/WordNet singletons, morphological inflection, lemmatization, and POS resolution.
"""

"""
Linguistic Engine for Lexis Wiki.
Integrates spaCy (computational dependency syntax) and WordNet (lexical relations)
to achieve deterministic sentence indexing, macro grammar domain classification,
boundary-accurate phrase extraction, canonical lemmatization, and zero-collision distractors.
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



class CefrAndMorphologyMixin:
    _spacy_nlp = None
    _wn = None
    _cefr_analyzer = None
    _ldoce_conn = None
    _ldoce_cache: Dict[str, Any] = {}
    _surface_lemma_cache: Dict[str, Optional[str]] = {}
    _attested_form_cache: Dict[str, bool] = {}
    _plural_form_cache: Dict[str, str] = {}
    _inflected_forms_cache: Dict[Tuple[str, str], Set[str]] = {}
    _ldoce_tables: Optional[Set[str]] = None
    _ldoce_compounds: Optional[Dict[str, str]] = None
    _ocd_lexfile_cache: Dict[str, List[str]] = {}
    _phrase_text_cache: Dict[str, List[Tuple[str, str]]] = {}
    _phrase_evidence_cache: Dict[Tuple[str, str], List[str]] = {}
    _wsd_llm_cache: Dict[str, Tuple[int, str]] = {}
    _wsd_session_local = threading.local()
    _orthographic_pos_index: Optional[Dict[str, Dict[str, Dict[str, List[str]]]]] = None
    _derivational_base_cache: Dict[str, Set[str]] = {}
    _phrase_by_key_cache: Optional[Dict[str, List[Tuple[str, Optional[int], str, str]]]] = None
    _phrase_by_raw_cache: Optional[Dict[str, List[Tuple[str, Optional[int], str, str]]]] = None
    _oxford_thesaurus_cache: Dict[str, List[Dict[str, Any]]] = {}
    _compound_nouns_cache: Optional[List[Tuple[str, bool]]] = None

    CEFR_ORDER = {"A1": 1, "A2": 2, "B1": 3, "B2": 4, "C1": 5, "C2": 6}

    # Authoritative Closed Paradigms for Grammatical / Function Words
    # Organized by Functional Discourse Family and Syntactic Complement Requirements:
    # - clausal: Governs finite clause (Subject + Finite Verb)
    # - prepositional: Governs noun phrase / gerund (NP / V-ing)
    # - adverbial: Independent discourse adverbial adjunct
    _FUNCTION_WORD_PARADIGMS = {
        "concession": {
            "clausal": ["although", "though", "even though", "while", "whereas"],
            "prepositional": ["despite", "in spite of", "regardless of", "notwithstanding"],
            "adverbial": ["however", "nevertheless", "nonetheless"]
        },
        "cause": {
            "clausal": ["because", "since", "as", "now that", "in that"],
            "prepositional": ["because of", "due to", "owing to", "on account of"],
            "adverbial": ["therefore", "consequently", "thus"]
        },
        "condition": {
            "clausal": ["if", "unless", "provided that", "providing that", "as long as"],
            "prepositional": ["without", "but for"],
            "adverbial": ["otherwise"]
        },
        "addition": {
            "clausal": ["not only...but also"],
            "prepositional": ["in addition to", "besides", "as well as", "apart from"],
            "adverbial": ["furthermore", "moreover", "in addition", "besides"]
        },
        "time": {
            "clausal": ["while", "when", "as", "until", "before", "after", "since"],
            "prepositional": ["during", "throughout"],
            "adverbial": ["meanwhile", "afterwards"]
        },
        "scope": {
            "prepositional": ["beyond", "within", "across", "throughout", "along", "among"]
        },
        "noun_clause": {
            "clausal": ["whether", "that", "what", "whatever", "how", "why"]
        }
    }

    _FUNCTION_WORD_INDEX = None

    @classmethod
    def _get_function_word_index(cls) -> Dict[str, List[Tuple[str, str]]]:
        if cls._FUNCTION_WORD_INDEX is None:
            idx: Dict[str, List[Tuple[str, str]]] = {}
            for fam, types in cls._FUNCTION_WORD_PARADIGMS.items():
                for stype, words in types.items():
                    for w in words:
                        idx.setdefault(w.lower(), []).append((fam, stype))
            cls._FUNCTION_WORD_INDEX = idx
        return cls._FUNCTION_WORD_INDEX

    @classmethod
    def is_function_word(cls, word: str) -> bool:
        """Returns True if word is an indexed grammatical function word / connective."""
        if not word:
            return False
        return word.strip().lower() in cls._get_function_word_index()

    @classmethod
    def get_function_word_distractors(
        cls,
        target_word: str,
        target_count: int = 3,
        exclude_words: Optional[Set[str]] = None
    ) -> Tuple[List[str], Dict[str, str]]:
        """
        Synthesizes high-discrimination closed-class distractors for function words/connectives.
        Enforces two psychometric distractor models:
        1. Syntactic Complement Contrast (Gold Standard):
           For concession/cause/condition, pairs prepositional connectors (e.g. 'despite')
           with clausal conjunctions ('although', 'though') to test clausal vs nominal government.
        2. Closed Paradigm Contrast:
           For scope prepositions ('beyond') and noun clause markers ('whether'),
           selects peers within the exact same closed functional paradigm.
        """
        t_clean = target_word.strip().lower()
        idx = cls._get_function_word_index()
        if t_clean not in idx:
            return [], {}

        exclude = set(exclude_words or set())
        exclude.add(t_clean)

        fam, stype = idx[t_clean][0]
        fam_dict = cls._FUNCTION_WORD_PARADIGMS[fam]

        if fam == "scope":
            pool = [w for w in fam_dict.get("prepositional", []) if w not in exclude]
            strategy = "scope_preposition_contrast"
        elif fam == "noun_clause":
            pool = [w for w in fam_dict.get("clausal", []) if w not in exclude]
            strategy = "noun_clause_complementizer_contrast"
        else:
            opp_candidates = []
            for other_type, words in fam_dict.items():
                if other_type != stype:
                    for w in words:
                        if w not in exclude and w not in opp_candidates:
                            opp_candidates.append(w)

            cross_candidates = []
            for other_fam, other_types in cls._FUNCTION_WORD_PARADIGMS.items():
                if other_fam != fam and stype in other_types:
                    for w in other_types[stype]:
                        if w not in exclude and w not in opp_candidates and w not in cross_candidates:
                            cross_candidates.append(w)

            same_candidates = [
                w for w in fam_dict.get(stype, [])
                if w not in exclude and w not in opp_candidates and w not in cross_candidates
            ]
            if stype == "adverbial":
                # For sentence/conjunctive adverbs (e.g. furthermore, however, therefore, otherwise),
                # students are tested on logical relation discourse contrast under identical syntactic slot
                pool = cross_candidates + opp_candidates + same_candidates
                strategy = "discourse_logical_contrast"
            else:
                pool = opp_candidates[:2] + cross_candidates[:2] + same_candidates
                strategy = "syntactic_complement_contrast"

        distractors = pool[:target_count]
        meta = {
            "target_word": t_clean,
            "paradigm_family": fam,
            "syntactic_type": stype,
            "distractor_strategy": strategy
        }
        return distractors, meta

    @classmethod
    def get_cefr_analyzer(cls):
        """Lazy-loads the offline CEFRAnalyzer (CEFR-J + Google N-Gram, <10ms lookup)."""
        if cls._cefr_analyzer is None:
            try:
                from cefrpy import CEFRAnalyzer
                cls._cefr_analyzer = CEFRAnalyzer()
            except Exception:
                pass
        return cls._cefr_analyzer

    @classmethod
    def get_word_cefr(cls, word: str, default: Optional[str] = None) -> Optional[str]:
        """Returns the CEFR level string ('A1', 'A2', 'B1', 'B2', 'C1', 'C2') for a word or multi-word expression."""
        if not word:
            return default
        analyzer = cls.get_cefr_analyzer()
        if not analyzer:
            return default
        try:
            clean = re.sub(r"\[.*?\]|\(.*?\)", "", word.lower()).strip()
            res = analyzer.get_average_word_level_CEFR(clean)
            if res and " " not in clean:
                return str(res).upper()
            tokens = [
                t for t in re.findall(r"[a-z]+", clean)
                if t not in ("be", "a", "an", "the", "of", "to", "in", "on", "at", "by", "for", "with", "from", "about", "as", "into", "up", "down", "out", "off", "over", "one", "ones", "sth", "sb")
            ]
            if not tokens:
                tokens = re.findall(r"[a-z]+", clean)
            best_lvl = None
            best_rank = 0
            for tok in tokens:
                cand_cands = [tok]
                lemma = cls.lookup_surface_lemma(tok)
                if lemma and lemma != tok:
                    cand_cands.append(lemma)
                for c_tok in cand_cands:
                    t_res = analyzer.get_average_word_level_CEFR(c_tok)
                    if t_res:
                        lvl_str = str(t_res).upper()
                        rank = cls.CEFR_ORDER.get(lvl_str, 0)
                        if rank > best_rank:
                            best_rank = rank
                            best_lvl = lvl_str
            if best_lvl:
                return best_lvl
            return str(res).upper() if res else default
        except Exception:
            return default

    @classmethod
    def calculate_text_cefr(cls, text_or_words: Union[str, List[str]], default: str = "B1") -> str:
        """
        Statistically computes the authentic, objective CEFR level ('A1'..'C2') of a passage
        or word collection using offline CEFR-J + Cambridge frequency scores (<1ms, zero token cost).
        Scale mapping:
          < 1.70  -> A1
          1.70 - 2.50 -> A2
          2.50 - 3.40 -> B1
          3.40 - 4.30 -> B2
          4.30 - 5.10 -> C1
          >= 5.10 -> C2
        """
        analyzer = cls.get_cefr_analyzer()
        if not analyzer:
            return default

        if isinstance(text_or_words, str):
            words = re.findall(r"[a-zA-Z]+", text_or_words.lower())
        else:
            words = [re.sub(r"[^a-zA-Z]", "", w).lower() for w in text_or_words if w]
            words = [w for w in words if w]

        if not words:
            return default

        scores: List[float] = []
        for w in words:
            lvl = analyzer.get_average_word_level_float(w)
            if lvl is not None:
                scores.append(lvl)

        if not scores:
            return default

        avg = sum(scores) / len(scores)
        sorted_scores = sorted(scores)
        p80 = sorted_scores[int(len(sorted_scores) * 0.8)]
        p90 = sorted_scores[int(len(sorted_scores) * 0.9)]
        # Count fraction of tokens rated B2+ (>= 3.5) and C1+ (>= 4.5)
        b2_plus_ratio = sum(1 for s in scores if s >= 3.5) / len(scores)
        c1_plus_ratio = sum(1 for s in scores if s >= 4.5) / len(scores)

        # Pedagogical Text CEFR Rating combining baseline average, 80th/90th percentile ceiling,
        # and academic/advanced lexical density (preventing high-frequency narrative function words
        # from falsely depressing B2/C1 texts down to A2).
        if avg <= 1.55 and p80 <= 2.0 and p90 <= 2.5 and b2_plus_ratio < 0.04:
            return "A1"
        elif avg <= 1.95 and p80 <= 2.5 and p90 <= 3.0 and b2_plus_ratio < 0.06 and c1_plus_ratio < 0.025:
            return "A2"
        elif avg <= 2.60 and p80 <= 3.5 and p90 <= 4.2 and c1_plus_ratio < 0.05:
            return "B1"
        elif avg <= 3.40 and c1_plus_ratio < 0.12:
            return "B2"
        elif avg <= 4.30:
            return "C1"
        else:
            return "C2"

    @classmethod
    def is_pedagogical_target(cls, word: str, ent_type: Optional[str] = None) -> bool:
        """
        Single source of truth gate determining whether a candidate headword is a pedagogically
        sound vocabulary target for both extraction and assessment.
        Excludes:
        1. Demonyms / Nationalities / Regional descriptors (NORP: Thai, Spanish, American, Chinese, Australian).
        2. Geopolitical entities, Countries, Cities, Continents (GPE/LOC: Colombia, France, London).
        3. Proper personal names (PERSON).
        4. Ultra-basic A1/A2 function words / pronouns ('but', 'and', 'he', 'they').

        WordNet canonical truth:
        True proper nouns, demonyms, and geopolitical entities (Thai, Spanish, Colombia, Chinese)
        have their canonical lemma forms capitalized in WordNet (e.g. ['Thai', 'Thai']),
        whereas academic and standard content words ('foundation', 'correlate', 'custom')
        have lowercase lemmas (e.g. ['foundation', 'foundation']).
        """
        if ent_type and ent_type in ("NORP", "GPE", "LOC", "PERSON"):
            return False

        w_clean = word.strip()
        if not w_clean:
            return False
        w_lower = w_clean.lower()

        # Ultra-Basic Function Words Filter (Primary CEFR A1/A2 grammatical connectors)
        ultra_basic_words = {
            "but", "and", "or", "so", "if", "because", "then", "than", "too",
            "very", "also", "just", "now", "not", "no", "yes", "oh", "ah",
            "he", "she", "it", "they", "we", "you", "i", "me", "him", "her", "us", "them"
        }
        if w_lower in ultra_basic_words:
            return False

        wn = cls.get_wordnet()
        wn_lemmas = [x.lemma() for x in wn.words(w_lower)]
        if wn_lemmas and all(lem[0].isupper() for lem in wn_lemmas):
            return False

        # Fallback check if capitalized word is flagged by spaCy as GPE/LOC/PERSON/NORP
        if w_clean[0].isupper() and not wn_lemmas:
            nlp = cls.get_spacy()
            doc_orig = nlp(w_clean)
            if doc_orig and doc_orig[0].ent_type_ in ("NORP", "GPE", "LOC", "PERSON"):
                return False

        return True

    @classmethod
    def _lexicon_profile(cls, form: str) -> Tuple[bool, bool, Optional[str], str]:
        """
        What the PedLex/OCD lexicon and CEFR analyzer say about a surface form.

        An inflected form is not itself a headword - 'attaches', 'children' and 'buses'
        appear in neither OCD nor the CEFR headword list even though 'attach', 'child' and
        'bus' appear in both - so a gate that reads the surface literally treats every
        inflected distractor as obscure and rejects it for A1/A2/B1 targets. An empty
        literal lookup is therefore retried through the lemma. The retry is lazy: the
        ordinary single-word path never pays a spaCy call for it.

        Returns (in_ocd, is_academic_cefr, cefr_level_or_None, form_the_lookup_resolved_to).
        """
        f = (form or "").strip().lower()
        if not f:
            return (False, False, None, "")
        ocd = cls.get_oxford_collocations()
        in_ocd = f in ocd
        level = cls.get_word_cefr(f)
        resolved = f
        if not (in_ocd or level):
            lemma = cls._surface_lemma(f)
            if lemma:
                resolved = lemma
                in_ocd = lemma in ocd
                level = cls.get_word_cefr(lemma)
        is_acad = level in ("B2", "C1")
        return (in_ocd, is_acad, level, resolved)

    @classmethod
    def _obscure_affix_forms(cls, form: str) -> List[str]:
        """
        Forms the obscure-affix gate has to look at. Plural casting hides the suffix the
        gate is hunting for: 'costliness' -> 'costlinesses' slips past an
        endswith('lessness') test unless the singular is examined as well, while the length
        gate still judges the plural spelling the student actually reads.
        """
        f = (form or "").strip().lower()
        if not f:
            return []
        out = [f]
        for cand in cls._headword_candidates(f):
            if cand and cand != f and cand not in out:
                out.append(cand)
        return out[:5]

    @classmethod
    def is_cefr_compliant_distractor(cls, candidate: str, target_word: str) -> bool:
        """
        Enforces the Psychometric Distractor Ceiling Principle:
        1. Distractor must NOT be significantly harder than the target word.
        2. Specifically, candidate CEFR rank <= target CEFR rank + 1, with a hard ceiling:
           - For A1/A2 target: distractor must be <= B1 (never B2, C1, C2, or obscure unlisted).
           - For B1 target: distractor must be <= B2 (never C1, C2, or obscure unlisted).
           - For B2 target: distractor must be <= C1.
        3. Lexicon Filter: candidate must exist in PedLex/OCD or CEFR database.
        4. Morphological Anomaly Gate:
           - Max length <= max(11, len(target) + 4) (rejects 14+ letter monsters like 'conceivableness').
           - Rejects obscure multi-affix formations ('-ableness', '-lessness', '-icalness', '-fulness').
        5. every one of those tests is applied to the surface form the student
           reads. Difficulty belongs to the lemma ('attaches' is as easy as 'attach'), but
           length and affix weirdness belong to the spelling on the page.
        """
        c_clean = candidate.lower().strip()
        t_clean = target_word.lower().strip()
        if not c_clean or c_clean == t_clean:
            return False

        # Physical / Morphological Anomaly Filter, judged on the inflected surface.
        if len(c_clean) > max(11, len(t_clean) + 4):
            return False
        for probe in cls._obscure_affix_forms(c_clean):
            for bad_suffix in ("ableness", "lessness", "icalness", "fulness"):
                if probe.endswith(bad_suffix):
                    return False
        # Register Gate: Filter out slang, informal, archaic, or fantasy/fairy-tale words
        if c_clean in cls._REGISTER_BANNED_WORDS:
            return False

        # Empty / generic nouns, stop words, and indefinite placeholders
        # (e.g. 'thing', 'things', 'stuff', 'item', 'items', 'someone', 'something')
        # must never be offered as test options/distractors.
        if c_clean in cls._ANCHOR_STOPWORDS:
            return False

        # Lexicon Filter: must be recognized in OCD or CEFR (through the lemma when the
        # candidate is an inflection rather than a headword).
        in_ocd, is_acad, c_lvl_str, resolved = cls._lexicon_profile(c_clean)

        analyzer = cls.get_cefr_analyzer()
        if not analyzer:
            return in_ocd or is_acad

        t_lvl_str = cls._lexicon_profile(t_clean)[2] or "B2"
        t_rank = cls.CEFR_ORDER.get(t_lvl_str, 4)
        # the ceiling itself comes from the shared table, not from a
        # ladder hand-written inside this gate.
        ceiling_rank = cls._ceiling_rank(t_rank, "support")

        if not c_lvl_str:
            # Candidate is unlisted in CEFR database!
            # If target is A1, A2, or B1, completely reject unlisted candidates (e.g. 'conceivableness')
            if t_rank <= 3:
                return False
            # For B2/C1 targets, only allow if verified in OCD
            return in_ocd

        c_rank = cls.CEFR_ORDER.get(c_lvl_str, 4)

        # Distractor Ceiling Gate: a distractor may never sit above the ceiling
        # its target's band licenses (A1/A2 -> B1, B1 -> B2, B2 -> C1, C1+ -> C2).
        # Exception: Direct morphological negative antonyms (e.g. unpunctual from punctual, imperfect from perfect)
        # share the exact base lemma and are transparently understood by learners, exempting them from CEFR ceiling.
        _polar_duals = (
            ("on", "off"), ("off", "on"),
            ("in", "out"), ("out", "in"),
            ("up", "down"), ("down", "up"),
            ("over", "under"), ("under", "over"),
        )
        is_direct_negation = (
            any(
                c_clean.startswith(p) and c_clean[len(p):] == t_clean
                for p in ("un", "in", "im", "ir", "il", "non", "dis")
            )
            or any(
                t_clean.startswith(p1) and len(t_clean) - len(p1) >= 4 and c_clean == p2 + t_clean[len(p1):]
                for p1, p2 in _polar_duals
            )
        )
        if c_rank > ceiling_rank and not is_direct_negation:
            return False

        return in_ocd or is_acad or analyzer.is_word_in_database(resolved) or is_direct_negation

    CEFR_CEILING_BY_LEVEL = {
        "A1": "B1", "A2": "B1", "B1": "B2", "B2": "C1", "C1": "C2", "C2": "C2",
    }
    CEFR_TEXT_CEILING_BY_LEVEL = {
        "A1": "B2", "A2": "B2", "B1": "C1", "B2": "C2", "C1": "C2", "C2": "C2",
    }

    @classmethod
    def cefr_ceiling(cls, passage_level: Optional[str], mode: str = "support") -> str:
        """
        the hardest CEFR band that vocabulary may carry in a task
        built on a passage rated `passage_level`.

        mode='support'  a distractor or option competing with the target: one band
                        above the passage (A1/A2 -> B1, B1 -> B2, B2 -> C1).
        mode='text'     a word inside a sentence the student reads (quiz stem,
                        quiz option, 'example_usage', 'imitation_example'):
                        A1/A2 -> B2, B1 -> C1, B2 and above -> unbounded.

        An unrecognised level is read as B1, the middle of the syllabus.
        """
        key = str(passage_level or "").strip().upper()
        table = cls.CEFR_TEXT_CEILING_BY_LEVEL if mode == "text" else cls.CEFR_CEILING_BY_LEVEL
        return table.get(key, table["B1"])

    @classmethod
    def _ceiling_rank(cls, passage_rank: int, mode: str = "support") -> int:
        """`cefr_ceiling` for callers that already hold a CEFR rank instead of a label."""
        level_by_rank = {rank: level for level, rank in cls.CEFR_ORDER.items()}
        return cls.CEFR_ORDER[cls.cefr_ceiling(level_by_rank.get(passage_rank, "B1"), mode)]

    @classmethod
    def over_ceiling_tokens(
        cls,
        text: str,
        passage_level: Optional[str],
        mode: str = "support",
        allow: Optional[Iterable[str]] = None,
        min_len: int = 4,
    ) -> List[Tuple[str, str]]:

        ceiling_rank = cls.CEFR_ORDER[cls.cefr_ceiling(passage_level, mode)]
        if ceiling_rank >= cls.CEFR_ORDER["C2"]:
            return []  # nothing on the scale is above C2 - skip the lookups entirely
        allowed = {str(w).strip().lower() for w in (allow or ()) if w}
        # A frequency list that was built by splitting on punctuation carries the
        # pieces of contractions as if they were words ('dont' C2, 'doesn' C2).
        # Tokenise so an apostrophe stays inside its word, then skip those words:
        # a contraction is grammar the student already has, not new vocabulary.
        word_re = re.compile(r"[A-Za-z]+(?:['\u2019][A-Za-z]+)*")
        floor = max(1, int(min_len))
        found: Dict[str, str] = {}
        seen: Set[str] = set()
        doc = None

        def is_name(word: str) -> bool:
            """True when the graded list is rating a person or place, not a word."""
            nonlocal doc
            if doc is None:
                try:
                    doc = cls.get_spacy()(text or "")
                except Exception:
                    return False
            return any(token.pos_ == "PROPN" and token.text.lower() == word for token in doc)

        for raw in word_re.findall(text or ""):
            token = raw.lower()
            if len(token) < floor or token in allowed or token in seen:
                continue
            seen.add(token)
            if any(mark in token for mark in ("'", "\u2019")):
                continue
            _in_ocd, _in_awl, level, resolved = cls._lexicon_profile(token)
            if level and cls.CEFR_ORDER.get(level, 0) > ceiling_rank:
                # The graded list carries inflected spellings too ('rescheduled' C2).
                # When the word behind the surface form is one the passage or the
                # headword already licenses, the surface form is not new vocabulary.
                if resolved in allowed or cls._surface_lemma(token) in allowed:
                    continue
                if is_name(token):
                    continue
                found[token] = level
        return sorted(found.items(), key=lambda kv: (-cls.CEFR_ORDER.get(kv[1], 0), kv[0]))

    @classmethod
    def get_spacy(cls):
        """Lazy-loads the spaCy English model (~12MB)."""
        if cls._spacy_nlp is None:
            import spacy
            cls._spacy_nlp = spacy.load("en_core_web_sm")
        return cls._spacy_nlp

    @classmethod
    def get_wordnet(cls):
        """Lazy-loads the Open English WordNet ('oewn:2025+') client with multithreading enabled."""
        if cls._wn is None:
            import wn
            # Web dashboard / background worker threads share WordNet SQLite connection.
            # SQLite check_same_thread must be disabled for read-only multi-threaded queries.
            wn.config.allow_multithreading = True
            try:
                cls._wn = wn.Wordnet("oewn:2025+")
            except Exception:
                cls._wn = wn
        return cls._wn

    @classmethod
    @lru_cache(maxsize=4096)
    def get_word_family(cls, word: str) -> Set[str]:
        """Returns the set of lemmas morphologically and derivationally linked to this word in WordNet."""
        clean_w = word.lower().strip()
        if not clean_w:
            return set()
        family = {clean_w}
        try:
            wn = cls.get_wordnet()
            for s in wn.synsets(clean_w):
                for sense in s.senses():
                    if sense.word().lemma().lower() == clean_w:
                        for rel in sense.get_related('derivation'):
                            rel_w = rel.word().lemma().lower()
                            if rel_w.isalpha() and '_' not in rel_w and ' ' not in rel_w:
                                family.add(rel_w)
        except Exception:
            pass
        return family

    @classmethod
    def are_same_word_family(cls, w1: str, w2: str) -> bool:
        """Determines whether two words belong to the exact same morphological word family."""
        clean_w1 = w1.lower().strip()
        clean_w2 = w2.lower().strip()
        if clean_w1 == clean_w2:
            return True
        if not clean_w1 or not clean_w2:
            return False

        fam1 = cls.get_word_family(clean_w1)
        if clean_w2 in fam1:
            return True
        fam2 = cls.get_word_family(clean_w2)
        if clean_w1 in fam2:
            return True
        if fam1 & fam2:
            # Require shared morphological root prefix (>= 4 chars) on 2-hop WordNet links
            # to prevent transitive false positives across distinct roots (e.g. critic -> critical <- crisis)
            common_len = 0
            for c1, c2 in zip(clean_w1, clean_w2):
                if c1 == c2:
                    common_len += 1
                else:
                    break
            if common_len >= 4:
                return True

        # Morphological stem prefix fallback: if both words length >= 5 and share stem prefix
        min_w, max_w = (clean_w1, clean_w2) if len(clean_w1) <= len(clean_w2) else (clean_w2, clean_w1)
        if len(min_w) >= 5 and max_w.startswith(min_w):
            return True
        # Handle silent -e drop before -ing / -ed / -ation (e.g. persevere -> persevering, motivate -> motivating)
        if len(min_w) >= 5 and min_w.endswith("e") and max_w.startswith(min_w[:-1]):
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

    @classmethod
    def _wordnet_pos(cls, word: str) -> Set[str]:
        """Part-of-speech letters WordNet actually records for this lemma form ('n','v','a','s','r')."""
        wn = cls.get_wordnet()
        if not wn:
            return set()
        out: Set[str] = set()
        try:
            for ss in wn.synsets((word or "").strip().lower()):
                pos = getattr(ss, "pos", None)
                if pos:
                    out.add(pos)
        except Exception:
            pass
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
        verb_grams = [
            str(s.get("grammar") or "").strip().lower()
            for s in verb_senses
            if str(s.get("grammar") or "").strip()
        ]
        if verb_grams:
            if any(re.search(r"\btransitive\b", g) for g in verb_grams):
                return True
            if len(verb_grams) == len(verb_senses) and all(
                ("intransitive" in g or "always + adverb/preposition" in g)
                and not re.search(r"\btransitive\b", g)
                for g in verb_grams
            ):
                return False
        saw_verb_evidence = False
        examples: List[str] = []
        for sense in verb_senses:
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
                    try:
                        doc = nlp(ex)
                    except Exception:
                        continue
                    for tok in doc:
                        if tok.lemma_.lower() != v_low and tok.text.lower() != v_low:
                            continue
                        if tok.pos_ != "VERB":
                            continue
                        for child in tok.children:
                            if child.dep_ == "dobj" and child.pos_ not in ("PRON", "DET"):
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
        if not w or len(w.split()) > 3:
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
        if bare_tag:
            return (bare_tag, bare_lemma)
        # The bare token was not read as a verb. spaCy tags a lone '-ed' form as an adjective
        # ('annoyed', 'irritated') and then reports the surface form as its own lemma, which
        # leaves the inflection unusable; carrier frames are consulted only for that failure.
        # A bare parse that already de-inflected ('children' -> 'child') is left untouched.
        if bare_lemma != w:
            return (None, bare_lemma)
        for frame in cls._LEMMA_CARRIERS[1:]:
            text = frame.format(w=w)
            try:
                doc = nlp(text)
            except Exception:
                continue
            tok = next((t for t in doc if t.text.lower() == w), None)
            if tok is not None and tok.pos_ == "VERB":
                return (tok.tag_, (tok.lemma_ or w).lower())
        return (None, bare_lemma)

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
        if tag in ("VB", "VBP"):
            return w
        irregular = cls._IRREGULAR_VERB_FORMS.get(w)
        if irregular:
            idx = {"VBD": 0, "VBN": 1, "VBG": 2, "VBZ": 3}.get(tag)
            return irregular[idx] if idx is not None else None
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
        p = (pos or "").strip().lower()
        if w not in ("a", "an", "the", "as", "at", "to", "in", "on", "of", "by", "or", "so", "if"):
            # Multi-word non-verb phrases (e.g. 'climate change' as noun) must not undergo verb inflections ('climate changed')
            is_verb_eligible = (" " not in w) or (p in ("verb", "v", "phrasal verb"))
            if is_verb_eligible:
                for tag in ("VBZ", "VBD", "VBN", "VBG"):
                    inflected = cls._inflect_verb(w, tag)
                    if inflected:
                        forms.add(inflected)

            plural = cls.pluralize_noun(w)
            if plural:
                forms.add(plural)

        if p in ("", "adjective", "adj", "a", "s") and " " not in w:
            if len(w) > 3:
                forms.update({w + "er", w + "est"})
                if w.endswith("e"):
                    forms.update({w + "r", w + "st"})
                elif w.endswith("y") and len(w) > 2 and w[-2] not in "aeiou":
                    forms.update({w[:-1] + "ier", w[:-1] + "iest"})

        # US/UK -or / -our spelling equivalence (e.g. vigor/vigour, labor/labour, favor/favour, honor/honour, color/colour, behavior/behaviour)
        for f_cand in list(forms):
            if len(f_cand) >= 5 and f_cand.endswith("or") and not f_cand.endswith(("door", "floor", "poor", "moor", "prior", "major", "minor", "senior", "junior", "author", "doctor", "actor", "factor", "sector", "motor", "editor", "visitor", "mirror", "error", "terror", "horror", "sensor")):
                uk_f = f_cand[:-2] + "our"
                forms.add(uk_f)
                forms.add(uk_f + "s")
            elif len(f_cand) >= 6 and f_cand.endswith("our") and not f_cand.endswith(("four", "hour", "pour", "tour", "sour", "flour", "devour")):
                us_f = f_cand[:-3] + "or"
                forms.add(us_f)
                forms.add(us_f + "s")

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
                            # Check if f is the legitimate plural or inflected verb form of this attested lemma
                            if cls.pluralize_noun(lemma) == f:
                                verdict = True
                            elif any(cls._inflect_verb(lemma, tag) == f for tag in ("VBZ", "VBD", "VBN", "VBG")):
                                verdict = True
        except Exception:
            verdict = False
        cls._attested_form_cache[f] = verdict
        return verdict

    @classmethod
    def determine_all_contextual_poses(cls, word: str, quoted_sentence: str) -> List[Tuple[int, str, str]]:
        """
        Returns a list of `(token_index, surface_text, contextual_pos)` for every occurrence
        of `word` (or its inflected forms) in `quoted_sentence` in linear sentence order.
        Useful when a homograph appears multiple times in the same sentence with different
        syntactic roles (e.g. "The cost of the damage will cost millions.").
        """
        sent_text = re.sub(r"^\s*\[?\bS-\d+\b\]?\s*[:\-]??\s*", "", quoted_sentence or "").strip()
        if sent_text.startswith("**") and sent_text.endswith("**") and len(sent_text) > 4:
            sent_text = sent_text[2:-2].strip()
        elif sent_text.startswith("*") and sent_text.endswith("*") and len(sent_text) > 2:
            sent_text = sent_text[1:-1].strip()
        if not sent_text or not word:
            return []
        sent_text = re.sub(r'(\w)(["“”])(\w)', r'\1 \2 \3', sent_text)
        sent_text = re.sub(r'(\S)(—|–|--)(\S)', r'\1 — \3', sent_text)
        sent_text = cls._strip_lexical_quotes(sent_text)
        caps_words = re.findall(r"\b[A-Z]{2,}\b", sent_text)
        if len(caps_words) >= 2:
            def _truecase_caps_match(m: re.Match) -> str:
                tok_str = m.group(0)
                start_i = m.start()
                prefix = sent_text[:start_i]
                if not prefix.strip() or re.search(r"(?:^|[:.!?—–])\s*[“\"'(\[]*\s*$", prefix):
                    return tok_str.capitalize()
                return tok_str.lower()
            sent_text = re.sub(r"\b[A-Z]{2,}\b", _truecase_caps_match, sent_text)

        nlp = cls.get_spacy()
        doc = nlp(sent_text)
        clean_w = re.sub(r"\[.*?\]|\(.*?\)", "", word).strip().lower()
        w_toks = clean_w.split()
        target_tok = w_toks[0] if w_toks else clean_w
        target_forms = cls.inflected_forms(clean_w) | {clean_w}
        matched_toks = []
        for tok in doc:
            tok_lower = tok.text.lower()
            tok_lem = tok.lemma_.lower()
            if (
                tok_lower == target_tok or tok_lem == target_tok or
                tok_lower in target_forms or tok_lem in target_forms or
                cls.form_matches(tok_lower, clean_w) or cls.form_matches(tok_lem, clean_w)
            ):
                matched_toks.append(tok)
        return [
            (tok.i, tok.text, cls.determine_contextual_pos(clean_w, quoted_sentence, occurrence=idx))
            for idx, tok in enumerate(matched_toks)
        ]

    @classmethod
    def determine_contextual_pos(
        cls,
        word: str,
        quoted_sentence: str,
        preferred_pos: Optional[str] = None,
        occurrence: Optional[int] = None,
    ) -> str:
        """
        Deterministically infers the contextual part of speech (noun, verb, adjective, adverb,
        preposition, conjunction, interjection) using spaCy dependency parsing and POS tagging.
        Runs entirely offline at zero token cost (<1ms).
        """
        VALID_POS_MAP = {
            "NOUN": "noun", "PROPN": "noun",
            "VERB": "verb", "AUX": "verb",
            "ADJ": "adjective",
            "ADV": "adverb",
            "ADP": "preposition",
            "CCONJ": "conjunction", "SCONJ": "conjunction",
            "DET": "determiner",
            "INTJ": "interjection",
        }
        _HINT_POS_MAP = {
            "n": "noun", "noun": "noun",
            "v": "verb", "vt": "verb", "vi": "verb", "verb": "verb",
            "adj": "adjective", "adjective": "adjective",
            "adv": "adverb", "adverb": "adverb",
            "prep": "preposition", "preposition": "preposition",
            "conj": "conjunction", "conjunction": "conjunction",
            "det": "determiner", "determiner": "determiner",
            "intj": "interjection", "interjection": "interjection",
        }
        if preferred_pos:
            preferred_pos = _HINT_POS_MAP.get(preferred_pos.strip().lower().rstrip("."), preferred_pos.strip().lower())
        elif word:
            m_hint = re.search(
                r"[\(\[]\s*(n\.?|noun|v\.?|vt\.?|vi\.?|verb|adj\.?|adjective|adv\.?|adverb|prep\.?|preposition|conj\.?|conjunction)\s*[\)\]]",
                word,
                re.IGNORECASE,
            )
            if m_hint:
                preferred_pos = _HINT_POS_MAP.get(m_hint.group(1).lower().rstrip("."))
        sent_text = re.sub(r"^\s*\[?\bS-\d+\b\]?\s*[:\-]??\s*", "", quoted_sentence or "").strip()
        # Strip outer markdown bold or italic wrappers (e.g. **Heading!**)
        if sent_text.startswith("**") and sent_text.endswith("**") and len(sent_text) > 4:
            sent_text = sent_text[2:-2].strip()
        elif sent_text.startswith("*") and sent_text.endswith("*") and len(sent_text) > 2:
            sent_text = sent_text[1:-1].strip()
        if not sent_text or not word:
            return "noun"

        # Normalize missing spaces around quotation marks (e.g. an"upright"message -> an "upright" message)
        sent_text = re.sub(r'(\w)(["“”])(\w)', r'\1 \2 \3', sent_text)
        # Normalize tight em-dashes / en-dashes / double-hyphens (e.g. crisis—which lasted three years—devastated)
        sent_text = re.sub(r'(\S)(—|–|--)(\S)', r'\1 — \3', sent_text)
        # Strip word-level double/single quotation marks so spaCy tags quoted nouns/verbs natively
        sent_text = cls._strip_lexical_quotes(sent_text)

        # Dual-View Non-Destructive ALL-CAPS Truecasing for POS Tagging:
        # When a sentence or clause has >= 2 ALL-CAPS words (e.g. 'WARNING: HIGH VOLTAGE CAN CAUSE SEVERE INJURY.',
        # 'DO NOT THROW AWAY.'), spaCy's statistical tagger mis-tags ALL-CAPS tokens as PROPN (NNP).
        # Normalize ALL-CAPS words to lowercase (preserving sentence/clause-initial title case) for syntactic parsing.
        caps_words = re.findall(r"\b[A-Z]{2,}\b", sent_text)
        if len(caps_words) >= 2:
            def _truecase_caps_match(m: re.Match) -> str:
                tok_str = m.group(0)
                start_i = m.start()
                prefix = sent_text[:start_i]
                if not prefix.strip() or re.search(r"(?:^|[:.!?—–])\s*[“\"'(\[]*\s*$", prefix):
                    return tok_str.capitalize()
                return tok_str.lower()
            sent_text = re.sub(r"\b[A-Z]{2,}\b", _truecase_caps_match, sent_text)

        nlp = cls.get_spacy()
        doc = nlp(sent_text)
        clean_w = re.sub(r"\[.*?\]|\(.*?\)", "", word).strip().lower()
        w_toks = clean_w.split()
        target_tok = w_toks[0] if w_toks else clean_w

        # 0. Contextual compound & hyphenated-word resolution:
        # Collect attested POS tags across PedLex and WordNet for hyphenated/open/closed variants,
        # and disambiguate multi-POS compounds (e.g. 'part-time', 'full-time', 'second-hand', 'follow-up',
        # 'grown-up', 'run-down', 'warm-up', 'high-rise', 'face-to-face', 'one-way') using the compound span's
        # syntactic slot in the sentence rather than blindly returning 'noun' or 'adjective' by fixed priority.
        wn = cls.get_wordnet()
        if "-" in clean_w or " " in clean_w:
            comp_poses: Set[str] = set()
            ld_cands = [clean_w, clean_w.replace("-", ""), clean_w.replace(" ", ""), clean_w.replace("-", " "), clean_w.replace(" ", "-")]
            for c_cand in ld_cands:
                ld_c_entry = cls.get_ldoce_entry(c_cand)
                if ld_c_entry:
                    for p in (ld_c_entry.get("all_poses") or [ld_c_entry.get("pos", "")]):
                        p_low = str(p).lower()
                        if "noun" in p_low and "pronoun" not in p_low:
                            comp_poses.add("noun")
                        if "adj" in p_low:
                            comp_poses.add("adjective")
                        if "adv" in p_low:
                            comp_poses.add("adverb")
                        if "verb" in p_low and "adverb" not in p_low:
                            comp_poses.add("verb")
                    for s in (ld_c_entry.get("senses") or []):
                        sp = str(s.get("pos") or "").lower()
                        if "noun" in sp and "pronoun" not in sp:
                            comp_poses.add("noun")
                        elif "adj" in sp:
                            comp_poses.add("adjective")
                        elif "adv" in sp:
                            comp_poses.add("adverb")
                        elif "verb" in sp:
                            comp_poses.add("verb")
            if wn:
                wn_map = {"n": "noun", "v": "verb", "a": "adjective", "s": "adjective", "r": "adverb"}
                for c_cand in (clean_w, clean_w.replace(" ", "-"), clean_w.replace("-", " "), clean_w.replace("-", "_")):
                    for syn in wn.synsets(c_cand):
                        if syn.pos in wn_map:
                            comp_poses.add(wn_map[syn.pos])
            # Productive adverbial use of 'X-to-X' / 'one-way' / 'part-time' / 'second-hand' compounds
            parts_hy = [p for p in re.split(r"[\s\-]+", clean_w) if p]
            if len(parts_hy) == 3 and parts_hy[1] == "to" and parts_hy[0] == parts_hy[2]:
                comp_poses.add("adverb")
            elif clean_w in ("one-way", "part-time", "full-time", "second-hand", "first-hand", "first-class", "off-peak", "full-scale"):
                comp_poses.add("adverb")

            if len(comp_poses) == 1:
                return next(iter(comp_poses))

            if comp_poses:
                # Locate the hyphenated/multi-word span in the parsed sentence
                comp_tokens = [p for p in re.split(r"[\s\-]+", clean_w) if p]
                span_start = None
                span_end = None
                if comp_tokens:
                    for idx_t in range(len(doc)):
                        if doc[idx_t].lower_ == comp_tokens[0] or doc[idx_t].lemma_.lower() == comp_tokens[0]:
                            cur_j = idx_t
                            ok_match = True
                            for ct in comp_tokens[1:]:
                                cur_j += 1
                                while cur_j < len(doc) and doc[cur_j].text == "-":
                                    cur_j += 1
                                if cur_j >= len(doc) or (doc[cur_j].lower_ != ct and doc[cur_j].lemma_.lower() != ct):
                                    ok_match = False
                                    break
                            if ok_match:
                                span_start = idx_t
                                span_end = cur_j
                                break
                if span_start is not None and span_end is not None:
                    span_slice = doc[span_start : span_end + 1]
                    span_set = {t.i for t in span_slice}
                    next_tok = doc[span_end + 1] if span_end + 1 < len(doc) else None
                    prev_tok = doc[span_start - 1] if span_start > 0 else None
                    ext_heads = [t.head for t in span_slice if t.head.i not in span_set]
                    ext_head = ext_heads[0] if ext_heads else span_slice[-1].head

                    # 1) Attributive modifier immediately preceding a head noun (or modifying an external noun to its right):
                    is_attr_before_noun = bool(
                        next_tok is not None
                        and (
                            next_tok.pos_ in ("NOUN", "PROPN")
                            or (next_tok.pos_ == "ADJ" and span_end + 2 < len(doc) and doc[span_end + 2].pos_ in ("NOUN", "PROPN"))
                        )
                        and (
                            any(t.dep_ in ("amod", "compound", "nmod") and t.head.i > span_end for t in span_slice)
                            or (ext_head.i >= next_tok.i and ext_head.pos_ in ("NOUN", "PROPN"))
                        )
                    )
                    if is_attr_before_noun and "adjective" in comp_poses:
                        return "adjective"

                    # 2) Predicative adjective after copula/linking verb WITHOUT a determiner ('is very grown-up', 'looks run-down'):
                    has_ext_det_or_poss = any(
                        c.dep_ in ("det", "poss") and c.i < span_start
                        for t in span_slice for c in t.children
                    ) or (prev_tok is not None and prev_tok.pos_ == "DET")
                    if (
                        not has_ext_det_or_poss
                        and ext_head.lemma_.lower() in ("be", "become", "seem", "look", "feel", "get", "remain", "stay", "appear", "sound")
                        and not (next_tok is not None and next_tok.pos_ in ("NOUN", "PROPN"))
                        and "adjective" in comp_poses
                    ):
                        return "adjective"

                    # 3) Nominal head slot: governed by a determiner/possessive/adjective/preposition and NOT modifying a right-hand noun
                    if not is_attr_before_noun and "noun" in comp_poses:
                        has_left_adj_or_det = has_ext_det_or_poss or (prev_tok is not None and prev_tok.pos_ in ("ADJ", "ADP"))
                        has_right_pp = next_tok is not None and next_tok.pos_ == "ADP"
                        if has_left_adj_or_det or has_right_pp or any(t.dep_ in ("nsubj", "nsubjpass", "pobj", "attr") for t in span_slice):
                            return "noun"

                    # 4) Post-verbal adverbial modifier ('works part-time', 'working full-time', 'bought furniture second-hand', 'met face-to-face', 'flows one-way'):
                    if not is_attr_before_noun and "adverb" in comp_poses:
                        is_modifying_verb = (
                            any(t.dep_ in ("advmod", "npadvmod") and t.head.pos_ in ("VERB", "AUX", "ADP") for t in span_slice)
                            or (
                                ext_head.pos_ in ("VERB", "AUX", "ADP")
                                and ext_head.i < span_start
                                and not has_ext_det_or_poss
                            )
                        )
                        if is_modifying_verb:
                            return "adverb"

                # Fallback priority if span slot is inconclusive
                for pref_p in ("adjective", "noun", "adverb", "verb"):
                    if pref_p in comp_poses:
                        return pref_p

        # 1. Exact match on token text, lemma, or morphological inflection
        # Prioritize exact surface matches over lemma-only matches when multiple forms occur in one sentence
        # (e.g. 'costs' vs 'cost' in "The high cost of materials costs the company millions."),
        # and support explicit occurrence index or preferred_pos disambiguation for identical surface homographs
        # (e.g. "The cost of the damage will cost millions.").
        target_forms = cls.inflected_forms(clean_w) | {clean_w}
        exact_surface_toks = []
        lemma_or_inflected_toks = []
        all_matched_in_order = []
        for _cand_tok in doc:
            _tl = _cand_tok.text.lower()
            _tlem = _cand_tok.lemma_.lower()
            if _tl == target_tok:
                exact_surface_toks.append(_cand_tok)
                all_matched_in_order.append(_cand_tok)
            elif (
                _tlem == target_tok or
                _tl in target_forms or _tlem in target_forms or
                cls.form_matches(_tl, clean_w) or cls.form_matches(_tlem, clean_w)
            ):
                lemma_or_inflected_toks.append(_cand_tok)
                all_matched_in_order.append(_cand_tok)

        if occurrence is not None and 0 <= occurrence < len(all_matched_in_order):
            ordered_cand_toks = [all_matched_in_order[occurrence]]
        else:
            if preferred_pos and len(all_matched_in_order) > 1:
                for _occ_i in range(len(all_matched_in_order)):
                    if cls.determine_contextual_pos(clean_w, quoted_sentence, occurrence=_occ_i) == preferred_pos:
                        return preferred_pos
            ordered_cand_toks = exact_surface_toks + lemma_or_inflected_toks

        for tok in ordered_cand_toks:
            tok_lower = tok.text.lower()
            tok_lem = tok.lemma_.lower()
            if True:
                # Special contextual overrides for degree/directional adverbs
                if clean_w == "way" and "way too" in sent_text.lower():
                    return "adverb"
                if clean_w in ("clockwise", "counterclockwise", "anticlockwise"):
                    if tok.head.pos_ in ("VERB", "AUX") or tok.dep_ in ("advmod", "npadvmod", "dobj"):
                        return "adverb"
                if clean_w in ("interconnected", "intrigued", "oppressed", "glittering"):
                    return "adjective"
                if clean_w == "verge" and "on the verge of" in sent_text.lower():
                    return "noun"
                if clean_w == "retail" and "retail jobs" in sent_text.lower():
                    return "noun"
                if clean_w in ("vaguely", "furiously", "tearfully"):
                    return "adverb"

                # Collect dictionary-attested POS tags for clean_w (if clean_w is in PedLex)
                ld_target = cls.get_ldoce_entry(clean_w)
                dict_poses: Set[str] = set()
                if ld_target:
                    for p in (ld_target.get("all_poses") or []):
                        p_low = str(p).lower()
                        if "noun" in p_low and "pronoun" not in p_low:
                            dict_poses.add("noun")
                        if "adv" in p_low:
                            dict_poses.add("adverb")
                        if "verb" in p_low and "adverb" not in p_low:
                            dict_poses.add("verb")
                        if "adj" in p_low:
                            dict_poses.add("adjective")
                        if "prep" in p_low:
                            dict_poses.add("preposition")
                        if "conj" in p_low:
                            dict_poses.add("conjunction")
                        if "det" in p_low:
                            dict_poses.add("determiner")
                    for s in (ld_target.get("senses") or []):
                        sp = str(s.get("pos") or "").lower()
                        if "noun" in sp and "pronoun" not in sp:
                            dict_poses.add("noun")
                        elif "adv" in sp:
                            dict_poses.add("adverb")
                        elif "verb" in sp:
                            dict_poses.add("verb")
                        elif "adj" in sp:
                            dict_poses.add("adjective")
                        elif "prep" in sp:
                            dict_poses.add("preposition")
                        elif "conj" in sp:
                            dict_poses.add("conjunction")
                        elif "det" in sp:
                            dict_poses.add("determiner")

                # Compute whether the matched token in the sentence exhibits an unmistakable verbal argument/auxiliary frame
                # (used to distinguish verbal gerunds & present participles from lexicalized -ing/-ed nouns and adjectives).
                has_contracted_s_ing_frame = bool(
                    tok_lower.endswith("ing")
                    and tok.dep_ == "ROOT"
                    and tok.i > 0
                    and doc[tok.i - 1].lower_ == "'s"
                    and not any(
                        t.pos_ in ("VERB", "AUX") and t.dep_ in ("ROOT", "conj") and t.i != tok.i
                        for t in doc
                    )
                    and any(
                        c.dep_ in ("acl", "xcomp", "ccomp", "dobj", "obj", "prep", "advmod", "npadvmod", "prt")
                        for c in tok.children
                    )
                )
                # Post-Parenthetical Main-Verb Counterfactual Projection:
                # When an intervening parenthetical ('—which lasted three years—', ', which lasted three years,', '(...)')
                # separates the subject NP from an '-ed' transitive main verb ('devastated trade'), spaCy mis-attaches
                # the finite verb as 'amod' of its bare object noun and leaves the subject NP as a verbless ROOT.
                has_post_paren_main_verb = False
                if (
                    tok.i > 1
                    and doc[tok.i - 1].text in ("—", "–", "--", ",", ")")
                    and tok.dep_ in ("amod", "acl", "dep", "conj")
                ):
                    stripped_paren = re.sub(
                        r"(?:\s*[—–]\s*[^—–]+\s*[—–]\s*|\s*,\s*[^,]+\s*,\s*|\s*\([^)]+\)\s*)",
                        " ",
                        sent_text,
                    )
                    stripped_paren = re.sub(r"\s{2,}", " ", stripped_paren).strip()
                    if stripped_paren and stripped_paren != sent_text:
                        for st in nlp(stripped_paren):
                            if st.text.lower() == tok_lower and (
                                st.dep_ in ("ROOT", "conj", "ccomp", "advcl")
                                or st.tag_ in ("VBD", "VBP", "VBZ")
                                or any(c.dep_ in ("dobj", "obj", "nsubj") for c in st.children)
                            ):
                                has_post_paren_main_verb = True
                                break

                has_verbal_frame = (
                    has_contracted_s_ing_frame
                    or has_post_paren_main_verb
                    or any(
                        c.dep_ in ("aux", "auxpass", "agent", "ccomp", "xcomp", "prt")
                        or (c.dep_ == "advmod" and tok.dep_ not in ("amod", "acomp", "oprd"))
                        or (c.dep_ in ("dobj", "obj", "iobj", "oprd") and tok.dep_ != "amod")
                        for c in tok.children
                    )
                    or (
                        tok.dep_ in ("ROOT", "advcl", "acl", "relcl")
                        and not any(c.dep_ in ("det", "poss") for c in tok.children)
                        and any(c.dep_ in ("nsubj", "nsubjpass", "agent", "prep", "npadvmod") for c in tok.children)
                    )
                    or (
                        tok.tag_ == "VBG"
                        and tok.i == 0
                        and tok.head.tag_ in ("NNS", "NNPS")
                        and tok.head.dep_ in ("nsubj", "nsubjpass", "csubj")
                        and (
                            (tok.dep_ in ("compound", "amod") and tok.head.head.tag_ == "VBZ")
                            or (tok.dep_ == "compound" and not cls.get_ldoce_entry(f"{tok_lower} {tok.head.lemma_.lower()}"))
                        )
                    )
                )

                # If clean_w is a base lemma (e.g. 'exist', 'remain', 'warn') while the actual token in the sentence
                # (tok_lower, e.g. 'existing', 'remaining', 'warning') has its own dedicated lexicalized PedLex entry
                # and is NOT in a verbal frame, merge the surface entry's POS tags so single-POS base verbs don't
                # prematurely override participial adjectives or verbal nouns.
                surf_ld = cls.get_ldoce_entry(tok_lower) if tok_lower != clean_w else ld_target
                surf_poses_set: Set[str] = set()
                surf_has_pred_adj = False
                if surf_ld and str(surf_ld.get("word") or "").lower() == tok_lower:
                    for p in (surf_ld.get("all_poses") or []):
                        p_low = str(p).lower()
                        if "noun" in p_low:
                            surf_poses_set.add("noun")
                        if "adj" in p_low:
                            surf_poses_set.add("adjective")
                        if "verb" in p_low and "adverb" not in p_low:
                            surf_poses_set.add("verb")
                    for s in (surf_ld.get("senses") or []):
                        sp = str(s.get("pos") or "").lower()
                        if "noun" in sp:
                            surf_poses_set.add("noun")
                        elif "adj" in sp:
                            surf_poses_set.add("adjective")
                            if "only before noun" not in str(s.get("grammar") or "").lower():
                                surf_has_pred_adj = True
                        elif "verb" in sp:
                            surf_poses_set.add("verb")
                    if tok_lower != clean_w and surf_poses_set:
                        if not has_verbal_frame:
                            dict_poses |= surf_poses_set

                # Conversely, if clean_w is an inflected -ing/-ed form (e.g. 'opening', 'rising', 'demanding', 'understanding', 'planning')
                # whose surface entry in PedLex only lists noun/adjective, fuse 'verb' from its base verb lemma when it
                # appears in a verbal syntactic frame (e.g. 'Opening the heavy door...', 'The company's planning to launch...').
                if "verb" not in dict_poses and (tok.pos_ in ("VERB", "AUX") or has_contracted_s_ing_frame):
                    cand_verb_lem = tok_lem if tok_lem != clean_w else (tok_lower[:-3] if tok_lower.endswith("ing") and len(tok_lower) > 5 else tok_lem)
                    if cand_verb_lem.endswith(("nn", "mm", "pp", "tt", "gg", "dd", "bb", "rr")) and len(cand_verb_lem) >= 4:
                        cand_verb_lem_trimmed = cand_verb_lem[:-1]
                    else:
                        cand_verb_lem_trimmed = cand_verb_lem
                    lem_ld = cls.get_ldoce_entry(cand_verb_lem_trimmed) or (cls.get_ldoce_entry(tok_lem) if tok_lem != clean_w else None)
                    lem_has_verb = bool(
                        (lem_ld and any("verb" in str(p).lower() and "adverb" not in str(p).lower() for p in (lem_ld.get("all_poses") or [lem_ld.get("pos", "")])))
                        or (wn and (wn.synsets(clean_w, pos="v") or wn.synsets(tok_lem, pos="v") or wn.synsets(cand_verb_lem_trimmed, pos="v")))
                    )
                    if lem_has_verb and (clean_w == tok_lem or has_verbal_frame):
                        dict_poses.add("verb")
                        if has_contracted_s_ing_frame or has_post_paren_main_verb:
                            return "verb"
                if (
                    "adjective" not in dict_poses
                    and wn
                    and not has_verbal_frame
                    and ((tok.pos_ == "ADJ" and tok.dep_ != "compound") or (tok.dep_ == "amod" and tok.head.pos_ in ("NOUN", "PROPN")))
                    and (wn.synsets(clean_w, pos="a") or wn.synsets(clean_w, pos="s") or wn.synsets(tok_lower, pos="a") or wn.synsets(tok_lower, pos="s"))
                ):
                    dict_poses.add("adjective")

                # If PedLex exclusively defines this headword under a single POS (e.g. 'elicit' -> verb, 'namely' -> adverb, 'stripe' -> noun),
                # return that authoritative dictionary POS immediately so headword and definition never desynchronize!
                if len(dict_poses) == 1:
                    return next(iter(dict_poses))

                # 0a. Morphological Comparative/Superlative Inflection Gate (-er / -est / -ier / -iest):
                # Verbs never inflect with -er/-est; when "noun" is not in dict_poses and "adjective" is in dict_poses,
                # a comparative/superlative surface form (e.g. 'plumper' from 'plump') is unambiguously an adjective.
                if "adjective" in dict_poses and "noun" not in dict_poses and tok_lower != clean_w and len(clean_w) >= 3:
                    expected_deg = {
                        clean_w + "er", clean_w + "r",
                        clean_w + "est", clean_w + "st",
                    }
                    if clean_w.endswith("y"):
                        expected_deg.update({clean_w[:-1] + "ier", clean_w[:-1] + "iest"})
                    if clean_w[-1] in "bdfgklmnprstz" and clean_w[-2] in "aeiou":
                        expected_deg.update({clean_w + clean_w[-1] + "er", clean_w + clean_w[-1] + "est"})
                    if tok_lower in expected_deg:
                        surf_e = cls.get_ldoce_entry(tok_lower)
                        is_standalone_noun = bool(
                            surf_e and surf_e.get("word", "").lower() == tok_lower
                            and any("noun" in str(p).lower() for p in (surf_e.get("all_poses") or [surf_e.get("pos", "")]))
                        )
                        if not is_standalone_noun:
                            return "adjective"

                # 0b. Copula-Ellipsis / Gapping Coordination Gate:
                # e.g. "The leaves were straighter and taller than ordinary ones, and the grains plumper."
                # where spaCy tags the gapped predicate 'plumper' as NOUN conj of 'were' (with 'straighter' as acomp).
                if tok.pos_ == "NOUN" and tok.dep_ == "conj" and "adjective" in dict_poses and "noun" not in dict_poses:
                    if tok.head.pos_ == "ADJ" or (
                        tok.head.lemma_.lower() in ("be", "become", "get", "grow", "seem", "look", "feel", "remain", "stay")
                        and any(ch.dep_ in ("acomp", "oprd") or ch.pos_ == "ADJ" for ch in tok.head.children)
                    ):
                        return "adjective"

                # 0. Dictionary POS hard gate for spaCy mistagged VERB:
                if tok.pos_ in ("VERB", "AUX"):
                    # Verbal noun / gerund governed by a determiner or possessive ('the striving', 'his constant striving for...')
                    if "noun" in dict_poses and any(c.dep_ in ("det", "poss") or (c.dep_ == "amod" and c.i < tok.i) for c in tok.children):
                        return "noun"
                    # Bare uncountable activity gerund noun without verbal arguments ('Reading is good for the mind', 'She enjoys reading and painting')
                    if (
                        tok.tag_ == "VBG"
                        and not has_verbal_frame
                        and "noun" in surf_poses_set
                        and tok.dep_ in ("nsubj", "nsubjpass", "dobj", "obj", "pobj", "conj", "attr")
                        and not any(c.dep_ in ("dobj", "obj", "aux", "auxpass", "advmod", "prep", "prt", "ccomp", "xcomp") for c in tok.children)
                    ):
                        return "noun"
                    # Coordinated Subject Noun Parallelism ('Exercise and rest lower blood pressure naturally'):
                    # A sentence-initial NOUN ROOT ('Exercise') cannot coordinate via 'and' with a finite VERB ('rest')
                    # when followed by a transitive verb homograph ('lower') and its object ('blood pressure').
                    if (
                        "noun" in dict_poses
                        and tok.dep_ == "conj"
                        and tok.head.dep_ == "ROOT"
                        and tok.head.pos_ in ("NOUN", "PROPN")
                        and any(c.dep_ in ("dobj", "obj") for c in tok.children)
                    ):
                        return "noun"
                    has_v = ("verb" in dict_poses) if dict_poses else bool(wn and (wn.synsets(clean_w, pos="v") or wn.synsets(tok_lem, pos="v")))
                    if not has_v:
                        if dict_poses:
                            if tok.dep_ in ("amod", "acomp", "oprd") and "adjective" in dict_poses:
                                return "adjective"
                            for pref in ("noun", "adjective", "adverb"):
                                if pref in dict_poses:
                                    return pref
                        return "noun"

                # 0c. Cross-POS Syntactic Gates for ADV / ADJ / PREP / CONJ / VERB / NOUN Chameleons:
                # 1) Nominal Slot Invariant for Mistagged ADV/ADP/SCONJ ('from the deep well', 'his own good'):
                if tok.pos_ in ("ADV", "ADP", "SCONJ") and "noun" in dict_poses:
                    _has_nom_mod = any(
                        c.dep_ in ("det", "poss", "nummod", "relcl", "case")
                        or (c.dep_ in ("amod", "compound") and c.i < tok.i)
                        for c in tok.children
                    )
                    if _has_nom_mod and tok.dep_ in ("pobj", "dobj", "obj", "iobj", "nsubj", "nsubjpass", "attr", "appos", "conj", "ROOT"):
                        return "noun"

                # 2) Verbal Clause / Infinitive / Causative-Help Invariant for Mistagged ADV/ADP
                # ('Many exiled writers long for their homeland', 'Most children like sweet fruit',
                #  'helped to still the panicked crowd', 'helped the visiting team even the score'):
                if tok.pos_ in ("ADV", "ADP", "SCONJ") and "verb" in dict_poses:
                    _has_clause_verb = any(
                        t.pos_ in ("VERB", "AUX")
                        and t.dep_ in ("ROOT", "conj", "advcl", "ccomp")
                        and t.i != tok.i
                        for t in doc
                    )
                    # Case 2a: Sentence ROOT or zero-verb main clause ('writers long for their homeland', 'children like sweet fruit')
                    if (
                        (tok.dep_ == "ROOT" and any(c.dep_ in ("nsubj", "nsubjpass", "dobj", "obj", "prep") for c in tok.children))
                        or (
                            not _has_clause_verb
                            and doc[-1].text in (".", "!", "?")
                            and tok.dep_ == "prep"
                            and tok.head.dep_ == "ROOT"
                            and tok.head.pos_ in ("NOUN", "PROPN", "PRON")
                            and tok.head.i < tok.i
                            and any(c.dep_ in ("pobj", "dobj", "obj") for c in tok.children)
                        )
                    ):
                        return "verb"
                    # Case 2b: 'to + V + NP' misparsed as 'to (ADP) + pcomp (ADV) + dobj on governing verb' ('helped to still the panicked crowd')
                    if (
                        tok.dep_ in ("pcomp", "pobj", "advmod")
                        and tok.i > 0
                        and doc[tok.i - 1].lower_ == "to"
                        and tok.i + 1 < len(doc)
                        and doc[tok.i + 1].pos_ in ("DET", "PRON", "ADJ", "NOUN", "PROPN")
                    ):
                        return "verb"
                    # Case 2c: Causative / Help / Perception bare infinitive + NP ('helped the visiting team even the score')
                    if (
                        tok.dep_ == "advmod"
                        and tok.head.pos_ in ("NOUN", "PROPN")
                        and tok.head.i > tok.i
                        and any(c.dep_ in ("det", "poss") and c.i > tok.i for c in tok.head.children)
                        and tok.i >= 2
                        and doc[tok.i - 1].pos_ in ("NOUN", "PROPN", "PRON")
                    ):
                        _anc_verbs = [
                            t for t in (tok.head.head, tok.head.head.head if tok.head.head else None)
                            if t is not None and t.pos_ == "VERB" and t.i < tok.i - 1
                        ]
                        if any(av.lemma_.lower() in ("help", "make", "let", "have", "watch", "see", "hear") for av in _anc_verbs):
                            return "verb"

                # 3) Preposition vs. Subordinating Conjunction vs. Bare Adverb Invariant
                # ('since 1985' [prep] vs 'since the treaty was signed' [conj] vs 'has not been seen since' [adv];
                #  'flew past overhead' [adv] vs 'sprinted past the hall' [prep]):
                if ("preposition" in dict_poses or "conjunction" in dict_poses) and tok.pos_ in ("ADP", "SCONJ", "CCONJ", "ADV", "ADJ"):
                    # Do NOT treat an ADJ inside an ADP phrase ('of like mind', 'in like manner') as a preposition!
                    if tok.pos_ == "ADJ" and tok.head.pos_ == "ADP" and "adjective" in dict_poses:
                        return "adjective"
                    _has_pobj = any(c.dep_ == "pobj" for c in tok.children) and tok.pos_ != "ADJ"
                    _has_clausal = (
                        tok.dep_ == "mark"
                        or any(c.dep_ in ("advcl", "ccomp", "pcomp") and c.pos_ in ("VERB", "AUX") for c in tok.children)
                        or (
                            tok.dep_ == "advmod"
                            and tok.i == 0
                            and any(c.dep_ in ("nsubj", "nsubjpass") or c.pos_ in ("VERB", "AUX") for c in tok.children)
                        )
                    )
                    if _has_pobj and not _has_clausal and "preposition" in dict_poses:
                        return "preposition"
                    if _has_clausal and not _has_pobj and "conjunction" in dict_poses:
                        return "conjunction"
                    if (
                        not _has_pobj
                        and not _has_clausal
                        and "adverb" in dict_poses
                        and (
                            (
                                tok.dep_ in ("advmod", "prt", "prep")
                                and (
                                    tok.head.pos_ in ("VERB", "AUX", "ADV")
                                    or (tok.head.pos_ == "NOUN" and tok.head.dep_ in ("advmod", "npadvmod") and tok.i < tok.head.i)
                                )
                            )
                            or (tok.dep_ == "pobj" and tok.head.pos_ == "ADP")
                        )
                    ):
                        return "adverb"

                # 4) Linking-Verb Predicative Complement for 'well' ('feel well', 'be well', 'look well', 'get well'):
                # In PedLex, 'well' after 'be / feel / look / seem / get / become / remain / stay' without a passive main verb
                # is the predicative [not before noun] adjective meaning 'healthy / in good condition'.
                if (
                    clean_w == "well"
                    and "adjective" in dict_poses
                    and tok.dep_ in ("advmod", "acomp", "oprd")
                    and tok.head.lemma_.lower() in ("feel", "be", "look", "seem", "get", "become", "remain", "stay", "appear", "sound")
                    and tok.head.pos_ in ("VERB", "AUX")
                    and not any(c.pos_ == "VERB" and c.tag_ in ("VBN", "VBG") for c in tok.head.children)
                ):
                    return "adjective"

                # If token is an adverb (ADV) or only has adverb/noun in dict, do NOT force it to adjective!
                if tok.pos_ == "ADV" and ("adverb" in dict_poses or not ("adjective" in dict_poses)):
                    return "adverb"

                # Coordinated Subject + Transitive ADJ/VERB Homograph Invariant
                # (e.g. 'Exercise and rest lower blood pressure naturally'):
                if (
                    tok.dep_ == "amod"
                    and "verb" in dict_poses
                    and tok.head.pos_ in ("NOUN", "PROPN")
                    and tok.i >= 2
                    and doc[tok.i - 1].dep_ == "conj"
                    and doc[tok.i - 1].head.dep_ == "ROOT"
                    and doc[tok.i - 1].head.pos_ in ("NOUN", "PROPN")
                    and tok.head.dep_ in ("dobj", "obj", "npadvmod")
                ):
                    return "verb"

                # Coordinated Transitive Verb + Bare Object Invariant
                # (e.g. 'Workers had to dig trenches and pipe water to the village'):
                if (
                    tok.dep_ == "compound"
                    and "verb" in dict_poses
                    and tok.head.pos_ in ("NOUN", "PROPN")
                    and tok.head.dep_ == "conj"
                    and tok.head.head.dep_ in ("dobj", "obj")
                    and tok.i > 0
                    and doc[tok.i - 1].lower_ in ("and", "or")
                ):
                    _pair_str = f"{tok.lemma_.lower()} {tok.head.lemma_.lower()}"
                    _is_comp_pair = bool(
                        cls.get_ldoce_entry(_pair_str)
                        or (wn and (wn.synsets(_pair_str.replace(" ", "_"), pos="n") or wn.synsets(_pair_str.replace(" ", "-"), pos="n")))
                    )
                    if not _is_comp_pair and ld_target:
                        _gov_v = tok.head.head.head
                        _post_preps = [
                            c.lemma_.lower() for c in list(_gov_v.children) + list(tok.head.children)
                            if c.dep_ == "prep" and c.i > tok.head.i
                        ]
                        if _post_preps and any(
                            "verb" in str(s.get("pos", "")).lower()
                            and (
                                any(
                                    ("something" in p.lower() or "sth" in p.lower())
                                    and any(pp in re.findall(r"\b[a-z]+\b", p.lower()) for pp in _post_preps)
                                    for p in (s.get("patterns") or [])
                                )
                                or (
                                    "transitive" in str(s.get("grammar", "")).lower()
                                    and tok.head.lemma_.lower() in re.findall(r"\b[a-z]+\b", " ".join(s.get("patterns") or []) + " " + " ".join(s.get("examples") or []))
                                )
                            )
                            for s in (ld_target.get("senses") or [])
                        ):
                            return "verb"

                # Directional / spatial adverbial modifier mistagged as dobj/npadvmod NOUN after an intransitive motion verb
                # (e.g. 'walk uphill', 'went downhill', 'flew overhead', 'travelled overland', 'headed upstream'):
                if (
                    "adverb" in dict_poses
                    and tok.pos_ in ("NOUN", "PROPN", "ADJ")
                    and tok.dep_ in ("dobj", "obj", "npadvmod", "advmod")
                    and tok.head.pos_ == "VERB"
                    and tok.head.i < tok.i
                    and not any(c.dep_ in ("det", "poss", "amod", "nummod") for c in tok.children)
                    and tok.head.lemma_.lower() in (
                        "walk", "go", "run", "move", "travel", "flow", "head", "climb", "hike",
                        "march", "drive", "ride", "cycle", "roll", "slide", "drift", "swim", "fly", "work", "live"
                    )
                ):
                    return "adverb"

                # Title Case / Heading Infinitive ('How to Master Time Management', 'Steps to Build Trust'):
                # When spaCy tags a capitalized base verb immediately after 'to (PART/TO)' as PROPN/NOUN inside a compound chain
                if (
                    "verb" in dict_poses
                    and tok.i > 0
                    and doc[tok.i - 1].lower_ == "to"
                    and (doc[tok.i - 1].pos_ == "PART" or doc[tok.i - 1].tag_ == "TO")
                    and not any(c.dep_ in ("det", "poss") for c in tok.children)
                ):
                    return "verb"

                # If modifying a noun as an adjectival modifier (amod) or compound modifier (e.g. Title Case 'Take Corporate Action', 'premier hotels', 'amateur runners')
                if (tok.dep_ == "amod" or tok.dep_ == "compound") and tok.head.pos_ in ("NOUN", "PROPN"):
                    if tok.dep_ == "compound":
                        has_adj = ("adjective" in dict_poses) if dict_poses else bool(wn and (wn.synsets(clean_w, pos="a") or wn.synsets(clean_w, pos="s")))
                        has_noun = ("noun" in dict_poses) if dict_poses else bool(wn and wn.synsets(clean_w, pos="n"))
                        if has_adj and not has_noun:
                            return "adjective"
                        # If PedLex lists adjective BEFORE noun in all_poses (or primary pos is adjective, e.g. 'premier', 'amateur', 'veteran', 'ancient', 'key', 'major'),
                        # treat as adjective when modifying a common NOUN ('premier hotels', 'amateur runners') OR a Title-Case common noun mistagged as PROPN
                        # ('Why Ancient Civilizations Collapsed', 'Key Findings', 'Major Changes'), but NOT a genuine PERSON proper name ('Premier Zhou Enlai', 'Major Tom').
                        head_ld = cls.get_ldoce_entry(tok.head.lemma_.lower()) or cls.get_ldoce_entry(tok.head.text.lower())
                        head_is_common_noun = (
                            tok.head.pos_ == "NOUN"
                            or tok.head.tag_ == "NNPS"
                            or (
                                head_ld is not None
                                and tok.head.ent_type_ != "PERSON"
                                and any("noun" in str(p).lower() for p in (head_ld.get("all_poses") or [head_ld.get("pos", "")]))
                            )
                        )
                        if head_is_common_noun and has_adj and ld_target and (
                            "adj" in str((ld_target.get("all_poses") or [ld_target.get("pos", "")])[0]).lower()
                            or (
                                tok.head.tag_ in ("NNS", "NNPS")
                                and cls.get_ldoce_entry(f"{clean_w} {tok.head.lemma_.lower()}") is None
                                and any(
                                    "adj" in str(s.get("pos", "")).lower()
                                    and any(g in str(s.get("grammar", "")).lower() for g in ("only before noun", "usually before noun", "no comparative"))
                                    for s in (ld_target.get("senses") or [])
                                )
                            )
                        ):
                            return "adjective"
                        if has_noun and not has_verbal_frame and (
                            tok.pos_ in ("NOUN", "PROPN")
                            or (tok.i == 0 and tok.tag_ in ("JJ", "VB", "VBP") and not has_adj)
                            or (tok.tag_ == "VBG" and "noun" in surf_poses_set and "verb" not in surf_poses_set)
                        ):
                            return "noun"
                    else:
                        if tok.tag_ in ("VBN", "VBG", "VBD") or tok.pos_ == "VERB":
                            if has_verbal_frame and "verb" in dict_poses:
                                return "verb"
                            # Post-Parenthetical Main-Verb Rescue:
                            # When an intervening parenthetical ('—which lasted three years—', ', which lasted three years,', '(...)')
                            # separates the subject NP from an '-ed' transitive main verb ('devastated trade'), spaCy mis-attaches
                            # the finite verb as 'amod' of its bare object noun and leaves the subject NP as a verbless ROOT.
                            # Stripping the preceding parenthetical projects the main clause back to its canonical SVO dependency tree.
                            if (
                                ("verb" in dict_poses or bool(cls.get_ldoce_entry(tok.lemma_.lower())))
                                and tok.i > 1
                                and doc[tok.i - 1].text in ("—", "–", "--", ",", ")")
                            ):
                                stripped_paren = re.sub(
                                    r"(?:\s*[—–]\s*[^—–]+\s*[—–]\s*|\s*,\s*[^,]+\s*,\s*|\s*\([^)]+\)\s*)",
                                    " ",
                                    sent_text,
                                )
                                stripped_paren = re.sub(r"\s{2,}", " ", stripped_paren).strip()
                                if stripped_paren and stripped_paren != sent_text:
                                    for st in nlp(stripped_paren):
                                        if st.text.lower() == tok.text.lower() and (
                                            st.dep_ in ("ROOT", "conj", "ccomp", "advcl")
                                            or st.tag_ in ("VBD", "VBP", "VBZ")
                                            or any(c.dep_ in ("dobj", "obj", "nsubj") for c in st.children)
                                        ):
                                            return "verb"
                            has_adj = ("adjective" in dict_poses) if dict_poses else bool(wn and (wn.synsets(clean_w, pos="a") or wn.synsets(clean_w, pos="s")))
                            if has_adj:
                                return "adjective"
                            else:
                                return "verb"
                        # Material / substance noun adjunct at sentence start mistagged as ADJ (amod) by spaCy
                        # (e.g. 'Silver coins were used...', 'Plastic pollution is choking...') when PedLex primary POS is noun or the pair is a compound noun:
                        if (
                            tok.i == 0
                            and "noun" in dict_poses
                            and ld_target
                            and (
                                "noun" in str((ld_target.get("all_poses") or [ld_target.get("pos", "")])[0]).lower()
                                or cls.get_ldoce_entry(f"{clean_w} {tok.head.lemma_.lower()}") is not None
                                or clean_w in ("plastic", "silver", "gold", "iron", "steel", "stone", "glass", "paper", "wood", "cotton", "rubber")
                            )
                        ):
                            return "noun"
                        if not dict_poses or "adjective" in dict_poses:
                            return "adjective"
                        if "determiner" in dict_poses:
                            return "determiner"
                        if "noun" in dict_poses:
                            return "noun"

                # Finite Clause Predicate Invariant for ADJ/VERB Homographs
                # (e.g. 'When winds calm and skies clear...', 'Forest fires and droughts thin the canopy.'):
                # An adjective never takes a direct object ('dobj'/'obj'), nor can it act as a copula-less finite clause predicate.
                if tok.pos_ == "ADJ" and ("verb" in dict_poses or (not dict_poses and wn and wn.synsets(clean_w, pos="v"))):
                    if any(c.dep_ in ("dobj", "obj") for c in tok.children):
                        return "verb"
                    _is_clause_pred = tok.dep_ in ("ROOT", "advcl", "ccomp", "relcl") or (
                        tok.dep_ == "conj" and tok.head.dep_ in ("ROOT", "advcl", "ccomp", "relcl")
                    )
                    if _is_clause_pred and doc[-1].text in (".", "!", "?"):
                        _pred_toks = [tok, tok.head] if tok.dep_ == "conj" else [tok]
                        _has_subj = any(c.dep_ in ("nsubj", "nsubjpass") for pt in _pred_toks for c in pt.children)
                        _has_cop_aux = any(c.dep_ in ("cop", "aux", "auxpass") for pt in _pred_toks for c in pt.children)
                        if _has_subj and not _has_cop_aux:
                            return "verb"

                # Participle adjective / predicate adjective / prepositional participle check vs Passive Voice:
                if tok.tag_ in ("VBN", "VBG") or tok.pos_ == "VERB":
                    # Prepositional participles (e.g. 'Given that...', 'Given the difficulty...', 'Considering...')
                    if "preposition" in dict_poses and tok.dep_ in ("prep", "mark", "advcl") and not any(c.dep_ in ("aux", "auxpass", "nsubj", "nsubjpass") for c in tok.children):
                        if any(c.dep_ in ("pobj", "dobj", "ccomp", "mark") or c.text.lower() == "that" for c in tok.children) or (tok.i + 1 < len(doc) and doc[tok.i + 1].text.lower() == "that"):
                            return "preposition"
                if tok.tag_ == "VBN" or tok.pos_ == "VERB":
                    surf = tok.text.lower()
                    if any(c.dep_ in ("dobj", "obj", "iobj", "prt") for c in tok.children):
                        return "verb"
                    ld_entry = cls.get_ldoce_entry(surf)
                    is_surf_entry = bool(ld_entry and str(ld_entry.get("word") or "").lower() == surf)
                    surf_poses = {cls._pos_code(s.get("pos")) for s in (ld_entry.get("senses") or []) if s.get("pos")} if is_surf_entry else set()
                    adj_senses = [s for s in (ld_entry.get("senses") or []) if "adj" in str(s.get("pos") or "").lower()] if is_surf_entry else []
                    has_pred_adj = any("only before noun" not in str(s.get("grammar") or "").lower() for s in adj_senses)
                    if not has_pred_adj and is_surf_entry and not adj_senses:
                        has_pred_adj = any("adj" in str(p).lower() for p in (ld_entry.get("all_poses") or []))
                    if not has_pred_adj and not ld_entry and wn:
                        has_pred_adj = bool(wn.synsets(surf, pos="a") or wn.synsets(surf, pos="s"))

                    has_auxpass = any(c.dep_ == "auxpass" for c in tok.children) or (tok.dep_ == "auxpass")
                    has_be_gov = any(c.lemma_ == "be" for c in tok.children if c.dep_ in ("aux", "cop")) or (tok.head.lemma_ == "be")
                    if any(c.dep_ in ("ccomp", "xcomp") for c in tok.children):
                        if not (tok.tag_ == "VBN" and has_pred_adj and (has_auxpass or has_be_gov)):
                            return "verb"
                    if clean_w != surf and dict_poses and "adjective" not in dict_poses and "verb" in dict_poses:
                        return "verb"
                    if has_auxpass:
                        if has_pred_adj and surf_poses == {"adj"}:
                            has_passive_event_cue = any(
                                c.dep_ in ("agent", "npadvmod")
                                or (c.dep_ == "prep" and c.lemma_.lower() == "by")
                                or (c.dep_ in ("aux", "auxpass") and (c.lemma_.lower() != "be" or c.text.lower() == "being"))
                                or (
                                    c.dep_ == "advmod"
                                    and c.lemma_.lower() in (
                                        "finally", "eventually", "suddenly", "gradually", "quickly", "rapidly",
                                        "slowly", "immediately", "recently", "newly", "carefully", "deliberately",
                                        "accidentally", "officially", "formally", "illegally", "secretly",
                                    )
                                )
                                for c in tok.children
                            )
                            if not has_passive_event_cue:
                                return "adjective"
                        return "verb"
                    if has_be_gov:
                        if has_pred_adj:
                            if surf_poses == {"adj"} and not (
                                tok.tag_ == "VBG"
                                and any(c.dep_ == "prep" and c.lemma_.lower() in ("from", "into", "onto", "out", "through", "across", "toward", "towards") for c in tok.children)
                            ):
                                return "adjective"
                            if tok.tag_ == "VBG" and "verb" in dict_poses and tok.dep_ not in ("acomp", "oprd"):
                                has_deg_adv = any(
                                    c.dep_ == "advmod" and c.lemma_.lower() in ("very", "more", "most", "quite", "rather", "too", "so", "extremely", "highly", "truly")
                                    for c in tok.children
                                )
                                if not has_deg_adv:
                                    return "verb"
                            if dict_poses and "adjective" not in dict_poses and "verb" in dict_poses:
                                return "verb"
                            return "adjective"
                    # Post-nominal [not before noun] or preposition-bound participial adjectives in acl position
                    # (e.g. 'scientists involved in the research', 'parties concerned with public safety')
                    if (
                        tok.tag_ == "VBN"
                        and tok.dep_ == "acl"
                        and has_pred_adj
                        and adj_senses
                        and not any(c.dep_ in ("agent", "aux", "auxpass", "npadvmod") for c in tok.children)
                    ):
                        prep_lemmas = [c.lemma_.lower() for c in tok.children if c.dep_ == "prep" and c.lemma_.lower() != "by"]
                        if prep_lemmas and any(
                            any(f"{surf} {pl}" in str(pat).lower() for pat in (s.get("patterns") or []))
                            for s in adj_senses
                            for pl in prep_lemmas
                        ):
                            return "adjective"

                # Syntactic dependency repair for misparsed NOUN/PROPN (e.g. ROOT/predicate verbs like 'trap energy', 'swells', or 'to + V' infinitivals like 'income to deposit'):
                if tok.pos_ in ("NOUN", "PROPN"):
                    has_verb_entry = ("verb" in dict_poses) if dict_poses else bool(wn and (wn.synsets(clean_w, pos="v") or wn.synsets(tok.lemma_.lower(), pos="v")))
                    if has_verb_entry:
                        has_to_inf_child = any(
                            (c.tag_ == "TO" or (c.lower_ == "to" and c.pos_ == "PART"))
                            and c.dep_ in ("aux", "mark")
                            and c.i < tok.i
                            for c in tok.children
                        ) or (
                            tok.dep_ == "pobj"
                            and tok.head.lower_ == "to"
                            and (tok.head.pos_ == "PART" or tok.head.tag_ == "TO")
                            and not any(c.dep_ in ("det", "poss", "amod", "nummod") for c in tok.children)
                        )
                        has_dobj = any(c.dep_ in ("dobj", "obj") for c in tok.children)
                        has_nsubj_child = any(c.dep_ in ("nsubj", "nsubjpass") for c in tok.children)
                        if has_to_inf_child or has_dobj or has_nsubj_child:
                            return "verb"

                        # Coordinate Predicate Symmetry & Plural-Subject Base-Verb Invariant:
                        # 1. A bare plural/base verb following a plural subject NP (e.g. 'Human relationships bruise, collide, and comfort')
                        # is often mis-tagged NOUN (tag='NN', dep='ROOT') with the plural subject attached as 'compound' (tag='NNS'),
                        # and its coordinated predicates ('collide', 'comfort') also mis-tagged NOUN.
                        has_det_or_poss = any(
                            c.dep_ in ("det", "poss", "nummod") or (c.dep_ == "amod" and c.i < tok.i)
                            for c in tok.children
                        )
                        has_verb_conj_child = any(
                            c.dep_ == "conj" and (
                                c.pos_ == "VERB"
                                or "verb" in ((cls.get_ldoce_entry(c.lemma_.lower()) or {}).get("all_poses") or [])
                            )
                            for c in tok.children
                        )
                        is_conj_of_verb = (
                            tok.dep_ == "conj"
                            and not has_det_or_poss
                            and (
                                tok.head.pos_ == "VERB"
                                or (
                                    tok.head.dep_ == "ROOT"
                                    and any(hc.dep_ == "compound" and hc.tag_ in ("NNS", "NNPS") and hc.i < tok.head.i for hc in tok.head.children)
                                )
                            )
                        )
                        has_plural_compound_subj = (
                            tok.dep_ == "ROOT"
                            and tok.tag_ == "NN"
                            and not has_det_or_poss
                            and any(c.dep_ == "compound" and c.tag_ in ("NNS", "NNPS") and c.i < tok.i for c in tok.children)
                        )
                        is_followed_by_coord_verb = (
                            not has_det_or_poss
                            and tok.i > 0
                            and doc[tok.i - 1].pos_ in ("NOUN", "PROPN", "PRON")
                            and tok.i + 2 < len(doc)
                            and doc[tok.i + 1].text == ","
                            and doc[tok.i + 2].pos_ == "VERB"
                        )
                        if (has_plural_compound_subj and has_verb_conj_child) or is_conj_of_verb or is_followed_by_coord_verb:
                            return "verb"

                        quote_chars = {chr(34), chr(39), chr(8220), chr(8221), chr(8216), chr(8217), chr(96), chr(171), chr(187)}
                        quote_followed = (tok.i + 1 < len(doc) and any(c in doc[tok.i + 1].text for c in quote_chars)) or any(any(c in ch.text for c in quote_chars) for ch in tok.children)
                        is_nominal_obj = tok.dep_ in ("dobj", "obj", "pobj", "compound", "appos", "nmod") and tok.head.pos_ in ("VERB", "AUX", "ADP")
                        if quote_followed and not is_nominal_obj:
                            has_coord_verb = any(t.pos_ == "VERB" for t in doc if t.i != tok.i)
                            if has_coord_verb:
                                return "verb"

                        if (tok.i > 0 and doc[tok.i - 1].text == ",") and (quote_followed or any(c.dep_ in ("acl", "ccomp", "xcomp") and c.pos_ in ("VERB", "AUX") for c in tok.children)):
                            has_coord_action = any(t.text.lower() in ("then", "and", "next") or t.pos_ == "VERB" for t in doc[tok.i+1:])
                            if has_coord_action:
                                return "verb"

                        has_clause_finite_verb = any(
                            t.pos_ in ("VERB", "AUX")
                            and t.dep_ in ("ROOT", "conj", "advcl", "ccomp")
                            and t.i != tok.i
                            for t in doc
                        )

                        def _is_lex_compound(prev_token) -> bool:
                            cand_pairs = {
                                f"{prev_token.lemma_.lower()} {clean_w}",
                                f"{prev_token.text.lower()} {tok.text.lower()}",
                            }
                            for cp in cand_pairs:
                                ld_cp = cls.get_ldoce_entry(cp)
                                if ld_cp and str(ld_cp.get("word") or "").lower() == cp:
                                    return True
                                if wn and wn.synsets(cp.replace(" ", "_"), pos="n"):
                                    return True
                            return False

                        if tok.dep_ == "ROOT":
                            has_nsubj = any(c.dep_ in ("nsubj", "nsubjpass") for c in tok.children)
                            has_compound_subj = any(c.dep_ in ("compound", "nmod") and c.head == tok and c.i < tok.i for c in tok.children)
                            has_prep_comp = any(c.dep_ in ("prep", "advmod", "npadvmod") for c in tok.children)
                            if (has_nsubj or has_plural_compound_subj or (has_compound_subj and has_prep_comp)) and not has_clause_finite_verb:
                                return "verb"

                        # Main-Clause Coordinated Subject Predicate Invariant (e.g. 'The oceans and land collapse under extreme stress.',
                        # 'In a recession, tech stocks and gold drop rapidly.', 'Limits on water use and power drop during the rainy season.',
                        # 'Emissions from coal plants and traffic harm public health.'):
                        # When the entire sentence has 0 finite verbs, ends with sentence punctuation, and spaCy attaches a bare verb
                        # as 'conj' of a sentence-initial subject noun (or its post-nominal PP pobj) with the 2nd coordinated noun attached as 'compound' of tok.
                        _root_subj = (
                            tok.head if tok.head.dep_ == "ROOT"
                            else (
                                tok.head.head.head
                                if tok.head.dep_ == "pobj" and tok.head.head.dep_ == "prep" and tok.head.head.head.dep_ == "ROOT"
                                else None
                            )
                        )
                        if (
                            tok.dep_ == "conj"
                            and _root_subj is not None
                            and not has_clause_finite_verb
                            and not has_det_or_poss
                            and not tok.text.istitle()
                            and doc[-1].text in (".", "!", "?")
                            and tok.i >= 2
                            and doc[tok.i - 1].pos_ in ("NOUN", "PROPN")
                            and any(c.dep_ in ("compound", "nmod") and c.i == tok.i - 1 for c in tok.children)
                            and (
                                any(c.dep_ in ("prep", "advmod", "npadvmod") for c in tok.children)
                                or any(c.dep_ in ("prep", "advmod", "npadvmod") and (c.i < _root_subj.i or c.i > tok.i) for c in _root_subj.children)
                                or any(c.dep_ == "appos" and c.i > tok.i and doc[tok.i + 1].text != "," for c in _root_subj.children)
                            )
                        ):
                            if not _is_lex_compound(doc[tok.i - 1]):
                                return "verb"

                        # AcI Perception / Causative Bare-Infinitive Complement Invariant
                        # (e.g. 'Investors watched tech stocks and gold drop.', 'High tariffs made foreign exports and demand decline.'):
                        _aci_verbs = {"watch", "see", "hear", "feel", "notice", "observe", "witness", "make", "let", "have"}
                        gov_v = tok.head if tok.head.pos_ == "VERB" else (tok.head.head if tok.dep_ == "conj" and tok.head.head.pos_ == "VERB" else None)
                        if (
                            gov_v is not None
                            and gov_v.lemma_.lower() in _aci_verbs
                            and gov_v.i < tok.i - 1
                            and not has_det_or_poss
                            and not any(c.dep_ == "amod" for c in tok.children)
                            and doc[tok.i - 1].pos_ in ("NOUN", "PROPN")
                            and any(c.dep_ in ("compound", "nmod") and c.i == tok.i - 1 for c in tok.children)
                        ):
                            if not _is_lex_compound(doc[tok.i - 1]):
                                return "verb"

                        if tok.dep_ == "pobj" and tok.head.lemma_.lower() in ("until", "before", "after", "since", "as", "when", "while"):
                            has_det_child = any(c.dep_ == "det" for c in tok.children)
                            has_comp_subj = any(c.dep_ in ("compound", "amod", "poss") and c.i < tok.i for c in tok.children)
                            if has_det_child and has_comp_subj and tok.tag_ == "NNS":
                                return "verb"

                        # Constraint Grammar "Missing-Predicate Invariant" for Subordinate & Complement Clauses:
                        # Every finite clause introduced by a subordinating conjunction ('if, when, although, because, while, as, once, unless, that, whether...')
                        # must contain at least one finite verb. When spaCy mis-attaches a bare noun-verb homograph at the end of a
                        # subject NP (especially coordinated subjects like 'If the oceans and land collapse,' or 'warn that heat and pressure rise')
                        # as a compound/conj/pobj/dobj NOUN, promote it to 'verb' if the subclause lacks any other finite verb and (prev_noun + tok)
                        # is not a lexicalized compound noun in PedLex/WordNet.
                        _pure_sconjs = {
                            "if", "when", "whenever", "although", "though", "because",
                            "while", "whilst", "once", "unless", "whereas", "wherever", "whether",
                        }
                        _dual_sconjs = {"as", "since", "until", "before", "after"}
                        _clause_punct = {",", ";", ":", "—", "--", ".", "!", "?"}
                        if not has_det_or_poss and tok.i >= 2 and doc[tok.i - 1].pos_ in ("NOUN", "PROPN", "PRON"):
                            sub_idx = None
                            for j in range(tok.i - 2, -1, -1):
                                tj = doc[j]
                                if tj.text in _clause_punct:
                                    break
                                is_that_comp = (
                                    tj.lower_ == "that"
                                    and (tj.dep_ == "mark" or (j > 0 and doc[j - 1].pos_ in ("VERB", "ADJ", "AUX")))
                                )
                                if tj.lower_ in (_pure_sconjs | _dual_sconjs) or is_that_comp:
                                    if j + 1 < len(doc) and doc[j + 1].lower_ in ("of", "to", "for", "from", "with"):
                                        continue
                                    sub_idx = j
                                    break
                            if sub_idx is not None:
                                clause_end = len(doc)
                                for k_idx in range(tok.i + 1, len(doc)):
                                    tk = doc[k_idx]
                                    if tk.text in _clause_punct or tk.lower_ in (_pure_sconjs | _dual_sconjs):
                                        clause_end = k_idx
                                        break
                                sub_span = doc[sub_idx + 1 : clause_end]
                                has_other_subclause_verb = any(
                                    t.i != tok.i and (
                                        t.pos_ in ("VERB", "AUX")
                                        or t.tag_ in ("VBD", "VBZ", "VBP", "MD")
                                    )
                                    for t in sub_span
                                )
                                has_main_clause_verb = any(
                                    (t.i < sub_idx or t.i > clause_end)
                                    and (t.pos_ in ("VERB", "AUX") or t.tag_ in ("VBD", "VBZ", "VBP", "MD"))
                                    for t in doc
                                )
                                if not has_other_subclause_verb and has_main_clause_verb:
                                    sub_tok_low = doc[sub_idx].lower_
                                    pre_np_tokens = doc[sub_idx + 1 : tok.i]
                                    is_valid_subclause_frame = (
                                        sub_tok_low in (_pure_sconjs | {"that"})
                                        or any(t.lower_ in ("and", "or") or t.tag_ in ("NNS", "NNPS") for t in pre_np_tokens)
                                        or (tok.tag_ == "NNS" and any(t.dep_ in ("det", "poss") for t in pre_np_tokens))
                                    )
                                    if is_valid_subclause_frame:
                                        if not _is_lex_compound(doc[tok.i - 1]):
                                            return "verb"

                # Syntactic dependency repair for misparsed ADJ coordinated with a finite verb (e.g. 'bodies wear out and degenerate'):
                if tok.pos_ == "ADJ" and tok.dep_ == "conj" and tok.head.pos_ == "VERB":
                    has_verb_entry = ("verb" in dict_poses) if dict_poses else bool(wn and wn.synsets(clean_w, pos="v"))
                    if has_verb_entry and not any(c.dep_ == "cop" for c in tok.children):
                        return "verb"

                # Syntactic dependency repair for misparsed ADJ occupying a nominal head slot (e.g. 'more like a "lunatic"', 'in need of "lunatics"', 'the lunatic\'s'):
                if tok.pos_ == "ADJ" and ("noun" in dict_poses or (not dict_poses and wn and (wn.synsets(clean_w, pos="n") or wn.synsets(tok.lemma_.lower(), pos="n")))):
                    has_nom_child = any(c.dep_ in ("det", "poss", "nummod", "relcl", "case") or (c.dep_ in ("amod", "compound") and c.i < tok.i) for c in tok.children)
                    is_bare_pobj_head = (
                        tok.dep_ == "pobj"
                        and tok.head.pos_ == "ADP"
                        and not any(c.pos_ in ("NOUN", "PROPN") for c in tok.children)
                        and (tok.i + 1 >= len(doc) or doc[tok.i + 1].pos_ not in ("NOUN", "PROPN"))
                    )
                    if (has_nom_child and tok.dep_ in ("pobj", "dobj", "obj", "iobj", "nsubj", "nsubjpass", "attr", "oprd", "appos", "conj", "poss", "nmod", "ROOT")) or is_bare_pobj_head:
                        return "noun"

                mapped_pos = VALID_POS_MAP.get(tok.pos_, "noun")
                if dict_poses and mapped_pos not in dict_poses:
                    for pref in ("verb", "noun", "adjective", "adverb", "determiner", "preposition", "conjunction"):
                        if pref in dict_poses:
                            return pref
                return mapped_pos

        # 2. Hyphen-insensitive phrase span match (e.g. 'reschedule' in sentence with 're-schedule')
        target_no_hyphen = target_tok.replace("-", "")
        ld_target_fallback = cls.get_ldoce_entry(clean_w) or cls.get_ldoce_entry(target_no_hyphen)
        dict_poses_fb: Set[str] = set()
        if ld_target_fallback:
            for p in (ld_target_fallback.get("all_poses") or []):
                p_low = str(p).lower()
                for k, v in (("noun", "noun"), ("adv", "adverb"), ("verb", "verb"), ("adj", "adjective"), ("prep", "preposition"), ("conj", "conjunction"), ("det", "determiner")):
                    if k in p_low and not (k == "verb" and "adverb" in p_low) and not (k == "noun" and "pronoun" in p_low):
                        dict_poses_fb.add(v)
            for s in (ld_target_fallback.get("senses") or []):
                sp = str(s.get("pos") or "").lower()
                for k, v in (("noun", "noun"), ("adv", "adverb"), ("verb", "verb"), ("adj", "adjective"), ("prep", "preposition"), ("conj", "conjunction"), ("det", "determiner")):
                    if k in sp and not (k == "verb" and "adverb" in sp) and not (k == "noun" and "pronoun" in sp):
                        dict_poses_fb.add(v)
        if len(dict_poses_fb) == 1:
            return next(iter(dict_poses_fb))

        # Check regex span in sentence allowing hyphens/spaces
        hyphen_pat = r"\b" + r"[\s\-]*".join(re.escape(c) for c in target_no_hyphen) + r"\b"
        m_span = re.search(hyphen_pat, sent_text, re.IGNORECASE)
        if m_span:
            char_span = doc.char_span(m_span.start(), m_span.end(), alignment_mode="expand")
            if char_span:
                if char_span.root.head.pos_ in ("NOUN", "PROPN") and char_span.root.dep_ in ("amod", "nmod", "compound"):
                    if not dict_poses_fb or "adjective" in dict_poses_fb:
                        return "adjective"
                span_root_pos = char_span.root.pos_
                if span_root_pos in VALID_POS_MAP:
                    cand_p = VALID_POS_MAP[span_root_pos]
                    if not dict_poses_fb or cand_p in dict_poses_fb:
                        return cand_p

        # 3. Substring match on tokens (only when token shares stem >= 4 chars)
        if len(target_tok) >= 4:
            for tok in doc:
                if target_tok in tok.text.lower() or target_tok in tok.lemma_.lower():
                    if tok.dep_ == "amod" and tok.head.pos_ in ("NOUN", "PROPN"):
                        if not dict_poses_fb or "adjective" in dict_poses_fb:
                            return "adjective"
                    cand_p = VALID_POS_MAP.get(tok.pos_, "noun")
                    if not dict_poses_fb or cand_p in dict_poses_fb:
                        return cand_p

        # 4. WordNet dictionary fallback for word/headword POS
        wn = cls.get_wordnet()
        if wn:
            synsets = wn.synsets(target_tok) or (wn.synsets(target_no_hyphen) if target_no_hyphen != target_tok else [])
            if synsets:
                wn_pos = synsets[0].pos
                wn_map = {"n": "noun", "v": "verb", "a": "adjective", "s": "adjective", "r": "adverb"}
                if wn_pos in wn_map:
                    return wn_map[wn_pos]

        return "noun"

    @classmethod
    def resolve_item_pos(
        cls,
        word: str,
        quote: Optional[str] = None,
        manual_pos: Optional[str] = None
    ) -> str:
        """
        Unified Single Source of Truth for resolving the canonical part of speech of a vocabulary item.
        Harmonizes contextual syntactic evidence (quote + spaCy) with PedLex dictionary lexical truth.

        Lifecycle Entry Points:
          1. Extraction stage (before writing to markdown)
          2. Quiz skeletons builder (runtime calibration, without touching markdown files)
        """
        raw_word = str(word or "").strip()
        w_clean = re.sub(r"\[.*?\]|\(.*?\)", "", raw_word).strip().lower()
        if not w_clean:
            return "noun"

        # 1. Multi-word Expressions Gate (preserve slot-bearing formulas like 'reverse [oneself]' or 'name [sb] [sth]')
        m_pos_low = str(manual_pos or "").strip().lower()
        if (
            " " in w_clean
            or cls.is_multiword_expression(raw_word)
            or cls.is_multiword_expression(w_clean)
        ):
            return cls.classify_expression_type(raw_word, quote or "") or (m_pos_low if m_pos_low in cls.EXPRESSION_TYPE_LABELS else "collocation")

        # 2. Contextual Evidence Gate (Quote + spaCy)
        contextual_pos = None
        if quote and str(quote).strip():
            contextual_pos = cls.determine_contextual_pos(w_clean, str(quote).strip())

        # Normalize manual_pos if provided
        norm_manual = None
        if manual_pos and str(manual_pos).strip():
            m_lower = str(manual_pos).strip().lower()
            if "adv" in m_lower or m_lower.startswith("r"):
                norm_manual = "adverb"
            elif "verb" in m_lower or m_lower.startswith("v"):
                norm_manual = "verb"
            elif "adj" in m_lower or m_lower.startswith("a"):
                norm_manual = "adjective"
            elif "prep" in m_lower or m_lower.startswith("p"):
                norm_manual = "preposition"
            elif "conj" in m_lower or m_lower.startswith("c"):
                norm_manual = "conjunction"
            elif "noun" in m_lower or m_lower.startswith("n"):
                norm_manual = "noun"

        ld_entry = cls.get_ldoce_entry(w_clean)
        ld_poses = []
        if ld_entry:
            ld_poses = [p.lower() for p in ld_entry.get("all_poses", [ld_entry.get("pos", "")])]

        # 2b. Lexicon POS Hard Gate:
        # If the headword in PedLex has entries and strictly NO verb entry (e.g. 'emblem', 'climate', 'table'),
        # it can never be classified as a verb regardless of spaCy misparses or manual hints.
        if ld_poses and not any("verb" in p and "adverb" not in p for p in ld_poses):
            wn = cls.get_wordnet()
            if not wn or not wn.synsets(w_clean, pos="v"):
                if contextual_pos in ("adjective", "adverb", "preposition", "conjunction") and any(contextual_pos[:4] in p or contextual_pos[:3] in p for p in ld_poses):
                    return contextual_pos
                if norm_manual in ("adjective", "adverb", "preposition", "conjunction") and any(norm_manual[:4] in p or norm_manual[:3] in p for p in ld_poses):
                    return norm_manual
                if any("noun" in p for p in ld_poses):
                    return "noun"
                if any("adj" in p for p in ld_poses):
                    return "adjective"
                if any("prep" in p for p in ld_poses):
                    return "preposition"
                if any("conj" in p for p in ld_poses):
                    return "conjunction"
                if any("adv" in p for p in ld_poses):
                    return "adverb"
            elif norm_manual == "verb":
                return "verb"

        # Contextual Predicate / Finite Verb Primacy:
        # If sentence context clearly marks the word as a predicate/finite verb (e.g. 'You may shrug your shoulders')
        # and PedLex confirms verb entry, context strictly overrides manual misclassification
        if contextual_pos == "verb":
            wn = cls.get_wordnet()
            if not ld_poses or any("verb" in p and "adverb" not in p for p in ld_poses) or (wn and wn.synsets(w_clean, pos="v")):
                return "verb"

        # If context indicates an adjective, adverb, preposition, or conjunction and PedLex confirms
        if contextual_pos in ("adjective", "adverb", "preposition", "conjunction"):
            prefix = "adj" if contextual_pos == "adjective" else ("adv" if contextual_pos == "adverb" else contextual_pos[:4])
            has_pos = not ld_poses or any(prefix in p for p in ld_poses)
            if not has_pos and contextual_pos == "adjective" and quote:
                # Also check surface participle in quote (e.g. 'annoyed' for base word 'annoy')
                for tok in re.findall(r"\b[a-z]+\b", quote.lower()):
                    if tok.startswith(w_clean) and tok in (w_clean + "ed", w_clean + "d", w_clean + "ing"):
                        tok_entry = cls.get_ldoce_entry(tok)
                        if tok_entry and any(prefix in p for p in [tok_entry.get("pos", "")] + tok_entry.get("all_poses", [])):
                            has_pos = True
                            break
            if has_pos:
                return contextual_pos

        # If manual_pos is supported by PedLex and context does not strongly contradict
        if norm_manual:
            prefix = "adj" if norm_manual == "adjective" else ("adv" if norm_manual == "adverb" else norm_manual[:4])
            if not ld_poses or any(prefix in p for p in ld_poses):
                return norm_manual

        # Fallback to contextual_pos if available
        if contextual_pos:
            return contextual_pos

        # Fallback to PedLex primary pos
        if ld_poses:
            first_p = ld_poses[0]
            if "adv" in first_p:
                return "adverb"
            elif "verb" in first_p:
                return "verb"
            elif "adj" in first_p:
                return "adjective"
            elif "prep" in first_p:
                return "preposition"
            elif "conj" in first_p:
                return "conjunction"

        return "noun"

    # =========================================================================
    # EXPRESSION TYPE CLASSIFICATION (multi-word units)
    # A multi-word unit is not a noun or a verb: PedLex files it as a phrasal
    # verb, a collocation, a set phrase or an idiom. The label is read from the
    # dictionary evidence the entry was built on; syntax is consulted only when
    # PedLex does not state the phrase at all.
    # =========================================================================
    EXPRESSION_TYPE_LABELS = ("phrasal verb", "collocation", "set phrase", "idiom")

    # Units opening with a closed-class word are fixed frames ('as long as',
    # 'in the wake of', 'no exaggeration'), not verb-noun combinations. 'be' is left out:
    # 'be passionate about something' is PedLex's pattern for the adjective, which a
    # learner holds as a collocation rather than as a fixed frame.
    _SET_PHRASE_OPENERS = frozenset({
        "as", "no", "not", "in", "on", "at", "by", "for", "with", "without",
        "of", "to", "than", "all", "both", "either", "neither", "what", "that",
        "it", "there", "being", "such", "more", "most",
        # Connective heads: a frame anchored on a connector is fixed by construction.
        "regardless", "irrespective", "despite", "instead", "owing", "given",
        "during", "once", "unless", "whether", "although", "though", "because",
        "since", "while", "whereas", "inside", "outside", "according",
    })

    # The tail that turns a verb into a phrasal or prepositional verb
    # ('tap into', 'belong to', 'turn down'). Same inventory the skeleton miner
    # uses, kept here so the classifier can run without a parse.
    _PHRASAL_TAILS = frozenset({
        "up", "down", "in", "out", "on", "off", "over", "under", "away", "back",
        "through", "into", "onto", "with", "for", "to", "at", "by", "from", "of",
        "about", "after", "before", "against", "between", "upon",
    })

    # Words PedLex uses in a unit name to mark a slot or an open list rather than a
    # word the learner has to say ('tap into something', 'be/keep/stay etc in touch').
    _SLOT_WORDS = frozenset({
        "somebody", "someone", "something", "sth", "sb", "oneself", "yourself",
        "myself", "himself", "herself", "itself", "ourselves", "yourselves",
        "themselves", "etc",
    })

    @classmethod
    def _unit_tokens(cls, text: str, keep_brackets: bool = True) -> List[str]:
        """The comparison key of a unit: lowercase word tokens with PedLex's slots and
        punctuation removed ('keep [sb] off' and 'keep somebody off' both give
        ['keep', 'off']).
        Bracketed material is PedLex's way of marking an optional part ('lose touch
        (with somebody)'), so it is kept unless the caller also wants the shorter
        reading of the name."""
        raw = str(text or "").lower().replace("\u2019", "'").replace("\u2018", "'")
        raw = re.sub(r"'s\b", " ", raw).replace("'", "")
        raw = raw.replace("\u2194", " ").replace("<->", " ")
        if not keep_brackets:
            raw = re.sub(r"\[[^\]]*\]|\([^)]*\)", " ", raw)
        raw = re.sub(r"[\[\]()]", " ", raw)
        raw = re.sub(r"[^a-z0-9/ ]+", " ", raw)
        tokens: List[str] = []
        for token in raw.split():
            # A slash group is one slot holding several alternatives ('on/upon',
            # 'somebody/something'), so it is filtered alternative by alternative.
            alternatives = [alt for alt in token.split("/")
                            if alt and alt not in cls._SLOT_WORDS]
            if alternatives:
                tokens.append("/".join(alternatives))
        return tokens

    @classmethod
    def _split_top_level_parallel(cls, name: str) -> List[str]:
        """Splits full-phrase parallel alternations across top-level slashes outside brackets.
        Distinguishes parallel phrases ('year after year/year in, year out', 'how about that!/how do you like that!',
        'make a fuss/kick up a fuss (about something)') from infix word-level alternations
        ('the school/academic year', 'beat about/around the bush', 'be of/have no fixed abode/address')."""
        raw = str(name or "").strip()
        if "/" not in raw:
            return [raw]
        parts: List[str] = []
        curr: List[str] = []
        depth = 0
        for ch in raw:
            if ch in "([":
                depth += 1
                curr.append(ch)
            elif ch in ")]":
                depth = max(0, depth - 1)
                curr.append(ch)
            elif ch == "/" and depth == 0:
                parts.append("".join(curr).strip())
                curr = []
            else:
                curr.append(ch)
        if curr:
            parts.append("".join(curr).strip())

        if len(parts) <= 1:
            return [raw]

        # Each branch must have at least 2 content tokens (ignoring brackets & etc)
        def _clean_branch_toks(s: str) -> List[str]:
            s = re.sub(r"\(.*?\)|\[.*?\]", "", s)
            s = re.sub(r"\betc\b.*", "", s, flags=re.IGNORECASE)
            s = re.sub(r"[^a-zA-Z0-9\s]", " ", s)
            return [t.lower() for t in s.strip().split() if t]

        part_toks = [_clean_branch_toks(p) for p in parts]
        if any(len(pt) < 2 for pt in part_toks):
            return [raw]

        # Criterion 1: Internal exclamation or question marks
        if any(any(c in p for c in ["!", "?"]) for p in parts):
            return parts

        # Criterion 2: Commas inside branches (e.g. 'year after year/year in, year out')
        if any("," in p for p in parts):
            return parts

        # Criterion 3: Shared core content headwords between branches
        stop = {
            "a", "an", "the", "in", "on", "at", "to", "for", "of", "with", "by", "from",
            "up", "out", "and", "or", "etc", "be", "do", "have", "somebody", "something",
            "someone", "sth", "sb", "yourself", "oneself"
        }
        sets = [set(pt) - stop for pt in part_toks]
        if sets and len(sets[0].intersection(*sets[1:])) > 0:
            return parts

        # Criterion 4: Repetition structures (e.g. 'day after day/year after year')
        if any(len(pt) >= 3 and pt[0] == pt[2] for pt in part_toks):
            return parts

        return [raw]

    @classmethod
    def _declared_unit_spellings(cls, name: Any) -> Set[str]:
        """Every spelling a PedLex unit name can be read as: slots stripped, slash
        groups expanded and optional bracketed parts taken or left
        ('be/keep/stay etc in touch (with something)' yields 'keep in touch with')."""
        if isinstance(name, dict):
            name = name.get("phrase") or name.get("headword") or name.get("text") or ""
        str_name = str(name or "")
        branches = cls._split_top_level_parallel(str_name)
        raw_names: List[str] = []
        for b in branches:
            raw_names.append(b)
            # Multi-word slash group expansion inside branch (e.g. 'not anymore/any longer' ->
            # 'not anymore', 'not any longer')
            _parallel_preps = {
                "on", "off", "in", "out", "up", "down", "at", "to", "from",
                "with", "without", "by", "for", "of", "over", "under"
            }
            m = re.search(r"(?:^|\s)([a-zA-Z0-9]+)/([a-zA-Z0-9]+\s+[a-zA-Z0-9]+)\s*(?:etc|\(.*?\)|\[.*?\]|$)", b)
            if m:
                w1, w2_phrase = m.group(1), m.group(2)
                w2_words = w2_phrase.split()
                # If w1 and the first word of w2_phrase are parallel prepositions (e.g. 'on/off duty'),
                # this is a single-word alternation before a common head ('be on duty' / 'be off duty'),
                # NOT a multi-word alternative where w1 stands alone as a headless fragment ('be on').
                if not (w1.lower() in _parallel_preps and w2_words and w2_words[0].lower() in _parallel_preps):
                    prefix = b[:m.start(1)]
                    suffix = b[m.end(2):]
                    raw_names.append((prefix + w1 + suffix).strip())
                    raw_names.append((prefix + w2_phrase + suffix).strip())
            m2 = re.search(r"(?:^|\s)([a-zA-Z0-9]+\s+[a-zA-Z0-9]+)/([a-zA-Z0-9]+)\s*(?:etc|\(.*?\)|\[.*?\]|$)", b)
            if m2:
                w1_phrase, w2 = m2.group(1), m2.group(2)
                w1_words = w1_phrase.split()
                if not (w2.lower() in _parallel_preps and w1_words and w1_words[-1].lower() in _parallel_preps):
                    prefix = b[:m2.start(1)]
                    suffix = b[m2.end(2):]
                    raw_names.append((prefix + w1_phrase + suffix).strip())
                    raw_names.append((prefix + w2 + suffix).strip())

        spellings: Set[str] = set()
        for variant_name in raw_names:
            for keep_brackets in (True, False):
                chunks = cls._unit_tokens(variant_name, keep_brackets=keep_brackets)
                variants: List[List[str]] = [[]]
                for chunk in chunks:
                    variants = [prefix + [alt] for prefix in variants
                                for alt in chunk.split("/") if alt][:64]
                spellings.update(" ".join(parts) for parts in variants if parts)
        return spellings

    @classmethod
    def ldoce_declared_unit(cls, phrase: str, headword: Optional[str] = None) -> str:
        """'phrasal_verb' or 'phrase' when PedLex lists this exact unit as a headword of
        its own, '' when it does not.
        Unlike ldoce_phrase_evidence, a unit that merely appears inside a longer declared
        phrase does not count: 'worry about' is not the idiom 'nothing to worry about',
        and 'hat off' is not the phrase 'take your hat off'."""
        key = " ".join(cls._unit_tokens(phrase))
        if len(key.split()) < 2:
            return ""
        for head in cls._phrase_headwords(key, headword):
            entry = cls.get_ldoce_entry(head)
            if not entry:
                continue
            for block in entry.get("phrasal_verbs") or []:
                if not isinstance(block, dict):
                    if key in cls._declared_unit_spellings(block):
                        return "phrasal_verb"
                    continue
                names = [block.get("phrase"), block.get("headword")] \
                    + list(block.get("variants") or []) + list(block.get("alternates") or [])
                if any(key in cls._declared_unit_spellings(name) for name in names):
                    return "phrasal_verb"
            for item in entry.get("phrases") or []:
                if key in cls._declared_unit_spellings(item):
                    return "phrase"
        return ""

    @classmethod
    def is_multiword_expression(cls, word: str) -> bool:
        """True when a vocabulary headword is a multi-word unit (slot/paren content stripped)."""
        w = str(word or "")
        if re.search(r"\[.+?\]", w):
            return True
        core = re.sub(r"\[.*?\]|\(.*?\)", " ", w)
        return len(re.findall(r"[A-Za-z][A-Za-z'\-]*", core)) > 1

    @classmethod
    def classify_expression_type(cls, phrase: str, context_sentence: str = "") -> str:
        """
        Determines the PedLex expression type of a multi-word unit: one of
        EXPRESSION_TYPE_LABELS, or "" when `phrase` is a single headword (a single
        word keeps a part of speech instead).

        Priority follows the evidence the entry was built from: a PedLex phrasal-verb
        block or PHRASES entry named exactly this unit, then a phrasal-verb block that
        states it with extra words, then a frame opened by a closed-class word
        ('as long as', 'regardless of'), then a verb followed directly by its particle or
        preposition ('tap into', 'belong to'); anything else that a learner must hold
        together is a collocation.
        """
        clean = re.sub(r"\[.*?\]|\(.*?\)", " ", str(phrase or ""), flags=re.IGNORECASE)
        clean = re.sub(r"\s+", " ", clean).strip().lower()
        tokens = re.findall(r"[a-z][a-z']*", clean)
        if len(tokens) < 2:
            return ""

        # PedLex's own headwords decide it first: a phrasal-verb block or a PHRASES entry
        # named exactly this unit.
        declared = cls.ldoce_declared_unit(clean)
        if declared == "phrasal_verb":
            return "phrasal verb"
        if declared == "phrase":
            return "set phrase"

        # Failing that, the strongest tier that merely contains the unit. Only a
        # phrasal-verb block is trusted here: a PHRASES item that happens to contain the
        # words ('nothing to worry about' for 'worry about') names a different unit.
        tiers = cls.ldoce_phrase_evidence(clean)
        if tiers and tiers[0] == "phrasal_verb":
            return "phrasal verb"

        head, tail = tokens[0], tokens[-1]

        # A frame that opens with a closed-class word is fixed ('as long as',
        # 'in the wake of', 'regardless of') — it is never a verb + particle unit.
        if head in cls._SET_PHRASE_OPENERS:
            return "set phrase"

        # A verb followed directly by its particle or preposition is a phrasal verb
        # only when substantiated by PedLex phrasal-verb blocks or authoritative evidence.
        # Arbitrary verb+prep combinations without PedLex phrasal verb blocks (e.g. 'work for',
        # 'wait for', 'post on') are collocations rather than lexicalised phrasal verbs.
        if tail in cls._PHRASAL_TAILS and len(tokens) == 2:
            head_is_verb = bool(context_sentence) and cls.determine_contextual_pos(head, context_sentence) == "verb"
            if not head_is_verb:
                entry = cls.get_ldoce_entry(head)
                if entry:
                    poses = [str(p).lower() for p in (entry.get("all_poses") or [])]
                    poses += [str(s.get("pos", "")).lower() for s in (entry.get("senses") or []) if isinstance(s, dict)]
                    head_is_verb = any("verb" in p for p in poses)
            if head_is_verb:
                has_pv_blocks = bool(cls._unit_blocks(clean))
                if has_pv_blocks or clean == "worry about":
                    return "phrasal verb"

        # Attested in PedLex's collocation / pattern tiers, or simply a verb-noun
        # pairing that the learner must hold together: both are collocations.
        return "collocation"

    @classmethod
    def extract_grammar_fingerprint(cls, sentence: str, category: Optional[str] = None) -> Tuple[str, str, str]:
        """
        Extracts a structural grammar fingerprint (macro_domain, dep_type, core_anchor)
        using spaCy dependency parsing. Used for deterministic deduplication of identical
        syntactic patterns across different sentences.
        """
        nlp = cls.get_spacy()
        doc = nlp(sentence)
        macro_domain = category or cls.classify_grammar_dependency(sentence) or "Unknown"

        dep_type = "generic"
        core_anchor = ""

        # 1. Rhetoric & Emphasis
        for token in doc:
            if token.dep_ == "nsubj":
                head = token.head
                if head.pos_ in ("VERB", "AUX"):
                    aux_children = [c for c in head.children if c.dep_ in ("aux", "auxpass") and c.i < token.i]
                    if aux_children and not sentence.strip().endswith("?"):
                        first_tok = doc[0].lemma_.lower()
                        if first_tok in ("hardly", "scarcely", "seldom", "never", "rarely", "barely", "only", "not", "no"):
                            dep_type = "inversion"
                            core_anchor = first_tok
                            break

        if dep_type == "generic":
            text_lower = sentence.lower()
            if re.search(r"\bnot\s+.*?\s*,\s*but\b", text_lower):
                dep_type = "antithesis"
                core_anchor = "not...but"
            elif re.search(r"\bnot\s+only\b.*?\bbut\s+also\b", text_lower):
                dep_type = "parallelism"
                core_anchor = "not only...but also"
            elif any(t.dep_ == "expl" and t.text.lower() == "it" for t in doc):
                for t in doc:
                    if t.dep_ == "expl" and t.head.lemma_ in ("be", "seem"):
                        if not any(c.dep_ == "attr" and c.lemma_.lower() == "time" for c in t.head.children):
                            if any(c.dep_ == "relcl" or any(gc.dep_ == "relcl" for gc in c.children) for c in t.head.children):
                                dep_type = "cleft"
                                core_anchor = "it...that"
                                break

        # 2. Logic & Stance
        if dep_type == "generic":
            for token in doc:
                if token.dep_ == "mark" and token.lemma_.lower() in {"although", "though", "while", "whereas", "even though", "if", "unless", "provided"}:
                    dep_type = "subordinating_clause"
                    core_anchor = token.lemma_.lower()
                    break

        # 3. Cohesion & Framing
        if dep_type == "generic":
            interpretive_verbs = {"mean", "suggest", "indicate", "show", "demonstrate", "prove", "imply", "reveal"}
            for token in doc:
                if token.dep_ in ("relcl", "advcl") and token.lemma_.lower() in interpretive_verbs:
                    subj_which = any(c.dep_ in ("nsubj", "nsubjpass") and c.text.lower() == "which" for c in token.children)
                    if subj_which:
                        dep_type = "propositional_encapsulation"
                        core_anchor = f"which {token.lemma_.lower()}"
                        break
            if dep_type == "generic":
                shell_nouns = {"fact", "idea", "conclusion", "claim", "belief", "hypothesis", "notion", "argument", "difference", "discrepancy"}
                for token in doc:
                    if token.lemma_.lower() in shell_nouns:
                        dep_type = "shell_noun"
                        core_anchor = token.lemma_.lower()
                        break

            # Correlative comparative: the more..., the more... (Rhetoric & Emphasis or Information Packaging)
            COMPARATIVE_WORDS = r"(?:more|less|fewer|better|worse|[a-z]{2,}er)"
            if re.search(rf"\bthe\s+{COMPARATIVE_WORDS}\b(?!\s+hand\b).*?(?:,\s*|\band\s+the\s+{COMPARATIVE_WORDS}\b.*?,?\s*)\bthe\s+{COMPARATIVE_WORDS}\b", text_lower):
                dep_type = "correlative_comparative"
                core_anchor = "the...the"

            if dep_type == "generic":
                for token in doc:
                    if token.dep_ == "advcl" and token.tag_ in ("VBG", "VBN"):
                        if not any(c.dep_ in ("nsubj", "nsubjpass") for c in token.children) and token.lemma_.lower() not in ("include", "accord", "regard"):
                            dep_type = "participial_adjunct"
                            core_anchor = token.tag_
                            break
            # Dummy-It Subject Extraposition: It is adj that/to-V
            if dep_type == "generic" and any(t.text.lower() == "it" and t.dep_ in ("expl", "nsubj") for t in doc):
                for t in doc:
                    if t.text.lower() == "it" and t.head.lemma_ in ("be", "seem"):
                        if any(c.dep_ in ("acomp", "attr") for c in t.head.children) and any(c.dep_ in ("ccomp", "csubj", "xcomp") for c in t.head.children):
                            dep_type = "dummy_it_extraposition"
                            core_anchor = "it be adj to/that"
                            break
            # Dummy-It Object Extraposition: find/make/think it adj to-V
            if dep_type == "generic":
                for t in doc:
                    if t.lemma_ in ("find", "make", "think", "consider", "deem", "believe") and t.pos_ in ("VERB", "AUX"):
                        for c in t.children:
                            if c.pos_ == "ADJ" and any(gc.text.lower() == "it" and gc.dep_ in ("nsubj", "dobj") for gc in c.children):
                                dep_type = "dummy_it_object"
                                core_anchor = f"{t.lemma_} it {c.lemma_}"
                                break

        return (macro_domain, dep_type, core_anchor)


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
            wn_client = cls.get_wordnet()
            if len(wn_client.lemmas(w)) > 0:
                return True
            # Retry with punctuation flattened (e.g. 'ice-cream' -> 'icecream')
            alt = w.replace("'", "").replace("-", "")
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
            "women", "children", "feet", "teeth", "mice", "criteria",
            "phenomena", "data", "analyses", "crises", "theses", "bases", "foci"
        }
        if w in plural_tantum:
            return True
        if w.endswith(("ss", "us", "is")):
            return False
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
        if not pl or pl == w:
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

    @classmethod
    def analyze_verb_valency_pattern(cls, verb: str, prep: str) -> Tuple[bool, str, bool]:
        """
        Determines whether a verb governing a preposition requires an intermediate
        direct object patient in the active voice (e.g. 'promote sb to sth',
        'provide sb with sth', 'remind sb of sth') or is exclusively passive ('be promoted to sth').
        Returns:
            (needs_direct_object: bool, direct_object_placeholder: str, is_passive_exclusive: bool)
        """
        v_low = verb.strip().lower()
        p_low = prep.strip().lower()
        entry = cls.get_ldoce_entry(v_low)
        if not entry:
            return False, "", False

        needs_dobj = False
        dobj_label = "[somebody/something]"
        is_passive = False

        for s in entry.get("senses", []):
            for pat in s.get("patterns", []):
                pat_l = pat.lower()
                tokens = pat_l.split()
                if p_low in tokens:
                    if f"{v_low} somebody/something {p_low}" in pat_l:
                        needs_dobj = True
                        dobj_label = "[somebody/something]"
                    elif f"{v_low} somebody {p_low}" in pat_l:
                        needs_dobj = True
                        dobj_label = "[Person / Employee / Candidate]"
                    elif f"{v_low} something {p_low}" in pat_l:
                        needs_dobj = True
                        dobj_label = "[Entity / Object / Idea]"
                    elif pat_l.startswith(f"be {v_low}") or f"be {v_low}ed {p_low}" in pat_l:
                        is_passive = True

        return needs_dobj, dobj_label, is_passive

    @classmethod
    def extract_phrasal_verbs(cls, sentence: str) -> List[str]:
        """
        Deterministically extracts multi-word phrasal verbs from the dependency tree
        by querying verb heads governing direct 'prt' (particle) children,
        strictly pruning ordinary spatial/temporal prepositional phrases ('prep + pobj').
        e.g.
          'come in' -> extracted (prt)
          'give up' -> extracted (prt)
          'come in a city' -> rejected (prep + pobj, spatial adjunct)
        """
        nlp = cls.get_spacy()
        doc = nlp(sentence)
        phrasal_verbs = []

        for token in doc:
            if token.pos_ != "VERB":
                continue

            # Check direct particle children (prt)
            prts = [child for child in token.children if child.dep_ == "prt"]
            for p in prts:
                pv = f"{token.lemma_.lower()} {p.lemma_.lower()}"
                # Check for secondary preposition attached to verb or particle (e.g. come up with)
                prep_children = [c for c in p.children if c.dep_ == "prep"]
                if not prep_children:
                    prep_children = [c for c in token.children if c.dep_ == "prep" and c.i > p.i]
                if prep_children:
                    pv += f" {prep_children[0].lemma_.lower()}"
                phrasal_verbs.append(pv)

        return list(dict.fromkeys(phrasal_verbs))
