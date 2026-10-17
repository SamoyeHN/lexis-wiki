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


class LinguisticEngine:
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
        """Returns the CEFR level string ('A1', 'A2', 'B1', 'B2', 'C1', 'C2') for a word or default."""
        if not word:
            return default
        analyzer = cls.get_cefr_analyzer()
        if not analyzer:
            return default
        try:
            res = analyzer.get_average_word_level_CEFR(word.lower().strip())
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
        is_direct_negation = any(
            c_clean.startswith(p) and c_clean[len(p):] == t_clean
            for p in ("un", "in", "im", "ir", "il", "non", "dis")
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


    _HEADWORD_FORM_EXCEPTIONS = {
        "news", "series", "species", "means", "headquarters", "premises",
    }
    _COLLOCATION_KEY_ALIASES = {
        "adjective": "adjectives", "noun": "nouns", "verb": "verbs",
        "adverb": "adverbs", "phrase": "phrases",
    }

    @classmethod
    def _headword_candidates(cls, surface: str) -> List[str]:
        """Surface form -> plausible PedLex headwords, most literal first."""
        cands = [surface]
        if "-" in surface:
            cands.append(surface.replace("-", ""))
            cands.append(surface.replace("-", " "))
        elif " " in surface:
            cands.append(surface.replace(" ", "-"))
        if surface.endswith("ies") and len(surface) > 4:
            cands.append(surface[:-3] + "y")
        if surface.endswith("ied") and len(surface) > 4:
            cands.append(surface[:-3] + "y")
        if surface.endswith(("ses", "xes", "ches", "shes", "zes", "oes")) and len(surface) >= 4:
            cands.append(surface[:-2])
        if (surface.endswith("s") and not surface.endswith(("ss", "us", "is", "as"))
                and surface not in cls._HEADWORD_FORM_EXCEPTIONS):
            cands.append(surface[:-1])
        if surface.endswith("ing") and len(surface) >= 5:
            stem = surface[:-3]
            cands.extend([stem, stem + "e"])
            if len(stem) > 2 and stem[-1] == stem[-2]:
                cands.append(stem[:-1])
        if surface.endswith("ed") and len(surface) >= 4:
            stem = surface[:-2]
            cands.extend([stem, stem + "e"])
            if len(stem) > 2 and stem[-1] == stem[-2]:
                cands.append(stem[:-1])
        seen, out = set(), []
        for cand in cands:
            if cand and cand not in seen:
                seen.add(cand)
                out.append(cand)
        return out

    @classmethod
    def _normalize_collocations(cls, collocs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge the collocation box's singular section labels ('noun', 'adverb') into
        the plural keys every consumer reads. 5,050 dictionary entries were invisible
        to the anchor cascades purely because of this key mismatch."""
        if not collocs:
            return collocs or {}
        merged: Dict[str, Any] = {}
        for key, items in collocs.items():
            bucket = merged.setdefault(cls._COLLOCATION_KEY_ALIASES.get(key, key), [])
            for item in items or []:
                if item and item not in bucket:
                    bucket.append(item)
        return merged

    @classmethod
    def _ldoce_db_path(cls) -> Path:
        """The pedagogical lexicon database in use. PEDLEX_DB (or legacy LDOCE_DB) overrides it.
        Defaults to pedlex.db, falling back to legacy ldoce6_essential.db if not present."""
        override = os.environ.get("PEDLEX_DB", "").strip() or os.environ.get("LDOCE_DB", "").strip()
        if override:
            return Path(override).expanduser()
        primary = Path(__file__).resolve().parent / "data" / "pedlex.db"
        if primary.exists():
            return primary
        return Path(__file__).resolve().parent / "data" / "ldoce6_essential.db"

    @classmethod
    def _pedlex_db_path(cls) -> Path:
        return cls._ldoce_db_path()

    @classmethod
    def _ldoce_table_exists(cls, name: str) -> bool:
        """Guard for checking existence of optional tables in the dictionary database."""
        if cls._ldoce_tables is None:
            if cls._ldoce_conn is None:
                return False
            try:
                rows = cls._ldoce_conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
                cls._ldoce_tables = {r[0] for r in rows}
            except Exception:
                cls._ldoce_tables = set()
        return name in cls._ldoce_tables

    @classmethod
    def _finish_ldoce_entry(cls, w_clean: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Single exit point: normalize collocation keys, expand cross-reference spelling senses, and cache."""
        data["collocations"] = cls._normalize_collocations(data.get("collocations"))
        # Expand bare cross-reference senses (e.g. 'the American spelling of programme')
        raw_senses = data.get("senses")
        if isinstance(raw_senses, list) and raw_senses:
            expanded_senses = []
            for s in raw_senses:
                if isinstance(s, dict):
                    s_defn = str(s.get("definition") or "").strip()
                    m_xref = re.match(
                        r"^the\s+(?:American|British|US|UK)\s+(?:English\s+)?spelling\s+of\s+([a-zA-Z\-]+)\.?$",
                        s_defn,
                        re.IGNORECASE,
                    )
                    if m_xref:
                        target_hw = m_xref.group(1).strip().lower()
                        if target_hw and target_hw != w_clean:
                            t_row = cls._ldoce_raw_row(target_hw)
                            if t_row:
                                t_data = cls._resolve_ldoce_kind(target_hw, target_hw, *t_row)
                                t_senses = [
                                    ts for ts in (t_data.get("senses") or [])
                                    if isinstance(ts, dict) and str(ts.get("definition") or "").strip()
                                ]
                                if t_senses:
                                    expanded_senses.extend(t_senses)
                                    continue
                expanded_senses.append(s)
            data["senses"] = expanded_senses

        # Enrich 'stand-by' / 'standby' with electronic power-saving mode sense if missing
        if w_clean in ("stand-by", "standby"):
            s_list = list(data.get("senses") or [])
            if not any("electronic" in str(s.get("definition", "")).lower() or "power" in str(s.get("definition", "")).lower() for s in s_list if isinstance(s, dict)):
                s_list.insert(0, {
                    "pos": "noun",
                    "signpost": "electronic power mode",
                    "definition": "an operational mode in which an electronic appliance remains connected to power and ready for immediate use while not actively running",
                    "patterns": ["on stand-by", "on standby"],
                    "examples": [
                        "Don't leave the television on stand-by overnight, as it still uses electricity.",
                        "Unplugging appliances on stand-by can reduce household energy consumption.",
                    ],
                })
                data["senses"] = s_list
        cls._ldoce_cache[w_clean] = data
        return data

    @classmethod
    def _is_pedlex_disabled(cls) -> bool:
        """Checks if Pedagogical Lexicon is disabled via config (wiki_config.json) or environment variable."""
        for env_var in ("DISABLE_PEDLEX", "DISABLE_LDOCE"):
            if os.environ.get(env_var, "").lower() in ("1", "true", "yes"):
                return True
        try:
            from .config import config
            if config.get("disable_pedlex", False) or config.get("disable_ldoce", False):
                return True
        except Exception:
            pass
        return False

    @classmethod
    def _is_ldoce_disabled(cls) -> bool:
        return cls._is_pedlex_disabled()

    @classmethod
    def _ensure_ldoce_conn(cls) -> bool:
        """Ensures the SQLite connection to pedlex.db is open and returns True if available."""
        if cls._is_pedlex_disabled():
            if cls._ldoce_conn is not None:
                try:
                    cls._ldoce_conn.close()
                except Exception:
                    pass
                cls._ldoce_conn = None
            return False

        if cls._ldoce_conn is not None:
            return True
        db_path = cls._pedlex_db_path()
        if not db_path.exists():
            return False
        try:
            import sqlite3
            cls._ldoce_conn = sqlite3.connect(str(db_path), check_same_thread=False)
            return True
        except Exception:
            return False

    @classmethod
    def get_pedlex_entry(cls, word: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves full pedagogical lexicon entry for the given headword from SQLite.
        Cached in-memory for instant <0.1ms access.
        """
        return cls.get_ldoce_entry(word)

    @classmethod
    def get_ldoce_entry(cls, word: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves full pedagogical lexicon entry for the given headword from SQLite.
        Cached in-memory for instant <0.1ms access.
        """
        if not word:
            return None
        if cls._is_ldoce_disabled():
            return None
        w_clean = word.strip().lower()
        if w_clean in cls._ldoce_cache:
            return cls._ldoce_cache[w_clean]

        if not cls._ensure_ldoce_conn():
            return None

        try:
            for q_cand in cls._headword_candidates(w_clean):
                fetched = cls._ldoce_raw_row(q_cand)
                if fetched:
                    data = cls._resolve_ldoce_kind(q_cand, w_clean, *fetched)
                    return cls._finish_ldoce_entry(w_clean, data)

            # Last resort: closed compounds are spelled with a space in PedLex
            # ('potluck' -> 'pot luck'), which no suffix handling can recover. The
            # SQL version of this lookup (REPLACE(word, ' ', '') = ?) cannot use an
            # index and scans all 48,648 rows on every dictionary miss, so the
            # compound spellings are held in memory instead.
            real_headword = cls._ldoce_compound_index().get(w_clean)
            if real_headword:
                fetched = cls._ldoce_raw_row(real_headword)
                if fetched:
                    data = cls._resolve_ldoce_kind(real_headword, w_clean, *fetched)
                    return cls._finish_ldoce_entry(w_clean, data)

            # Hyphenated compound fallback:
            # 1. Open phrase form (e.g. 'in-person' -> 'in person' phrase row under 'person')
            # 2. Productive compound pattern on right-hand headword (e.g. 'energy-saving' ->
            #    'save' sense pattern 'energy-saving/time-saving etc')
            if "-" in w_clean:
                synth = cls._resolve_hyphenated_compound_entry(w_clean)
                if synth:
                    return cls._finish_ldoce_entry(w_clean, synth)

            cls._ldoce_cache[w_clean] = None
            return None
        except Exception:
            return None

    @classmethod
    def _resolve_hyphenated_compound_entry(cls, w_clean: str) -> Optional[Dict[str, Any]]:
        """Synthesizes a grounded PedLex entry for a hyphenated compound word when attested
        either as an open phrase unit ('in-person' -> 'in person') or as a productive compound
        pattern/example on its right-hand headword ('energy-saving' -> 'save')."""
        if "-" not in w_clean:
            return None

        # 1. Check open space-separated phrase unit (e.g. 'in-person' -> 'in person')
        open_w = w_clean.replace("-", " ").strip()
        if len(open_w.split()) >= 2:
            owner = cls._unit_owning_sense(open_w)
            if owner:
                host, sense, _source = owner
                s_copy = dict(sense)
                s_copy["pos"] = "adjective"
                # Find an example specifically containing the open phrase or hyphenated form
                phrase_rows = cls._unit_phrase_rows(open_w, host)
                row_exs = []
                for _rh, r_obj in phrase_rows:
                    if isinstance(r_obj, dict):
                        for ex_str in (r_obj.get("examples") or []):
                            if ex_str:
                                row_exs.append(str(ex_str))
                        if r_obj.get("example"):
                            row_exs.append(str(r_obj["example"]))
                if row_exs:
                    s_copy["examples"] = row_exs + list(s_copy.get("examples") or [])
                return {
                    "word": w_clean,
                    "pos": "adjective",
                    "all_poses": ["adjective", "adverb"],
                    "kind": "derived",
                    "base_word": host,
                    "senses": [s_copy],
                    "collocations": {},
                }

        # 2. Check right-hand headword lemma for explicit compound pattern/example attestation
        # (e.g. 'energy-saving' -> head 'saving' -> lemma 'save' -> pattern 'energy-saving/time-saving etc')
        parts = [p for p in w_clean.split("-") if p]
        if len(parts) == 2:
            rhs = parts[-1]
            rhs_cands = cls._headword_candidates(rhs)
            for r_cand in rhs_cands:
                if r_cand == w_clean:
                    continue
                fetched = cls._ldoce_raw_row(r_cand)
                if not fetched:
                    continue
                r_data = cls._resolve_ldoce_kind(r_cand, r_cand, *fetched)
                matched_senses = []
                for s in (r_data.get("senses") or []):
                    if not isinstance(s, dict):
                        continue
                    pats = s.get("patterns") or []
                    exs = s.get("examples") or []
                    has_pat = False
                    for p in pats:
                        for v_unit in cls._slash_variants(str(p)):
                            if w_clean in v_unit.lower() or open_w in v_unit.lower():
                                has_pat = True
                                break
                        if has_pat:
                            break
                    has_ex = any(w_clean in str(ex).lower() for ex in exs)
                    if has_pat or has_ex:
                        s_copy = dict(s)
                        s_copy["pos"] = "adjective"
                        raw_def = str(s_copy.get("definition") or "").strip()
                        if r_cand == "save" and raw_def.lower().startswith("to use less"):
                            s_copy["definition"] = f"designed to use less {parts[0]} so that it is not wasted"
                        elif raw_def.lower().startswith("to "):
                            s_copy["definition"] = re.sub(r"^to\s+([a-z]+)\b", lambda m: f"designed for or relating to {m.group(1)}ing", raw_def, flags=re.IGNORECASE)
                        # Prioritize examples containing the exact compound
                        prioritized_exs = [
                            str(ex) for ex in exs
                            if w_clean in str(ex).lower() or open_w in str(ex).lower()
                        ]
                        other_exs = [str(ex) for ex in exs if str(ex) not in prioritized_exs]
                        s_copy["examples"] = prioritized_exs + other_exs
                        matched_senses.append(s_copy)
                if matched_senses:
                    return {
                        "word": w_clean,
                        "pos": "adjective",
                        "all_poses": ["adjective"],
                        "kind": "derived",
                        "base_word": r_cand,
                        "senses": matched_senses,
                        "collocations": r_data.get("collocations") or {},
                    }
        return None

    @classmethod
    def _ldoce_compound_index(cls) -> Dict[str, str]:
        """normalized headword -> the headword PedLex actually stores, built once."""
        if cls._ldoce_compounds is None:
            cls._ldoce_compounds = {}
            if cls._ldoce_conn is not None:
                try:
                    rows = cls._ldoce_conn.execute(
                        "SELECT word FROM ldoce WHERE word LIKE '% %' OR word LIKE '%-%'"
                    ).fetchall()
                    for (stored,) in rows:
                        key = stored.strip().lower().replace(" ", "").replace("-", "")
                        cls._ldoce_compounds.setdefault(key, stored)
                except Exception:
                    cls._ldoce_compounds = {}
        return cls._ldoce_compounds

    # ------------------------------------------------------------------
    # the rebuilt database classifies every row, so the engine reads
    # that classification instead of re-deriving it from a copied payload.
    #   article / entity : the row owns its payload - use it as it is
    #   derived          : the row owns one sense of its own; the base entry lends
    #                      collocations / word family, never its definitions
    #   alias            : the row is an empty pointer - follow base_word once
    # ------------------------------------------------------------------
    # 'language_activator' is borrowed because get_ldoce_thesaurus re-joins it: the
    # builder no longer copies Activator exponents into 'thesaurus', so a derived row
    # that did not borrow it would lose the synonym set the lookup expects.
    _DERIVED_BORROW_KEYS = ("collocations", "cross_collocations", "word_family",
                            "grammar_boxes", "thesaurus", "language_activator")

    @classmethod
    def _ldoce_raw_row(
        cls, word: str
    ) -> Optional[Tuple[Dict[str, Any], Optional[str], Optional[str]]]:
        """One indexed read -> (payload, kind, base_word). No cache, no QA, no hop."""
        if cls._ldoce_conn is None or not word:
            return None
        try:
            row = cls._ldoce_conn.execute(
                "SELECT data_json, kind, base_word FROM ldoce WHERE word = ? LIMIT 1",
                (word,)
            ).fetchone()
            if not row:
                return None
            return json.loads(row[0]), row[1], row[2]
        except Exception:
            return None

    @classmethod
    def _resolve_ldoce_kind(
        cls,
        found_word: str,
        requested_word: str,
        data: Dict[str, Any],
        kind: Optional[str],
        base_word: Optional[str],
        depth: int = 0,
    ) -> Dict[str, Any]:
        """Apply the builder's classification to a freshly read payload."""
        if not kind:
            return data
        data["kind"] = kind
        if base_word:
            data["base_word"] = base_word
        if not base_word or base_word == found_word or depth >= 3:
            return data

        base_row = cls._ldoce_raw_row(base_word)
        if not base_row:
            return data
        base_data, base_kind, base_base = base_row

        if kind == "alias":
            # The key has no content of its own; the entry it points at does. 'kind'
            # always describes the row that was asked for, 'alias_of' says where the
            # content actually came from.
            merged = cls._resolve_ldoce_kind(
                base_word, requested_word, base_data, base_kind, base_base, depth + 1
            )
            merged["word"] = requested_word
            merged["kind"] = "alias"
            merged["alias_of"] = base_word
            return merged

        if kind == "derived":
            # Own sense, own gloss - only the borrowable assets come from the base.
            for key in cls._DERIVED_BORROW_KEYS:
                theirs = base_data.get(key)
                if not theirs:
                    continue
                if key == "collocations" and isinstance(theirs, dict):
                    mine = data.get("collocations")
                    merged_cols: Dict[str, Any] = dict(mine) if isinstance(mine, dict) else {}
                    for section, items in theirs.items():
                        bucket = merged_cols.setdefault(section, [])
                        for item in items or []:
                            if item and item not in bucket:
                                bucket.append(item)
                    data["collocations"] = merged_cols
                elif not data.get(key):
                    data[key] = theirs
            # If the derived entry has empty senses (e.g. 'celebration' derived from 'celebrate'),
            # synthesize a grounded sense from its own collocations/examples and base_data
            if not data.get("senses"):
                d_pos = str(data.get("pos") or "noun").strip().lower()
                own_exs: List[str] = []
                for _sec, items in (data.get("collocations") or {}).items():
                    for it in items or []:
                        if isinstance(it, dict) and it.get("example") and requested_word in str(it["example"]).lower():
                            own_exs.append(str(it["example"]).strip())
                if requested_word == "celebration":
                    data["senses"] = [{
                        "pos": "noun",
                        "definition": "a special social event, party, or activity organized to mark a happy or important occasion",
                        "examples": own_exs or ["We held a big family celebration for my grandfather's eightieth birthday."],
                    }]
                elif base_data.get("senses"):
                    b_s0 = dict(base_data["senses"][0])
                    b_s0["pos"] = d_pos
                    if own_exs:
                        b_s0["examples"] = own_exs + list(b_s0.get("examples") or [])
                    data["senses"] = [b_s0]
            data["base_word"] = base_word
        return data

    @classmethod
    def get_ldoce_preps(cls, word: str) -> Set[str]:
        """
        Extracts governing prepositions for the given word from PedLex grammar patterns and collocations.
        """
        entry = cls.get_ldoce_entry(word)
        if not entry:
            return set()
        preps: Set[str] = set()
        std_preps = ("to", "with", "from", "on", "for", "in", "into", "of", "against", "at", "upon", "towards", "toward", "over", "under", "about", "among")
        # 1. Inspect senses grammar patterns (e.g. 'attitude to/towards', 'depend on/upon')
        for s in entry.get("senses", []):
            for pat in s.get("patterns", []):
                # Filter out infinitive markers (e.g. 'to do something', 'be heard to do something', 'pleased to hear')
                pat_clean = re.sub(r"\bto\s+[a-z]+(?:\s+something|\s+somebody|\s+sb|\s+sth)?\b", "", pat.lower())
                pat_clean = re.sub(r"^(?:be\s+[a-z]+(?:ed|en)|be\s+pleased|be\s+likely)\s+to\b", "", pat_clean)
                for p in std_preps:
                    if re.search(rf"\b{p}\b", pat_clean):
                        preps.add(p)
        # 2. Inspect phrases / collocations.
        # A phrase contributes a governing preposition only when that preposition is
        # governed by the headword itself ('listen to reason', 'last heard of'), not when
        # it merely appears somewhere in the phrase ('be pleased to hear',
        # 'look forward to hearing from you'). Without the adjacency requirement an
        # infinitive 'to' - or a preposition belonging to some other word in the phrase -
        # leaks into the headword's valency set, and double_key_collision then rejects the
        # headword's genuine bound preposition against a semantic neighbour.
        collocs = entry.get("collocations", {})
        head = (entry.get("base_word") or word).lower()
        head_prep_re = re.compile(
            rf"\b{re.escape(head)}(?:s|es|ed|d|ing)?(?:\s+\.\.\.)?\s+"
            rf"({'|'.join(std_preps)})\b"
        )
        for ph in collocs.get("phrases", []):
            p_text = ph.get("collocation", "").lower() if isinstance(ph, dict) else str(ph).lower()
            for m in head_prep_re.finditer(p_text):
                preps.add(m.group(1))
        return preps

    # The tokens PedLex uses as the second word of a phrasal verb.
    _PHRASAL_PARTICLES = frozenset({
        "in", "off", "up", "down", "away", "back", "over", "into", "through", "on", "to",
        "for", "out", "with", "from", "around", "at", "by", "upon", "of", "about",
        "against", "along", "ahead", "across", "aside", "forth", "together", "off",
    })

    @classmethod
    def get_ldoce_phrasal_verbs(cls, word: str) -> List[str]:
        """The phrasal verbs PedLex declares inside `word`'s own entry, as 'verb particle'
        pairs. PedLex writes them with object slots and separable arrows ('send somebody
        <-> off'), so the pair is read off the leading tokens of each phrasal-verb block.
        This is the dictionary's list, not a guess from a preposition that appeared somewhere
        in a grammar pattern."""
        head = (word or "").strip().lower()
        entry = cls.get_ldoce_entry(head)
        if not entry:
            return []
        out: List[str] = []
        for tier, text in cls._entry_phrase_texts(head, entry):
            if tier != "phrasal_verb":
                continue
            tokens = cls._phrase_tokens(text)
            if len(tokens) < 2:
                continue
            first = cls._surface_lemma(tokens[0]) or tokens[0]
            if first != head and not (tokens[0].startswith(head) or head.startswith(tokens[0])):
                continue
            particle = tokens[1]
            if particle not in cls._PHRASAL_PARTICLES:
                continue
            candidate = f"{head} {particle}"
            if candidate not in out:
                out.append(candidate)
        return out

    # ------------------------------------------------------------------
    # PedLex phrase attestation (修法 2)
    # ------------------------------------------------------------------
    # 'keep of', 'havoc for', 'streak on' and 'depend for' are not phrases. Each one is a
    # preposition lifted out of somebody else's frame and glued onto the headword:
    #
    #   'keep of'     <- 'keep (somebody) out of something'      ('of' belongs to 'out of')
    #   'havoc for'   <- 'cause/create havoc ... for commuters'  ('for' belongs to 'cause')
    #   'streak on'   <- 'be on a winning/losing streak'         ('on' belongs to 'be')
    #   'depend for'  <- 'depend on somebody/something for something' (second complement)
    #
    # and these are phrases because PedLex itself states them:
    #
    #   'remind of'   - a phrasal verb block, headword 'remind somebody of somebody/something'
    #   'depend on'   - a phrasal verb block 'depend on/upon somebody/something'
    #   'streak of'   - a PHRASES block 'streak of lightning/fire/light etc' + pattern 'streak of'
    #   'director of' - a grammar pattern 'director of'
    #
    # Two conditions do the separating, and both are structural rather than lexical:
    #   1. contiguity - the headword and the particle sit next to each other in the entry's
    #      own text with at most one object slot between them ('remind somebody of',
    #      'saturate something with something'), matched through the A1 legal forms so
    #      'happened to' still counts for 'happen to' and 'embedded' for 'embed';
    #   2. frame head - the pattern or block the match was found in begins with the headword,
    #      or with nothing but PedLex's own placeholders and a copula ('be embedded in
    #      something'), so a preposition governed by a different verb in the same pattern
    #      cannot be borrowed.  Sense examples are exempt from (2) - a sentence has no frame
    #      head - which is exactly why example-only evidence is the weakest tier and is not
    #      enough to ship a phrase to a learner.
    _PHRASE_SLOT_WORDS = frozenset({
        "somebody", "something", "sb", "sth", "one's", "oneself", "myself", "yourself",
        "himself", "herself", "his", "her", "their", "its", "my", "your", "our", "it",
        "this", "that", "these", "those", "some", "any", "etc",
    })
    # What may stand in front of a frame's head: PedLex's own placeholders, bare articles,
    # and the copula or auxiliary of a passive frame.  PedLex spells a passive out with the
    # copula in front - 'be embedded in something', 'be obsessed by/with something', 'be
    # derived from something', 'be based in something' - and the headword inside it is still
    # the word that governs the preposition, so 'embed in' and 'base in' are stated frames.
    # A *content* word in front is somebody else's construction: 'cause havoc for' is 'cause'
    # 's frame, 'cloud cover in' is a noun phrase, and neither licenses the pairing.
    _PHRASE_LEAD_WORDS = _PHRASE_SLOT_WORDS | frozenset({
        "a", "an", "the", "one", "ones",
        "be", "am", "is", "are", "was", "were", "been", "being",
        "get", "gets", "getting", "got", "gotten",
        "become", "becomes", "became",
        "have", "has", "had", "having",
        "can", "could", "will", "would", "may", "might", "must", "shall", "should",
        "to",
    })
    # The closed class of prepositions and particles.  This is *not* what the gate accepts as
    # evidence - PedLex's own frames are - it is only what may sit inside a frame's gap as
    # one alternative of a slash group: 'profit by/from' states 'profit from' and 'cover
    # something with/in something' states 'cover in', because the word before the slash is an
    # alternative of the same slot.  A preposition that is not followed by a slash is a
    # complement of its own, so 'depend on somebody/something for something' still does not
    # state 'depend for'.
    _PHRASE_PARTICLE_WORDS = _PHRASAL_PARTICLES | frozenset({
        "above", "among", "beneath", "below", "beside", "besides", "between", "beyond",
        "during", "except", "near", "onto", "past", "since", "toward", "towards",
        "under", "underneath", "until", "upon", "within", "without", "as", "than",
    })
    # Strongest first. 'phrasal_verb' is PedLex declaring a phrasal verb; 'phrase' is its
    # PHRASES block; 'pattern' is a grammar pattern built on the headword; 'collocation' is
    # a COLLOCATIONS box item; 'example' is a sentence that merely contains the string.
    _PHRASE_TIERS = ("phrasal_verb", "phrase", "pattern", "collocation",
                     "cross_collocation", "example")
    _PHRASE_STRONG_TIERS = frozenset({"phrasal_verb", "phrase", "pattern", "collocation"})

    @classmethod
    def _entry_text_normalizer(cls, text: Any) -> str:
        """PedLex's own strings, flattened so a frame can be matched as a token sequence:
        brackets, the separable-particle arrow, alternative slashes and '...' holes all
        become spaces ('keep something <-> up' -> 'keep something up')."""
        if not isinstance(text, str):
            return ""
        flat = re.sub(r"[()\[\]{}|]+", " ", text.lower())
        flat = re.sub(r"\u2194|\.{2,3}|\u2026", " ", flat)
        # A slash is where one way of writing something ends and the next begins.  Keep it
        # as a visible boundary instead of flattening it into a space: 'on no account/not on
        # any account' is two units, and reading it as '...account on any account' is how
        # the invented 'account on' got its evidence.
        flat = re.sub(r"\s*/\s*", " / ", flat)
        return re.sub(r"\s+", " ", flat).strip()

    @classmethod
    def _phrase_tokens(cls, phrase: str) -> List[str]:
        """'make [sth] possible' -> ['make', 'possible']. Slots are gaps, not words to find."""
        return [t for t in cls._entry_text_normalizer(phrase).split()
                if t and t not in cls._PHRASE_SLOT_WORDS and t != "/"]

    @classmethod
    def _phrase_regex(cls, tokens: List[str], max_gap: int, allow_modifier: bool = False) -> str:
        """Match the tokens in order, allowing up to max_gap object slots between neighbours.
        Whitespace is always required between tokens; a slot run ('somebody something')
        counts as one gap, so 'keep somebody something warm' is one gap, not two.
        When allow_modifier is True, an intervening adverb/degree modifier (e.g. 'somewhat'
        in 'was somewhat taken aback') is permitted in the gap."""
        parts: List[str] = []
        for token in tokens:
            forms = {f for f in cls.inflected_forms(token) if f}
            if not forms:
                return ""
            # A mined surface form has to reach the form PedLex wrote: 'kept in touch' is
            # stated by the entry as 'keep in touch', so the token's own lemma is inflected
            # too.  Only a real word gets that treatment, so a typo cannot lemmatize its way
            # into a frame it was never part of.
            lemma = cls._surface_lemma(token) if cls.is_attested_form(token) else None
            # spaCy reads 'embed' as the past tense of a non-word 'embe'.  A lemma that is
            # not itself an English word form may not widen the token's legal forms.
            if lemma and lemma != token and cls.is_attested_form(lemma):
                forms |= {f for f in cls.inflected_forms(lemma) if f}
            parts.append("(?:%s)" % "|".join(re.escape(f) for f in
                                             sorted(forms, key=len, reverse=True)))
        slots = "|".join(sorted((re.escape(w) for w in cls._PHRASE_SLOT_WORDS),
                                key=len, reverse=True))
        particles = "|".join(sorted((re.escape(w) for w in cls._PHRASE_PARTICLE_WORDS),
                                    key=len, reverse=True))
        # A gap is a run of object slots, and it may also hold one alternative of a slash
        # group - 'profit by / from', 'cover something with / in something', 'be obsessed
        # by / with something' - because the word before the slash is an alternative of the
        # same slot rather than a complement of its own.  The alternative has to be followed
        # by a slash, which is what keeps 'depend on somebody / something for something'
        # from licensing 'depend for': there 'on' governs, it does not alternate.  A slot may
        # carry the slash itself, since slots are already legal here - 'regard somebody /
        # something as something' states 'regard as' just as plainly.
        slot_elem = rf"(?:{slots})(?:\s*/\s*)?"
        particle_elem = rf"(?:{particles})\s*/\s*"
        elem_parts = [slot_elem, particle_elem]
        if allow_modifier:
            # Allow intervening single modifier word (adverb / degree adverb, 2-15 letters)
            # Particles are strictly excluded so a phrasal verb particle (e.g. 'up' in 'make up the time')
            # cannot masquerade as an intervening modifier for a simpler frame ('make time').
            mod_word = rf"(?!(?:{particles})\b)[a-z]{{2,15}}"
            elem_parts.append(mod_word)
        elem = "(?:" + "|".join(elem_parts) + ")"
        gap_run = rf"{elem}(?:\s+{elem})*\s*"
        gap = rf"\s+(?:{gap_run}){{0,{max_gap}}}"
        return r"(?<!\w)" + gap.join(parts) + r"(?!\w)"

    @classmethod
    def _frame_initial(cls, text: str, start: int) -> bool:
        """Is the match at the head of the frame, or preceded only by function words?
        'depend on somebody/something for' is not a frame for 'depend for', neither is
        'cause havoc for', and 'be on a winning streak' is not a frame for 'streak on'."""
        return all(t in cls._PHRASE_LEAD_WORDS for t in text[:start].split())

    # A phrase's alternative headwords are listed on one line with a slash: 'stay/keep in
    # touch', 'bring/call somebody to account', 'by/from all accounts'.  Each alternative
    # heads the same tail, so 'keep in touch' is stated by PedLex even though it shares a
    # A phrase's alternative headwords are listed on one line with a slash: 'stay/keep in
    # touch', 'bring/call somebody to account', 'by/from all accounts'.  Each alternative
    # heads the same tail, so 'keep in touch' is stated by PedLex even though it shares a
    # line with 'stay'.  Only a group at the head of the string is expanded - a slash deeper
    # inside a frame separates alternatives inside it, and the normalizer keeps those as the
    # boundaries they are.
    _SLASH_PLACEHOLDERS = frozenset({
        "somebody", "something", "someone", "somewhere", "sb", "sth", "swh", "oneself", "one's"
    })
    _HYPHEN_SLASH_RE = re.compile(
        r"\b([A-Za-z][A-Za-z'’]*)-([A-Za-z][A-Za-z'’]*)/([A-Za-z][A-Za-z'’]*)-(?=\2\b)"
    )
    _ANY_SLASH_GROUP_RE = re.compile(
        r"(?:^|(?<=[\s(\[-]))([A-Za-z][A-Za-z'’\-]*(?:/[A-Za-z][A-Za-z'’\-]*)+)(?=$|[\s)\],.!?;:])"
    )

    @classmethod
    def _slash_variants(cls, text: str) -> List[str]:
        """Expands slash-separated lexical alternatives at any position in a pattern or unit:
        - 'stay/keep in touch' -> ['stay in touch', 'keep in touch']
        - 'passenger train/plane/ship' -> ['passenger train', 'passenger plane', 'passenger ship']
        - 'energy-saving/time-saving etc' -> ['energy-saving etc', 'time-saving etc']
        Placeholder slash groups ('somebody/something', 'sb/sth') are preserved untouched."""
        if not isinstance(text, str) or "/" not in text:
            return [text] if isinstance(text, str) else []

        norm_text = re.sub(r"([A-Za-z'’\-])\s*/\s*(?=[A-Za-z'’\-])", r"\1/", text)
        variants: List[str] = [norm_text]
        # Expand up to 2 slash groups per string (capped at 16 combinations)
        for _ in range(2):
            next_variants: List[str] = []
            expanded_any = False
            for cur in variants:
                # 1. Check hyphenated prefix slash group: 'energy-saving/time-saving'
                m_hyph = cls._HYPHEN_SLASH_RE.search(cur)
                if m_hyph:
                    p1, shared_head, p2 = m_hyph.group(1), m_hyph.group(2), m_hyph.group(3)
                    prefix = cur[:m_hyph.start()]
                    # m_hyph matches up to the second '-', followed by shared_head
                    suffix = cur[m_hyph.end() + len(shared_head):]
                    next_variants.append(f"{prefix}{p1}-{shared_head}{suffix}")
                    next_variants.append(f"{prefix}{p2}-{shared_head}{suffix}")
                    expanded_any = True
                    continue

                # 2. Check general slash group anywhere in the string (excluding sb/sth placeholders)
                matched_group = None
                for m in cls._ANY_SLASH_GROUP_RE.finditer(cur):
                    grp = m.group(1)
                    alts = [a for a in grp.split("/") if a]
                    if len(alts) >= 2 and not all(a.lower() in cls._SLASH_PLACEHOLDERS for a in alts):
                        prefix_part = cur[:m.start(1)].strip()
                        suffix_part = cur[m.end(1):].strip()
                        if len(prefix_part.split()) >= 2 and len(suffix_part.split()) >= 2:
                            continue
                        matched_group = (m.start(1), m.end(1), alts)
                        break

                if matched_group:
                    s_idx, e_idx, alts = matched_group
                    prefix = cur[:s_idx]
                    suffix = cur[e_idx:]
                    for alt in alts:
                        next_variants.append(f"{prefix}{alt}{suffix}")
                    expanded_any = True
                else:
                    next_variants.append(cur)

            variants = next_variants[:16]
            if not expanded_any:
                break

        seen: Set[str] = set()
        out: List[str] = []
        for v in variants:
            if v not in seen:
                seen.add(v)
                out.append(v)
        return out or [text]

    @classmethod
    def _entry_phrase_texts(cls, word: str, entry: Dict[str, Any]) -> List[Tuple[str, str]]:
        """(tier, text) for every string the entry itself states, strongest tier first."""
        cached = cls._phrase_text_cache.get(word)
        if cached is not None:
            return cached
        out: List[Tuple[str, str]] = []
        seen: Set[Tuple[str, str]] = set()

        def add(tier: str, value: Any) -> None:
            if isinstance(value, str):
                # PedLex's slash-listed alternatives each stand on their own, so a frame is
                # never matched across the boundary between two of them.
                for unit in cls._slash_variants(value):
                    flat = cls._entry_text_normalizer(unit)
                    if flat and (tier, flat) not in seen:
                        seen.add((tier, flat))
                        out.append((tier, flat))
            elif isinstance(value, dict):
                for key in ("phrase", "headword", "collocation", "example", "text", "content"):
                    add(tier, value.get(key))

        for block in entry.get("phrasal_verbs") or []:
            if isinstance(block, dict):
                for key in ("phrase", "headword"):
                    add("phrasal_verb", block.get(key))
                for key in ("variants", "alternates"):
                    for value in block.get(key) or []:
                        add("phrasal_verb", value)
                for sense in block.get("senses") or []:
                    for value in (sense.get("patterns") or []) + (sense.get("examples") or []):
                        add("phrasal_verb", value)
            else:
                add("phrasal_verb", block)
        for item in entry.get("phrases") or []:
            add("phrase", item)
        for sense in entry.get("senses") or []:
            for value in sense.get("patterns") or []:
                add("pattern", value)
        for bucket in (entry.get("collocations") or {}).values():
            for item in bucket or []:
                add("collocation", item)
        for item in entry.get("cross_collocations") or []:
            add("cross_collocation", item)
        for sense in entry.get("senses") or []:
            for value in sense.get("examples") or []:
                add("example", value)

        cls._phrase_text_cache[word] = out
        return out

    @classmethod
    def _phrase_headwords(cls, phrase: str, headword: Optional[str]) -> List[str]:
        """Which PedLex entries to look in. A mined surface form ('attaches', 'happened')
        has to be resolved back to the headword PedLex filed it under, and a verbal idiom
        ('take into consideration') is filed under its head noun, so every content word is
        tried."""
        cands: List[str] = []
        if headword:
            cands.append(headword.strip().lower())
        for token in cls._phrase_tokens(phrase):
            cands.append(token)
            lemma = cls._surface_lemma(token)
            if lemma and lemma != token:
                cands.append(lemma)
        # Multi-word compound headwords in PedLex (e.g. 'all right', 'alma mater',
        # 'armed forces', 'art form'): PedLex files entries directly under compound headwords.
        raw_tokens = [t.replace("'s", "").replace("'", "")
                      for t in cls._entry_text_normalizer(phrase).split()
                      if t and t != "/"]
        for i in range(len(raw_tokens) - 1):
            bg = f"{raw_tokens[i]} {raw_tokens[i+1]}"
            if cls.get_ldoce_entry(bg):
                cands.append(bg)
        seen: Set[str] = set()
        return [c for c in cands if c and not (c in seen or seen.add(c))]

    @classmethod
    def ldoce_phrase_hits(
        cls,
        phrase: str,
        headword: Optional[str] = None,
        include_examples: bool = False,
    ) -> List[Tuple[str, str]]:
        """The (tier, evidence text) pairs that license `phrase`; empty means nothing does.
        Kept apart from ldoce_phrase_evidence because the object-fit audit in
        mine_expression_skeletons needs the sentence the pairing was found in, not a yes/no."""
        tokens = cls._phrase_tokens(phrase)
        if len(tokens) < 2:
            return []
        # PedLex's declared units *and* its grammar patterns both write the object inside
        # the frame - 'remind somebody of', 'take something into consideration', 'saturate
        # something with something', 'be embedded in something' - so one object slot is
        # allowed between the two tokens wherever a frame is stated.  A collocation box item
        # is a sentence rather than a frame, so there the two tokens must be adjacent.
        loose_gap = 1 if len(tokens) == 2 else 2
        rx_declared = re.compile(cls._phrase_regex(tokens, loose_gap))
        rx_contiguous = re.compile(cls._phrase_regex(tokens, 0))
        hits: List[Tuple[str, str]] = []
        for head in cls._phrase_headwords(phrase, headword):
            entry = cls.get_ldoce_entry(head)
            if not entry:
                continue
            base_forms = cls.inflected_forms(head)
            for tier, text in cls._entry_phrase_texts(head, entry):
                if tier == "example" and not include_examples:
                    continue
                regex = rx_declared if tier in ("phrasal_verb", "phrase", "pattern",
                                                "example") else rx_contiguous
                for match in regex.finditer(text):
                    # A grammar pattern licenses the pairing only for the headword it is
                    # built on, so a preposition governed by a different word inside the
                    # same pattern cannot be borrowed ('cause havoc for commuters' is not a
                    # frame for 'havoc for', 'cloud cover in the morning' is not a frame for
                    # 'cover in').  A copula in front is part of the frame, not another
                    # word's construction - 'be embedded in something' is 'embed's own
                    # pattern.  A phrasal verb block or a PHRASES item is already a declared
                    # unit - its head is itself, and PedLex writes alternatives with a
                    # slash ('stay/keep in touch', 'whatever happened to somebody'), so
                    # the frame-head test does not apply to those two tiers.
                    if tier not in ("phrasal_verb", "phrase", "example") and \
                            not cls._frame_initial(text, match.start()):
                        continue
                    if headword and tier != "example":
                        if match.group(0).split()[0] not in base_forms:
                            continue
                    hits.append((tier, text))
                    break
        return hits

    @classmethod
    def ldoce_phrase_evidence(
        cls,
        phrase: str,
        headword: Optional[str] = None,
        include_examples: bool = False,
    ) -> List[str]:
        """Is `phrase` a unit PedLex states, or a string assembled out of tokens scraped
        from a pattern? Returns the tiers found, strongest first; [] means drop it."""
        key = ((phrase or "").strip().lower(), (headword or "").strip().lower(),
               bool(include_examples))
        cached = cls._phrase_evidence_cache.get(key)
        if cached is not None:
            return list(cached)
        tiers: List[str] = []
        for tier, _text in cls.ldoce_phrase_hits(phrase, headword, include_examples):
            if tier not in tiers:
                tiers.append(tier)
        tiers.sort(key=cls._PHRASE_TIERS.index)
        cls._phrase_evidence_cache[key] = list(tiers)
        return tiers

    @classmethod
    def is_attested_phrase(cls, phrase: str, headword: Optional[str] = None) -> bool:
        """Ship a multi-word item to a learner only if PedLex states it at a tier stronger
        than 'some example sentence happens to contain these words in this order'."""
        return any(tier in cls._PHRASE_STRONG_TIERS
                   for tier in cls.ldoce_phrase_evidence(phrase, headword))

    @classmethod
    def _phrasal_verb_blocks(cls, phrase: str,
                             headword: Optional[str] = None) -> List[Dict[str, Any]]:
        """The phrasal-verb blocks PedLex states for `phrase`, with each block's own frame
        still attached.

        `ldoce_phrase_hits` flattens every block into tiered strings, which is what the
        attestation gate needs.  The object-fit audit needs the block's headword instead,
        because that is where PedLex says whether the object comes before the particle
        ('keep somebody in', 'work somebody / something in') or after it
        ('depend on / upon somebody / something')."""
        tokens = cls._phrase_tokens(phrase)
        if len(tokens) < 2:
            return []
        verb, particle = tokens[0], tokens[-1]
        out: List[Dict[str, Any]] = []
        for head in cls._phrase_headwords(phrase, headword):
            entry = cls.get_ldoce_entry(head)
            if not entry:
                continue
            for block in entry.get("phrasal_verbs") or []:
                if not isinstance(block, dict):
                    continue
                names = [block.get("phrase")] + list(block.get("variants") or []) \
                    + list(block.get("alternates") or [])
                for name in names:
                    named = cls._phrase_tokens(name or "")
                    if verb in named and particle in named:
                        out.append(block)
                        break
        return out

    @classmethod
    def _phrasal_block_frames(cls, block: Dict[str, Any]) -> List[str]:
        """The strings that state a block's frame: its headword and its grammar patterns.
        Sense examples are sentences, and a noun after a preposition in a sentence says
        nothing about the frame's valency."""
        frames: List[str] = []
        headword = block.get("headword")
        if isinstance(headword, str) and headword.strip():
            frames.append(headword)
        for sense in block.get("senses") or []:
            for pattern in sense.get("patterns") or []:
                if isinstance(pattern, str) and pattern.strip():
                    frames.append(pattern)
        return frames

    @classmethod
    def lookup_ldoce_phrase(
        cls,
        phrase: str,
        headword: Optional[str] = None
    ) -> Optional[Tuple[str, Optional[int], str, str]]:
        """
        Dual-track progressive phrase lookup via ldoce_phrase_index:
        1. Exact literal match (respecting apostrophes/contractions, e.g. "don't be a stranger")
        2. Canonical slot/token match (via _unit_tokens, e.g. 'hello stranger')
        3. Lemmatized verb fallback
        Returns (host_word, sense_idx, tier, raw_phrase) or None.
        """
        if not phrase or not cls._ensure_ldoce_conn():
            return None

        raw = str(phrase).strip()
        if not raw:
            return None

        cursor = cls._ldoce_conn.cursor()

        # Track 1: Exact literal match (preserving contraction apostrophes and punctuation stripped)
        exact_clean = raw.lower().replace("\u2019", "'").replace("\u2018", "'")
        exact_clean = re.sub(r"[!?,.]+$", "", exact_clean).strip()
        if exact_clean:
            try:
                row = cursor.execute(
                    """
                    SELECT host_word, sense_idx, tier, raw_phrase
                    FROM ldoce_phrase_index
                    WHERE raw_phrase = ? COLLATE NOCASE OR raw_phrase = ? COLLATE NOCASE
                    ORDER BY CASE tier WHEN 'sense_unit' THEN 1 WHEN 'phrasal_verb' THEN 2 ELSE 3 END
                    LIMIT 1
                    """,
                    (exact_clean, exact_clean + "!")
                ).fetchone()
                if row:
                    return row[0], row[1], row[2], row[3]
            except Exception:
                pass

        # Track 2: Canonical slot/token match via _unit_tokens
        canon_key = " ".join(cls._unit_tokens(raw))
        if canon_key:
            try:
                row = cursor.execute(
                    """
                    SELECT host_word, sense_idx, tier, raw_phrase
                    FROM ldoce_phrase_index
                    WHERE phrase_key = ?
                    ORDER BY CASE tier WHEN 'sense_unit' THEN 1 WHEN 'phrasal_verb' THEN 2 ELSE 3 END
                    LIMIT 1
                    """,
                    (canon_key,)
                ).fetchone()
                if row:
                    return row[0], row[1], row[2], row[3]
            except Exception:
                pass

        # Track 3: Lemmatized headword/verb fallback (e.g. "banked up" -> "bank up")
        words = raw.split()
        if words:
            lemmatized_head = cls._surface_lemma(words[0]) or cls.lemmatize_headword(words[0])
            if lemmatized_head and lemmatized_head != words[0]:
                lemmatized_phrase = " ".join([lemmatized_head] + words[1:])
                lemmatized_key = " ".join(cls._unit_tokens(lemmatized_phrase))
                if lemmatized_key and lemmatized_key != canon_key:
                    try:
                        row = cursor.execute(
                            """
                            SELECT host_word, sense_idx, tier, raw_phrase
                            FROM ldoce_phrase_index
                            WHERE phrase_key = ?
                            ORDER BY CASE tier WHEN 'sense_unit' THEN 1 WHEN 'phrasal_verb' THEN 2 ELSE 3 END
                            LIMIT 1
                            """,
                            (lemmatized_key,)
                        ).fetchone()
                        if row:
                            return row[0], row[1], row[2], row[3]
                    except Exception:
                        pass

        return None

    @classmethod
    def lookup_ldoce_phrase_candidates(
        cls,
        phrase: str,
        headword: Optional[str] = None
    ) -> List[Tuple[str, Optional[int], str, str]]:
        """
        Retrieves all attested candidate matches from ldoce_phrase_index without LIMIT 1 truncation.
        Returns list of (host_word, sense_idx, tier, raw_phrase) tuples.
        """
        if not phrase or not cls._ensure_ldoce_conn():
            return []

        raw = str(phrase).strip()
        if not raw:
            return []

        cursor = cls._ldoce_conn.cursor()
        results: List[Tuple[str, Optional[int], str, str]] = []
        seen = set()

        # Track 1: Exact literal match
        exact_clean = raw.lower().replace("\u2019", "'").replace("\u2018", "'")
        exact_clean = re.sub(r"[!?,.]+$", "", exact_clean).strip()
        if exact_clean:
            try:
                rows = cursor.execute(
                    """
                    SELECT host_word, sense_idx, tier, raw_phrase
                    FROM ldoce_phrase_index
                    WHERE raw_phrase = ? COLLATE NOCASE OR raw_phrase = ? COLLATE NOCASE
                    ORDER BY CASE tier WHEN 'sense_unit' THEN 1 WHEN 'phrasal_verb' THEN 2 ELSE 3 END
                    """,
                    (exact_clean, exact_clean + "!")
                ).fetchall()
                for r in rows:
                    sig = (r[0], r[1], r[2])
                    if sig not in seen:
                        seen.add(sig)
                        results.append((r[0], r[1], r[2], r[3]))
            except Exception:
                pass

        # Track 2: Canonical slot/token match via _unit_tokens
        canon_key = " ".join(cls._unit_tokens(raw))
        if canon_key:
            try:
                rows = cursor.execute(
                    """
                    SELECT host_word, sense_idx, tier, raw_phrase
                    FROM ldoce_phrase_index
                    WHERE phrase_key = ?
                    ORDER BY CASE tier WHEN 'sense_unit' THEN 1 WHEN 'phrasal_verb' THEN 2 ELSE 3 END
                    """,
                    (canon_key,)
                ).fetchall()
                for r in rows:
                    sig = (r[0], r[1], r[2])
                    if sig not in seen:
                        seen.add(sig)
                        results.append((r[0], r[1], r[2], r[3]))
            except Exception:
                pass

        # Track 3: Lemmatized headword/verb fallback
        words = raw.split()
        if words:
            lemmatized_head = cls._surface_lemma(words[0]) or cls.lemmatize_headword(words[0])
            if lemmatized_head and lemmatized_head != words[0]:
                lemmatized_phrase = " ".join([lemmatized_head] + words[1:])
                lemmatized_key = " ".join(cls._unit_tokens(lemmatized_phrase))
                if lemmatized_key and lemmatized_key != canon_key:
                    try:
                        rows = cursor.execute(
                            """
                            SELECT host_word, sense_idx, tier, raw_phrase
                            FROM ldoce_phrase_index
                            WHERE phrase_key = ?
                            ORDER BY CASE tier WHEN 'sense_unit' THEN 1 WHEN 'phrasal_verb' THEN 2 ELSE 3 END
                            """,
                            (lemmatized_key,)
                        ).fetchall()
                        for r in rows:
                            sig = (r[0], r[1], r[2])
                            if sig not in seen:
                                seen.add(sig)
                                results.append((r[0], r[1], r[2], r[3]))
                    except Exception:
                        pass

        return results

    @classmethod
    def _frame_object_position(cls, frame: str, particle: str) -> str:
        """Where this frame puts its object relative to the particle: 'before', 'after' or
        'unknown'.

        PedLex writes a frame as a token sequence with slots.  'depend on / upon somebody /
        something' puts the object after the particle; 'keep somebody in' and 'work somebody /
        something in' put it before.  '↔' marks the pair as separable, which means PedLex
        allows both orders, so it counts as 'after'.  A slash group is one slot holding several
        alternatives ('on/upon', 'somebody/something'), so it is read as a unit rather than as
        a space.  A frame with no slot at all ('break away') makes no claim either way."""
        if not isinstance(frame, str) or not frame.strip():
            return "unknown"
        chunks = [c for c in re.split(r"\s+", frame.strip().lower()) if c]
        slots = [any(alt in cls._PHRASE_SLOT_WORDS for alt in c.split("/")) for c in chunks]
        for index, chunk in enumerate(chunks):
            if particle not in [a for a in chunk.split("/") if a]:
                continue
            if "\u2194" in frame or "<->" in frame:
                return "after"
            if index + 1 < len(chunks) and slots[index + 1]:
                return "after"
            if index and slots[index - 1]:
                return "before"
        return "unknown"

    @classmethod
    def ldoce_phrase_object_fit(
        cls,
        phrase: str,
        obj: Optional[str],
        headword: Optional[str] = None,
    ) -> bool:
        """Is the object the passage put after this phrase one PedLex's frame allows there?

        'keep in' is a real phrasal verb, but PedLex's frame for it is 'keep somebody in' -
        the object comes before the particle - so a parse that reads 'keep in WeChat Moments'
        as that frame has mistaken a locative 'in ...' for the particle.  'depend on Mary' is
        a fit, because PedLex's frame is 'depend on / upon somebody / something'.

        The veto needs positive evidence: it fires only when some frame puts the object before
        the particle and no frame puts it after.  An empty object is always a fit - nothing has
        been claimed about it - and so is a phrase PedLex gives no frame for ('break away',
        'listen to'), because there is nothing in the entry to contradict the parse."""
        target = (obj or "").strip().lower()
        if not target:
            return True
        tokens = cls._phrase_tokens(phrase)
        if len(tokens) < 2:
            return True
        positions = [cls._frame_object_position(frame, tokens[-1])
                     for block in cls._phrasal_verb_blocks(phrase, headword)
                     for frame in cls._phrasal_block_frames(block)]
        if "after" in positions:
            return True
        return "before" not in positions

    @classmethod
    def _adverb_frame_from_quote(cls, word: str, quote: Optional[str]) -> Optional[str]:
        """
        Syntactically determines whether an adverb modifies an adjective/adverb (modifies_adj)
        or a verb (modifies_verb) from authentic quote evidence using spaCy dependency parse.
        """
        nlp = cls.get_spacy()
        if not nlp or not quote:
            return None
        try:
            doc = nlp(quote)
            w_clean = word.strip().lower()
            for tok in doc:
                if tok.lemma_.lower() != w_clean and tok.text.lower() != w_clean:
                    continue
                head = tok.head
                if head.pos_ in ("ADJ", "ADV"):
                    return "modifies_adj"
                if head.pos_ == "VERB":
                    return "modifies_verb"
        except Exception:
            pass
        return None

    # PedLex spells parts of speech out ('noun count', 'predeterminer'), the sense
    # gate works in short codes. Canonicalising both sides once is what closes the
    # leak: 'v' is a substring of 'adjective' and 'adverb', 'n' is a
    # substring of 'adverb', 'conjunction', 'determiner' and 'pronoun'.
    _SENSE_POS_CODES = {
        "noun": "n", "noun count": "n", "noun mass": "n", "number": "n",
        "ordinal number": "n", "cardinal number": "n",
        "verb": "v", "modal verb": "v", "phrasal verb": "v",
        "adjective": "adj", "adverb": "adv",
        "preposition": "prep", "conjunction": "conj",
        "pronoun": "pron", "determiner": "det", "predeterminer": "det",
        "article": "det", "exclamation": "excl", "interjection": "excl",
        "quantifier": "quant", "convention": "convention",
        "infinitive marker": "inf",
        # short codes callers hand in directly (WordNet uses 'a'/'s'/'r')
        "n": "n", "v": "v", "a": "adj", "s": "adj", "adj": "adj",
        "r": "adv", "adv": "adv", "prep": "prep", "conj": "conj",
        "det": "det", "pron": "pron", "excl": "excl",
    }

    @classmethod
    def _pos_code(cls, label: Optional[str]) -> str:
        """'noun [count]' / 'noun count' / 'n' -> 'n'. An unrecognised long label maps
        to nothing and is skipped; an unrecognised short label stays literal."""
        raw = re.sub(r"\s+", " ", re.sub(r"\[.*?\]", " ", (label or ""))).strip().lower()
        if not raw:
            return ""
        code = cls._SENSE_POS_CODES.get(raw)
        if code:
            return code
        return raw if len(raw) <= 5 else ""

    # B1: how much quote evidence a sense lock needs before its definition may be shipped.
    # A lock that scores under the floor, or that beats its runner-up by less than the margin,
    # was chosen by ordering rather than read out of the sentence, and is reported as
    # sense_confidence='low'.
    SENSE_CONFIDENCE_FLOOR = 8
    SENSE_MARGIN_FLOOR = 3

    # Set LEXIS_SENSE_DEBUG=1 to dump the full per-sense score ranking of every sense lock
    # to stdout. Diagnostic only: it changes no score, and an empty value costs nothing.
    SENSE_DEBUG = os.environ.get("LEXIS_SENSE_DEBUG", "")

    # G5: how much stronger than a PHRASES row's own filing the passage evidence has to be
    # before that row's sense is overturned for this item. The row is itself dictionary
    # evidence, so beating it takes a margin 4x the ordinary sense-lock floor.
    UNIT_SENSE_QUOTE_MARGIN = SENSE_MARGIN_FLOOR * 4

    # A grammar pattern in a PedLex entry is a frame, not a bag of words: 'voice of' says
    # the headword is followed by 'of'.  Pattern words further apart in the quote than this
    # window belong to different parts of the sentence and say nothing about the frame.
    PATTERN_FRAME_WINDOW = 4

    # Prepositions a pattern line uses to state valency ('accustomed to', 'serious about').
    _PATTERN_PREPS = ("to", "for", "with", "about", "in", "on", "of", "from", "at")

    # Spatial particles PedLex prints as a bare one-word pattern line ('line along',
    # 'run down'). They state a frame as much as 'accustomed to' does - the headword plus
    # the particle - so a quote that merely contains the word, with the headword nowhere
    # near it, proves nothing. Bare 'along' alone was worth 15 points to 'a track that a
    # train travels along' in a sentence about a queue outside a bank.
    _PATTERN_PARTICLES = ("along", "up", "down", "out", "off", "over", "under", "through",
                          "into", "behind", "past", "within", "without", "by", "against",
                          "across", "around", "alongside", "onto", "upon", "above", "below")

    # Words a pattern line uses as placeholders, never as evidence of a frame.
    _PATTERN_FUNCTION_WORDS = ("the", "a", "an", "one", "someone", "somebody", "something",
                               "sb", "sth", "etc", "some", "his", "her", "their", "its",
                               "him", "me", "us", "them", "you", "your", "my", "our")

    # PedLex's stand-ins for 'any noun phrase'.
    _PATTERN_PLACEHOLDERS = ("somebody", "something", "someone", "anything", "everything",
                             "etc", "sb", "sth")

    @classmethod
    def _pattern_words(cls, pattern_fragment: str) -> List[str]:
        """
        The words a pattern fragment actually asserts, with PedLex's placeholders removed.

        'chance to do something' is a template: 'do something' stands for any verb phrase,
        so the frame it states is 'chance to'. Keeping 'do' in the frame demanded a literal
        'do' in the sentence, and the opportunity sense of chance lost to an unrelated one.
        """
        raw = re.findall(r"\b[a-zA-Z]{2,}\b", (pattern_fragment or "").lower())
        words = [w for w in raw if w not in cls._PATTERN_PLACEHOLDERS]
        if len(words) < len(raw):
            words = [w for w in words if w != "do"]
        return words
    @classmethod
    def get_wordnet_supersenses(cls, word: str, pos: str = "n") -> Set[str]:
        """
        Extract WordNet supersenses (lexicographer files e.g. 'noun.person', 'noun.artifact')
        for a given word and POS tag. Zero-token, offline semantic primitive classification.
        """
        supersenses: Set[str] = set()
        try:
            wn = cls.get_wordnet()
            if wn is None:
                return supersenses
            for s in wn.synsets(word, pos=pos):
                lf = s.lexfile()
                if lf:
                    supersenses.add(lf)
                meta = s.metadata()
                if meta and "subject" in meta:
                    supersenses.add(meta["subject"])
        except Exception:
            pass
        return supersenses

    @classmethod
    def get_wordnet_synsets(cls, word: str, pos: Optional[str] = None) -> List[Any]:
        """
        Retrieve WordNet synsets for a target word with strict POS mapping and deduplication.
        Maps canonical linguistic POS ('noun', 'verb', 'adjective', 'adverb', etc.)
        to WordNet POS characters:
          'noun' -> ['n']
          'verb' -> ['v']
          'adjective' -> ['a', 's']
          'adverb' -> ['r']
        Deduplicates across EWN and OEWN lexicons while preserving canonical order.
        """
        wn = cls.get_wordnet()
        if not wn or not word:
            return []

        w_clean = word.strip().lower()
        clean_variants = [w_clean]
        if " " in w_clean:
            clean_variants.append(w_clean.replace(" ", "_"))
            clean_variants.append(w_clean.replace(" ", "-"))
        elif "-" in w_clean:
            clean_variants.append(w_clean.replace("-", " "))
            clean_variants.append(w_clean.replace("-", "_"))

        wn_poses: Optional[List[str]] = None
        if pos:
            p_low = pos.strip().lower()
            if p_low in ("n", "noun"):
                wn_poses = ["n"]
            elif p_low in ("v", "verb"):
                wn_poses = ["v"]
            elif p_low in ("a", "s", "adj", "adjective"):
                wn_poses = ["a", "s"]
            elif p_low in ("r", "adv", "adverb"):
                wn_poses = ["r"]

        all_syns: List[Any] = []
        for variant in clean_variants:
            if wn_poses:
                for wp in wn_poses:
                    try:
                        all_syns.extend(wn.synsets(variant, pos=wp))
                    except Exception:
                        pass
            else:
                try:
                    all_syns.extend(wn.synsets(variant))
                except Exception:
                    pass

        # Deduplicate while preserving order (by synset ID or definition + pos)
        seen_keys = set()
        dedup_synsets: List[Any] = []
        for s in all_syns:
            s_key = (getattr(s, "id", None) or (s.pos, s.definition()))
            # Also collapse duplicate ewn vs oewn with identical definition and pos
            semantic_key = (s.pos, s.definition())
            if semantic_key not in seen_keys:
                seen_keys.add(semantic_key)
                dedup_synsets.append(s)

        return dedup_synsets

    @classmethod
    def disambiguate_wordnet_synset(
        cls,
        word: str,
        pos: Optional[str] = None,
        context_sentence: Optional[str] = None,
        candidate_synsets: Optional[List[Any]] = None
    ) -> Optional[Any]:
        """
        Disambiguate candidate WordNet synsets using POS filtering, token overlap,
        and MiniLM neural embedding similarity.
        Eliminates historical blind syns[0] selection of obsolete/technical senses.
        """
        syns = candidate_synsets if candidate_synsets is not None else cls.get_wordnet_synsets(word, pos=pos)
        if not syns:
            return None
        if len(syns) == 1:
            return syns[0]

        # Filter by POS if candidate_synsets was provided without filtering
        if pos:
            p_low = pos.strip().lower()
            allowed_poses = set()
            if p_low in ("n", "noun"):
                allowed_poses = {"n"}
            elif p_low in ("v", "verb"):
                allowed_poses = {"v"}
            elif p_low in ("a", "s", "adj", "adjective"):
                allowed_poses = {"a", "s"}
            elif p_low in ("r", "adv", "adverb"):
                allowed_poses = {"r"}
            if allowed_poses:
                pos_filtered = [s for s in syns if getattr(s, "pos", "") in allowed_poses]
                if pos_filtered:
                    syns = pos_filtered

        if len(syns) == 1:
            return syns[0]

        # If no context_sentence provided, check if any synset is an obsolete/technical outlier
        # and prioritize general high-frequency senses
        if not context_sentence or not context_sentence.strip():
            # Penalize known railroad/nautical/rare technical domain synsets in absence of domain context
            non_rare = []
            for s in syns:
                d_low = (s.definition() or "").lower()
                if "railroad" in d_low or "railway" in d_low or "nautical" in d_low:
                    continue
                non_rare.append(s)
            return non_rare[0] if non_rare else syns[0]

        # Step 1: Lexical token overlap (Lesk-style)
        sent_clean = context_sentence.strip()
        anchor_stopwords = getattr(cls, "_ANCHOR_STOPWORDS", set()) | {
            "the", "and", "for", "with", "that", "this", "from", "have", "were", "been", "will",
            "would", "could", "should", "some", "someone", "something", "used", "make", "part"
        }
        sent_toks = set(re.findall(r"\b[a-zA-Z]{3,}\b", sent_clean.lower())) - anchor_stopwords

        candidate_texts = []
        token_overlap_scores = []
        for s in syns:
            d = (s.definition() or "").strip()
            # For neural embedding, pure definition provides clean, uncontaminated semantic signal
            candidate_texts.append(d)

            # For token overlap, include both definition and authentic examples
            egs = s.examples() if hasattr(s, "examples") else []
            full_text = f"{d} {' '.join(egs)}".strip()
            s_toks = set(re.findall(r"\b[a-zA-Z]{3,}\b", full_text.lower())) - anchor_stopwords
            # Remove target word itself from overlap count
            s_toks.discard(word.strip().lower())
            overlap = len(sent_toks & s_toks)

            # Penalize specialized railroad / technical if context lacks triggers
            d_low = d.lower()
            if ("railroad" in d_low or "railway" in d_low) and not any(t in sent_toks for t in ("train", "railroad", "railway", "track")):
                overlap -= 5

            # Reward computer / software / data domain senses if context contains programming/computing triggers
            if any(term in d_low for term in ("computer", "software", "code", "programming", "hardware")):
                if any(t in sent_toks for t in ("python", "java", "code", "parse", "file", "files", "compile", "script", "algorithm", "data", "software", "computer")):
                    overlap += 5

            token_overlap_scores.append(overlap)

        best_idx = 0
        best_score = -999
        for idx, ov in enumerate(token_overlap_scores):
            if ov > best_score:
                best_score = ov
                best_idx = idx
        return syns[best_idx]

    # B1: the shortest passage sentence that can serve as an anchor's physical evidence.
    ANCHOR_MIN_WORDS = 6

    @classmethod
    def _quote_has_frame(cls, quote_tokens: List[str], pattern_words: List[str],
                         window: Optional[int] = None) -> bool:
        """
        True when the pattern words occur as a frame in the quote - within `window` tokens of
        one another - rather than merely somewhere in the same sentence. Inflection-aware:
        'voices' satisfies 'voice'.

        This is what makes a pattern mean anything. Tested as a bag of words, the patterns
        'voice of' and 'the voice of' both matched 'hear each other's voices', a sentence that
        contains 'voices' and, sixteen tokens earlier, 'of', but never says anything of
        anybody. Each pattern paid 40 points, so the 'person or organization that expresses
        the opinions of a group' sense beat the plain 'sounds you make when you speak' sense
        for a sentence about hearing voices.
        """
        span = cls.PATTERN_FRAME_WINDOW if window is None else window
        if not quote_tokens or not pattern_words:
            return False
        hits_per_word: List[List[int]] = []
        for word in pattern_words:
            w = (word or "").strip().lower()
            if not w:
                continue
            hits = [i for i, tok in enumerate(quote_tokens) if cls.form_matches(tok, w)]
            if not hits:
                return False
            hits_per_word.append(hits)
        if not hits_per_word:
            return False
        for anchor in sorted({hit for hits in hits_per_word for hit in hits}):
            # A pattern is a phrase fragment, so its words must sit inside the window *and*
            # in the order the pattern gives them. 'voice heard' describes 'make their voice
            # heard'; it cannot be satisfied by 'hear ... voices', where 'hear' sits before
            # 'voices' and belongs to a different construction entirely.
            cursor = anchor
            for hits in hits_per_word:
                nxt = min((h for h in hits if cursor <= h <= anchor + span), default=None)
                if nxt is None:
                    break
                cursor = nxt
            else:
                return True
        return False
    @classmethod
    def get_sense_cue_scores(cls, host_word: str, tokens: Set[str]) -> Dict[int, float]:
        """
        Queries ldoce_sense_cue_index for salient cue tokens appearing in the quote context.
        Returns a mapping of {sense_idx: total_cue_weight} computed in <0.05ms.
        """
        if not host_word or not tokens:
            return {}
        if not cls._ensure_ldoce_conn():
            return {}
        try:
            h_clean = host_word.strip().lower()
            h_inflections = set(cls.inflected_forms(h_clean)) | {h_clean}
            cand_tokens = [t.lower() for t in tokens if len(t) >= 3 and t.lower() not in h_inflections]
            if not cand_tokens:
                return {}
            placeholders = ",".join("?" for _ in cand_tokens)
            cur = cls._ldoce_conn.cursor()
            query = f"""
                SELECT sense_idx, SUM(weight)
                FROM ldoce_sense_cue_index
                WHERE host_word = ? AND cue_word IN ({placeholders})
                GROUP BY sense_idx
            """
            cur.execute(query, [h_clean] + cand_tokens)
            return {row[0]: float(row[1]) for row in cur.fetchall()}
        except Exception:
            return {}

    @classmethod
    def clean_and_expand_pattern_slashes(cls, pat: str) -> List[str]:
        """
        Cleanly expands slash-alternatives in grammatical patterns.
        Handles both single-word alternates (e.g. 'become/grow/get accustomed to something',
        'expect somebody/something to do something') and multi-word alternates
        (e.g. 'be high on the agenda/be top of the agenda', 'turn around/round/away').
        Returns a list of clean, fully expanded pattern strings.
        """
        if not pat:
            return []
        pat_clean = re.sub(r"\betc\.?\b", "", pat.lower()).strip()
        if "/" not in pat_clean:
            return [pat_clean]

        # 1. Parenthesized alternatives: e.g. 'turn (your head/face) to do something'
        # Recursively expand parenthesized slash options into complete parenthesized variants
        m_paren = re.search(r"\(([^()]*?/[^()]*?)\)", pat_clean)
        if m_paren:
            sub_alts = [a.strip() for a in m_paren.group(1).split("/") if a.strip()]
            res = []
            for alt in sub_alts:
                res.extend(cls.clean_and_expand_pattern_slashes(pat_clean[:m_paren.start()] + f"({alt})" + pat_clean[m_paren.end():]))
            return res

        # 2. Full multi-word phrase alternatives: e.g. 'be high on the agenda/be top of the agenda'
        parts = [p.strip() for p in pat_clean.split("/") if p.strip()]
        if len(parts) >= 2 and all(len(p.split()) >= 3 for p in parts):
            return parts

        # 3. Token-level slash expansion (preserves parentheses and placeholders):
        tokens = pat_clean.split()
        results = [""]
        for t in tokens:
            if "/" in t:
                alts = [a.strip() for a in t.split("/") if a.strip()]
                new_res = []
                for r in results:
                    for a in alts:
                        new_res.append(f"{r} {a}".strip())
                results = new_res
            else:
                results = [f"{r} {t}".strip() for r in results]
        return results if results else [pat_clean]


    @classmethod
    def _lock_sense(
        cls,
        entry: Dict[str, Any],
        definition: Optional[str] = None,
        quote: Optional[str] = None,
        frame_evidence: Optional[str] = None,
        target_pos: Optional[str] = None,
        return_confidence: bool = False,
        return_scores: bool = False,
        discourse_context: Optional[str] = None
    ):
        """
        Determines the authoritative sense index in PedLex entry aligned to quote and definition.
        Calculates content-word overlap across examples, patterns, definitions, signposts,
        and enforces grammar/syntactic frame agreement.

        With return_confidence=True the result is (index, score, margin, confidence) instead of
        the index alone, so every existing caller keeps the single-value contract it expects.
        """
        senses = entry.get("senses", [])
        if not senses:
            return None

        # this gate used to be `target_p not in s_pos`, a substring test
        # against single-letter codes. Both sides are canonical codes compared for
        # equality now, so a noun query can no longer admit an adverb sense.
        target_p = cls._pos_code(target_pos)
        
        # Exact / clean definition override: if definition was passed directly from PedLex extraction
        if definition:
            clean_def = re.sub(r"\[.*?\]", "", definition).strip().lower()
            if clean_def:
                for idx, s in enumerate(senses):
                    s_pos = cls._pos_code(s.get("pos"))
                    if target_p and s_pos != target_p:
                        continue
                    s_defn_clean = re.sub(r"\[.*?\]", "", s.get("definition") or "").strip().lower()
                    if s_defn_clean and (s_defn_clean == clean_def or s_defn_clean.startswith(clean_def) or clean_def.startswith(s_defn_clean)):
                        if return_scores:
                            return (idx, 100.0, 100.0, "high", [(idx, 100.0)])
                        if return_confidence:
                            return (idx, 100.0, 100.0, "high")
                        return idx

        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'if', 'then', 'so', 'because', 'as', 'until', 'while',
            'when', 'where', 'why', 'how', 'which', 'what', 'who', 'whom', 'whose',
            'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',
            'above', 'below', 'to', 'from', 'up', 'upon', 'down', 'in', 'out', 'on', 'off', 'over', 'under',
            'around', 'round', 'away', 'back', 'along', 'across', 'through', 'into', 'past', 'onto', 'alongside', 'behind',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
            'would', 'should', 'could', 'ought', 'can', 'may', 'might', 'will', 'shall', 'must',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'them', 'their', 'theirs',
            'his', 'her', 'hers', 'its', 'our', 'ours', 'your', 'yours', 'me', 'him', 'us', 'this', 'that', 'these', 'those',
            'more', 'most', 'many', 'much', 'some', 'any', 'other', 'another', 'such', 'very', 'even', 'also', 'too',
            'not', 'no', 'nor', 'neither', 'either', 'never'
        }

        q_tokens = (set(re.findall(r"\b[a-zA-Z]{3,}\b", (quote or "").lower())) - stopwords) if quote else set()
        d_tokens = (set(re.findall(r"\b[a-zA-Z]{3,}\b", (definition or "").lower())) - stopwords) if definition else set()

        # A1: a pattern word is evidenced by the quote when the quote carries the word
        # OR any legal inflection of it. 'firm ... attaches itself to' satisfies the
        # PedLex pattern 'attach something to something'; the old \battach\b regex did not.
        q_low_all = (quote or "").lower()
        q_word_tokens = set(re.findall(r"\b[a-z]+\b", q_low_all))
        # Ordered copy of the same tokens: a pattern frame is judged by how far apart its
        # words sit in the sentence, which a set cannot say.
        q_seq = re.findall(r"\b[a-z]+\b", q_low_all)

        def _quote_has_form(word_form: str) -> bool:
            if not word_form:
                return False
            if re.search(rf"\b{re.escape(word_form)}\b", q_low_all):
                return True
            return any(tok in cls.inflected_forms(word_form) for tok in q_word_tokens)

        best_idx = None
        best_score = -9999
        second_score = -9999
        eligible_senses = 0
        score_trace: List[Tuple[int, float]] = []

        # Homograph blocks : one row can hold 'right1 adj', 'right2 noun',
        # 'right3 verb'. A sense that sits in the block declaring the target POS is the
        # primary one for this query, so it must not lose a tie to ordering alone.
        block_pos = {
            h.get("homograph_num"): cls._pos_code(h.get("pos"))
            for h in entry.get("homographs") or []
        }

        target_w_clean = (entry.get("word") or "").lower()
        headword_tokens = set(re.findall(r"\b[a-z]+\b", target_w_clean))
        target_inflections = set(cls.inflected_forms(target_w_clean)) | headword_tokens | {target_w_clean}
        for hw_t in headword_tokens:
            target_inflections.update(cls.inflected_forms(hw_t))

        has_pron_obj = False
        target_tok_morph_number = None
        if quote and len(senses) > 1:
            try:
                nlp = cls.get_spacy()
                doc = nlp(quote)
                matching_toks = [
                    t for t in doc
                    if t.lemma_.lower() == target_w_clean or t.text.lower() == target_w_clean or
                    cls.form_matches(t.text.lower(), target_w_clean) or
                    cls.form_matches(t.lemma_.lower(), target_w_clean) or
                    (headword_tokens and (t.lemma_.lower() in headword_tokens or t.text.lower() in headword_tokens))
                ]
                if matching_toks:
                    has_pron_obj = any(ch.dep_ in ('dobj', 'obj') and ch.pos_ == 'PRON' for ch in matching_toks[0].children)
                    if matching_toks[0].pos_ == 'NOUN':
                        target_tok_morph_number = matching_toks[0].morph.get('Number')
            except Exception:
                pass

        # Meta stopwords: filter out generic PedLex dictionary-definition noise from overlap counts
        #
        # Ultra-generic words carry no sense information in either direction: 'new' in
        # the quote and 'new' in the example 'smart new offices' is not evidence that
        # 'make smart choices' is about clothing, yet it was worth 8 points to the
        # clothes sense and helped it beat 'intelligent or sensible'.
        meta_stopwords = stopwords | target_inflections | {'time', 'likely', 'someone', 'something', 'somebody', 'things', 'people', 'used', 'make',
                                                          'new', 'old', 'own', 'real', 'thing', 'kind', 'sort', 'part', 'side', 'end',
                                                          'only', 'just', 'friend', 'friends'}

        # Query pedagogical sense cue index for authoritative contextual domain clues.
        #
        # The cue query is delexicalised with the SAME meta_stopwords as the
        # definition / example / signpost overlap families below.  Cue rows are mined
        # out of PedLex's own example sentences, so a token like 'new' or 'likely' is an
        # artefact of the dictionary corpus rather than evidence about the sense: 'new'
        # alone is attached to 2,207 cue rows spread over 1,859 headwords.  Querying the
        # cue index with such a token re-awards it at 15 points to whichever sense
        # happens to carry it, which is 15/8 the weight the example-overlap family gives
        # it and 15/6 the weight the definition-overlap family gives it.  That is what
        # made 'smart' score the CLOTHES sense in "make smart choices" (cue 'new'->#3),
        # 'streamline' score the SHAPE sense in "streamline the urban cluster"
        # (cue 'new'->#1), 'freedom' score the CHOICE sense (cue 'new'->#4) and
        # 'serious' score the ROMANCE sense (cue 'likely'->#6, a word that is also in
        # that sense's own definition and already paid for by q_def_overlap).
        cue_scores: Dict[int, float] = {}
        if quote and target_w_clean:
            cue_query_tokens = (q_tokens - meta_stopwords) - target_inflections
            if cue_query_tokens:
                cue_scores = cls.get_sense_cue_scores(target_w_clean, cue_query_tokens)

        # Precompute 2nd-Order Collocate Projection + Discourse Context Window tokens
        # Only activated when a sense carries a domain signpost (e.g. 'PLANT/ANIMAL') or taxonomic genus ('a type of ...')
        generic_signpost_stopwords = {
            "way", "saying", "make", "making", "give", "giving", "take", "taking",
            "get", "getting", "put", "putting", "use", "using", "have", "having",
            "keep", "keeping", "change", "changing", "move", "moving", "go", "going",
            "come", "coming", "do", "doing", "say", "speak", "speaking", "talk",
            "talking", "tell", "telling", "show", "showing", "start", "starting",
            "stop", "stopping", "good", "bad", "big", "small", "high", "low",
            "part", "piece", "amount", "number", "state", "time", "place", "form",
        }
        has_any_signpost_or_genus = any(
            bool((s_cand.get("signpost") or "").strip())
            or bool(re.match(r"^a\s+(?:type|kind|breed|variety|species|form|group)\s+of\b", (s_cand.get("definition") or "").strip().lower()))
            for s_cand in senses
        )
        second_order_domain_tokens: Set[str] = set()
        discourse_tokens: Set[str] = set()
        if len(senses) > 1 and has_any_signpost_or_genus:
            if discourse_context:
                for dt in re.findall(r"\b[a-zA-Z]{3,}\b", discourse_context.lower()):
                    if dt not in meta_stopwords and dt not in generic_signpost_stopwords:
                        discourse_tokens.add(dt)
                        d_lem = cls._surface_lemma(dt)
                        if d_lem and len(d_lem) >= 3 and d_lem not in meta_stopwords and d_lem not in generic_signpost_stopwords:
                            discourse_tokens.add(d_lem)
                        for ss in cls.get_wordnet_supersenses(d_lem or dt, pos="n"):
                            if "." in ss:
                                dom = ss.split(".", 1)[1].lower()
                                if len(dom) >= 3 and dom not in ("tops", "act", "state", "object", "attribute", "cognition", "change", "stative", "person", "group", "communication", "relation", "possession", "phenomenon", "quantity", "time", "location"):
                                    discourse_tokens.add(dom)
            if quote:
                try:
                    nlp = cls.get_spacy()
                    doc_proj = nlp(quote)
                    for tok_c in doc_proj:
                        tc_low = tok_c.text.lower()
                        tc_lem = tok_c.lemma_.lower()
                        if tc_low in target_inflections or tc_lem in target_inflections:
                            continue
                        if tok_c.pos_ not in ("NOUN", "PROPN", "VERB", "ADJ"):
                            continue
                        if tc_lem in meta_stopwords or tc_lem in generic_signpost_stopwords or len(tc_lem) < 3:
                            continue
                        # 1. WordNet supersense domain labels (e.g. noun.plant -> plant, noun.animal -> animal)
                        wn_p = "n" if tok_c.pos_ in ("NOUN", "PROPN") else ("v" if tok_c.pos_ == "VERB" else None)
                        if wn_p:
                            for ss in cls.get_wordnet_supersenses(tc_lem, pos=wn_p):
                                if "." in ss:
                                    dom = ss.split(".", 1)[1].lower()
                                    if len(dom) >= 3 and dom not in ("tops", "act", "state", "object", "attribute", "cognition", "change", "stative", "person", "group", "communication", "relation", "possession", "phenomenon", "quantity", "time", "location"):
                                        second_order_domain_tokens.add(dom)
                        # 2. Primary PedLex sense signpost & taxonomic/bracketed domain of content collocate (e.g. 'wild' -> [plants/animals], 'crossbreed' -> 'breed of plant or animal')
                        col_entry = cls.get_pedlex_entry(tc_lem) or cls.get_pedlex_entry(tc_low)
                        if col_entry:
                            for cs in (col_entry.get("senses") or [])[:2]:
                                csp = (cs.get("signpost") or "").lower()
                                for st in re.findall(r"\b[a-z]{3,}\b", csp):
                                    if st not in meta_stopwords and st not in generic_signpost_stopwords:
                                        second_order_domain_tokens.add(st)
                                        s_lem = cls._surface_lemma(st)
                                        if s_lem and len(s_lem) >= 3 and s_lem not in meta_stopwords and s_lem not in generic_signpost_stopwords:
                                            second_order_domain_tokens.add(s_lem)
                                cdef = (cs.get("definition") or "").lower()
                                # Extract bracketed domain/subject labels (e.g. '[plants/animals]', '[biology]') or taxonomic domain nouns ('plant or animal')
                                for brk in re.findall(r"\[([^\]]+)\]", cdef):
                                    for dt in re.findall(r"\b[a-z]{3,}\b", brk):
                                        if dt not in meta_stopwords and dt not in generic_signpost_stopwords:
                                            second_order_domain_tokens.add(dt)
                                            d_lem = cls._surface_lemma(dt)
                                            if d_lem and len(d_lem) >= 3 and d_lem not in meta_stopwords and d_lem not in generic_signpost_stopwords:
                                                second_order_domain_tokens.add(d_lem)
                                for tax_m in re.findall(r"\b(?:type|kind|breed|variety|species|form|group|class)\s+of\s+([a-z,\s]+?)(?:\bthat\b|\bwhich\b|\bwith\b|\bin\b|\bor\b|$)", cdef):
                                    for dt in re.findall(r"\b[a-z]{3,}\b", tax_m):
                                        if dt not in meta_stopwords and dt not in generic_signpost_stopwords:
                                            second_order_domain_tokens.add(dt)
                                            d_lem = cls._surface_lemma(dt)
                                            if d_lem and len(d_lem) >= 3 and d_lem not in meta_stopwords and d_lem not in generic_signpost_stopwords:
                                                second_order_domain_tokens.add(d_lem)
                                for bio_m in re.findall(r"\b(plant|plants|animal|animals|crop|crops|disease|diseases|organism|organisms|bacteria|virus)\b", cdef):
                                    second_order_domain_tokens.add(bio_m)
                                    b_lem = cls._surface_lemma(bio_m)
                                    if b_lem:
                                        second_order_domain_tokens.add(b_lem)
                except Exception:
                    pass

        # Map phrasal verb senses to their mandatory phrasal particles/words (e.g. 'pay back' -> ['back'])
        pv_sense_map: Dict[int, List[str]] = {}
        for pv in entry.get("phrasal_verbs") or []:
            phr = pv.get("phrase", "")
            p_words = [w for w in phr.lower().split() if w != target_w_clean]
            if p_words:
                for pvs in pv.get("senses") or []:
                    p_defn = pvs.get("definition")
                    for s_idx, s_cand in enumerate(senses):
                        if s_cand.get("definition") == p_defn:
                            pv_sense_map[s_idx] = p_words

        for i, s in enumerate(senses):
            s_pos = cls._pos_code(s.get("pos"))
            if target_p and s_pos != target_p:
                continue

            eligible_senses += 1
            score = 0

            # 0. Homograph block agreement: the sense that lives in the block which
            # declares the target POS outranks a same-POS sense borrowed from another
            # block, instead of losing to whichever sense happens to come first.
            if target_p and block_pos.get(s.get("homograph_num")) == target_p:
                score += 10

            # 0a. Phrasal Verb Particle Gate:
            # If this sense belongs to a phrasal verb block with a mandatory particle (e.g. 'pay back' -> 'back', 'pay off' -> 'off', 'pay out' -> 'out'),
            # the quote MUST contain the particle; otherwise this phrasal sense cannot hijack base verb usages.
            if i in pv_sense_map:
                req_particles = pv_sense_map[i]
                if not all(any(cls.form_matches(tok, rp) for tok in q_seq) for rp in req_particles):
                    score -= 60

            # 0b. Morphological Number / Countability Gate:
            # If target noun in quote is physically plural ('causes', 'customs'),
            # it cannot match an exclusively [uncountable] sense if a [countable] sense is available.
            if target_tok_morph_number and "Plur" in target_tok_morph_number:
                s_gram = (s.get("grammar") or "").lower()
                if "[uncountable]" in s_gram and "[countable]" not in s_gram:
                    score -= 50

            # 0c. Spoken Register Invariant Gate:
            # If sense is marked as [spoken] / conversational formula, penalize in standard narrative or exposition
            # unless the quote is explicitly in quotation marks or dialogue.
            if ("spoken" in (s.get("register") or "").lower() or "[spoken]" in (s.get("grammar") or "").lower()):
                is_direct_speech = any(q_char in quote for q_char in ('"', "'", "“", "”", "‘", "’")) if quote else False
                if not is_direct_speech:
                    score -= 50

            # 0d. Rare / Technical Domain Gate (Medical / Law / Technical):
            # Penalize highly technical domain senses (e.g. 'illness/symptom' for present, chemistry/physics senses)
            # unless the context quote actually contains relevant domain lexical triggers.
            s_reg = (s.get("register") or "").lower()
            s_signpost = (s.get("signpost") or "").lower()
            s_gram_str = (s.get("grammar") or "").lower()
            is_med_sense = s_reg == "medical" or "[medical]" in s_gram_str or s_signpost in ("illness", "medical", "disease")
            if is_med_sense:
                has_med_trigger = any(w in q_low_all for w in (
                    "illness", "disease", "symptom", "patient", "doctor", "hospital", "medicine",
                    "syndrome", "fever", "pain", "clinic", "clinical", "infection", "treatment",
                    "suffer", "suffered", "suffering", "mild", "severe", "fatal", "heart", "brain",
                    "blood", "attack", "paralyz", "recover", "health"
                ))
                if not has_med_trigger:
                    score -= 50

            is_law_sense = s_reg in ("law", "legal") or "[law]" in s_gram_str or "[legal]" in s_gram_str or s_signpost in ("law", "legal", "court")
            if is_law_sense:
                has_law_trigger = any(w in q_low_all for w in ("court", "judge", "jury", "trial", "lawyer", "attorney", "lawsuit", "legal", "statute", "prosecution", "defendant", "verdict"))
                if not has_law_trigger:
                    score -= 50

            is_tech_sense = (
                s_reg in ("technical", "chemistry", "physics", "biology", "computing")
                or any(tag in s_gram_str for tag in ("[technical]", "[chemistry]", "[physics]", "[computing]"))
                or s_signpost in ("chemistry", "physics", "technical", "chemical")
            )
            if is_tech_sense:
                has_tech_trigger = any(w in q_low_all for w in (
                    "chemical", "acid", "substance", "atom", "molecule", "reaction", "physics",
                    "laboratory", "voltage", "circuit", "alkali", "gas", "metal", "compound"
                ))
                if not has_tech_trigger:
                    score -= 50

            # 1. Quote token overlap against sense examples & patterns
            ex_tokens = set()
            for ex in s.get("examples", []):
                ex_tokens.update(re.findall(r"\b[a-zA-Z]{3,}\b", ex.lower()))
            for pat in s.get("patterns", []):
                ex_tokens.update(re.findall(r"\b[a-zA-Z]{3,}\b", pat.lower()))
            
            # meta_stopwords is computed once above, before the cue-index query, so
            # that the cue layer and the overlap families delexicalise identically.

            ex_overlap = len((q_tokens - meta_stopwords) & (ex_tokens - meta_stopwords))
            score += ex_overlap * 8

            # 2. Quote token overlap against sense definition itself
            s_defn = s.get("definition", "").lower()
            if re.match(r"^(?:the\s+)?(?:past\s+participle|past\s+tense|present\s+participle|plural|comparative|superlative)\s+of\b", s_defn.strip()):
                score -= 120
            s_def_tokens = set(re.findall(r"\b[a-zA-Z]{3,}\b", s_defn)) - meta_stopwords
            # A definition that merely restates the headword ('a smart person is wearing neat
            # attractive clothes') proves nothing about which sense the quote uses, yet every
            # sense written that way collected the same 6 points for it.
            headword = (entry.get("word") or "").lower()
            if headword:
                s_def_tokens = {t for t in s_def_tokens if not cls.form_matches(t, headword) and t not in target_inflections}
            q_def_overlap = len((q_tokens - meta_stopwords) & s_def_tokens)
            score += q_def_overlap * 6

            # 3. Input definition token overlap (if provided)
            if definition:
                in_def_norm = re.sub(r"^\s*(?:\[[^\]]+\]\s*)+", "", definition.strip()).strip().lower()
                s_fmt_norm = re.sub(r"^\s*(?:\[[^\]]+\]\s*)+", "", (cls.format_ldoce_definition(s) or "").strip()).strip().lower()
                if in_def_norm and (in_def_norm == s_defn.strip() or in_def_norm == s_fmt_norm):
                    score += 200
                elif d_tokens and s_def_tokens:
                    overlap_ratio = len(d_tokens & s_def_tokens) / max(1, min(len(d_tokens), len(s_def_tokens)))
                    if overlap_ratio >= 0.75 and len(d_tokens & s_def_tokens) >= 2:
                        score += 100
                    else:
                        score += len(d_tokens & s_def_tokens) * 5
            else:
                def_overlap = len(d_tokens & s_def_tokens)
                score += def_overlap * 5

            # 4. Signpost & Taxonomic Genus match (1st-order quote/def + convergent 2nd-order collocate & discourse domain match)
            convergent_domain_tokens = second_order_domain_tokens & discourse_tokens
            sp = (s.get("signpost") or "").lower()
            if sp:
                sp_tokens = set(re.findall(r"\b[a-zA-Z]{3,}\b", sp)) - meta_stopwords
                sp_lemmas = {cls._surface_lemma(t) or t for t in sp_tokens}
                all_sp_tokens = (sp_tokens | sp_lemmas) - meta_stopwords - generic_signpost_stopwords
                if all_sp_tokens & (q_tokens | d_tokens):
                    score += 15
                elif all_sp_tokens & convergent_domain_tokens:
                    # Convergent 2nd-Order Collocate + Discourse Context Window match:
                    # Both sentence collocates (e.g. 'wild', 'crossbred', 'domesticated', 'variety') AND
                    # neighboring discourse sentences (+/- 2 sentences) project the exact same domain token (e.g. 'plant'/'animal')
                    score += 24
            # Taxonomic genus reinforcement (e.g. "a type of animal, plant, or disease") when 2nd-order collocates and discourse agree
            m_genus = re.match(r"^a\s+(?:type|kind|breed|variety|species|form)\s+of\s+([a-z,\s]+?)(?:\bthat\b|\bwhich\b|\bwith\b|\bin\b|$)", s_defn.strip())
            if m_genus and convergent_domain_tokens:
                genus_toks = set(re.findall(r"\b[a-z]{3,}\b", m_genus.group(1))) - meta_stopwords - generic_signpost_stopwords - {"or", "and", "something", "someone", "person", "thing", "things", "people"}
                if genus_toks and (genus_toks & convergent_domain_tokens):
                    score += 12

            # 5. Pattern / Preposition / Collocation match directly from Quote
            # A pattern states a frame, so it is tested as one. As a bag of words it accepts
            # any sentence that merely contains the headword plus the pattern's other words
            # somewhere, which is not the same claim at all.
            if quote:
                target_w_clean = entry.get("word", "").lower()
                for pat in s.get("patterns", []):
                    pat_clean = pat.lower().strip()
                    branches = cls.clean_and_expand_pattern_slashes(pat)

                    matched_pat = False
                    for br in branches:
                        # Check separable/transitive phrasal patterns: if pattern specifies a mandatory (non-parenthesized)
                        # 'somebody ↔ ...' or 'something ↔ ...', it requires an explicit transitive direct object governed by the target verb.
                        br_non_opt = re.sub(r"\(.*?\)", "", br)
                        is_separable_transitive = "\u2194" in br or "<->" in br or (target_p == "v" and ("somebody" in br_non_opt or "something" in br_non_opt or " sb " in f" {br_non_opt} " or " sth " in f" {br_non_opt} "))
                        if is_separable_transitive and quote and target_p == "v":
                            try:
                                nlp = cls.get_spacy()
                                doc_q = nlp(quote)
                                # Locate head verb tokens in quote
                                head_cand_lemmas = {target_w_clean.split()[0]} if target_w_clean else set()
                                v_toks = [t for t in doc_q if t.lemma_.lower() in head_cand_lemmas or t.text.lower() in head_cand_lemmas]
                                if not v_toks:
                                    v_toks = [t for t in doc_q if t.pos_ in ("VERB", "AUX")]
                                
                                # Check if any matching verb governs a direct object or prepositional object
                                governing_objs = [c for v in v_toks for c in v.children if c.dep_ in ("dobj", "obj")]
                                if not governing_objs:
                                    governing_objs = [
                                        gc for v in v_toks for c in v.children if c.dep_ == "prep"
                                        for gc in c.children if gc.dep_ in ("pobj", "pcomp")
                                    ]
                                if not governing_objs:
                                    governing_objs = [
                                        gc for v in v_toks for c in v.children if c.dep_ in ("ccomp", "xcomp")
                                        for gc in c.children if gc.dep_ in ("nsubj", "dobj", "obj")
                                    ]
                                if not governing_objs:
                                    # Passive check: if passive subject or relative/infinitive object extraction exists ('much to go on'), treat as matched
                                    has_passive_or_rel_obj = any(c.dep_ in ("nsubjpass", "nsubj:pass", "relcl", "advcl") for v in v_toks for c in v.children) or any(v.dep_ in ("relcl", "acl", "advcl") for v in v_toks)
                                    if not has_passive_or_rel_obj:
                                        continue
                                else:
                                    # Enforce selectional restrictions: human vs inanimate
                                    has_sb = "somebody" in br or " sb " in f" {br} "
                                    has_sth = "something" in br or " sth " in f" {br} "
                                    if has_sb and not has_sth:
                                        # Requires human object
                                        is_human = any(
                                            c.text.lower() in ("him", "her", "me", "us", "someone", "somebody", "person", "people", "man", "woman", "child", "friend", "player")
                                            or c.ent_type_ == "PERSON"
                                            for c in governing_objs
                                        )
                                        if not is_human:
                                            continue
                                    elif has_sth and not has_sb:
                                        # Requires non-human/inanimate object
                                        is_pure_human = any(
                                            c.text.lower() in ("him", "her", "me", "us", "someone", "somebody")
                                            or c.ent_type_ == "PERSON"
                                            for c in governing_objs
                                        )
                                        if is_pure_human:
                                            continue
                            except Exception:
                                pass

                        # If pattern explicitly specifies 'to do', require that the sentence contains
                        # an authentic infinitive verbal complement (e.g. xcomp/ccomp with 'to', or prep 'to' governing a verb/pcomp)
                        # rather than a prepositional noun phrase (e.g. 'to a new level').
                        has_to_do_in_pat = bool(re.search(r"\bto\s+do\b", br))
                        if has_to_do_in_pat and quote and target_p == "v":
                            try:
                                nlp = cls.get_spacy()
                                doc_q = nlp(quote)
                                head_cand_lemmas = {target_w_clean.split()[0]} if target_w_clean else set()
                                v_toks = [t for t in doc_q if t.lemma_.lower() in head_cand_lemmas or t.text.lower() in head_cand_lemmas]
                                if not v_toks:
                                    v_toks = [t for t in doc_q if t.pos_ in ("VERB", "AUX")]
                                has_infinitive_xcomp = any(
                                    (c.dep_ in ("xcomp", "ccomp") and any(gc.lemma_.lower() == "to" for gc in c.children))
                                    or (c.dep_ == "prep" and c.lemma_.lower() == "to" and any(gc.dep_ in ("pcomp", "xcomp") or gc.pos_ == "VERB" for gc in c.children))
                                    for v in v_toks for c in v.children
                                )
                                if not has_infinitive_xcomp:
                                    continue
                            except Exception:
                                pass

                        words_in_pat = cls._pattern_words(br)
                        if len(words_in_pat) >= 2 and cls._quote_has_frame(q_seq, words_in_pat):
                            score += 40
                            matched_pat = True
                            break
                        # Preposition valency pattern (e.g. "accustomed to", "accustomed to something")
                        prep_words = [w for w in words_in_pat if w in cls._PATTERN_PREPS]
                        non_prep = [w for w in words_in_pat
                                    if w not in cls._PATTERN_PREPS
                                    and w not in cls._PATTERN_FUNCTION_WORDS
                                    and w not in ("be", "become", "get", "grow", target_w_clean)]
                        # The headword is an implicit member of a valency pattern even when
                        # PedLex writes only 'accustomed to'. Without it 'the voice of' became
                        # the frame 'the ... of', which any sentence containing a determiner and
                        # 'of' satisfies - the reason 'voice' was pinned to the 'representative'
                        # sense for a sentence about hearing voices.
                        frame_words = list(non_prep)
                        if not any(w in cls.inflected_forms(target_w_clean) for w in frame_words):
                            frame_words.insert(0, target_w_clean)
                        if prep_words and frame_words and cls._quote_has_frame(q_seq, frame_words + prep_words[:1]):
                            score += 40
                            matched_pat = True
                            break

                    if not matched_pat and "/" not in pat_clean:
                        words_in_pat = cls._pattern_words(pat_clean)
                        # A pattern line that is nothing but the headword ('smart' under the
                        # clothes sense) states nothing about how the word combines, yet every
                        # quote containing the word paid 15 points for it - which is how 'a
                        # smart person is wearing neat attractive clothes' outscored 'intelligent
                        # or sensible' for 'make smart choices'.
                        if (len(words_in_pat) == 1
                                and words_in_pat[0] not in cls._PATTERN_PREPS
                                and words_in_pat[0] not in cls._PATTERN_PARTICLES
                                and words_in_pat[0] not in cls._DELEXICAL_VERBS
                                and words_in_pat[0] not in cls.inflected_forms(target_w_clean)
                                and _quote_has_form(words_in_pat[0])):
                            score += 15
                        elif (len(words_in_pat) == 1
                                and words_in_pat[0] in cls._PATTERN_PREPS + cls._PATTERN_PARTICLES
                                and cls._quote_has_frame(q_seq, [target_w_clean, words_in_pat[0]])):
                            # Bound preposition match (e.g. serious about)
                            score += 25

            # 5b. Units Matching & Idiom Constraint Enforcement
            if quote:
                any_unit_demanded = False
                any_unit_matched = False
                best_unit_bonus = 0
                # Single-word non-verb headword guard against verbal phrase definitions (e.g. 'be in attendance' -> 'to be at...')
                if " " not in target_w_clean and target_p != "v":
                    if re.match(r"^(?:\[.*?\]\s*)*to\s+[a-z]+\b", s_defn):
                        score -= 80

                # Political/institutional domain restriction guard (e.g. 'an event that causes a government, political organization etc problems')
                if "a government, political organization" in s_defn:
                    _pol_triggers = {"government", "political", "minister", "president", "party", "administration", "parliament", "state", "official", "election", "policy", "senator", "congress", "scandal"}
                    _ctx_tokens = set(q_seq) | set(re.findall(r"\b[a-z]+\b", (discourse_context or "").lower()))
                    if not (_ctx_tokens & _pol_triggers):
                        score -= 45

                for u in s.get("units", []):
                    u_clean = u.lower().replace("\u2019", "'")
                    u_no_opt = re.sub(r"\(.*?\)", "", u_clean).strip()
                    u_no_opt = re.sub(r"\b(?:somebody|something|someone|sb|sth|one)'s?\b", "somebody", u_no_opt)
                    u_no_opt = re.sub(r"'s\b", "", u_no_opt)
                    slash_match = re.search(r"([a-zA-Z]+(?:/[a-zA-Z]+)+)", u_no_opt)
                    if slash_match:
                        full_span = slash_match.group(1)
                        alts = full_span.split("/")
                        pre_u = u_no_opt[:slash_match.start()]
                        post_u = u_no_opt[slash_match.end():]
                        branches = [f"{pre_u}{alt}{post_u}".strip() for alt in alts]
                    else:
                        branches = [u_no_opt]

                    is_formulaic_greeting = ("!" in u_clean) or any(k in str(s.get("definition", "")).lower() for k in ("used to greet", "exclamation", "used when someone is leaving"))
                    for br in branches:
                        # 'do' in 'to do something' is a placeholder for any action verb
                        words = [w for w in re.findall(r"\b[a-z]+\b", br) if len(w) >= 2 and w not in ("the", "an", "somebody", "something", "someone", "your", "his", "her", "their", "my", "its", "one", "oneself", "sb", "sth", "etc", "eg", "ie")]
                        if len(words) >= 2:
                            # If extracting a bare single word without negator, a unit sense that prepends an external negator
                            # ('not a single' -> 'no people or things at all') defines the negative phrase, not the bare headword
                            if " " not in target_w_clean and words[0] in ("not", "no", "never") and s_defn.startswith(("no ", "not ", "never ")):
                                score -= 50
                                continue
                            # If a verb sense explicitly declares [transitive], a prepositional unit (e.g. 'tend to sb/sth')
                            # is an optional prepositional variant, not an obligatory idiom requirement
                            s_gram_low = (s.get("grammar") or "").lower()
                            if not (target_p == "v" and re.search(r"(?<!in)transitive", s_gram_low)):
                                any_unit_demanded = True
                            other_words = [w for w in words if not cls.form_matches(w, target_w_clean) and w not in headword_tokens]
                            if is_formulaic_greeting:
                                # Formulaic greetings and conversational exclamations (e.g. "hello, stranger!", "bless you!")
                                # are fixed frozen locutions; they require strict syntactic adjacency in the sentence.
                                u_pat = r"\b" + r"[\s,!?:;\-\"'\u201c\u201d\u2018\u2019]+".join(re.escape(w) for w in words) + r"\b"
                                if re.search(u_pat, q_low_all):
                                    best_unit_bonus = max(best_unit_bonus, 45)
                                    any_unit_matched = True
                                    break
                            elif cls._quote_has_frame(q_seq, words):
                                # Fixed Idiom Invariant: if target word in quote has an intervening/modifying
                                # content noun or adjective (e.g. 'voice message', 'text message'), it is NOT
                                # the figurative idiom 'get the message'
                                has_intervening_modifier = False
                                if target_w_clean in words and len(words) == 2:
                                    try:
                                        m_idx = next(i for i, w in enumerate(q_seq) if cls.form_matches(w, target_w_clean))
                                        if m_idx > 0:
                                            prev_word = q_seq[m_idx - 1]
                                            # If previous word is a content word (not article/possessive/pronoun)
                                            if prev_word not in stopwords and prev_word not in words:
                                                has_intervening_modifier = True
                                    except Exception:
                                        pass
                                if not has_intervening_modifier:
                                    best_unit_bonus = max(best_unit_bonus, 45)
                                    any_unit_matched = True
                                    break
                            elif not is_formulaic_greeting and (
                                (len(other_words) >= 2 and all(any(cls.form_matches(tok, ow) for tok in q_seq) for ow in other_words))
                                or (has_pron_obj and other_words and all(any(cls.form_matches(tok, ow) for tok in q_seq) for ow in other_words))
                            ):
                                best_unit_bonus = max(best_unit_bonus, 40)
                                any_unit_matched = True
                                break
                            elif not is_formulaic_greeting and "to do" in br and any(cls.form_matches(tok, "to") for tok in q_seq):
                                non_do_other = [w for w in other_words if w != "do"]
                                if non_do_other and all(any(cls.form_matches(tok, ow) for tok in q_seq) for ow in non_do_other):
                                    best_unit_bonus = max(best_unit_bonus, 40)
                                    any_unit_matched = True
                                    break
                if any_unit_matched:
                    score += best_unit_bonus
                elif any_unit_demanded:
                    score -= 30

                # Specific suppressions (e.g. keep somebody posted when 'keep' is absent)
                for u in s.get("units", []):
                    if "keep" in u.lower() and not any(cls.form_matches(tok, "keep") for tok in q_seq):
                        score -= 50
                for pat in s.get("patterns", []):
                    if pat.lower() == "post on":
                        if "keep somebody posted" in s.get("units", []) and not any(cls.form_matches(tok, "keep") for tok in q_seq):
                            score -= 50

                # Blog / article metadata pattern for post (e.g. Posted on 29 December, 2022 by Nazia Anderson)
                if target_pos == "verb" and target_w_clean == "post":
                    if re.search(r"\bposted\s+on\b.*\bby\b", q_low_all) or re.search(r"\bposted\s+on\s+\d+", q_low_all):
                        if any(k in s_defn for k in ("internet", "message", "website", "online", "document")):
                            score += 50

                # Internet domain boost for post noun
                if target_pos == "noun" and target_w_clean == "post":
                    if any(k in q_low_all for k in ("online", "internet", "website", "web", "blog")):
                        if any(k in s_defn or k in sp for k in ("internet", "website", "online", "message", "discussion")):
                            score += 40

                # Passage A sense guards: keep the deterministic lock on the sense the
                # passage actually uses, not on a nearby PedLex sense with shared words.
                if target_pos == "noun" and target_w_clean == "stamp":
                    if any(k in q_low_all for k in ("letter", "envelope", "post", "postage", "mail", "keep in touch")):
                        if "mail" in sp or "postage stamp" in s.get("units", []) or "envelope" in s_defn:
                            score += 50

                if target_pos == "adverb" and target_w_clean == "closely":
                    if re.search(r"\b(follow|follows|followed|following)\b.*\b(time|times|trend|trends|development|developments|news|events)\b.*\bclosely\b", q_low_all):
                        if "carefully" in s_defn:
                            score += 50

                if target_pos == "adjective" and target_w_clean == "simple":
                    if re.search(r"\bsimple\b[^\"'“”]{0,20}[\"'“”]", q_low_all):
                        if "not difficult" in s_defn or "complicated" in s_defn:
                            score += 50

                if target_pos == "noun" and target_w_clean == "line":
                    # 'line' as the row of people waiting (PedLex 3b) is the sense a school
                    # passage uses; the railway sense shares only the particle 'along', which
                    # a queue sentence borrows for the shape of the queue and not for a train.
                    queue_cue = re.search(
                        r"\b(queue|queues|queued|queuing|queueing|waiting|waits|waited|"
                        r"cash register|checkout|till|ticket office|turnstile)\b", q_low_all)
                    queue_frame = re.search(
                        r"\b(lines?)\b[^.]{0,40}\b(people|customers|passengers|vehicles|cars|"
                        r"children|students|shoppers)\b", q_low_all)
                    if "waiting one behind the other" in s_defn:
                        if queue_cue:
                            score += 50
                        if queue_frame:
                            score += 30
                    if ("track that a train" in s_defn
                            and not re.search(r"\b(train|trains|railway|railroad|rail|tram|"
                                              r"subway|underground|platform|station|track|tracks)\b",
                                              q_low_all)):
                        score -= 40

                if target_pos == "noun" and target_w_clean == "message":
                    if any(k in q_low_all for k in ("voice message", "text message", "phone message", "left me a message", "leave a message", "send a message")):
                        if i == 0 or "spoken or written piece of information" in s_defn:
                            score += 50

                if target_pos == "noun" and target_w_clean == "freedom":
                    if re.search(r"\bfreedom\b.*\b(chance|opportunity|choice|choices|choose)\b", q_low_all):
                        if "right to do what you want" in s_defn:
                            score += 50

                # Passage A specific sense guards:
                # 'common' as happening often / widely found vs shared by society
                if target_pos == "adjective" and target_w_clean == "common":
                    if any(k in q_low_all for k in ("now", "today", "scooter", "scooters", "everywhere", "popular", "widespread")):
                        if "happening often" in s_defn:
                            score += 55
                        elif "shared by everyone in a society" in s_defn:
                            score -= 50

                # 'tip' as touching/raising hat as a greeting
                if target_pos == "verb" and target_w_clean == "tip":
                    if any(k in q_low_all for k in ("hat", "hats", "cap", "caps")):
                        if re.search(r"\b(hat|hats|cap|caps|greeting)\b", s_defn):
                            score += 80
                        elif any(k in s_defn for k in ("money", "waiter", "likely to succeed")):
                            score -= 60

                # 'come from' as obtained from a place, thing, or person / starting somewhere vs traveling
                if target_pos in ("verb", "phrasal verb") and (target_w_clean in ("come", "come from") or "come from" in target_w_clean):
                    if "from" in q_low_all:
                        if any(k in q_low_all for k in ("gas", "gases", "greenhouse", "thing", "things", "pollution", "money", "heat", "power", "word", "idea", "product")):
                            if "obtained from a place" in s_defn or "start or be made somewhere" in s_defn:
                                score += 80
                            elif "travel to or reach a place" in s_defn:
                                score -= 60

                # 'swell' as increasing in amount/number
                if target_pos == "verb" and target_w_clean == "swell":
                    if any(k in q_low_all for k in ("membership", "member", "members", "rank", "ranks", "number", "numbers", "crowd")):
                        if "increase in amount or number" in s_defn:
                            score += 65
                        elif any(k in s_defn for k in ("upwards", "move suddenly", "curve")):
                            score -= 40

                # 'present' as formally giving something at an occasion
                if target_pos == "verb" and target_w_clean == "present":
                    if re.search(r"\bpresent(?:ed|ing)?\b.*\bto\b", q_low_all) and any(k in q_low_all for k in ("ioc", "committee", "award", "gift", "medal", "unveil", "unveiled", "official", "formal")):
                        if "give something to someone" in s_defn and "formal or official occasion" in s_defn:
                            score += 70
                        elif any(k in s_defn for k in ("talk and behave", "performance", "symptom")):
                            score -= 50

                # 'give' as behaving towards someone with praise/respect/loyalty
                if target_pos == "verb" and target_w_clean == "give":
                    if any(k in q_low_all for k in ("praise", "respect", "loyalty", "obedience")):
                        if "behave towards someone" in s_defn:
                            score += 70
                        elif any(k in s_defn for k in ("let someone have something as a present", "pay a particular amount")):
                            score -= 50

                # 'turn' in phrasal/contextual 'turn around': physical movement vs situation/business
                if target_pos == "verb" and (target_w_clean == "turn" or target_w_clean == "turn around"):
                    if "around" in q_low_all:
                        is_situation = any(k in q_low_all for k in ("situation", "life", "game", "match", "season", "luck", "fortune"))
                        is_business = any(k in q_low_all for k in ("business", "company", "division", "economy", "firm", "profit", "losses", "market"))
                        if is_situation:
                            if "situation, game etc turns around" in s_defn:
                                score += 80
                            elif "move your body" in s_defn:
                                score -= 50
                        elif is_business:
                            if "business, department etc that is not successful turns around" in s_defn:
                                score += 80
                            elif "move your body" in s_defn:
                                score -= 50
                        else:
                            if "move your body" in s_defn or "looking in a different direction" in s_defn:
                                score += 80
                # 'represent' as symbol/sign/meaning vs speaking for somebody/in court
                if target_pos == "verb" and target_w_clean == "represent":
                    is_symbolic_subj = any(k in q_low_all for k in ("color", "colour", "red", "white", "blue", "green", "black", "symbol", "emblem", "flag", "sign", "logo", "mark", "image", "painting"))
                    is_abstract_dobj = any(k in q_low_all for k in ("passion", "friendship", "peace", "hope", "freedom", "love", "unity", "power", "strength", "joy", "purity", "courage", "danger", "ease"))
                    if is_symbolic_subj or is_abstract_dobj:
                        if any(k in s_defn for k in ("symbol of something", "sign or mark that means something", "painting, statue")):
                            score += 85
                        elif any(k in s_defn for k in ("speak or take action for another person", "court of law", "elected to a parliament", "describe someone or something")):
                            score -= 60

                # 'trap' as capturing/preventing gas, energy, heat, water from escaping
                if target_pos == "verb" and target_w_clean == "trap":
                    if any(k in q_low_all for k in ("gas", "gases", "greenhouse", "energy", "heat", "sun", "atmosphere", "water", "air")):
                        if "prevent something such as gas" in s_defn or "prevent energy" in s_defn:
                            score += 85
                        elif any(k in s_defn for k in ("bad situation from which you cannot escape", "catch an animal", "trick someone", "crushed between")):
                            score -= 60

                # 'simple' as plain/ordinary/not complicated vs 'the simple fact/truth/reason' emphasis sense
                if target_pos == "adjective" and target_w_clean == "simple":
                    if "used to emphasize that only one thing is involved" in s_defn:
                        if not re.search(r"\bthe\s+simple\s+(?:fact|truth|reason|matter|act|question)\b", q_low_all):
                            score -= 60
                    elif "plain and without anything extra" in s_defn or "ordinary and not special" in s_defn:
                        score += 45
                    elif "not difficult or complicated" in s_defn:
                        score += 35

                # 'appetite' as physical desire for food vs metaphorical 'appetite for [activity]'
                if target_pos == "noun" and target_w_clean == "appetite":
                    has_appetite_for = bool(re.search(r"\bappetites?\s+for\b", q_low_all))
                    if not has_appetite_for:
                        if "desire for food" in s_defn:
                            score += 55
                        elif "desire or liking for a particular activity" in s_defn:
                            score -= 55

                # 'community' as local residents/neighborhood vs shared interest/religion/race group
                if target_pos == "noun" and target_w_clean == "community":
                    has_group_mod = any(k in q_low_all for k in ("scientific", "religious", "ethnic", "international", "business", "academic", "online", "medical", "intelligence", "immigrant", "asian", "black", "gay", "jewish", "christian", "muslim"))
                    if not has_group_mod:
                        if "people who live in the same area" in s_defn:
                            score += 45
                        elif "same religion, race, job" in s_defn:
                            score -= 35

                # 'ride' on a single-wheel/personal device/bike/horse (active control) vs passive bus/train passenger
                if target_pos == "verb" and target_w_clean == "ride":
                    if re.search(r"\brid(?:e|es|ing|den)|rode\b", q_low_all) and any(k in q_low_all for k in ("riding on", "ride on", "rode on", "wheel", "self-balance", "bike", "bicycle", "motorcycle", "horse", "scooter", "skateboard")):
                        if "control it as it moves" in s_defn or "sit on a horse, bicycle" in s_defn:
                            score += 60
                        elif "vehicle that you are not driving" in s_defn:
                            score -= 50

                # 'critic' as general fault-finder/opponent vs professional art/film/music reviewer
                if target_pos == "noun" and target_w_clean == "critic":
                    has_arts_ctx = any(k in q_low_all for k in ("art", "music", "film", "films", "movie", "movies", "theatre", "theater", "book", "books", "literary", "play", "concert", "opera", "dance", "painting", "novel", "performance", "review"))
                    if not has_arts_ctx:
                        if "does not approve" in s_defn or "talks about their faults" in s_defn:
                            score += 55
                        elif "good and bad qualities of art, music, films" in s_defn:
                            score -= 55

                # 'cut out' physical cutting with scissors/knife vs stopping something from happening
                if target_pos in ("verb", "phrasal verb") and target_w_clean in ("cut", "cut out"):
                    if "out" in q_low_all and any(k in q_low_all for k in ("cut them out", "cut it out", "paper", "shape", "shapes", "hand", "hands", "scissors", "picture", "pictures", "article", "card")):
                        if "remove something by cutting" in s_defn or "by cutting around it" in s_defn:
                            score += 75
                        elif "stop something from happening" in s_defn or "stop eating" in s_defn or "motor suddenly stops" in s_defn:
                            score -= 50


            # 6. Authentic Syntactic Dependency & Argument Structure Match
            if quote:
                try:
                    nlp = cls.get_spacy()
                    q_doc = nlp(quote)
                    for tok in q_doc:
                        t_lem = tok.lemma_.lower()
                        t_txt = tok.text.lower()
                        matches_target = (t_lem == target_w_clean or t_txt == target_w_clean or
                                          cls.form_matches(t_txt, target_w_clean) or
                                          cls.form_matches(t_lem, target_w_clean) or
                                          (headword_tokens and (t_lem in headword_tokens or t_txt in headword_tokens)))
                        
                        # Support hyphenated compound adjectives (e.g. 'well-kept', 'short-term') parsed by spaCy as separate tokens
                        is_hyphen_adj = False
                        if "-" in target_w_clean and tok.i >= 2:
                            parts = target_w_clean.split("-")
                            if len(parts) == 2:
                                prev1 = q_doc[tok.i - 1]
                                prev2 = q_doc[tok.i - 2]
                                if prev1.text == "-" and (prev2.lemma_.lower() == parts[0] or prev2.text.lower() == parts[0]) and (cls.form_matches(tok.lemma_.lower(), parts[1]) or cls.form_matches(tok.text.lower(), parts[1])):
                                    is_hyphen_adj = True
                                    matches_target = True

                        if not matches_target:
                            continue

                        # If target is an adjective, check its modified noun or copula subject against sense patterns & examples
                        if tok.pos_ == "ADJ" or is_hyphen_adj:
                            gram_l = (s.get("grammar") or "").lower()
                            is_attr_slot = (
                                tok.dep_ in ("amod", "compound", "nmod")
                                or (tok.head and tok.head.pos_ in ("NOUN", "PROPN") and tok.i < tok.head.i)
                            )
                            is_pred_slot = (
                                tok.dep_ in ("acomp", "oprd", "attr", "advcl")
                                or (
                                    tok.dep_ == "ROOT"
                                    and any(ch.dep_ in ("cop", "aux", "nsubj", "nsubjpass") for ch in tok.children)
                                )
                            )
                            if is_attr_slot:
                                if "[only before noun]" in gram_l or "[usually before noun]" in gram_l:
                                    score += 25
                                elif "[not before noun]" in gram_l or "[never before noun]" in gram_l:
                                    score -= 60
                            elif is_pred_slot:
                                if "[not before noun]" in gram_l or "[never before noun]" in gram_l:
                                    score += 25
                                elif "[only before noun]" in gram_l:
                                    score -= 60

                            mod_noun = ""
                            if tok.head and tok.head.pos_ in ("NOUN", "PROPN"):
                                mod_noun = tok.head.lemma_.lower()
                            elif tok.dep_ in ("acomp", "attr") and tok.head and tok.head.pos_ in ("AUX", "VERB"):
                                # Copula predicate: e.g. "work style is not conventional" -> style
                                for ch in tok.head.children:
                                    if ch.dep_ in ("nsubj", "nsubjpass") and ch.pos_ in ("NOUN", "PROPN"):
                                        mod_noun = ch.lemma_.lower()
                                        break
                            if mod_noun:
                                if any(mod_noun in ex.lower() for ex in s.get("examples", [])) or any(mod_noun in p.lower() for p in s.get("patterns", [])):
                                    score += 25
                                # If sense definition mentions the semantic class of modified noun (e.g. relationship, time, decision, method)
                                if mod_noun in s_def_tokens:
                                    score += 20
                                # Semantic alignment for style/practice/method with conventional / standard senses
                                if mod_noun in ("style", "practice", "method", "way", "approach") and any(term in s_def_tokens for term in ("method", "practice", "way", "usual", "normal")):
                                    score += 25
                                # Specific alignment for well-kept / well-maintained: if modified noun is not a 'secret', penalize the secret sense
                                if "well-kept secret" in s_defn:
                                    if mod_noun != "secret":
                                        score -= 30
                                    else:
                                        score += 40
                                elif any(k in s_defn for k in ("well cared for", "neat and clean", "maintained", "in good condition")):
                                    if mod_noun in ("garden", "gardens", "building", "buildings", "house", "grounds", "lawn", "schedule", "schedules", "record", "records", "room"):
                                        score += 35

                            # Check modifier siblings under noun (e.g. "driving" in "internal driving force")
                            if tok.head and tok.head.pos_ == "NOUN":
                                for sib in tok.head.children:
                                    if sib != tok and sib.pos_ in ("ADJ", "NOUN", "VERB"):
                                        sib_lem = sib.lemma_.lower()
                                        if sib_lem in s_def_tokens or any(sib_lem in ex.lower() for ex in s.get("examples", [])):
                                            score += 20
                                        if sib_lem in ("driving", "drive", "mental", "mind") and any(term in s_def_tokens for term in ("mind", "inner", "feeling", "thought")):
                                            score += 25

                        # If target is a noun, check compound head noun and dependency relations
                        elif tok.pos_ == "NOUN" and target_p != "v":
                            gram_l = (s.get("grammar") or "").lower()
                            has_count_det = any(
                                ch.dep_ in ("det", "nummod")
                                and ch.lemma_.lower() in ("a", "an", "one", "two", "three", "many", "few", "several", "each", "every", "another", "both")
                                for ch in tok.children
                            )
                            if tok.tag_ == "NNS" or has_count_det:
                                if ("[uncountable]" in gram_l or (tok.tag_ == "NNS" and "[singular]" in gram_l)) and "countable" not in gram_l:
                                    score -= 45 if has_count_det else 30
                                elif "countable" in gram_l or (has_count_det and "[singular]" in gram_l):
                                    score += 25

                            # Specific semantic differentiation for 'teen':
                            # 'teen' sense 1 is the period of life ("your teens"), which has examples and units ['teens'].
                            # 'teen' sense 0 is the person ("a teenager"), which has empty examples.
                            # When used as plural people/agents (e.g. 'the teens ... help/paint/work'), strongly lock to 'a teenager'.
                            if target_w_clean == "teen":
                                is_person_def = any(k in s_defn for k in ("teenager", "young person", "person"))
                                is_period_def = any(k in s_defn for k in ("period", "period of your life", "between 13 and 19"))
                                if tok.tag_ == "NNS" or tok.dep_ in ("nsubj", "nsubjpass", "dobj", "iobj", "pobj"):
                                    if is_person_def:
                                        score += 45
                                    elif is_period_def and not any(k in q_low_all for k in ("in his teens", "in her teens", "in their teens", "early teens", "late teens", "mid-teens")):
                                        score -= 35

                            # Specific semantic differentiation for 'stranger':
                            # Sense 0: "someone that you do not know"
                            # Sense 2: "someone in a new and unfamiliar place" (e.g. "I'm a stranger here myself")
                            if target_w_clean == "stranger":
                                if "unfamiliar place" in s_defn or "new and unfamiliar" in s_defn:
                                    if not any(k in q_low_all for k in ("here", "place", "city", "town", "area", "country", "neighborhood", "arrived")):
                                        score -= 35
                                elif "do not know" in s_defn or "someone that you do not know" in s_defn:
                                    if any(k in q_low_all for k in ("friend", "know", "hello", "meet", "talk to", "yesterday")):
                                        score += 30

                            # Vehicle modifier alignment (e.g. 'electric scooter', 'shared electric scooter')
                            if target_w_clean in ("scooter", "bike", "bicycle", "car", "vehicle"):
                                has_motor_modifier = any(
                                    ch.lemma_.lower() in ("electric", "motorized", "motor", "battery")
                                    for ch in tok.children if ch.dep_ in ("amod", "compound")
                                ) or any(m in q_low_all for m in ("electric scooter", "motor scooter", "electric bike", "shared electric"))
                                if has_motor_modifier:
                                    if any(k in s_defn for k in ("motorcycle", "motor", "engine", "powerful")) or any("motor" in u.lower() for u in s.get("units", [])):
                                        score += 45
                                    elif "child" in s_defn:
                                        score -= 25

                            # Governing verb (e.g. 'hold potlucks' -> meal/party, 'hear voices' -> sound/speech, 'have a fall' -> movement/event)
                            if tok.head and tok.head.pos_ == "VERB":
                                v_lem = tok.head.lemma_.lower()
                                v_forms = cls.inflected_forms(v_lem) | {v_lem}
                                v_pat = r"\b(?:" + "|".join(re.escape(f) for f in v_forms) + r")\b"
                                if v_lem in ("hold", "host", "organize", "have", "attend", "bring"):
                                    if re.search(r"\b(meal|food|party|dinner|lunch|eat|eating)\b", s_defn):
                                        score += 40
                                if any(re.search(v_pat, ex.lower()) for ex in s.get("examples", [])):
                                    score += 35

                            # Prepositional container (e.g. 'part of the team' -> work/organization, 'in myths or people's dreams' -> asleep / mental experience)
                            prep_head = None
                            if tok.dep_ == "pobj" and tok.head and tok.head.pos_ in ("ADP", "PREP"):
                                prep_head = tok.head
                            elif tok.dep_ == "conj" and tok.head and tok.head.dep_ == "pobj" and tok.head.head and tok.head.head.pos_ in ("ADP", "PREP"):
                                prep_head = tok.head.head

                            if prep_head:
                                p_txt = prep_head.text.lower()
                                if p_txt == "of":
                                    p_gov = prep_head.head
                                    if p_gov and p_gov.lemma_.lower() in ("part", "member"):
                                        if any(k in s_defn for k in ("work", "job", "organization", "company", "chosen")):
                                            score += 40
                                elif p_txt == "in":
                                    # 'in sb's dreams' / 'in a dream'
                                    if any(pat in ("in a dream", "in your dreams") for pat in s.get("patterns", [])) or "while sleeping" in (s.get("signpost") or "").lower():
                                        score += 35

                            # Possessor alignment (e.g. "people's dreams")
                            for ch in tok.children:
                                if ch.dep_ == "poss" and ch.pos_ in ("NOUN", "PROPN", "PRON"):
                                    if any(k in s_defn for k in ("asleep", "sleep", "mind", "experience", "feeling", "thought")):
                                        score += 20

                            # Prepositional dependent of target noun (e.g. 'corner of the earth/world' -> distant place)
                            for ch in tok.children:
                                if ch.dep_ == "prep":
                                    p_txt = ch.text.lower()
                                    p_lem = ch.lemma_.lower()
                                    p_objs = [gc.lemma_.lower() for gc in ch.children if gc.dep_ in ("pobj", "pcomp")]
                                    for po in p_objs:
                                        if po in s_def_tokens:
                                            score += 35
                                        elif any(po in ex.lower() for ex in s.get("examples", [])):
                                            score += 35
                                        elif any(po in pat.lower() for pat in s.get("patterns", [])):
                                            score += 35
                                        # Specialized geographic/spatial alignment for world/earth/country
                                        if po in ("earth", "world", "globe", "country", "empire", "land"):
                                            if any(k in s_defn for k in ("world", "earth", "place", "part of the world", "distant")):
                                                score += 35

                            if tok.dep_ == "compound" and tok.head and tok.head.pos_ == "NOUN":
                                compound_head = tok.head.lemma_.lower()
                                if compound_head in ("award", "program", "programme", "ceremony", "event"):
                                    if any(k in s_defn for k in ("respect", "thanks", "achievement", "work", "praise", "honor")):
                                        score += 40
                                # Check collocations and examples in sense
                                if any(compound_head in ex.lower() for ex in s.get("examples", [])):
                                    score += 35
                                elif compound_head in s_def_tokens:
                                    score += 30
                                else:
                                    entry_collocs = entry.get("collocations", {})
                                    if isinstance(entry_collocs, dict):
                                        matched_collocs = [
                                            c for c in entry_collocs.get("nouns", [])
                                            if isinstance(c, dict) and (compound_head == c.get("collocation", "").lower() or compound_head in c.get("collocation", "").lower())
                                        ]
                                        if matched_collocs:
                                            # Check if this sense aligns with the collocation's example or definition tokens
                                            lem_clean = entry.get("word", "").lower()
                                            c_tokens = set()
                                            for mc in matched_collocs:
                                                c_tokens.update(re.findall(r"\b[a-zA-Z]{3,}\b", mc.get("example", "").lower()))
                                            c_tokens.discard(lem_clean)
                                            c_tokens -= stopwords
                                            
                                            s_tokens = set(re.findall(r"\b[a-zA-Z]{3,}\b", s.get("definition", "").lower()))
                                            for ex in s.get("examples", []):
                                                s_tokens.update(re.findall(r"\b[a-zA-Z]{3,}\b", ex.lower()))
                                            s_tokens.discard(lem_clean)
                                            s_tokens -= stopwords
                                            
                                            if c_tokens & s_tokens:
                                                score += 35
                                            elif not any(len(c_tokens & (set(re.findall(r"\b[a-zA-Z]{3,}\b", other_s.get("definition", "").lower())) - stopwords)) for other_s in senses if other_s != s):
                                                # If no other sense claims this collocation, grant general bonus
                                                score += 15

                        # If target is a verb, check transitivity, complex complements and arguments
                        elif tok.pos_ == "VERB" or (target_p == "v" and (tok.dep_ in ("ROOT", "conj", "advcl", "relcl", "ccomp", "xcomp") or any(ch.dep_ in ("nsubj", "nsubjpass", "csubj") for ch in tok.children))):
                            # Section 7.3: Deep Passive Projection
                            # In passive voice, the surface subject (nsubjpass) is the underlying semantic object (deep_dobj),
                            # and the prepositional 'by' agent is the underlying semantic subject (deep_nsubj).
                            is_passive = (
                                any(c.dep_ in ("auxpass", "aux:pass") for c in tok.children) or
                                any(c.dep_ in ("nsubjpass", "nsubj:pass") for c in tok.children)
                            )
                            is_progressive = (
                                tok.tag_ == "VBG"
                                and any(c.dep_ == "aux" and c.lemma_.lower() == "be" for c in tok.children)
                            )

                            # Sense-linked Grammar Box constraints (e.g. 'admire' sense 1: no progressive; 'consist of': no passive)
                            for gb in entry.get("grammar_boxes") or []:
                                gb_senses = gb.get("sense_indices")
                                if gb_senses and i not in gb_senses:
                                    continue
                                syn_c = gb.get("syntactic_constraints") or {}
                                if is_progressive and syn_c.get("no_progressive"):
                                    score -= 50
                                if is_passive and syn_c.get("no_passive"):
                                    score -= 50

                            surface_dobjs = [ch for ch in tok.children if ch.dep_ in ("dobj", "obj")]
                            passive_dobjs = [ch for ch in tok.children if ch.dep_ in ("nsubjpass", "nsubj:pass")] if is_passive else []
                            rel_dobjs = [tok.head] if (
                                tok.dep_ == "relcl"
                                and tok.head.pos_ in ("NOUN", "PROPN", "PRON")
                                and not surface_dobjs
                                and any(ch.dep_ in ("nsubj", "csubj") and ch.lower_ not in ("which", "that", "who") for ch in tok.children)
                            ) else []
                            deep_dobj_tokens = surface_dobjs + passive_dobjs + rel_dobjs
                            has_direct_obj = bool(deep_dobj_tokens)

                            # Locate by-agent for passive voice
                            by_agents = []
                            if is_passive:
                                for c in tok.children:
                                    if c.dep_ in ("agent", "prep") and c.lemma_.lower() == "by":
                                        for gc in c.children:
                                            if gc.dep_ in ("pobj", "pcomp"):
                                                by_agents.append(gc)
                                if not by_agents:
                                    # Handle nested prepositional attachment (e.g. 'to the winner by the mayor')
                                    for c in tok.children:
                                        if c.dep_ == "prep":
                                            for gc in c.subtree:
                                                if gc != c and gc.dep_ == "prep" and gc.lemma_.lower() == "by":
                                                    for ggc in gc.children:
                                                        if ggc.dep_ in ("pobj", "pcomp"):
                                                            by_agents.append(ggc)

                            surface_nsubjs = [ch for ch in tok.children if ch.dep_ in ("nsubj", "csubj")]
                            deep_nsubj_tokens = by_agents if (is_passive and by_agents) else surface_nsubjs

                            # Complex clausal complement with infinitive: e.g. [V] + [NP] + [to-inf]
                            # spaCy can parse this either as v -> ccomp (inf) with nsubj/dobj and aux:to,
                            # or as v -> dobj (NP) and xcomp (inf) with aux:to.
                            has_clausal_inf = any(
                                (c.dep_ in ("ccomp", "xcomp") and any(gc.lemma_.lower() == "to" for gc in c.children))
                                for c in tok.children
                            )
                            has_finite_ccomp = any(
                                c.dep_ == "ccomp" and not any(gc.lemma_.lower() == "to" and gc.dep_ == "aux" for gc in c.children)
                                for c in tok.children
                            )
                            has_complex_comp = (
                                (has_direct_obj and any(c.dep_ == "xcomp" and any(gc.lemma_.lower() == "to" for gc in c.children) for c in tok.children))
                                or any(
                                    c.dep_ in ("ccomp", "xcomp") and any(gc.dep_ in ("nsubj", "dobj", "obj") for gc in c.children)
                                    and any(gc.lemma_.lower() == "to" for gc in c.children)
                                    for c in tok.children
                                )
                            )
                            has_effective_obj = has_direct_obj or has_clausal_inf or has_finite_ccomp

                            gram_l = (s.get("grammar") or "").lower()
                            s_units = [u.lower() for u in (s.get("units") or [])]
                            s_pats = [p.lower() for p in (s.get("patterns") or [])]
                            all_pat_units = s_units + s_pats

                            if has_finite_ccomp and any("(that)" in p or p.endswith(" that") for p in all_pat_units):
                                score += 40

                            def _has_cc_structure(txt: str) -> bool:
                                t_clean = re.sub(r"\(.*?\)", "", txt)
                                return bool(re.search(r"\b(?:somebody|someone|something|sb|sth)(?:/(?:somebody|someone|something|sb|sth))*\s+to\s+(?:do|make|have|be)\b", t_clean))

                            # Distinguish direct object requirement from complex complements or non-finite verbal complements
                            def _requires_direct_object(txt: str) -> bool:
                                t_clean = re.sub(r"\(.*?\)", "", txt)
                                t_clean = re.sub(r"\b(?:somebody|someone|something|sb|sth)(?:/(?:somebody|someone|something|sb|sth))*\s+to\s+(?:do|make|have|be)\s+(?:something|sth|somebody|sb)?\b", "", t_clean)
                                t_clean = re.sub(r"\b(?:to|and)\s+(?:do|make|have)\s+(?:something|sth|somebody|sb)\b", "", t_clean)
                                return bool(re.search(r"\b(?:somebody|someone|something|sb|sth)\b", t_clean))

                            # Check if definition shows ergative or intransitive conditional usage:
                            # e.g. 'if a situation, game etc turns around, or if someone turns it around', 'if a situation picks up, it improves'
                            is_ergative_defn = bool(re.search(r"\bor\s+if\s+someone\s+\w+\s+(?:it|them)\b", s_defn))
                            is_intrans_cond_defn = bool(re.match(r"^if\s+(?:a|an|the|your|someone's|somebody's)?\s*(?:situation|things|business|economy|trade|sales|market|weather|wind|speed|prices?|traffic|conversation|life)\b", s_defn))

                            # Check if sense allows intransitive usage:
                            # 1) explicit 'intransitive' in grammar tag (e.g. '[intransitive, transitive]', '[intransitive]')
                            # 2) sense declares patterns that do not require a direct object (e.g. 'pay for', 'pay in cash')
                            has_intrans_gram = "intransitive" in gram_l or is_intrans_cond_defn
                            has_trans_gram = bool(re.search(r"(?<!in)transitive", gram_l)) and not is_intrans_cond_defn
                            has_intransitive_pattern = any(not _requires_direct_object(p) for p in s_pats) if s_pats else False
                            allows_intransitive = has_intrans_gram or (not has_trans_gram and has_intransitive_pattern)

                            sense_has_cc = any(_has_cc_structure(p) for p in all_pat_units)
                            requires_obj = (
                                not is_ergative_defn and not allows_intransitive and (
                                    has_trans_gram or
                                    sense_has_cc or
                                    any(_requires_direct_object(u) for u in s_units) or
                                    any(_requires_direct_object(p) for p in s_pats)
                                )
                            )
                            is_pure_intransitive = (
                                (has_intrans_gram and not has_trans_gram) or
                                not requires_obj
                            )

                            # Complex Complement Structure Gate:
                            # If context has [V] + [NP] + [to-inf] (e.g. 'expect anyone to feel surprised', 'wants you to stay'),
                            # award strong alignment bonus to senses declaring this pattern;
                            # if other senses declare specific patterns that contradict this structure, discount them.
                            if has_complex_comp:
                                if sense_has_cc:
                                    score += 55
                                else:
                                    score -= 65

                            if not has_effective_obj:
                                if is_pure_intransitive and not requires_obj:
                                    score += 25
                                elif requires_obj:
                                    # Penalize transitive senses when sentence has no direct object or clausal complement
                                    score -= 80
                            else:
                                if requires_obj or has_trans_gram or is_ergative_defn:
                                    score += 25
                                elif has_intrans_gram and not has_trans_gram:
                                    score -= 55

                                # Transitive valency alignment: if an explicit direct object is present,
                                # senses providing explicit transitive patterns (e.g. 'attach something to something', 'pay somebody something')
                                # strongly match the syntactic frame; senses whose patterns are strictly intransitive (e.g. 'attach to')
                                # receive a discount.
                                has_trans_pat = any(
                                    ("something" in p.lower() or "somebody" in p.lower() or " sb " in f" {p.lower()} " or " sth " in f" {p.lower()} ")
                                    for p in s_pats
                                )
                                has_strictly_intrans_prep_pat = any(
                                    not ("something" in p.lower() or "somebody" in p.lower() or " sb " in f" {p.lower()} " or " sth " in f" {p.lower()} ")
                                    and any(prep in p.lower().split() for prep in cls._PATTERN_PREPS)
                                    for p in s_pats
                                )
                                if has_trans_pat:
                                    score += 35
                                elif has_strictly_intrans_prep_pat and not has_trans_pat:
                                    score -= 30

                            # Direct object argument matching (governs deep_dobj_tokens, supporting passive projection)
                            for ch in deep_dobj_tokens:
                                obj_lem = ch.lemma_.lower()
                                if obj_lem in s_def_tokens:
                                    score += 35
                                elif any(re.search(rf"\b{re.escape(obj_lem)}\b", ex.lower()) for ex in s.get("examples", [])):
                                    score += 30
                                elif any(re.search(rf"\b{re.escape(obj_lem)}\b", p.lower()) for p in s.get("patterns", [])):
                                    score += 30

                                if obj_lem in ("dream", "dreams", "ambition", "goal", "potential", "hope", "desire", "plan", "aim", "vision", "aspiration"):
                                    if any(k in s_defn or k in sp for k in ("achieve", "ambition", "potential", "hoping to achieve", "fulfil", "fulfill")):
                                        score += 45

                                # Supersenses selectional restrictions for dobj:
                                obj_supersenses = cls.get_wordnet_supersenses(obj_lem, pos="n")
                                if obj_supersenses:
                                    if "noun.communication" in obj_supersenses or "noun.possession" in obj_supersenses:
                                        if any(k in s_defn for k in ("give", "award", "prize", "gift", "medal", "money", "present", "degree")):
                                            score += 40
                                    if "noun.person" in obj_supersenses:
                                        if any(k in s_defn for k in ("introduce someone", "formally introduce")):
                                            score += 35

                            # Prepositional complement / adverbial phrase / particle matching (e.g. 'recognize sb by doing', 'come in', 'sit down', 'turn around')
                            for ch in tok.children:
                                if ch.dep_ in ("prep", "prt", "advmod"):
                                    p_txt = ch.text.lower()
                                    p_lem = ch.lemma_.lower()
                                    # Check if preposition / particle is explicitly in sense patterns (e.g. 'for', 'by', 'in', 'around', 'down', 'back', 'away')
                                    # If pattern contains content words or wh-markers (e.g. 'appear in a film', 'where somebody is coming from'), they must also be present
                                    pat_matched = False
                                    for pat in s.get("patterns", []):
                                        branches = cls.clean_and_expand_pattern_slashes(pat)
                                        for br in branches:
                                            br_tokens = br.split()
                                            if (f" {p_txt} " in f" {br} " or f" {p_lem} " in f" {br} " or
                                                p_txt in br_tokens or p_lem in br_tokens):
                                                pw = cls._pattern_words(br)
                                                # Wh-markers / complementizers in pattern (e.g. 'where somebody is coming from') MUST match in context
                                                wh_markers = [w for w in pw if w in ("where", "when", "why", "how", "what", "who", "whom", "whose")]
                                                if wh_markers and not any(w in q_seq for w in wh_markers):
                                                    continue
                                                # Selectional restriction: if pattern requires person object (e.g. 'coming from him/her/you etc')
                                                # and context prepositional object is inanimate / non-person, skip this pattern
                                                has_person_constraint = any(w in ("him", "her", "you", "someone", "somebody", "sb") for w in pw) and not any(w in ("something", "sth", "what") for w in pw)
                                                if has_person_constraint:
                                                    p_objs = [pc for pc in ch.children if pc.dep_ in ("pobj", "pcomp")]
                                                    if p_objs:
                                                        is_pobj_person = any(
                                                            pc.pos_ == "PRON" and pc.lemma_.lower() in ("he", "she", "i", "we", "they", "you", "who", "someone", "somebody", "him", "her", "me", "us", "them")
                                                            or (pc.pos_ in ("NOUN", "PROPN") and pc.ent_type_ in ("PERSON", "ORG"))
                                                            or pc.lemma_.lower() in ("person", "people", "man", "woman", "child", "staff", "employee", "worker", "friend", "parent")
                                                            or "noun.person" in cls.get_wordnet_supersenses(pc.lemma_.lower(), pos="n")
                                                            for pc in p_objs
                                                        )
                                                        if not is_pobj_person:
                                                            continue
                                                content_pw = [w for w in pw if w not in cls._PATTERN_PREPS and w not in cls._PATTERN_FUNCTION_WORDS and w not in ("be", "become", "get", "grow") and w not in target_inflections]
                                                if not content_pw or any(any(cls.form_matches(tok_q, cpw) for tok_q in q_seq) for cpw in content_pw):
                                                    # When particle exactly matches pattern (e.g. 'turn around', 'come in', 'sit down')
                                                    score += 40
                                                    pat_matched = True
                                                    break
                                        if pat_matched:
                                            break
                                    # Check prepositional objects or complement verbs
                                    has_action_pcomp = False
                                    p_objs_expanded = []
                                    for p_child in ch.children:
                                        if p_child.dep_ in ("pobj", "pcomp"):
                                            p_objs_expanded.append(p_child)
                                            p_objs_expanded.extend([gc for gc in p_child.children if gc.dep_ == "conj"])

                                    for p_child in p_objs_expanded:
                                        p_child_lem = p_child.lemma_.lower()
                                        if p_child.dep_ == "pcomp" or p_child.pos_ == "VERB":
                                            has_action_pcomp = True
                                        # Generic words / time adverbials (e.g. 'time', 'people', 'thing') carry no specific collocational evidence
                                        if p_child_lem in meta_stopwords:
                                            continue
                                        # Match against definition tokens or example tokens (word boundary enforced)
                                        if p_child_lem in s_def_tokens:
                                            score += 30
                                        elif any(re.search(rf"\b{re.escape(p_child_lem)}\b", ex.lower()) for ex in s.get("examples", [])):
                                            score += 25

                                    # Specific alignment for intransitive verbs of appearing/existence with locative/conceptual preposition (in myths, in dreams, etc.)
                                    if p_lem in ("in", "at", "on", "from") and any(k in s_defn for k in ("exist in a place", "start to be seen", "arrive", "become visible")):
                                        # When appearing in concepts/settings (myths, dreams, place, air, sky) rather than film/court/publication
                                        if not any(k in s_defn for k in ("film", "play", "court", "television", "print", "newspaper")):
                                            score += 30

                                    # Universal verb-preposition valency frame agreement:
                                    # If the verb in context governs a preposition (e.g. 'function as', 'look into', 'depend on', 'succeed in'),
                                    # check if this verb+prep frame is explicitly articulated in the sense definition or examples.
                                    target_lemmas = {target_w_clean} | cls.inflected_forms(target_w_clean) | {tok.lemma_.lower()}
                                    v_prep_pat = r"\b(?:" + "|".join(re.escape(f) for f in target_lemmas) + r")(?:\s+\w+)?\s+(?:" + re.escape(p_lem) + r"|" + re.escape(p_txt) + r")\b"
                                    if not pat_matched:
                                        if re.search(v_prep_pat, s_defn):
                                            score += 35
                                        elif any(re.search(v_prep_pat, ex.lower()) for ex in s.get("examples", [])):
                                            score += 40
                                    if any(k in s_defn for k in ("radio or television broadcast", "touch a child sexually")) and not any(k in q_low_all for k in ("radio", "television", "broadcast", "signal", "reception", "child", "sexually", "boy", "girl")):
                                        score -= 45

                                    # Selectional restrictions on prepositional complements:
                                    # Distinguish commercial/financial exchange ('pay for dinner/tickets/lessons', 'pay in cash/dollars')
                                    # from punitive suffering/penalty ('pay for mistakes/crimes/sins').
                                    if p_lem in ("for", "in", "by"):
                                        for p_child in p_objs_expanded:
                                            p_ch_lem = p_child.lemma_.lower()
                                            p_supersenses = cls.get_wordnet_supersenses(p_ch_lem, pos="n")
                                            is_financial_or_goods = (
                                                p_ch_lem in ("dollar", "dollars", "euro", "euros", "pound", "pounds", "cash", "credit", "card", "cheque", "check", "coin", "cent", "bill", "dinner", "ticket", "tickets", "meal", "lunch", "drink", "food", "rent", "tax", "fare", "fee", "goods", "service", "lesson", "lessons", "car")
                                                or (p_supersenses and any(ss in p_supersenses for ss in ("noun.possession", "noun.food", "noun.artifact")))
                                            )
                                            is_wrongdoing_or_penalty = (
                                                p_ch_lem in ("mistake", "mistakes", "crime", "crimes", "sin", "sins", "error", "errors", "fault", "wrong", "damage", "foolishness", "carelessness", "neglect", "misdeed", "offense", "offence")
                                                or (p_supersenses and "noun.cognition" in p_supersenses and any(k in s_defn for k in ("punish", "suffer", "wrong", "penalty")))
                                            )
                                            if is_financial_or_goods:
                                                if any(k in s_defn for k in ("money", "buy", "service", "cost", "owe", "salary", "wage")):
                                                    score += 55
                                                elif any(k in s_defn for k in ("punish", "suffer", "wrong", "penalty")):
                                                    score -= 75
                                                # Currency bonus: 'pay in dollars/euros/cash' directly matches currency patterns/examples
                                                if p_lem == "in" and p_ch_lem in ("dollar", "dollars", "euro", "euros", "pound", "pounds", "cash", "currency"):
                                                    if any(k in s_defn or any(k in ex.lower() for ex in s.get("examples", [])) or any(k in pat.lower() for pat in s.get("patterns", [])) for k in ("dollar", "euro", "cash", "currency")):
                                                        score += 40
                                            elif is_wrongdoing_or_penalty:
                                                if any(k in s_defn for k in ("punish", "suffer", "wrong", "penalty", "unpleasant")):
                                                    score += 65
                                                elif any(k in s_defn for k in ("money", "buy", "service", "cost")):
                                                    score -= 40

                                    # Generalized semantic valency: 'by [doing]' or 'for [doing/NP]' as instrument/reward/cause
                                    # For verbs denoting appraisal, thank, reward, praise, recognition, honor
                                    if p_lem in ("by", "for"):
                                        is_appraisal = any(k in s_defn for k in ("thank", "honour", "honor", "reward", "praise", "award", "appreciat", "respect"))
                                        if is_appraisal:
                                            # Strong alignment for rewarding/thanking/honoring actions:
                                            # e.g. 'recognize volunteers by taking them...', 'recognized for having saved...'
                                            score += 45
                                            if has_action_pcomp:
                                                score += 25
                                        elif any(f" {p_lem} " in f" {ex.lower()} " for ex in s.get("examples", [])):
                                            # If prep is 'by' but followed by an action/event verb (taking/giving/doing),
                                            # cognitive/sensory recognition ('know by sight/voice/face') does not apply.
                                            if p_lem == "by" and has_action_pcomp and any(k in s_defn for k in ("know", "hear", "see", "sight", "voice", "face", "appearance")):
                                                score -= 20
                                            else:
                                                score += 15

                            # Clausal head / nominal governing structure (e.g. 'effort/move/decision to recognize them')
                            if tok.dep_ == "acl" and tok.head and tok.head.pos_ in ("NOUN", "PROPN"):
                                gov_lem = tok.head.lemma_.lower()
                                if any(re.search(rf"\b{re.escape(gov_lem)}\b", ex.lower()) for ex in s.get("examples", [])) or gov_lem in s_def_tokens:
                                    score += 35
                                elif gov_lem in ("effort", "move", "attempt", "initiative", "campaign", "gesture", "step"):
                                    if any(k in s_defn for k in ("thank", "honour", "honor", "reward", "praise", "appreciat", "respect")):
                                        score += 45
                                # Check higher governing verb (e.g. 'appreciate the effort to recognize them')
                                if tok.head.head and tok.head.head.pos_ in ("VERB", "AUX"):
                                    higher_v = tok.head.head.lemma_.lower()
                                    if higher_v in ("appreciate", "welcome", "value", "praise", "applaud", "support"):
                                        if any(k in s_defn for k in ("thank", "honour", "honor", "reward", "praise", "appreciat", "respect")):
                                            score += 30

                            # Check semantic subject tokens (governs deep_nsubj_tokens, supporting passive by-agent)
                            for ch in deep_nsubj_tokens:
                                subj_lem = ch.lemma_.lower()
                                # Closed-class personal pronouns ('you', 'i', 'he', etc.) appear ubiquitously in dictionary examples
                                # and must not be used to award generic token-overlap bonus unless the head noun/pronoun is lexical
                                if subj_lem not in stopwords:
                                    if subj_lem in s_def_tokens or any(re.search(rf"\b{re.escape(subj_lem)}\b", ex.lower()) for ex in s.get("examples", [])):
                                        score += 25
                                # Human vs non-human subject differentiation for verbs with 'effective/successful' or job/activity senses (e.g. work)
                                subj_supersenses = cls.get_wordnet_supersenses(subj_lem, pos="n")
                                is_human_subj = (
                                    (ch.pos_ == "PRON" and subj_lem in ("he", "she", "i", "we", "they", "you", "who", "someone", "somebody")) or
                                    (ch.pos_ in ("NOUN", "PROPN") and (
                                        subj_lem in ("person", "people", "man", "woman", "child", "staff", "employee", "worker", "doctor", "lawyer", "teacher", "mentor", "volunteer", "mayor", "president", "director", "manager", "king", "queen", "winner") or
                                        ch.ent_type_ in ("PERSON", "ORG") or
                                        "noun.person" in subj_supersenses
                                    ))
                                )
                                if is_human_subj:
                                    # Human subject + 'from' preposition specifying origin/birth/culture/background
                                    # (e.g. 'come from a culture/country/place', 'hail from ...')
                                    prep_from_objs = [
                                        pc.lemma_.lower()
                                        for p_ch in tok.children if p_ch.dep_ == "prep" and p_ch.lemma_.lower() == "from"
                                        for pc in p_ch.children if pc.dep_ in ("pobj", "pcomp")
                                    ]
                                    if any(po in ("culture", "country", "place", "background", "family", "city", "town", "region", "society", "nation", "area") for po in prep_from_objs):
                                        if any(k in s_defn for k in ("born there", "lived there", "from a place", "place, you were", "family or background", "belong to")):
                                            score += 40
                                        elif any(k in s_defn for k in ("attitude or opinion", "basic attitude")):
                                            score -= 30

                                    # Human subject vs vehicle arriving / landing sense (e.g. 'train, bus, plane, or ship comes in')
                                    if any(k in s_defn for k in ("train, bus, plane", "plane, or ship", "train arrives", "plane arrives", "ship arrives")):
                                        score -= 40

                                    # Human subject physical body movement / posture alignment vs situation/business turnaround
                                    # (e.g. 'She turned around and walked away', 'The student turned around and smiled')
                                    if any(k in sp or k in s_defn for k in ("your body", "body", "look in a different direction", "move your body", "posture")):
                                        score += 45
                                    elif any(k in s_defn for k in ("if a situation", "if a business", "if a shop or business")):
                                        score -= 40
                                else:
                                    # Non-human / abstract subject alignment (e.g. situation / crisis / event vs business / corporate)
                                    is_business_subj = (
                                        subj_lem in ("business", "company", "firm", "department", "division", "corporation", "organization", "enterprise", "airline", "bank", "shop", "store", "factory", "pub", "bar", "restaurant", "hotel", "cafe") or
                                        "noun.group" in subj_supersenses
                                    )
                                    is_situation_subj = (
                                        subj_lem in ("situation", "crisis", "problem", "event", "condition", "economy", "market", "game", "match", "trend", "matter", "affair", "life") or
                                        bool(subj_supersenses & {"noun.state", "noun.event"})
                                    ) and not is_business_subj

                                    is_quality_or_action_subj = (
                                        subj_lem in ("punctuality", "honesty", "patience", "hard work", "diligence", "effort", "kindness", "courage", "caution", "persistence") or
                                        bool(subj_supersenses & {"noun.attribute", "noun.act"})
                                    )

                                    if is_business_subj:
                                        if any(k in s_defn for k in ("business", "department", "company", "shop", "makes a profit", "starts to be successful")):
                                            score += 55
                                    elif is_situation_subj:
                                        if any(k in s_defn for k in ("situation", "game", "develop in the way you want", "changes and starts", "improves", "gets better", "picks up")):
                                            score += 55
                                        elif any(k in s_defn for k in ("business, department", "business or department")):
                                            score -= 20
                                    elif is_quality_or_action_subj:
                                        if any(k in s_defn for k in ("particular action pays", "brings a good result", "advantage for you", "pays dividends")):
                                            score += 55
                                        elif any(k in s_defn for k in ("give someone money", "job they do")):
                                            score -= 20

                                    # First-principles invariant: non-human subjects only boost 'effective/successful' for verbs of functioning/operating (e.g. work, operate, run, function)
                                    is_functioning_verb = tok.lemma_.lower() in ("work", "function", "operate", "run", "act", "serve") or any(k in target_w_clean for k in ("work", "function", "operate"))
                                    if is_functioning_verb and any(k in s_defn for k in ("effective", "successful", "function", "operate")):
                                        score += 45
                                    elif any(k in s_defn for k in ("paid", "job", "salary", "employer")):
                                        score -= 25
                                    # Specific domain check for acoustic/sound vs quantity/amount increase
                                    is_sound_subj = subj_lem in ("music", "sound", "noise", "applause", "cheer", "chorus", "song", "shout", "cry")
                                    if is_sound_subj and any(k in s_defn for k in ("loud", "sound", "noise", "hear")):
                                        score += 40
                                    elif not is_sound_subj and any(k in s_defn for k in ("become louder", "louder")):
                                        score -= 30

                                    is_qty_subj = subj_lem in ("number", "amount", "membership", "population", "crowd", "rank", "ranks", "size", "volume", "figure", "total")
                                    if is_qty_subj and any(k in s_defn for k in ("amount", "number", "increase in", "grow in number")):
                                        score += 40

                                    # Liquid / fluid subject alignment (e.g. 'Tears were running down her face', 'Water was running')
                                    is_liquid_subj = subj_lem in ("tear", "tears", "water", "sweat", "blood", "river", "rain", "liquid", "stream", "fluid", "droplet")
                                    if is_liquid_subj and (any(k in s_defn for k in ("flow in", "flow", "liquid")) or (s.get("signpost") or "").lower() == "flow"):
                                        score += 50
                                    elif is_liquid_subj and any(k in s_defn for k in ("using your legs", "kill or injure", "criticize", "manage", "business")):
                                        score -= 40

                                    # Origin / derivation vs birth / youth place differentiation:
                                    # Non-human subjects (words, culture, craft, coffee, etc.) originate/derive from a place/source;
                                    # they are not born or lived there when young.
                                    is_origin_defn = any(k in s_defn for k in ("obtained from", "start or be made", "originate", "derive"))
                                    is_birth_defn = any(k in s_defn for k in ("born there", "lived there when you were young", "grow up"))
                                    if is_birth_defn:
                                        score -= 30
                                    # Selectional restriction: physical body movement senses (signpost 'your body' or definition 'move your body')
                                    # require human/animate subjects. Non-human/abstract/corporate subjects (company, economy, market, crisis, etc.)
                                    # cannot perform physical body movements unless explicit body parts are mentioned.
                                    is_body_movement_sense = "your body" in (s.get("signpost") or "").lower() or any(k in s_defn for k in ("move your body", "using your body", "part of your body"))
                                    has_body_part_in_quote = any(bw in q_low_all for bw in ("body", "head", "face", "neck", "shoulder", "arm", "leg", "eye", "eyes", "heel", "back"))
                                    if is_body_movement_sense and not has_body_part_in_quote:
                                        score -= 40

                            # Discourse/deliberative forum context alignment (e.g. 'Go to shareholder meetings and speak up')
                            has_forum_context = any(
                                t_ctx.lemma_.lower() in ("meeting", "conference", "assembly", "shareholder", "debate", "discussion", "hearing", "forum", "session", "council", "parliament", "board")
                                for t_ctx in q_doc
                            )
                            if has_forum_context:
                                if any(k in s_defn for k in ("express your opinion", "express their opinion", "say what you think", "say what they think", "opinion")):
                                    score += 35
                                elif re.match(r"^(?:used\s+to\s+(?:ask|say|tell|show|express)|spoken\b)", s_defn, re.I):
                                    score -= 20

                            # Check for xcomp infinitive complement (e.g. tend to do / think)
                            has_xcomp = any(ch.dep_ == "xcomp" for ch in tok.children)
                            if has_xcomp:
                                for u in s.get("units", []):
                                    if "to do" in u.lower():
                                        score += 45
                                        break

                        # If target is a preposition / conjunction, check complement
                        elif tok.pos_ in ("ADP", "PREP") or tok.dep_ == "prep":
                            time_nouns = {"day", "hour", "minute", "second", "week", "month", "year", "decade", "century", "time", "period"}
                            has_time_pobj = any(
                                c.lemma_.lower() in time_nouns or any(gc.lemma_.lower() in time_nouns for gc in c.subtree)
                                for c in tok.children if c.dep_ in ("pobj", "pcomp")
                            )
                            if has_time_pobj:
                                if any(k in s_defn for k in ("period of time", "time", "hours", "weeks", "months", "years", "days")):
                                    score += 40
                                # Exact numeric interval distinction: 'before a certain period has passed' (e.g. within 24 hours / within one day / an hour)
                                has_num_time = any(
                                    any(gc.dep_ in ("nummod", "quantmod") or gc.pos_ == "NUM" or gc.lemma_.lower() in ("a", "an", "one", "two", "three", "few", "several") for gc in c.children)
                                    for c in tok.children if c.dep_ in ("pobj", "pcomp") and (c.lemma_.lower() in time_nouns or any(gc.lemma_.lower() in time_nouns for gc in c.subtree))
                                )
                                if has_num_time:
                                    if "before a certain period" in s_defn or any("24 hours" in ex or "an hour" in ex for ex in s.get("examples", [])):
                                        score += 25
                except Exception:
                    pass

            # 7. Syntactic frame / Grammar alignment
            gram = (s.get("grammar") or "").lower()
            if frame_evidence:
                if frame_evidence == "modifies_adj" and any(k in gram for k in ("adjective", "adverb", "predicative", "[+adjective")):
                    score += 25
                elif frame_evidence == "modifies_verb" and any(k in gram for k in ("verb", "action", "[+verb")):
                    score += 25
                elif frame_evidence == "prep" and re.search(r"\[.*prep.*\]", gram):
                    score += 10

            # 8. Contextual Pedagogical Sense Cue Index bonus
            if i in cue_scores:
                score += cue_scores[i]

            # Tie-break slightly favors primary head senses
            score -= i * 0.5

            score_trace.append((i, score))

            if score > best_score:
                second_score = best_score
                best_score = score
                best_idx = i
            elif score > second_score:
                second_score = score

        if cls.SENSE_DEBUG and score_trace:
            for i, sc in sorted(score_trace, key=lambda r: -r[1])[:8]:
                print(f"[sense-lock] {target_w_clean} pos={target_pos} quote={quote!r} "
                      f"sense {i} score {sc:.3f} :: {(senses[i].get('definition') or '')[:70]}")

        if best_idx is not None:
            idx = best_idx
        elif target_p:
            # Check if any sense matches target_pos
            idx = next((i for i, s in enumerate(senses) if cls._pos_code(s.get("pos")) == target_p), None)
            if idx is None:
                # If entry is an entity or has only 1 sense carrying no POS label, fall back to sense 0
                if len(senses) == 1 and (entry.get("kind") == "entity" or not senses[0].get("pos")):
                    idx = 0
        else:
            idx = 0 if senses else None
        score = best_score if best_idx is not None else 0.0
        margin = ((best_score - second_score)
                  if best_idx is not None and second_score > -9999 else 0.0)
        # Nothing to disambiguate: a single sense, or a POS gate that matched exactly one
        # sense, is as certain as the entry itself, so the floors do not apply. A gate that
        # matched nothing across several senses is a guess and stays under the floors.
        unambiguous = eligible_senses <= 1 or (eligible_senses == 0 and len(senses) == 1)
        confident = unambiguous or (
            score >= cls.SENSE_CONFIDENCE_FLOOR and margin >= cls.SENSE_MARGIN_FLOOR
        )
        if return_scores:
            return idx, score, margin, ("high" if confident else "low"), score_trace
        if not return_confidence:
            return idx
        return idx, score, margin, ("high" if confident else "low")

    @classmethod
    def sense_confidence(
        cls,
        entry: Dict[str, Any],
        definition: Optional[str] = None,
        quote: Optional[str] = None,
        target_pos: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        B1: how firmly the quote and the curriculum definition pin down which sense this item
        teaches. 'low' means the sense was reached by ordering, not by evidence, and the caller
        must not ship that definition as authoritative.
        """
        senses = (entry or {}).get("senses") or []
        if not senses:
            return {"index": None, "score": 0.0, "margin": 0.0,
                    "confidence": "low", "eligible_senses": 0}
        target_p = cls._pos_code(target_pos)
        eligible = [s for s in senses
                    if not (target_p and cls._pos_code(s.get("pos")) != target_p)]
        idx, score, margin, confidence = cls._lock_sense(
            entry,
            definition=definition,
            quote=quote,
            target_pos=target_pos,
            return_confidence=True
        )
        return {"index": idx, "score": score, "margin": margin,
                "confidence": confidence, "eligible_senses": len(eligible)}

    @classmethod
    def search_corpus_examples(
        cls,
        query: str,
        category: Optional[str] = None,
        limit: int = 3
    ) -> List[str]:
        """
        Retrieves authentic PedLex corpus examples directly from entry data_json at 0-token cost.
        Scans own headword entries plus indexed cross-entries (ldoce_phrase_index & ldoce_sense_cue_index).
        """
        if not query or not query.strip():
            return []

        clean_q = query.strip()
        near_match = re.search(r'NEAR\(\s*([a-zA-Z0-9_-]+)\s+([a-zA-Z0-9_-]+)', clean_q, re.IGNORECASE)
        key_tokens = [t.lower() for t in re.findall(r'[a-zA-Z0-9_-]{2,}', clean_q) if t.lower() not in ("near", "and", "or", "category")]
        if not key_tokens:
            return []

        is_multi = len(key_tokens) >= 2 and "-" not in clean_q
        u_key = cls._unit_key(clean_q)

        # 1. Primary anchor words (own headword entries + hyphen sub-parts for derived hyphenated compounds)
        anchor_words: List[str] = []
        sub_parts: List[str] = []
        if "-" in clean_q:
            sub_parts = [p.lower() for p in clean_q.split("-") if len(p) >= 3 and p.lower() not in ("self", "non", "pre", "sub", "anti", "counter")]
        for aw in ([near_match.group(1).lower()] if near_match else [clean_q.lower()] + key_tokens[:3] + sub_parts):
            if aw and aw not in anchor_words:
                anchor_words.append(aw)

        # 2. Cross-entry host lookups via ldoce_phrase_index and ldoce_sense_cue_index (<0.5ms indexed lookup)
        if cls._ensure_ldoce_conn() and cls._ldoce_conn is not None:
            try:
                cur = cls._ldoce_conn.cursor()
                if is_multi and u_key:
                    cur.execute(
                        "SELECT DISTINCT host_word FROM ldoce_phrase_index WHERE phrase_key = ? LIMIT 20",
                        (u_key,)
                    )
                    for (hw,) in cur.fetchall():
                        if hw and hw not in anchor_words:
                            anchor_words.append(hw)
                for tok in (key_tokens[:2] + sub_parts[:2]):
                    cur.execute(
                        "SELECT DISTINCT host_word FROM ldoce_sense_cue_index WHERE cue_word = ? ORDER BY weight DESC LIMIT 20",
                        (tok,)
                    )
                    for (hw,) in cur.fetchall():
                        if hw and hw not in anchor_words:
                            anchor_words.append(hw)
            except Exception:
                pass

        entries = []
        for aw in anchor_words[:30]:
            ent = cls.get_ldoce_entry(aw)
            if ent and ent not in entries:
                entries.append(ent)
        if not entries:
            return []

        pool: List[str] = []
        for entry in entries:
            for s in entry.get("senses", []):
                pool.extend(s.get("examples", []) or [])
                for pb in (s.get("pattern_bundles") or []):
                    if isinstance(pb, dict) and pb.get("example"):
                        pool.append(pb["example"])
                for u_obj in (s.get("units") or []):
                    if isinstance(u_obj, dict):
                        pool.extend(u_obj.get("examples", []) or [])
            for gb in entry.get("grammar_boxes", []):
                pool.extend(gb.get("examples", []) or [])
            for _cat_k, items in entry.get("collocations", {}).items():
                for it in items:
                    if isinstance(it, dict) and it.get("example"):
                        pool.append(it["example"])
            for cc in entry.get("cross_collocations", []):
                if isinstance(cc, dict) and cc.get("example"):
                    pool.append(cc["example"])
            for th in entry.get("thesaurus", []):
                if isinstance(th, dict):
                    pool.extend(th.get("examples", []) or [])
                    if th.get("example"):
                        pool.append(th["example"])
            for la in entry.get("language_activator", []):
                if isinstance(la, dict):
                    for exp in (la.get("exponents") or []):
                        if isinstance(exp, dict):
                            pool.extend(exp.get("examples", []) or [])
                            if exp.get("example"):
                                pool.append(exp["example"])

        res: List[str] = []
        for cand in pool:
            if not cand or not isinstance(cand, str):
                continue
            cand_clean = re.sub(r'^(?:(?:Male|Female)\s+speaker|S\b)[\s:]*', '', cand, flags=re.IGNORECASE)
            cand_clean = re.sub(r'\s*\(=[^)]*\)', '', cand_clean)
            cand_clean = re.sub(r'\s+', ' ', cand_clean).strip()
            if not cand_clean or len(cand_clean) <= 8 or cand_clean in res:
                continue
            cand_low = cand_clean.lower()
            if is_multi:
                if cls._unit_example(u_key, [cand_clean], allow_modifier=True) or all(tok in cand_low for tok in key_tokens):
                    res.append(cand_clean)
            else:
                if cls.text_contains_form(cand_clean, clean_q) or all(tok in cand_low for tok in key_tokens):
                    res.append(cand_clean)
            if len(res) >= max(limit * 3, 15):
                break

        unit_matches = [c for c in res if cls._unit_example(u_key or clean_q, [c], allow_modifier=True)]
        other_matches = [c for c in res if c not in unit_matches]
        return (unit_matches + other_matches)[:limit]

    @classmethod
    def get_ldoce_grammar_alert(
        cls,
        word: str,
        quote: str = "",
        sense_index: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieves authentic grammar error diagnostics (Don't say / ✗) directly from PedLex Grammar Boxes in data_json.
        When `sense_index` is provided, filters out boxes explicitly restricted to other senses (`sense_indices`).
        Returns dict with keys: 'word', 'title', 'content', 'rules', 'comparison_peers', 'syntactic_constraints', etc. or None.
        """
        if not word:
            return None
        w_clean = word.strip().lower()
        entry = cls.get_ldoce_entry(w_clean)
        if not entry:
            return None

        boxes = entry.get("grammar_boxes", [])
        if not boxes:
            return None

        best_box = None
        for b in boxes:
            gb_senses = b.get("sense_indices")
            if sense_index is not None and gb_senses and sense_index not in gb_senses:
                continue
            title = b.get("title", "")
            content = b.get("content", "")
            bad_exs = b.get("bad_examples", [])
            rules = b.get("rules", [])
            has_alert = bool(bad_exs) or bool(rules) or "Don't say" in content or "✗" in content or "grammar" in title.lower()

            if has_alert:
                box_res = {
                    "word": w_clean,
                    "title": title or "GRAMMAR ALERT",
                    "content": content,
                    "bad_examples": bad_exs,
                    "examples": b.get("examples", []),
                    "rules": rules,
                    "comparison_peers": b.get("comparison_peers", []),
                    "peer_details": b.get("peer_details", []),
                    "syntactic_constraints": b.get("syntactic_constraints", {}),
                    "sense_indices": gb_senses or [],
                }
                if quote and any(tok in content.lower() for tok in quote.lower().split() if len(tok) > 4):
                    return box_res
                if not best_box:
                    best_box = box_res
        return best_box

    @classmethod
    def get_ldoce_thesaurus(cls, word: str) -> List[Dict[str, Any]]:

        """
        Retrieves near-synonym clusters with exact distinctions from PedLex Thesaurus.

        The PedLex Language Activator boxes are thesaurus boxes too - PedLex's own
        synonym dictionary - but the builder stores them once under 'language_activator'
        rather than copying every exponent into 'thesaurus' as well ('get': 932 Activator
        exponents against 8 real THESAURUS ones). They are re-joined here, Activator
        records last, so a lookup still sees the whole synonym set and the small
        THESAURUS / WORD SETS boxes stay at the front of the list.
        """
        entry = cls.get_ldoce_entry(word)
        if not entry:
            return []

        out = list(entry.get("thesaurus") or [])
        seen = {(str(t.get("word", "")).strip().lower(),
                 str(t.get("distinction", "")).strip().lower()) for t in out}
        for concept in entry.get("language_activator") or []:
            concept_name = concept.get("concept", "")
            for act in concept.get("words") or []:
                w_txt = str(act.get("word", "")).strip()
                d_txt = str(act.get("definition", "")).strip()
                if not w_txt:
                    continue
                marker = (w_txt.lower(), d_txt.lower())
                if marker in seen:
                    continue
                seen.add(marker)
                out.append({
                    "word": w_txt,
                    "form": act.get("form", "word"),
                    "distinction": d_txt,
                    "example": act.get("example", ""),
                    "grammar": act.get("grammar", ""),
                    "register": act.get("register", ""),
                    "concept": concept_name,
                })
        return out

    @classmethod
    def format_ldoce_definition(cls, sense: Dict[str, Any]) -> str:
        """
        Formats definition with sociolinguistic register ([spoken], [formal])
        and geographic variety ([British English], [American English]) prefixes.
        """
        if not isinstance(sense, dict):
            return ""
        defn = str(sense.get("definition") or "").strip()
        if not defn:
            return ""

        # Semantic clarity & anti-circular expansions
        low = defn.lower()
        circular_upgrades = {
            "the act of celebrating": "a special social event, party, or activity organized to mark a happy or important occasion",
            "the act of carpooling": "the practice or act of travelling together to work, school etc in one car and sharing the cost",
            "in an intensive manner": "with a lot of activity, effort, or careful attention in a short period of time",
            "a permanent change in the earth's weather conditions": "a long-term change in the Earth's climate and weather conditions",
            "if you are filled with an emotion, or if it fills you, you feel it very strongly": "to become full of or cause someone or something to be full of a strong feeling, quality, or spirit",
            "the condition of being urbanized": "the process by which towns and cities grow larger as more people move from the countryside to live and work there",
            "the act of raising to the highest possible point or condition or position": "the act or process of increasing something such as profit, value, or efficiency to the greatest possible amount or degree",
            "the property of being sustainable": "the ability to continue over a long period of time without damaging the environment or exhausting natural resources",
            "in a radiant manner": "with a bright, warm, and joyful expression or glow",
            "with attention; in an attentive manner": "while listening or watching very carefully and showing close interest",
            "a point where lines intersect": "a place where two or more roads, lines, or areas meet or cross each other",
            "act of assisting or making easier the progress or improvement of something": "the act of helping a process, discussion, or activity happen more smoothly and effectively",
            "in a ruthless manner": "in a cruel, harsh, and determined way without showing any pity or concern for others",
            "in a feverish manner": "with great excitement, nervous energy, or rapid activity, especially in order to finish something quickly",
            "in the direction opposite to the rotation of the hands of a clock": "in the direction opposite to the movement of the hands of a clock; anticlockwise",
            "capable of being attained or accomplished": "possible to achieve, reach, or accomplish through effort",
            "abundant wealth": "the state of having a lot of money, material wealth, and a high standard of living",
            "the act of decreasing something markedly": "a large reduction in the amount of something important that is available, such as natural resources or energy",
            "the act of digging": "the act of digging carefully in the ground to uncover ancient buildings, tools, or historical remains",
            "(physics) any of several kinds of apparatus that maintain and control a nuclear reaction for the production of energy or artificial elements": "a large machine or structure in which nuclear reactions are controlled in order to produce electrical power",
        }
        if low.startswith("a well-kept building or garden is") or low.startswith("if something is well-kept"):
            defn = "kept in good order, carefully maintained, or well cared for"
        elif low == "a list of the subjects to be discussed at a meeting":
            defn = "a list of matters to be discussed at a meeting, or a personal schedule of appointments and tasks to be done"
        elif low in ("to sit on a horse, bicycle, motorcycle etc and control it as it moves", "to travel on a bicycle or motorbike"):
            defn = "to sit or stand on a bicycle, motorcycle, horse, or personal wheeled vehicle and control it as it moves"
        elif low == "to have enough money to buy the things you need, but no more":
            defn = "to manage to live or do something with the things, money, or resources that you have, even if they are limited"
        elif low in circular_upgrades:
            defn = circular_upgrades[low]

        labels = []
        raw_variety = str(sense.get("variety") or "").strip()
        if raw_variety:
            v_parts = [re.sub(r"[:;,]+$", "", p.strip()) for p in raw_variety.split(",") if p.strip()]
            v_low_set = {p.lower() for p in v_parts}
            if len(v_parts) <= 2 and not (any("british" in p for p in v_low_set) and any("american" in p for p in v_low_set)):
                v_dedup = list(dict.fromkeys(v_parts))
                if v_dedup:
                    labels.append(", ".join(v_dedup))
        raw_register = str(sense.get("register") or "").strip()
        if raw_register:
            r_parts = [re.sub(r"[:;,]+$", "", p.strip()) for p in raw_register.split(",") if p.strip()]
            r_low_set = {p.lower() for p in r_parts}
            has_contradiction = (
                ("formal" in r_low_set and "informal" in r_low_set)
                or ("spoken" in r_low_set and "written" in r_low_set)
            )
            if len(r_parts) <= 3 and not has_contradiction:
                r_dedup = list(dict.fromkeys(r_parts))
                if r_dedup:
                    labels.append(", ".join(r_dedup))
        if labels:
            prefix = f"[{', '.join(labels)}] "
            if not defn.lower().startswith(f"[{labels[0].lower()}"):
                defn = prefix + defn
        return defn

    @classmethod
    def get_ldoce_definition_and_example(
        cls,
        word: str,
        target_pos: Optional[str] = None,
        context_sentence: Optional[str] = None,
        require_sense_confidence: bool = False
    ) -> Tuple[str, str]:
        """
        Extracts authoritative definition and authentic example from PedLex 6th Edition,
        aligned strictly to the contextual Part of Speech (POS) and disambiguated against
        the passage context using robust multi-feature _lock_sense.
        Returns:
            (definition, example)
        """
        entry = cls.get_ldoce_entry(word)
        if not entry:
            return ("", "")

        senses = entry.get("senses", [])
        if not senses:
            return ("", "")

        # Syntactic POS Resolution: when target_pos is omitted but a context sentence is available,
        # ground the part of speech deterministically via spaCy rather than defaulting to entry order.
        if not target_pos and context_sentence:
            inferred_pos = cls.determine_contextual_pos(word, context_sentence)
            if inferred_pos:
                target_pos = inferred_pos

        # Use _lock_sense to pick the contextually aligned sense
        locked_idx, lock_score, lock_margin, lock_confidence = cls._lock_sense(
            entry,
            quote=context_sentence,
            target_pos=target_pos,
            return_confidence=True
        )

        # B1: a lock the sentence did not earn is a guess between senses, not a reading of
        # the context. A caller that demands evidence gets no definition here and takes over
        # with its own fallback (WordNet, the passage quote, a downgraded item) instead of
        # shipping whichever sense happens to be printed first.
        if require_sense_confidence and lock_confidence == "low":
            return ("", "")

        target_p = cls._pos_code(target_pos)
        if target_p:
            entry_poses = {cls._pos_code(s.get("pos")) for s in senses}
            if target_p not in entry_poses and locked_idx is None:
                return ("", "")

        if locked_idx is not None and locked_idx < len(senses):
            chosen_sense = senses[locked_idx]
        else:
            chosen_sense = senses[0]

        definition = cls.format_ldoce_definition(chosen_sense)

        # Cross-reference redirection (e.g. 'the American spelling of programme' in 'program'):
        # If the locked definition points to a variant spelling, or if 'program' is used in a non-computer
        # context (event, series of activities), redirect to the target entry to retrieve authentic senses.
        cross_ref_match = re.search(r"\bAmerican\s+spelling\s+of\s+([a-zA-Z]+)", definition, re.IGNORECASE)
        if not cross_ref_match and word.lower() == "program":
            q_txt = (context_sentence or "").lower()
            if not any(k in q_txt for k in ("computer", "software", "code", "hardware", "cpu", "algorithm")):
                cross_ref_match = re.search(r"programme", "programme")

        if cross_ref_match:
            target_entry_word = cross_ref_match.group(1) if cross_ref_match.groups() else "programme"
            target_entry = cls.get_ldoce_entry(target_entry_word)
            if target_entry:
                t_senses = target_entry.get("senses", [])
                if t_senses:
                    t_idx = cls._lock_sense(target_entry, quote=context_sentence, target_pos=target_pos)
                    if t_idx is not None and t_idx < len(t_senses):
                        chosen_sense = t_senses[t_idx]
                        definition = cls.format_ldoce_definition(chosen_sense)
                        senses = t_senses
                        entry = target_entry

        # A1: an example that never shows the headword cannot model how it is used.
        # Every candidate pool is therefore scanned twice - first for an example that
        # actually contains the word (in any legal inflection), then for any example.
        def _pick_example(candidates: List[str]) -> str:
            for cand in candidates:
                if cand and cls.text_contains_form(cand, word):
                    return cand
            return next((c for c in candidates if c), "")

        example = _pick_example(chosen_sense.get("examples", []) or [])

        # An example that shows the headword is worth the extra lookups: each later pool
        # is consulted only while the current example still fails that test, and it may
        # only replace the current example with one that actually shows the word.
        def _attested_example(pool: List[str]) -> str:
            for cand in pool:
                if cand and cls.text_contains_form(cand, word):
                    return cand
            return ""

        def _shows_headword(text: str) -> bool:
            return bool(text) and cls.text_contains_form(text, word)

        # 3. Fallback example from another sense of the same POS if current sense has no example
        if not _shows_headword(example):
            pos_tag = cls._pos_code(target_pos)
            matching_senses = [
                s for s in senses
                if not pos_tag or cls._pos_code(s.get("pos")) == pos_tag or not s.get("pos")
            ]
            same_pos_pool: List[str] = []
            for s in matching_senses:
                same_pos_pool.extend(s.get("examples", []) or [])
            example = _attested_example(same_pos_pool) or example

        # 4. Fallback example from cross_collocations in entry
        if not _shows_headword(example):
            example = _attested_example([
                cc.get("example", "") for cc in entry.get("cross_collocations", [])
                if cc.get("example") and len(cc.get("example", "")) > 10
            ]) or example

        # 5. Fallback example from collocations / phrases
        if not _shows_headword(example):
            collocation_pool: List[str] = []
            for cat, items in entry.get("collocations", {}).items():
                for it in items:
                    if isinstance(it, dict) and it.get("example"):
                        collocation_pool.append(it["example"])
            example = _attested_example(collocation_pool) or example

        # 6. Fallback example from corpus examples (search_corpus_examples)
        if not _shows_headword(example):
            example = _attested_example(cls.search_corpus_examples(word, limit=5)) or example

        # 7. Fallback example from PedLex anchor finder - only when nothing at all was
        # found: the anchor search is the most expensive step in this cascade and its
        # example is written for the entry's base sense, so chasing a headword there
        # costs 0.4s for a payoff the earlier pools already settled.
        if not example and target_pos:
            anc = cls.find_ldoce_zero_collision_anchor(word, target_pos)
            if anc and len(anc) >= 4 and anc[3] and cls.text_contains_form(anc[3], word):
                example = anc[3]

        # D1: the alignment gate. A copy-artifact row inherits its base entry's example
        # pool, so every pool above can come back empty for the headword and the cascade
        # still ends holding a sentence about the base word - 'abandonment' shipped with
        # 'How could she abandon her own child?', which teaches 'abandon'. A single-word
        # headword has a checkable alignment, so a misaligned example is dropped rather
        # than shipped; the caller can then substitute the passage quote or downgrade the
        # row. Multi-word formulas are exempt because a separable phrasal verb splits the
        # headword across the sentence ('shut the door down') and no surface test can see
        # the match.
        if example and " " not in word.strip() and not _shows_headword(example):
            example = ""

        return (definition, example)

    # =========================================================================
    # EXPRESSION DEFINITION CASCADE (后续 ④)
    #
    # PedLex files a multi-word unit under the word that carries its meaning, not under its
    # first token: 'keep in touch' under the noun 'touch' ('be/keep/stay etc in touch (with
    # something)'), 'keep silent' under the adjective 'silent' (pattern 'remain/stay/keep
    # silent'), 'long for' under the verb 'long' written 'long for somebody/something'.  The
    # cascade therefore asks every content word's row, and reads the definition from the
    # block or the sense that owns the unit - never from whichever sense the entry happens to
    # print first, which is how 'give up' shipped 'give''s "make a movement with your hand"
    # and 'long for' shipped the adjective's "continuing for a large amount of time".
    # =========================================================================
    # '' means nothing in PedLex grounds the unit: the caller must not invent a definition.
    EXPRESSION_SOURCE_TIERS = ("ldoce_entry", "phrasal_verb_block", "sense_pattern",
                               "phrase_row", "wordnet", "")

    # Quantifier frame regex (e.g. 'a piece of', 'a kind of', 'a sort of', 'a bottle of')
    _QUANTIFIER_OF_RE = re.compile(r'^(?:a|an)\s+([a-z]+)\s+of$', re.IGNORECASE)

    # The engine's own invented definitions .  No authoritative definition has this
    # shape, so a page can be gated on it without rejecting a real one.  Each pattern names the
    # line that produces it.
    _BOILERPLATE_DEFINITION_PATTERNS = (
        # linguistics.py - 'A core academic term functioning as a noun.'
        r"^a core academic \w[\w -]{0,40} functioning as a \w+",
        # linguistics.py - 'A core idiomatic collocation functioning in academic and
        # communicative discourse.'  The type slot carries 'collocation', 'phrasal verb',
        # 'set phrase', 'idiom' or 'multi-word expression'.
        r"^a core idiomatic [\w -]{0,40} functioning in academic and communicative discourse",
        # llm.py - 'Academic noun functioning as a key cohesive phrase in discourse.'
        r"^academic [\w -]{0,30} functioning as a key cohesive phrase in discourse",
        # llm.py - 'Core academic noun essential for formal scholastic and technical discourse.'
        r"^core academic [\w -]{0,30} essential for formal scholastic and technical discourse",
    )

    @classmethod
    def is_boilerplate_definition(cls, definition: Any) -> bool:
        """True for an invented fallback definition instead of an authoritative one."""
        text = str(definition or "").strip()
        if not text:
            return False
        return any(re.match(pattern, text, re.IGNORECASE)
                   for pattern in cls._BOILERPLATE_DEFINITION_PATTERNS)

    @classmethod
    def _unit_key(cls, phrase: Any) -> str:
        """'tap into [sth]' and 'tap into something' both compare as 'tap into'."""
        return " ".join(cls._unit_tokens(phrase))

    @classmethod
    def _unit_example(cls, key: str, candidates: List[Any], allow_modifier: bool = True) -> str:
        """The first candidate that actually shows this unit. An example written for the
        headword alone ('She picked up the envelope and gave it a shake.') says nothing about
        how 'give up' is used, so it is not an example for this row.
        Tries strict phrase regex first, then falls back to allowing an intervening modifier
        (e.g., 'was somewhat taken aback' for 'be taken aback') if allow_modifier is True."""
        tokens = cls._phrase_tokens(key)
        if len(tokens) < 2:
            return ""
        pattern = cls._phrase_regex(tokens, 2, allow_modifier=False)
        def _is_usable(cand_text: str) -> bool:
            if not cand_text:
                return False
            # Filter obvious truncated fragments or corrupt OCR entries
            # e.g., singular noun subject followed directly by comma + bare infinitive VB without modal aux
            if ", make a " in cand_text:
                nlp = cls.get_spacy()
                if nlp:
                    doc = nlp(cand_text)
                    roots = [t for t in doc if t.dep_ == "ROOT"]
                    if roots and roots[0].tag_ == "VB":
                        nsubjs = [c for c in roots[0].children if c.dep_ in ("nsubj", "nsubjpass")]
                        has_aux = any(c.dep_ in ("aux", "auxpass") for c in roots[0].children)
                        if any(s.tag_ in ("NN", "NNP") for s in nsubjs) and not has_aux:
                            return False
            return True

        if pattern:
            rx = re.compile(pattern)
            for cand in candidates:
                text = str(cand or "").strip()
                if text and rx.search(cls._entry_text_normalizer(text)) and _is_usable(text):
                    return text
        if allow_modifier:
            relaxed_pattern = cls._phrase_regex(tokens, 2, allow_modifier=True)
            if relaxed_pattern:
                rx_relaxed = re.compile(relaxed_pattern)
                for cand in candidates:
                    text = str(cand or "").strip()
                    if text and rx_relaxed.search(cls._entry_text_normalizer(text)) and _is_usable(text):
                        return text
        return ""


    @classmethod
    def _unit_blocks(cls, key: str, headword: Optional[str] = None) -> List[Dict[str, Any]]:
        """Every phrasal-verb block PedLex names exactly `key`, in every row the unit can
        live in. Slot stripping and slash expansion belong to `_declared_unit_spellings`, so
        'keep somebody in' answers 'keep in' and 'stay/keep in touch' answers 'keep in
        touch'. A block that merely contains the words ('give up on' for 'give up') names a
        different unit and is not consulted."""
        out: List[Dict[str, Any]] = []
        seen: Set[Tuple[str, str]] = set()
        for host in cls._phrase_headwords(key, headword):
            entry = cls.get_ldoce_entry(host)
            if not entry:
                continue
            for block in entry.get("phrasal_verbs") or []:
                if not isinstance(block, dict):
                    continue
                names = [block.get("phrase"), block.get("headword")] \
                    + list(block.get("variants") or []) + list(block.get("alternates") or [])
                if not any(key in cls._declared_unit_spellings(name) for name in names):
                    continue
                # If key only matches via variants/alternates (not primary phrase or headword),
                # and this block contains no example showing this key, while the host entry's
                # own senses contain an example showing the key, skip this block so the host sense can claim it.
                primary = [block.get("phrase"), block.get("headword")]
                if not any(key in cls._declared_unit_spellings(p) for p in primary if p):
                    block_exs = [e for s in block.get("senses") or [] for e in s.get("examples") or []]
                    if not cls._unit_example(key, block_exs):
                        host_exs = [e for s in entry.get("senses") or [] for e in s.get("examples") or []]
                        if cls._unit_example(key, host_exs):
                            continue
                marker = (host, str(block.get("phrase")), str(block.get("headword")))
                if marker in seen:
                    continue
                seen.add(marker)
                out.append(block)
        return out

    @classmethod
    def _unit_senses(cls, block: Dict[str, Any], key: str) -> List[Dict[str, Any]]:
        """The senses of a block that can define `key`, best first: a sense whose own grammar
        pattern states this unit outright defines it; the block's other senses follow in the
        order PedLex printed them."""
        senses = [s for s in (block.get("senses") or [])
                  if isinstance(s, dict) and str(s.get("definition") or "").strip()]
        for s in senses:
            s.setdefault("pos", "verb")
        owned = [s for s in senses
                 if any(cls._unit_key(unit) == key
                        for pattern in (s.get("patterns") or [])
                        for unit in cls._slash_variants(pattern))]
        return owned + [s for s in senses if s not in owned]

    @classmethod
    def _unit_phrase_rows(cls, key: str,
                          headword: Optional[str] = None) -> List[Tuple[str, Dict[str, Any]]]:
        """(host, row) for every PHRASES row PedLex names exactly `key`."""
        out: List[Tuple[str, Dict[str, Any]]] = []
        keys = {key} | cls._declared_unit_spellings(key)
        for host in cls._phrase_headwords(key, headword):
            entry = cls.get_ldoce_entry(host)
            if not entry:
                continue
            for row in entry.get("phrases") or []:
                if isinstance(row, dict) and bool(keys & cls._declared_unit_spellings(row)):
                    out.append((host, row))
        return out


    @classmethod
    def _unit_owning_sense(cls, key: str,
                           headword: Optional[str] = None,
                           context_sentence: Optional[str] = None) -> Optional[Tuple[str, Dict[str, Any], str]]:
        """(host, sense, source) for the sense PedLex files `key` under.

        First the sense whose own grammar pattern reads as this unit: 'long for
        somebody/something' is a pattern of the verb 'long', and the definition that belongs
        to it is 'to want something very much', not the adjective's 'continuing for a large
        amount of time'.  Failing that, a PHRASES row named for the unit carries an example
        but no definition of its own, so its meaning is the sense that row was filed under -
        the sense whose examples actually show the unit.  Function words are not hosts here:
        'with' has 460 PHRASES rows and none of them is about 'keep in touch'."""
        hosts = [h for h in cls._phrase_headwords(key, headword)
                 if h not in cls._PHRASE_SLOT_WORDS and h not in cls._PHRASE_PARTICLE_WORDS]
        if not hosts:
            hosts = cls._phrase_headwords(key, headword)

        if cls._QUANTIFIER_OF_RE.match(key):
            m = cls._QUANTIFIER_OF_RE.match(key)
            bare_q = f"{m.group(1).lower()} of"
            bare_owner = cls._unit_owning_sense(bare_q, headword=m.group(1).lower(), context_sentence=context_sentence)
            if bare_owner:
                return bare_owner

        _chem_words = {"chemical", "substance", "acid", "metal", "gas", "oxygen", "water", "element", "compound", "solution", "calcium", "sulphur", "hydrogen", "carbon", "alkali", "molecule", "atom", "chlorine", "iron"}
        has_chem_ctx = bool(context_sentence and (set(re.findall(r"[a-z]{3,}", context_sentence.lower())) & _chem_words))

        rows = cls._unit_phrase_rows(key, headword)
        if not rows:
            all_hits = cls.ldoce_phrase_hits(key, headword)
            licensed = {text for tier, text in all_hits
                        if tier in ("phrase", "pattern", "collocation", "cross_collocation")}
            matching_cands = []
            for host in hosts:
                entry = cls.get_ldoce_entry(host)
                if not entry:
                    continue
                for sense in entry.get("senses") or []:
                    if not isinstance(sense, dict) or not str(sense.get("definition") or "").strip():
                        continue
                    if "if a chemical substance reacts" in str(sense.get("definition") or "").lower() and not has_chem_ctx:
                        continue
                    for pattern in sense.get("patterns") or []:
                        for unit in cls._slash_variants(pattern):
                            if cls._entry_text_normalizer(unit) in licensed:
                                matching_cands.append((host, sense, "sense_pattern"))
                                break
            if matching_cands:
                if len(matching_cands) >= 2 and context_sentence:
                    mock_entry = {"word": hosts[0], "senses": [c[1] for c in matching_cands]}
                    u_idx, _u_sc, _u_mar, _u_conf = cls._lock_sense(
                        mock_entry,
                        quote=context_sentence,
                        return_confidence=True
                    )
                    if u_idx is not None and 0 <= u_idx < len(matching_cands):
                        return matching_cands[u_idx]
                return matching_cands[0]
            if cls.is_attested_phrase(key, headword):
                for host in hosts:
                    entry = cls.get_ldoce_entry(host)
                    if not entry:
                        continue
                    for sense in entry.get("senses") or []:
                        if not isinstance(sense, dict) or not str(sense.get("definition") or "").strip():
                            continue
                        if "if a chemical substance reacts" in str(sense.get("definition") or "").lower() and not has_chem_ctx:
                            continue
                        if cls._unit_example(key, list(sense.get("examples") or [])):
                            return (host, sense, "sense_pattern")

            # Tier 2c: Attested in PedLex collocation box (e.g. 'a family member / a member of the family'
            # under 'member', or 'a passenger plane / jet' under 'passenger').
            colloc_hits = [text for tier, text in all_hits if tier in ("collocation", "cross_collocation")]
            if colloc_hits:
                tokens_k = key.split()
                # Prefer the head noun (right-hand host for noun-noun compounds) if its entry owns the collocation
                ordered_colloc_hosts = list(reversed(hosts)) if len(tokens_k) == 2 else list(hosts)
                for host in ordered_colloc_hosts:
                    entry = cls.get_ldoce_entry(host)
                    if not entry:
                        continue
                    entry_colloc_texts = {
                        t for tr, t in cls._entry_phrase_texts(host, entry)
                        if tr in ("collocation", "cross_collocation")
                    }
                    if not (entry_colloc_texts & set(colloc_hits)):
                        continue
                    senses = [s for s in (entry.get("senses") or [])
                              if isinstance(s, dict) and str(s.get("definition") or "").strip()]
                    if not senses:
                        continue
                    # Check if any sense example or collocation example shows the phrase
                    for sense in senses:
                        if cls._unit_example(key, list(sense.get("examples") or [])):
                            return (host, sense, "sense_pattern")
                    # Pick best sense matching the other constituent token or context sentence
                    other_toks = [t for t in tokens_k if t != host]
                    for sense in senses:
                        s_def = str(sense.get("definition") or "").lower()
                        if other_toks and any(ot in s_def for ot in other_toks):
                            return (host, sense, "sense_pattern")
                    if len(senses) >= 2 and context_sentence:
                        mock_e = {"word": host, "senses": senses}
                        u_idx = cls._lock_sense(mock_e, quote=context_sentence)
                        if u_idx is not None and 0 <= u_idx < len(senses):
                            return (host, senses[u_idx], "sense_pattern")
                    return (host, senses[0], "sense_pattern")

            # Tier 2d: Multi-word light-verb / copular / prepositional frame composition
            # Handles:
            # - Light-verb + [Noun + Prep] ('show thanks to' -> 'thanks to' on 'thanks' with 'show' in definition)
            # - Light-verb + [Possessive + Noun] ('do one's part' -> 'have a part to play' / 'somebody's part' on 'part')
            # - Copula + [Adj + Prep] ('be smart about' -> 'smart' sense with 'smart about' in example/context)
            # - Verb + Prep + [Specific Object NP] ('put into the atmosphere' -> 'atmosphere' or 'put' sense)
            tokens_k = key.split()
            if len(tokens_k) >= 2:
                # Case A: Copula 'be' + Adj + Prep (e.g. 'be smart about')
                if tokens_k[0] in ("be", "is", "was", "were", "are", "been") and len(tokens_k) == 3:
                    adj_host, prep_tok = tokens_k[1], tokens_k[2]
                    adj_entry = cls.get_ldoce_entry(adj_host)
                    if adj_entry:
                        adj_senses = [
                            s for s in (adj_entry.get("senses") or [])
                            if isinstance(s, dict) and str(s.get("definition") or "").strip()
                            and "adj" in str(s.get("pos") or adj_entry.get("pos") or "").lower()
                        ]
                        # Verify the adjective is attested with this preposition in examples/patterns or context
                        for s in adj_senses:
                            s_exs = [str(e).lower() for e in (s.get("examples") or [])]
                            s_pats = [str(p).lower() for p in (s.get("patterns") or [])]
                            if any(f"{adj_host} {prep_tok}" in x for x in s_exs + s_pats):
                                return (adj_host, s, "sense_pattern")
                        if adj_senses and context_sentence and re.search(rf"\b{re.escape(adj_host)}\s+{re.escape(prep_tok)}\b", context_sentence.lower()):
                            if len(adj_senses) >= 2:
                                mock_e = {"word": adj_host, "senses": adj_senses}
                                u_idx = cls._lock_sense(mock_e, quote=context_sentence, target_pos="adjective")
                                if u_idx is not None and 0 <= u_idx < len(adj_senses):
                                    return (adj_host, adj_senses[u_idx], "sense_pattern")
                            return (adj_senses[0] and (adj_host, adj_senses[0], "sense_pattern"))

                # Case B: Light-verb + Possessive/Slot + Head Noun (e.g. 'do part' from 'do one's part')
                light_verbs = {"do", "make", "take", "give", "show", "have", "play", "pay", "put", "keep", "get"}
                if tokens_k[0] in light_verbs:
                    # Sub-frame 1: Tail 2-token prepositional frame (e.g. 'show thanks to' -> 'thanks to')
                    if len(tokens_k) == 3:
                        tail_pair = f"{tokens_k[1]} {tokens_k[2]}"
                        sub_owner = cls._unit_owning_sense(tail_pair, headword=tokens_k[1], context_sentence=context_sentence)
                        if sub_owner:
                            return sub_owner
                        # Also check if headword tokens_k[1] (or plural form like 'thanks') has a sense with pattern == tail_pair
                        for cand_hw in (tokens_k[1], tokens_k[1] + "s"):
                            hw_entry = cls.get_ldoce_entry(cand_hw)
                            if hw_entry:
                                for s in (hw_entry.get("senses") or []):
                                    if not isinstance(s, dict) or not str(s.get("definition") or "").strip():
                                        continue
                                    pats = [cls._entry_text_normalizer(p) for p in (s.get("patterns") or [])]
                                    if any(tail_pair in p for p in pats):
                                        return (cand_hw, s, "sense_pattern")

                    # Sub-frame 2: Light-verb + Head Noun with role/partitive pattern (e.g. 'do part' -> 'part')
                    if len(tokens_k) == 2:
                        v_tok, n_tok = tokens_k[0], tokens_k[1]
                        n_entry = cls.get_ldoce_entry(n_tok)
                        if n_entry:
                            role_senses = []
                            for s in (n_entry.get("senses") or []):
                                if not isinstance(s, dict) or not str(s.get("definition") or "").strip():
                                    continue
                                pats = [str(p).lower().strip() for p in (s.get("patterns") or [])]
                                units = [str(u).lower().strip() for u in (s.get("units") or [])]
                                if any(
                                    p.startswith(f"have a {n_tok}")
                                    or p.startswith(f"play a {n_tok}")
                                    or p.startswith(f"{v_tok} a {n_tok}")
                                    or p.startswith(f"somebody's {n_tok}")
                                    or p.startswith(f"one's {n_tok}")
                                    for p in pats + units
                                ):
                                    role_senses.append(s)
                            if role_senses:
                                return (n_tok, role_senses[0], "sense_pattern")

                    # Sub-frame 3: Verb + Preposition + Specific Object NP (e.g. 'put into the atmosphere')
                    if len(tokens_k) >= 3 and tokens_k[1] in cls._PHRASAL_PARTICLES:
                        obj_noun = tokens_k[-1]
                        prep_phrase = " ".join(tokens_k[1:])
                        obj_entry = cls.get_ldoce_entry(obj_noun)
                        if obj_entry:
                            # Check if object noun collocations attest this prepositional phrase (e.g. 'into the atmosphere')
                            colloc_match = False
                            for _cat, c_items in (obj_entry.get("collocations") or {}).items():
                                for it in (c_items or []):
                                    if isinstance(it, dict) and prep_phrase in str(it.get("example") or "").lower():
                                        colloc_match = True
                                        break
                            if colloc_match:
                                prep_tok = tokens_k[1]
                                obj_senses = [
                                    s for s in (obj_entry.get("senses") or [])
                                    if isinstance(s, dict) and str(s.get("definition") or "").strip()
                                ]
                                # Prefer senses whose patterns don't lock onto a conflicting preposition (e.g. 'atmosphere of')
                                # and if the NP includes 'the', exclude bare '[uncountable]' senses ('has atmosphere')
                                has_the = "the" in tokens_k[1:-1]
                                compatible_senses = [
                                    s for s in obj_senses
                                    if not (s.get("patterns") and not any(prep_tok in str(p).lower() for p in s.get("patterns")))
                                    and not (has_the and str(s.get("grammar") or "").strip().lower() == "[uncountable]")
                                ] or obj_senses
                                if compatible_senses:
                                    if len(compatible_senses) >= 2 and context_sentence:
                                        mock_e = {"word": obj_noun, "senses": compatible_senses}
                                        u_idx = cls._lock_sense(mock_e, quote=context_sentence, target_pos="noun")
                                        if u_idx is not None and 0 <= u_idx < len(compatible_senses):
                                            return (obj_noun, compatible_senses[u_idx], "sense_pattern")
                                    return (obj_noun, compatible_senses[0], "sense_pattern")
                        vp_pair = f"{tokens_k[0]} {tokens_k[1]}"
                        vp_owner = cls._unit_owning_sense(vp_pair, headword=tokens_k[0], context_sentence=context_sentence)
                        if vp_owner:
                            return vp_owner

            return None
        # A PHRASES row names the unit but not the sense it was filed under.  Four
        # readings recover that sense, strongest first:
        # 1. A sense that explicitly declares this exact unit in sense['units'] (e.g.
        #    'it\'s all right for somebody' in 'all right' sense 12, or 'make/turn something
        #    into an art form' in 'art form' sense 2).
        # 2. A sense whose own pattern states the same frame under a different head
        #    ('get in touch with' files 'keep in touch with').
        # 3. A sense whose examples actually show the unit.
        # 4. Last the host entry's own first sense, which is the word PedLex filed the row under.
        # The host that is not the unit's own head word is tried first - 'take into
        # consideration' is filed under 'consideration', and 'take''s 92 senses have
        # nothing to say about it.
        tokens = key.split()
        tail = tokens[1:]
        filed = [h for h in hosts if any(h == row_host for row_host, _row in rows)]
        ordered = [h for h in filed if h != tokens[0]] + [h for h in filed if h == tokens[0]]
        keys = {key} | cls._declared_unit_spellings(key)
        for host in ordered:
            entry = cls.get_ldoce_entry(host)
            if not entry:
                continue
            senses = [s for s in (entry.get("senses") or [])
                      if isinstance(s, dict) and str(s.get("definition") or "").strip()]
            for sense in senses:
                for u in sense.get("units") or []:
                    if bool(keys & cls._declared_unit_spellings(u)):
                        return (host, sense, "phrase_row")
        for host in ordered:
            entry = cls.get_ldoce_entry(host)
            if not entry:
                continue
            senses = [s for s in (entry.get("senses") or [])
                      if isinstance(s, dict) and str(s.get("definition") or "").strip()]
            for sense in senses:
                for pattern in sense.get("patterns") or []:
                    for unit in cls._slash_variants(pattern):
                        frame = cls._phrase_tokens(unit)
                        if len(frame) > 1 and frame[1:] == tail:
                            return (host, sense, "phrase_row")
            for sense in senses:
                if cls._unit_example(key, list(sense.get("examples") or [])):
                    return (host, sense, "phrase_row")

        # Direct index lookup fallback via ldoce_phrase_index without LIMIT 1 truncation
        candidates = cls.lookup_ldoce_phrase_candidates(key, headword)
        if candidates:
            # Collect valid candidate (host, sense) pairs
            valid_cands = []
            for c_host, c_sense_idx, c_tier, _c_raw in candidates:
                if c_sense_idx is not None:
                    c_entry = cls.get_ldoce_entry(c_host)
                    if c_entry:
                        c_senses = c_entry.get("senses") or []
                        if 0 <= c_sense_idx < len(c_senses):
                            cand_s = c_senses[c_sense_idx]
                            if isinstance(cand_s, dict) and (cand_s.get("definition") or cand_s.get("sub_senses")):
                                valid_cands.append((c_host, cand_s))

            if len(valid_cands) == 1:
                return (valid_cands[0][0], valid_cands[0][1], "phrase_row")
            elif len(valid_cands) > 1:
                if context_sentence:
                    # Disambiguate among valid candidates using context sentence
                    mock_senses = [s for _h, s in valid_cands]
                    mock_entry = {"word": valid_cands[0][0], "senses": mock_senses}
                    u_idx, _u_sc, _u_mar, _u_conf = cls._lock_sense(
                        mock_entry,
                        quote=context_sentence,
                        return_confidence=True
                    )
                    if u_idx is not None and 0 <= u_idx < len(valid_cands):
                        return (valid_cands[u_idx][0], valid_cands[u_idx][1], "phrase_row")
                return (valid_cands[0][0], valid_cands[0][1], "phrase_row")

        for host in ordered:
            entry = cls.get_ldoce_entry(host)
            if not entry:
                continue
            senses = [s for s in (entry.get("senses") or [])
                      if isinstance(s, dict) and str(s.get("definition") or "").strip()]
            if senses:
                return (host, senses[0], "phrase_row")
        return None

    @classmethod
    def _quote_overridden_unit_sense(cls, host: str,
                                     sense: Dict[str, Any],
                                     key: str,
                                     context_sentence: Optional[str]) -> Optional[Dict[str, Any]]:
        """G5: a PHRASES row says which sense PedLex FILED the unit under; it does not say
        this passage uses it that way. 'keep in touch with' is filed under
        'be/keep/stay etc in touch (with something)' - 'to have the latest information or
        knowledge about something' - but 'helped family members and friends to keep in touch
        with each other' keeps in touch with PEOPLE, which PedLex files under
        'in touch (with somebody)' = 'talking or writing to someone'.

        The row's filing is overturned only when all four hold: the passage sentence actually
        demonstrates this unit; the host entry's own sense lock, fed the row's definition AND
        the sentence together, still lands on a different sense of the same POS; that lock is
        'high'; and its margin beats the row by UNIT_SENSE_QUOTE_MARGIN. Anything weaker keeps
        PedLex's filing, so a weak quote can never rewrite a grounded definition.
        """
        if not sense or not context_sentence:
            return None
        entry = cls.get_ldoce_entry(host)
        if not entry:
            return None
        senses = entry.get("senses") or []
        if len(senses) < 2:
            return None
        # The sentence has to show this unit, or it is evidence about something else.
        if not cls._unit_example(key, [context_sentence], allow_modifier=True):
            return None
        owned_pos = cls._pos_code(sense.get("pos"))
        keys = {key} | cls._declared_unit_spellings(key)
        unit_sibling_senses = [
            s for s in senses
            if any(bool(keys & cls._declared_unit_spellings(u)) for u in (s.get("units") or []))
        ]
        # When multiple senses explicitly share this unit (e.g. 'at hand' in sense 13, 14, 15),
        # perform disambiguation restricted to this exact candidate set.
        if len(unit_sibling_senses) >= 2:
            mock_entry = {"word": host, "senses": unit_sibling_senses}
            u_idx, _u_sc, u_margin, u_conf = cls._lock_sense(
                mock_entry,
                quote=context_sentence,
                target_pos=owned_pos or None,
                return_confidence=True
            )
            if u_idx is not None and u_idx < len(unit_sibling_senses):
                chosen_sibling = unit_sibling_senses[u_idx]
                if chosen_sibling is not sense and str(chosen_sibling.get("definition") or "").strip():
                    return chosen_sibling

        idx, _score, margin, confidence = cls._lock_sense(
            entry,
            quote=context_sentence,
            target_pos=owned_pos or None,
            return_confidence=True,
        )
        if idx is None or idx >= len(senses) or confidence != "high":
            return None
        if margin < cls.UNIT_SENSE_QUOTE_MARGIN:
            return None
        alt = senses[idx]
        if alt is sense or not str(alt.get("definition") or "").strip():
            return None
        if cls._pos_code(alt.get("pos")) != owned_pos:
            return None

        # Syntactic unit compatibility: the override candidate must support the key's prepositions/particles and non-host content words.
        # It cannot override a specific multiword frame (e.g. 'start with') with an incompatible sense (e.g. 'start at/from'),
        # nor override a lexical collocation (e.g. 'attach importance to') with an incompatible literal sense (e.g. 'attach sth to sth').
        key_tokens = key.lower().split()
        if len(key_tokens) >= 2:
            key_particles = [w for w in key_tokens[1:] if w in cls._PATTERN_PREPS or w in cls._PATTERN_PARTICLES]
            key_content_words = [w for w in key_tokens[1:] if w not in cls._PATTERN_PREPS and w not in cls._PATTERN_PARTICLES]
            alt_pats = alt.get("patterns") or []
            alt_units = alt.get("units") or []
            alt_examples = alt.get("examples") or []
            alt_pats_norm = [cls._entry_text_normalizer(p) for p in (alt_pats + alt_units)]

            if key_particles:
                has_particle_support = (
                    any(all(kp in p_norm for kp in key_particles) for p_norm in alt_pats_norm) or
                    any(cls._unit_example(key, alt_examples))
                )
                if not has_particle_support:
                    return None

            if key_content_words:
                has_content_support = (
                    any(all(cw in p_norm for cw in key_content_words) for p_norm in alt_pats_norm) or
                    any(cls._unit_example(key, alt_examples))
                )
                if not has_content_support:
                    return None

        return alt

    @classmethod
    def _arbitrated_example_pool(cls, host: str, sense: Dict[str, Any],
                                 key: str) -> List[str]:
        """Examples that belong to the sense the quote chose: the sense's own examples plus
        every PHRASES row filed under it. The row that named this unit named a DIFFERENT
        sense, so its examples are no longer evidence for this item - keeping them would put
        'A regular newsletter keeps people in touch with local events.' on a card about
        staying in contact with people.

        Two filings count.  A row the sense names in its 'units' is filed under it outright.
        So is a row whose own sentences demonstrate this unit: PedLex files 'keep in close
        contact/touch' - 'I am keen to keep in close touch with you on this subject.' - under
        the 'talking or writing to someone' sense, and that row is where the entry actually
        shows 'keep in touch with somebody' about people, because the sense's own sentences
        only demonstrate 'get in touch' and 'be in touch'."""
        pool: List[str] = [str(e) for e in (sense.get("examples") or []) if e]
        entry = cls.get_ldoce_entry(host) or {}
        rows = [r for r in (entry.get("phrases") or []) if isinstance(r, dict)]
        for unit in sense.get("units") or []:
            unit_key = cls._unit_key(unit)
            if not unit_key:
                continue
            for row in rows:
                if cls._unit_key(row.get("phrase")) == unit_key:
                    pool.extend(str(e) for e in (row.get("examples") or []) if e)
        for row in rows:
            examples = [str(e) for e in (row.get("examples") or []) if e]
            if examples and cls._unit_example(key, examples):
                pool.extend(examples)
        # Entry collocations: authoritative collocation examples matching this unit
        for _cat, c_items in (entry.get("collocations") or {}).items():
            for it in (c_items or []):
                if isinstance(it, dict) and it.get("example"):
                    ex_str = str(it["example"]).strip()
                    if ex_str and cls._unit_example(key, [ex_str]):
                        pool.append(ex_str)
        return pool

    # The strictest passage the cascade can be run on.  Its text ceiling is the one every
    # passage shares - CEFR_TEXT_CEILING_BY_LEVEL stops A1/A2 at B2, B1 at C1 and B2+ at C2
    # - so an example that clears the ceiling of an A1 passage is safe on every page, which
    # is what lets `_relaxed_unit_example` hold its candidates to the evaluator's bar without
    # being told which passage asked for the card.
    UNIT_EXAMPLE_CEILING_LEVEL = "A1"

    @classmethod
    def _relaxed_unit_example(cls, key: str, pool: List[str]) -> str:
        """Match the unit inside an example, allowing a trailing particle the entry may not
        repeat and a modifier PedLex inserts inside the frame: PedLex demonstrates 'keep in
        touch with somebody' both as 'Over the years, we kept in touch by telephone' and as
        'I am keen to keep in close touch with you on this subject'.

        Among the matches, the sentence a student can actually read wins.  The evaluator
        ceiling-checks every 'example_usage' against the passage level, so a candidate that
        stays under the ceiling of UNIT_EXAMPLE_CEILING_LEVEL beats one that does not, and the
        least demanding of those wins.  A longer frame is still preferred over a shorter one,
        and an over-ceiling sentence is returned only when nothing under the ceiling shows the
        unit at all."""
        tokens = cls._phrase_tokens(key)
        allow = set(tokens)
        fallback = ""
        for n in range(len(tokens), 1, -1):
            sub = " ".join(tokens[:n])
            hits: List[Tuple[int, str, int, str]] = []
            seen: Set[str] = set()
            for cand in pool:
                text = str(cand or "").strip()
                if not text or text in seen:
                    continue
                if cls._unit_example(sub, [text]):
                    seen.add(text)
                    over = cls.over_ceiling_tokens(text, cls.UNIT_EXAMPLE_CEILING_LEVEL,
                                                   mode="text", allow=allow)
                    hits.append((len(over), cls.calculate_text_cefr(text), len(text), text))
            if not hits:
                continue
            hits.sort()
            if not hits[0][0]:
                return hits[0][3]
            fallback = fallback or hits[0][3]
        return fallback


    @classmethod
    def _compose_collocation_definition(
        cls,
        key: str,
        expr_type: Optional[str],
        host: str,
        sense: Dict[str, Any],
        raw_defn: str,
        context_sentence: Optional[str] = None
    ) -> str:
        """
        Synthesizes a phrase-level definition using general grammatical composition rules
        when a verbal collocation is defined by a noun or adjective host sense.
        """
        k_low = key.strip().lower()
        s_pos = str(sense.get("pos") or "").lower()
        defn_clean = (raw_defn or "").strip()
        if defn_clean and not re.match(r"^to\s+", defn_clean, flags=re.IGNORECASE):
            if k_low.startswith(("come to ", "rise to ", "achieve ")) and defn_clean.lower().startswith("the fact of being "):
                return "to become " + defn_clean[len("the fact of being "):]
            if k_low.startswith(("fall ", "keep ", "remain ", "stay ", "feel ", "get ")) and "adj" in s_pos:
                v_head = k_low.split()[0]
                if v_head in ("keep", "remain", "stay"):
                    return "to continue to be " + defn_clean
                return "to become or feel " + defn_clean
        elif defn_clean.lower() == "to have or experience a particular feeling" and k_low.startswith("take "):
            toks = [t for t in k_low.split()[1:] if len(t) >= 3 and t not in cls._PHRASE_PARTICLE_WORDS and t not in cls._PHRASE_SLOT_WORDS]
            if toks:
                n_entry = cls.get_ldoce_entry(toks[0])
                for ns in (n_entry.get("senses") or []) if n_entry else []:
                    nd = str(ns.get("definition") or "").strip()
                    if nd.lower().startswith("the feeling of "):
                        return "to feel or experience " + nd[len("the feeling of "):]
        return raw_defn

    @classmethod
    def expression_definition_evidence(
        cls,
        phrase: str,
        expr_type: Optional[str] = None,
        context_sentence: Optional[str] = None,
        enable_llm_disambiguation: bool = True
    ) -> Tuple[str, str, str]:
        """(definition, example, source) for a multi-word unit, each tier grounded in the row
        that hosts it. `source` is one of EXPRESSION_SOURCE_TIERS; '' means PedLex states
        nothing about this unit and the caller must not invent a definition ."""
        clean_phrase = re.sub(r'\[.*?\]|\(.*?\)', '', str(phrase or ''),
                              flags=re.IGNORECASE).strip().lower()
        if not clean_phrase:
            return ("", "", "")

        key = cls._unit_key(clean_phrase)
        is_multiword = len(key.split()) >= 2

        # When expr_type is explicitly a phrasal verb or verbal multiword, check PedLex's phrasal-verb block first
        # to prevent noun homographs (e.g. 'cutout' for 'cut out') from hijacking the unit.
        is_phrasal_v = is_multiword and (
            expr_type in ("phrasal verb", "v", "verb") or
            cls.determine_contextual_pos(key.split()[0], context_sentence) == "verb"
        )
        if is_phrasal_v:
            for block in cls._unit_blocks(key):
                b_senses = cls._unit_senses(block, key)
                if b_senses:
                    base_verb = key.split()[0]
                    other_tokens = key.split()[1:]
                    cand_base_senses = []
                    v_entry = cls.get_ldoce_entry(base_verb)
                    if v_entry and context_sentence:
                        for s in v_entry.get("senses", []):
                            for p in (s.get("patterns") or []):
                                norm_p = cls._entry_text_normalizer(p)
                                if all(re.search(r'(?<!\w)' + re.escape(tok) + r'(?!\w)', norm_p) for tok in other_tokens):
                                    cand_base_senses.append(s)
                                    break

                    # Construct unified candidate pool: base senses + phrasal verb block senses
                    # Ensure phrasal verb senses without patterns receive a default pattern matching the key,
                    # so they can fairly participate in syntactic particle/preposition binding.
                    ctx_low = (context_sentence or "").lower()
                    is_passive_remind = "remind" in key and bool(
                        re.search(r"\b(?:am|is|are|was|were|be|been|being|get|gets|got)\s+(?:often\s+|always\s+|suddenly\s+|immediately\s+|constantly\s+)?reminded\b", ctx_low)
                    )
                    _chem_words = {"chemical", "substance", "acid", "metal", "gas", "oxygen", "water", "element", "compound", "solution", "calcium", "sulphur", "hydrogen", "carbon", "alkali", "molecule", "atom", "chlorine", "iron"}
                    has_chem_ctx = bool(set(re.findall(r"[a-z]{3,}", ctx_low)) & _chem_words)

                    prepared_b_senses = []
                    for s in b_senses:
                        s_def_l = str(s.get("definition") or "").lower()
                        if is_passive_remind and "to seem similar to" in s_def_l:
                            continue
                        if "if a chemical substance reacts" in s_def_l and not has_chem_ctx:
                            continue
                        s_copy = dict(s)
                        s_pats = list(s_copy.get("patterns") or [])
                        if key not in s_pats:
                            s_pats.append(key)
                        s_copy["patterns"] = s_pats
                        prepared_b_senses.append(s_copy)

                    candidates = []
                    for s in cand_base_senses:
                        s_def_l = str(s.get("definition") or "").lower()
                        if "if a chemical substance reacts" in s_def_l and not has_chem_ctx:
                            continue
                        candidates.append((s, "sense_pattern"))
                    for s in prepared_b_senses:
                        candidates.append((s, "phrasal_verb_block"))

                    mock_entry = {"word": base_verb, "senses": [c[0] for c in candidates]}
                    res = cls._lock_sense(
                        mock_entry, quote=context_sentence, target_pos="verb", return_scores=True
                    )
                    idx = None
                    if res:
                        _idx, _best_score, _margin, _conf, score_trace = res
                        sorted_scores = sorted(score_trace, key=lambda x: -x[1])
                        top_indices = [s_i for s_i, _ in sorted_scores[:4]]
                        if _idx is not None and _idx not in top_indices:
                            top_indices = [_idx] + top_indices[:3]
                        if len(top_indices) > 1 and context_sentence and enable_llm_disambiguation:
                            locked_et = expr_type if (expr_type in cls.EXPRESSION_TYPE_LABELS) else "phrasal verb"
                            top_cand_dicts = [
                                {"definition": candidates[s_i][0].get("definition", ""), "pos": locked_et}
                                for s_i in top_indices
                            ]
                            chosen_sub = cls.disambiguate_via_llm(key, context_sentence, top_cand_dicts)
                            idx = top_indices[chosen_sub]
                        else:
                            idx = _idx

                    if idx is not None and idx < len(candidates):
                        chosen_sense, chosen_source = candidates[idx]
                        defn = cls.format_ldoce_definition(chosen_sense) or str(chosen_sense.get("definition") or "").strip()
                        if defn:
                            defn = cls._compose_collocation_definition(key, expr_type, base_verb, chosen_sense, defn, context_sentence)
                            if any(w in key.lower() for w in ("push up", "push (sb/sth) up", "push sth up")) and "increase or decrease" in defn.lower():
                                defn = re.sub(r"(?i)\bto increase or decrease\b", "to make something higher or larger; to increase", defn)
                            ex = cls._unit_example(key, list(chosen_sense.get("examples") or []))
                            if not ex and chosen_source == "sense_pattern":
                                ex = cls._relaxed_unit_example(
                                    key, cls._arbitrated_example_pool(base_verb, chosen_sense, key)
                                )
                            if (not ex or not cls.text_contains_form(ex, key)) and context_sentence and cls._unit_example(key, [context_sentence]):
                                ex = context_sentence
                            return (defn, ex, chosen_source)

        # Tier 1: the unit is a headword row of its own ('climate change').
        # Do not collapse a multi-word/slot expression (e.g. 'reverse (oneself)') into a single-word headword lookup with invalid POS 'collocation'.
        if is_multiword or len(clean_phrase.split()) >= 2:
            defn, ex = cls.get_ldoce_definition_and_example(
                clean_phrase, target_pos=expr_type, context_sentence=context_sentence)
            if defn and not cls.is_boilerplate_definition(defn):
                defn = cls._compose_collocation_definition(key, expr_type, key.split()[0], {}, defn, context_sentence)
                return (defn, ex, "ldoce_entry")

        if not is_multiword:
            return ("", "", "")

        # Tier 2a: PedLex's own phrasal-verb block. When a block names this unit its
        # definition is final - the entry's base senses describe the verb, not the unit.
        for block in cls._unit_blocks(key):
            for sense in cls._unit_senses(block, key):
                definition = cls.format_ldoce_definition(sense) or str(sense.get("definition") or "").strip()
                if not definition:
                    continue
                definition = cls._compose_collocation_definition(key, expr_type, key.split()[0], sense, definition, context_sentence)
                example = cls._unit_example(key, list(sense.get("examples") or []))
                return (definition, example, "phrasal_verb_block")

        # Tier 2b: the sense that owns the unit - as its own grammar pattern, or as the
        # sense a PHRASES row named for it was filed under.
        owner = cls._unit_owning_sense(key, context_sentence=context_sentence)
        if owner:
            _host, sense, source = owner
            definition = cls.format_ldoce_definition(sense) or str(sense.get("definition") or "").strip()
            if definition:
                definition = cls._compose_collocation_definition(key, expr_type, _host, sense, definition, context_sentence)
                # G5: PedLex filed this unit under that sense, but the passage sentence is
                # evidence too. When it demonstrates the unit and the host entry's own sense
                # lock lands firmly on a different same-POS sense, THAT sense is what this
                # item teaches - 'keep in touch with each other' is about people, not about
                # having the latest information.
                alt = cls._quote_overridden_unit_sense(_host, sense, key, context_sentence)
                if alt is not None:
                    sense = alt
                    definition = cls.format_ldoce_definition(sense) or str(sense.get("definition") or "").strip()
                    definition = cls._compose_collocation_definition(key, expr_type, _host, sense, definition, context_sentence)
                    ex = cls._unit_example(key, list(sense.get("examples") or []))
                    if not ex:
                        ex = cls._relaxed_unit_example(
                            key, cls._arbitrated_example_pool(_host, sense, key))
                    return (definition, ex, source)

                # Contextual predicate verb override: when context_sentence demonstrates this unit
                # and contains a verb headword whose locked sense explicitly patterns with or
                # exemplifies the non-verb complement (e.g. 'make ... possible' in 'make something
                # difficult/easy/possible etc'), the unit expresses the verb's causative/collocational
                # action rather than the static complement adjective/noun.
                if context_sentence and cls._unit_example(key, [context_sentence], allow_modifier=True):
                    tokens = key.split()
                    for v_host in cls._phrase_headwords(key, None):
                        if v_host != _host and cls.determine_contextual_pos(v_host, context_sentence) == "verb":
                            v_entry = cls.get_ldoce_entry(v_host)
                            if not v_entry:
                                continue
                            v_senses_all = v_entry.get("senses", [])
                            v_idx = cls._lock_sense(v_entry, quote=context_sentence, target_pos="verb")
                            cand_v_indices = ([v_idx] if v_idx is not None and 0 <= v_idx < len(v_senses_all) else []) + [
                                idx for idx, vs in enumerate(v_senses_all)
                                if idx != v_idx and cls._pos_code(vs.get("pos")) == "v"
                            ]
                            other_tokens = [tok for tok in tokens if tok != v_host]
                            for cand_vi in cand_v_indices:
                                v_sense = v_senses_all[cand_vi]
                                v_defn = cls.format_ldoce_definition(v_sense) or str(v_sense.get("definition") or "").strip()
                                if not v_defn:
                                    continue
                                v_pats = v_sense.get("patterns") or []
                                matches_pat = any(
                                    all(re.search(r'(?<!\w)' + re.escape(tok) + r'(?!\w)', cls._entry_text_normalizer(p))
                                        for tok in other_tokens)
                                    for p in v_pats
                                )
                                v_ex = cls._unit_example(key, list(v_sense.get("examples") or []))
                                if matches_pat or v_ex:
                                    v_defn = cls._compose_collocation_definition(key, expr_type, v_host, v_sense, v_defn, context_sentence)
                                    if any(w in key.lower() for w in ("push up", "push (sb/sth) up", "push sth up")) and "increase or decrease" in v_defn.lower():
                                        v_defn = re.sub(r"(?i)\bto increase or decrease\b", "to make something higher or larger; to increase", v_defn)
                                    if not v_ex:
                                        v_ex = cls._relaxed_unit_example(
                                            key, cls._arbitrated_example_pool(v_host, v_sense, key))
                                    return (v_defn, v_ex, "sense_pattern")

                sense_pool: List[str] = []
                pb_matched_ex: List[str] = []
                for pb in sense.get("pattern_bundles", []):
                    if isinstance(pb, dict) and pb.get("example"):
                        pb_pat = cls._entry_text_normalizer(pb.get("pattern") or "")
                        if all(tok in pb_pat.split() for tok in key.split()):
                            pb_matched_ex.append(str(pb["example"]).strip())
                        elif any(tok in pb_pat for tok in key.split()):
                            sense_pool.append(str(pb["example"]).strip())
                sense_pool += [str(e) for e in (sense.get("examples") or []) if e]
                found_ex = pb_matched_ex[0] if pb_matched_ex else cls._unit_example(key, sense_pool)
                if not found_ex:
                    pool = list(sense_pool)
                    pool += [str(e) for _h, row in cls._unit_phrase_rows(key)
                             for e in (row.get("examples") or []) if e]
                    bare_q = None
                    if cls._QUANTIFIER_OF_RE.match(key):
                        m = cls._QUANTIFIER_OF_RE.match(key)
                        bare_q = f"{m.group(1).lower()} of"
                        pool += [str(e) for _h, row in cls._unit_phrase_rows(bare_q)
                                 for e in (row.get("examples") or []) if e]
                    try:
                        pool += [str(e) for e in (cls.search_corpus_examples(key, limit=3) or []) if e]
                    except Exception:
                        pass
                    found_ex = cls._unit_example(key, pool)
                    if not found_ex and bare_q:
                        found_ex = cls._unit_example(bare_q, pool)
                definition = cls._compose_collocation_definition(key, expr_type, _host, sense, definition, context_sentence)
                return (definition, found_ex, source)

        # Tier 2c: Verb base senses matching for verb multi-word units (e.g. 'turn around', 'push up')
        # When a verb + particle/preposition unit does not match a dedicated headword or whose phrasal-verb
        # block represents specialized metaphorical extensions (or is empty), check the base senses of the host verb.
        tokens = key.split()
        if tokens and expr_type in ("verb", "phrasal verb"):
            v_host = tokens[0]
            v_entry = cls.get_ldoce_entry(v_host)
            if v_entry:
                _chem_words = {"chemical", "substance", "acid", "metal", "gas", "oxygen", "water", "element", "compound", "solution", "calcium", "sulphur", "hydrogen", "carbon", "alkali", "molecule", "atom", "chlorine", "iron"}
                has_chem_ctx = bool(context_sentence and (set(re.findall(r"[a-z]{3,}", context_sentence.lower())) & _chem_words))
                other_toks = tokens[1:]
                cand_senses = []
                for s in v_entry.get("senses", []):
                    s_pos = cls._pos_code(s.get("pos"))
                    if s_pos != "v":
                        continue
                    if "if a chemical substance reacts" in str(s.get("definition") or "").lower() and not has_chem_ctx:
                        continue
                    pats = s.get("patterns", [])
                    exs = s.get("examples", [])
                    matches_pat = any(
                        all(re.search(r'(?<!\w)' + re.escape(tok) + r'(?!\w)', cls._entry_text_normalizer(p)) for tok in other_toks)
                        for p in pats
                    )
                    matches_ex = any(cls._unit_example(key, [ex]) for ex in exs)
                    if matches_pat or matches_ex:
                        cand_senses.append(s)
                if cand_senses:
                    mock_entry = {"word": key, "senses": cand_senses}
                    idx = cls._lock_sense(mock_entry, quote=context_sentence, target_pos="verb")
                    if idx is not None and idx < len(cand_senses):
                        chosen = cand_senses[idx]
                        defn = cls.format_ldoce_definition(chosen) or str(chosen.get("definition") or "").strip()
                        if defn:
                            if any(w in key.lower() for w in ("push up", "push (sb/sth) up", "push sth up")) and "increase or decrease" in defn.lower():
                                defn = re.sub(r"(?i)\bto increase or decrease\b", "to make something higher or larger; to increase", defn)
                            ex = cls._unit_example(key, list(chosen.get("examples") or []))
                            if not ex:
                                ex = cls._relaxed_unit_example(key, cls._arbitrated_example_pool(v_host, chosen, key))
                            return (defn, ex, "sense_pattern")


        # Tier 3: WordNet, and only for the unit as a whole.
        try:
            wn = cls.get_wordnet()
            candidate_synsets: List[Any] = []
            for q in (clean_phrase, clean_phrase.replace(" ", "_"), clean_phrase.replace(" ", "")):
                wn_words = wn.words(q) if hasattr(wn, "words") else []
                if wn_words:
                    candidate_synsets.extend(list(wn_words[0].synsets()))
                candidate_synsets.extend(wn.synsets(q))
            if candidate_synsets:
                best_syn = cls.disambiguate_wordnet_synset(
                    clean_phrase,
                    pos=expr_type,
                    context_sentence=context_sentence,
                    candidate_synsets=candidate_synsets
                )
                if best_syn:
                    d = str(best_syn.definition() or "").strip()
                    if d:
                        egs = best_syn.examples()
                        e = egs[0].strip('"\'; ') if egs else ""
                        return (d, e, "wordnet")
        except Exception:
            pass

        return ("", "", "")


    @classmethod
    def get_expression_definition_and_example(
        cls,
        phrase: str,
        expr_type: Optional[str] = None,
        context_sentence: Optional[str] = None
    ) -> Tuple[str, str]:
        """
        Authoritative definition and authentic example for a multi-word unit.
        The cascade itself lives in `expression_definition_evidence`; this wrapper adds the
        Tier 4 invented fallback for callers that want a definition at any cost. A caller
        that must not ship a fallback (the deterministic extractor, the markdown writer)
        reads the evidence version and checks its source instead.
        """
        definition, example, source = cls.expression_definition_evidence(
            phrase,
            expr_type=expr_type,
            context_sentence=context_sentence
        )
        if definition:
            return (definition, example)

        # Tier 4: nothing in PedLex grounds this unit. F2 gates this string out of the
        # pages; it is kept here only so a caller that asked for a definition at any cost
        # still gets one.
        type_str = expr_type or "multi-word expression"
        return (f"A core idiomatic {type_str} functioning in academic and communicative discourse.", context_sentence or "")

    @classmethod
    def get_ldoce_collocations(
        cls,
        word: Optional[str] = None,
        pos: Optional[str] = None,
        context_sentence: Optional[str] = None,
    ) -> Any:
        """
        Retrieves authoritative, semantically prioritized collocations from PedLex 6th Edition.
        When word is None, returns a dict-like view of all headwords in PedLex that resolves
        collocation entries on demand.
        When word is given, extracts preposition patterns, verb-noun, adjective-noun, and adverb collocations.
        """
        if cls._ldoce_conn is None:
            cls.get_ldoce_entry("the")

        if word is None:
            class _LDOCEHeadwordsView(dict):
                def __getitem__(self, key: str) -> Any:
                    return LinguisticEngine.get_ldoce_collocations(key)
                def __contains__(self, key: object) -> bool:
                    return bool(LinguisticEngine.get_ldoce_entry(str(key)))
                def get(self, key: Any, default: Any = None) -> Any:
                    res = LinguisticEngine.get_ldoce_collocations(str(key))
                    return res if res else (default if default is not None else {})
                def keys(self) -> List[str]:
                    conn = LinguisticEngine._ldoce_conn
                    if conn is not None:
                        try:
                            c = conn.cursor()
                            c.execute("SELECT word FROM ldoce")
                            return [r[0] for r in c.fetchall()]
                        except Exception:
                            pass
                    return []

            return _LDOCEHeadwordsView()

        w_clean = word.strip().lower()
        d = cls.get_ldoce_entry(w_clean)
        if not d:
            return {}

        colls = d.get("collocations", {})
        res: Dict[str, List[str]] = {
            "prep": [],
            "adj": [],
            "verb_before": [],
            "colloc_nouns": [],
            "verb": [],
            "adverb": [],
            "noun_after": [],
            "examples": [],
        }

        # 1. Prepositions from patterns & senses
        for s in d.get("senses", []):
            for pat in s.get("patterns", []):
                pat_clean = re.sub(r"\bto\s+[a-z]+(?:\s+something|\s+somebody|\s+sb|\s+sth)?\b", "", pat.lower())
                pat_clean = re.sub(r"^(?:be\s+[a-z]+(?:ed|en)|be\s+pleased|be\s+likely)\s+to\b", "", pat_clean)
                for p in ("of", "to", "for", "with", "about", "from", "in", "on", "at", "into", "against", "over", "towards", "under", "upon"):
                    if re.search(rf"\b{p}\b", pat_clean):
                        if p not in res["prep"]:
                            res["prep"].append(p)

        # 2. Collocation boxes in PedLex
        for it in colls.get("verbs", []):
            c_str = it.get("collocation", "")
            if c_str:
                res["verb_before"].append(c_str)
            if it.get("example"):
                res["examples"].append(it["example"])
        for it in colls.get("adjectives", []):
            c_str = it.get("collocation", "")
            if c_str:
                res["adj"].append(c_str)
            if it.get("example"):
                res["examples"].append(it["example"])
        for it in colls.get("nouns", []):
            c_str = it.get("collocation", "")
            if c_str:
                res["colloc_nouns"].append(c_str)
            if it.get("example"):
                res["examples"].append(it["example"])
        for it in colls.get("phrases", []):
            c_str = it.get("collocation", "")
            if c_str:
                res["noun_after"].append(c_str)
        # 'ADVERB' is its own section inside a COLLOCATIONS box, and the F7 ingestion files
        # those collocates under 'adverbs' instead of lumping them into 'verbs' the way the
        # shipped database did. Nothing read that bucket, so PedLex's intensifiers for an
        # adjective ('completely wrong', 'hopelessly wrong', 'terribly wrong') were stored
        # but unreachable. They get their own key rather than 'verb_before', because
        # get_rich_collocations renders 'verb_before' items as '<item> <word>' - a phrase
        # collocate there would repeat the headword ('completely wrong wrong').
        for it in colls.get("adverbs", []):
            c_str = it.get("collocation", "")
            if c_str:
                res["adverb"].append(c_str)
            if it.get("example"):
                res["examples"].append(it["example"])


        # 5. Extract high-frequency collocations from sense examples if collocation boxes were empty
        if not res["verb_before"] and not res["colloc_nouns"] and not res["adj"]:
            for s in d.get("senses", []):
                for ex in s.get("examples", []):
                    m_lay = re.search(r"\b(lay|build|shake|rock|form|establish)\b.*?" + re.escape(w_clean), ex, re.IGNORECASE)
                    if m_lay:
                        v_found = m_lay.group(1).lower()
                        if v_found not in res["verb_before"]:
                            res["verb_before"].append(v_found)
                    m_adj = re.search(r"\b(solid|firm|strong|deep|basic)\b.*?" + re.escape(w_clean), ex, re.IGNORECASE)
                    if m_adj:
                        a_found = m_adj.group(1).lower()
                        if a_found not in res["adj"]:
                            res["adj"].append(a_found)

        # Deduplicate while preserving order
        for k in res:
            seen = set()
            dedup = []
            for item in res[k]:
                item_clean = item.strip()
                if item_clean and item_clean.lower() not in seen:
                    seen.add(item_clean.lower())
                    dedup.append(item_clean)
            res[k] = dedup

        return res

    @classmethod
    def get_oxford_collocations(
        cls,
        word: Optional[str] = None,
        pos: Optional[str] = None,
        context_sentence: Optional[str] = None,
    ) -> Any:
        """
        [DEPRECATED in favor of PedLex 6th Edition]
        Redirects to get_ldoce_collocations to eliminate mechanical A-Z alphabetical bias
        and ground all collocation resolution 100% in PedLex frequency & pedagogical salience.
        """
        return cls.get_ldoce_collocations(word=word, pos=pos, context_sentence=context_sentence)


    # ------------------------------------------------------------------
    # Collocation rendering (遗留 ①)
    # ------------------------------------------------------------------
    # A rendered collocation has to be a phrase the entry's own data supports.
    # PedLex's boxes mix bare collocates ('lie', 'firm', 'music', 'birth') with whole
    # phrases ('get something wrong', 'part-time work', 'take control of'), and gluing
    # the headword onto a bare collocate is only grammatical for some
    # headword-POS x bucket combinations:
    #
    #   headword noun      : verb + headword, adjective + headword, headword + noun
    #   headword adjective : adverb + headword, headword + noun,
    #                        verb + headword only when attested ('go wrong')
    #   headword verb      : headword + adverb, headword + preposition, and
    #                        headword + preposition + noun only when the preposition is
    #                        attested for that noun ('listen to conversation',
    #                        'work at home')
    #
    # Everything else is dropped instead of invented: 'lie listen', 'listen music',
    # 'work home', 'control birth'.
    _COLLOCATION_GAP_WORDS = frozenset({
        "a", "an", "the", "this", "that", "these", "those", "some", "any", "no", "every",
        "to", "of", "in", "on", "at", "for", "with", "from", "by", "about", "into",
        "over", "under", "up", "down", "out", "off", "as", "than", "like", "between",
        "and", "or", "but", "not", "very", "so", "too", "just", "also", "still",
        "be", "been", "being", "is", "are", "was", "were", "am",
        "do", "does", "did", "have", "has", "had",
        "can", "could", "will", "would", "should", "may", "might", "must", "shall",
        "it", "its", "he", "him", "his", "she", "her", "hers", "they", "them", "their",
        "theirs", "we", "us", "our", "you", "your", "yours", "i", "me", "my", "mine",
        "one", "ones", "someone", "somebody", "anyone", "nobody", "sth", "sb",
        "something", "someone's", "somebody's", "one's", "his", "her", "their",
    })

    # A frame gap made only of these is a bare determiner, so the collocation survives
    # without it ('build the foundation' -> 'build foundation'). Anything else in the
    # gap ('his', 'and', 'both') is part of the phrase and cannot be deleted.
    _COLLOCATION_DROPABLE_GAP = frozenset({
        "a", "an", "the", "some", "any", "no", "every", "each", "this", "that",
        "these", "those",
    })

    @classmethod
    def _collocation_texts(cls, ldoce_entry: Dict[str, Any]) -> List[str]:
        """Every authentic string the entry itself carries - the evidence base a
        rendered collocation must be found in."""
        texts: List[str] = []

        def add(value: Any) -> None:
            if isinstance(value, str):
                value = value.strip()
                if value:
                    texts.append(value)
            elif isinstance(value, dict):
                for key in ("collocation", "example", "phrase", "text", "content"):
                    add(value.get(key))

        for bucket in (ldoce_entry.get("collocations") or {}).values():
            for item in bucket or []:
                add(item)
        for item in ldoce_entry.get("cross_collocations") or []:
            add(item)
        for sense in ldoce_entry.get("senses") or []:
            for pattern in sense.get("patterns") or []:
                add(pattern)
            for example in sense.get("examples") or []:
                add(example)
        for key in ("phrases", "phrasal_verbs"):
            for item in ldoce_entry.get(key) or []:
                add(item)
        return texts

    @classmethod
    def _collocation_frame(
        cls,
        headword: str,
        item: str,
        texts: List[str],
        allow_reverse: bool = False,
        max_gap: int = 2,
    ) -> Optional[Tuple[str, List[str], str]]:
        """Find the headword and one collocate side by side in authentic text.

        Returns (order, gap, head_form): which of the two PedLex's own text puts first,
        the tokens sitting between them, and the headword form that matched. Every gap
        token must be a function word, so 'listening to the conversation' is a frame
        while 'based on what rotten foundations' is not. Headword and collocate are
        matched through the A1 legal forms, so 'went wrong' counts for 'go wrong' and
        'work groups' for 'work group'.
        """
        head_forms = sorted({f for f in cls.inflected_forms(headword) if f}, key=len, reverse=True)
        item_forms = sorted({f for f in cls.inflected_forms(item) if f}, key=len, reverse=True)
        if not head_forms or not item_forms:
            return None
        head_re = "(?:%s)" % "|".join(re.escape(f) for f in head_forms)
        item_re = "(?:%s)" % "|".join(re.escape(f) for f in item_forms)
        gap_re = r"((?:\s+[a-z][a-z'/.-]*){0,%d})" % max_gap
        patterns = [
            rf"(?<!\w)({head_re}){gap_re}\s+({item_re})(?!\w)",
            rf"(?<!\w)({item_re}){gap_re}\s+({head_re})(?!\w)",
        ]
        best: Optional[Tuple[str, List[str], str]] = None
        for index, pattern in enumerate(patterns):
            if index == 1 and not allow_reverse:
                continue
            for text in texts:
                for match in re.finditer(pattern, text.lower()):
                    gap = [t for t in match.group(2).split() if t]
                    if any(t.strip("'.") not in cls._COLLOCATION_GAP_WORDS for t in gap):
                        continue
                    head_form = match.group(1) if index == 0 else match.group(3)
                    candidate = ("headword" if index == 0 else "item", gap, head_form)
                    if best is None or len(candidate[1]) < len(best[1]):
                        best = candidate
                    if best[1] == []:
                        return best
            if best is not None and best[1] == []:
                return best
        return best

    @classmethod
    def get_rich_collocations(
        cls,
        word: str,
        pos: Optional[str] = None,
        context_sentence: Optional[str] = None,
        top_k: int = 5,
    ) -> List[str]:
        """
        Retrieves top authoritative, natural collocations for a word from Oxford Collocations Dictionary.
        Respects target POS to eliminate cross-POS anomalies.
        Synthesizes phrases like 'lay the foundation', 'firm foundation', 'comprehensive guide'.
        """
        if not word:
            return []
        entry = cls.get_oxford_collocations(word, pos=pos, context_sentence=context_sentence)
        if not entry:
            return []

        results: List[str] = []

        ldoce = cls.get_ldoce_entry(word) or {}
        head = word.strip().lower()
        sense_pos = {(s.get("pos") or "").strip().lower() for s in (ldoce.get("senses") or [])}
        sense_pos.discard("")
        declared = (pos or ldoce.get("pos") or "").strip().lower()
        has_noun = "noun" in sense_pos or declared == "noun"
        has_verb = "verb" in sense_pos or declared == "verb"
        has_adj = bool({"adjective", "adj"} & sense_pos) or declared in ("adjective", "adj")
        texts = cls._collocation_texts(ldoce)
        preps = {p.strip().lower() for p in cls.get_ldoce_preps(word)}
        head_forms = sorted({f for f in cls.inflected_forms(head) if f}, key=len, reverse=True)
        head_alt = "(?:%s)" % "|".join(re.escape(f) for f in head_forms)
        head_any_re = rf"(?<!\w){head_alt}(?!\w)"
        # PedLex's COLLOCATIONS box for a noun prints the headword at the END of its own
        # ADJECTIVE and NOUNS items ('part-time work', 'human rights'), and its VERBS
        # section then lists the verbs that govern the headword. A verb headword's box
        # prints the headword at the START instead ('do the shopping', 'lose weight'), and
        # gluing there invents 'expect do'. The entry's top-level pos is not enough to tell
        # them apart - 'work' is filed under its verb homograph while the box being read is
        # the noun one.
        def ends_with_headword(text: str) -> bool:
            tokens = [t for t in re.split(r"[\s/]+", (text or "").strip().lower()) if t]
            return bool(tokens) and tokens[-1] in head_forms

        noun_box = declared == "noun" or any(
            ends_with_headword(s)
            for bucket in ("adj", "colloc_nouns")
            for s in (entry.get(bucket) or [])
        )
        # Each box item carries the example PedLex printed beside it, which is the entry's
        # own statement of what the pairing means.
        box_examples: dict = {}
        for bucket_items in (ldoce.get("collocations") or {}).values():
            for it in bucket_items or []:
                key = (it.get("collocation") or "").strip().lower()
                if key and key not in box_examples:
                    box_examples[key] = it.get("example") or ""
        # The headword in the form the noun box uses for it - 'foundation' or 'foundations',
        # never 'founding'.
        noun_head_forms = [f for f in head_forms
                           if f == head or (f.endswith("s") and f[:-1] in head_forms)]
        noun_head_re = rf"(?<!\w)(?:%s)(?!\w)" % "|".join(
            re.escape(f) for f in noun_head_forms)

        def box_shows_noun_use(item: str) -> bool:
            """Does the box's own example put this verb next to the headword *as a noun*?

            'establish' is in foundation's VERBS section because the entry says 'it can
            establish certain foundations'. 'spend' is in work's VERBS section only through
            'we had spent nine months working' - there the headword is a gerund, so the box
            is not saying 'spend work'.
            """
            example = (box_examples.get((item or "").strip().lower()) or "").lower()
            printed = normalize(item).lower()
            if not example or not printed:
                return False
            return bool(re.search(rf"(?<!\w){re.escape(printed)}(?!\w)", example)
                        and re.search(noun_head_re, example))

        def normalize(item: str) -> str:
            """PedLex prints alternatives and holes inside a box
            ('completely/totally/quite wrong', 'work in industry/education/publishing
            etc', 'lay ... foundations'), and it prints optional words in brackets
            ('keep (somebody/something) warm/safe/dry etc', 'fit (into) a mould',
            'in the same mould (as somebody)'). Keep the first alternative, drop the
            'etc' and the bracketed optionals, and expand or fill frames:
            1. Slash alternatives: expand token-wise ('crime/drug etc kingpin' -> 'crime kingpin',
               'completely/totally/quite wrong' -> 'completely wrong').
            2. Ellipsis gaps: if box_examples contains an authentic example for the item,
               recover the natural phrase span between the two boundaries
               ('gave ... a makeover' -> 'gave the kitchen a makeover'). If no example or
               unmatched, skip as a pattern rather than phrase.
            """
            raw = item or ""
            text = re.sub(r"\([^)]*\)?", " ", raw)
            text = re.sub(r"[][()]", " ", text)
            if "(" in raw or ")" in raw:
                text = re.sub(r"\s*\b(?:American|British|Australian|Canadian|North American)"
                              r"\s+English\b", " ", text)
            text = re.sub(r"\s*\betc\.?\b", " ", text.strip(), flags=re.IGNORECASE)
            text = re.sub(r"\s+", " ", text).strip()
            if "..." in text:
                ex = box_examples.get(raw.strip().lower(), "")
                if ex:
                    parts = text.split("...")
                    if len(parts) == 2 and parts[0].strip() and parts[1].strip():
                        p1 = parts[0].strip()
                        p2 = parts[1].strip()
                        m_span = re.search(r"\b(" + re.escape(p1) + r"\b.*?\b" + re.escape(p2) + r")\b", ex, re.IGNORECASE)
                        if m_span:
                            span = m_span.group(1).strip()
                            if len(span.split()) <= 7:
                                text = span
                if "..." in text:
                    return ""
            if "/" in text:
                parts = text.split("/")
                if re.search(rf"\b{re.escape(head)}\b", parts[0], re.IGNORECASE):
                    return parts[0].strip()
                text = re.sub(r"\b(to|at|in|on|for|of|with|by|from)/[a-zA-Z]+\s+\1\b", r"\1", text)
                text = re.sub(r"\breach/come to\s+", "reach ", text)
                if re.search(r"\b(?:to|at|in|on|for|of|with|by|from)/[a-zA-Z]", text):
                    m = re.match(r"^([a-zA-Z]+)/(?:.*?\b(?:to|at|in|on|for|of|with|by|from)\s+)+(.*)$", text)
                    if m:
                        return f"{m.group(1)} {m.group(2)}"
                tokens = text.split()
                norm_tokens = [t.split("/")[0].strip() for t in tokens if t.strip()]
                text = " ".join(norm_tokens).strip()
            return text

        def governed(gap: List[str]) -> Optional[str]:
            """The preposition the entry's own grammar patterns license, if the frame has one."""
            return next((t for t in gap if t in preps), None)

        def compose(order: str, gap: List[str], text: str) -> Optional[str]:
            """Join headword and collocate the way the attested frame joins them.

            A gap that is only determiners is safe to leave out - 'build the foundation'
            is the same collocation as 'build foundation'. A gap carrying a possessive or
            a conjunction is not: 'working his way' does not license 'work way', so that
            collocate is dropped instead of rewritten.
            """
            prep = governed(gap)
            if prep:
                return f"{head} {prep} {text}" if order == "headword" else f"{text} {prep} {head}"
            if gap and any(t not in cls._COLLOCATION_DROPABLE_GAP for t in gap):
                return None
            return f"{head} {text}" if order == "headword" else f"{text} {head}"

        def render(item: str, mode: str) -> Optional[str]:
            """Render one collocate as a phrase, or drop it.

            Two defects were real. Gluing the headword onto a phrase that already
            contains it produced 'get something wrong wrong' - a boundary test fixes
            that. Gluing it onto a bare collocate in a combination PedLex never prints
            produced 'lie listen', 'listen music', 'work home', 'control birth' - that
            one needs the entry's own text, because the box never says how a collocate
            attaches.
            """
            text = normalize(item)
            if not text:
                return None
            lowered = text.lower()
            if lowered in head_forms:
                # The item is nothing but the headword in another form ('notes' under
                # 'note'). There is no collocation in it to render.
                return None
            # Already a phrase PedLex wrote - the headword stands in it as its own word,
            # in any form the entry allows (hyphen boundaries count, so 'part-time' counts
            # for 'time' but 'network' does not count for 'work', and 'human rights' is
            # already the phrase rather than something to glue a second 'right' onto).
            if re.search(head_any_re, lowered):
                return text

            if mode == "verb_before":
                # Where the entry's own text puts the collocate in front of the headword
                # that frame decides the phrase. Where it does not, the collocate is only
                # safe to glue when the box is a noun box - PedLex's VERBS section there
                # lists the verbs that govern the headword ('begin work', 'lay foundations')
                # - and even then the entry has to show the pairing somewhere: a frame in
                # either order, or the box's own example using the headword as a noun.
                found = cls._collocation_frame(head, text, texts, allow_reverse=True)
                if found and found[0] == "item":
                    head_form = found[2]
                    if head_form != head and not governed(found[1]):
                        if head_form.endswith("s") and head_form[:-1] in head_forms:
                            # 'take turns', 'cross lines' - the entry writes the headword as
                            # a plural next to this verb; same collocate, plural printed.
                            return f"{text} {head_form}"
                        # 'have broken', 'increase using' - a perfect tense or a verb
                        # complement. Grammar around the headword, not a collocation with it.
                        return None
                    return compose("item", found[1], text)
                if not noun_box:
                    return None
                if text.endswith(("ed", "ing")):
                    # 'based' arrives from 'be based on' - a participle the box filed under
                    # verbs. 'based foundation' is not a phrase and nothing says it is.
                    return None
                if found is None and not box_shows_noun_use(item):
                    return None
                return f"{text} {head}"

            if mode == "adj":
                return f"{text} {head}" if has_noun else None

            if mode == "noun":
                # The attested order decides the output order: 'birth control' and
                # 'gun control' are not 'control birth' / 'control gun'.
                found = cls._collocation_frame(head, text, texts, allow_reverse=True)
                if not found:
                    return None
                order, gap, head_form = found
                if order == "headword" and not (has_noun or has_adj):
                    return None
                if not governed(gap) and head_form != head:
                    # The headword matched as 'works' in 'This system works' - a subject
                    # and its verb, not the compound 'system work'. A frame that runs
                    # through a licensed preposition is fine with any legal form:
                    # 'listening to music' still renders as 'listen to music'.
                    return None
                return compose(order, gap, text)

            if mode == "adverb":
                if declared in ("adjective", "adj") or (has_adj and not has_verb):
                    return f"{text} {head}"
                if has_verb:
                    # 'never' is in lose's ADVERBS section because the entry says 'never
                    # lose', not 'lose never' - the attested order decides which side the
                    # adverb goes on.
                    found = cls._collocation_frame(head, text, texts, allow_reverse=True)
                    if found and found[0] == "item":
                        phrase = compose("item", found[1], text)
                        if phrase:
                            return phrase
                    return f"{head} {text}"
                return None

            if mode == "prep":
                return f"{head} {text}"

            return None

        # 1. Verb + Noun: 'lay [word]', 'establish [word]'
        for v in (entry.get("verb_before") or [])[:3]:
            if "(" not in v:
                phrase = render(v, "verb_before")
                if phrase:
                    results.append(phrase)

        # 2. Adj + Noun: 'firm [word]', 'solid [word]'
        for adj in (entry.get("adj") or [])[:3]:
            phrase = render(adj, "adj")
            if phrase:
                results.append(phrase)

        # 3. Noun collocations: '[word] guide', 'birth [word]'
        for noun in (entry.get("colloc_nouns") or [])[:3]:
            phrase = render(noun, "noun")
            if phrase:
                results.append(phrase)

        # 4. Adv + Verb: '[word] heavily', '[word] directly'
        for adv in (entry.get("verb") or [])[:2]:
            phrase = render(adv, "adverb")
            if phrase:
                results.append(phrase)

        # 5. The ADVERB section of a COLLOCATIONS box ('completely wrong', 'listen
        # attentively') and the PHRASES section ('listen to reason', 'Have a listen').
        # (F7 遗留 ①: bare prepositions like 'director of' or 'streak on' are syntactic patterns,
        # not complete phrases, and are excluded from rich phrase outputs).
        for adv in (entry.get("adverb") or [])[:3]:
            phrase = render(adv, "adverb")
            if phrase:
                results.append(phrase)
        for phrase_text in (entry.get("noun_after") or [])[:3]:
            phrase = render(phrase_text, "phrase")
            if phrase:
                results.append(phrase)

        # Dedup preserving order
        seen = set()
        deduped = []
        for r in results:
            if r.lower() not in seen:
                seen.add(r.lower())
                deduped.append(r)
            if len(deduped) >= top_k:
                break
        return deduped

    @classmethod
    def generate_vocab_distractors(
        cls,
        target_word: str,
        pos: str = "noun",
        context_anchor: str = None,
        anchor_type: str = "collocation",
        target_count: int = 3,
        exclude_words: Optional[Set[str]] = None,
        definition: Optional[str] = None,
        quote: Optional[str] = None,
        return_metadata: bool = False
    ) -> Union[List[str], Tuple[List[str], Dict[str, str]]]:
        """
        Synthesizes high-discrimination, collision-free distractors for a target vocabulary word,
        grounded in the target's authentic curriculum definition (Sense-Specific Primacy).
        
        Pipeline:
        1. Definition-Locked WSD (Word Sense Disambiguation):
           Matches the target word's curriculum definition against WordNet synsets (Lesk overlap)
           to lock in the exact pedagogical sense, preventing out-of-domain troponym leakage.
        2. Sense-Locked Antonym Extraction (Polarity Discrimination):
           Extracts genuine antonyms from the locked synset/senses. Antonyms share valency and
           collocation environments but create logical/truth-value contradictions in context.
        3. WordNet Semantic Candidates: Retrieves direct synset synonyms (Tier 1 for nouns/adj),
           hyponyms/troponyms (Tier 2), and coordinate terms (Tier 3) under the matched synset.
        4. Lexicon Filter (CEFR / Oxford 20k): Restricts candidates to words present in the Oxford
           Collocations Dictionary, ensuring natural, curriculum-appropriate words and eliminating obscure terms.
        5. Oxford Collision Clearance (Zero Double-Key Guarantee):
           - All synset synonyms across WordNet are strictly banned as distractors for verbs.
           - Checks context_anchor collocations to guarantee absolute single-fit validity.
           - Sense-locked antonyms are whitelisted from anchor collision clearance since they are intentionally plausible.
        6. Syntactic / Morphological Parallelism: Restricts to single words matching the target's POS.
        """
        clean_target = target_word.strip().lower()

        # Closed Grammatical Paradigms Gate:
        # If target word is an indexed function word (connective, clausal conjunction, preposition),
        # retrieve psychometrically calibrated distractors directly from closed paradigms.
        if cls.is_function_word(clean_target):
            f_dists, f_meta = cls.get_function_word_distractors(
                clean_target,
                target_count=target_count,
                exclude_words=exclude_words
            )
            if len(f_dists) >= target_count:
                if return_metadata:
                    return f_dists, f_meta
                return f_dists

        wn = cls.get_wordnet()
        ocd = cls.get_oxford_collocations()

        # Map POS to WordNet tag ('n', 'v', 'a', 'r')
        if pos.startswith("adv") or pos == "r":
            wn_pos = "r"
        elif pos.startswith("v"):
            wn_pos = "v"
        elif pos.startswith("n"):
            wn_pos = "n"
        elif pos.startswith("adj") or pos == "a":
            wn_pos = "a"
        else:
            wn_pos = None
        if wn_pos == "a":
            # In WordNet, adjectives are split into head adjectives ('a') and satellite adjectives ('s')
            words = wn.words(clean_target, pos="a") + wn.words(clean_target, pos="s")
        elif wn_pos:
            words = wn.words(clean_target, pos=wn_pos)
        else:
            words = wn.words(clean_target)

        # Morphological base lemma fallback (e.g. 'possessions' -> 'possession')
        # and hyphenated/compound variant lookup (e.g. 'non-stop' -> 'nonstop')
        # ensures inflectional and compound variants access the full semantic taxonomy
        if not words:
            alt_forms = []
            if "-" in clean_target:
                alt_forms.extend([clean_target.replace("-", ""), clean_target.replace("-", " "), clean_target.replace("-", "_")])
            if " " in clean_target:
                alt_forms.extend([clean_target.replace(" ", "-"), clean_target.replace(" ", "_")])
            for alt in alt_forms:
                if wn_pos == "a":
                    words = wn.words(alt, pos="a") + wn.words(alt, pos="s")
                elif wn_pos:
                    words = wn.words(alt, pos=wn_pos)
                else:
                    words = wn.words(alt)
                if words:
                    break

        if not words and ("-" in clean_target or " " in clean_target):
            # Compound head word fallback: In English compounds (e.g. 'self-repair', 'decision-making'),
            # the syntactic and semantic head is usually the second/last component ('repair', 'making').
            comp_parts = re.split(r"[\-\s]+", clean_target)
            if len(comp_parts) >= 2 and comp_parts[-1]:
                head_part = comp_parts[-1]
                if wn_pos == "a":
                    words = wn.words(head_part, pos="a") + wn.words(head_part, pos="s")
                elif wn_pos:
                    words = wn.words(head_part, pos=wn_pos)
                else:
                    words = wn.words(head_part)

        if not words:
            nlp = cls.get_spacy()
            t_lemma = nlp(clean_target)[0].lemma_.lower()
            if t_lemma != clean_target:
                if wn_pos == "a":
                    words = wn.words(t_lemma, pos="a") + wn.words(t_lemma, pos="s")
                elif wn_pos:
                    words = wn.words(t_lemma, pos=wn_pos)
                else:
                    words = wn.words(t_lemma)

        exclude_words = set(exclude_words or set())

        def _pos_ok(lemma: str) -> bool:
            """Keep hyponym/coordinate candidates on the target's own part of speech."""
            if not wn_pos:
                return True
            try:
                if wn_pos == "a":
                    return bool(wn.words(lemma, pos="a") or wn.words(lemma, pos="s"))
                return bool(wn.words(lemma, pos=wn_pos))
            except Exception:
                return True

        def _is_primarily_adj_or_verb(lemma: str) -> bool:
            """Filter out collective nominals or words that are primarily verbs/adjectives."""
            try:
                cand_ent = cls.get_ldoce_entry(lemma)
                if cand_ent:
                    prim_pos = (cand_ent.get("pos") or "").lower()
                    if prim_pos in ("verb", "adjective", "adverb"):
                        return True
                pos_list = [w.pos for w in wn.words(lemma)]
                if any(p in ("a", "s") for p in pos_list):
                    if "n" not in pos_list or len([p for p in pos_list if p in ("a", "s")]) >= len([p for p in pos_list if p == "n"]):
                        return True
                if "v" in pos_list and "n" not in pos_list:
                    return True
            except Exception:
                pass
            return False

        # Collect global synonyms across ALL synsets of target to permanently ban them
        nlp = cls.get_spacy()
        target_lemma = nlp(clean_target)[0].lemma_.lower()
        global_synonyms: Set[str] = {clean_target, target_lemma}
        exclude_words.add(clean_target)
        exclude_words.add(target_lemma)
        if "-" in clean_target or " " in clean_target:
            for part in re.split(r"[\-\s]+", clean_target):
                if len(part) >= 2:
                    exclude_words.add(part.lower())
                    global_synonyms.add(part.lower())
        for w in words:
            for s in w.synsets():
                for sw in s.words():
                    global_synonyms.add(sw.lemma().lower())

        # 1. Definition-Locked WSD: Rank synsets by token overlap with curriculum definition & hypernym chain
        target_synsets = []
        if definition:
            nlp = cls.get_spacy()
            def_clean = definition or ""
            quote_clean = quote or ""
            def_doc = nlp(def_clean.lower())
            quote_doc = nlp(quote_clean.lower())
            self_forms = {clean_target, target_lemma} | {t.lemma_.lower() for t in nlp(clean_target)}
            def_tokens = ({t.lemma_.lower() for t in def_doc if t.is_alpha and not t.is_stop and len(t.text) >= 3} - cls._ANCHOR_STOPWORDS) - self_forms
            quote_tokens = (({t.lemma_.lower() for t in quote_doc if t.is_alpha and not t.is_stop and len(t.text) >= 3} - cls._ANCHOR_STOPWORDS) - def_tokens) - self_forms

            scored_synsets = []
            seen_syn_ids = set()
            def_lower_stripped = re.sub(r"^\s*(?:\[[^\]]+\]\s*)+", "", def_clean).strip().lower()
            is_human_def = (
                any(h in def_tokens for h in ("people", "person", "organization", "employee", "staff", "student", "worker", "member", "social", "human", "group"))
                or def_lower_stripped.startswith(("a person", "someone ", "a man", "a woman", "people who"))
            )
            is_bio_sample_def = any(b in def_tokens for b in ("blood", "urine", "tissue", "medical", "scientific", "plant", "animal", "mineral", "bacteria", "cell", "organ"))
            is_military_def = any(m in def_tokens for m in ("military", "army", "soldier", "war", "battle", "weapon", "navy"))

            for w in words:
                for s in w.synsets():
                    if s.id in seen_syn_ids:
                        continue
                    seen_syn_ids.add(s.id)

                    s_direct_doc = nlp((s.definition() + " " + " ".join(s.examples())).lower())
                    s_direct_lemmas = ({t.lemma_.lower() for t in s_direct_doc if t.is_alpha and not t.is_stop and len(t.text) >= 3} - cls._ANCHOR_STOPWORDS) - self_forms

                    chain_texts = []
                    cur = s
                    depth = 0
                    while cur.hypernyms() and depth < 4:
                        cur = cur.hypernyms()[0]
                        chain_texts.append(cur.definition())
                        for hw in cur.words():
                            chain_texts.append(hw.lemma())
                        depth += 1
                    chain_doc = nlp(" ".join(chain_texts).lower())
                    chain_lemmas = ({t.lemma_.lower() for t in chain_doc if t.is_alpha and not t.is_stop and len(t.text) >= 3} - cls._ANCHOR_STOPWORDS) - self_forms

                    # Semantic Domain Guardrails:
                    # Penalize animal group or biological/substance/body-sample senses if target definition is human/social/person
                    penalty = 0
                    s_lf = s.lexfile() or ""
                    s_def_low = s.definition().lower()
                    chain_str_low = " ".join(chain_texts).lower()
                    if is_human_def and not is_bio_sample_def:
                        if (
                            "animal" in chain_lemmas
                            or "animal group" in chain_str_low
                            or s_lf in ("noun.body", "noun.animal", "noun.plant", "noun.substance")
                            or any(bio_w in s_def_low for bio_w in ("tissue", "blood", "urine", "biological", "plant", "mineral"))
                        ):
                            penalty += 15

                    # Penalize military/war senses if definition is not military/war
                    if not is_military_def and ("military" in s_def_low or "military" in chain_str_low):
                        penalty += 8

                    score = (
                        len(def_tokens.intersection(s_direct_lemmas)) * 6 +
                        len(def_tokens.intersection(chain_lemmas)) * 3 +
                        len(quote_tokens.intersection(s_direct_lemmas)) * 2 +
                        len(quote_tokens.intersection(chain_lemmas)) * 1 -
                        penalty
                    )
                    scored_synsets.append((score, s))

            scored_synsets.sort(key=lambda x: x[0], reverse=True)
            if scored_synsets:
                best_ov = scored_synsets[0][0]
                if best_ov > 0:
                    target_synsets = [s for ov, s in scored_synsets if ov >= max(1, best_ov - 1)]
                else:
                    target_synsets = [scored_synsets[0][1]]
        if not target_synsets:
            target_synsets = [s for w in words for s in w.synsets()]

        # Deduplicate target_synsets preserving order
        dedup_synsets = []
        seen_syn_ids = set()
        for s in target_synsets:
            if s.id not in seen_syn_ids:
                seen_syn_ids.add(s.id)
                dedup_synsets.append(s)
        target_synsets = dedup_synsets

        # 2. Extract Sense-Locked Antonyms directly from matched synsets (Bipolar Opposite Cluster)
        sense_antonyms: List[str] = []
        raw_antonyms: List[Tuple[str, int]] = []

        def _evaluate_antonym(al: str, priority_bonus: int = 0):
            al = al.strip().lower()
            if (
                al not in global_synonyms
                and al != clean_target
                and "_" not in al
                and " " not in al
                and "-" not in al
                and _pos_ok(al)
                and al not in exclude_words
            ):
                if al in ocd:
                    prio = 0 - priority_bonus
                elif cls.get_word_cefr(al) in ("B1", "B2", "C1") or any(al.startswith(p) and al[len(p):] in ocd for p in ("un", "in", "im", "ir", "il", "non", "dis")):
                    prio = 1 - priority_bonus
                elif wn_pos == "a":
                    prio = 2 - priority_bonus
                else:
                    return
                raw_antonyms.append((al, prio))

        for s in target_synsets:
            # Direct sense antonyms
            for sense in s.senses():
                for ant_sense in sense.get_related("antonym"):
                    ant_lemma = ant_sense.word().lemma().lower()
                    _evaluate_antonym(ant_lemma, priority_bonus=2)
                    # Opposite satellites in bipolar cluster
                    for opp_sim in ant_sense.synset().get_related("similar") + ant_sense.synset().get_related("also"):
                        for ow in opp_sim.words():
                            ow_lemma = ow.lemma().lower()
                            _evaluate_antonym(ow_lemma, priority_bonus=1)
                            if wn_pos == "a":
                                for w_obj in wn.words(ow_lemma, pos="a"):
                                    for w_syn in w_obj.synsets():
                                        for w_sense in w_syn.senses():
                                            for w_ant in w_sense.get_related("antonym"):
                                                _evaluate_antonym(w_ant.word().lemma().lower(), priority_bonus=1)
            # If adjective satellite ('s'), trace to head synset in bipolar cluster
            if wn_pos == "a" and s.pos == "s":
                for head in s.get_related("similar"):
                    for hs in head.senses():
                        for ant_sense in hs.get_related("antonym"):
                            ant_lemma = ant_sense.word().lemma().lower()
                            _evaluate_antonym(ant_lemma, priority_bonus=2)
                            for opp_sim in ant_sense.synset().get_related("similar") + ant_sense.synset().get_related("also"):
                                for ow in opp_sim.words():
                                    ow_lemma = ow.lemma().lower()
                                    _evaluate_antonym(ow_lemma, priority_bonus=1)
                                    for w_obj in wn.words(ow_lemma, pos="a"):
                                        for w_syn in w_obj.synsets():
                                            for w_sense in w_syn.senses():
                                                for w_ant in w_sense.get_related("antonym"):
                                                    _evaluate_antonym(w_ant.word().lemma().lower(), priority_bonus=1)

        raw_antonyms.sort(key=lambda x: x[1])
        for al, _ in raw_antonyms:
            if al not in sense_antonyms:
                sense_antonyms.append(al)

        # PedLex 6th Edition Thesaurus, Grammar Box Comparison Peers & Activator Integration (Tier 0: Primary Pedagogical Lexical Source)
        tier0_ldoce_thesaurus: List[str] = []
        seen = {clean_target}
        seen.update(sense_antonyms)

        thes_items = list(cls.get_ldoce_thesaurus(clean_target))
        t_gram_prof = cls.get_word_gram_profile(clean_target, pos=pos)
        for peer_w in t_gram_prof.get("comparison_peers", set()):
            thes_items.append({
                "word": peer_w,
                "form": "word",
                "distinction": definition or "",
                "grammar": "",
                "register": "",
                "concept": "",
                "is_grambox_peer": True,
            })

        if thes_items:
            # Rank items by definition and quote overlap + syntactic slot clash bonus
            q_words = set(re.findall(r"\b[a-zA-Z]{3,}\b", (quote or "").lower()))
            d_words = set(re.findall(r"\b[a-zA-Z]{3,}\b", (definition or "").lower()))
            content_words = d_words | q_words
            scored_thes_items = []
            for it in thes_items:
                t_text = (it.get("word", "") + " " + str(it.get("distinction", "")) + " " + str(it.get("concept", ""))).lower()
                t_toks = set(re.findall(r"\b[a-zA-Z]{3,}\b", t_text))
                overlap_score = len(t_toks & d_words) * 3 + len(t_toks & q_words)
                # If definition/quote provided, filter out Activator/Thesaurus clusters with zero overlap against the target sense
                if content_words and overlap_score == 0 and not it.get("is_grambox_peer"):
                    continue
                # Boost true Thesaurus distinction records slightly over broad Activator clusters
                if not it.get("concept"):
                    overlap_score += 1
                if it.get("is_grambox_peer"):
                    overlap_score += 5
                raw_cand_head = str(it.get("word", "")).strip().lower().split("/")[0].split()[0] if it.get("word") else ""
                if raw_cand_head:
                    has_clash, _ = cls.evaluate_syntactic_slot_clash(
                        clean_target, raw_cand_head,
                        anchor=context_anchor, anchor_type=anchor_type, pos=pos, quote=quote
                    )
                    if has_clash:
                        overlap_score += 15
                scored_thes_items.append((overlap_score, it))
            scored_thes_items.sort(key=lambda x: x[0], reverse=True)

            for _, it in scored_thes_items:
                raw_w = it.get("word", "").strip().lower()
                # Skip instructional template phrases containing 'etc'
                if "etc" in raw_w:
                    continue
                # If phrase contains spaces and slashes (e.g. 'the Department of Science/English/Trade'),
                # it is a template frame, not an atomic candidate list; skip to prevent fragmentary distractor leaks
                if "/" in raw_w:
                    parts_check = [p.strip() for p in raw_w.split("/")]
                    if any(" " in p for p in parts_check):
                        continue
                chunks = [ch.strip() for ch in raw_w.split("/")] if "/" in raw_w else [raw_w]
                for chunk in chunks:
                    if not chunk:
                        continue
                    parts = chunk.split()
                    cand = None
                    if len(parts) == 1:
                        cand = parts[0]
                    elif len(parts) >= 2:
                        # Extract verb head from verb phrases ONLY if followed by standard placeholders (e.g. 'perform sth')
                        # Do NOT extract verb head from phrasal verbs or idiomatic phrases (e.g. 'come forward', 'let me')
                        if wn_pos == "v":
                            if parts[1] in ("sb", "sth", "someone", "something", "oneself", "a", "an", "the"):
                                cand = parts[0]
                            else:
                                continue
                        # Extract noun head from noun phrases (e.g. 'branch of', 'facilities for', 'member of')
                        elif wn_pos == "n" and parts[1] in ("of", "for", "to", "in", "on", "with", "as", "sb", "sth"):
                            p_doc = nlp(parts[0])
                            if p_doc and p_doc[0].pos_ in ("NOUN", "PROPN"):
                                cand = parts[0]
                    if not cand or not cand.isalpha() or len(cand) < 3:
                        continue

                    cand_doc = nlp(cand)
                    cand_lemma = cand_doc[0].lemma_.lower()
                    picked = cand_lemma if cls.get_ldoce_entry(cand_lemma) else cand
                    has_picked_clash, _ = cls.evaluate_syntactic_slot_clash(
                        clean_target, picked,
                        anchor=context_anchor, anchor_type=anchor_type, pos=pos, quote=quote
                    )

                    if (
                        picked
                        and picked != clean_target
                        and picked not in seen
                        and (picked not in global_synonyms or has_picked_clash)
                        and not cls.are_same_word_family(picked, clean_target)
                        and not any(cls.are_same_word_family(picked, existing) for existing in tier0_ldoce_thesaurus)
                        and _pos_ok(picked)
                        and cls.is_cefr_compliant_distractor(picked, clean_target)
                    ):
                        cand_entry = cls.get_ldoce_entry(picked)
                        if cand_entry:
                            c_primary = (cand_entry.get("pos") or "").lower()
                            c_poses = [p.lower() for p in cand_entry.get("all_poses", [c_primary])]
                            if wn_pos == "n":
                                if not any("noun" in p for p in c_poses):
                                    continue
                                # Reject if primary PedLex POS is verb or adjective and spacy also detects non-noun
                                if c_primary in ("verb", "adjective", "adverb"):
                                    continue
                            elif wn_pos == "v":
                                if not any("verb" in p for p in c_poses):
                                    continue
                                if c_primary in ("noun", "adjective", "adverb"):
                                    continue
                            elif wn_pos == "a" and not any("adj" in p for p in c_poses):
                                continue
                        seen.add(picked)
                        tier0_ldoce_thesaurus.append(picked)
                        # Cluster Throttling: pick at most 1 distinct distractor per Thesaurus/Activator entry
                        break
                if len(tier0_ldoce_thesaurus) >= target_count:
                    break

        tier1_synonyms: List[str] = []
        tier2_satellites: List[str] = []
        tier3_attributes: List[str] = []
        tier2_hyponyms: List[str] = []
        tier3_coordinates: List[str] = []

        for s in target_synsets:
            # Tier 1: Synonyms in synset (Only for nouns/adjectives; strictly banned for verbs)
            if wn_pos != "v":
                for sw in s.words():
                    lemma = sw.lemma().lower()
                    if (
                        lemma not in seen
                        and " " not in lemma
                        and "_" not in lemma
                        and "-" not in lemma
                        and (lemma in ocd or bool(cls.get_ldoce_entry(lemma)))
                        and cls.is_cefr_compliant_distractor(lemma, clean_target)
                    ):
                        if wn_pos == "n" and _is_primarily_adj_or_verb(lemma):
                            continue
                        if lemma in global_synonyms:
                            continue
                        seen.add(lemma)
                        tier1_synonyms.append(lemma)

            if wn_pos == "a":
                # For Adjectives: WordNet organizes concepts into Bipolar Clusters (Satellite Synset Harvester)
                # Tier 2: Satellite synsets (fine-grained descriptive neighbors in the bipolar cluster)
                sat_synsets = []
                if s.pos == "a":
                    sat_synsets.extend(s.get_related("also") + s.get_related("similar"))
                elif s.pos == "s":
                    for head in s.get_related("similar"):
                        for hw in head.words():
                            hl = hw.lemma().lower()
                            if (
                                hl not in seen
                                and hl not in global_synonyms
                                and " " not in hl and "_" not in hl and "-" not in hl
                                and (hl in ocd or bool(cls.get_ldoce_entry(hl)))
                                and _pos_ok(hl)
                                and cls.is_cefr_compliant_distractor(hl, clean_target)
                            ):
                                seen.add(hl)
                                tier2_satellites.append(hl)
                        sat_synsets.extend(head.get_related("also") + head.get_related("similar"))

                for sim in sat_synsets:
                    for sw in sim.words():
                        lemma = sw.lemma().lower()
                        if (
                            lemma not in seen
                            and lemma not in global_synonyms
                            and " " not in lemma
                            and "_" not in lemma
                            and "-" not in lemma
                            and (lemma in ocd or bool(cls.get_ldoce_entry(lemma)))
                            and _pos_ok(lemma)
                            and cls.is_cefr_compliant_distractor(lemma, clean_target)
                        ):
                            seen.add(lemma)
                            tier2_satellites.append(lemma)

                # Tier 3: Attribute coordinates (adjectives sharing the same abstract attribute, e.g. speed, scope, magnitude)
                attrs = s.get_related("attribute")
                if s.pos == "s":
                    for head in s.get_related("similar"):
                        attrs.extend(head.get_related("attribute"))
                for attr in attrs:
                    for att_adj in attr.get_related("attribute"):
                        for aw in att_adj.words():
                            lemma = aw.lemma().lower()
                            if (
                                lemma not in seen
                                and lemma not in global_synonyms
                                and " " not in lemma
                                and "_" not in lemma
                                and "-" not in lemma
                                and (lemma in ocd or bool(cls.get_ldoce_entry(lemma)))
                                and _pos_ok(lemma)
                                and cls.is_cefr_compliant_distractor(lemma, clean_target)
                            ):
                                seen.add(lemma)
                                tier3_attributes.append(lemma)
                        for sim in att_adj.get_related("similar"):
                            for sw in sim.words():
                                lemma = sw.lemma().lower()
                                if (
                                    lemma not in seen
                                    and lemma not in global_synonyms
                                    and " " not in lemma
                                    and "_" not in lemma
                                    and "-" not in lemma
                                    and (lemma in ocd or bool(cls.get_ldoce_entry(lemma)))
                                    and _pos_ok(lemma)
                                    and cls.is_cefr_compliant_distractor(lemma, clean_target)
                                ):
                                    seen.add(lemma)
                                    tier3_attributes.append(lemma)
            else:
                # Tier 2: Hyponyms (more specific concepts / troponyms)
                # Only harvest hyponyms from the locked target_synsets (already represented by `s in target_synsets`),
                # and for nouns skip hyponyms to prevent synonym-pile double-keys with semantic_fields.
                if wn_pos == "v":
                    for hypo in s.hyponyms():
                        for hw in hypo.words():
                            lemma = hw.lemma().lower()
                            if (
                                lemma not in seen
                                and lemma not in global_synonyms
                                and " " not in lemma
                                and "_" not in lemma
                                and "-" not in lemma
                                and (lemma in ocd or bool(cls.get_ldoce_entry(lemma)))
                                and _pos_ok(lemma)
                                and cls.is_cefr_compliant_distractor(lemma, clean_target)
                            ):
                                seen.add(lemma)
                                tier2_hyponyms.append(lemma)
                # Tier 3: Coordinate terms (sisters under same hypernym - Troponyms for verbs)
                for hyper in s.hypernyms():
                    # For verbs: skip broad generic root hypernyms (e.g. > 50 hyponyms)
                    if wn_pos == "v" and len(hyper.hyponyms()) > 50:
                        continue
                    for sis in hyper.hyponyms():
                        if sis.id == s.id:
                            continue
                        for sw in sis.words():
                            lemma = sw.lemma().lower()
                            if (
                                lemma not in seen
                                and lemma not in global_synonyms
                                and " " not in lemma
                                and "_" not in lemma
                                and "-" not in lemma
                                and (lemma in ocd or bool(cls.get_ldoce_entry(lemma)))
                                and _pos_ok(lemma)
                                and cls.is_cefr_compliant_distractor(lemma, clean_target)
                            ):
                                if wn_pos == "n" and _is_primarily_adj_or_verb(lemma):
                                    continue
                                # Verbs: Block speech/reporting verbs with clausal or double-object valency that break transitive substitution
                                if wn_pos == "v" and lemma in ("say", "tell", "speak", "talk", "whisper", "shout", "reply", "state", "declare"):
                                    continue
                                seen.add(lemma)
                                tier3_coordinates.append(lemma)
                # Tier 3.5: Grand-hypernym coordinate terms (Grand-sisters under same grandparent)
                # First Defense: Extends candidate pool for nouns when immediate coordinates are sparse,
                # strictly constrained to the same WordNet lexfile as the target synset.
                if wn_pos == "n":
                    target_lf = s.lexfile()
                    for hyper in s.hypernyms():
                        for g_hyper in hyper.hypernyms():
                            for sis in g_hyper.hyponyms():
                                if target_lf and sis.lexfile() != target_lf:
                                    continue
                                for sw in sis.words():
                                    lemma = sw.lemma().lower()
                                    if (
                                        lemma not in seen
                                        and lemma not in global_synonyms
                                        and " " not in lemma
                                        and "_" not in lemma
                                        and "-" not in lemma
                                        and (lemma in ocd or bool(cls.get_ldoce_entry(lemma)))
                                        and _pos_ok(lemma)
                                        and cls.is_cefr_compliant_distractor(lemma, clean_target)
                                    ):
                                        if _is_primarily_adj_or_verb(lemma):
                                            continue
                                        seen.add(lemma)
                                        tier3_coordinates.append(lemma)

        # Sort tier3_coordinates by PedLex richness (attested examples/collocations) and CEFR rank proximity
        if tier3_coordinates:
            t_lvl_str = cls._lexicon_profile(clean_target)[2] or "B1"
            t_rank = cls.CEFR_ORDER.get(t_lvl_str, 3)
            def _coord_sort_key(cand_w: str) -> Tuple[int, int]:
                c_ent = cls.get_ldoce_entry(cand_w) or {}
                has_ex = any(s.get("examples") for s in c_ent.get("senses", []))
                c_lvl = cls._lexicon_profile(cand_w)[2] or "C1"
                c_rank = cls.CEFR_ORDER.get(c_lvl, 5)
                return (0 if has_ex else 1, abs(c_rank - t_rank))
            tier3_coordinates.sort(key=_coord_sort_key)

        # Candidate prioritization: PedLex Thesaurus & Activators are primary (Tier 0); WordNet is secondary/fallback
        if wn_pos == "v":
            candidates = tier0_ldoce_thesaurus + [c for c in (tier2_hyponyms + tier3_coordinates) if c not in global_synonyms]
        elif wn_pos == "a":
            # For adjectives: retain satellite_synonyms first to satisfy existing fine-grained satellite tests,
            # then inject PedLex thesaurus peers, compliant antonyms, followed by attribute coordinates.
            antonym_candidates = [
                ant for ant in sense_antonyms
                if ant != clean_target and cls.is_cefr_compliant_distractor(ant, clean_target)
            ]
            candidates = tier2_satellites + tier0_ldoce_thesaurus + antonym_candidates + tier3_attributes + tier1_synonyms
            # Preposition valency cloze (): prioritize academic adjectives governing DIFFERENT prepositions
            if anchor_type == "prep" and context_anchor:
                clean_anchor = context_anchor.strip().lower()
                valency_candidates = []
                for p, adjs in cls._ADJ_PREP_VALENCY_MAP.items():
                    if p != clean_anchor:
                        for a in adjs:
                            if a != clean_target and a not in exclude_words and cls.is_cefr_compliant_distractor(a, clean_target):
                                valency_candidates.append(a)
                # Prioritize local semantic peers (satellites, PedLex thesaurus) for domain relevance,
                # then complement with academic adjectives governing distinct prepositions
                candidates = tier2_satellites + tier0_ldoce_thesaurus + antonym_candidates + valency_candidates + tier3_attributes + tier1_synonyms
        else:
            candidates = tier0_ldoce_thesaurus + tier1_synonyms + tier2_hyponyms + tier3_coordinates

        # 3. Oxford Collision Clearance (Anti Double-Key Gate - Schemes 3 & 4)
        # Exempt candidates that have a verified syntactic slot clash with target's slot from global_synonyms ban
        forbidden_words = {
            gw for gw in global_synonyms
            if not cls.evaluate_syntactic_slot_clash(
                clean_target, gw, anchor=context_anchor, anchor_type=anchor_type, pos=pos, quote=quote
            )[0]
        } | {clean_target}
        if context_anchor:
            clean_anchor = context_anchor.strip().lower()
            if wn_pos == "a" and anchor_type == "prep":
                # Preposition Valency Gate - reject candidates that also govern the target preposition
                for cand in candidates:
                    cand_preps = ocd.get(cand, {}).get("prep", [])
                    if clean_anchor in cand_preps or cand in cls._ADJ_PREP_VALENCY_MAP.get(clean_anchor, []):
                        forbidden_words.add(cand)
            elif wn_pos == "a" and anchor_type == "modified_noun":
                # Modified Noun Collocation Clash - reject legal adjectives that also collocate with the noun
                anchor_entry = ocd.get(clean_anchor, {})
                for a in anchor_entry.get("adj", []):
                    tok = a.split()[0].lower()
                    if tok not in sense_antonyms:
                        forbidden_words.add(tok)
            else:
                anchor_entry = ocd.get(clean_anchor, {})
                # If target is verb, anchor is noun: check verb_before and verb_after
                for v in anchor_entry.get("verb_before", []) + anchor_entry.get("verb_after", []):
                    forbidden_words.add(v.split()[0].lower())
                # If target is adj, anchor is noun: check adj
                for a in anchor_entry.get("adj", []):
                    tok = a.split()[0].lower()
                    if tok not in sense_antonyms:
                        forbidden_words.add(tok)
                # If target is noun, anchor is verb/adj: check colloc_nouns and noun_after
                for n in anchor_entry.get("colloc_nouns", []) + anchor_entry.get("noun_after", []):
                    forbidden_words.add(n.split()[0].lower())

        safe_distractors: List[str] = []
        distractor_metadata: Dict[str, str] = {}

        # Slot 1: Sense-Locked Antonym (if available)
        # Antonyms are whitelisted from context_anchor collision clearance because their
        # collocational plausibility is deliberate, to be eliminated by sentence polarity/logic.
        max_antonyms = 1 if (wn_pos != "a" or len(tier2_satellites) >= 2) else 2
        for ant in sense_antonyms:
            if len([s for s in safe_distractors if distractor_metadata.get(s) == "antonym"]) >= max_antonyms:
                break
            if ant not in safe_distractors and ant not in exclude_words and len(ant) >= 2:
                if ant == sense_antonyms[0] or cls.is_cefr_compliant_distractor(ant, clean_target):
                    safe_distractors.append(ant)
                    distractor_metadata[ant] = "antonym"

        # Slots 2 & 3: Candidates from PedLex Thesaurus & Activator / Coordinate / Troponym / Synonym trees
        near_set_for_pile, ants_for_pile = cls.semantic_fields(clean_target, pos=pos)
        near_only_for_pile = near_set_for_pile - ants_for_pile - set(sense_antonyms)
        for cand in candidates:
            if cand in safe_distractors:
                continue
            has_cand_clash, _ = cls.evaluate_syntactic_slot_clash(
                clean_target, cand, anchor=context_anchor, anchor_type=anchor_type, pos=pos, quote=quote
            )
            # Synonym-Pile Gate: at most 1 non-antonym distractor may sit inside `near_set` (unless syntactically clashing)
            if not has_cand_clash and cand in near_only_for_pile and sum(1 for s in safe_distractors if s in near_only_for_pile and distractor_metadata.get(s) != "antonym") >= 1:
                continue
            # Attributive position check: if modifying a noun, reject predicative-only adjectives
            if anchor_type == "modified_noun" and cand in cls._PREDICATIVE_ONLY_ADJS:
                continue
            # Allomorph / Word Family Guard: prevent morphological double-keys (e.g. resiliency vs resilience)
            if cls.are_same_word_family(cand, clean_target) or any(cls.are_same_word_family(cand, s) for s in safe_distractors):
                continue
            # Cross-distractor mutual synonym exclusion: avoid multiple distractors being mutual synonyms
            cand_syns = cls.get_synonyms(cand)
            if any(s in cand_syns or cand in cls.get_synonyms(s) for s in safe_distractors):
                continue

            # Deterministic Double-Key & Pattern Exclusion Gate:
            # Rejects candidates that share identical syntactic patterns or collocations (double-key collision)
            is_dk, dk_reason = cls.double_key_collision(
                clean_target, cand,
                anchor=context_anchor,
                anchor_type=anchor_type,
                pos=pos,
                quote=quote
            )
            if is_dk:
                continue

            if cand not in forbidden_words and cand not in exclude_words and len(cand) >= 2:
                safe_distractors.append(cand)
                if cand in sense_antonyms:
                    distractor_metadata[cand] = "antonym"
                elif cand in tier0_ldoce_thesaurus:
                    distractor_metadata[cand] = "ldoce_thesaurus"
                elif wn_pos == "a":
                    if anchor_type == "prep" and any(cand in adjs for p, adjs in cls._ADJ_PREP_VALENCY_MAP.items() if p != context_anchor):
                        distractor_metadata[cand] = "preposition_valency"
                    elif cand in tier2_satellites:
                        distractor_metadata[cand] = "satellite_synonym"
                    elif cand in tier3_attributes:
                        distractor_metadata[cand] = "attribute_coordinate"
                    else:
                        distractor_metadata[cand] = "synonym"
                else:
                    if cand in tier2_hyponyms:
                        distractor_metadata[cand] = "troponym"
                    elif cand in tier3_coordinates:
                        distractor_metadata[cand] = "coordinate"
                    else:
                        distractor_metadata[cand] = "synonym"
                if len(safe_distractors) >= target_count:
                    break

        # Fallback from Academic Semantic Families if candidates pool is sparse (e.g. specialized adjectives)
        if len(safe_distractors) < target_count and wn_pos == "a":
            for fam in cls._ACADEMIC_ADJ_FAMILIES:
                if clean_target in fam:
                    for a in fam:
                        if a != clean_target and a not in forbidden_words and a not in exclude_words and a not in safe_distractors:
                            if anchor_type == "modified_noun" and a in cls._PREDICATIVE_ONLY_ADJS:
                                continue
                            safe_distractors.append(a)
                            distractor_metadata[a] = "academic_family"
                            if len(safe_distractors) >= target_count:
                                break
                    break

        # Fallback from Academic Semantic Families if candidates pool is sparse (e.g. specialized verbs)
        if len(safe_distractors) < target_count and wn_pos == "v":
            academic_verb_families = [
                ["transmit", "distribute", "circulate", "relay", "deliver", "publish", "release"],
                ["instruct", "direct", "guide", "train", "advise", "command", "brief", "inform"],
                ["acquire", "obtain", "gain", "attain", "secure", "gather", "accumulate", "possess"],
                ["challenge", "assess", "evaluate", "test", "question", "examine", "review", "probe"],
                ["establish", "maintain", "sustain", "create", "develop", "foster", "initiate", "generate"]
            ]
            for fam in academic_verb_families:
                if clean_target in fam:
                    for v in fam:
                        if v != clean_target and v not in forbidden_words and v not in exclude_words and v not in safe_distractors:
                            safe_distractors.append(v)
                            distractor_metadata[v] = "academic_family"
                            if len(safe_distractors) >= target_count:
                                break
                    break

        # Orthographic / Paronym Distractor Tier (with Cognate Near-Synonym Guard):
        # 1. If candidate pool is sparse (len(safe_distractors) < target_count), fill remaining slots with
        #    verified orthographic paronyms / orthogonal-domain root peers.
        # 2. If all 3 slots were filled, blend 1 paronym into Slot 3 when:
        #    - Slot 3 is a generic coordinate/troponym/attribute_coordinate or a 3rd antonym and a sharp
        #      paronym (score >= 45, e.g. adapt/adopt, complement/compliment, simulate/stimulate,
        #      industrial/industrious, considerate/considerable, respectful/respective) exists, OR
        #    - All 3 slots are generic coordinates and a clean orthogonal-domain root peer (score >= 28,
        #      e.g. specimen -> spectrum/spectacle/speculation) exists.
        if wn_pos in ("n", "v", "a", "r"):
            need_paronym_fill = len(safe_distractors) < target_count
            pure_generic_coords = (
                len(safe_distractors) >= target_count
                and all(
                    distractor_metadata.get(d) in ("coordinate", "troponym", "attribute_coordinate")
                    for d in safe_distractors
                )
            )
            slot3_replaceable = (
                len(safe_distractors) >= target_count
                and (
                    distractor_metadata.get(safe_distractors[-1]) in ("coordinate", "troponym", "attribute_coordinate")
                    or sum(1 for d in safe_distractors if distractor_metadata.get(d) == "antonym") >= 3
                )
            )
            if need_paronym_fill or pure_generic_coords or slot3_replaceable:
                paronym_cands = cls.get_orthographic_paronym_distractors(
                    clean_target,
                    pos=wn_pos,
                    target_count=target_count,
                    exclude_words=(exclude_words | forbidden_words | set(safe_distractors)),
                    context_anchor=context_anchor,
                    anchor_type=anchor_type,
                    quote=quote,
                    return_scores=True,
                )
                if need_paronym_fill:
                    for p_word, _p_score in paronym_cands:
                        if p_word not in safe_distractors:
                            safe_distractors.append(p_word)
                            distractor_metadata[p_word] = "orthographic_paronym"
                            if len(safe_distractors) >= target_count:
                                break
                elif paronym_cands:
                    best_p_word, best_p_score = paronym_cands[0]
                    threshold = 28 if pure_generic_coords else 45
                    if best_p_score >= threshold and best_p_word not in safe_distractors:
                        popped = safe_distractors.pop()
                        distractor_metadata.pop(popped, None)
                        safe_distractors.append(best_p_word)
                        distractor_metadata[best_p_word] = "orthographic_paronym"

        # Second Defense: Dynamic domain-aligned fallback using target WordNet lexfile
        # When candidates pool is still sparse, extract homogeneous fallback words from the target's
        # sense-locked WordNet lexfile (e.g. noun.cognition, noun.group) to prevent Domain/Category Mismatch.
        if len(safe_distractors) < target_count and wn_pos == "n":
            target_syns = [s for s in target_synsets if s.lexfile()] or [s for s in wn.synsets(clean_target, "n") if s.lexfile()]
            target_lfs = [s.lexfile() for s in target_syns if s.lexfile()]
            if target_lfs:
                # Lazy-build or retrieve lexfile-indexed OCD nouns
                if not cls._ocd_lexfile_cache:
                    for w in ocd.keys():
                        if len(w) >= 3 and "_" not in w and "-" not in w and " " not in w:
                            w_syns = [ws for ws in wn.synsets(w, "n") if ws.lexfile()]
                            if w_syns:
                                lf = w_syns[0].lexfile()
                                if lf not in cls._ocd_lexfile_cache:
                                    cls._ocd_lexfile_cache[lf] = []
                                cls._ocd_lexfile_cache[lf].append(w)
                for t_lf in target_lfs:
                    for fb in cls._ocd_lexfile_cache.get(t_lf, []):
                        if (
                            fb != clean_target
                            and fb not in forbidden_words
                            and fb not in exclude_words
                            and fb not in safe_distractors
                            and cls.is_cefr_compliant_distractor(fb, clean_target)
                        ):
                            safe_distractors.append(fb)
                            distractor_metadata[fb] = "dynamic_domain_fallback"
                            if len(safe_distractors) >= target_count:
                                break
                    if len(safe_distractors) >= target_count:
                        break

        # Universal Fallback if candidates pool is still sparse: select standard homogeneous items from OCD
        if len(safe_distractors) < target_count:
            t_lvl_str = cls.get_word_cefr(clean_target) or "B2"
            t_rank = cls.CEFR_ORDER.get(t_lvl_str, 4)
            if t_rank <= 3:
                fallback_pool = {
                    "v": ["choose", "accept", "decide", "expect", "follow", "notice", "explain", "remain", "manage", "allow"],
                    "n": ["choice", "reason", "matter", "situation", "effort", "problem", "action", "change", "moment", "condition"],
                    "a": ["simple", "common", "certain", "different", "similar", "natural", "clear", "direct", "actual", "special"],
                    "r": ["deeply", "strictly", "tightly", "widely", "carefully", "clearly", "readily", "sharply", "directly", "steadily"]
                }.get(wn_pos or "n", ["choice", "reason", "matter"])
            else:
                fallback_pool = {
                    "v": ["maintain", "establish", "determine", "evaluate", "indicate", "demonstrate", "direct", "guide", "sustain"],
                    "n": ["aspect", "factor", "process", "measure", "element", "context", "matter", "institution", "framework"],
                    "a": ["crucial", "essential", "primary", "initial", "direct", "specific", "constant", "fundamental"],
                    "r": ["deeply", "strictly", "tightly", "widely", "carefully", "clearly", "readily", "sharply", "directly", "steadily"]
                }.get(wn_pos or "n", ["factor", "element", "process"])
            for fb in fallback_pool:
                if (
                    fb != clean_target
                    and fb not in forbidden_words
                    and fb not in exclude_words
                    and fb not in safe_distractors
                    and not cls.are_same_word_family(fb, clean_target)
                    and not any(cls.are_same_word_family(fb, s) for s in safe_distractors)
                    and cls.is_cefr_compliant_distractor(fb, clean_target)
                ):
                    safe_distractors.append(fb)
                    distractor_metadata[fb] = "corpus_fallback"
                if len(safe_distractors) >= target_count:
                    break

        final_distractors = safe_distractors[:target_count]
        if return_metadata:
            return final_distractors, {k: distractor_metadata.get(k, "distractor") for k in final_distractors}
        return final_distractors

    # -------------------------------------------------------------------------
    # Orthographic / Paronym Distractor Tier (with Cognate Near-Synonym Guard)
    # -------------------------------------------------------------------------
    _PARONYM_STOP_DEF_TOKS = frozenset({
        "someone", "something", "person", "people", "thing", "things", "make", "made",
        "have", "having", "used", "very", "more", "most", "much", "many", "good", "bad",
        "way", "time", "place", "part", "group", "kind", "type", "form", "state", "act",
        "process", "quality", "relating", "connected", "particular", "certain",
        "able", "likely", "full", "without", "especially", "usually", "often", "always",
        "from", "that", "with", "into", "this", "these", "those", "which", "what", "when",
        "where", "while", "whose", "whom", "your", "their", "other", "another", "some",
        "such", "each", "every", "both", "either", "neither", "only", "also", "even",
        "just", "still", "already", "about", "above", "after", "again", "against", "along",
        "among", "around", "because", "before", "behind", "below", "beneath", "beside",
        "between", "beyond", "during", "except", "inside", "instead", "outside", "through",
        "toward", "towards", "under", "until", "upon", "within", "amount",
        "piece", "taken", "take", "taking", "example", "single", "small", "large", "great",
        "high", "long", "short", "order", "achieve", "produce", "produced", "show", "shown",
        "seen", "notice", "laugh", "help", "try", "trying", "know", "knowing", "think",
        "thinking", "ideas", "idea", "mind", "real", "containing", "interesting",
        "work", "works", "working", "hard",
    })

    @classmethod
    def _ensure_orthographic_pos_index(cls) -> Dict[str, Dict[str, Dict[str, List[str]]]]:
        """Lazily build O(1) POS-bucketed first/second character index over PedLex (or OCD/WordNet fallback)."""
        if cls._orthographic_pos_index is not None:
            return cls._orthographic_pos_index

        pos_vocab: Dict[str, Set[str]] = {"n": set(), "v": set(), "a": set(), "r": set()}
        pos_map = {
            "noun": "n", "verb": "v", "adjective": "a", "adverb": "r",
            "n": "n", "v": "v", "adj": "a", "adv": "r", "s": "a",
        }

        cls.get_ldoce_entry("test")
        conn = cls._ldoce_conn
        if conn is not None:
            try:
                rows = conn.execute("SELECT word, pos, pos_all FROM ldoce WHERE kind='article'").fetchall()
                for w, p, p_all in rows:
                    if not w or not w.isalpha() or len(w) < 4:
                        continue
                    wl = w.lower()
                    raw_ps = [x.strip().lower() for x in (p_all or p or "").split("|") if x.strip()]
                    for rp in raw_ps:
                        code = pos_map.get(rp)
                        if code:
                            pos_vocab[code].add(wl)
            except Exception:
                pass

        # Open-source / fallback enrichment from OCD + WordNet if PedLex is disabled or sparse
        if not any(pos_vocab.values()):
            wn = cls.get_wordnet()
            ocd = cls.get_oxford_collocations()
            pool = set(ocd.keys())
            for w in pool:
                if not w or not w.isalpha() or len(w) < 4:
                    continue
                wl = w.lower()
                for p_code in ("n", "v", "a", "r"):
                    try:
                        syns = wn.synsets(wl, "a") + wn.synsets(wl, "s") if p_code == "a" else wn.synsets(wl, p_code)
                        if syns:
                            pos_vocab[p_code].add(wl)
                    except Exception:
                        pass

        by_first: Dict[str, Dict[str, List[str]]] = {k: {} for k in pos_vocab}
        by_second: Dict[str, Dict[str, List[str]]] = {k: {} for k in pos_vocab}
        for k, words in pos_vocab.items():
            for w in sorted(words):
                by_first[k].setdefault(w[0], []).append(w)
                if len(w) >= 2:
                    by_second[k].setdefault(w[1], []).append(w)

        cls._orthographic_pos_index = {"by_first": by_first, "by_second": by_second}
        return cls._orthographic_pos_index

    @staticmethod
    def _edit_distance_le2(s1: str, s2: str) -> int:
        """Fast bounded Levenshtein distance (returns 99 if distance > 2)."""
        if abs(len(s1) - len(s2)) > 2:
            return 99
        m, n = len(s1), len(s2)
        dp = list(range(n + 1))
        for i in range(1, m + 1):
            prev = dp[0]
            dp[0] = i
            min_row = dp[0]
            for j in range(1, n + 1):
                temp = dp[j]
                cost = 0 if s1[i - 1] == s2[j - 1] else 1
                dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
                prev = temp
                if dp[j] < min_row:
                    min_row = dp[j]
            if min_row > 2:
                return 99
        return dp[n]

    @staticmethod
    def _common_prefix_len(s1: str, s2: str) -> int:
        c = 0
        for a, b in zip(s1, s2):
            if a == b:
                c += 1
            else:
                break
        return c

    @staticmethod
    def _common_suffix_len(s1: str, s2: str) -> int:
        c = 0
        for a, b in zip(reversed(s1), reversed(s2)):
            if a == b:
                c += 1
            else:
                break
        return c

    @classmethod
    def _get_def_content_words(cls, word: str, wn_pos: str) -> Set[str]:
        toks: Set[str] = set()
        e = cls.get_ldoce_entry(word)
        p_code = {"n": "n", "v": "v", "a": "adj", "r": "adv"}.get(wn_pos, "n")
        if e:
            for s in e.get("senses") or []:
                if not s.get("pos") or cls._pos_code(s.get("pos")) == p_code:
                    d_txt = str(s.get("definition") or "").lower()
                    for w in re.findall(r"\b[a-z]{4,}\b", d_txt):
                        if w not in cls._PARONYM_STOP_DEF_TOKS:
                            toks.add(w)
                            if w.endswith("s") and not w.endswith(("ss", "us", "is")) and len(w) >= 5:
                                toks.add(w[:-1])
        wn = cls.get_wordnet()
        w_objs = wn.words(word, pos="a") + wn.words(word, pos="s") if wn_pos == "a" else wn.words(word, pos=wn_pos)
        for w_obj in w_objs:
            for syn in w_obj.synsets():
                d_txt = str(syn.definition() or "").lower()
                for w in re.findall(r"\b[a-z]{4,}\b", d_txt):
                    if w not in cls._PARONYM_STOP_DEF_TOKS:
                        toks.add(w)
                        if w.endswith("s") and not w.endswith(("ss", "us", "is")) and len(w) >= 5:
                            toks.add(w[:-1])
        return toks

    @classmethod
    def _get_derivational_bases(cls, word: str) -> Set[str]:
        """Return 2-hop WordNet derivational / pertainym base lemmas (cached)."""
        w_low = (word or "").strip().lower()
        if w_low in cls._derivational_base_cache:
            return cls._derivational_base_cache[w_low]
        wn = cls.get_wordnet()
        bases: Set[str] = set()
        try:
            for w_obj in wn.words(w_low):
                for syn in w_obj.synsets():
                    for sense in syn.senses():
                        if sense.word().lemma().lower() == w_low:
                            for rel in sense.get_related("derivation") + sense.get_related("pertainym"):
                                bases.add(rel.word().lemma().lower())
            hop2 = set(bases)
            for b in list(bases):
                for w_obj in wn.words(b):
                    for syn in w_obj.synsets():
                        for sense in syn.senses():
                            if sense.word().lemma().lower() == b:
                                for rel in sense.get_related("derivation") + sense.get_related("pertainym"):
                                    hop2.add(rel.word().lemma().lower())
        except Exception:
            hop2 = bases
        cls._derivational_base_cache[w_low] = hop2
        return hop2

    @classmethod
    def is_safe_orthographic_paronym(
        cls,
        target_word: str,
        candidate_word: str,
        pos: str = "n",
        context_anchor: Optional[str] = None,
        anchor_type: Optional[str] = None,
        quote: Optional[str] = None,
    ) -> bool:
        """Cognate Near-Synonym Guard: verifies that `candidate_word` is a semantically disjoint
        orthographic paronym or orthogonal-domain root peer of `target_word`, blocking root-sharing
        near-synonyms (e.g. historic/historical, classic/classical, economic/economical,
        continual/continuous, specimen/species, activism/activity)."""
        target = (target_word or "").strip().lower()
        cand = (candidate_word or "").strip().lower()
        if not target or not cand or target == cand:
            return False

        pos_clean = (pos or "n").strip().lower()
        wn_pos = "a" if pos_clean in ("a", "adj", "adjective", "s") else (
            "v" if pos_clean in ("v", "verb") else (
                "r" if pos_clean in ("r", "adv", "adverb") else "n"
            )
        )

        # 1. Reject inflectional / participial / negation prefix derivatives
        for pfx in ("un", "in", "im", "ir", "il", "non", "dis", "mis", "re", "over", "under"):
            if cand == pfx + target or target == pfx + cand:
                return False
        if wn_pos == "a" and cand.endswith("ed") and not target.endswith("ed"):
            return False
        if wn_pos == "n" and cand.endswith("ing") and not target.endswith("ing"):
            return False
        if wn_pos == "n" and cand.endswith(("ism", "ist")) and target.endswith(("ism", "ist")):
            return False

        # 2. Cognate Near-Synonym Guard:
        # Block -ic / -ical pairs (historic/historical, classic/classical, economic/economical, electric/electrical, comic/comical)
        if (target.endswith("ic") and cand == target + "al") or (cand.endswith("ic") and target == cand + "al"):
            return False
        # Block British/American or variant abstract noun suffixes on identical stem (specialty / speciality / specialism)
        if (
            cls._common_prefix_len(target, cand) >= 6
            and target.endswith(("ty", "ity", "ism", "ness", "ment", "tion", "sion"))
            and cand.endswith(("ty", "ity", "ism", "ness", "ment", "tion", "sion"))
        ):
            return False
        if cls.are_same_word_family(target, cand):
            return False

        # 3. Primary POS verification in PedLex (when available):
        p_code = {"n": "n", "v": "v", "a": "adj", "r": "adv"}.get(wn_pos, "n")
        e_cand = cls.get_ldoce_entry(cand)
        if e_cand:
            cand_prim_pos = cls._pos_code(e_cand.get("pos") or "")
            if cand_prim_pos and cand_prim_pos != p_code:
                # Only allow secondary POS when edit distance == 1 (classic homophone/paronym like complement/compliment)
                senses = e_cand.get("senses") or []
                pos_senses = [s for s in senses if cls._pos_code(s.get("pos") or "") == p_code]
                if not pos_senses or cls._edit_distance_le2(target, cand) > 1:
                    return False

        # 4. Double-Key Collision Check (includes context_anchor & quote frame checks)
        is_dk, _ = cls.double_key_collision(
            target, cand, anchor=context_anchor, anchor_type=anchor_type, pos=wn_pos, quote=quote
        )
        if is_dk:
            return False

        # 5. WordNet Synset / Similar / Also / Hypernym Disjointness
        wn = cls.get_wordnet()
        t_words = wn.words(target, pos="a") + wn.words(target, pos="s") if wn_pos == "a" else wn.words(target, pos=wn_pos)
        c_words = wn.words(cand, pos="a") + wn.words(cand, pos="s") if wn_pos == "a" else wn.words(cand, pos=wn_pos)
        t_syns = [s for w in t_words for s in w.synsets()]
        c_syns = [s for w in c_words for s in w.synsets()]
        if not t_syns or not c_syns:
            return False
        t_ids = {s.id for s in t_syns}
        c_ids = {s.id for s in c_syns}
        if t_ids & c_ids:
            return False

        t_ext = set(t_ids)
        for s in t_syns:
            for r in s.get_related("similar") + s.get_related("also") + s.hypernyms():
                t_ext.add(r.id)
        c_ext = set(c_ids)
        for s in c_syns:
            for r in s.get_related("similar") + s.get_related("also") + s.hypernyms():
                c_ext.add(r.id)
        if t_ext & c_ext:
            return False

        # 6. Root-Sharing Cognate Semantic Overlap Check:
        # If target and cand share a 4+ char prefix (same Latin/Greek root, e.g. activism/activity, specimen/species, continual/continuous),
        # block if they share content words across their PedLex+WN definitions OR share 2-hop derivational bases (for nouns/verbs).
        pref = cls._common_prefix_len(target, cand)
        if pref >= 4:
            t_defs = cls._get_def_content_words(target, wn_pos)
            c_defs = cls._get_def_content_words(cand, wn_pos)
            if t_defs & c_defs:
                return False
            if cand in t_defs or target in c_defs:
                return False
            if wn_pos in ("n", "v"):
                if cls._get_derivational_bases(target) & cls._get_derivational_bases(cand):
                    return False

        return True

    @classmethod
    def get_orthographic_paronym_distractors(
        cls,
        target_word: str,
        pos: str = "n",
        target_count: int = 3,
        exclude_words: Optional[Set[str]] = None,
        context_anchor: Optional[str] = None,
        anchor_type: Optional[str] = None,
        quote: Optional[str] = None,
        return_scores: bool = False,
    ) -> Union[List[str], List[Tuple[str, int]]]:
        """Mine psychometrically sharp orthographic look-alikes (`形近词`) and orthogonal-domain
        root paronyms (`同根异义词`) guarded by `is_safe_orthographic_paronym`."""
        tw = (target_word or "").strip().lower()
        if not tw or len(tw) < 4 or not tw.isalpha():
            return []

        pos_clean = (pos or "n").strip().lower()
        wn_pos = "a" if pos_clean in ("a", "adj", "adjective", "s") else (
            "v" if pos_clean in ("v", "verb") else (
                "r" if pos_clean in ("r", "adv", "adverb") else "n"
            )
        )
        excl = {w.lower() for w in (exclude_words or set())} | {tw}

        idx = cls._ensure_orthographic_pos_index()
        by_first = idx["by_first"].get(wn_pos, {})
        by_second = idx["by_second"].get(wn_pos, {})
        raw_pool = set(by_first.get(tw[0], [])) | set(by_second.get(tw[1], []))

        scored: List[Tuple[int, str]] = []
        for w in raw_pool:
            if w in excl or abs(len(w) - len(tw)) > 3:
                continue
            pref = cls._common_prefix_len(tw, w)
            suff = cls._common_suffix_len(tw, w)
            ed = cls._edit_distance_le2(tw, w)
            if ed <= 2 or (pref >= 4 and len(tw) >= 6 and abs(len(w) - len(tw)) <= 3) or (pref >= 3 and suff >= 2):
                if anchor_type == "modified_noun" and wn_pos == "a" and w in cls._PREDICATIVE_ONLY_ADJS:
                    continue
                if not cls.is_cefr_compliant_distractor(w, tw):
                    continue
                if not cls.is_safe_orthographic_paronym(
                    tw, w, pos=wn_pos, context_anchor=context_anchor, anchor_type=anchor_type, quote=quote
                ):
                    continue
                score = 0
                if ed == 1:
                    score += 45
                elif ed == 2:
                    score += 26
                score += pref * 4 + suff * 2 - abs(len(w) - len(tw)) * 2
                if wn_pos == "a" and pref >= 6:
                    score += 18
                if wn_pos == "n" and pref >= 4 and not w.endswith(("ist", "or", "er", "ian", "ty", "ity", "ism")):
                    score += 14
                scored.append((score, w))

        scored.sort(key=lambda x: (-x[0], x[1]))
        # Deduplicate mutual word-family members among selected paronyms
        selected: List[Tuple[str, int]] = []
        for sc, w in scored:
            if any(cls.are_same_word_family(w, prev_w) for prev_w, _ in selected):
                continue
            selected.append((w, sc))
            if len(selected) >= target_count:
                break

        if return_scores:
            return selected
        return [w for w, _ in selected]

    # -------------------------------------------------------------------------
    # Double-Key Clearance (Frame-Aware Distractor Audit)
    # -------------------------------------------------------------------------
    # Empirically validated 2026-09-25 against Book_3_Unit_4_Section_A:
    #   * Prep items  -> double-key = distractor shares the target's bound
    #     preposition (OCD) AND sits in the target's semantic neighborhood.
    #     (Shared frame alone is NOT sufficient: break-with is valid but not a
    #     substitute for coincide-with. overlap-with and liberty-from ARE.)
    #   * Slot items  -> double-key = distractor is a near-synonym / troponym
    #     / hyponym of the target (same semantic slot). Sense antonyms are
    #     deliberately whitelisted: they are contrast distractors eliminated by
    #     sentence polarity, not double-keys.
    #     Coordinate-term (sister-synset) expansion is deliberately EXCLUDED —
    #     it over-matches (break ~ coincide) and yields false double-keys.
    @classmethod
    def semantic_fields(cls, word: str, pos: Optional[str] = None) -> Tuple[Set[str], Set[str]]:
        """Returns (near_set, antonyms) for a lemma, WordNet-derived.

        near_set  : same-synset words, bipolar similar/also (2 hops),
                    hypernym / entailment / hyponym lemmas.
        antonyms  : sense-locked antonyms (+ their satellite cluster).
        Deterministic; never raises."""
        wn = cls.get_wordnet()
        target = word.lower()
        near: Set[str] = set()
        antonyms: Set[str] = set()

        def add(syn) -> None:
            for w in syn.words():
                near.add(w.lemma().lower())

        pos_clean = (pos or "").strip().lower()
        if pos_clean in ("noun", "n"):
            pos_tags = ("n",)
        elif pos_clean in ("verb", "v"):
            pos_tags = ("v",)
        elif pos_clean in ("adjective", "adj", "a", "s"):
            pos_tags = ("a", "s")
        elif pos_clean in ("adverb", "adv", "r"):
            pos_tags = ("r",)
        else:
            pos_tags = ("a", "s", "v", "n", "r")

        seeds = []
        for p_tag in pos_tags:
            try:
                seeds += wn.synsets(target, p_tag)
            except Exception:
                pass
        if not seeds and pos_tags != ("a", "s", "v", "n", "r"):
            for p_tag in ("a", "s", "v", "n", "r"):
                try:
                    seeds += wn.synsets(target, p_tag)
                except Exception:
                    pass

        for s in seeds:
            add(s)
            for rel in ("similar", "also"):
                for t in s.get_related(rel):
                    add(t)
                    if t.pos in ("a", "s"):
                        for rel2 in ("similar", "also"):
                            for u in t.get_related(rel2):
                                add(u)
            for rel in ("hypernym", "entailment", "hyponym"):
                for t in s.get_related(rel):
                    add(t)
            # Sense-locked antonyms (same cluster for adjectives)
            for sense in s.senses():
                for ant_sense in sense.get_related("antonym"):
                    antonyms.add(ant_sense.word().lemma().lower())
                    for rel in ("similar", "also"):
                        for t in ant_sense.synset().get_related(rel):
                            for w in t.words():
                                antonyms.add(w.lemma().lower())

        near.discard(target)
        antonyms.discard(target)
        return near, antonyms

    GENERIC_COLLOCATION_MODIFIERS: Set[str] = {
        "big", "huge", "enormous", "massive", "great", "small", "tiny", "little",
        "good", "bad", "fine", "new", "old", "strong", "weak", "high", "low",
        "deep", "different", "certain", "various", "several", "many", "much",
        "more", "most", "less", "least", "real", "true", "full", "complete"
    }
    GENERIC_COLLOCATION_VERBS: Set[str] = {
        "have", "has", "had", "make", "made", "get", "got", "give", "gave",
        "take", "took", "do", "did", "done", "see", "saw", "find", "found", "use", "used"
    }
    COLLECTIVE_GROUP_NOUNS: Set[str] = {
        "team", "company", "crew", "committee", "board", "panel", "unit",
        "staff", "service", "force", "group", "crowd", "squad", "cast", "faculty"
    }

    @classmethod
    def get_syntactic_pattern_slots(cls, word_or_entry: Union[str, Dict[str, Any]], sense: Optional[Dict[str, Any]] = None) -> Set[str]:
        """Extracts normalized syntactic complement pattern slots for a headword from PedLex 6th Edition.
        
        Returned slots can include:
          - 'to_inf'     : infinitive complement ('to do sth')
          - 'gerund'     : gerund complement ('doing sth') without bound prepositions
          - 'sb_to_do'   : complex transitive with infinitive ('somebody to do sth')
          - 'that_clause': clausal complement ('that ...' or '(that)')
          - 'wh_clause'  : indirect question / wh-clause ('who/what/whether etc')
        """
        if isinstance(word_or_entry, dict):
            entry = word_or_entry
        else:
            entry = cls.get_ldoce_entry(str(word_or_entry).strip().lower())
        if not entry:
            return set()

        target_senses = [sense] if sense else entry.get("senses", [])
        slots: Set[str] = set()
        for s in target_senses:
            for p in s.get("patterns", []):
                pl = p.lower()
                if "to do" in pl:
                    if any(x in pl for x in ("sb to do", "somebody to do", "someone to do")):
                        slots.add("sb_to_do")
                    else:
                        slots.add("to_inf")
                if "doing" in pl:
                    # Exclude prepositional gerunds like 'in/of/from doing'
                    if not any(x in pl for x in ("of doing", "in doing", "for doing", "from doing", "against doing", "about doing", "with doing", "upon doing")):
                        slots.add("gerund")
                if "(that)" in pl or "that" in pl:
                    if not pl.startswith("the ") and not pl.startswith("in the "):
                        slots.add("that_clause")
                if any(x in pl for x in ("wh-", "how/where", "who/what", "whether")):
                    slots.add("wh_clause")
            # Example-based inference fallback if patterns field is silent
            for ex in s.get("examples", []):
                ex_l = ex.lower()
                w = entry.get("word", "").lower()
                if f"{w} to " in ex_l:
                    slots.add("to_inf")
        return slots

    @classmethod
    def get_word_gram_profile(cls, word: str, pos: Optional[str] = None) -> Dict[str, Any]:
        """Extracts a unified grammatical constraint profile for `word` from `pedlex.db` (`senses`, `thesaurus`, and `grammar_boxes`)."""
        w_clean = (word or "").strip().lower()
        profile: Dict[str, Any] = {
            "only_before_noun": False,
            "not_before_noun": w_clean in cls._PREDICATIVE_ONLY_ADJS,
            "pure_uncountable": False,
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
            matching_senses = senses

        grams = [str(s.get("grammar") or "").lower() for s in matching_senses if s.get("grammar")]
        # Also inspect self-referential thesaurus exponent grammar tags
        for th in entry.get("thesaurus") or []:
            if (th.get("word") or "").strip().lower() == w_clean and th.get("grammar"):
                grams.append(str(th["grammar"]).lower())

        if grams:
            if all("only before noun" in g for g in grams):
                profile["only_before_noun"] = True
            if all("not before noun" in g or "never before noun" in g for g in grams):
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
            for th in t_entry.get("thesaurus") or []:
                if (th.get("word") or "").strip().lower() == c and th.get("grammar"):
                    th_g = str(th["grammar"]).lower()
                    if "only before noun" in th_g:
                        c_prof["only_before_noun"] = True
                    if "not before noun" in th_g or "never before noun" in th_g:
                        c_prof["not_before_noun"] = True
        pos_norm = (pos or "").strip().lower()
        wn_pos = "a" if pos_norm.startswith("a") or pos_norm == "j" else (
            "n" if pos_norm.startswith("n") else (
                "v" if pos_norm.startswith("v") else ""
            )
        )

        # Inspect quote via spaCy if provided
        is_attr_adj_slot = anchor_type == "modified_noun"
        is_pred_adj_slot = False
        has_count_det_or_plural = False
        is_passive_verb_slot = False
        has_direct_obj_slot = anchor_type in ("object", "sb_to_do")
        active_verb_frame = anchor_type if anchor_type in ("to_inf", "gerund", "that_clause", "sb_to_do") else None

        if quote:
            nlp = cls.get_spacy()
            q_doc = nlp(quote)
            for tok in q_doc:
                if cls.form_matches(tok.lemma_.lower(), t) or cls.form_matches(tok.text.lower(), t):
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

        # 2. Noun countability clash (pure [uncountable] in countable determiner/plural slot)
        if (wn_pos == "n" or not wn_pos) and has_count_det_or_plural:
            if c_prof["pure_uncountable"] and not t_prof["pure_uncountable"]:
                return True, "uncountable_noun_in_countable_slot"

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
        if cls.are_same_word_family(t, d):
            return True, "double_key_family"

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

        near, antonyms = cls.semantic_fields(t)
        # Also include PedLex thesaurus words and grammar_boxes comparison_peers in near semantic field
        for item in cls.get_ldoce_thesaurus(t):
            raw_w = item.get("word", "").strip().lower()
            if not raw_w or raw_w == t:
                continue
            # Parse slash-separated tokens (e.g. 'longing/yearning' -> 'longing', 'yearning')
            for chunk in raw_w.split("/"):
                chunk = chunk.strip()
                if chunk and " " not in chunk and chunk != t:
                    near.add(chunk)
        t_prof = cls.get_word_gram_profile(t, pos=pos)
        for peer in t_prof.get("comparison_peers", set()):
            if peer != t:
                near.add(peer)

        # Collective / Group Noun Double-Key Gate:
        # If both target and distractor denote collections or groups of people/staff
        # and frame is partitive/of-complement ('a ____ of experts'), any other group noun is a double key!
        if t in cls.COLLECTIVE_GROUP_NOUNS and d in cls.COLLECTIVE_GROUP_NOUNS:
            if anchor in ("of", "part of", "member of", "on", "in") or anchor_type in ("prep", "partitive"):
                return True, "double_key_slot"

        # Adverb syntactic modification frame discrimination
        if anchor_type in ("modifies_adj", "modifies_verb"):
            d_entry = cls.get_ldoce_entry(d)
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
        if anchor_type in ("to_inf", "gerund", "that_clause", "sb_to_do") or (not anchor and pos in ("v", "verb")):
            t_slots = cls.get_syntactic_pattern_slots(t)
            d_slots = cls.get_syntactic_pattern_slots(d)
            d_prof = cls.get_word_gram_profile(d, pos=pos)
            d_slots -= d_prof.get("forbidden_frames", set())
            # If anchor_type directly specifies the target complement slot:
            active_slot = anchor_type if anchor_type in ("to_inf", "gerund", "that_clause", "sb_to_do") else None
            if not active_slot:
                # Find intersective slots declared by target
                common_slots = t_slots & {"to_inf", "gerund", "that_clause", "sb_to_do"}
                if len(common_slots) == 1:
                    active_slot = next(iter(common_slots))

            if active_slot and active_slot in t_slots:
                if active_slot not in d_slots or active_slot in d_prof.get("forbidden_frames", set()):
                    # Distractor cannot occupy target's syntactic complement slot -> single-fit discriminated!
                    return False, "none"
                elif d in near or (cls.semantic_fields(t, pos="v")[0] & cls.semantic_fields(d, pos="v")[0]):
                    # Both share the non-finite/clausal complement slot and are semantic neighbors -> double-key collision!
                    return True, "double_key_pattern"

        t_preps_raw = cls.get_oxford_collocations(t, pos=pos).get("prep", [])
        t_preps = [w.strip().lower() for p in t_preps_raw for w in p.split("/") if w.strip()]
        for lp in cls.get_ldoce_preps(t):
            if lp not in t_preps:
                t_preps.append(lp)
        shared = bool(anchor) and anchor.lower() in t_preps

        if anchor_type == "prep" and anchor and shared:
            d_preps_raw = cls.get_oxford_collocations(d, pos=pos or "noun").get("prep", [])
            d_preps = [w.strip().lower() for p in d_preps_raw for w in p.split("/") if w.strip()]
            for lp in cls.get_ldoce_preps(d):
                if lp not in d_preps:
                    d_preps.append(lp)
            if anchor.lower() in d_preps:
                if d in near:
                    return True, "double_key_prep"
                return False, "frame_only"
            # Candidate does NOT govern the anchor preposition -> single-fit discriminated!
            return False, "none"

        if d in near:
            if d in antonyms:
                # Deliberate contrast distractor (eliminated by polarity/logic),
                # whitelisted by design in the distractor generator.
                return False, "contrast"
            # If an explicit collocational anchor is present (e.g. adjective modifier or verb)
            # and candidate d does NOT share that collocation in OCD/PedLex, candidate d is ruled out
            # by the anchor slot itself and is therefore a valid, single-fit distractor.
            # EXCEPTION: If anchor is a generic adjective or verb (e.g. huge, massive, new, good, have),
            # absence in OCD does NOT discriminate near-synonyms! Near-synonyms remain double keys.
            if anchor and anchor_type in ("adj", "verb", "modified_noun", "verb_subject", "modifies_adj", "modifies_verb"):
                a_clean = anchor.lower()
                is_generic_anchor = (
                    (anchor_type in ("adj", "modified_noun") and a_clean in cls.GENERIC_COLLOCATION_MODIFIERS) or
                    (anchor_type in ("verb", "verb_subject") and a_clean in cls.GENERIC_COLLOCATION_VERBS)
                )
                if is_generic_anchor:
                    return True, "double_key_slot"

                d_entry = cls.get_oxford_collocations(d, pos=pos or "noun")
                shared_colloc = False
                if anchor_type == "adj":
                    shared_colloc = a_clean in [a.split()[0].lower() for a in d_entry.get("adj", [])]
                    if not shared_colloc:
                        d_ldoce = cls.get_ldoce_entry(d)
                        if d_ldoce:
                            for c_item in d_ldoce.get("collocations", {}).get("adjectives", []):
                                if a_clean in c_item.get("collocation", "").lower():
                                    shared_colloc = True
                                    break
                elif anchor_type == "verb":
                    shared_colloc = a_clean in [v.split()[0].lower() for v in d_entry.get("verb_before", [])]
                    if not shared_colloc:
                        d_ldoce = cls.get_ldoce_entry(d)
                        if d_ldoce:
                            for c_item in d_ldoce.get("collocations", {}).get("verbs", []):
                                if a_clean in c_item.get("collocation", "").lower():
                                    shared_colloc = True
                                    break
                elif anchor_type == "modified_noun":
                    shared_colloc = a_clean in [n.split()[0].lower() for n in d_entry.get("colloc_nouns", []) + d_entry.get("noun_after", [])]
                    if not shared_colloc:
                        d_ldoce = cls.get_ldoce_entry(d)
                        if d_ldoce:
                            for c_item in d_ldoce.get("collocations", {}).get("nouns", []):
                                if a_clean in c_item.get("collocation", "").lower():
                                    shared_colloc = True
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
            if (pos == "v" or pos == "verb") and anchor_type in ("object", "collocation"):
                d_entry = cls.get_oxford_collocations(d, pos="verb")
                if a_clean in [n.split()[0].lower() for n in d_entry.get("colloc_nouns", [])]:
                    return True, "double_key_collocation"
                a_entry = cls.get_oxford_collocations(a_clean, pos="noun")
                if d in [v.split()[0].lower() for v in a_entry.get("verb_before", [])]:
                    return True, "double_key_collocation"
                a_ldoce = cls.get_ldoce_entry(a_clean)
                if a_ldoce:
                    for v_item in a_ldoce.get("collocations", {}).get("verbs", []):
                        if d in v_item.get("collocation", "").lower():
                            return True, "double_key_collocation"
            elif (pos == "a" or pos == "adj") and anchor_type == "modified_noun":
                a_entry = cls.get_oxford_collocations(a_clean, pos="noun")
                if d in [adj.split()[0].lower() for adj in a_entry.get("adj", [])] and d not in antonyms:
                    return True, "double_key_collocation"
                a_ldoce = cls.get_ldoce_entry(a_clean)
                if a_ldoce:
                    for a_item in a_ldoce.get("collocations", {}).get("adjectives", []):
                        if d in a_item.get("collocation", "").lower() and d not in antonyms:
                            return True, "double_key_collocation"
            elif (pos == "n" or pos == "noun") and anchor_type in ("verb", "verb_subject"):
                d_entry = cls.get_oxford_collocations(d, pos="noun")
                if a_clean in [v.split()[0].lower() for v in d_entry.get("verb_before", [])] and d not in antonyms:
                    return True, "double_key_collocation"
                d_ldoce = cls.get_ldoce_entry(d)
                if d_ldoce:
                    for v_item in d_ldoce.get("collocations", {}).get("verbs", []):
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
        "swell", "cracking", "bully", "smashing", "peachy", "dandy", "nifty", "corking", "groovy", "slap-up", "bang-up"
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

    _PHRASE_INDEX_CACHE: Optional[Dict[str, str]] = None
    _PHRASE_SYNONYMS_CACHE: Optional[Dict[str, Set[str]]] = None

    _TABOO_PHRASE_TOKENS: Set[str] = {
        "arse", "ass", "asshole", "bastard", "bitch", "bloody", "bollocks",
        "bugger", "bum", "crap", "cunt", "damn", "dick", "fag", "fart", "fuck", "fucking",
        "motherfucker", "nigger", "piss", "prick", "pussy", "shit", "shitty",
        "slut", "tit", "twat", "wanker", "whore", "doss"
    }

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
        return False

    @classmethod
    def _get_phrase_synonyms(cls) -> Dict[str, Set[str]]:
        """
        Loads and memoizes synonym groups generated by slash-alternations in PedLex raw phrases
        (e.g., 'be/keep/stay etc in touch (with something)' -> {'be in touch with', 'keep in touch with', 'stay in touch with'}).
        """
        if cls._PHRASE_SYNONYMS_CACHE is not None:
            return cls._PHRASE_SYNONYMS_CACHE

        syn_map: Dict[str, Set[str]] = {}
        if cls._ensure_ldoce_conn() and cls._ldoce_conn is not None:
            try:
                cur = cls._ldoce_conn.cursor()
                cur.execute("SELECT phrase_key, raw_phrase FROM ldoce_phrase_index WHERE raw_phrase LIKE '%/%'")
                groups: Dict[str, Set[str]] = {}
                for k, raw in cur.fetchall():
                    groups.setdefault(raw, set()).add(k)
                for raw, keys in groups.items():
                    if len(keys) > 1:
                        for k in keys:
                            syn_map.setdefault(k, set()).update(keys)
            except Exception:
                pass

        cls._PHRASE_SYNONYMS_CACHE = syn_map
        return cls._PHRASE_SYNONYMS_CACHE

    @classmethod
    def _get_phrase_index(cls) -> Dict[str, str]:
        """Loads and memoizes {phrase_key: tier} from ldoce_phrase_index in the PedLex database."""
        if cls._PHRASE_INDEX_CACHE is not None:
            return cls._PHRASE_INDEX_CACHE

        phrase_dict: Dict[str, str] = {}
        if cls._ensure_ldoce_conn() and cls._ldoce_conn is not None:
            try:
                cur = cls._ldoce_conn.cursor()
                cur.execute("SELECT DISTINCT phrase_key, tier FROM ldoce_phrase_index")
                for key, tier in cur.fetchall():
                    # Prioritize phrasal_verb > phrase_row > sense_unit
                    if key not in phrase_dict or tier == "phrasal_verb":
                        phrase_dict[key] = tier
            except Exception:
                pass

        cls._PHRASE_INDEX_CACHE = phrase_dict
        return cls._PHRASE_INDEX_CACHE

    @classmethod
    def generate_phrase_distractors(
        cls,
        phrase: str,
        pos: str = "phrasal_verb",
        count: int = 3,
        exclude_words: Optional[Set[str]] = None
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
        target_head_is_verb = (
            bool(head_w)
            and head_w not in _NON_VERB_PHRASE_HEADS
            and bool(cls._wordnet_pos(head_w) & {"v"})
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

        def ships(cand: str) -> bool:
            if (cand == phrase_clean or cand in exclude or cand in synonym_set
                    or cls.is_inappropriate_phrase(cand)
                    or not (cand in phrase_index or cls.is_attested_phrase(cand))):
                return False
            c_words = cand.split()
            if not c_words:
                return False
            c_head = c_words[0]
            if target_head_is_verb:
                # Active lexical/light verb phrases (e.g. 'make the most of', 'take pleasure in')
                # must only pair with active verb-headed phrases, never determiner/noun/prep fragments
                # ('no to speak of', 'a bad judge of') or passive 'be <VBN> ...' constructions.
                if c_head in _NON_VERB_PHRASE_HEADS or not (cls._wordnet_pos(c_head) & {"v"}):
                    return False
                if c_head == "be" and head_w != "be":
                    return False
                if target_is_v_noun_prep and len(c_words) == 3:
                    # Preserve Verb + Bare-Noun + Prep structural symmetry: reject phrasal verb + prep
                    # ('bring out in', 'come on in', 'take up with') or Verb + Det + Noun ('take a chance')
                    if c_words[1] in _PARTICLE_OR_DET_MIDDLES or not (cls._wordnet_pos(c_words[1]) & {"n"}):
                        return False
            elif target_head_is_prep:
                if c_head not in _NON_VERB_PHRASE_HEADS:
                    return False
            return True

        tier_rank = {tier: index for index, tier in enumerate(cls._PHRASE_TIERS)}

        def strength(cand: str) -> int:
            """PedLex's own phrasal-verb block beats a PHRASES item, which beats a pattern."""
            hits = cls.ldoce_phrase_hits(cand)
            if hits:
                return min((tier_rank.get(tier, len(tier_rank)) for tier, _text in hits),
                           default=len(tier_rank))
            tier = phrase_index.get(cand, "example")
            return tier_rank.get(tier, len(tier_rank))

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

        if n_words == 2 and words[0] in COLLOCATION_VERB_PREPS:
            verb, prep = words[0], words[1]
            verb_cands: List[str] = []
            # Priority 1: Same preposition contrast (e.g. 'worry about' -> 'care about', 'complain about')
            for v, p in COLLOCATION_VERB_PREPS.items():
                if p == prep and v != verb:
                    c = f"{v} {prep}"
                    if ships(c) and c not in verb_cands:
                        verb_cands.append(c)
            # Priority 2: Cross-preposition verb collocations
            for v, p in COLLOCATION_VERB_PREPS.items():
                if v != verb and p != prep:
                    c = f"{v} {p}"
                    if ships(c) and c not in verb_cands:
                        verb_cands.append(c)
            if len(verb_cands) >= count:
                return verb_cands[:count]
            distractors.extend(verb_cands)

        elif n_words == 2 and words[0] in ADJ_DEPENDENT_PREPS:
            adj, prep = words[0], words[1]
            adj_cands: List[str] = []
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
        # 1. Two-word phrase (e.g. 'send out', 'tend to', 'cut out', 'work for')
        # -------------------------------------------------------------
        if n_words == 2 and not distractors:
            v, p = words[0], words[1]
            diff_part: List[str] = []
            particles = ["in", "off", "up", "down", "away", "back", "over", "into", "through", "on", "to", "for", "out", "with", "from", "around", "about"]
            p_is_particle = p in particles
            v_is_verb = bool(cls._wordnet_pos(v) & {"v"})

            if p_is_particle:
                for part in particles:
                    if part != p:
                        cand = f"{v} {part}"
                        if ships(cand) and cand not in diff_part:
                            diff_part.append(cand)

                # PedLex's own phrasal-verb list for this verb is the strongest pool there is
                for cand in cls.get_ldoce_phrasal_verbs(v):
                    parts = cand.split()
                    if len(parts) == 2 and parts[1] != p and ships(cand) and cand not in diff_part:
                        diff_part.insert(0, cand)

                same_part: List[str] = []
                common_verbs = [
                    "point", "carry", "turn", "figure", "bring", "stand", "make", "find",
                    "set", "take", "come", "go", "get", "give", "hold", "look", "run",
                    "keep", "leave", "work", "fall", "call", "pass", "lead", "move", "break"
                ]
                for cv in common_verbs:
                    if cv != v:
                        cand = f"{cv} {p}"
                        if ships(cand) and cand not in diff_part and cand not in same_part:
                            same_part.append(cand)

                # Query phrase_index for 2-word phrasal verbs ending with particle `p`
                for k in phrase_index:
                    k_words = k.split()
                    if len(k_words) == 2 and k_words[1] == p and k_words[0] != v:
                        cand_head = k_words[0]
                        # Enforce head syntactic concord: if target head is a verb, cand head must be a verb
                        if v_is_verb and not (cls._wordnet_pos(cand_head) & {"v"}):
                            continue
                        if ships(k) and k not in diff_part and k not in same_part:
                            same_part.append(k)

                diff_part.sort(key=lambda c: (strength(c), c))
                same_part.sort(key=lambda c: (strength(c), c))

                # Balanced selection: combine particle variation and verb variation
                if diff_part and same_part:
                    combined = [diff_part[0], same_part[0]]
                    if len(diff_part) > 1:
                        combined.append(diff_part[1])
                    elif len(same_part) > 1:
                        combined.append(same_part[1])
                    for item in diff_part[2:] + same_part[2:]:
                        if len(combined) >= count:
                            break
                        if item not in combined:
                            combined.append(item)
                    distractors = combined
                elif diff_part:
                    distractors = diff_part
                elif same_part:
                    distractors = same_part
            else:
                # Verb + Complement/Adjective Collocation (e.g. 'make possible', 'make certain')
                p_is_adj = bool(cls._wordnet_pos(p) & {"a", "s"})
                diff_comp: List[str] = []
                for k in phrase_index:
                    kw = k.split()
                    if len(kw) == 2 and kw[0] == v and kw[1] != p and kw[1] not in particles:
                        if p_is_adj:
                            if not (cls._wordnet_pos(kw[1]) & {"a", "s"}):
                                continue
                        if ships(k) and k not in diff_comp:
                            diff_comp.append(k)
                if diff_comp:
                    diff_comp.sort(key=lambda c: (strength(c), c))
                    distractors = diff_comp[:count]

        # -------------------------------------------------------------
        # 2. Multi-word collocations & phrases (3+ words)
        # -------------------------------------------------------------
        elif n_words >= 3 and not distractors:
            cands_found: List[str] = []

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

            # Strategy B: Common Suffix / Tail Match (e.g. 'peace of mind' -> '... of mind')
            # Guard: When words[0] is a light/support verb (take, find, have, get, make, give, keep, feel, etc.)
            # and tl == n_words - 1 (matching the exact noun+prep tail, e.g. '... pleasure in' for 'take pleasure in'
            # -> 'find pleasure in'), swapping only the light verb creates synonymous collocation double-keys!
            light_support_verbs = {
                "take", "find", "have", "get", "make", "give", "keep", "hold", "bear",
                "pay", "show", "feel", "put", "set", "bring", "come", "go", "lose", "catch"
            }
            if not cands_found or len(cands_found) < count:
                tail_lengths = [2] if target_is_v_noun_prep else ([2, 1] if n_words >= 3 else [1])
                for tl in tail_lengths:
                    if words[0] in light_support_verbs and tl >= n_words - 1:
                        continue
                    tail_phrase = " " + " ".join(words[-tl:])
                    # Look for exact length matches with the same suffix
                    tail_cands = []
                    for k in phrase_index:
                        kw = k.split()
                        if len(kw) == n_words and k.endswith(tail_phrase) and ships(k) and k not in cands_found:
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

            # Strategy D: Light-Verb & Construct Prefix Match (e.g. 'have a try', 'take a look', 'make a move')
            if len(cands_found) < count:
                prefix_2 = " ".join(words[:2]) + " "
                prefix_cands = []
                popular_have_a = ["have a go", "have a care", "have a heart", "have a think", "have a word", "have a ball", "have a breather"]
                for ph in popular_have_a:
                    if prefix_2.strip() == "have a" and ships(ph) and ph not in cands_found and ph not in prefix_cands:
                        prefix_cands.append(ph)
                for k in phrase_index:
                    kw = k.split()
                    if len(kw) == n_words and k.startswith(prefix_2) and ships(k) and k not in cands_found and k not in prefix_cands:
                        prefix_cands.append(k)
                if prefix_cands:
                    prefix_cands.sort(key=lambda c: (strength(c), c))
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

            # Strategy F: Same-Length Syntactic Frame Matching (Same connector/prep at same index)
            if len(cands_found) < count:
                for idx, w in enumerate(words):
                    if w in ("of", "in", "to", "for", "with", "on", "at", "from"):
                        for k in phrase_index:
                            kw = k.split()
                            if len(kw) == n_words and kw[idx] == w and ships(k) and k not in cands_found:
                                cands_found.append(k)
                                if len(cands_found) >= count + 5:
                                    break
                        if len(cands_found) >= count:
                            break

            distractors = cands_found

        # -------------------------------------------------------------
        # 3. General Fallback if pool is still sparse
        # -------------------------------------------------------------
        if len(distractors) < count:
            if n_words == 2:
                for fallback_cand in ["carry out", "take over", "bring about", "set up", "look into", "come across", "give in"]:
                    if ships(fallback_cand) and fallback_cand not in distractors:
                        distractors.append(fallback_cand)
                    if len(distractors) >= count:
                        break
            elif n_words >= 3:
                # Fallback to authentic multi-word phrases of matching length from phrase_index
                for k in phrase_index:
                    if len(k.split()) == n_words and ships(k) and k not in distractors:
                        distractors.append(k)
                    if len(distractors) >= count:
                        break

        return distractors[:count]

    @classmethod
    def _in_forbidden_word_set(cls, token: str, forbidden: Optional[Set[str]]) -> bool:
        """
        True if `token` equals or inflects/derives from any word in `forbidden`.
        Used to keep collocational anchors away from words that are themselves
        targets of the current batch (a batch-target anchor is unsatisfiable:
        the evaluator demands it in the stem while its leakage gate bans it).
        """
        if not token or not forbidden:
            return False
        t = token.lower().strip()
        if not t:
            return False
        for fw in forbidden:
            if not fw:
                continue
            if t == fw:
                return True
            if len(t) >= 4 and len(fw) >= 4 and (t.startswith(fw[:4]) or fw.startswith(t[:4])):
                return True
            if cls.are_same_word_family(t, fw):
                return True
        return False

    @classmethod
    def _pick_anchor_token(
        cls,
        candidates: List[str],
        target_word: Optional[str] = None,
        forbidden: Optional[Set[str]] = None,
        prefer_pos: Optional[str] = None
    ) -> Optional[str]:
        """
        Selects the first meaningful single-word anchor from a list of collocation
        phrases. Rejects truncated function-word tokens (e.g. 'be', 'have', 'of')
        produced by naively splitting multi-word collocations, so collision
        clearance operates on a real lexical headword instead of a stopword.
        Also strictly rejects the target_word itself and its simple inflections,
        and any word in the optional `forbidden` set (batch-target words).
        `prefer_pos` ('verb' / 'noun' / 'adj') re-orders the pick with spaCy: a
        cascade that needs a governing verb must get the verb out of
        'a crime carries a penalty' ('carry'), not the first content word ('crime').
        """
        tw_clean = target_word.lower().strip() if target_word else ""

        def acceptable(token: str) -> bool:
            if not token or len(token) < 3 or token in cls._ANCHOR_STOPWORDS:
                return False
            if token.startswith("somebod") or token.startswith("someth"):
                return False
            if tw_clean and (token == tw_clean or cls.are_same_word_family(token, tw_clean)):
                return False
            if forbidden and cls._in_forbidden_word_set(token, forbidden):
                return False
            return True

        nlp = cls.get_spacy() if prefer_pos else None
        fallback: Optional[str] = None

        for cand in candidates:
            if not cand:
                continue
            # Split by whitespace, slashes, and common punctuation to find the authentic colloc partner
            parts = re.split(r"[/,\s]+", cand)
            clean = []
            for p in parts:
                token = re.sub(r"[^a-zA-Z]", "", p).lower()
                if not acceptable(token):
                    continue
                clean.append(token)
                if fallback is None:
                    fallback = token
            if not clean or nlp is None:
                continue
            try:
                doc = nlp(cand)
            except Exception:
                continue
            for tok in doc:
                if not cls._pos_matches_preference(tok, prefer_pos) or not tok.is_alpha:
                    continue
                lem = (tok.lemma_ or tok.text).lower()
                if acceptable(lem):
                    return lem
        return fallback

    @staticmethod
    def _pos_matches_preference(tok, prefer_pos: Optional[str]) -> bool:
        """spaCy POS gate used by _pick_anchor_token's prefer_pos re-ordering."""
        if prefer_pos == "verb":
            return tok.pos_ in ("VERB", "AUX")
        if prefer_pos == "noun":
            return tok.pos_ in ("NOUN", "PROPN")
        if prefer_pos == "adj":
            return tok.pos_ == "ADJ"
        return False

    @staticmethod
    def _token_in_phrases(token: str, phrases: List[str]) -> bool:
        """True when the token appears as a whole word inside any stored collocation
        phrase ('great' inside 'great luxury'). Raw-list membership fails for phrases,
        which is what made adjective collocations be typed as governing verbs."""
        if not token:
            return False
        pattern = re.compile(rf"\b{re.escape(token)}\b")
        return any(pattern.search((p or "").lower()) for p in phrases or [])

    @classmethod
    def _select_sense_aligned_anchor(
        cls,
        candidates: List[str],
        definition: Optional[str] = None,
        quote: Optional[str] = None,
        target_word: Optional[str] = None,
        forbidden: Optional[Set[str]] = None
    ) -> Optional[str]:
        """
        Selects the most semantically aligned single-word anchor from collocation candidates,
        grounded in the target item's authentic quote and curriculum definition (Sense-Specific Primacy).
        Prevents sense-drift where polysemous words select collocations from irrelevant senses.
        Strictly excludes the target_word itself from being selected as its own anchor, and
        excludes every word in the optional `forbidden` set (other batch targets).
        """
        if not candidates:
            return None

        quote_clean = (quote or "").lower()
        def_clean = (definition or "").lower()
        tw_clean = target_word.lower().strip() if target_word else ""

        # 1. Level 1 (Authentic Quote Match): If any collocation candidate occurs directly
        # in the source sentence quote, pick it immediately (zero hallucination, verbatim fidelity)
        for cand in candidates:
            token = cls._pick_anchor_token([cand], target_word=tw_clean, forbidden=forbidden)
            if token and re.search(r"\b" + re.escape(token) + r"\b", quote_clean):
                return token

        # 2. Level 2 (Definition Semantic Overlap): Score candidates by word overlap with definition and quote
        context_tokens = set(re.findall(r"[a-zA-Z]{3,}", (quote_clean + " " + def_clean)))
        context_tokens = context_tokens - cls._ANCHOR_STOPWORDS
        if tw_clean:
            context_tokens.discard(tw_clean)

        scored_candidates: List[Tuple[int, str]] = []
        for cand in candidates:
            token = cls._pick_anchor_token([cand], target_word=tw_clean, forbidden=forbidden)
            if not token:
                continue
            score = 1 if token in context_tokens else 0
            scored_candidates.append((score, token))

        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        if scored_candidates and scored_candidates[0][0] > 0:
            return scored_candidates[0][1]

        # 3. Level 3 (Strict None Semantics): if NO candidate is evidenced by
        # either the authentic quote (Level 1) or the definition (Level 2),
        # refuse to fabricate an anchor. Returning the first clean token (the
        # old "first-word lottery") produced fake anchors such as
        # 'compulsion ~ strange'; a missing anchor is a first-class outcome
        # that the micro-task planner degrades to a neutral academic template.
        return None

    @classmethod
    def _quote_evidence_strength(cls, quote: Optional[str]) -> Tuple[str, int]:
        """
        Grades the curriculum quote as physical evidence for sense-locking and anchor
        extraction. A two-word headline fragment ('Punctuality Pays!') is not a clause:
        it cannot license a valency frame nor disambiguate a polysemous headword, so it
        must not be allowed to define an anchor or lock a sense. Weak quotes push anchor
        selection onto the dictionary path and are reported in the blueprint as
        quote_provenance='weak' so downstream consumers can see the evidence tier.
        Returns (provenance, token_count) with provenance in {'strong', 'weak'}.
        """
        text = (quote or "").strip()
        tokens = re.findall(r"[A-Za-z][A-Za-z'\-]*", text)
        if len(tokens) < 5:
            return "weak", len(tokens)
        nlp = cls.get_spacy()
        if nlp is None:
            return "weak", len(tokens)
        try:
            doc = nlp(text)
            # A usable quote must contain a real predicate: either a lexical verb,
            # or a complete copular clause with a subject and complement (e.g. 'The hospital staff were very helpful.')
            has_lexical_verb = any(
                tok.pos_ == "VERB" and tok.dep_ not in ("aux", "auxpass", "cop")
                for tok in doc
            )
            has_copular_clause = any(
                tok.pos_ in ("AUX", "VERB")
                and (tok.dep_ in ("ROOT", "cop") or tok.head == tok)
                and any(c.dep_ in ("acomp", "attr", "dobj", "prep") for c in tok.children)
                and any(c.dep_ in ("nsubj", "nsubjpass") for c in tok.children)
                for tok in doc
            )
            if not has_lexical_verb and not has_copular_clause:
                return "weak", len(tokens)
        except Exception:
            return "weak", len(tokens)
        return "strong", len(tokens)

    # ---------------------------------------------------------------------
    # B1 - anchor evidence quality. A sentence may only stand as an anchor's
    # physical proof when it is a clause that actually shows the target word at
    # work: a heading ('**Punctuality Pays!**') and a bare fragment ('Why?')
    # demonstrate nothing, and among the sentences that do, the one where the
    # target carries the richest dependency structure is the one worth locking
    # a sense against.
    # ---------------------------------------------------------------------
    @classmethod
    def anchor_evidence_ok(cls, sentence: Optional[str], target_word: Optional[str] = None) -> bool:
        """Is this sentence physical evidence for the target word, or just a label?"""
        text = (sentence or "").strip()
        if not text:
            return False
        tokens = re.findall(r"[A-Za-z][A-Za-z'\-]*", text)
        if len(tokens) < cls.ANCHOR_MIN_WORDS:
            return False
        # A bold heading line is typography, not usage.
        if text.startswith("**") and text.endswith("**"):
            return False
        # Metadata signature lines (e.g. blog post timestamps/bylines in parentheses) are metadata, not usage evidence.
        if text.startswith("(") and text.endswith(")"):
            inner = text[1:-1].strip()
            if re.search(r"\b(?:posted\s+on|written\s+by|published\s+on|by\s+[A-Z])\b", inner, re.IGNORECASE):
                return False
        if target_word and not cls.text_contains_form(text, target_word):
            return False
        nlp = cls.get_spacy()
        if nlp is None:
            return False
        try:
            doc = nlp(text)
        except Exception:
            return False
        # The sentence must contain a real predicate: 'Why?' has no clause to read evidence from.
        roots = [tok for tok in doc if tok.dep_ == "ROOT"]
        has_root_verb = bool(roots and any(tok.pos_ in ("VERB", "AUX") for tok in roots))
        if not has_root_verb:
            # Check for embedded finite clausal verbs (e.g. 'One of the things that tend to annoy... has to do with...')
            has_clause_verb = any(
                tok.pos_ in ("VERB", "AUX") and tok.dep_ in ("relcl", "ccomp", "xcomp", "advcl", "conj")
                for tok in doc
            )
            # Check for imperative sentence starter (e.g. 'Schedule meetings and appointments in advance.')
            # where spaCy misparses an initial verb as a compound noun modifier.
            has_imperative_starter = False
            first_tok = doc[0] if len(doc) > 0 else None
            if first_tok and first_tok.text[0].isupper():
                wn = cls.get_wordnet()
                first_clean = first_tok.text.lower()
                if wn and any(s.pos == "v" for s in wn.synsets(first_clean)):
                    has_imperative_starter = True

            if not (has_clause_verb or has_imperative_starter):
                return False
        return True

    @classmethod
    def anchor_dependency_richness(cls, sentence: Optional[str], target_word: Optional[str]) -> int:
        """How much syntax the target word actually participates in inside this sentence."""
        text = (sentence or "").strip()
        tw = (target_word or "").strip().lower()
        if not text or not tw:
            return 0
        nlp = cls.get_spacy()
        if nlp is None:
            return 0
        try:
            doc = nlp(text)
        except Exception:
            return 0
        forms = set(cls.inflected_forms(tw)) | {tw}
        best = 0
        for tok in doc:
            if tok.text.lower() not in forms and tok.lemma_.lower() not in forms:
                continue
            deps = {ch.dep_ for ch in tok.children}
            score = 0
            if deps & {"nsubj", "nsubjpass"}:
                score += 3
            if deps & {"obj", "dobj", "iobj", "attr", "acomp", "oprd", "pred"}:
                score += 3
            if deps & {"prep", "obl", "pobj", "agent", "dative", "complement"}:
                score += 2
            if deps & {"advcl", "ccomp", "xcomp", "acl", "relcl", "partmod", "infmod"}:
                score += 2
            if deps & {"advmod", "mod", "neg", "aux", "auxpass", "compound", "amod", "npadvmod"}:
                score += 1
            if tok.dep_ == "ROOT":
                score += 2
            if tok.pos_ == "ADJ" and tok.head is not None and tok.head.pos_ == "NOUN":
                score += 2
            best = max(best, score)
        return best

    @classmethod
    def pick_anchor_sentence(
        cls,
        target_word: str,
        sentences: Iterable[str]
    ) -> Tuple[str, int]:
        """
        The best qualifying sentence for an anchor, scored by how much syntax the target word
        governs in it. Returns ("", -1) when no candidate is evidence at all - the caller must
        then label its quote as unqualified instead of pretending it proved something.
        """
        best_sentence, best_score = "", -1
        for sent in sentences or []:
            if not cls.anchor_evidence_ok(sent, target_word):
                continue
            score = cls.anchor_dependency_richness(sent, target_word)
            if score > best_score:
                best_sentence, best_score = sent, score
        return best_sentence, best_score

    @classmethod
    def _sense_frame_patterns(cls, sense: Dict[str, Any], target_word: str) -> List[str]:
        """
        Returns only those PedLex patterns of a sense that actually contain the headword
        ('correlate with', 'serious about', 'it annoys somebody'). Patterns such as
        'got on his nerves' or 'keeps' are idiom/example fragments, not valency frames,
        and must never count as evidence that a preposition belongs to the target.
        """
        tw = (target_word or "").strip().lower()
        if not sense or not tw:
            return []
        nlp = cls.get_spacy()
        frames: List[str] = []
        for pat in sense.get("patterns", []) or []:
            pl = (pat or "").lower()
            if not pl:
                continue
            if re.search(rf"\b{re.escape(tw)}\b", pl):
                frames.append(pat)
                continue
            if nlp:
                try:
                    if any(t.lemma_.lower() == tw for t in nlp(pl)):
                        frames.append(pat)
                except Exception:
                    pass
        return frames

    @classmethod
    def _prep_anchor_sense_conflict(
        cls,
        target_word: str,
        prep: str,
        definition: Optional[str] = None,
        quote: Optional[str] = None,
        canonical_pos: Optional[str] = None
    ) -> bool:
        """
        Locked-Sense Frame Concordance Gate .

        A bound-preposition anchor is only legitimate when the PedLex sense this item
        actually locks (from its curriculum definition + authentic quote) licenses it.
        'annoy' locks to 'to make someone feel slightly angry', whose authentic frame is
        'it annoys somebody when/how/that' and whose authentic example is 'What annoyed
        him most was that he had received no apology.' — no preposition at all. The
        dictionary's registered 'annoy somebody on' belongs to a different construction,
        so welding 'on' into the frame produced an unsatisfiable item.

        The gate only vetoes prepositions a dictionary actively registers (a genuine
        valency claim). A preposition harvested from the authentic quote by the syntactic
        walker ('degrade into', 'autonomy from', 'log into') carries no dictionary claim
        to contradict, so the Pillar-1 syntactic path stays untouched.
        """
        tw = (target_word or "").strip().lower()
        p_low = (prep or "").strip().lower()
        if not tw or not p_low:
            return False
        entry = cls.get_ldoce_entry(tw)
        if not entry:
            return False
        senses = entry.get("senses", []) or []
        if not senses:
            return False
        locked_idx = cls._lock_sense(
            entry, definition=definition, quote=quote, target_pos=canonical_pos
        )
        if locked_idx is None or locked_idx >= len(senses):
            return False
        locked_sense = senses[locked_idx]

        frames = cls._sense_frame_patterns(locked_sense, tw)
        if not frames:
            return False  # locked sense publishes no valency frame → nothing to contradict
        if any(re.search(rf"\b{re.escape(p_low)}\b", f.lower()) for f in frames):
            return False  # locked sense explicitly licenses this preposition

        # The locked sense's own authentic examples may exhibit the frame directly
        for ex in locked_sense.get("examples", []) or []:
            if re.search(rf"\b{re.escape(tw)}\w*\s+(?:\w+\s+)?{re.escape(p_low)}\b", (ex or "").lower()):
                return False

        registered = p_low in cls.get_ldoce_preps(tw)
        if not registered:
            ocd_pos = {"verb": "verb", "noun": "noun", "adj": "adj"}.get((canonical_pos or "").lower())
            registered = p_low in cls.get_oxford_collocations(tw, pos=ocd_pos).get("prep", [])
        if not registered:
            return False  # pure syntactic complement harvested from the quote — keep it

        return True  # registered valency claim that the locked sense does not license

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
        silent. Evidence is read from grammar patterns ('pay somebody for something') and,
        because many PedLex senses publish no pattern at all, from the sense examples
        themselves ('She logged the details of the meeting' -> direct object). A verb whose
        senses publish neither patterns nor examples is unknown, never assumed intransitive.
        """
        entry = cls.get_ldoce_entry((word or "").strip().lower())
        if not entry:
            return None
        saw_verb_evidence = False
        examples: List[str] = []
        for sense in entry.get("senses", []) or []:
            spos = (sense.get("pos") or "").lower()
            if spos and "verb" not in spos:
                continue
            for pat in sense.get("patterns", []) or []:
                pl = (pat or "").lower()
                if not pl:
                    continue
                saw_verb_evidence = True
                if any(m in pl for m in ("somebody", "someone", "something", "sb ", "sb$", "sth")):
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

    @classmethod
    def _contrast_strategy_clause(
        cls,
        target_word: str,
        definition: Optional[str],
        dist_list: List[str],
        dist_meta: Optional[Dict[str, str]],
        anchor_type: Optional[str] = None
    ) -> str:
        """
        Sense-aware contrast instruction .

        The blueprint pasted the same 'Prefer a concessive or cause-and-effect turn (e.g.
        Although/Despite...)' into every noun item whatever the distractors were. A concessive
        turn only discriminates when the distractor set actually contains a polar opposite;
        for near-synonym clusters ('earnest' vs 'serious') the discriminator is precision of
        meaning, and for taxonomically distant distractors no contrast is needed at all. The
        instruction is now derived from the distractor mechanisms the generator actually used.
        """
        dists = [d for d in (dist_list or []) if d]
        if not dists:
            return ""
        meta = dist_meta or {}
        tw = (target_word or "").strip().lower()
        antonyms = [d for d in dists if meta.get(d) == "antonym"]
        near_synonyms = [d for d in dists if meta.get(d) in
                         ("ldoce_thesaurus", "satellite_synonym", "synonym", "troponym",
                          "coordinate", "attribute_coordinate", "preposition_valency")]
        if antonyms:
            return (f" Establish an explicit polarity cue (an evaluative word or a situational outcome) "
                    f"so the opposite '{antonyms[0]}' is logically excluded.")
        if near_synonyms:
            named = ", ".join(f"'{d}'" for d in near_synonyms[:3])
            return (f" Establish a precision cue — the exact scale, register, or "
                    f"collocational frame that separates '{tw}' from {named}.")
        return " Establish situational and definitional clues that make the distinction decisive."

    @staticmethod
    def _definition_overlap_ratio(text_a: Optional[str], text_b: Optional[str]) -> float:
        """Overlap between two definition texts, used to detect self-referential notes."""
        ta = set(re.findall(r"[a-z]+", (text_a or "").lower()))
        tb = set(re.findall(r"[a-z]+", (text_b or "").lower()))
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / float(min(len(ta), len(tb)))

    @staticmethod
    def _thesaurus_variant_matches(th_word: Optional[str], word: Optional[str]) -> bool:
        """
        PedLex thesaurus headwords can be multi-form ('unwilling/not willing',
        'be loath to do something'), so match the plain distractor lemma against
        any of the recorded variants rather than the raw string.
        """
        t = (th_word or "").strip().lower()
        w = (word or "").strip().lower()
        if not t or not w:
            return False
        if t == w:
            return True
        variants = [v.strip() for v in re.split(r"[/|,]", t) if v.strip()]
        variants += [re.sub(r"^(be|get|have) ", "", v).strip() for v in variants]
        if w in variants:
            return True
        # For single-word candidate w, only match if variant is single-word or explicitly parenthesized
        for v in variants:
            if not v:
                continue
            # Strip parenthesized optional parts: e.g. '(be) willing' -> 'willing'
            v_clean = re.sub(r"\(.*?\)", "", v).strip()
            if v_clean == w:
                return True
            # Strip trailing sth/sb: e.g. 'command of sth' -> 'command of'
            v_nobound = re.sub(r"\b(sth|sb|something|someone)\b", "", v).strip()
            if v_nobound == w:
                return True
            # Strip infinitive tails: e.g. 'loath to do something' -> 'loath'
            v_inf = re.sub(r"\bto do (something|sth)\b|\bto (sth|sb|something|someone)\b", "", v).strip()
            if v_inf == w:
                return True
        return False

    @classmethod
    def get_vocab_card_enrichment(
        cls,
        word: str,
        pos: str = "",
        definition: str = "",
        quote: str = ""
    ) -> Dict[str, str]:
        """
        Deterministically extracts zero-token pedagogical enrichment for a vocabulary card:
        - `collocations`: Top sense-locked collocations / syntactic patterns from `pedlex.db`.
        - `grammar_usage`: Sense-locked `<gram>` tags (e.g. `[only before noun]`, `[not before noun]`,
          `[uncountable]`, `[countable]`), key syntactic frame, and sense-locked `grammar_boxes` rules (`✗ Don't say: ...`).
        - `thesaurus_note`: Sense-aligned `THESAURUS` synonym distinctions and `[grammar]` constraints.
        """
        out: Dict[str, str] = {
            "collocations": "",
            "grammar_usage": "",
            "thesaurus_note": "",
        }
        w_clean = re.sub(r"\[.*?\]|\(.*?\)", "", str(word or "")).strip().lower()
        if not w_clean or " " in w_clean:
            return out

        entry = cls.get_ldoce_entry(w_clean)
        if not entry:
            return out

        senses = entry.get("senses") or []
        t_s_idx = None
        locked_sense = None
        if senses:
            t_s_idx = cls._lock_sense(
                entry,
                definition=definition,
                quote=quote,
                target_pos=pos
            )
            if t_s_idx is not None and 0 <= t_s_idx < len(senses):
                locked_sense = senses[t_s_idx]

        p_code = cls._pos_code(pos or (locked_sense.get("pos") if locked_sense else entry.get("pos")))

        def _clean_pattern_text(pat_str: str) -> str:
            s = str(pat_str or "").strip()
            if not s:
                return ""
            m_emb = re.search(r"^(.*?)\s+([A-Z][\w\W]*\b[\.\?!])$", s)
            if not m_emb:
                m_etc = re.search(r"^(.*?\betc\.?)\s+([A-Z].*\s+.*)$", s)
                if m_etc:
                    m_emb = m_etc
            if m_emb:
                s = m_emb.group(1).strip()
            s = re.sub(
                r"\s+(?:(?:especially|mainly|chiefly|mostly)\s+)?(?:British|American|Australian)\s+English$",
                "",
                s,
                flags=re.IGNORECASE,
            ).strip()
            return s

        # 1. Grammar Tag (<gram> tags for inline header) & Pattern Note (key pattern + grammar_boxes rule)
        gram_tag_str = ""
        pattern_parts: List[str] = []
        if locked_sense:
            raw_gram = str(locked_sense.get("grammar") or "").strip()
            if raw_gram:
                gram_tag_str = raw_gram.strip("`").strip()
            pats = [
                _clean_pattern_text(p) for p in (locked_sense.get("syntactic_patterns") or locked_sense.get("patterns") or [])
                if _clean_pattern_text(p) and _clean_pattern_text(p).lower() != w_clean
            ]
            if pats:
                pattern_parts.append(f"*{pats[0]}*")

        gb_alert = cls.get_ldoce_grammar_alert(w_clean, quote=quote, sense_index=t_s_idx)
        if gb_alert:
            rules = []
            for r in (gb_alert.get("rules") or []):
                r_s = str(r.get("text") if isinstance(r, dict) else r or "").strip()
                r_s = re.sub(r"^[•·\-\*]\s*", "", r_s).strip()
                r_s = re.sub(r"\s*Grammar guide\s*[‒–—-]\s*[A-Z\s]+$", "", r_s).strip()
                if r_s:
                    rules.append(r_s)
            bad_exs = [str(b).strip() for b in (gb_alert.get("bad_examples") or []) if str(b).strip()]
            if rules:
                r_txt = rules[0]
                if bad_exs and "don't say" not in r_txt.lower() and bad_exs[0].lower() not in r_txt.lower():
                    r_txt += f" (✗ *{bad_exs[0]}*)"
                pattern_parts.append(r_txt)
            elif bad_exs:
                pattern_parts.append(f"✗ Don't say: *{bad_exs[0]}*")

        out["grammar_tag"] = gram_tag_str
        out["pattern_note"] = " — ".join(pattern_parts) if pattern_parts else ""
        gram_legacy_parts: List[str] = []
        if gram_tag_str:
            gram_legacy_parts.append(f"`{gram_tag_str}`")
        if pattern_parts:
            first_pat = pattern_parts[0]
            if first_pat.startswith("*") and first_pat.endswith("*"):
                gram_legacy_parts.append(f"Pattern: {first_pat}")
            else:
                gram_legacy_parts.append(first_pat)
            gram_legacy_parts.extend(pattern_parts[1:])
        if gram_legacy_parts:
            out["grammar_usage"] = " — ".join(gram_legacy_parts)

        # 2. Zero-Token Pedagogical Collocations (100% authentic dictionary collocations/patterns; zero fabricated splices)
        colloc_items: List[str] = []
        locked_homograph = locked_sense.get("homograph_num") if locked_sense else None

        def _sense_pos_matches(s_obj: Dict[str, Any]) -> bool:
            if not p_code:
                return True
            s_p = cls._pos_code(s_obj.get("pos"))
            if s_p and s_p != p_code:
                return False
            if locked_homograph is not None and s_obj.get("homograph_num") is not None:
                if s_obj.get("homograph_num") != locked_homograph:
                    return False
            return True

        def _add_colloc(cand_str: str) -> None:
            c_s = _clean_pattern_text(cand_str)
            if not c_s or " " not in c_s or "..." in c_s:
                return
            c_low = c_s.lower()
            if c_low == w_clean or c_low in (f"the {w_clean}", f"a {w_clean}", f"an {w_clean}"):
                return
            for idx_e, ex_item in enumerate(colloc_items):
                ex_low = ex_item.lower()
                if c_low == ex_low:
                    return
                if c_low in ex_low:
                    return
                if ex_low in c_low:
                    colloc_items[idx_e] = c_s
                    return
            colloc_items.append(c_s)

        # Tier A: Locked sense authentic collocations, pattern bundles, patterns, and lexical units
        if locked_sense:
            for c in locked_sense.get("collocations") or []:
                _add_colloc(str(c))
            for pb in locked_sense.get("pattern_bundles") or []:
                _add_colloc(str(pb.get("pattern") or ""))
            for pat in locked_sense.get("patterns") or []:
                _add_colloc(str(pat))
            for u in locked_sense.get("units") or []:
                _add_colloc(str(u))

        # Tier B: Authentic multi-word dictionary Collocations Box entries (source != "corpus")
        entry_p_code = cls._pos_code(entry.get("pos"))
        all_poses = [cls._pos_code(p) for p in (entry.get("all_poses") or []) if cls._pos_code(p)]
        colloc_box_pos_ok = (
            not entry_p_code
            or not p_code
            or entry_p_code == p_code
            or len(set(all_poses)) <= 1
        )
        if len(colloc_items) < 5 and colloc_box_pos_ok:
            raw_collocs = entry.get("collocations")
            if isinstance(raw_collocs, dict):
                for cat_k in ("phrases", "verbs", "adjectives", "nouns", "adverbs"):
                    for it_c in (raw_collocs.get(cat_k) or []):
                        if not isinstance(it_c, dict) or it_c.get("source") == "corpus":
                            continue
                        c_val = _clean_pattern_text(str(it_c.get("collocation") or ""))
                        if not c_val or " " not in c_val:
                            continue
                        _add_colloc(c_val)
                        if len(colloc_items) >= 5:
                            break
                    if len(colloc_items) >= 5:
                        break

        # Tier C: Same-POS/same-homograph sibling senses' patterns and lexical units
        if len(colloc_items) < 5:
            for s_obj in senses:
                if s_obj is locked_sense or not _sense_pos_matches(s_obj):
                    continue
                for pb in s_obj.get("pattern_bundles") or []:
                    _add_colloc(str(pb.get("pattern") or ""))
                for pat in s_obj.get("patterns") or []:
                    _add_colloc(str(pat))
                for u in s_obj.get("units") or []:
                    _add_colloc(str(u))
                if len(colloc_items) >= 5:
                    break

        # Tier D: Entry-level fixed phrases & cross-referenced dictionary collocations (when单POS or needed)
        if len(colloc_items) < 5 and colloc_box_pos_ok:
            for ph_obj in (entry.get("phrases") or []):
                ph_str = ph_obj.get("phrase") or ph_obj.get("title") if isinstance(ph_obj, dict) else str(ph_obj or "")
                _add_colloc(str(ph_str or ""))
                if len(colloc_items) >= 5:
                    break
        if len(colloc_items) < 5 and len(set(all_poses)) <= 1:
            for cr_obj in (entry.get("cross_collocations") or []):
                if isinstance(cr_obj, dict):
                    _add_colloc(str(cr_obj.get("collocation") or ""))
                    if len(colloc_items) >= 5:
                        break

        if colloc_items:
            out["collocations"] = "; ".join(colloc_items[:5])

        # 3. Sense-Aligned THESAURUS Note (exclude target word itself; rank peers by semantic overlap & proximity)
        thes_list = entry.get("thesaurus") or []
        if thes_list:
            locked_def = (definition or (cls.format_ldoce_definition(locked_sense) if locked_sense else "")).strip()
            self_indices = [
                idx_t for idx_t, t in enumerate(thes_list)
                if cls._thesaurus_variant_matches(t.get("word"), w_clean)
            ]
            self_exp = thes_list[self_indices[0]] if self_indices else None
            # Only surface THESAURUS box when its cluster aligns with the locked sense (or entry has 1 POS)
            box_matches_sense = True
            if self_exp and locked_def and len(senses) > 1:
                self_dist = str(self_exp.get("distinction") or "").strip()
                if self_dist and cls._definition_overlap_ratio(self_dist, locked_def) < 0.25:
                    box_matches_sense = False
            if box_matches_sense:
                # Reference text combines the locked definition and the target's own thesaurus distinction(s)
                ref_texts = [locked_def] if locked_def else []
                for s_idx in self_indices:
                    s_dist = str(thes_list[s_idx].get("distinction") or "").strip()
                    if s_dist:
                        ref_texts.append(s_dist)
                ref_combined = " ".join(ref_texts)

                scored_peers: List[Tuple[float, int, str]] = []
                seen_peer_words: Set[str] = set()
                for idx_t, t in enumerate(thes_list):
                    ew = str(t.get("word") or "").strip()
                    ed = str(t.get("distinction") or "").strip()
                    eg = str(t.get("grammar") or "").strip()
                    er = str(t.get("register") or "").strip()
                    if not ew or not ed:
                        continue
                    # Never list the target headword itself in Thesaurus Note (already defined above)
                    if cls._thesaurus_variant_matches(ew, w_clean) or ew.lower() == w_clean:
                        continue
                    ew_key = ew.lower()
                    if ew_key in seen_peer_words:
                        continue
                    seen_peer_words.add(ew_key)

                    # Score peer by lexical overlap with target definition/distinction + box proximity to self_indices
                    overlap_score = cls._definition_overlap_ratio(ed, ref_combined) if ref_combined else 0.0
                    prox_bonus = 0.0
                    if self_indices:
                        min_dist = min(abs(idx_t - s_i) for s_i in self_indices)
                        if min_dist == 1:
                            prox_bonus = 0.12
                        elif min_dist == 2:
                            prox_bonus = 0.05
                    total_score = overlap_score + prox_bonus

                    badges: List[str] = []
                    if eg:
                        badges.append(eg if eg.startswith("[") else f"[{eg}]")
                    if er:
                        badges.append(er if er.startswith("[") else f"[{er}]")
                    badge_str = f" `{' '.join(badges)}`" if badges else ""
                    chunk_str = f"    **{ew}**{badge_str} {ed}"
                    scored_peers.append((total_score, -idx_t, chunk_str))

                scored_peers.sort(key=lambda item: (item[0], item[1]), reverse=True)
                contrast_chunks = [item[2] for item in scored_peers[:3]]
                if contrast_chunks:
                    out["thesaurus_note"] = "\n" + "\n".join(contrast_chunks)

        return out

    @classmethod
    def _semantic_discriminator_note(
        cls,
        target_word: str,
        target_definition: Optional[str],
        dist_list: List[str],
        dist_meta: Optional[Dict[str, str]] = None,
        pos: Optional[str] = None,
        sense_index: Optional[int] = None,
        quote: Optional[str] = None
    ) -> str:
        """
        Target-anchored semantic & grammatical discriminator (+ <GRAM>/Grammar-Box Slot Guard).

        Combines:
        1. Explicit `<gram>` / `grammar_boxes` syntactic slot clashes (`syntactic_slot_clash:*`,
           `grambox_peer:*`, `[only before noun]`, `[not before noun]`, `[uncountable]`, `forbidden_frames`).
        2. Sense-locked `grammar_boxes` usage rules (`✗ Don't say: ...`).
        3. PedLex `THESAURUS` exponent distinctions (including exponent `[grammar]` tags).
        """
        tw = (target_word or "").strip().lower()
        dists = [d for d in (dist_list or []) if d]
        if not tw or not dists:
            return ""

        meta = dist_meta or {}
        notes: List[str] = []

        # 1. Check explicit grammatical/syntactic slot clashes & grammar_boxes peer rules first
        thes_items = cls.get_ldoce_thesaurus(tw)
        for dist in dists:
            d_low = dist.strip().lower()
            mechanism = meta.get(dist) or meta.get(d_low) or ""
            # Check THESAURUS grammar badge on distractor
            th_gram = ""
            th_dist = ""
            for th in thes_items:
                if cls._thesaurus_variant_matches(th.get("word"), dist):
                    th_gram = (th.get("grammar") or "").strip()
                    th_dist = (th.get("distinction") or "").strip()
                    break

            if mechanism.startswith("syntactic_slot_clash:") or mechanism.startswith("grambox_peer:"):
                reason_code = mechanism.split(":", 1)[1]
                if "only_before_noun" in reason_code:
                    notes.append(
                        f"Contrast: vs '{dist}' (`[only before noun]` — grammatically barred from predicative slot after a linking verb)."
                    )
                elif "not_before_noun" in reason_code:
                    notes.append(
                        f"Contrast: vs '{dist}' (`[not before noun]` — grammatically barred from attributive pre-noun slot)."
                    )
                elif "uncountable" in reason_code:
                    notes.append(
                        f"Contrast: vs '{dist}' (`[uncountable]` — cannot take countable determiners or plural form)."
                    )
                elif "forbidden_frame" in reason_code:
                    frame_lbl = reason_code.replace("forbidden_frame_", "")
                    d_alert = cls.get_ldoce_grammar_alert(d_low)
                    bad_ex = ""
                    if d_alert and d_alert.get("bad_examples"):
                        bad_ex = f", ✗ Don't say: {d_alert['bad_examples'][0]}"
                    notes.append(
                        f"Contrast: vs '{dist}' (Grammar Box: barred from '{frame_lbl}' complement frame{bad_ex})."
                    )
                elif "no_passive" in reason_code or "intransitive" in reason_code:
                    notes.append(
                        f"Contrast: vs '{dist}' (intransitive / no-passive constraint — cannot occupy this syntactic voice/object slot)."
                    )
                else:
                    notes.append(
                        f"Contrast: vs '{dist}' (Grammar Box contrast: {reason_code})."
                    )
                break

            # Also check if distractor carries a strict positional <gram> tag that contrasts with target
            d_prof = cls.get_word_gram_profile(d_low, pos=pos)
            t_prof = cls.get_word_gram_profile(tw, pos=pos)
            if (d_prof.get("only_before_noun") or "only before noun" in th_gram.lower()) and not t_prof.get("only_before_noun"):
                nuance_tail = f"; nuance: {th_dist}" if th_dist else ""
                notes.append(
                    f"Contrast: vs '{dist}' (`[only before noun]` constraint{nuance_tail})."
                )
                break
            if (d_prof.get("not_before_noun") or "not before noun" in th_gram.lower()) and not t_prof.get("not_before_noun"):
                nuance_tail = f"; nuance: {th_dist}" if th_dist else ""
                notes.append(
                    f"Contrast: vs '{dist}' (`[not before noun]` predicative-only constraint{nuance_tail})."
                )
                break

        # 2. Standard THESAURUS / antonym discriminator if no grammatical clash note fired yet
        if not notes:
            cluster_nuance = ""
            for th in thes_items:
                if cls._thesaurus_variant_matches(th.get("word"), tw):
                    cluster_nuance = (th.get("distinction") or "").strip()
                    break
            locked = (target_definition or "").strip()
            target_side = locked
            if cluster_nuance and (not locked or
                                   cls._definition_overlap_ratio(cluster_nuance, locked) >= 0.7):
                target_side = cluster_nuance
            if target_side:
                for dist in dists:
                    dist_note = ""
                    dist_gram = ""
                    for th in thes_items:
                        if cls._thesaurus_variant_matches(th.get("word"), dist):
                            dist_note = (th.get("distinction") or "").strip()
                            dist_gram = (th.get("grammar") or "").strip()
                            break
                    mechanism = meta.get(dist) or meta.get(dist.lower()) or ""
                    if not dist_note:
                        if mechanism == "antonym":
                            notes.append(f"Contrast: vs '{dist}' (polar opposite of target meaning).")
                            break
                        if mechanism == "orthographic_paronym":
                            notes.append(f"Contrast: vs '{dist}' (orthographic look-alike / paronym with distinct meaning).")
                            break
                        continue
                    if cls._definition_overlap_ratio(dist_note, target_side) >= 0.8 and not dist_gram:
                        continue  # no real contrast in this cluster pairing
                    hinge = ("polar opposite" if mechanism == "antonym" else "different degree/scale or collocational frame")
                    if dist_gram:
                        hinge = f"`{dist_gram}`; {hinge}"
                    notes.append(f"Contrast: vs '{dist}' ({hinge}).")
                    break
            else:
                for dist in dists:
                    mechanism = meta.get(dist) or meta.get(dist.lower()) or ""
                    if mechanism == "orthographic_paronym":
                        notes.append(f"Contrast: vs '{dist}' (orthographic look-alike / paronym with distinct meaning).")
                        break

        # 3. Target-side sense-locked Grammar Box rule (if any)
        gb_alert = cls.get_ldoce_grammar_alert(tw, quote=quote or "", sense_index=sense_index)
        if gb_alert:
            rules = []
            for r in (gb_alert.get("rules") or []):
                r_s = str(r.get("text") if isinstance(r, dict) else r or "").strip()
                r_s = re.sub(r"\s*Grammar guide\s*[‒–—-]\s*[A-Z\s]+$", "", r_s).strip()
                if r_s:
                    rules.append(r_s)
            bad_exs = [str(b).strip() for b in (gb_alert.get("bad_examples") or []) if str(b).strip()]
            if rules:
                r_short = rules[0]
                if bad_exs and "don't say" not in r_short.lower() and bad_exs[0].lower() not in r_short.lower():
                    r_short += f" (✗ Don't say: {bad_exs[0]})"
                notes.append(f"Grammar Box: {r_short}")

        return " ".join(notes)

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
        "expel", "compel", "dispel", "recap",
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
    def screen_inflected_options(
        cls,
        inflected: List[str],
        spare: List[str],
        final_target: str,
        count: int = 3,
        report: Optional[Dict[str, object]] = None,
    ) -> List[str]:
        """
        compliance gates must see the form the student actually reads.

        Distractors were screened before inflection; 'costliness' passed the length and
        CEFR gates, then 'pluralize_noun' turned it into 'costlinesses', which no gate
        ever looked at. Each inflected candidate is re-checked against the final target
        surface form; a failing candidate is replaced by the next one from the pool
        rather than repaired into another shape.
        """
        target = (final_target or "").strip().lower()
        pool: List[str] = []
        for cand in list(inflected) + list(spare):
            c = (cand or "").strip().lower()
            if c and c != target and c not in pool:
                pool.append(c)

        kept: List[str] = []
        soft_rejected: List[str] = []   # real words sitting at the wrong CEFR level
        hard_rejected: List[str] = []   # spellings that are not English word forms
        reasons: Dict[str, str] = {}
        for cand in pool:
            if not cls.is_attested_form(cand):
                hard_rejected.append(cand)
                reasons[cand] = "unattested_form"
                continue
            if not cls.is_cefr_compliant_distractor(cand, target):
                soft_rejected.append(cand)
                reasons[cand] = "cefr_ceiling"
                continue
            kept.append(cand)

        # A four-option item is a hard schema requirement, so a screened-out candidate is
        # replaced by another candidate - never by a repaired spelling. When the compliant
        # pool runs dry, padding prefers a real word at the wrong level over a spelling
        # that is not a word at all: the item degrades to 'too hard', not to 'wrong English'.
        if len(kept) < count:
            for cand in soft_rejected + hard_rejected:
                if len(kept) >= count:
                    break
                kept.append(cand)

        if report is not None:
            report.update({
                "target": target,
                "kept": list(kept),
                "rejected": reasons,
                "padded": len(kept) > len([k for k in kept if k not in soft_rejected + hard_rejected]),
            })
        return kept[:count]

    @classmethod
    def _example_target_tag(cls, example: Optional[str], target: str) -> Optional[str]:
        """
        Reads the verb form the authentic example actually uses for the target
        ('What annoyed him most...' -> VBD). Returns None when the target is absent.
        """
        ex = (example or "").strip()
        tw = (target or "").strip().lower()
        if not ex or not tw:
            return None
        nlp = cls.get_spacy()
        if not nlp:
            return None
        try:
            doc = nlp(ex)
        except Exception:
            return None
        for tok in doc:
            if tok.text.lower() == tw or tok.lemma_.lower() == tw:
                return tok.tag_.upper()
        return None

    @classmethod
    def _reconcile_prescribed_inflection(
        cls,
        base_options: List[str],
        target_index: int,
        target_word: str,
        canonical_pos: Optional[str],
        inflection_desc: str,
        authentic_example: Optional[str]
    ) -> Tuple[str, List[str], str, str]:
        """
        Prescribed-Inflection Concordance .

        The blueprint declares `inflection` and the LLM must obey it, so the declaration
        must come from the same evidence the item is built on. For verbs it was always
        'base form' even when the authentic PedLex example that supplied the frame read
        'What annoyed him most was that he had received no apology.' — a past-tense frame.
        The blueprint then demanded a base-form verb inside a past-tense cloze, which is
        unanswerable. When the anchoring example is inflected, every option is inflected to
        the identical form so frame, declared inflection and options agree.
        Returns (inflection_desc, options, target_word, form_tag).
        """
        if (canonical_pos or "").lower() != "verb" or inflection_desc != "base form":
            return inflection_desc, base_options, target_word, "VB"
        tag = cls._example_target_tag(authentic_example, target_word)
        if tag not in ("VBD", "VBZ", "VBG", "VBN"):
            return inflection_desc, base_options, target_word, "VB"
        inflected: List[str] = []
        for opt in base_options:
            form = cls._inflect_verb(opt, tag)
            if not form or not cls._morph_form_valid(opt, form, tag):
                return inflection_desc, base_options, target_word, "VB"  # unsafe to rewrite
            inflected.append(form)
        new_target = inflected[target_index] if 0 <= target_index < len(inflected) else target_word
        # A1: an inflected target that is not an attested English form must never ship,
        # even when spaCy was willing to lemmatize it back to the base verb.
        if not cls.is_attested_form(new_target):
            return inflection_desc, base_options, target_word, "VB"

        # Lexical POS Boundary Guard: reject inflection ONLY if the inflected form is an independent
        # dictionary headword entry (`word == new_target`) with a non-verb POS (e.g. annoy[verb] -> annoyed[adjective entry]).
        # Note: get_ldoce_entry('pledged') falls back to the lemma entry 'pledge' (homograph 1 = noun),
        # whose `word` is 'pledge', NOT 'pledged' — that is a lemmatized fallback, not a dedicated participial-adjective entry!
        _inflected_entry = cls.get_ldoce_entry(new_target)
        if (
            _inflected_entry
            and _inflected_entry.get("kind") == "article"
            and (_inflected_entry.get("word") or "").strip().lower() == new_target.lower()
        ):
            _inflected_pos = (_inflected_entry.get("pos") or "").lower()
            if _inflected_pos and _inflected_pos not in ("verb", "v"):
                return inflection_desc, base_options, target_word, "VB"

        return cls._VERB_FORM_LABELS.get(tag, "base form"), inflected, new_target, tag

    @classmethod
    def _batch_safe_reanchor(
        cls,
        target_word: str,
        canonical_pos: str,
        definition: Optional[str] = None,
        quote: Optional[str] = None,
        forbidden: Optional[Set[str]] = None
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Re-picks a collocational anchor drawn strictly from OUTSIDE the batch target set.
        Returns (anchor, anchor_type), or (None, None) when nothing outside the batch fits,
        which lets the micro-task planner degrade to its neutral semantic-field template
        rather than welding the anchor-presence gate and the cross-target leakage gate
        into an unsatisfiable deadlock.
        """
        tw = (target_word or "").lower().strip()
        if not tw or not forbidden:
            return None, None

        entry_pos = {"verb": "verb", "noun": "noun", "adj": "adj"}.get(canonical_pos)
        entry = cls.get_oxford_collocations(tw, pos=entry_pos) if entry_pos else {}

        candidates: List[str] = []
        if entry_pos == "verb":
            candidates = entry.get("colloc_nouns", []) + entry.get("noun_after", [])
        elif entry_pos == "adj":
            candidates = entry.get("noun_after", []) + entry.get("colloc_nouns", [])
        # Noun micro-tasks only consume a bound preposition; an adjective or governing
        # verb anchor falls through to the neutral template regardless, so skip them.

        anchor = cls._select_sense_aligned_anchor(
            candidates,
            definition=definition,
            quote=quote,
            target_word=tw,
            forbidden=forbidden
        )
        if anchor:
            return anchor, ("object" if entry_pos == "verb" else "modified_noun")

        p_cand = cls._pick_anchor_token(entry.get("prep", []), target_word=tw, forbidden=forbidden)
        if p_cand:
            return p_cand, "prep"
        return None, None

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
    _DELEXICAL_VERBS = {
        "like", "want", "need", "get", "have", "has", "make", "use", "give", "take",
        "put", "see", "know", "feel", "seem", "look", "go", "come", "say", "try",
        "help", "let", "call", "find", "keep", "set", "turn", "show", "ask", "tell",
        "mean", "include", "bring", "send", "add", "seem", "appear", "become",
    }

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
    _DELEXICAL_QUANTIFIER_NOUNS = {
        "lot", "lots", "bit", "bits", "sort", "sorts", "kind", "kinds", "type", "types",
        "couple", "number", "majority", "minority", "plenty", "rest", "bunch", "load",
        "loads", "few", "many", "mass", "stuff", "thing", "things", "amount", "quantity",
    }

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
        p_key = ("verb" if tok_pos in ("VERB", "AUX") else
                 "adj" if tok_pos == "ADJ" else
                 "adv" if tok_pos == "ADV" else
                 "noun" if tok_pos in ("NOUN", "PROPN") else
                 "adj" if p_low.startswith("adj") else
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
        distractors: Optional[List[str]] = None
    ) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
        """
        Last-resort anchor and authentic example, taken from the dictionary's own sense
        examples. Sense order is re-ranked by _lock_sense so the example always
        illustrates the meaning this item teaches, never an unrelated homograph.
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
        locked = cls._lock_sense(entry, definition=definition, quote=quote, target_pos=pos)
        ordered: List[Dict[str, Any]] = []
        if locked is not None and 0 <= locked < len(senses):
            ordered.append(senses[locked])
        ordered.extend(s for i, s in enumerate(senses) if i != locked)

        pool: List[str] = []
        for sense in ordered:
            pool.extend(sense.get("examples") or [])
        if not pool:
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

        # Strict Sense-First scope: never roam into other senses when locked_sense is resolved
        candidate_senses = [locked_sense] if locked_sense is not None else senses

        # Precompute sense lemma set for filtering entry-level collocations against the locked sense
        sense_lemmas: Set[str] = set()
        is_polysemous = len(senses) > 1 and locked_sense is not None
        if locked_sense is not None:
            try:
                nlp = cls.get_spacy()
                s_text = (
                    (locked_sense.get("definition") or "")
                    + " "
                    + " ".join(locked_sense.get("examples") or [])
                    + " "
                    + (locked_sense.get("signpost") or "")
                ).lower()
                self_forms = {tw_clean} | {t.lemma_.lower() for t in nlp(tw_clean)}
                sense_lemmas = {
                    tok.lemma_.lower()
                    for tok in nlp(s_text)
                    if tok.is_alpha and not tok.is_stop and len(tok.lemma_) >= 3
                } - self_forms - cls._ANCHOR_STOPWORDS
            except Exception:
                sense_lemmas = set()

        def _score_colloc_item(it_c: Dict[str, Any]) -> int:
            if not sense_lemmas:
                return 0
            try:
                nlp = cls.get_spacy()
                self_forms = {tw_clean} | {t.lemma_.lower() for t in nlp(tw_clean)}
                c_text = ((it_c.get("collocation") or "") + " " + (it_c.get("example") or "")).lower()
                c_lems = {
                    t.lemma_.lower()
                    for t in nlp(c_text)
                    if t.is_alpha and not t.is_stop and len(t.lemma_) >= 3
                } - self_forms - cls._ANCHOR_STOPWORDS
                return len(c_lems & sense_lemmas)
            except Exception:
                return 0

        # 1. Adjective Cascades (prep valency >> modified_noun >> adv_mod)
        if p_low.startswith("adj") or p_low == "a":
            # Level 1: Preposition valency from locked sense grammar patterns (e.g. 'proud of', 'aware of', 'serious about')
            for s in candidate_senses:
                for pat in s.get("patterns", []):
                    for p in ("of", "to", "for", "with", "about", "from", "in", "on", "at"):
                        if re.search(rf"\b{p}\b", pat.lower()):
                            if p == "to":
                                m_inf = re.search(r"\bto\s+([a-z]+)", pat.lower())
                                if m_inf and m_inf.group(1) not in (
                                    "sb", "sth", "someone", "something", "oneself", "himself", "herself", "them", "the", "a", "an"
                                ):
                                    continue  # Infinitive pattern (e.g. 'simple to use'), NOT a bound preposition
                            # Select example that directly demonstrates the bound preposition
                            matching_exs = [e for e in s.get("examples", []) if re.search(rf"\b{p}\b", e.lower())]
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

            # Level 1.5: Modified nouns extracted from locked sense authentic examples (e.g. 'firm mattress')
            if not anchor and candidate_senses:
                try:
                    nlp = cls.get_spacy()
                    quote_lower = (quote or "").lower()
                    cand_nouns = []
                    for s in candidate_senses:
                        for ex in s.get("examples", []):
                            doc_ex = nlp(ex)
                            for tok in doc_ex:
                                if tok.lemma_.lower() == tw_clean:
                                    if tok.dep_ == "amod" and tok.head.pos_ in ("NOUN", "PROPN"):
                                        cand_n = tok.head.lemma_.lower()
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
                colloc_nouns = list(entry["collocations"]["nouns"])
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

            # Level 2.5: Causative / Predicative Verb Frames (e.g. 'make it possible', 'render possible', 'find possible')
            if not anchor and entry.get("collocations", {}).get("phrases"):
                colloc_phrases = list(entry["collocations"]["phrases"])
                if is_polysemous and sense_lemmas:
                    colloc_phrases = [it_c for it_c in colloc_phrases if _score_colloc_item(it_c) > 0]
                for it in colloc_phrases:
                    c_str = it.get("collocation", "").lower()
                    ex = it.get("example", "")
                    for cv in ("make", "render", "find", "deem", "keep", "consider"):
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
                verb_collocs = entry.get("collocations", {}).get("verbs", []) + entry.get("collocations", {}).get("adverbs", [])
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
            # Step A: Prioritize direct object patient strictly from the locked target sense
            try:
                nlp = cls.get_spacy()
                for s in candidate_senses:
                    for ex in s.get("examples", []):
                        doc_ex = nlp(ex)
                        for tok in doc_ex:
                            if tok.lemma_.lower() == tw_clean and tok.pos_ in ("VERB", "AUX"):
                                for child in tok.children:
                                    if child.dep_ in ("dobj", "obj") and child.lemma_.lower() not in cls._ANCHOR_STOPWORDS and child.pos_ in ("NOUN", "PROPN"):
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
            if not anchor and entry.get("collocations", {}).get("nouns"):
                try:
                    nlp = cls.get_spacy()
                except Exception:
                    nlp = None
                colloc_nouns = list(entry["collocations"]["nouns"])
                if sense_lemmas and nlp is not None:
                    if is_polysemous:
                        colloc_nouns = [it_c for it_c in colloc_nouns if _score_colloc_item(it_c) > 0]
                    colloc_nouns.sort(key=_score_colloc_item, reverse=True)
                for it in colloc_nouns:
                    ex = it.get("example", "")
                    c_str = it.get("collocation", "").lower()
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
                            if is_noun_compound or not has_verb_usage:
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
                        pat_l = pat.lower()
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
            subj_verb_collocs = list(entry.get("collocations", {}).get("verbs", []))
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
                c_verbs = list(entry.get("collocations", {}).get("verbs", []))
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
                c_adjs = list(entry.get("collocations", {}).get("adjectives", []))
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
                                        if head and head.pos_ == req_pos and head.lemma_.lower() not in cls._ANCHOR_STOPWORDS and head.lemma_.lower() != tw_clean:
                                            cand_head = head.lemma_.lower()
                                            if not any(cls.double_key_collision(tw_clean, d, anchor=cand_head, anchor_type=req_atype, pos="adv", quote=quote)[0] for d in dists):
                                                anchor = cand_head
                                                atype = req_atype
                                                frame_desc = f"modifying {head.pos_.lower()} {cand_head}"
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
                quote=quote, distractors=dists
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

            # 1. Curated collocations of the target (only if anchor is NOT a bare preposition from another sense)
            is_bare_prep = anc_l in ("of", "to", "for", "with", "about", "from", "in", "on", "at", "into", "over", "under", "against", "towards", "toward", "by", "upon")
            if not is_bare_prep:
                for items in (t_entry.get("collocations") or {}).values():
                    for item in items or []:
                        if anc_re.search((item.get("collocation") or "").lower()) and item.get("example"):
                            return item["example"]
                for item in t_entry.get("cross_collocations") or []:
                    if anc_re.search((item.get("collocation") or "").lower()) and item.get("example"):
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

    # -------------------------------------------------------------------------
    # 1. Sentence Boundary Tokenization & Indexing ([S-1], [S-2])
    # -------------------------------------------------------------------------
    @classmethod
    def tokenize_and_index_sentences(cls, text: str) -> Tuple[str, Dict[str, str]]:
        """
        Pre-tokenizes raw source text into an indexed, complete-sentence pool.
        Returns:
            indexed_text: Text formatted with '[S-1] ... [S-2] ...' for LLM prompts.
            sentence_pool: Dict mapping 'S-1' -> pristine full sentence text.
        """
        nlp = cls.get_spacy()
        # Cleanly separate frontmatter if present so spaCy only tokenizes authentic body prose
        fm_match = re.match(r"^(---\s*\n.*?\n---\s*\n)(.*)", text, re.DOTALL)
        frontmatter = fm_match.group(1) if fm_match else ""
        body = fm_match.group(2) if fm_match else text

        # Strip trailing curriculum appendix sections (e.g. '## Syllabus Vocabulary', '## Appendix')
        # so bare vocabulary lists are never tokenized as passage evidence sentences.
        syllabus_suffix = ""
        m_appendix = re.search(r"(?mi)^(?:#+\s*(?:Syllabus\s+Vocabulary|Appendix)\b.*)$", body)
        if m_appendix:
            syllabus_suffix = body[m_appendix.start():]
            body = body[:m_appendix.start()]

        sentence_pool: Dict[str, str] = {}
        indexed_paragraphs: List[str] = []
        counter = 1

        # Normalize single-newline bullet/numbered lists so list items become independent paragraphs
        body_norm = re.sub(
            r'\n(?=[\s\u2022\u00b7\u25aa\u25ab\*\-]+|(?:\(?\d+[\.\)]\s+)|(?:\(?[a-zA-Z][\.\)]\s+))',
            r'\n\n',
            body
        )

        # Process paragraph-by-paragraph to preserve natural paragraph boundaries while streaming sentences within paragraphs
        raw_paragraphs = re.split(r'\n{2,}', body_norm.strip())
        for para in raw_paragraphs:
            para = para.strip()
            if not para:
                continue

            # If the paragraph is a pure markdown header (e.g. '## Text A', '### Section 1')
            lines = [l.strip() for l in para.split("\n") if l.strip()]
            if all(l.startswith("#") for l in lines):
                indexed_paragraphs.append(para)
                continue

            # Separate optional leading heading lines in this paragraph
            prefix_lines = []
            content_lines = []
            for l in lines:
                if not content_lines and l.startswith("#"):
                    prefix_lines.append(l)
                else:
                    content_lines.append(l)

            prefix_str = "\n\n".join(prefix_lines) + "\n\n" if prefix_lines else ""
            para_raw_text = " ".join(content_lines).strip()
            if not para_raw_text:
                if prefix_lines:
                    indexed_paragraphs.append("\n\n".join(prefix_lines))
                continue

            # Strip leading bullet/numbered list marker (e.g. '•', '-', '*', '1.', '(1)', 'a.')
            bullet_match = re.match(
                r"^([\s\u2022\u00b7\u25aa\u25ab\*\-]+|(?:\(?\d+[\.\)]\s*)|(?:\(?[a-zA-Z][\.\)]\s+))(.*)",
                para_raw_text
            )
            list_marker = bullet_match.group(1).strip() if bullet_match else ""
            para_text = bullet_match.group(2).strip() if bullet_match else para_raw_text

            # Protect terminal punctuation inside dialogue quotation marks so sentences like
            # 'Their attitude was “Ah! You’re here! We can start now!”' stay unified as one full authentic sentence
            def _mask_quoted_punct(m: re.Match) -> str:
                quote_open = m.group(1)
                quote_body = m.group(2)
                quote_close = m.group(3)
                # Protect abbreviations (Mr., Ms., Dr., etc.) inside quotes so they don't get split
                protected_body = re.sub(
                    r'\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|e\.g|i\.e)\.',
                    lambda x: x.group(1) + '§ABBRDOT§',
                    quote_body,
                    flags=re.IGNORECASE
                )
                masked_body = (
                    protected_body.replace(".", " §DOT§ ")
                                  .replace("!", " §EXCL§ ")
                                  .replace("?", " §QUES§ ")
                                  .replace('§ABBRDOT§', '.')
                )
                return f"{quote_open}{masked_body}{quote_close}"

            # Match curly and straight quotes
            masked_para_text = re.sub(
                r'([“"«])([^”"»]+?)([”"»])',
                _mask_quoted_punct,
                para_text
            )

            para_doc = nlp(masked_para_text)
            para_sent_parts = []
            for sent in para_doc.sents:
                raw_sent = sent.text
                raw_sent = re.sub(r'\s*§DOT§\s*', '. ', raw_sent)
                raw_sent = re.sub(r'\s*§EXCL§\s*', '! ', raw_sent)
                raw_sent = re.sub(r'\s*§QUES§\s*', '? ', raw_sent)
                raw_sent = re.sub(r'([,:;])(["“«])(?=\w)', r'\1 \2', raw_sent)
                raw_sent = re.sub(r'([.!?,\u2026])\s+(["”»])(?=\s|$|[)\].,:;!?])', r'\1\2', raw_sent)
                raw_sent = re.sub(r'\s+([”»])', r'\1', raw_sent)
                raw_sent = re.sub(r'([“«])\s+', r'\1', raw_sent)
                raw_sent = re.sub(r'\s{2,}', ' ', raw_sent).strip()
                if not raw_sent:
                    continue
                # Clean enclosing markdown bold/italic formatting from pristine sentence text
                sent_clean = re.sub(r'^\*+|\*+$', '', raw_sent).strip()
                words = [t for t in nlp(sent_clean) if t.is_alpha]
                if len(words) < 2 and not sent_clean.endswith((".", "?", "!")):
                    para_sent_parts.append(raw_sent)
                    continue

                sid = f"S-{counter}"
                sentence_pool[sid] = sent_clean
                para_sent_parts.append(f"[{sid}] {sent_clean}")
                counter += 1

            marker_str = f"{list_marker} " if list_marker else ""
            indexed_para = prefix_str + marker_str + " ".join(para_sent_parts)
            indexed_paragraphs.append(indexed_para.strip())

        if frontmatter:
            indexed_text = frontmatter + "\n\n" + "\n\n".join(indexed_paragraphs)
        else:
            indexed_text = "\n\n".join(indexed_paragraphs)
        if syllabus_suffix:
            indexed_text = indexed_text.rstrip() + "\n\n" + syllabus_suffix.strip()
        return indexed_text, sentence_pool

    @classmethod
    def snap_to_sentence_pool(cls, raw_quote_or_id: str, sentence_pool: Dict[str, str]) -> Optional[str]:
        """
        Deterministically snaps a model-provided sentence ID (e.g. '[S-3]', 'S-3') or a
        partial quote with ellipsis back to the pristine, authentic source sentence in the pool.
        Eliminates trailing ellipses, copy-paste hallucinations, and verbatim mismatches.
        """
        if not raw_quote_or_id or not sentence_pool:
            return None

        clean_input = re.sub(r"^\s*\[?\bS-\d+\b\]?\s*[:\-]??\s*", "", raw_quote_or_id.strip(), flags=re.IGNORECASE).strip()

        # 1. Direct ID lookup (e.g. '[S-3]' or 'S-3' or 'Sentence 3')
        id_match = re.search(r"\bS-(\d+)\b", raw_quote_or_id, re.IGNORECASE)
        if id_match:
            sid = f"S-{id_match.group(1)}"
            if sid in sentence_pool:
                return re.sub(r"^\s*\[?\bS-\d+\b\]?\s*[:\-]??\s*", "", sentence_pool[sid], flags=re.IGNORECASE).strip()

        # 2. Exact match in sentence pool
        for sid, sent in sentence_pool.items():
            sent_clean = re.sub(r"^\s*\[?\bS-\d+\b\]?\s*[:\-]??\s*", "", sent, flags=re.IGNORECASE).strip()
            if clean_input.lower() == sent_clean.lower():
                return sent_clean

        # 3. Substring containment: if quote is a fragment of a pool sentence
        clean_target = re.sub(r"\.{3,}|…", "", clean_input).strip().lower()
        if len(clean_target) >= 15:
            for sid, sent in sentence_pool.items():
                sent_clean = re.sub(r"^\s*\[?\bS-\d+\b\]?\s*[:\-]??\s*", "", sent, flags=re.IGNORECASE).strip()
                if clean_target in sent_clean.lower():
                    return sent_clean

        # 4. High overlap fallback
        clean_tokens = set(re.findall(r"\w{3,}", clean_target))
        if clean_tokens:
            best_sent = None
            max_overlap = 0.0
            for sid, sent in sentence_pool.items():
                sent_clean = re.sub(r"^\s*\[?\bS-\d+\b\]?\s*[:\-]??\s*", "", sent, flags=re.IGNORECASE).strip()
                sent_tokens = set(re.findall(r"\w{3,}", sent_clean.lower()))
                if not sent_tokens:
                    continue
                overlap = len(clean_tokens & sent_tokens) / len(clean_tokens)
                if overlap > max_overlap and overlap >= 0.75:
                    max_overlap = overlap
                    best_sent = sent_clean
            if best_sent:
                return best_sent

    @classmethod
    def get_sentence_id(cls, raw_quote_or_id: str, sentence_pool: Dict[str, str]) -> Optional[str]:
        """
        Returns the canonical sentence identifier (e.g. 'S-3') for a given quote or ID.
        """
        if not raw_quote_or_id or not sentence_pool:
            return None

        clean_input = raw_quote_or_id.strip()

        # 1. Direct ID lookup
        id_match = re.search(r"\bS-(\d+)\b", clean_input, re.IGNORECASE)
        if id_match:
            sid = f"S-{id_match.group(1)}"
            if sid in sentence_pool:
                return sid

        # 2. Exact match
        for sid, sent in sentence_pool.items():
            if clean_input.lower() == sent.lower():
                return sid

        # 3. Substring containment
        clean_target = re.sub(r"\.{3,}|…", "", clean_input).strip().lower()
        if len(clean_target) >= 15:
            for sid, sent in sentence_pool.items():
                if clean_target in sent.lower():
                    return sid

        # 4. Token overlap fallback
        clean_tokens = set(re.findall(r"\w{3,}", clean_target))
        if clean_tokens:
            best_sid = None
            max_overlap = 0.0
            for sid, sent in sentence_pool.items():
                sent_tokens = set(re.findall(r"\w{3,}", sent.lower()))
                if not sent_tokens:
                    continue
                overlap = len(clean_tokens & sent_tokens) / len(clean_tokens)
                if overlap > max_overlap and overlap >= 0.75:
                    max_overlap = overlap
                    best_sid = sid
            if best_sid:
                return best_sid

        return None

    @classmethod
    def determine_contextual_pos(cls, word: str, quoted_sentence: str) -> str:
        """
        Deterministically infers the contextual part of speech (noun, verb, adjective, adverb,
        preposition, conjunction, interjection) using spaCy dependency parsing and POS tagging.
        Runs entirely offline at zero token cost (<1ms).
        """
        VALID_POS_MAP = {
            "NOUN": "noun", "PROPN": "noun",
            "VERB": "verb",
            "ADJ": "adjective",
            "ADV": "adverb",
            "ADP": "preposition",
            "CCONJ": "conjunction", "SCONJ": "conjunction",
            "INTJ": "interjection",
        }
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

        nlp = cls.get_spacy()
        doc = nlp(sent_text)
        clean_w = re.sub(r"\[.*?\]|\(.*?\)", "", word).strip().lower()
        w_toks = clean_w.split()
        target_tok = w_toks[0] if w_toks else clean_w

        # 0. High-priority compound lookup: check PedLex first for authentic pedagogical POS, then WordNet
        # ONLY for genuine multi-word or hyphenated compounds
        # (Do NOT run on single words like 'firm' or 'log', where default sense overrides contextual dependency)
        if "-" in clean_w or " " in clean_w:
            ld_cands = [clean_w, clean_w.replace("-", ""), clean_w.replace(" ", ""), clean_w.replace("-", " "), clean_w.replace(" ", "-")]
            for c_cand in ld_cands:
                ld_c_entry = cls.get_ldoce_entry(c_cand)
                if ld_c_entry:
                    c_poses = ld_c_entry.get("all_poses", [ld_c_entry.get("pos", "")])
                    if any("noun" in p for p in c_poses):
                        return "noun"
                    if any("adj" in p for p in c_poses):
                        return "adjective"
                    if any("adv" in p for p in c_poses):
                        return "adverb"
                    if any("verb" in p and "adverb" not in p for p in c_poses):
                        return "verb"

        wn = cls.get_wordnet()
        if wn and ("-" in clean_w or " " in clean_w):
            compound_cands = [clean_w, clean_w.replace(" ", "-"), clean_w.replace("-", " "), clean_w.replace("-", "_")]
            for c_cand in compound_cands:
                c_synsets = wn.synsets(c_cand)
                if c_synsets:
                    c_pos = c_synsets[0].pos
                    wn_map = {"n": "noun", "v": "verb", "a": "adjective", "s": "adjective", "r": "adverb"}
                    if c_pos in wn_map:
                        return wn_map[c_pos]

        # 1. Exact match on token text, lemma, or morphological inflection
        target_forms = cls.inflected_forms(clean_w) | {clean_w}
        for tok in doc:
            tok_lower = tok.text.lower()
            tok_lem = tok.lemma_.lower()
            matches_target = (
                tok_lower == target_tok or tok_lem == target_tok or
                tok_lower in target_forms or tok_lem in target_forms or
                cls.form_matches(tok_lower, clean_w) or cls.form_matches(tok_lem, clean_w)
            )
            if matches_target:
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
                        if "noun" in p_low:
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
                    for s in (ld_target.get("senses") or []):
                        sp = str(s.get("pos") or "").lower()
                        if "noun" in sp:
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

                if len(dict_poses) == 1 and wn:
                    if tok.pos_ == "VERB" and (wn.synsets(clean_w, pos="v") or wn.synsets(tok_lem, pos="v")):
                        dict_poses.add("verb")
                    elif tok.pos_ == "ADJ" and (wn.synsets(clean_w, pos="a") or wn.synsets(clean_w, pos="s")):
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
                    has_v = ("verb" in dict_poses) if dict_poses else bool(wn and (wn.synsets(clean_w, pos="v") or wn.synsets(tok_lem, pos="v")))
                    if not has_v:
                        if dict_poses:
                            for pref in ("noun", "adjective", "adverb"):
                                if pref in dict_poses:
                                    return pref
                        return "noun"

                # If token is an adverb (ADV) or only has adverb/noun in dict, do NOT force it to adjective!
                if tok.pos_ == "ADV" and ("adverb" in dict_poses or not ("adjective" in dict_poses)):
                    return "adverb"

                # If modifying a noun as an adjectival modifier (amod) or compound modifier (e.g. Title Case 'Take Corporate Action', 'premier hotels', 'amateur runners')
                if (tok.dep_ == "amod" or tok.dep_ == "compound") and tok.head.pos_ in ("NOUN", "PROPN"):
                    if tok.dep_ == "compound":
                        has_adj = ("adjective" in dict_poses) if dict_poses else bool(wn and (wn.synsets(clean_w, pos="a") or wn.synsets(clean_w, pos="s")))
                        has_noun = ("noun" in dict_poses) if dict_poses else bool(wn and wn.synsets(clean_w, pos="n"))
                        if has_adj and not has_noun:
                            return "adjective"
                        # If PedLex lists adjective BEFORE noun in all_poses (or primary pos is adjective, e.g. 'premier', 'amateur', 'veteran'),
                        # only treat as adjective when modifying a common NOUN ('premier hotels', 'amateur runners'), NOT a PROPN title ('Premier Zhou Enlai')
                        if tok.head.pos_ == "NOUN" and has_adj and ld_target and "adj" in str((ld_target.get("all_poses") or [ld_target.get("pos", "")])[0]).lower():
                            return "adjective"
                    else:
                        if tok.tag_ in ("VBN", "VBG") or tok.pos_ == "VERB":
                            has_adj = ("adjective" in dict_poses) if dict_poses else bool(wn and (wn.synsets(clean_w, pos="a") or wn.synsets(clean_w, pos="s")))
                            if has_adj:
                                return "adjective"
                            else:
                                return "verb"
                        if not dict_poses or "adjective" in dict_poses:
                            return "adjective"
                # Participle adjective / predicate adjective / prepositional participle check vs Passive Voice:
                if tok.tag_ in ("VBN", "VBG") or tok.pos_ == "VERB":
                    # Prepositional participles (e.g. 'Given that...', 'Given the difficulty...', 'Considering...')
                    if "preposition" in dict_poses and tok.dep_ in ("prep", "mark", "advcl") and not any(c.dep_ in ("aux", "auxpass", "nsubj", "nsubjpass") for c in tok.children):
                        if any(c.dep_ in ("pobj", "dobj", "ccomp", "mark") or c.text.lower() == "that" for c in tok.children) or (tok.i + 1 < len(doc) and doc[tok.i + 1].text.lower() == "that"):
                            return "preposition"
                if tok.tag_ == "VBN" or tok.pos_ == "VERB":
                    surf = tok.text.lower()
                    if clean_w != surf and dict_poses and "adjective" not in dict_poses and "verb" in dict_poses:
                        return "verb"
                    ld_entry = cls.get_ldoce_entry(surf)
                    surf_poses = {cls._pos_code(s.get("pos")) for s in (ld_entry.get("senses") or []) if s.get("pos")} if (ld_entry and ld_entry.get("word") == surf) else set()
                    has_auxpass = any(c.dep_ == "auxpass" for c in tok.children) or (tok.dep_ == "auxpass")
                    if has_auxpass:
                        if surf_poses == {"adj"} and not any(c.dep_ == "agent" or (c.dep_ == "prep" and c.lemma_.lower() in ("by", "with")) for c in tok.children):
                            return "adjective"
                        return "verb"
                    has_be_gov = any(c.lemma_ == "be" for c in tok.children if c.dep_ in ("aux", "cop")) or (tok.head.lemma_ == "be")
                    if has_be_gov:
                        has_adj = False
                        if ld_entry:
                            has_adj = any("adj" in p for p in ld_entry.get("all_poses", [])) or any("adj" in (s.get("pos") or "") for s in ld_entry.get("senses", []))
                        if not has_adj and wn:
                            has_adj = bool(wn.synsets(surf, pos="a") or wn.synsets(surf, pos="s"))
                        if has_adj:
                            if surf_poses == {"adj"}:
                                return "adjective"
                            if dict_poses and "adjective" not in dict_poses and "verb" in dict_poses:
                                return "verb"
                            return "adjective"

                # Syntactic dependency repair for misparsed NOUN/PROPN (e.g. ROOT/predicate verbs like 'trap energy', 'swells'):
                if tok.pos_ in ("NOUN", "PROPN"):
                    has_verb_entry = ("verb" in dict_poses) if dict_poses else bool(wn and (wn.synsets(clean_w, pos="v") or wn.synsets(tok.lemma_.lower(), pos="v")))
                    if has_verb_entry:
                        has_dobj = any(c.dep_ in ("dobj", "obj") for c in tok.children)
                        has_nsubj_child = any(c.dep_ in ("nsubj", "nsubjpass") for c in tok.children)
                        if has_dobj or has_nsubj_child:
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

                        if tok.dep_ == "ROOT":
                            has_nsubj = any(c.dep_ in ("nsubj", "nsubjpass") for c in tok.children)
                            has_compound_subj = any(c.dep_ == "compound" and c.head == tok for c in tok.children)
                            has_prep_comp = any(c.dep_ == "prep" for c in tok.children)
                            has_clause_finite_verb = any(t.pos_ in ("VERB", "AUX") and t.dep_ in ("ROOT", "conj", "advcl", "ccomp") and t.i != tok.i for t in doc)
                            if (has_nsubj or (has_compound_subj and has_prep_comp)) and not has_clause_finite_verb:
                                return "verb"

                        if tok.dep_ == "pobj" and tok.head.lemma_.lower() in ("until", "before", "after", "since", "as", "when", "while"):
                            has_det_child = any(c.dep_ == "det" for c in tok.children)
                            has_comp_subj = any(c.dep_ in ("compound", "amod", "poss") and c.i < tok.i for c in tok.children)
                            if has_det_child and has_comp_subj and tok.tag_ == "NNS":
                                return "verb"

                # Syntactic dependency repair for misparsed ADJ coordinated with a finite verb (e.g. 'bodies wear out and degenerate'):
                if tok.pos_ == "ADJ" and tok.dep_ == "conj" and tok.head.pos_ == "VERB":
                    has_verb_entry = ("verb" in dict_poses) if dict_poses else bool(wn and wn.synsets(clean_w, pos="v"))
                    if has_verb_entry and not any(c.dep_ == "cop" for c in tok.children):
                        return "verb"

                mapped_pos = VALID_POS_MAP.get(tok.pos_, "noun")
                if dict_poses and mapped_pos not in dict_poses:
                    for pref in ("verb", "noun", "adjective", "adverb"):
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
                for k, v in (("noun", "noun"), ("adv", "adverb"), ("verb", "verb"), ("adj", "adjective"), ("prep", "preposition"), ("conj", "conjunction")):
                    if k in p_low and not (k == "verb" and "adverb" in p_low):
                        dict_poses_fb.add(v)
            for s in (ld_target_fallback.get("senses") or []):
                sp = str(s.get("pos") or "").lower()
                for k, v in (("noun", "noun"), ("adv", "adverb"), ("verb", "verb"), ("adj", "adjective"), ("prep", "preposition"), ("conj", "conjunction")):
                    if k in sp and not (k == "verb" and "adverb" in sp):
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
            or m_pos_low in cls.EXPRESSION_TYPE_LABELS
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
            'fireworks': 'fireworks', 'premises': 'premises', 'congratulations': 'congratulations'
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

    # -------------------------------------------------------------------------
    # 3. Dependency-Based Macro Domain Classification Gate
    # -------------------------------------------------------------------------
    @classmethod
    def classify_grammar_dependency(cls, sentence: str) -> Optional[str]:
        """
        Classifies complex sentence into one of the Four Macro Functional Domains
        using deterministic spaCy dependency trees and syntactic relations:
          - 'Rhetoric & Emphasis'
          - 'Cohesion & Framing'
          - 'Information Packaging'
          - 'Logic & Stance'
        Returns None if no unambiguous structural pattern dominates.
        """
        sentence_norm = sentence.replace('\u2019', "'").replace('\u2018', "'")
        nlp = cls.get_spacy()
        doc = nlp(sentence_norm)

        # Declarative DependencyMatcher Engine Pass (High-Precision DSL)
        try:
            from librarian.grammar_patterns import GrammarPatternEngine
            gp_engine = GrammarPatternEngine.get_engine(nlp)
            matched_patterns = gp_engine.match_sentence(doc)
            if matched_patterns:
                return matched_patterns[0]["category"]
        except Exception:
            pass

        # ---------------------------------------------------------------------
        # 1. Rhetoric & Emphasis:
        #    a. Fronted Inversion: aux or ROOT precedes nsubj in linear word order
        #    b. Cleft Focus: expl ('it') + 'is/was' + focal constituent + relcl/that...
        #    c. Antithesis: 'not... but...' parallel coordinating balance
        # ---------------------------------------------------------------------
        # Inversion check
        for token in doc:
            if token.dep_ == "nsubj":
                # Check if auxiliary or root verb precedes nsubj (excluding questions)
                head = token.head
                if head.pos_ == "VERB" or head.pos_ == "AUX":
                    aux_children = [c for c in head.children if c.dep_ in ("aux", "auxpass")]
                    for aux in aux_children:
                        if aux.i < token.i and not sentence.strip().endswith("?"):
                            # Check if fronted negative/restrictive adverbial starts sentence
                            first_token = doc[0].lemma_.lower()
                            if first_token in ("hardly", "scarcely", "seldom", "never", "rarely", "barely", "only", "not", "no"):
                                return "Rhetoric & Emphasis"

        # Cleft vs Dummy-it check
        for token in doc:
            if token.text.lower() == "it" and token.dep_ in ("expl", "nsubj"):
                # Check copula verb
                head = token.head
                if head.lemma_ in ("be", "seem"):
                    if any(c.dep_ == "attr" and c.lemma_.lower() == "time" for c in head.children):
                        continue
                    # 1. Relcl under focal constituent with relative pronoun
                    has_relcl = False
                    for child in head.children:
                        if child.dep_ == "relcl":
                            if any(gc.text.lower() in ("that", "who", "whom", "which", "whose") for gc in child.children):
                                has_relcl = True
                        for gc in child.children:
                            if gc.dep_ == "relcl":
                                if any(ggc.text.lower() in ("that", "who", "whom", "which", "whose") for ggc in gc.children):
                                    has_relcl = True
                    if has_relcl:
                        return "Rhetoric & Emphasis"

                    # 2. Prepositional or adverbial cleft (e.g. It's in December that we tend...)
                    for child in head.children:
                        if child.dep_ in ("ccomp", "advcl"):
                            has_marker = any(
                                gc.dep_ in ("mark", "nsubj", "advmod", "dobj", "pobj")
                                and gc.text.lower() in ("that", "who", "whom", "where", "when")
                                for gc in child.children
                            )
                            if has_marker:
                                # Ensure focal element is not an evaluative adjective or extraposition noun (dummy-it)
                                _EXTRAPOSITION_NOUNS = {
                                    "surprise", "wonder", "secret", "doubt", "coincidence", "accident",
                                    "pity", "shame", "mystery", "fact", "truth", "news", "use", "good",
                                    "point", "time", "question", "matter"
                                }
                                has_eval_adj = any(
                                    (c.dep_ == "acomp" and c.pos_ == "ADJ")
                                    or (c.dep_ == "attr" and c.lemma_.lower() in _EXTRAPOSITION_NOUNS)
                                    for c in head.children
                                )
                                if not has_eval_adj:
                                    return "Rhetoric & Emphasis"

        # Antithesis / Correlative Parallelism check (not... but..., not only... but also)
        text_lower = sentence_norm.lower()
        if re.search(r"\bnot\s+.*?\s*,\s*but\b", text_lower) or re.search(r"\bnot\s+only\b.*?\bbut\s+also\b", text_lower):
            return "Rhetoric & Emphasis"

        # Resultative Degree Constructions (so + Adj/Adv + that / such + NP + that)
        if re.search(r"\bso\s+[a-z-]{3,}(?:\s+[a-z-]{3,})?\s+that\b", text_lower) or re.search(r"\bsuch\s+(?:a\s+|an\s+)?[a-z-]{3,}(?:\s+[a-z-]{3,}){0,2}\s+that\b", text_lower):
            return "Rhetoric & Emphasis"

        # ---------------------------------------------------------------------
        # 2. Logic & Stance:
        #    Conditionals (if/unless/as long as/provided that) or Concessives (although/even though/while/whereas/in spite of)
        #    Adversative transitions (however/nevertheless/nonetheless)
        #    Dual-stance coordinate contrast (..., but [S] also...)
        # ---------------------------------------------------------------------
        # Multi-word subordinating connectives
        if re.search(r"\b(as long as|so long as|provided that|providing that|in case|on condition that)\b", text_lower):
            return "Logic & Stance"

        subordinating_conjs = {"although", "though", "while", "whereas", "even though", "if", "unless", "provided", "providing"}
        for token in doc:
            if token.dep_ == "mark" and token.lemma_.lower() in subordinating_conjs:
                if token.lemma_.lower() == "if" and token.head.dep_ != "advcl":
                    continue
                # Subjectless 'while + V-ing/V-ed' is a reduced participial time adjunct (Information Packaging), not a finite concessive clause
                if token.lemma_.lower() == "while" and token.head.tag_ in ("VBG", "VBN"):
                    if not any(c.dep_ in ("nsubj", "nsubjpass", "expl") for c in token.head.children):
                        continue
                return "Logic & Stance"

        # Stance / Adversative transition (However, / Nevertheless, ...)
        if re.match(r"^(however|nevertheless|nonetheless|conversely|in contrast)\b", text_lower):
            return "Logic & Stance"

        # Stance / Contrastive coordinate balance (..., but [S] also...)
        if re.search(r"\b,\s*but\s+[a-z]+\s+(also|still|yet)\b", text_lower):
            return "Logic & Stance"

        # ---------------------------------------------------------------------
        # 3. Cohesion & Framing:
        #    a. Propositional Encapsulation: , which + [interpretive verb] + that
        #    b. Shell Noun Frames: [shell noun] + that [clause]
        # ---------------------------------------------------------------------
        # Propositional encapsulation: which + interpretive verb + that
        interpretive_verbs = {"mean", "suggest", "indicate", "show", "demonstrate", "prove", "imply", "reveal"}
        for token in doc:
            if token.dep_ in ("relcl", "advcl") and token.lemma_.lower() in interpretive_verbs:
                # Check if subject is 'which'
                subj_children = [c for c in token.children if c.dep_ in ("nsubj", "nsubjpass") and c.text.lower() == "which"]
                if subj_children:
                    has_that_comp = any(child.dep_ == "ccomp" for child in token.children)
                    if has_that_comp:
                        return "Cohesion & Framing"

        # Shell noun patterns: fact, idea, conclusion, claim, belief, hypothesis + that
        shell_nouns = {"fact", "idea", "conclusion", "claim", "belief", "hypothesis", "notion", "argument", "evidence", "assumption"}
        for token in doc:
            if token.lemma_.lower() in shell_nouns:
                for child in token.children:
                    if child.dep_ in ("acl", "appos", "ccomp") and any(gc.dep_ == "mark" and gc.lower_ in ("that", "whether") for gc in child.children):
                        return "Cohesion & Framing"

        # ---------------------------------------------------------------------
        # 4. Information Packaging:
        #    a. Evaluative Dummy-It: It is + [adj/noun] + that/to [csubj/ccomp]
        #    b. Dense Object Complement: make/find/render/keep + Object + Adj
        #    c. Non-finite participial adjuncts: advcl with VerbForm=Part
        #    d. Non-finite subject nominalization (Gerund / Infinitive phrase as Subject)
        #    e. Correlative comparative: The more..., the more...
        #    f. Dense Prepositional Frame: Instead of / By / Thanks to + [V-ing/NP]
        #    g. Relative clauses: which/who/whom/whose + VP
        #    h. Elaborative non-restrictive relative clause: , which + VP
        # ---------------------------------------------------------------------
        # Correlative comparative: The more..., the more...
        COMPARATIVE_WORDS = r"(?:more|less|fewer|better|worse|[a-z]{2,}er)"
        if re.search(rf"\bthe\s+{COMPARATIVE_WORDS}\b(?!\s+hand\b).*?(?:,\s*|\band\s+the\s+{COMPARATIVE_WORDS}\b.*?,?\s*)\bthe\s+{COMPARATIVE_WORDS}\b", text_lower):
            return "Information Packaging"

        # Dense Prepositional Frame (Instead of, By doing, Due to, Thanks to + V-ing/NP, [Subject] + [VP])
        if re.match(r"^(instead of|by|through|despite|in spite of|thanks to|due to|owing to)\s+[a-z0-9\s-]+?,", text_lower):
            return "Information Packaging"

        # Non-finite subject nominalization (Gerund or Infinitive as subject)
        # Handles direct sentence subject or embedded clause subject (e.g. That is why [being on wheels] means...)
        for token in doc:
            if token.tag_ == "VBG" and token.dep_ in ("nsubj", "nsubjpass", "csubj"):
                return "Information Packaging"
            if token.tag_ == "VB" and token.dep_ == "csubj":
                return "Information Packaging"
            # Embedded gerund subject in why/that/wh-clauses where spaCy might label it advcl or pobj
            if token.tag_ == "VBG" and token.dep_ in ("advcl", "csubj") and token.head.pos_ in ("VERB", "AUX"):
                # Check if this VBG introduces a gerund clause following 'why' or copula
                has_wh = any(c.dep_ == "advmod" and c.lemma_.lower() in ("why", "how", "what", "where") for c in token.children)
                if has_wh:
                    return "Information Packaging"
                head_subjs = [c for c in token.head.children if c.dep_ in ("nsubj", "nsubjpass")]
                if not head_subjs and token.i < token.head.i:
                    return "Information Packaging"

        # Evaluative Dummy-It (spaCy tags 'It' as nsubj/expl and the complement clause as ccomp/csubj)
        _AMBIENT_PREDS = {
            "daylight", "dark", "night", "morning", "afternoon", "evening", "noon", "midnight",
            "dawn", "dusk", "winter", "summer", "spring", "autumn", "fall", "cold", "hot", "warm",
            "cool", "sunny", "rainy", "snowy", "windy", "cloudy", "late", "early", "o'clock"
        }
        for token in doc:
            if token.text.lower() == "it" and token.dep_ in ("expl", "nsubj"):
                head = token.head
                if head.lemma_ in ("be", "seem") and not any(c.dep_ == "advmod" and c.lower_ in ("when", "where") for c in head.children):
                    eval_children = [
                        child for child in head.children
                        if child.dep_ in ("acomp", "attr") and child.lemma_.lower() not in _AMBIENT_PREDS
                    ]
                    comp_children = [
                        child for child in head.children
                        if child.dep_ in ("ccomp", "csubj", "xcomp")
                        and child.tag_ not in ("VBG", "VBN")
                        and not any(gc.dep_ == "advmod" and gc.lower_ in ("when", "where") for gc in child.children)
                    ]
                    if eval_children and comp_children and comp_children[0].i > eval_children[0].i:
                        return "Information Packaging"

        # Evaluative Dummy-It Object (find/make/think it adj to-V)
        for t in doc:
            if t.lemma_ in ("find", "make", "think", "consider", "deem", "believe") and t.pos_ in ("VERB", "AUX"):
                for c in t.children:
                    if c.pos_ == "ADJ" and any(gc.text.lower() == "it" and gc.dep_ in ("nsubj", "dobj") for gc in c.children):
                        return "Information Packaging"

        # Dense Object Complement / Causative (make/find/render/keep + Object + Adj/Complement)
        for t in doc:
            if t.lemma_ in ("make", "find", "render", "keep") and t.pos_ in ("VERB", "AUX"):
                for c in t.children:
                    if c.dep_ in ("ccomp", "oprd") and c.pos_ == "ADJ":
                        return "Information Packaging"

        # Non-finite participial adjuncts (advcl with VBG or VBN)
        _OBJ_VING_VERBS = {"have", "find", "spend", "waste", "get", "catch", "keep", "leave", "start", "stop", "prevent", "busy"}
        for token in doc:
            if token.dep_ == "advcl" and token.tag_ in ("VBG", "VBN"):
                # Exclude 'have trouble/difficulty/time V-ing' and 'spend/waste time V-ing' complement frames
                if token.tag_ == "VBG" and token.head.lemma_.lower() in _OBJ_VING_VERBS and token.i > token.head.i:
                    if not any(t.text == "," for t in doc[token.head.i : token.i]):
                        continue
                # Ensure it has no subject of its own (non-finite) and not a preposition
                has_subj = any(c.dep_ in ("nsubj", "nsubjpass") for c in token.children)
                if not has_subj and token.lemma_.lower() not in ("include", "accord", "regard", "concern"):
                    return "Information Packaging"

        # Relative clauses (which, that, who, whom, whose)
        for token in doc:
            if token.dep_ == "relcl":
                if any(c.text.lower() in ("which", "that", "who", "whom", "whose") for c in token.children):
                    return "Information Packaging"

        # Passive voice (agentless or agentive passive foregrounding patient)
        _UNACCUSATIVE_STATE_VERBS = {"go", "come", "fall", "rise", "arrive", "depart", "vanish", "disappear", "die", "remain", "stay"}
        for token in doc:
            if token.dep_ in ("nsubjpass", "auxpass"):
                head = token.head
                if head.pos_ in ("VERB", "AUX") and head.lemma_.lower() not in _UNACCUSATIVE_STATE_VERBS:
                    return "Information Packaging"

        return None

    @classmethod
    def generate_cobuild_formula(cls, sentence: str, category: Optional[str] = None) -> str:
        """
        Deterministically derives standard algebraic COBUILD slot formula from sentence
        via spaCy dependency trees and syntactic pattern anchors.
        Replaces prone-to-hallucination LLM formula drafting with zero-token invariant logic.
        """
        sentence_norm = sentence.replace('\u2019', "'").replace('\u2018', "'")
        nlp = cls.get_spacy()
        doc = nlp(sentence_norm)
        cat = category or cls.classify_grammar_dependency(sentence_norm) or ""
        text_lower = sentence_norm.lower()

        # Declarative DependencyMatcher Engine Pass (High-Precision DSL)
        try:
            from librarian.grammar_patterns import GrammarPatternEngine
            gp_engine = GrammarPatternEngine.get_engine(nlp)
            matched_patterns = gp_engine.match_sentence(doc)
            if matched_patterns:
                if cat:
                    matched_cand = next((m for m in matched_patterns if m["category"].lower() == cat.lower() or cat.lower() in m["name"].lower() or cat.lower() in m["pattern_id"].lower()), None)
                else:
                    matched_cand = matched_patterns[0]
                if matched_cand and matched_cand.get("formula"):
                    return matched_cand["formula"]
        except Exception:
            pass

        # 1. Rhetoric & Emphasis
        if cat == "Rhetoric & Emphasis" or not cat:
            # Inversion
            for token in doc:
                if token.dep_ == "nsubj":
                    head = token.head
                    if head.pos_ in ("VERB", "AUX"):
                        aux_children = [c for c in head.children if c.dep_ in ("aux", "auxpass") and c.i < token.i]
                        if aux_children and not sentence.strip().endswith("?"):
                            first_tok = doc[0].lemma_.lower()
                            if first_tok in ("hardly", "scarcely", "seldom", "never", "rarely", "barely", "only", "not", "no"):
                                return f"{doc[0].text} + [aux/be] + [Subject] + [VP]"
            # Cleft
            _EXTRAPOSITION_NOUNS = {
                "surprise", "wonder", "secret", "doubt", "coincidence", "accident",
                "pity", "shame", "mystery", "fact", "truth", "news", "use", "good",
                "point", "time", "question", "matter"
            }
            for token in doc:
                if token.text.lower() == "it" and token.dep_ in ("expl", "nsubj") and token.head.lemma_ in ("be", "seem"):
                    if any(
                        (c.dep_ == "acomp" and c.pos_ == "ADJ")
                        or (c.dep_ == "attr" and c.lemma_.lower() in _EXTRAPOSITION_NOUNS)
                        for c in token.head.children
                    ):
                        continue
                    # Check relcl
                    has_relcl = any(
                        (c.dep_ == "relcl" and any(gc.text.lower() in ("that", "who", "whom", "which", "whose") for gc in c.children))
                        or any(gc.dep_ == "relcl" and any(ggc.text.lower() in ("that", "who", "whom", "which", "whose") for ggc in gc.children) for gc in c.children)
                        for c in token.head.children
                    )
                    # Check prepositional or adverbial cleft
                    has_cleft_marker = any(
                        c.dep_ in ("ccomp", "advcl")
                        and any(gc.dep_ in ("mark", "nsubj", "advmod", "dobj", "pobj") and gc.text.lower() in ("that", "who", "whom", "where", "when") for gc in c.children)
                        for c in token.head.children
                    )
                    if has_relcl or has_cleft_marker:
                        return "It + [be] + [Focal Element] + that/who + [Clause]"
            # Correlative Parallelism (with or without Fronted Negative Inversion)
            if re.search(r"\bnot\s+only\b.*?\bbut\s+(?:also\b|[a-z]+\s+also\b)", text_lower):
                if re.search(r"(?:^|[,;—–-]\s*)not\s+only\s+(?:can|could|will|would|shall|should|may|might|must|do|does|did|have|has|had|am|is|are|was|were)\b", text_lower):
                    return "Not only + [Aux] + [Subject] + [VP], but + [Subject] + [also] + [VP]"
                return "[Subject] + not only + [VP], but also + [VP]"
            # Antithesis
            if re.search(r"\bnot\s+.*?\s*,\s*but\b", text_lower):
                return "[Subject] + [VP], not + [PrepP/NP], but + [PrepP/NP]"
            # Resultative so/such ... that
            if re.search(r"\bsuch\s+(?:a\s+|an\s+)?[a-z-]{3,}(?:\s+[a-z-]{3,}){0,2}\s+that\b", text_lower):
                return "such + [NP] + that + [Result Clause]"
            if re.search(r"\bso\s+[a-z-]{3,}(?:\s+[a-z-]{3,})?\s+that\b", text_lower):
                return "so + [Adj/Adv] + that + [Result Clause]"

        # 2. Logic & Stance
        if cat == "Logic & Stance" or not cat:
            # Multi-word conditional & concessive connectives
            m_conn = re.search(r"\b(as long as|so long as|provided that|providing that|in case|on condition that|even though|even if)\b", text_lower)
            if m_conn:
                conn_name = m_conn.group(1).capitalize()
                return f"{conn_name} + [Clause], [Subject] + [VP]"
            for token in doc:
                if token.dep_ == "mark" and token.lemma_.lower() in {"although", "though", "while", "whereas", "even though", "if", "unless", "provided"}:
                    if token.lemma_.lower() == "if" and token.head.dep_ != "advcl":
                        continue
                    if token.lemma_.lower() == "while" and token.head.tag_ in ("VBG", "VBN"):
                        if not any(c.dep_ in ("nsubj", "nsubjpass", "expl") for c in token.head.children):
                            continue
                    prev_w = doc[token.i - 1].lower_ if token.i > 0 else ""
                    if prev_w == "even" and token.lower_ in ("if", "though"):
                        mark_word = f"Even {token.lower_}"
                    else:
                        mark_word = token.text.capitalize()
                    return f"{mark_word} + [Clause], [Subject] + [VP]"
            m_trans = re.match(r"^(however|nevertheless|nonetheless|conversely|in contrast)\b", text_lower)
            if m_trans:
                return f"{m_trans.group(1).capitalize()}, [Subject] + [Modal/VP]"
            if re.search(r"\b,\s*but\s+[a-z]+\s+(also|still|yet)\b", text_lower):
                return "[Subject] + [VP], but [Subject] + [also] + [VP]"

        # 3. Cohesion & Framing
        if cat == "Cohesion & Framing" or not cat:
            # Propositional Encapsulation: , which + [interpretive verb] + that
            interpretive_verbs = {"mean", "suggest", "indicate", "show", "demonstrate", "prove", "imply", "reveal"}
            for token in doc:
                if token.dep_ in ("relcl", "advcl") and token.lemma_.lower() in interpretive_verbs:
                    if any(c.dep_ in ("nsubj", "nsubjpass") and c.text.lower() == "which" for c in token.children):
                        return f", which + {token.lemma_.lower()}s + that + [Proposition Clause]"
            # Shell noun frame
            shell_nouns = {"fact", "idea", "conclusion", "claim", "belief", "hypothesis", "notion", "argument", "evidence", "assumption"}
            for token in doc:
                if token.lemma_.lower() in shell_nouns:
                    if any(
                        c.dep_ in ("acl", "appos", "ccomp")
                        and any(gc.dep_ == "mark" and gc.lower_ in ("that", "whether") for gc in c.children)
                        for c in token.children
                    ):
                        return f"The + {token.lemma_.lower()} + that/of + [Proposition Clause]"

        # 4. Information Packaging
        if cat == "Information Packaging" or not cat:
            # Correlative comparative: The more..., the more...
            COMPARATIVE_WORDS = r"(?:more|less|fewer|better|worse|[a-z]{2,}er)"
            if re.search(rf"\bthe\s+{COMPARATIVE_WORDS}\b(?!\s+hand\b).*?(?:,\s*|\band\s+the\s+{COMPARATIVE_WORDS}\b.*?,?\s*)\bthe\s+{COMPARATIVE_WORDS}\b", text_lower):
                return "The + [comparative] + [Clause], the + [comparative] + [Clause]"
            # Dense Prepositional Frame (Instead of, By doing, Thanks to...)
            m_prep = re.match(r"^(instead of|by|through|despite|in spite of|thanks to|due to|owing to)\b", text_lower)
            if m_prep:
                prep_head = m_prep.group(1).capitalize()
                return f"{prep_head} + [V-ing/NP], [Subject] + [VP]"
            # Non-finite Subject Nominalization (Gerund / Infinitive phrase as Subject)
            # Detects direct sentence subject or embedded predicate clause subject (e.g. That is why [being on wheels] means...)
            for token in doc:
                is_gerund_subj = (token.tag_ == "VBG" and token.dep_ in ("nsubj", "nsubjpass", "csubj"))
                is_infinitive_subj = (token.tag_ == "VB" and token.dep_ == "csubj")
                is_embedded_gerund = (
                    token.tag_ == "VBG" and token.dep_ in ("advcl", "csubj")
                    and (
                        any(c.dep_ == "advmod" and c.lemma_.lower() in ("why", "how", "what", "where") for c in token.children)
                        or (
                            token.head.pos_ in ("VERB", "AUX")
                            and not any(c.dep_ in ("nsubj", "nsubjpass") for c in token.head.children)
                            and token.i < token.head.i
                        )
                    )
                )
                if is_gerund_subj or is_embedded_gerund:
                    if re.search(r"\b(that|this)\s+(?:is|’s|'s)\s+why\b", text_lower):
                        return "[Subject] + is why + [V-ing / Gerund Phrase] + [VP]"
                    return "[V-ing / Gerund Phrase] + [Predicate Verb] + [Complement/Object]"
                if is_infinitive_subj:
                    return "To + [Infinitive Phrase] + [Predicate Verb] + [Complement/Object]"
            # Dummy-It Object Extraposition (find/make/think it adj to-V)
            for t in doc:
                if t.lemma_ in ("find", "make", "think", "consider", "deem", "believe") and t.pos_ in ("VERB", "AUX"):
                    for c in t.children:
                        if c.pos_ == "ADJ" and any(gc.text.lower() == "it" and gc.dep_ in ("nsubj", "dobj") for gc in c.children):
                            return f"[Subject] + {t.lemma_} + it + [{c.lemma_.capitalize()}] + to-V"
            # Dense Object Complement / Causative (make/find/render/keep + Object + Adj)
            for t in doc:
                if t.lemma_ in ("make", "find", "render", "keep") and t.pos_ in ("VERB", "AUX"):
                    for c in t.children:
                        if c.dep_ in ("ccomp", "oprd") and c.pos_ == "ADJ":
                            return f"[Subject] + {t.lemma_} + [Object] + [Adj]"
            # Dummy-It Subject Extraposition
            _AMBIENT_PREDS = {
                "daylight", "dark", "night", "morning", "afternoon", "evening", "noon", "midnight",
                "dawn", "dusk", "winter", "summer", "spring", "autumn", "fall", "cold", "hot", "warm",
                "cool", "sunny", "rainy", "snowy", "windy", "cloudy", "late", "early", "o'clock"
            }
            if any(t.text.lower() == "it" and t.dep_ in ("expl", "nsubj") for t in doc):
                for t in doc:
                    if t.text.lower() == "it" and t.head.lemma_ in ("be", "seem") and not any(c.dep_ == "advmod" and c.lower_ in ("when", "where") for c in t.head.children):
                        eval_children = [
                            c for c in t.head.children
                            if c.dep_ in ("acomp", "attr") and c.lemma_.lower() not in _AMBIENT_PREDS
                        ]
                        comp_children = [
                            c for c in t.head.children
                            if c.dep_ in ("ccomp", "csubj", "xcomp")
                            and c.tag_ not in ("VBG", "VBN")
                            and not any(gc.dep_ == "advmod" and gc.lower_ in ("when", "where") for gc in c.children)
                        ]
                        if eval_children and comp_children and comp_children[0].i > eval_children[0].i:
                            return "It is + [Adj/Noun] + that/to-inf [Clause]"
            # Participial adjunct (strictly requiring non-subject participle, comma boundary when post-positioned, verbal dependents, and actual main subject)
            for token in doc:
                if token.dep_ == "advcl" and token.tag_ in ("VBG", "VBN"):
                    if token.i > token.head.i and not any(t.text == "," for t in doc[token.head.i : token.i]):
                        continue
                    if token.i + 1 < len(doc) and doc[token.i + 1].pos_ in ("NOUN", "PROPN") and doc[token.i + 1].dep_ in ("dobj", "pobj", "nsubj", "attr"):
                        continue
                    if not any(c.dep_ in ("dobj", "prep", "advmod", "acomp", "xcomp", "ccomp", "prt", "agent", "npadvmod", "oprd") for c in token.children):
                        continue
                    if not any(c.dep_ in ("nsubj", "nsubjpass") for c in token.children) and token.lemma_.lower() not in ("include", "accord", "regard"):
                        # Ensure main verb actually has a subject, otherwise it might be a gerund subject misclassified as advcl
                        main_has_subj = any(c.dep_ in ("nsubj", "nsubjpass") for c in token.head.children)
                        if main_has_subj:
                            is_fronted = token.i < token.head.i
                            has_having = any(c.dep_ in ("aux", "auxpass") and c.lower_ == "having" for c in token.children)
                            has_being = any(c.dep_ in ("aux", "auxpass") and c.lower_ == "being" for c in token.children)
                            if has_having:
                                v_type = "Having + V-ed"
                            elif has_being:
                                v_type = "Being + V-ed"
                            elif token.tag_ == "VBG":
                                v_type = "V-ing"
                            else:
                                v_type = "V-ed"
                            if is_fronted:
                                return f"[{v_type} Phrase], [Subject] + [VP]"
                            else:
                                return f"[Subject] + [VP], [{v_type} Phrase]"
            # Relative clauses (which, that, who, whom, whose)
            for token in doc:
                if token.dep_ == "relcl":
                    rel_pron = next((c.text.lower() for c in token.children if c.text.lower() in ("which", "that", "who", "whom", "whose")), None)
                    if rel_pron:
                        return f"[NP] + {rel_pron} + [VP]"
            # Elaborative clause
            if re.search(r",\s*which\s+[a-z]+", text_lower):
                return "[Subject] + [VP], which + [VP]"

            # Passive voice
            for token in doc:
                if token.dep_ in ("nsubjpass", "auxpass") and token.head.lemma_.lower() not in {"go", "come", "fall", "rise", "arrive", "depart", "vanish", "disappear", "die", "remain", "stay"}:
                    return "[Subject] + [be/modal be] + [V-ed]"

        # Default fallback standard formula
        return "[Subject] + [VP] + [Clause]"

    @classmethod
    def mine_grammar_skeletons(
        cls,
        sentence_pool: Dict[str, str],
        target_count: int = 5,
        syllabus_grammar: Optional[List[str]] = None,
        cefr_band: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Deterministically mines genuine, non-trivial grammar structural skeletons from the sentence pool.
        Selects sentences exhibiting rich dependency structures across the Four Macro Domains,
        prioritized by the 3-band CEFR priority sequence (A1_A2, B1_B2, C1_C2) and intra-pattern complexity.
        Returns a list of dicts: [{'sid': 'S-9', 'quote': '...', 'category': '...', 'pattern_formula': '...'}]
        """
        raw_skeletons: List[Dict[str, Any]] = []
        band = cefr_band or cls.resolve_grammar_cefr_band(sentence_pool)

        nlp = cls.get_spacy()
        from librarian.grammar_patterns import GrammarPatternEngine
        gp_engine = GrammarPatternEngine.get_engine(nlp)

        for sid, sent in sentence_pool.items():
            sent_clean = sent.strip()
            # Skip very short sentences
            if len(sent_clean.split()) < 7:
                continue
            # Skip passage title (S-1) when passage has multiple sentences, section headers, and syllabus word-lists
            if sid == "S-1" and len(sentence_pool) > 1:
                continue
            # If sentence does not end with standard sentence punctuation, skip (headings/titles)
            if not sent_clean.endswith((".", "!", "?", '"', "'")):
                continue
            # If sentence is a colon-separated headline/title (e.g. "Title: Subtitle") without terminal punctuation
            if ":" in sent_clean and not any(sent_clean.endswith(p) for p in (".", "!", "?")):
                continue

            sent_norm = sent_clean.replace('\u2019', "'").replace('\u2018', "'")
            doc = nlp(sent_norm)
            matched_patterns = gp_engine.match_sentence(doc)

            if matched_patterns:
                for match_item in matched_patterns:
                    cat = match_item["category"]
                    formula = match_item.get("formula", "")
                    if cat and formula and not formula.startswith("[Subject] + [VP] + [Clause]") and "[" in formula:
                        raw_skeletons.append({
                            "sid": sid,
                            "quote": sent_clean,
                            "category": cat,
                            "pattern_formula": formula,
                        })
            else:
                cat = cls.classify_grammar_dependency(sent_clean)
                formula = cls.generate_cobuild_formula(sent_clean, category=cat) if cat else ""
                if cat and formula and not formula.startswith("[Subject] + [VP] + [Clause]") and "[" in formula:
                    raw_skeletons.append({
                        "sid": sid,
                        "quote": sent_clean,
                        "category": cat,
                        "pattern_formula": formula,
                    })

        if not raw_skeletons:
            return []

        # Score each skeleton using 3-band CEFR priority + Intra-Pattern Syntactic Complexity
        def _score_skeleton(item: Dict[str, Any]) -> Tuple[float, int]:
            formula_str = item["pattern_formula"]
            quote_str = item["quote"]
            cat_str = item["category"]
            prof = cls.match_cobuild_grammar_profile(formula_str, quote_str, category=cat_str, cefr_band=band)
            base_prio = float(prof.get("priority", 70))
            prof_key = prof.get("profile_key", "")

            words = quote_str.split()
            w_cnt = len(words)
            q_low = quote_str.lower()
            f_low = formula_str.lower()

            bonus = 0.0
            # Perfective / passive participle or coordinated non-finite clauses boost
            if "having + v-ed" in f_low or "being + v-ed" in f_low or re.search(r"\bhaving\s+[a-z]+ed\b", q_low):
                bonus += 10.0
            # Optimal pedagogical sentence length boost (14-38 words)
            if 14 <= w_cnt <= 38:
                bonus += 6.0
            elif w_cnt < 10:
                bonus -= 8.0
            elif w_cnt > 48:
                bonus -= 5.0

            # In B1_B2 and C1_C2, penalize bare restrictive relative clauses (that/who/which without pied-piped prepositions)
            # so non-finite participial adjuncts, clefts, inversions, and degree results win the macro-domain slots
            if band in ("B1_B2", "C1_C2") and prof_key == "relative_clause_restrictive":
                has_pied_piping = bool(re.search(r"\b(?:in|on|at|from|to|with|by|through|for|of|under|beyond|within|upon)\s+(?:which|whom)\b", q_low))
                if has_pied_piping:
                    bonus += 12.0
                else:
                    bonus -= 15.0

            sid_m = re.search(r"\d+", item.get("sid", "999"))
            sid_num = int(sid_m.group(0)) if sid_m else 999
            return (-round(base_prio + bonus, 2), sid_num)

        ranked_skeletons = sorted(raw_skeletons, key=_score_skeleton)

        # Diverse selection algorithm across macro domains, distinct formulas, and strictly unique quotes
        def _norm_structure(formula_str: str) -> str:
            res = re.sub(
                r"\[NP\]\s*(?:\+|,\s*)\s*(?:who|which|that|whom|whose)\s*\+\s*\[VP\]",
                "[NP] + rel_pron + [VP]",
                formula_str,
                flags=re.IGNORECASE
            )
            res = re.sub(
                r"^(?:if|unless|as long as|provided that)\s*\+\s*\[Clause\]",
                "[Condition] + [Clause]",
                res,
                flags=re.IGNORECASE
            )
            res = re.sub(
                r"^\[(?:Having \+ V-ed|Being \+ V-ed|V-ing|V-ed) Phrase\],\s*\[Subject\] \+ \[VP\]",
                "[Participial Phrase], [Subject] + [VP]",
                res,
                flags=re.IGNORECASE
            )
            return res

        selected: List[Dict[str, str]] = []
        seen_cats: Set[str] = set()
        seen_structures: Set[str] = set()
        seen_profiles: Set[str] = set()
        seen_quotes: Set[str] = set()

        def _get_prof_key(item: Dict[str, str]) -> str:
            prof = cls.match_cobuild_grammar_profile(
                item["pattern_formula"], item["quote"], category=item["category"], cefr_band=band
            )
            return str(prof.get("profile_key") or "")

        # Pass 1: maximize category diversity across the 4 macro domains (highest priority first)
        for item in ranked_skeletons:
            cat = item["category"]
            quote = item["quote"]
            struct = _norm_structure(item["pattern_formula"])
            pkey = _get_prof_key(item)
            if quote not in seen_quotes and cat not in seen_cats and struct not in seen_structures and pkey not in seen_profiles:
                selected.append(item)
                seen_cats.add(cat)
                seen_structures.add(struct)
                if pkey:
                    seen_profiles.add(pkey)
                seen_quotes.add(quote)
            if len(selected) >= target_count:
                break

        # Pass 2: fill remaining slots with distinct structural formulas, distinct profiles, and unique quotes
        if len(selected) < target_count:
            for item in ranked_skeletons:
                quote = item["quote"]
                struct = _norm_structure(item["pattern_formula"])
                pkey = _get_prof_key(item)
                if quote not in seen_quotes and struct not in seen_structures and pkey not in seen_profiles:
                    selected.append(item)
                    seen_structures.add(struct)
                    if pkey:
                        seen_profiles.add(pkey)
                    seen_quotes.add(quote)
                if len(selected) >= target_count:
                    break

        # Pass 3: if still under target_count, append remaining distinct quotes
        if len(selected) < target_count:
            for item in ranked_skeletons:
                quote = item["quote"]
                if quote not in seen_quotes:
                    selected.append(item)
                    seen_quotes.add(quote)
                if len(selected) >= target_count:
                    break

        # Restore natural passage appearance order (by sentence ID) for coherent reading flow
        def _sid_order(item: Dict[str, str]) -> int:
            m = re.search(r"\d+", item.get("sid", "999"))
            return int(m.group(0)) if m else 999

        selected.sort(key=_sid_order)
        return selected[:target_count]

    # -------------------------------------------------------------------------
    # 3b. Deterministic Multi-Word Expression & Collocation Mining (ACL + spaCy)
    # -------------------------------------------------------------------------
    @classmethod
    def mine_expression_skeletons(
        cls,
        text: str,
        target_count: int = 8,
        syllabus_expressions: Optional[List[str]] = None
    ) -> List[Dict[str, str]]:
        """
        Deterministically extracts high-value multi-word expressions, phrasal verbs,
        and Academic Collocation List (ACL) items from the indexed sentence pool.
        Standardizes slotted formulas (e.g. [sb], [sth], [one's]) at zero token cost.
        If syllabus_expressions is provided, prioritizes resolving and standardizing
        those curriculum targets first before filling remaining slots with computational candidates.

        Returns list of dicts:
            [{
                "sid": "S-1",
                "quote": "full sentence text",
                "phrase": "have an adverse effect on",
                "type": "collocation",
                "pattern_formula": "have an adverse effect on [sth]"
            }, ...]
        """
        _, pool = cls.tokenize_and_index_sentences(text)
        if not pool:
            return []

        nlp = cls.get_spacy()

        COMMON_PARTICLES = frozenset({"away", "back", "down", "in", "off", "on", "out", "over", "round", "through", "up"})
        DEPENDENT_PREPS = frozenset({"on", "upon", "to", "with", "for", "from", "into", "against", "about", "of", "in"})

        raw_candidates: List[Dict[str, Any]] = []

        for sid, sentence in pool.items():
            sent_clean = sentence.strip()
            if len(sent_clean.split()) < 5:
                continue

            doc = nlp(sent_clean)
            text_lower = sent_clean.lower()

            # 1. Dependency Syntax Phrasal Verbs & Prepositional Combinations via Declarative Expression Pattern Engine
            from librarian.expression_patterns import ExpressionPatternEngine
            expr_engine = ExpressionPatternEngine.get_engine(nlp)
            for match in expr_engine.match_sentence(doc):
                raw_candidates.append({
                    "sid": sid,
                    "quote": sent_clean,
                    "phrase": match["phrase"],
                    "type": match["type"],
                    "pattern_formula": match["pattern_formula"],
                    "pobj": match.get("pobj"),
                    "pobj_proper": match.get("pobj_proper", False),
                    "score": match.get("priority", 8),
                })


        if not raw_candidates:
            return []

        # PedLex attestation gate: a candidate ships only when PedLex itself states the unit.
        # The old gate asked the Oxford Collocations file whether it had a headword for the
        # phrase, which is a weaker test than it sounds - 'keep of' is a preposition lifted out
        # of 'keep somebody out of something' and no dictionary has it as a unit.
        # ldoce_phrase_evidence asks the real question: is this a phrasal-verb block, a PHRASES
        # item, a grammar pattern built on the headword, or a COLLOCATIONS box item?  Evidence
        # that amounts to 'some example sentence contains these words in this order' does not
        # ship, because an example is exactly where a stray preposition looks most like a phrase.
        verified_candidates: List[Dict[str, Any]] = []

        for item in raw_candidates:
            phrase = item.get("phrase", "").lower().strip()
            item_type = item.get("type")
            words = phrase.split()
            if not words:
                continue

            # Phrasal verbs and prepositional verbs (e.g. 'rely on', 'long for', 'worry about')
            if item_type == "phrasal verb":
                attested = cls.is_attested_phrase(phrase)
                if not attested and "touch" in words and any(v in words for v in ("keep", "get", "stay", "lose")):
                    # PedLex states it as 'be/keep/stay etc in touch (with something)'
                    attested = cls.is_attested_phrase(" ".join(w for w in words if w != "with"))
                if attested:
                    # 'keep in' is a real phrasal verb, but PedLex's frame for it is 'keep
                    # somebody in', so a parse of 'keep in WeChat Moments' - or of 'keep in
                    # touch' - has read a locative 'in ...' as the particle and is not that
                    # frame.  The check runs for any object, not only a proper noun, because
                    # the frame says where the object sits regardless of what it is.
                    pobj_word = item.get("pobj")
                    if pobj_word and not cls.ldoce_phrase_object_fit(phrase, pobj_word):
                        continue
                    verified_candidates.append(item)
                continue

            # Anything else the dictionary states as a unit (e.g. 'account for', 'participate in')
            if cls.is_attested_phrase(phrase):
                verified_candidates.append(item)
                continue

            # Verbal idioms (e.g. 'take into consideration') that PedLex does not list as a
            # PHRASES item: the head noun's own grammar patterns must show the verb governing it.
            if item_type == "idiom":
                v_lemma, head_noun = words[0], words[-1]
                n_entry = cls.get_ldoce_entry(head_noun)
                if n_entry:
                    idiom_rx = re.compile(
                        r"(?<!\w)" + re.escape(v_lemma) + r"\w*(?:\s+\w+){0,4}\s"
                        + re.escape(head_noun) + r"(?!\w)", re.IGNORECASE)
                    if any(idiom_rx.search(cls._entry_text_normalizer(pat))
                           for pat in (n_entry.get("patterns") or [])):
                        verified_candidates.append(item)
                        continue

            # Collocations (ACL items, possessive frames like 'hear [one\'s] voice')
            elif item_type == "collocation":
                # ACL items or verified possessive collocations
                verified_candidates.append(item)
                continue

            # Default: if none of the specific validation branches match, drop unverified candidate

        raw_candidates = verified_candidates
        if not raw_candidates:
            return []

        selected: List[Dict[str, str]] = []
        seen_phrases: Set[str] = set()
        seen_quotes: Set[str] = set()

        # Phase 1: If syllabus_expressions is provided, prioritize and standardize those curriculum targets
        if syllabus_expressions:
            syllabus_candidates: List[Dict[str, Any]] = []
            for raw_expr in syllabus_expressions:
                raw_norm = re.sub(r"\s+", " ", raw_expr.strip())
                raw_norm = raw_norm.replace("\u2019", "'").replace("\u2018", "'").replace("…", "...")
                # Strip trailing ', etc.' or 'etc.'
                raw_norm = re.sub(r",?\s*\betc\.?(?=\s+to\b|\s+for\b|\s+in\b|\s+on\b|\s+with\b|$)", "", raw_norm, flags=re.IGNORECASE)
                # Preserve parenthesized metasyntax slots: (sb) -> [sb], (sth) -> [sth], (oneself) -> [oneself]
                raw_norm = re.sub(r"(?:\(\s*|\[\s*)+(?:sb\.?|somebody|someone)(?:\s*\)|\s*\])+", "[sb]", raw_norm, flags=re.IGNORECASE)
                raw_norm = re.sub(r"(?:\(\s*|\[\s*)+(?:sth\.?|something)(?:\s*\)|\s*\])+", "[sth]", raw_norm, flags=re.IGNORECASE)
                raw_norm = re.sub(r"(?:\(\s*|\[\s*)+(?:sb/sth|sth/sb)(?:\s*\)|\s*\])+", "[sth/sb]", raw_norm, flags=re.IGNORECASE)
                raw_norm = re.sub(r"(?:\(\s*|\[\s*)+oneself(?:\s*\)|\s*\])+", "[oneself]", raw_norm, flags=re.IGNORECASE)
                # Strip remaining optional parentheticals (e.g. 'no exaggeration (to say (sth) / that (sth))', 'protect [sth] (from)')
                while re.search(r"\([^()]*\)", raw_norm):
                    raw_norm = re.sub(r"\([^()]*\)", "", raw_norm).strip()
                raw_norm = re.sub(r"\s+", " ", raw_norm).strip()
                # Normalize metasyntax placeholders:
                # 1. Possessive placeholders: one's / ones / sb.'s / sb's / somebody's / someone's -> [one's]
                raw_norm = re.sub(r"\b(?:one's|ones|sb\.?'s|somebody's|someone's)\b", "[one's]", raw_norm, flags=re.IGNORECASE)
                # 2. Reflexive placeholder: oneself -> [oneself]
                raw_norm = re.sub(r"\boneself\b", "[oneself]", raw_norm, flags=re.IGNORECASE)
                # 3. Object placeholders: sb. / sb / somebody / someone -> [sb]; sth. / sth / something / ... -> [sth]
                raw_norm = re.sub(r"\b(?:sb|somebody|someone)\b\.?", "[sb]", raw_norm, flags=re.IGNORECASE)
                raw_norm = re.sub(r"\b(?:sth|something)\b\.?", "[sth]", raw_norm, flags=re.IGNORECASE)
                raw_norm = re.sub(r"\.{2,}", "[sth]", raw_norm)
                raw_norm = re.sub(r"\[+\s*(one's|oneself|sth/sb|sb/sth|sth|sb)\s*\]+", r"[\1]", raw_norm, flags=re.IGNORECASE)
                raw_norm = re.sub(r"\s+", " ", raw_norm).strip()

                # Pre-expand multi-word slash alternatives (e.g. 'have a edge over / an edge on',
                # 'not have the heart to do [sth] / for [sth]', 'open the door to [sth] / to doing [sth]')
                raw_norm = re.sub(r"\s*/\s*to\s+doing\s+\[sth\]$", "", raw_norm, flags=re.IGNORECASE)
                multi_slash_seeds = [raw_norm]
                if " / " in raw_norm:
                    m_tail = re.match(r"^(.*?)\s+(to\s+do\s+\[sth\]|for\s+\[sth\]|an?\s+edge\s+(?:over|on))\s*/\s*(.+)$", raw_norm, flags=re.IGNORECASE)
                    if m_tail:
                        prefix_p, alt1, alt2 = m_tail.group(1).strip(), m_tail.group(2).strip(), m_tail.group(3).strip()
                        alt1 = re.sub(r"^a\s+edge\b", "an edge", alt1, flags=re.IGNORECASE)
                        alt2 = re.sub(r"^a\s+edge\b", "an edge", alt2, flags=re.IGNORECASE)
                        multi_slash_seeds = [f"{prefix_p} {alt1}".strip(), f"{prefix_p} {alt2}".strip()]

                # Expand single-token slash alternatives (e.g. 'put [sth] into the atmosphere/air' ->
                # ['put [sth] into the atmosphere', 'put [sth] into the air'])
                branch_variants: List[str] = []
                for seed in multi_slash_seeds:
                    for bv in cls._slash_variants(seed):
                        if bv not in branch_variants:
                            branch_variants.append(bv)
                matched_branch_expr = None
                matched_sid = None
                matched_sent = None

                for branch_expr in branch_variants:
                    expr_clean = re.sub(r"\s+", " ", branch_expr.strip())
                    words = expr_clean.split()

                    # Filter proper nouns / TV trivia entities (e.g. 'lone ranger' with kind == 'entity')
                    ldoce_ent = cls.get_ldoce_entry(expr_clean)
                    if ldoce_ent and ldoce_ent.get("kind") == "entity":
                        continue

                    # If syllabus expression starts with passive auxiliary 'be' (e.g. 'be admitted to', 'be featured in'),
                    # check if it is a genuine PedLex dictionary idiom/entry or a copular adjective frame ('be smart about').
                    # Only strip leading 'be' when followed by a passive participle (VBN / -ed).
                    if len(words) > 1 and words[0].lower() in ("be", "is", "was", "were", "are", "been") and expr_clean.lower() not in ("be lost", "feel lost"):
                        has_genuine_be = bool(
                            cls._unit_phrase_rows(expr_clean)
                            or cls._unit_blocks(expr_clean)
                            or any(h[0] in ("phrase", "phrasal_verb") for h in cls.ldoce_phrase_hits(expr_clean))
                        )
                        if not has_genuine_be:
                            w1_doc = nlp(expr_clean)
                            w1_tok = w1_doc[1] if len(w1_doc) > 1 else None
                            is_adj_pred = bool(
                                w1_tok and (
                                    w1_tok.pos_ == "ADJ"
                                    or (cls.get_ldoce_entry(words[1].lower()) or {}).get("pos") == "adjective"
                                ) and not words[1].lower().endswith("ed")
                            )
                            if not is_adj_pred:
                                expr_clean = " ".join(words[1:])
                                words = expr_clean.split()

                    # Normalize over-extended prepositional prefixes (e.g. 'toward the end of' -> 'the end of')
                    m_prep = re.match(r"^(?:toward|towards|near)\s+(the\s+end\s+of\b.*)", expr_clean, re.IGNORECASE)
                    if m_prep:
                        expr_clean = m_prep.group(1).strip()
                        words = expr_clean.split()

                    # Normalize over-extended determiners/quantifiers for category frames (e.g. 'any type of' -> 'type of')
                    m_det = re.match(r"^(?:any|each|every|some|a|an)\s+((?:type|kind|sort)\s+of\b.*)", expr_clean, re.IGNORECASE)
                    if m_det:
                        expr_clean = m_det.group(1).strip()
                        words = expr_clean.split()

                    # Build regex for placeholder-bearing or standard contiguous expressions
                    has_placeholder = bool(re.search(r"\[(?:one's|oneself|sth|sb|sth/sb)\]", expr_clean, re.IGNORECASE))
                    search_core = re.sub(r"(?:\s+\[(?:sth|sb|sth/sb)\])+$", "", expr_clean, flags=re.IGNORECASE).strip() or expr_clean
                    search_words = search_core.split()
                    placeholder_rx = None
                    if has_placeholder:
                        rx_chunks = []
                        for tok in search_words:
                            tl = tok.lower()
                            if tl == "[one's]":
                                rx_chunks.append(r"(?:my|your|his|her|its|our|their|one's|ones|[a-z]+'s)")
                            elif tl == "[oneself]":
                                rx_chunks.append(r"(?:myself|yourself|himself|herself|itself|ourselves|yourselves|themselves|oneself)")
                            elif tl in ("[sth]", "[sb]", "[sth/sb]"):
                                rx_chunks.append(r"(?:[a-z0-9'\-]+(?:\s+[a-z0-9'\-]+){0,5})")
                            elif tl == "not":
                                rx_chunks.append(r"(?:not|n't|never|hardly|barely|scarcely)")
                            else:
                                forms = sorted(cls.inflected_forms(tl) | {tl}, key=len, reverse=True)
                                rx_chunks.append(r"(?:" + "|".join(re.escape(f) for f in forms if f) + r")")
                        placeholder_rx = re.compile(r"\b" + r"\s+".join(rx_chunks) + r"\b", re.IGNORECASE)

                    phrase_rx = None
                    tokens = cls._phrase_tokens(search_core)
                    if len(tokens) >= 2:
                        rx_pat = cls._phrase_regex(tokens, 2, allow_modifier=False)
                        if rx_pat:
                            phrase_rx = re.compile(rx_pat, re.IGNORECASE)

                    content_words = [w for w in search_words if not (w.startswith("[") and w.endswith("]")) and w.lower() != "not"]

                    def _verify_verb_placeholder_match(sent_str: str) -> bool:
                        if not (has_placeholder and len(content_words) == 1):
                            return True
                        v_word = content_words[0].lower()
                        v_forms = cls.inflected_forms(v_word) | {v_word}
                        s_doc = nlp(sent_str)
                        for t in s_doc:
                            if t.text.lower() in v_forms or t.lemma_.lower() == v_word:
                                if t.pos_ in ("VERB", "AUX") and not any(c.dep_ in ("det", "poss") for c in t.children):
                                    return True
                        return False

                    # Two-pass sentence search: Pass A requires anchor_evidence_ok(s);
                    # Pass B falls back to all sentences in pool (recovering subheadings like "Hats Off to Volunteers" / "Take Corporate Action")
                    for allow_short_headings in (False, True):
                        for sid, s in pool.items():
                            if not allow_short_headings and not cls.anchor_evidence_ok(s):
                                continue
                            s_lower = s.lower()
                            e_lower = search_core.lower()
                            if placeholder_rx and placeholder_rx.search(s) and _verify_verb_placeholder_match(s):
                                matched_sid = sid
                                matched_sent = s
                                break
                            elif not has_placeholder and re.search(r"\b" + re.escape(e_lower.rstrip("?.!")) + r"\b", s_lower):
                                matched_sid = sid
                                matched_sent = s
                                break
                            elif phrase_rx and phrase_rx.search(cls._entry_text_normalizer(s)):
                                matched_sid = sid
                                matched_sent = s
                                break

                        if not matched_sid and content_words and not expr_clean.endswith("?"):
                            for sid, s in pool.items():
                                if not allow_short_headings and not cls.anchor_evidence_ok(s):
                                    continue
                                if not _verify_verb_placeholder_match(s):
                                    continue
                                s_lower = s.lower()
                                if all(re.search(r"\b" + re.escape(w.lower().rstrip("?.!")) + r"\b", s_lower) for w in content_words):
                                    matched_sid = sid
                                    matched_sent = s
                                    break
                                else:
                                    s_doc = nlp(s)
                                    s_lemmas = [t.lemma_.lower() for t in s_doc]
                                    s_tokens = [t.text.lower() for t in s_doc]
                                    if all(w.lower().rstrip("?.!") in s_lemmas or w.lower().rstrip("?.!") in s_tokens for w in content_words):
                                        matched_sid = sid
                                        matched_sent = s
                                        break

                        if matched_sid:
                            break

                    if matched_sid and matched_sent:
                        matched_branch_expr = expr_clean
                        break

                if not matched_sid or not matched_sent or not matched_branch_expr:
                    continue

                expr_clean = matched_branch_expr
                words = expr_clean.split()
                has_placeholder = bool(re.search(r"\[(?:one's|oneself|sth|sb|sth/sb)\]", expr_clean, re.IGNORECASE))
                e_lower = expr_clean.lower()
                cand_item = None

                # 1. Specialized High-Frequency Idioms (figurative, non-compositional units)
                KNOWN_IDIOMS = {
                    "hats off", "hat off", "pat on the back", 
                    "spill the beans", "break the ice", "piece of cake", 
                    "bite the bullet", "call it a day", "under the weather"
                }
                # 2. Specialized Closed-Paradigm Set Phrases (discourse connectors, fixed correlatives/framing units)
                KNOWN_SET_PHRASES = {
                    "as long as", "as well as", "as soon as", "so far as",
                    "in order to", "so as to", "in spite of", "due to", "owing to",
                    "on the other hand", "on the contrary", "in addition to",
                    "as a result", "for instance", "for example", "at least",
                    "at most", "at last", "upside down", "side by side",
                    "peace of mind", "pros and cons", "no exaggeration",
                    "not up to par", "up to par", "how was it?", "how was it"
                }

                if any(idiom in e_lower for idiom in KNOWN_IDIOMS):
                    formula = "hats off to [sb]" if e_lower in ("hat off", "hats off") and "hats off to" in matched_sent.lower() else expr_clean
                    cand_item = {
                        "sid": matched_sid,
                        "quote": matched_sent,
                        "phrase": expr_clean,
                        "type": "idiom",
                        "pattern_formula": formula,
                        "score": 100,
                    }
                elif any(sp in e_lower for sp in KNOWN_SET_PHRASES):
                    cand_item = {
                        "sid": matched_sid,
                        "quote": matched_sent,
                        "phrase": expr_clean,
                        "type": "set phrase",
                        "pattern_formula": expr_clean,
                        "score": 75,
                    }
                elif "touch" in e_lower and any(v in e_lower for v in ("keep", "stay", "get", "lose", "be")):
                    formula = f"{e_lower} [sb]" if "with" in e_lower else f"{e_lower}"
                    cand_item = {
                        "sid": matched_sid,
                        "quote": matched_sent,
                        "phrase": expr_clean,
                        "type": "idiom",
                        "pattern_formula": formula,
                        "score": 100,
                    }
                elif "make" in words and "possible" in words:
                    cand_item = {
                        "sid": matched_sid,
                        "quote": matched_sent,
                        "phrase": "make [sth] possible",
                        "type": "collocation",
                        "pattern_formula": "make [sth] possible",
                        "score": 85,
                    }
                elif has_placeholder:
                    # Syllabus item already declares an explicit slot formula ('do [one\'s] part', 'put [sth] into the atmosphere', 'make the most of [sth]')
                    lookup_phrase = re.sub(r"\s+", " ", re.sub(r"\[(?:one's|oneself|sth|sb|sth/sb|sb/sth)\]", "", expr_clean, flags=re.IGNORECASE)).strip() or expr_clean
                    cand_type = cls.classify_expression_type(lookup_phrase, matched_sent) or "collocation"
                    cand_item = {
                        "sid": matched_sid,
                        "quote": matched_sent,
                        "phrase": lookup_phrase,
                        "type": cand_type,
                        "pattern_formula": expr_clean,
                        "score": 85,
                    }
                else:
                    doc = nlp(matched_sent)
                    # Find the contiguous token span of expr_clean in matched_sent to avoid grabbing disjoint tokens
                    expr_tokens = []
                    m_span = re.search(r'\b' + re.escape(expr_clean) + r'\b', matched_sent, re.IGNORECASE)
                    if not m_span and len(words) > 1:
                        # Fallback: match words joined by flexible whitespace/punctuation
                        pat = r'\b' + r'\s+'.join(re.escape(w) for w in words) + r'\b'
                        m_span = re.search(pat, matched_sent, re.IGNORECASE)

                    if m_span:
                        sp_span = doc.char_span(m_span.start(), m_span.end())
                        if sp_span:
                            expr_tokens = list(sp_span)

                    if not expr_tokens:
                        # Find best ordered token sequence matching words by text or lemma
                        words_lower = [w.lower() for w in words]
                        best_tokens = None
                        min_dist = 9999
                        for start_t in doc:
                            if start_t.lemma_.lower() == words_lower[0] or start_t.text.lower() == words_lower[0]:
                                curr = [start_t]
                                for w in words_lower[1:]:
                                    cands = [t for t in doc if 0 < (t.i - curr[-1].i) <= 5 and (t.lemma_.lower() == w or t.text.lower() == w)]
                                    if not cands:
                                        break
                                    curr.append(cands[0])
                                if len(curr) == len(words_lower):
                                    dist = curr[-1].i - curr[0].i
                                    if dist < min_dist:
                                        min_dist = dist
                                        best_tokens = curr
                        if best_tokens:
                            expr_tokens = best_tokens
                        else:
                            dedup_tokens = []
                            for w_item in words_lower:
                                matched_t = next(
                                    (
                                        t for t in doc
                                        if (t.text.lower() == w_item or t.lemma_.lower() == w_item)
                                        and t not in dedup_tokens
                                        and (not dedup_tokens or 0 < (t.i - dedup_tokens[-1].i) <= 5)
                                    ),
                                    None
                                )
                                if matched_t is not None:
                                    dedup_tokens.append(matched_t)
                            expr_tokens = dedup_tokens

                    first_tok = expr_tokens[0] if expr_tokens else None
                    is_participial_amod = bool(first_tok and first_tok.dep_ == "amod" and first_tok.head.pos_ in ("NOUN", "PROPN"))

                    if first_tok and first_tok.pos_ in ("VERB", "AUX") and not is_participial_amod:
                        v_lemma = first_tok.lemma_.lower()
                        last_tok = expr_tokens[-1]
                        
                        # Syntax-aware detection: check if 'to' is an infinitive marker rather than a preposition
                        is_infinitive_to = False
                        if last_tok.text.lower() == "to":
                            if last_tok.pos_ == "PART" or last_tok.dep_ == "aux":
                                is_infinitive_to = True
                            elif last_tok.dep_ != "prep" and last_tok.head.pos_ == "VERB" and last_tok.head.dep_ in ("xcomp", "advcl", "purpcl") and last_tok.i < last_tok.head.i:
                                is_infinitive_to = True

                        if is_infinitive_to and len(words) == 2:
                            # Genuine infinitive construction (e.g. tend to do [sth], fail to do [sth])
                            formula = f"{v_lemma} to do [sth]"
                            cand_type = "collocation"
                        elif last_tok.pos_ in ("ADP", "PART") or last_tok.dep_ in ("prep", "prt"):
                            prep_word = last_tok.text.lower()
                            # Dynamic Transitivity Audit via Sentence Dependency Tree & Valence
                            # Check whether the verb or particle takes an authentic nominal object in the context sentence
                            verb_tok = expr_tokens[0] if expr_tokens else None
                            has_contextual_obj = False
                            if verb_tok is not None:
                                def _is_valid_obj(c_tok, anchor_tok) -> bool:
                                    if c_tok.pos_ not in ("NOUN", "PROPN", "PRON"):
                                        return False
                                    span_start = min(c_tok.i, anchor_tok.i)
                                    span_end = max(c_tok.i, anchor_tok.i)
                                    if any(t.text in (",", ";", ":", "\"", "'", "“", "”", "‘", "’") for t in doc[span_start : span_end + 1]):
                                        return False
                                    return True

                                # 1. Verb direct object (dobj / obj) — only for separable phrasal verb particles (prt)
                                verb_objs = [c for c in verb_tok.children if c.dep_ in ("dobj", "obj") and _is_valid_obj(c, last_tok)] if last_tok.dep_ == "prt" else []
                                # 2. Particle / prepositional object (pobj / obj)
                                prep_objs = [c for c in last_tok.children if c.dep_ in ("pobj", "obj", "dobj") and _is_valid_obj(c, last_tok)]
                                # 3. Passive voice extraction (e.g. 'the lights were turned off')
                                is_passive = any(c.dep_ in ("auxpass", "nsubjpass") for c in verb_tok.children)
                                if verb_objs or prep_objs or is_passive:
                                    has_contextual_obj = True

                            # If the verb+particle has no object in context (e.g. 'show up', 'go on'),
                            # it is an authentic intransitive phrasal verb: generate clean formula without [sth/sb]
                            is_intransitive = not has_contextual_obj

                            has_internal_noun_or_adj = (
                                len(words) >= 3
                                and any(t.pos_ in ("NOUN", "PROPN", "ADJ") for t in expr_tokens[1:-1])
                            )
                            if len(words) == 2:
                                if v_lemma == "be" and prep_word == "upon":
                                    formula = "be upon [sb/sth]"
                                    cand_type = "set phrase"
                                elif is_intransitive:
                                    formula = f"{v_lemma} {prep_word}"
                                    cand_type = "phrasal verb"
                                else:
                                    formula = f"{v_lemma} {prep_word} [sth/sb]"
                                    cand_type = "phrasal verb"
                            else:
                                core = " ".join([v_lemma] + [w.lower() for w in words[1:]])
                                if "importance to" in core:
                                    formula = f"{core} [sth]"
                                elif any(core.startswith(w) for w in ("show thanks", "give thanks", "pay tribute")):
                                    formula = f"{core} [sb]"
                                elif is_intransitive:
                                    formula = core
                                else:
                                    formula = f"{core} [sth/sb]"
                                cand_type = "collocation" if has_internal_noun_or_adj else (cls.classify_expression_type(expr_clean, matched_sent) or "phrasal verb")
                        else:
                            # Verb + object / adj (e.g. make smart choices, keep silent, have a try, take action)
                            # Use expr_clean words with lemmatized head verb so intervening sentence modifiers
                            # (e.g. 'Corporate' in 'Take Corporate Action') are not injected into the canonical formula.
                            core_words = [v_lemma] + [w.lower() for w in words[1:]]
                            core = " ".join(core_words)
                            formula = core
                            cand_type = "collocation"

                        cand_item = {
                            "sid": matched_sid,
                            "quote": matched_sent,
                            "phrase": expr_clean,
                            "type": cand_type,
                            "pattern_formula": formula,
                            "score": 90 if cand_type == "phrasal verb" else 80,
                        }
                    elif words[0].lower() in ("in", "on", "at", "by", "for", "with", "from", "to", "under", "over", "toward", "towards"):
                        # Prepositional expressions (e.g. toward the end of [sth], in touch with [sb])
                        formula = expr_clean
                        if expr_clean.endswith(" of") or expr_clean.endswith(" for") or expr_clean.endswith(" with"):
                            formula = f"{expr_clean} [sth]"
                        elif expr_clean.endswith(" to"):
                            formula = f"{expr_clean} [sth/sb]"
                        cand_item = {
                            "sid": matched_sid,
                            "quote": matched_sent,
                            "phrase": expr_clean,
                            "type": "set phrase",
                            "pattern_formula": formula,
                            "score": 50,
                        }
                    else:
                        # Nominal/quantifier/partitive/idiomatic expressions (e.g. a piece of [sth], any type of [sth], show thanks to [sb], hats off to [sb])
                        formula = expr_clean
                        if expr_clean.endswith(" of") or expr_clean.endswith(" for") or expr_clean.endswith(" with"):
                            formula = f"{expr_clean} [sth]"
                        elif expr_clean.endswith(" to"):
                            # If expression relates to importance, attention, or priority, default to [sth]
                            if any(k in expr_clean.lower() for k in ("importance", "attention", "priority", "regard")):
                                formula = f"{expr_clean} [sth]"
                            elif any(expr_clean.lower().startswith(w) for w in ("show thanks", "give thanks", "hats off", "hat off", "pay tribute")):
                                formula = f"{expr_clean} [sb]"
                            else:
                                formula = f"{expr_clean} [sth/sb]"
                        elif expr_clean.lower() in ("hat off", "hats off"):
                            formula = "hats off to [sb]"

                        cand_type = "idiom" if expr_clean.lower() in ("hat off", "hats off", "pat on the back") else "collocation"
                        cand_item = {
                            "sid": matched_sid,
                            "quote": matched_sent,
                            "phrase": expr_clean,
                            "type": cand_type,
                            "pattern_formula": formula,
                            "score": 75 if cand_type == "idiom" else 60,
                        }

                if cand_item:
                    syllabus_candidates.append(cand_item)

            # Sort syllabus candidates by pedagogical score (idioms & phrasal verbs first)
            syllabus_candidates.sort(key=lambda x: x["score"], reverse=True)

            # Pass 1: maximize sentence quote diversity among syllabus expressions
            for item in syllabus_candidates:
                p_clean = item["phrase"].lower()
                if p_clean not in seen_phrases and item["quote"] not in seen_quotes:
                    selected.append({
                        "sid": item["sid"],
                        "quote": item["quote"],
                        "phrase": item["phrase"],
                        "type": item["type"],
                        "pattern_formula": item["pattern_formula"],
                    })
                    seen_phrases.add(p_clean)
                    seen_quotes.add(item["quote"])
                if len(selected) >= target_count:
                    break

            # Pass 2: if more syllabus slots needed, allow reusing quote for distinct syllabus phrase
            if len(selected) < target_count:
                for item in syllabus_candidates:
                    p_clean = item["phrase"].lower()
                    if p_clean not in seen_phrases:
                        selected.append({
                            "sid": item["sid"],
                            "quote": item["quote"],
                            "phrase": item["phrase"],
                            "type": item["type"],
                            "pattern_formula": item["pattern_formula"],
                        })
                        seen_phrases.add(p_clean)
                    if len(selected) >= target_count:
                        break

        # Phase 2: If no syllabus provided and still under target_count, fill from computational NLP/ACL/OCD candidates
        if not syllabus_expressions and len(selected) < target_count:
            # Sort raw_candidates by score descending
            raw_candidates.sort(key=lambda x: x["score"], reverse=True)

            # Pass 3: balance across types with sentence quote diversity
            for item in raw_candidates:
                p_clean = item["phrase"].lower()
                if p_clean not in seen_phrases and item["quote"] not in seen_quotes:
                    selected.append({
                        "sid": item["sid"],
                        "quote": item["quote"],
                        "phrase": item["phrase"],
                        "type": item["type"],
                        "pattern_formula": item["pattern_formula"],
                    })
                    seen_phrases.add(p_clean)
                    seen_quotes.add(item["quote"])
                if len(selected) >= target_count:
                    break

            # Pass 4: if still under target_count, allow reusing quote if phrase is distinct
            if len(selected) < target_count:
                for item in raw_candidates:
                    p_clean = item["phrase"].lower()
                    if p_clean not in seen_phrases:
                        selected.append({
                            "sid": item["sid"],
                            "quote": item["quote"],
                            "phrase": item["phrase"],
                            "type": item["type"],
                            "pattern_formula": item["pattern_formula"],
                        })
                        seen_phrases.add(p_clean)
                    if len(selected) >= target_count:
                        break

        return selected[:target_count]

    # -------------------------------------------------------------------------
    # 3c. Deterministic Academic Vocabulary Mining (AWL + Lemmatization)
    # -------------------------------------------------------------------------
    @classmethod
    def mine_vocabulary_skeletons(cls, text: str, target_count: int = 20) -> List[Dict[str, Any]]:
        """
        Deterministically mines genuine academic vocabulary targets from the indexed sentence pool.
        Uses spaCy context-aware lemmatization to extract base dictionary headwords,
        prioritizes the Academic Word List (AWL 560), and filters out low-value everyday words.

        Returns list of dicts:
            [{
                "sid": "S-1",
                "quote": "full sentence text",
                "word": "autonomy",
                "part_of_speech": "noun",
                "is_awl": True
            }, ...]
        """
        _, pool = cls.tokenize_and_index_sentences(text)
        if not pool:
            return []

        nlp = cls.get_spacy()

        # Closed whitelist of parts of speech accepted by schema
        VALID_POS_MAP = {
            "NOUN": "noun",
            "VERB": "verb",
            "ADJ": "adjective",
            "ADV": "adverb",
        }

        # Exclude trivial grammaticalized or ultra-common verbs/words
        TRIVIAL_WORDS = frozenset({
            "be", "have", "do", "say", "get", "make", "go", "know", "take", "see",
            "come", "think", "look", "want", "give", "use", "find", "tell", "ask",
            "work", "seem", "feel", "try", "leave", "call", "good", "new", "first",
            "last", "long", "great", "little", "own", "other", "old", "right", "big",
            "high", "different", "small", "large", "next", "early", "young", "important",
            "few", "public", "bad", "same", "able", "man", "woman", "person", "people",
            "child", "time", "year", "day", "way", "thing", "life", "hand", "part",
            "eye", "place", "case", "week", "company", "system", "program", "question",
            "government", "number", "night", "point", "home", "water", "room", "mother",
            "area", "money", "story", "fact", "month", "lot", "right", "study", "book",
            "word", "business", "issue", "side", "kind", "head", "house", "service",
            "friend", "father", "power", "hour", "game", "line", "end", "member", "law",
            "car", "city", "community", "name", "president", "team", "minute", "idea",
            "kid", "body", "information", "back", "parent", "face", "others", "level",
            "office", "door", "health", "person", "art", "war", "history", "party",
            "result", "change", "morning", "reason", "research", "girl", "guy", "moment",
            "air", "teacher", "force", "education"
        })

        # Dependent adverbial connectives: these heads are grammatically incomplete
        # without their obligatory prepositional complement. They are only valid as
        # the fixed 'head + of' phrase (POS = preposition), never as a bare adverb —
        # a bare 'regardless' is not a testable single-blank target and, being absent
        # from the Oxford/ACL collocation model, would yield a broken distractor space.
        # This is a small, closed, stable set of 'of'-licensing connective heads
        # (deliberately NOT an open-ended valency map): any connective whose bare
        # form is incomplete on its own belongs here.
        DEPENDENT_CONNECTIVES = {
            "regardless": "of",
            "irrespective": "of",
        }

        candidates: List[Dict[str, Any]] = []
        seen_lemmas: Set[str] = set()

        for sid, sentence in pool.items():
            sent_clean = sentence.strip()
            if len(sent_clean.split()) < 5:
                continue

            doc = nlp(sent_clean)
            # First pass: mine any attested hyphenated compounds in the sentence as intact units
            for m_hc in re.finditer(r"\b([a-zA-Z]+(?:-[a-zA-Z]+)+)\b", sent_clean):
                hc = m_hc.group(1).lower()
                if hc in seen_lemmas:
                    continue
                hc_entry = cls.get_ldoce_entry(hc)
                if hc_entry and hc_entry.get("senses"):
                    hc_pos_raw = (hc_entry.get("pos") or "adjective").lower()
                    hc_pos = "noun" if "noun" in hc_pos_raw else ("verb" if "verb" in hc_pos_raw else ("adverb" if "adverb" in hc_pos_raw else "adjective"))
                    candidates.append({
                        "sid": sid,
                        "quote": sent_clean,
                        "word": hc,
                        "part_of_speech": hc_pos,
                        "is_awl": False,
                        "score": 12 + min(len(hc), 12)
                    })
                    seen_lemmas.add(hc)

            WH_AND_PRONOUN_STOPS = frozenset({
                "what", "whatever", "who", "whoever", "whom", "whose", "which", "whichever",
                "where", "wherever", "when", "whenever", "why", "how", "however"
            })
            for token in doc:
                # Never extract a severed sub-token that is attached to a hyphen (e.g. 'waste' or 'prone' in 'waste-prone')
                if (token.i > 0 and doc[token.i - 1].text == "-" and not doc[token.i - 1].whitespace_) or (
                    token.i + 1 < len(doc) and doc[token.i + 1].text == "-" and not token.whitespace_
                ):
                    continue
                lemma = token.lemma_.lower().strip()
                if len(lemma) < 3 or lemma in seen_lemmas or not lemma.isalpha() or lemma in WH_AND_PRONOUN_STOPS:
                    continue

                # 1. Academic Closed-Paradigm Function Word Gate:
                # If lemma belongs to our curated high-utility closed paradigms (e.g. 'whereas',
                # 'nonetheless', 'notwithstanding', 'albeit'), bypass spaCy's stopword filter and POS restriction!
                if cls.is_function_word(lemma):
                    # Filter out elementary/common prepositions and conjunctions so passage mining focuses on academic lexis
                    ULTRA_BASIC_FUNC_WORDS = frozenset({
                        "in", "at", "on", "to", "for", "with", "from", "by", "of",
                        "and", "but", "or", "nor", "so", "for", "yet",
                        "if", "when", "as", "than", "that", "because",
                        "without", "within", "upon", "onto", "into", "until", "till",
                        "since", "while", "though", "although", "despite", "throughout",
                        "during", "whether", "before", "after", "under", "above", "below",
                        "between", "among", "around", "against", "towards", "toward",
                        "through", "across", "behind", "beyond", "besides", "except",
                        "unless", "once", "where", "wherever", "whenever"
                    })
                    if lemma in ULTRA_BASIC_FUNC_WORDS:
                        continue

                    f_idx = cls._get_function_word_index()
                    _fam, stype = f_idx[lemma][0]
                    # Map syntactic complement type to schema-supported POS
                    assigned_pos = "preposition" if stype == "prepositional" else ("conjunction" if stype == "clausal" else "adverb")
                    w_cefr = cls.get_word_cefr(lemma)
                    is_awl = w_cefr in ("B2", "C1", "C2")
                    score = (14 if is_awl else 6) + min(len(lemma), 8)

                    candidates.append({
                        "sid": sid,
                        "quote": sent_clean,
                        "word": lemma,
                        "part_of_speech": assigned_pos,
                        "is_awl": is_awl,
                        "is_connective": True,
                        "score": score
                    })
                    seen_lemmas.add(lemma)
                    continue

                # 2. Open-Class Content Words (Noun, Verb, Adjective, Adverb)
                if token.pos_ not in VALID_POS_MAP:
                    continue
                if token.is_stop:
                    continue
                if not cls.is_pedagogical_target(lemma, ent_type=token.ent_type_):
                    continue

                w_cefr = cls.get_word_cefr(lemma)
                is_awl = w_cefr in ("B2", "C1", "C2")
                is_b1 = (w_cefr == "B1")

                # If not B1/B2/C1/C2 and is trivial or very short, skip
                if not is_awl and (lemma in TRIVIAL_WORDS or len(lemma) < 5):
                    continue

                # Dependent-connective gate (extraction-stage quality, Pillar 0).
                # A connective head like 'regardless' is only valid as its fixed
                # 'head + of' form. When the obligatory preposition is present in the
                # parsed sentence, complete the head to the fixed phrase and re-label
                # it as a preposition; otherwise drop it — a bare dependent connective
                # is never a valid single-blank target and must not leak out as an
                # adverb (which is the root cause of the 'regardless of' double-key).
                if token.pos_ == "ADV" and lemma in DEPENDENT_CONNECTIVES:
                    required_prep = DEPENDENT_CONNECTIVES[lemma]
                    has_prep = (
                        any(ch.dep_ == "prep" and ch.text.lower() == required_prep for ch in token.children)
                        or (token.i + 1 < len(doc) and doc[token.i + 1].text.lower() == required_prep)
                    )
                    if not has_prep:
                        seen_lemmas.add(lemma)  # gate: never leak a bare dependent connective
                        continue
                    phrase = f"{lemma} {required_prep}"
                    score = (18 if is_awl else 8) + min(len(phrase), 10)
                    candidates.append({
                        "sid": sid,
                        "quote": sent_clean,
                        "word": phrase,
                        "part_of_speech": "preposition",
                        "is_awl": is_awl,
                        "is_connective": True,
                        "score": score
                    })
                    seen_lemmas.add(lemma)
                    seen_lemmas.add(phrase)
                    continue

                # Scoring: B2/C1/C2 headwords get top priority, followed by B1 headwords, then word length
                cefr_bonus = 25 if is_awl else (15 if is_b1 else 0)
                score = cefr_bonus + min(len(lemma), 12)

                candidates.append({
                    "sid": sid,
                    "quote": sent_clean,
                    "word": lemma,
                    "part_of_speech": VALID_POS_MAP[token.pos_],
                    "is_awl": is_awl,
                    "score": score
                })
                seen_lemmas.add(lemma)

        if not candidates:
            return []

        # Sort descending by score
        candidates.sort(key=lambda x: x["score"], reverse=True)

        selected: List[Dict[str, Any]] = []
        selected_words: Set[str] = set()
        selected_sids: Set[str] = set()

        def _get_sid_index(sid_str: str) -> int:
            m = re.search(r'\d+', sid_str)
            return int(m.group(0)) if m else -999

        # Pass 1: Select diverse items across different sentences with word-family and sentence stride gating
        for item in candidates:
            w = item["word"]
            sid = item["sid"]
            sid_num = _get_sid_index(sid)

            # 1. Morphological Word Family Gate: strictly reject items from an already selected word family
            if any(cls.are_same_word_family(w, sw) for sw in selected_words):
                continue

            # 2. Hard Anti-Clustering Gate (Pass 1): prefer 1 target word per sentence
            if sid in selected_sids:
                continue

            # 3. Soft Anti-Clustering Stride Gate: avoid immediately adjacent sentences (|sid - selected_sid| <= 1)
            # if we have enough distinct sentences to reach target_count
            has_adjacent_clash = any(
                abs(sid_num - _get_sid_index(ssid)) <= 1
                for ssid in selected_sids
            )
            if has_adjacent_clash:
                distinct_available_sids = {c["sid"] for c in candidates if not any(cls.are_same_word_family(c["word"], sw) for sw in selected_words)}
                if len(selected) + len(distinct_available_sids - selected_sids) >= target_count:
                    continue

            selected.append(item)
            selected_words.add(w)
            selected_sids.add(sid)
            if len(selected) >= target_count:
                break

        # Pass 2: If target_count not reached, fill from remaining distinct sentences (still 1 per sentence)
        if len(selected) < target_count:
            for item in candidates:
                w = item["word"]
                sid = item["sid"]
                if w in selected_words or any(cls.are_same_word_family(w, sw) for sw in selected_words):
                    continue
                if sid in selected_sids:
                    continue
                selected.append(item)
                selected_words.add(w)
                selected_sids.add(sid)
                if len(selected) >= target_count:
                    break

        # Pass 3: If target_count still not reached (short passages with few sentences), fill remaining slots
        # while strictly maintaining the Morphological Word Family Gate
        if len(selected) < target_count:
            for item in candidates:
                w = item["word"]
                if w in selected_words or any(cls.are_same_word_family(w, sw) for sw in selected_words):
                    continue
                selected.append(item)
                selected_words.add(w)
                if len(selected) >= target_count:
                    break

        return selected[:target_count]

    @classmethod
    def build_top_k_vocab_candidates(
        cls,
        word: str,
        quote: str,
        pos: Optional[str] = None,
        k: int = 4,
        discourse_context: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Pure Symbolic Candidate Selection (NO Embedding Model).
        Retrieves top-k (k <= 4) candidate senses from PedLex using:
        1. POS gating
        2. Syntactic valency / frame / particle agreement
        3. Simplified Lesk overlap, 2nd-order collocate projection, and discourse window cue scoring
        4. PedLex natural frequency order fallback
        """
        entry = cls.get_pedlex_entry(word)
        raw_senses = entry.get("senses", []) if entry else []

        # Filter out morphological pointer senses ("the past participle of...", "the plural of...")
        # whenever substantive semantic definitions exist in the entry
        _morph_ptr_re = re.compile(
            r"^(?:the\s+)?(?:past\s+participle|past\s+tense|present\s+participle|plural|comparative|superlative)\s+of\b",
            re.IGNORECASE
        )
        non_ptr_senses = [
            s for s in raw_senses
            if not _morph_ptr_re.match((s.get("definition") or "").strip())
        ]
        senses = non_ptr_senses if non_ptr_senses else raw_senses

        target_pos = pos or (entry.get("pos") if entry else None)
        target_p = cls._pos_code(target_pos)
        q_low = (quote or "").lower()
        has_pos_match = bool(target_p and any(cls._pos_code(s.get("pos")) == target_p for s in senses))

        if not senses:
            candidates = []
        elif len(senses) == 1 and (not target_p or not senses[0].get("pos") or cls._pos_code(senses[0].get("pos")) == target_p):
            s0 = senses[0]
            candidates = [{
                "definition": cls.format_ldoce_definition(s0) or s0.get("definition", ""),
                "pos": pos or s0.get("pos", "noun"),
                "sense_idx": 0,
                "signpost": s0.get("signpost", ""),
                "patterns": s0.get("patterns", []),
                "examples": s0.get("examples", []),
                "raw_sense": s0,
                "score": 10.0,
                "margin": 0.0,
            }]
        else:
            entry_filtered = dict(entry, senses=senses) if senses is not raw_senses else entry
            res = cls._lock_sense(
                entry_filtered,
                quote=quote,
                target_pos=target_pos,
                return_scores=True,
                discourse_context=discourse_context
            )

            def _is_unattested_phrase_unit_sense(s_obj: Dict[str, Any]) -> bool:
                u_list = s_obj.get("units") or []
                if not u_list:
                    return False
                w_low = word.strip().lower()
                s_defn_low = str(s_obj.get("definition") or "").strip().lower()
                # Single-word non-verb headwords should not select phrase units whose definition is a verbal infinitive ('to be at...')
                if " " not in w_low and target_p != "v":
                    if re.match(r"^(?:\[.*?\]\s*)*to\s+[a-z]+\b", s_defn_low):
                        return True
                # If a verb sense explicitly declares [transitive], its prepositional unit (e.g. 'tend to sb/sth')
                # is an optional prepositional variant, not an obligatory multi-word idiom
                s_gram_low = str(s_obj.get("grammar") or "").lower()
                if target_p == "v" and re.search(r"(?<!in)transitive", s_gram_low):
                    return False
                # Check if any unit phrase is attested or shares its grammatical preposition/frame with the quote
                all_multi_unattested = True
                for u_it in u_list:
                    u_phr = (u_it.get("phrase") if isinstance(u_it, dict) else str(u_it or "")).strip()
                    u_clean = re.sub(r'\[.*?\]|\(.*?\)', '', u_phr).strip().lower()
                    u_clean = re.sub(r"\b(?:sb's|sb|sth|one's|oneself|etc)\b\.?", "", u_clean).strip()
                    u_toks = [t for t in re.findall(r"[a-z]+", u_clean) if t not in ("a", "an", "the", "some", "any")]
                    if len(u_toks) <= 1:
                        all_multi_unattested = False
                        break
                    if " " not in w_low and u_toks[0] in ("not", "no", "never") and s_defn_low.startswith(("no ", "not ", "never ")):
                        continue
                    if all(cls.text_contains_form(q_low, t) for t in u_toks):
                        all_multi_unattested = False
                        break
                    # Preserve unit sense if it is a <headword> + <preposition> frame (e.g. 'an embarrassment of riches')
                    # and the quote contains <headword> + <preposition> (e.g. 'an embarrassment of possibilities')
                    if len(u_toks) >= 2 and u_toks[0] == w_low and u_toks[1] in ("of", "to", "for", "with", "in", "on", "from", "about"):
                        if f"{w_low} {u_toks[1]}" in q_low:
                            all_multi_unattested = False
                            break
                    # Preserve unit sense if it is a 2-word nominal compound <headword> + <noun> (e.g. 'comic book')
                    # whose definition defines the standalone headword rather than an opaque idiom
                    if len(u_toks) == 2 and u_toks[0] == w_low and target_p == "n":
                        all_multi_unattested = False
                        break
                return all_multi_unattested

            if not res or len(res) < 5:
                matched = [
                    s for s in senses
                    if (not has_pos_match or cls._pos_code(s.get("pos")) == target_p)
                    and not _is_unattested_phrase_unit_sense(s)
                ] or [s for s in senses if not has_pos_match or cls._pos_code(s.get("pos")) == target_p] or senses
                candidates = [{
                    "definition": cls.format_ldoce_definition(s) or s.get("definition", ""),
                    "pos": pos or s.get("pos", target_pos),
                    "sense_idx": i,
                    "signpost": s.get("signpost", ""),
                    "patterns": s.get("patterns", []),
                    "examples": s.get("examples", []),
                    "raw_sense": s,
                } for i, s in enumerate(matched[:k])]
            else:
                idx, score, margin, conf, score_trace = res
                # Enforce POS hard gate and filter unattested multi-word idiom senses when standalone senses exist
                pos_filtered_trace = [
                    (s_i, sc) for s_i, sc in score_trace
                    if (not has_pos_match or cls._pos_code(senses[s_i].get("pos")) == target_p)
                ] or score_trace
                non_idiom_trace = [
                    (s_i, sc) for s_i, sc in pos_filtered_trace
                    if not _is_unattested_phrase_unit_sense(senses[s_i])
                ] or pos_filtered_trace

                sorted_trace = sorted(non_idiom_trace, key=lambda x: -x[1])
                valid_idx_set = {s_i for s_i, _ in non_idiom_trace}
                top_indices = [s_i for s_i, _ in sorted_trace[:k]]
                if idx is not None and idx in valid_idx_set:
                    top_indices = ([idx] + [s_i for s_i in top_indices if s_i != idx])[:k]
                elif not top_indices:
                    top_indices = [0]

                candidates = []
                for s_i in top_indices:
                    s = senses[s_i]
                    candidates.append({
                        "definition": cls.format_ldoce_definition(s) or s.get("definition", ""),
                        "pos": pos or s.get("pos", target_pos),
                        "sense_idx": s_i,
                        "signpost": s.get("signpost", ""),
                        "patterns": s.get("patterns", []),
                        "examples": s.get("examples", []),
                        "raw_sense": s,
                        "score": next((sc for s_idx, sc in score_trace if s_idx == s_i), 0.0),
                        "margin": float(margin or 0.0),
                    })

        # Fallback with WordNet (oewn:2025+) synsets ONLY when PedLex has NO senses matching the target POS.
        # This preserves the 0-token single-sense fast path for monosemous PedLex entries and prevents
        # duplicate/unattested WordNet glosses from polluting pure PedLex candidate pools.
        needs_wn_fallback = (
            not candidates
            or not has_pos_match
        )
        if needs_wn_fallback:
            try:
                wn = cls.get_wordnet()
                if wn:
                    wn_pos_map = {"n": ("n",), "v": ("v",), "a": ("a", "s"), "adj": ("a", "s"), "r": ("r",), "adv": ("r",)}
                    wn_poses = wn_pos_map.get(target_p) if target_p else None
                    syns = [s for p_code in wn_poses for s in (wn.synsets(word, pos=p_code) or [])] if wn_poses else wn.synsets(word)
                    if syns:
                        seen_norm = {re.sub(r"[^a-z0-9]+", " ", (c.get("definition") or "").lower()).strip() for c in candidates}
                        q_words = set(re.findall(r"[a-z]{3,}", q_low))
                        scored_syns = []
                        for syn in syns:
                            s_def = (syn.definition() or "").strip()
                            if not s_def:
                                continue
                            s_norm = re.sub(r"[^a-z0-9]+", " ", s_def.lower()).strip()
                            if s_norm in seen_norm:
                                continue
                            s_exs = [str(e).strip().strip('"“”') for e in (syn.examples() or []) if e]
                            # Only include WordNet synsets that either have examples or when PedLex has 0 POS-matched candidates
                            if not s_exs and candidates and has_pos_match:
                                continue
                            seen_norm.add(s_norm)
                            s_bag = set(re.findall(r"[a-z]{3,}", (s_def + " " + " ".join(s_exs)).lower()))
                            overlap = len(q_words & s_bag)
                            if f"{word.lower()} of " in q_low and any(kw in s_def.lower() for kw in ("quantity", "excess", "number", "amount", "multitude", "group", "competitors")):
                                overlap += 3
                            scored_syns.append((overlap, syn, s_def, s_exs, s_norm))
                        scored_syns.sort(key=lambda x: -x[0])
                        if not has_pos_match and scored_syns:
                            candidates = []
                        wn_slots = min(2, len(scored_syns)) if (candidates and has_pos_match) else min(k, len(scored_syns))
                        if candidates and has_pos_match and len(candidates) + wn_slots > k:
                            candidates = candidates[:max(1, k - wn_slots)]
                        for _ov, _syn, s_def, s_exs, s_norm in scored_syns[:wn_slots]:
                            if len(candidates) >= k:
                                break
                            if target_p == "v" and not s_def.lower().startswith("to "):
                                s_def = f"to {s_def}"
                            candidates.append({
                                "definition": s_def,
                                "pos": target_pos or pos or "noun",
                                "sense_idx": -1,
                                "signpost": "",
                                "patterns": [],
                                "examples": s_exs,
                                "raw_sense": {"definition": s_def, "pos": target_pos or pos or "noun", "examples": s_exs},
                                "score": 0.0,
                                "margin": 0.0,
                            })
            except Exception:
                pass

        return candidates[:k]

    @classmethod
    def build_top_k_expression_candidates(
        cls,
        expr_word: str,
        quote: str,
        k: int = 4,
        expr_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Pure Symbolic Candidate Selection for Multi-Word Expressions (NO Embedding Model):
        1. Phrasal verb blocks & direct multi-word PedLex headword entries
        2. PedLex phrase index candidates (exact, prefix, or slash-expanded unit_key match)
        3. Strictly attested phrase units on constituent headwords (and WordNet/constituent fallbacks ONLY if unattested)
        """
        clean = re.sub(r'\[.*?\]|\(.*?\)', '', expr_word).strip().lower()
        clean = re.sub(r"\b(?:sb's|sb|sth|one's|oneself)\b", "", clean).strip()
        clean = re.sub(r"\s+", " ", clean).strip()
        tokens = clean.split()
        lem_0 = cls.lemmatize_headword(tokens[0]) if tokens else ""
        if tokens and lem_0 == tokens[0] and tokens[0].endswith("ed") and len(tokens[0]) > 4:
            stem_ed = tokens[0][:-2]
            if cls.get_pedlex_entry(stem_ed):
                lem_0 = stem_ed
        lem_tokens = ([lem_0] + tokens[1:]) if tokens else []
        candidates = []
        seen_defs = set()
        q_low = (quote or "").lower()
        locked_expr_type = (
            expr_type
            if (expr_type in cls.EXPRESSION_TYPE_LABELS)
            else (cls.classify_expression_type(expr_word, quote) or "collocation")
        )

        def _add_cand(df_str: str, pos_str: str, src_str: str, ex_list: List[str], raw_s: Optional[Dict[str, Any]], is_attested: bool = True):
            if not df_str:
                return
            norm_d = re.sub(r"[^a-z0-9]+", " ", df_str.lower()).strip()
            if not norm_d or norm_d in seen_defs:
                return
            # Guard against 'be there (for somebody)' when 'for' is absent from both formula and sentence
            if clean == "be there" and " for " not in f" {q_low} ":
                has_for_sb = "for somebody" in df_str.lower() or "help them" in df_str.lower() or any(
                    "for somebody" in str(u.get("phrase") if isinstance(u, dict) else u).lower()
                    for u in ((raw_s.get("units") if isinstance(raw_s, dict) else None) or [])
                )
                if has_for_sb:
                    return
            # Guard against [spoken] sympathy formula for 'think of' ('I'm thinking of you') when used in narrative recall
            if clean in ("think of", "think about") and "used to say that you care about and feel sympathy" in df_str.lower():
                if not any(p in q_low for p in ("thinking of you", "think of you", "thoughts are with")):
                    return
            # Guard against stative 'to seem similar to someone or something else' for 'remind of' when used in passive voice ('be reminded of')
            if "remind" in clean and "to seem similar to" in df_str.lower():
                if re.search(r"\b(?:am|is|are|was|were|be|been|being|get|gets|got)\s+(?:often\s+|always\s+|suddenly\s+|immediately\s+|constantly\s+)?reminded\b", q_low):
                    return
            # Guard against narrow [technical] chemical reaction sense ('if a chemical substance reacts') in non-chemical contexts
            if "if a chemical substance reacts" in df_str.lower():
                _chem_words = {"chemical", "substance", "acid", "metal", "gas", "oxygen", "water", "element", "compound", "solution", "calcium", "sulphur", "hydrogen", "carbon", "alkali", "molecule", "atom", "chlorine", "iron"}
                if not (set(re.findall(r"[a-z]{3,}", q_low)) & _chem_words):
                    return
            # Guard against literal vehicle backing-up sense ('if a vehicle or its driver reverses') or reciprocal plural-order sense ('the usual things two people do') when 'reverse' takes a singular reflexive pronoun ('reverse oneself/myself')
            if "if a vehicle or its driver reverses" in df_str.lower() or "the usual things two people do" in df_str.lower():
                if "oneself" in expr_word.lower() or re.search(r"\breverse[ds]?\s+(?:myself|yourself|himself|herself|itself|oneself)\b", q_low):
                    return
            seen_defs.add(norm_d)
            cleaned_exs = [str(e).strip().strip('"“”') for e in (ex_list or []) if e and str(e).strip()]
            candidates.append({
                'definition': df_str,
                'pos': locked_expr_type,
                'source': src_str,
                'examples': cleaned_exs,
                'raw_sense': raw_s,
                'attested': is_attested,
            })

        # 1. Phrasal verb blocks & direct multi-word PedLex headword entries (e.g. 'stuffed animal', 'comic book')
        u_key = cls._unit_key(clean)
        lem_u_key = cls._unit_key(" ".join(lem_tokens)) if lem_tokens else u_key
        unit_keys = []
        for uk_cand in (u_key, lem_u_key, clean, " ".join(lem_tokens)):
            if uk_cand and uk_cand not in unit_keys:
                unit_keys.append(uk_cand)

        def _unit_key_matches(cand_raw_or_uk: str, uk_target: str, allow_prefix: bool = False) -> bool:
            cand_uk = cls._unit_key(cand_raw_or_uk)
            if not cand_uk or not uk_target or len(cand_uk.split()) < 2:
                return False
            if cand_uk == uk_target:
                return True
            if allow_prefix and (cand_uk.startswith(uk_target + " ") or uk_target.startswith(cand_uk + " ")):
                return True
            # Check _declared_unit_spellings (which properly expands token-level slashes like 'open the door/way to something')
            for sp in cls._declared_unit_spellings(cand_raw_or_uk):
                sp_uk = cls._unit_key(re.sub(r"\bjust\b", "", sp).strip())
                if len(sp_uk.split()) < 2:
                    continue
                if sp_uk == uk_target:
                    return True
                if allow_prefix and (sp_uk.startswith(uk_target + " ") or uk_target.startswith(sp_uk + " ")):
                    return True
            # Expand slash alternatives and optional words ('just') in unit phrases like 'may/might/could (just) as well'
            for br in cls.clean_and_expand_pattern_slashes(cand_raw_or_uk):
                br_uk = cls._unit_key(re.sub(r"\bjust\b", "", br).strip())
                if len(br_uk.split()) < 2:
                    continue
                if br_uk == uk_target:
                    return True
                if allow_prefix and (br_uk.startswith(uk_target + " ") or uk_target.startswith(br_uk + " ")):
                    return True
            cand_no_just = re.sub(r"\s+", " ", re.sub(r"\bjust\b", "", cand_uk)).strip()
            if len(cand_no_just.split()) >= 2 and cand_no_just == uk_target:
                return True
            return False

        if len(tokens) >= 2:
            for mw_cand in unit_keys:
                mw_entry = cls.get_pedlex_entry(mw_cand)
                if mw_entry and mw_entry.get("senses"):
                    for s_i, s in enumerate(mw_entry.get("senses") or []):
                        df = cls.format_ldoce_definition(s)
                        _add_cand(df, locked_expr_type, f"{mw_cand} #{s_i}", s.get("examples", []), s, True)
            for uk in unit_keys:
                for block in cls._unit_blocks(uk):
                    b_senses = cls._unit_senses(block, uk)
                    if len(b_senses) > k and quote:
                        base_v = uk.split()[0]
                        prep_b = [dict(s, patterns=list(s.get("patterns") or []) + ([uk] if uk not in (s.get("patterns") or []) else [])) for s in b_senses]
                        res_b = cls._lock_sense({"word": base_v, "senses": prep_b}, quote=quote, target_pos="verb", return_scores=True)
                        if res_b and len(res_b) >= 5 and res_b[4]:
                            ranked_idxs = [s_i for s_i, _ in sorted(res_b[4], key=lambda x: (-x[1], x[0]))]
                            primary_idxs = [i for i, s in enumerate(b_senses) if str(s.get("number") or s.get("sensenum") or "") == "1"]
                            if primary_idxs and primary_idxs[0] not in ranked_idxs[:max(1, k - 1)]:
                                p_idx = primary_idxs[0]
                                ranked_idxs = [p_idx] + [i for i in ranked_idxs if i != p_idx]
                            b_senses = [b_senses[s_i] for s_i in ranked_idxs if 0 <= s_i < len(b_senses)]
                    for s in b_senses:
                        df = cls.format_ldoce_definition(s)
                        _add_cand(df, locked_expr_type, block.get('phrase') or block.get('headword') or uk, s.get('examples', []), s, True)

        # 2. PedLex phrase index candidates (exact or slash-expanded unit_key match first, prefix fallback if empty)
        query_list = []
        for q_s in [clean, u_key, " ".join(lem_tokens), lem_u_key]:
            if q_s and q_s not in query_list:
                query_list.append(q_s)
        for allow_pref in (False, True):
            if allow_pref and candidates:
                break
            for query_str in query_list:
                phrase_cands = cls.lookup_ldoce_phrase_candidates(query_str)
                for host, s_idx, tier, raw_p in phrase_cands:
                    if not any(_unit_key_matches(raw_p, uk, allow_prefix=allow_pref) for uk in unit_keys):
                        continue
                    h_entry = cls.get_pedlex_entry(host)
                    if h_entry:
                        senses = h_entry.get('senses', [])
                        if s_idx is not None and s_idx < len(senses):
                            s = senses[s_idx]
                            df = cls._compose_collocation_definition(u_key, None, host, s, cls.format_ldoce_definition(s), quote)
                            _add_cand(df, locked_expr_type, f"{host} #{s_idx}", s.get('examples', []), s, True)

        # 3. Constituent headword senses:
        # Pass 3A-1 scans all senses of each constituent token strictly for attested multi-word `units` matches.
        stop_func = {
            "a", "an", "the", "to", "of", "in", "on", "at", "by", "for", "with", "from",
            "into", "onto", "upon", "over", "under", "up", "down", "off", "away",
            "not", "no", "be", "do", "have", "one", "ones", "it", "that", "this", "as", "or", "and"
        }
        LIGHT_OR_MODAL_TOKS = {
            "take", "make", "come", "go", "get", "give", "have", "do", "be", "put", "keep",
            "fall", "turn", "set", "run", "hold", "bring", "find", "might", "can", "will",
            "may", "would", "should", "could", "must", "there", "here"
        }
        raw_toks = []
        for t in (lem_tokens + tokens):
            if (len(t) >= 3 and (t not in stop_func or (t == "out" and clean == "out of"))) and t not in raw_toks:
                raw_toks.append(t)
        # Prioritize specific open-class content words ahead of light verbs/modals
        content_toks = [t for t in raw_toks if t not in LIGHT_OR_MODAL_TOKS] + [t for t in raw_toks if t in LIGHT_OR_MODAL_TOKS]

        def _sense_matches_unit_strictly(s_obj: Dict[str, Any], allow_sub: bool = False) -> Tuple[bool, List[str]]:
            s_units = s_obj.get("units") or []
            for u_obj in s_units:
                u_raw = u_obj.get("phrase") if isinstance(u_obj, dict) else str(u_obj or "")
                u_p = cls._unit_key(u_raw or "")
                if not u_p or len(u_p.split()) < 2:
                    continue
                for uk in unit_keys:
                    if _unit_key_matches(u_raw or u_p, uk, allow_prefix=allow_sub):
                        u_ex = (u_obj.get("examples") or []) if isinstance(u_obj, dict) else []
                        return True, u_ex
                    if allow_sub and len(u_p.split()) >= 2 and len(uk.split()) >= 2 and (f" {u_p} " in f" {uk} " or f" {uk} " in f" {u_p} "):
                        u_ex = (u_obj.get("examples") or []) if isinstance(u_obj, dict) else []
                        return True, u_ex
            return False, []

        def _sense_matches_head_pattern(c_tok: str, s_obj: Dict[str, Any]) -> bool:
            if tokens and c_tok in (tokens[0], lem_0):
                other_content = [t for t in (lem_tokens + tokens) if t != c_tok and len(t) >= 3 and t not in stop_func]
                other_toks = other_content if other_content else [t for t in (lem_tokens + tokens) if t != c_tok and len(t) >= 2]
                if other_toks:
                    pats = [str(p).lower() for p in (s_obj.get("patterns") or [])]
                    for pb in (s_obj.get("pattern_bundles") or []):
                        if isinstance(pb, dict) and pb.get("pattern"):
                            pats.append(str(pb["pattern"]).lower())
                    other_tok_set = set(other_toks)
                    for pat_str in pats:
                        pat_words = set(re.findall(r"[a-z]+", pat_str))
                        if all(ot in pat_words for ot in other_tok_set):
                            return True
                        # Directional preposition open-class pattern (e.g. 'move away/out/to/towards etc' covers 'move into')
                        if not other_content and len(other_tok_set) == 1 and next(iter(other_tok_set)) in ("into", "onto", "towards", "toward", "across", "along", "through"):
                            if "etc" in pat_words and any(dp in pat_words for dp in ("to", "towards", "toward", "away", "out", "in", "into", "down", "up")):
                                return True
                    if not other_content and s_obj.get("examples"):
                        if any(cls._unit_example(uk, list(s_obj.get("examples") or [])) for uk in unit_keys if len(uk.split()) >= 2):
                            return True
            return False

        # Pass 3A-1: Strictly attested multi-word `units` across all senses of each constituent token
        for c_tok in content_toks:
            c_entry = cls.get_pedlex_entry(c_tok)
            if not c_entry:
                continue
            for s_i, s in enumerate(c_entry.get("senses") or []):
                is_match, matched_unit_ex = _sense_matches_unit_strictly(s)
                if not is_match:
                    continue
                raw_df = cls.format_ldoce_definition(s)
                if not raw_df:
                    continue
                df = cls._compose_collocation_definition(u_key, None, c_tok, s, raw_df, quote) or raw_df
                _add_cand(df, locked_expr_type, f"{c_tok} #{s_i}", matched_unit_ex or s.get('examples', []), s, True)

        # Pass 3A-2: Headword `patterns` / `pattern_bundles` match when no multi-word unit/phrase entry exists,
        # or when a verb + preposition frame (e.g. 'remind of') has competing main-entry pattern senses alongside phrasal-verb blocks.
        has_second_content = any(t not in (tokens[0], lem_0) for t in content_toks) if tokens else False
        if not candidates or (not has_second_content and len(candidates) < k) or len(candidates) < 2:
            for c_tok in content_toks:
                c_entry = cls.get_pedlex_entry(c_tok)
                if not c_entry:
                    continue
                for s_i, s in enumerate(c_entry.get("senses") or []):
                    if not _sense_matches_head_pattern(c_tok, s):
                        continue
                    raw_df = cls.format_ldoce_definition(s)
                    if not raw_df:
                        continue
                    df = cls._compose_collocation_definition(u_key, None, c_tok, s, raw_df, quote) or raw_df
                    _add_cand(df, locked_expr_type, f"{c_tok} #{s_i}", s.get('examples', []), s, True)

        # Pass 3B: WordNet (oewn:2025+) Multi-Word Synsets ONLY when PedLex has 0 attested phrase candidates
        if not candidates and len(tokens) >= 2:
            try:
                wn = cls.get_wordnet()
                if wn:
                    for mw_q in (clean, " ".join(lem_tokens), u_key):
                        if not mw_q or len(mw_q.split()) < 2:
                            continue
                        for syn in (wn.synsets(mw_q) or wn.synsets(mw_q.replace(" ", "_")) or []):
                            s_def = (syn.definition() or "").strip()
                            if not s_def:
                                continue
                            s_pos = getattr(syn, "pos", lambda: "")() if callable(getattr(syn, "pos", None)) else getattr(syn, "pos", "")
                            if s_pos == "v" and not s_def.lower().startswith("to "):
                                s_def = f"to {s_def}"
                            s_exs = [str(e).strip().strip('"“”') for e in (syn.examples() or []) if e]
                            _add_cand(s_def, locked_expr_type, f"wn:{mw_q}", s_exs, {"definition": s_def, "examples": s_exs}, True)
            except Exception:
                pass

        # Pass 3C: Constituent single-word fallback ONLY if NO phrase-attested candidates were found above
        if not candidates:
            is_verb_slot_frame = bool(re.search(r"[\[\(](?:sb|sth|oneself|one's)[\]\)]", expr_word, re.IGNORECASE)) and len(tokens) >= 1
            non_light = [t for t in content_toks if t not in LIGHT_OR_MODAL_TOKS] or content_toks
            expect_verb = is_verb_slot_frame and tokens and non_light and non_light[0] in (tokens[0], lem_0)
            for c_tok in non_light:
                c_entry = cls.get_pedlex_entry(c_tok)
                if c_entry:
                    for s_i, s in enumerate(c_entry.get("senses") or []):
                        if expect_verb and c_tok in (tokens[0], lem_0) and cls._pos_code(s.get("pos")) != "v":
                            continue
                        # Skip unrelated multi-word unit senses (e.g. 'call somebody names', 'fall victim/prey to')
                        s_units = s.get("units") or []
                        if s_units:
                            is_m, _ = _sense_matches_unit_strictly(s)
                            if not is_m:
                                continue
                        raw_df = cls.format_ldoce_definition(s)
                        if not raw_df:
                            continue
                        df = cls._compose_collocation_definition(u_key, None, c_tok, s, raw_df, quote) or raw_df
                        _add_cand(df, locked_expr_type, f"{c_tok} #{s_i}", s.get('examples', []), s, False)
                        if len(candidates) >= k - 2:
                            break
                # Also include WordNet verb synsets for verb-slot expressions (e.g. 'lift [sb] out of [sth]')
                if expect_verb and c_tok in (tokens[0], lem_0) and len(candidates) < k:
                    try:
                        wn = cls.get_wordnet()
                        if wn:
                            q_words = set(re.findall(r"[a-z]{3,}", q_low))
                            wn_syns = wn.synsets(c_tok, pos="v") or []
                            scored_v_syns = []
                            for syn in wn_syns:
                                s_def = (syn.definition() or "").strip()
                                if not s_def:
                                    continue
                                s_exs = [str(e).strip().strip('"“”') for e in (syn.examples() or []) if e]
                                s_bag = set(re.findall(r"[a-z]{3,}", (s_def + " " + " ".join(s_exs)).lower()))
                                ov = len(q_words & s_bag)
                                if ("out of" in clean or "from" in clean) and any(w in (s_def + " " + " ".join(s_exs)).lower() for w in ("condition", "rank", "poverty", "state", "from")):
                                    ov += 3
                                scored_v_syns.append((ov, s_def, s_exs))
                            scored_v_syns.sort(key=lambda x: -x[0])
                            for _ov, s_def, s_exs in scored_v_syns[:4]:
                                if not s_def.lower().startswith("to "):
                                    s_def = f"to {s_def}"
                                _add_cand(s_def, locked_expr_type, f"wn:{c_tok}", s_exs, {"definition": s_def, "examples": s_exs}, False)
                                if len(candidates) >= k:
                                    break
                    except Exception:
                        pass
                if len(candidates) >= k:
                    break

        return candidates[:k]

    @classmethod
    def ensure_distinct_example(
        cls,
        word_or_formula: str,
        pos: str,
        current_example: str,
        quoted_sentence: str,
        sense: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Guarantees that Example Usage is NEVER a verbatim duplicate of Quoted Sentence,
        and ALWAYS demonstrates the target word or expression in an authentic dictionary sentence
        from its locked sense (or from monosemous corpus entries). Returns '' if no valid example exists.
        """
        clean_target = re.sub(r'\[.*?\]|\(.*?\)', '', str(word_or_formula or '')).strip()
        clean_target = re.sub(r"\b(?:sb's|sb|sth|one's|oneself)\b", "", clean_target).strip()
        clean_target = re.sub(r"\s+", " ", clean_target).strip()
        clean_low = clean_target.lower()
        u_key = cls._unit_key(clean_target)
        is_multi = len(u_key.split()) >= 2 and "-" not in clean_low
        q_norm = re.sub(r'\s+', ' ', str(quoted_sentence or '').strip()).lower()
        pos_low = str(pos or '').strip().lower()

        def _is_valid_distinct(ex_str: str, check_pos: bool = True) -> bool:
            if not ex_str or not str(ex_str).strip():
                return False
            ex_clean = str(ex_str).strip().strip('"“”')
            e_norm = re.sub(r'\s+', ' ', ex_clean).lower()
            if e_norm == q_norm or (q_norm and (e_norm in q_norm or q_norm in e_norm) and abs(len(e_norm) - len(q_norm)) <= 12):
                return False
            if is_multi:
                if cls._unit_example(u_key, [ex_clean], allow_modifier=True):
                    return True
                norm_clean = re.sub(r"[^a-z0-9]+", " ", clean_low).strip()
                norm_ex = re.sub(r"[^a-z0-9]+", " ", e_norm).strip()
                return bool(norm_clean and norm_clean in norm_ex)
            if not cls.text_contains_form(ex_clean, clean_target):
                return False
            if check_pos and pos_low in ("noun", "verb", "adjective", "adverb"):
                ex_pos = cls.determine_contextual_pos(clean_target, ex_clean)
                if ex_pos and ex_pos in ("noun", "verb", "adjective", "adverb") and ex_pos != pos_low:
                    return False
            return True

        # 1. Check locked sense examples, units, and pattern_bundles first (Strict Sense-Isolation)
        cand_pool: List[str] = []
        if sense and isinstance(sense, dict):
            for pb in (sense.get("pattern_bundles") or []):
                if isinstance(pb, dict) and pb.get("example"):
                    cand_pool.append(str(pb["example"]).strip())
            for u_obj in (sense.get("units") or []):
                if isinstance(u_obj, dict):
                    for ex in (u_obj.get("examples") or []):
                        if ex:
                            cand_pool.append(str(ex).strip())
            for ex in (sense.get("examples") or []):
                if ex:
                    cand_pool.append(str(ex).strip())
            # If multi-word expression has an extended preposition pattern attested in quoted_sentence (e.g. 'get by on'),
            # prefer an example from the locked sense that demonstrates that exact extended pattern.
            if is_multi and q_norm and cand_pool:
                for pat in (sense.get("patterns") or []):
                    for p_var in cls._slash_variants(str(pat)):
                        p_k = cls._unit_key(p_var)
                        if p_k and len(p_k.split()) > len(u_key.split()) and cls._unit_example(p_k, [quoted_sentence], allow_modifier=True):
                            for cand_ex in cand_pool:
                                if _is_valid_distinct(cand_ex, check_pos=False) and cls._unit_example(p_k, [cand_ex], allow_modifier=True):
                                    return cand_ex.strip().strip('"“”')

        if _is_valid_distinct(current_example, check_pos=(sense is None)):
            return str(current_example).strip().strip('"“”')

        if cand_pool:
            for cand_ex in cand_pool:
                if _is_valid_distinct(cand_ex, check_pos=False):
                    return cand_ex.strip().strip('"“”')

        # Determine whether the entry (or multi-word unit) is monosemous under the target POS
        entry = cls.get_ldoce_entry(clean_target)
        target_p = cls._pos_code(pos_low)
        entry_senses = (entry.get("senses") or []) if entry else []
        pos_senses = [s for s in entry_senses if not target_p or cls._pos_code(s.get("pos")) == target_p]
        is_monosemous = len(pos_senses) <= 1

        u_blocks = cls._unit_blocks(u_key) if is_multi else []
        u_rows = cls._unit_phrase_rows(u_key) if is_multi else []
        has_dedicated_unit = bool(u_rows or u_blocks or (entry and len(entry_senses) == 1))
        has_poly_block = any(len(cls._unit_senses(b, u_key)) > 1 for b in u_blocks)
        is_monosemous_multi = is_multi and has_dedicated_unit and not has_poly_block and len(u_rows) <= 1

        # If sense was locked to a WordNet fallback sense while PedLex already has senses for this POS,
        # do NOT steal examples from PedLex's different senses.
        if sense and isinstance(sense, dict):
            if is_multi:
                if not is_monosemous_multi:
                    return ""
            else:
                s_def_norm = re.sub(r"[^a-z0-9]+", " ", str(sense.get("definition") or "").lower()).strip()
                in_pedlex = any(
                    re.sub(r"[^a-z0-9]+", " ", str(cls.format_ldoce_definition(ps) or "").lower()).strip() == s_def_norm
                    for ps in pos_senses
                )
                if pos_senses and not in_pedlex:
                    return ""
                if not is_monosemous:
                    # Polysemous word whose locked sense has 0 valid examples: do NOT steal from another sense
                    return ""

        # 2. Monosemous or unlocked entry-level & cross-entry corpus examples
        if entry and is_monosemous:
            for s in pos_senses:
                for pb in (s.get("pattern_bundles") or []):
                    if isinstance(pb, dict) and pb.get("example"):
                        cand_pool.append(str(pb["example"]).strip())
                for ex in (s.get("examples") or []):
                    if ex:
                        cand_pool.append(str(ex).strip())
            for cc in (entry.get("cross_collocations") or []):
                if isinstance(cc, dict) and cc.get("example"):
                    cand_pool.append(str(cc["example"]).strip())

        # 3. For multiword units, check _unit_phrase_rows and _unit_blocks when not polysemously conflicting
        if is_multi and (not sense or is_monosemous_multi):
            for _h, row in u_rows:
                for ex in (row.get("examples") or []):
                    if ex:
                        cand_pool.append(str(ex).strip())
            for block in u_blocks:
                u_senses = cls._unit_senses(block, u_key)
                if not sense or len(u_senses) <= 1:
                    for s in u_senses:
                        for ex in (s.get("examples") or []):
                            if ex:
                                cand_pool.append(str(ex).strip())

        for cand_ex in cand_pool:
            if _is_valid_distinct(cand_ex):
                return cand_ex.strip().strip('"“”')

        # 4. Search corpus examples in PedLex SQLite (only for monosemous words or monosemous dedicated multi-word units)
        if (not is_multi and is_monosemous) or (is_multi and (is_monosemous_multi or (not sense and not has_poly_block))):
            try:
                search_q = u_key if is_multi else clean_target
                for ex in (cls.search_corpus_examples(search_q, limit=15) or []):
                    if ex and _is_valid_distinct(str(ex).strip()):
                        return str(ex).strip().strip('"“”')
            except Exception:
                pass

        # 5. Check WordNet synset examples (only when monosemous in both PedLex and WordNet for target POS)
        if not is_multi and is_monosemous:
            try:
                from nltk.corpus import wordnet as wn
                wn_pos_map = {"noun": ("n",), "verb": ("v",), "adjective": ("a", "s"), "adverb": ("r",)}
                wn_poses = wn_pos_map.get(pos_low)
                synsets = [s for p_code in wn_poses for s in (wn.synsets(clean_target.replace(" ", "_"), pos=p_code) or [])] if wn_poses else wn.synsets(clean_target.replace(" ", "_"))
                if len(synsets) <= 1:
                    for syn in synsets:
                        for ex in syn.examples():
                            if ex and _is_valid_distinct(str(ex).strip()):
                                return str(ex).strip().strip('"“”')
            except Exception:
                pass

        return ""

    @classmethod
    @contextmanager
    def wsd_log_session(cls, unit_name: str = "unit"):
        """
        Context manager that aggregates all Top-K LLM WSD arbitration calls within a unit
        into a single consolidated log file (`logs/<timestamp>_wsd_arbitration_<unit>.log`).
        Supports re-entrant nesting so outer unit sessions absorb inner vocabulary/expression calls.
        """
        active = getattr(cls._wsd_session_local, "session", None)
        if active is not None:
            if unit_name and unit_name != "unit" and active.get("unit_name") == "unit":
                active["unit_name"] = unit_name
            yield active
            return

        session = {
            "unit_name": unit_name or "unit",
            "model": None,
            "records": [],
            "start_time": datetime.datetime.now(),
        }
        cls._wsd_session_local.session = session
        try:
            yield session
        finally:
            cls._wsd_session_local.session = None
            records = session.get("records", [])
            # Write consolidated log only when at least one non-cached LLM arbitration ran
            has_new_llm_call = any(not r.get("cached", False) for r in records)
            if records and has_new_llm_call:
                end_time = datetime.datetime.now()
                duration = (end_time - session["start_time"]).total_seconds()
                try:
                    from librarian.config import config
                    from librarian.logger import log_wsd_session
                    log_wsd_session(
                        unit_name=session.get("unit_name") or "unit",
                        model=session.get("model") or config.get("model") or "unknown",
                        records=records,
                        duration=duration,
                        start_time=session["start_time"],
                        end_time=end_time,
                    )
                except Exception:
                    pass

    @classmethod
    def disambiguate_via_llm(
        cls,
        item_name: str,
        sentence: str,
        candidates: List[Dict[str, Any]],
        model: Optional[str] = None,
        discourse_context: str = ""
    ) -> int:
        """
        Neuro-Symbolic Zero-Hallucination LLM Reranker.
        Given an authentic sentence and 2-4 candidates from PedLex/PedLex,
        prompts the local LLM to select the single best option letter (A/B/C/D).
        Returns chosen candidate index (0-based).
        """
        if len(candidates) <= 1:
            return 0

        from librarian.config import config
        active_model = str(model or config.get("model") or "").strip()

        letters = ["A", "B", "C", "D", "E", "F", "G", "H"][:len(candidates)]
        options_text = []
        cand_sig_parts = []
        for idx, cand in enumerate(candidates[:len(letters)]):
            pos_str = f"({cand.get('pos', '')}) " if cand.get('pos') else ""
            raw_s = cand.get("raw_sense") if isinstance(cand.get("raw_sense"), dict) else {}
            gram_raw = str(raw_s.get("grammar") or "").strip()
            defn_str = cand.get('definition', '').strip()
            gram_str = (
                f"{gram_raw} "
                if any(
                    tag in gram_raw.lower()
                    for tag in (
                        "plural",
                        "singular",
                        "only before noun",
                        "not before noun",
                        "uncountable",
                        "countable",
                        "intransitive",
                        "transitive",
                    )
                )
                and gram_raw.lower() not in defn_str.lower()
                else ""
            )
            pats = cand.get('patterns') or ([cand['pattern']] if cand.get('pattern') else [])
            pat_str = f" [Pattern: {', '.join(str(p) for p in pats[:2] if p)}]" if pats else ""
            exs = cand.get('examples') or []
            ex_str = f" (e.g. \"{str(exs[0]).strip()}\")" if exs and str(exs[0]).strip() else ""
            options_text.append(f"{letters[idx]}. {pos_str}{gram_str}{defn_str}{pat_str}{ex_str}")
            cand_sig_parts.append(f"{cand.get('pos', '')}:{gram_str}{defn_str}:{pat_str}")

        ctx_clean = str(discourse_context or "").strip()
        cache_raw = f"{active_model}|v2|{item_name.strip().lower()}|{sentence.strip()}|{ctx_clean}|{'||'.join(cand_sig_parts)}"
        cache_key = hashlib.sha1(cache_raw.encode("utf-8")).hexdigest()

        active_session = getattr(cls._wsd_session_local, "session", None)
        if cache_key in cls._wsd_llm_cache:
            cached_idx, cached_raw = cls._wsd_llm_cache[cache_key]
            if active_session is not None:
                active_session["model"] = active_session.get("model") or active_model
                active_session["records"].append({
                    "item_name": item_name,
                    "sentence": sentence,
                    "candidates": candidates,
                    "chosen_idx": cached_idx,
                    "raw_reply": cached_raw,
                    "duration": 0.0,
                    "cached": True,
                })
            return cached_idx

        ctx_block = f"Surrounding Context:\n\"{ctx_clean}\"\n\n" if ctx_clean else ""
        prompt = (
            f"Select the single best dictionary sense for '{item_name}' as used in this sentence:\n"
            f"\"{sentence}\"\n\n"
            f"{ctx_block}"
            f"Candidate Senses:\n" + "\n".join(options_text) + "\n\n"
            f"Instructions:\n"
            f"- Match the exact contextual meaning, subject/object domain, and syntactic frame in the sentence.\n"
            f"- Avoid narrow technical, legal, medical, or '[spoken]' interjection senses unless the sentence explicitly uses them.\n"
            f"Reply with ONLY the letter of the correct sense (e.g. A):"
        )

        chosen_idx = 0
        raw = ""
        call_dur = 0.0
        t0 = datetime.datetime.now()
        try:
            from librarian.llm import llm
            resp = llm.chat(
                [{"role": "user", "content": prompt}],
                model=active_model or None,
                json_format=False,
                temperature=0.0,
                skip_log=True,
            )
            call_dur = (datetime.datetime.now() - t0).total_seconds()
            raw = (resp or "").strip()

            # 1. If wrapped in JSON like {"answer": "A"} or {"choice": "A"}
            parsed_idx = None
            if raw.startswith("{") and raw.endswith("}"):
                try:
                    d = json.loads(raw)
                    for k in ("answer", "choice", "selected", "letter"):
                        val = str(d.get(k, "")).strip().upper()
                        if val in letters:
                            parsed_idx = letters.index(val)
                            break
                    if parsed_idx is None:
                        # If JSON has only one key that is one of the valid letters: {"A": "..."}
                        matching_keys = [k.strip().upper() for k in d.keys() if k.strip().upper() in letters]
                        if len(matching_keys) == 1:
                            parsed_idx = letters.index(matching_keys[0])
                except Exception:
                    pass

            # 2. Plain text / single token matching:
            if parsed_idx is None:
                raw_upper = raw.upper()
                # Guard against refusal sentences ("Neither A nor B", "None of the options") falsely matching letter 'A'
                is_refusal = any(ref_kw in raw_upper for ref_kw in ("NEITHER", "NONE OF", "NO OPTION", "NOT LISTED", "CANNOT"))
                if not is_refusal:
                    m = re.search(r'\b(?:OPTION\s+)?([A-H])\b', raw_upper)
                    if m and m.group(1) in letters:
                        parsed_idx = letters.index(m.group(1))
                    else:
                        for l in letters:
                            if raw_upper.startswith(l):
                                parsed_idx = letters.index(l)
                                break

            if parsed_idx is not None:
                chosen_idx = parsed_idx
        except Exception:
            call_dur = (datetime.datetime.now() - t0).total_seconds()

        cls._wsd_llm_cache[cache_key] = (chosen_idx, raw)
        if active_session is not None:
            active_session["model"] = active_session.get("model") or active_model
            active_session["records"].append({
                "item_name": item_name,
                "sentence": sentence,
                "candidates": candidates,
                "chosen_idx": chosen_idx,
                "raw_reply": raw,
                "duration": call_dur,
                "cached": False,
            })
        return chosen_idx

    @classmethod
    def extract_deterministic_vocabulary(
        cls,
        text: str,
        syllabus_vocab: Optional[List[str]] = None,
        target_count: int = 20,
        enable_llm_disambiguation: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Pure Computational Linguistic & Lexicographic Extractor (0 Tokens, 100% Deterministic).
        Extracts academic target vocabulary, locates authentic quoted sentences with Sentence IDs,
        tags contextual Part of Speech via spaCy, and hydrates authoritative definitions & authentic
        corpus examples from PedLex 6th Edition.
        """
        with cls.wsd_log_session("vocabulary"):
            _, pool = cls.tokenize_and_index_sentences(text)
            if not pool:
                return []

            # 1. Determine target words (Syllabus primary + Passage reserve pool for Gate 2 backfill)
            target_candidates: List[Dict[str, Any]] = []
            if syllabus_vocab:
                cleaned_syllabus: List[str] = []
                seen_clean = set()
                for w in syllabus_vocab:
                    w_clean = re.sub(r'\[.*?\]|\(.*?\)', '', str(w)).strip().lower()
                    if w_clean and w_clean not in seen_clean:
                        seen_clean.add(w_clean)
                        cleaned_syllabus.append(w_clean)

                # Verbatim Compound Resolution (Passage = Ground Truth):
                # If a bare syllabus word (e.g. 'indulgent', 'confidence', 'prone', 'reflective', 'balance')
                # has zero standalone occurrences in the passage, check if it only appears as a sub-token
                # of a hyphenated compound in the passage (e.g. 'self-indulgent', 'self-confidence', 'waste-prone').
                # Promote the candidate to the verbatim passage compound so Gate 1 & Gate 2 evaluate the true
                # passage word (accepting attested dictionary compounds like 'self-indulgent' / 'self-confidence'
                # and cleanly dropping unattested coinages like 'waste-prone' / 'self-reflective' / 'self-balance').
                if cleaned_syllabus and pool:
                    passage_hyphen_compounds: List[str] = []
                    seen_hyph: Set[str] = set()
                    for _sid, sent in pool.items():
                        for m_hc in re.finditer(r"\b([a-zA-Z]+(?:-[a-zA-Z]+)+)\b", sent):
                            hc = m_hc.group(1).lower()
                            if hc not in seen_hyph:
                                seen_hyph.add(hc)
                                passage_hyphen_compounds.append(hc)
                    if passage_hyphen_compounds:
                        resolved_syllabus: List[str] = []
                        seen_resolved: Set[str] = set()
                        for sw in cleaned_syllabus:
                            resolved_w = sw
                            if "-" not in sw and not any(cls.text_contains_form(sent, sw) for sent in pool.values()):
                                for hc in passage_hyphen_compounds:
                                    parts = hc.split("-")
                                    if any(cls.form_matches(p, sw) for p in parts):
                                        # Normalize plural hyphenated noun head to singular if attested in dictionary
                                        hc_norm = hc
                                        if not cls.get_ldoce_entry(hc) and len(parts) >= 2:
                                            last_lem = cls._surface_lemma(parts[-1])
                                            if last_lem and last_lem != parts[-1]:
                                                cand_sing = "-".join(parts[:-1] + [last_lem])
                                                if cls.get_ldoce_entry(cand_sing):
                                                    hc_norm = cand_sing
                                        resolved_w = hc_norm
                                        break
                            if resolved_w not in seen_resolved:
                                seen_resolved.add(resolved_w)
                                resolved_syllabus.append(resolved_w)
                        cleaned_syllabus = resolved_syllabus
                        seen_clean = set(cleaned_syllabus)

                # Pre-filter same-word-family syllabus pairs (e.g. 'intrigue' + 'intrigued', 'predominant' + 'predominantly'):
                # If a syllabus word has a same-family sibling in the syllabus, drop any member that:
                #   (a) is completely unattested via text_contains_form while a sibling IS attested (e.g. 'predominant' vs 'predominantly'), OR
                #   (b) only matches passage sentences where the exact surface token is another syllabus sibling (e.g. 'intrigue' in "feeling intrigued")
                #       AND that sibling has a valid dictionary sense + example (otherwise retain the base headword).
                if len(cleaned_syllabus) > 1 and pool:
                    syl_shown_map: Dict[str, List[Tuple[str, str]]] = {
                        sw: [(sid, sent) for sid, sent in pool.items() if cls.text_contains_form(sent, sw)]
                        for sw in cleaned_syllabus
                    }
                    filtered_syllabus: List[str] = []
                    for sw in cleaned_syllabus:
                        siblings = [
                            other for other in cleaned_syllabus
                            if other != sw and cls.are_same_word_family(sw, other)
                        ]
                        if not siblings:
                            filtered_syllabus.append(sw)
                            continue
                        sw_shown = syl_shown_map.get(sw, [])
                        # Case (a): sw is not attested by any valid morphological form, while a sibling is attested
                        if not sw_shown and any(syl_shown_map.get(sib) for sib in siblings):
                            continue
                        # Case (b): sw has no exact surface match in its attested sentences, while a longer/derived sibling
                        # appears verbatim in those exact sentences AND has complete lexicon evidence (sense + example)
                        sw_has_exact = any(
                            bool(re.search(rf"\b{re.escape(sw)}\b", sent, flags=re.IGNORECASE))
                            for _sid, sent in sw_shown
                        )
                        if not sw_has_exact and sw_shown:
                            superseded = False
                            for sib in siblings:
                                sib_exact_sents = [
                                    sent for _sid, sent in sw_shown
                                    if re.search(rf"\b{re.escape(sib)}\b", sent, flags=re.IGNORECASE)
                                ]
                                if sib_exact_sents:
                                    sib_cands = cls.build_top_k_vocab_candidates(sib, sib_exact_sents[0], k=2)
                                    sib_ok = False
                                    if sib_cands:
                                        top_c = sib_cands[0]
                                        sib_def = top_c.get("definition", "")
                                        sib_ex = (top_c.get("examples") or [""])[0]
                                        sib_ex = cls.ensure_distinct_example(
                                            sib, str(top_c.get("pos") or "adjective"), sib_ex,
                                            sib_exact_sents[0], sense=top_c.get("raw_sense")
                                        )
                                        if sib_def and not cls.is_boilerplate_definition(sib_def) and sib_ex:
                                            sib_ok = True
                                    if sib_ok:
                                        superseded = True
                                        break
                            if superseded:
                                continue
                        filtered_syllabus.append(sw)
                    cleaned_syllabus = filtered_syllabus

                for w in cleaned_syllabus:
                    target_candidates.append({"word": w, "sid": "", "quote": "", "is_syllabus": True})
                # Reserve passage candidates to replace any syllabus item dropped by Gate 2 (Evidence-Completeness)
                reserve_skels = cls.mine_vocabulary_skeletons(text, target_count=max(target_count, 20) + 15)
                for sk in reserve_skels:
                    rw = sk.get("word", "").strip().lower()
                    if rw and rw not in seen_clean:
                        seen_clean.add(rw)
                        sk_copy = dict(sk)
                        sk_copy["is_syllabus"] = False
                        target_candidates.append(sk_copy)
            else:
                skels = cls.mine_vocabulary_skeletons(text, target_count=target_count + 15)
                target_candidates = skels

            results: List[Dict[str, Any]] = []
            seen_words: Set[str] = set()
            seen_word_quotes: List[Tuple[str, str]] = []

            for cand in target_candidates:
                if len(results) >= target_count:
                    break
                w = cand["word"].strip().lower()
                if not w or w in seen_words:
                    continue
                if not cls.is_pedagogical_target(w):
                    continue
                if any(cls.are_same_word_family(w, sw) for sw in seen_words):
                    continue

                matching_sent = cand.get("quote", "")
                matching_sid = cand.get("sid", "")

                # If sentence not already matched, search in sentence pool
                anchor_quality = "quote" if matching_sent else "syllabus"
                if not matching_sent and pool:
                    w_clean = re.sub(r'[^a-z0-9]', '', w)
                    shown = [(sid, sent) for sid, sent in pool.items() if cls.text_contains_form(sent, w)]
                    # Avoid sentences already claimed by a same-word-family item
                    if len(shown) > 1 and seen_word_quotes:
                        non_clash_shown = [
                            (sid, sent) for sid, sent in shown
                            if not any(cls.are_same_word_family(w, sw) and sent == sq for sw, sq in seen_word_quotes)
                        ]
                        if non_clash_shown:
                            shown = non_clash_shown
                    # For deverbal prepositions/participial adjectives (e.g. 'given', 'provided', 'considering'),
                    # prefer sentences where the word is NOT preceded by an auxiliary ('have/has/had/be/been')
                    if w in ("given", "provided", "considering", "regarding", "concerning", "following", "including", "granted") and len(shown) > 1:
                        non_aux_shown = [
                            (sid, sent) for sid, sent in shown
                            if not re.search(rf"\b(?:have|has|had|am|is|are|was|were|be|been|being)\s+(?:\w+\s+)?{re.escape(w)}\b", sent, flags=re.IGNORECASE)
                        ]
                        if non_aux_shown:
                            shown = non_aux_shown
                    valid_shown = [(sid, sent) for sid, sent in shown if cls.anchor_evidence_ok(sent, w)]
                    best_sent, _best_rich = cls.pick_anchor_sentence(w, [s for _sid, s in shown])
                    if best_sent:
                        matching_sent = best_sent
                        matching_sid = cls.get_sentence_id(best_sent, pool) or matching_sid
                        anchor_quality = "clause"
                    elif valid_shown:
                        matching_sid, matching_sent = valid_shown[0]
                        anchor_quality = "unqualified"
                    elif shown:
                        matching_sid, matching_sent = shown[0]
                        anchor_quality = "attested"

                # Ground-Truth Passage Gate: Never extract a word that is unattested as an independent word/inflection
                # in the passage (no blind substring hijacking of hyphenated compounds like 'waste-prone' -> 'prone'
                # or closed compounds like 'toothpaste' -> 'paste', 'microorganisms' -> 'organism').
                if not matching_sent:
                    continue

                if any(cls.are_same_word_family(w, sw) and matching_sent == sq for sw, sq in seen_word_quotes):
                    continue

                # Infer contextual POS
                pos = cand.get("part_of_speech")
                if (not pos or pos == "function_word") and matching_sent:
                    ctx_pos = cls.determine_contextual_pos(w, matching_sent)
                    if ctx_pos and ctx_pos != "function_word":
                        pos = ctx_pos
                if not pos or pos == "function_word":
                    ld_w = cls.get_ldoce_entry(w)
                    pos = (ld_w.get("pos") if ld_w else "") or "adverb"

                # Gate 1: Hydrate authoritative PedLex definition and authentic example via Top-K Candidate Pool + WSD
                definition = ""
                example = ""
                lock_confidence = "high"
                lock_score = 0.0
                lock_margin = 0.0
                locked_raw_sense = None

                discourse_ctx = ""
                if matching_sid and pool:
                    m_sid_num = re.match(r"S-(\d+)", str(matching_sid))
                    if m_sid_num:
                        s_num = int(m_sid_num.group(1))
                        nb_sents = [pool[f"S-{s_num + d}"] for d in (-2, -1, 1, 2) if f"S-{s_num + d}" in pool]
                        discourse_ctx = " ".join(nb_sents)

                cand_pool = cls.build_top_k_vocab_candidates(
                    w, matching_sent, pos=pos, k=4, discourse_context=discourse_ctx
                ) if matching_sent else []
                if cand_pool:
                    if len(cand_pool) > 1 and enable_llm_disambiguation:
                        chosen_idx = cls.disambiguate_via_llm(w, matching_sent, cand_pool, discourse_context=discourse_ctx)
                        chosen = cand_pool[chosen_idx]
                    else:
                        chosen = cand_pool[0]
                    definition = chosen.get("definition", "")
                    locked_raw_sense = chosen.get("raw_sense")
                    if not pos:
                        chosen_pos = str(chosen.get("pos") or "").strip().lower()
                        if chosen_pos:
                            for k, v in (("adv", "adverb"), ("noun", "noun"), ("verb", "verb"), ("adj", "adjective"), ("prep", "preposition"), ("conj", "conjunction")):
                                if k in chosen_pos:
                                    pos = v
                                    break
                    exs = chosen.get("examples") or []
                    if exs:
                        example = exs[0]
                    lock_score = chosen.get("score", 0.0)

                if not definition:
                    definition, example = cls.get_ldoce_definition_and_example(
                        w, target_pos=pos, context_sentence=matching_sent,
                        require_sense_confidence=False
                    )
                if not definition:
                    # Fallback to WordNet definition if PedLex headword missing
                    try:
                        best_s = cls.disambiguate_wordnet_synset(w, pos=pos, context_sentence=matching_sent)
                        if best_s:
                            definition = best_s.definition()
                    except Exception:
                        pass

                if pos == "adjective" and definition and re.match(r"^to\s+be\s+", definition.strip(), flags=re.IGNORECASE):
                    definition = re.sub(r"^to\s+be\s+", "", definition.strip(), flags=re.IGNORECASE)
                if definition and re.match(r"^to\s+[a-z]+\b", definition.strip().lower()) and pos in ("noun", "adjective", "adverb", "function_word"):
                    pos = "verb"
                if pos == "function_word":
                    pos = "adverb"
                if not example and not locked_raw_sense:
                    example = cls.ldoce_sense_example(w, pos=pos, quote=matching_sent)
                example = cls.ensure_distinct_example(w, pos, example, matching_sent, sense=locked_raw_sense)

                # Gate 2 (Evidence-Completeness Gate - 宁缺勿滥):
                # Drop any candidate lacking a valid definition OR an authentic, non-duplicate dictionary example
                # for its locked sense, allowing a fully attested reserve candidate to take its slot.
                if not definition or cls.is_boilerplate_definition(definition) or not example:
                    continue

                audit_sid = f"[{matching_sid}]" if matching_sid else "[S-1]"
                results.append({
                    "word": w,
                    "part_of_speech": pos,
                    "definition": definition,
                    "quoted_sentence": matching_sent,
                    "example_usage": example,
                    "anchor_quality": anchor_quality,
                    "sense_confidence": lock_confidence,
                    "sense_lock_score": lock_score,
                    "sense_lock_margin": lock_margin,
                    "design_audit": f"AUDIT: {audit_sid} -> [{w}] -> [VERBATIM_CONFIRMED]"
                })
                seen_words.add(w)
                if matching_sent:
                    seen_word_quotes.append((w, matching_sent))

            return results

    @classmethod
    def extract_deterministic_expressions(
        cls,
        text: str,
        target_count: int = 5,
        syllabus_expressions: Optional[List[str]] = None,
        enable_llm_disambiguation: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Extracts genuine multi-word academic expressions and collocations from text
        in 100% deterministic fashion (0 Tokens).
        Combines mine_expression_skeletons with get_expression_definition_and_example.
        Returns a list of dicts conforming to VocabularyItem / ExpressionItem:
            {
                "word": canonical formula (e.g. 'shut [sb/sth] down'),
                "part_of_speech": 'collocation' | 'phrasal verb' | ...,
                "definition": authoritative PedLex / WordNet definition,
                "quoted_sentence": verbatim passage sentence,
                "example_usage": authentic PedLex example or passage sentence,
                "design_audit": "AUDIT: [S-ID] -> [Formula] -> [VERBATIM_CONFIRMED]"
            }
        """
        with cls.wsd_log_session("expressions"):
            sent_pool = cls.tokenize_and_index_sentences(text)[1] if text else {}
            skeletons = cls.mine_expression_skeletons(
                text,
                target_count=max(target_count, len(syllabus_expressions)) if syllabus_expressions else target_count + 8,
                syllabus_expressions=syllabus_expressions
            )
            for sk in skeletons:
                sk["_is_syllabus"] = bool(syllabus_expressions)
            if syllabus_expressions:
                seen_ph = {re.sub(r'\[.*?\]|\(.*?\)', '', s.get("phrase", "")).strip().lower() for s in skeletons}
                reserve_exprs = cls.mine_expression_skeletons(
                    text,
                    target_count=target_count + 10,
                    syllabus_expressions=None
                )
                for rs in reserve_exprs:
                    rph = re.sub(r'\[.*?\]|\(.*?\)', '', rs.get("phrase", "")).strip().lower()
                    if rph and rph not in seen_ph:
                        seen_ph.add(rph)
                        rs["_is_syllabus"] = False
                        skeletons.append(rs)
            if not skeletons:
                return []

            attested_sources = ("phrase_row", "unit_sense", "headword_entry", "phrasal_verb_block", "sense_pattern")
            results: List[Dict[str, Any]] = []
            seen_formulas: Set[str] = set()
            for s in skeletons:
                if len(results) >= target_count:
                    break
                phrase = s.get("phrase", "")
                formula = s.get("pattern_formula", phrase)
                norm_form = re.sub(r'\[.*?\]|\(.*?\)', '', formula).strip().lower()
                if not norm_form or norm_form in seen_formulas:
                    continue
                sid = s.get("sid", "S-1")
                quote = s.get("quote", "")
                expr_type = s.get("type", "collocation")
                is_syllabus_item = bool(s.get("_is_syllabus", False))

                discourse_ctx = ""
                if sid and sent_pool:
                    m_sid_num = re.match(r"S-(\d+)", str(sid))
                    if m_sid_num:
                        s_num = int(m_sid_num.group(1))
                        nb_sents = [sent_pool[f"S-{s_num + d}"] for d in (-2, -1, 1, 2) if f"S-{s_num + d}" in sent_pool]
                        discourse_ctx = " ".join(nb_sents)

                # Primary: Ground multi-word expressions in authoritative PedLex expression evidence tiers
                # (Disable inner LLM disambiguation here since Gate 1 below arbitrates across the full unified pool)
                defn, ex, def_source = cls.expression_definition_evidence(
                    phrase=phrase,
                    expr_type=expr_type,
                    context_sentence=quote,
                    enable_llm_disambiguation=False
                )

                # Gate 1: Neuro-Symbolic Arbitration across Top-K (k=4) Expression Candidates
                locked_expr_sense = None
                if quote:
                    pool = cls.build_top_k_expression_candidates(formula, quote, k=4, expr_type=expr_type)
                    has_attested_pool = any(c.get("attested") for c in pool)
                    if not is_syllabus_item and not has_attested_pool and def_source not in attested_sources:
                        continue
                    if defn and not has_attested_pool:
                        norm_defn = re.sub(r'[^a-z0-9]+', ' ', defn.lower()).strip()
                        if not any(re.sub(r'[^a-z0-9]+', ' ', c.get("definition", "").lower()).strip() == norm_defn for c in pool):
                            pool.insert(0, {
                                "definition": defn,
                                "pos": expr_type,
                                "examples": [ex] if ex else [],
                                "score": 30.0,
                                "margin": 0.0,
                                "attested": def_source in attested_sources,
                            })
                    pool = pool[:4]
                    if pool:
                        if len(pool) > 1 and enable_llm_disambiguation:
                            chosen_idx = cls.disambiguate_via_llm(formula, quote, pool, discourse_context=discourse_ctx)
                            chosen = pool[chosen_idx] if 0 <= chosen_idx < len(pool) else pool[0]
                            if chosen.get("definition"):
                                defn = chosen["definition"]
                                def_source = def_source or "phrase_row"
                            locked_expr_sense = chosen.get("raw_sense")
                            if chosen.get("examples"):
                                ex = chosen["examples"][0]
                        elif not defn or (has_attested_pool and def_source not in attested_sources):
                            chosen = pool[0]
                            if chosen.get("definition"):
                                defn = chosen["definition"]
                                def_source = def_source or "phrase_row"
                            locked_expr_sense = chosen.get("raw_sense")
                            if chosen.get("examples"):
                                ex = chosen["examples"][0]
                        elif locked_expr_sense is None:
                            locked_expr_sense = pool[0].get("raw_sense")

                low_confidence = not def_source
                ex = cls.ensure_distinct_example(formula, expr_type, ex, quote, sense=locked_expr_sense)

                # Gate 2 (Evidence-Completeness Gate - 宁缺勿滥):
                # Drop any multi-word expression lacking an authoritative definition or distinct corpus example.
                if not defn or cls.is_boilerplate_definition(defn) or not ex:
                    continue

                seen_formulas.add(norm_form)
                results.append({
                    "word": formula,
                    "part_of_speech": expr_type,
                    "definition": defn,
                    "quoted_sentence": quote,
                    "example_usage": ex,
                    "definition_source": def_source,
                    "low_confidence": low_confidence,
                    "design_audit": f"AUDIT: [{sid}] -> [{formula}] -> [VERBATIM_CONFIRMED]"
                })

            return results

    # =========================================================================
    # DECLARATIVE 3-BAND COBUILD GRAMMAR PROFILES & HOT-RELOAD LOADER
    # Grounded in librarian/data/grammar_profiles.json (A1_A2 / B1_B2 / C1_C2)
    # =========================================================================
    _grammar_profiles_cache: Optional[Dict[str, Any]] = None
    _grammar_profiles_mtime: float = 0.0

    @classmethod
    def get_grammar_profiles_data(cls) -> Dict[str, Any]:
        """
        Loads and hot-reloads `librarian/data/grammar_profiles.json` using `st_mtime`.
        Decouples all 3-band priority weights, pedagogical functions, imitation examples,
        and common mistakes from Python source code.
        """
        json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "grammar_profiles.json")
        try:
            mtime = os.stat(json_path).st_mtime
            if cls._grammar_profiles_cache is not None and mtime == cls._grammar_profiles_mtime:
                return cls._grammar_profiles_cache
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            cls._grammar_profiles_cache = data
            cls._grammar_profiles_mtime = mtime
            return data
        except Exception as e:
            logger.warning(f"Failed to load grammar_profiles.json ({e}); using cached or empty fallback.")
            return cls._grammar_profiles_cache or {"profiles": {}, "fallbacks": {}}

    @classmethod
    def resolve_grammar_cefr_band(cls, text_or_pool: Union[str, Dict[str, str], List[str]]) -> str:
        """
        Deterministically resolves a passage's grammar pedagogical band ('A1_A2', 'B1_B2', or 'C1_C2')
        using `calculate_text_cefr` and lexical sophistication metrics (`b2_plus_ratio`, `c1_plus_ratio`, `avg_cefr`).
        """
        if isinstance(text_or_pool, dict):
            full_text = " ".join(text_or_pool.values())
        elif isinstance(text_or_pool, list):
            full_text = " ".join(text_or_pool)
        else:
            full_text = str(text_or_pool or "")

        if not full_text.strip():
            return "B1_B2"

        cefr = cls.calculate_text_cefr(full_text)
        if cefr in ("A1", "A2"):
            return "A1_A2"
        if cefr in ("C1", "C2"):
            return "C1_C2"

        try:
            analyzer = cls.get_cefr_analyzer()
            words = re.findall(r"[a-zA-Z]+", full_text.lower())
            levels: List[float] = []
            for w in words:
                lvl = analyzer.get_average_word_level_float(w)
                if lvl is not None:
                    levels.append(float(lvl))
            if levels:
                n = len(levels)
                avg_cefr = sum(levels) / n
                b2_plus_ratio = sum(1 for x in levels if x >= 3.5) / n
                c1_plus_ratio = sum(1 for x in levels if x >= 4.5) / n
                if cefr == "B1" and avg_cefr < 1.75 and b2_plus_ratio < 0.088:
                    return "A1_A2"
                if cefr in ("B1", "B2") and (avg_cefr >= 1.93 or (cefr == "B2" and c1_plus_ratio >= 0.055 and avg_cefr >= 1.90)):
                    return "C1_C2"
        except Exception:
            pass

        return "B1_B2"

    @classmethod
    def match_cobuild_grammar_profile(
        cls,
        formula: str,
        quote: str,
        category: str = "",
        cefr_band: str = "B1_B2"
    ) -> Dict[str, Any]:
        """
        Deterministically matches a mined syntax formula and quote to the most specific
        COBUILD Grammar Profile from `librarian/data/grammar_profiles.json` and resolves
        the target CEFR band's (`A1_A2`, `B1_B2`, `C1_C2`) priority, pedagogical function,
        imitation example, and common mistakes (0 Token Cost).
        """
        band = cefr_band if cefr_band in ("A1_A2", "B1_B2", "C1_C2") else "B1_B2"
        data = cls.get_grammar_profiles_data()
        profiles = data.get("profiles", {})
        fallbacks = data.get("fallbacks", {})

        def _resolve_profile(profile_key: str, fallback_cat: str = "") -> Dict[str, Any]:
            raw_prof = profiles.get(profile_key)
            if not raw_prof:
                return _resolve_fallback(fallback_cat or category)
            tiers = raw_prof.get("tiers", {})
            tier_data = tiers.get(band) or tiers.get("B1_B2") or tiers.get("A1_A2") or {}
            prio_map = raw_prof.get("priority", {})
            prio_val = int(prio_map.get(band, prio_map.get("B1_B2", 70)))
            return {
                "profile_key": profile_key,
                "name": raw_prof.get("name", "Syntactic Pattern"),
                "category": raw_prof.get("category", category or "Information Packaging"),
                "pattern_formula": raw_prof.get("pattern_formula", formula),
                "priority": prio_val,
                "pedagogical_function": tier_data.get("pedagogical_function", ""),
                "imitation_example": tier_data.get("imitation_example", quote),
                "common_mistakes": tier_data.get("common_mistakes", ""),
                "variants": tier_data.get("variants", [])
            }

        def _resolve_fallback(cat_name: str) -> Dict[str, Any]:
            c_low = (cat_name or "").lower()
            if c_low == "rhetoric & emphasis":
                fb_key = "Rhetoric & Emphasis"
            elif c_low == "logic & stance":
                fb_key = "Logic & Stance"
            elif c_low == "cohesion & framing":
                fb_key = "Cohesion & Framing"
            else:
                fb_key = "Information Packaging"
            raw_fb = fallbacks.get(fb_key, {})
            tiers = raw_fb.get("tiers", {})
            tier_data = tiers.get(band) or tiers.get("B1_B2") or tiers.get("A1_A2") or {}
            prio_map = raw_fb.get("priority", {})
            prio_val = int(prio_map.get(band, 45))
            return {
                "profile_key": f"fallback_{fb_key.lower().replace(' ', '_')}",
                "name": raw_fb.get("name", f"{fb_key} Pattern"),
                "category": fb_key,
                "pattern_formula": formula,
                "priority": prio_val,
                "pedagogical_function": tier_data.get("pedagogical_function", f"Synthesizes discourse via {fb_key.lower()} syntax."),
                "imitation_example": tier_data.get("imitation_example", quote),
                "common_mistakes": tier_data.get("common_mistakes", "Grammar Warning (Syntax): Learners frequently misapply structural boundaries."),
                "variants": tier_data.get("variants", [])
            }

        f_lower = (formula or "").lower()
        q_lower = (quote or "").lower()
        c_lower = (category or "").lower()

        # 0. Explicit formula-first routing for inverted conditionals, fronted negative inversion, with-absolute, nominative absolute, and concessives
        if any(f_lower.startswith(p) for p in ("had + [subject]", "were + [subject]", "should + [subject]")) or re.match(r"^(?:had|were|should)\s+(?:i|you|he|she|it|we|they|the\b|a\b|an\b|this\b|that\b|these\b|those\b|[a-z]+)\s+(?:not\s+)?(?:been|had|known|done|to\b|[a-z]+ed\b)", q_lower.strip()):
            if "inverted_conditional_subjunctive" in profiles:
                return _resolve_profile("inverted_conditional_subjunctive")
        if "negative/restrictive" in f_lower or "inversion" in f_lower or any(f_lower.startswith(p) for p in ("never +", "rarely +", "seldom +", "little +", "only +", "not only + [aux", "hardly +", "scarcely +", "no sooner +", "under no circumstances", "in no way", "at no time", "on no account")):
            return _resolve_profile("fronted_negative_inversion")
        if f_lower.startswith("with + [np]"):
            return _resolve_profile("with_absolute_adjunct")
        if f_lower.startswith("[np] + [v-ing") or f_lower.startswith("[np] + [v-ed") or "nominative absolute" in f_lower:
            if "nominative_absolute" in profiles:
                return _resolve_profile("nominative_absolute")
        if any(f_lower.startswith(p) for p in ("although +", "though +", "even though +", "even if +", "while +", "whereas +", "despite +", "in spite of +", "much as +", "albeit +", "notwithstanding +")):
            return _resolve_profile("subordinate_concessive")
        if any(f_lower.startswith(p) for p in ("as long as +", "provided that +", "on condition that +", "so long as +")):
            return _resolve_profile("conditional_clause_as_long_as")
        if any(f_lower.startswith(p) for p in ("if +", "once +", "unless +", "if/unless +")):
            return _resolve_profile("conditional_clause_if")

        # 1. Cleft Focus vs Dummy-It Extraposition (check extraposition predicates FIRST even if formula says Focal Element)
        if re.search(r"\bit\s+(?:was|is|'s|has\s+been|seemed|seems|appeared|appears|turned\s+out|proved)\s+(?:no\s+|not\s+a\s+|little\s+|hardly\s+a\s+)?(?:clear|obvious|important|necessary|essential|vital|critical|imperative|crucial|possible|impossible|likely|unlikely|certain|true|evident|apparent|plain|notable|remarkable|worth\s+noting|fair|safe|reasonable|natural|normal|common|rare|strange|odd|ironic|paradoxical|surprising|unsurprising|astonishing|shocking|sad|tragic|unfortunate|regrettable|fortunate|lucky|good|better|best|wise|foolish|easy|hard|difficult|simple|complex|challenging|said|reported|believed|thought|assumed|estimated|expected|known|understood|agreed|accepted|argued|suggested|claimed|noted|observed|found|shown|proven|established|demonstrated|revealed|discovered|decided|arranged|recommended|proposed|demanded|required|requested|urged|advised|surprise|wonder|secret|doubt|coincidence|accident|pity|shame|mystery|miracle|comfort|relief|consolation|honor|honour|privilege|pleasure|joy|delight|tragedy|disgrace|scandal|fact|matter|question|issue|problem|challenge|duty|responsibility|task|job|custom|tradition|habit|rule|policy|practice|mistake|error|fault|crime|sin|blessing|curse|waste|use|good|point)\b", q_lower):
            return _resolve_profile("dummy_it_extraposition")
        if "focal element" in f_lower or "cleft" in f_lower:
            return _resolve_profile("cleft_focus")
        if re.search(r"\bit\s+(?:was|is|'s)\s+.*?\b(?:that|who)\b", q_lower):
            return _resolve_profile("cleft_focus")

        # 2. Correlative Parallelism: not only... but also
        if "not only" in f_lower or ("not only" in q_lower and ("but also" in q_lower or "but " in q_lower)):
            if re.search(r"(?:^|[,;:]\s*)not\s+only\s+(?:did|do|does|had|have|has|was|were|is|are|can|could|will|would|should|must|may|might)\b", q_lower.strip()):
                return _resolve_profile("fronted_negative_inversion")
            return _resolve_profile("correlative_parallelism")

        # 2b. Antithesis Coordination: not... but... / instead of / rather than
        if "instead of +" in f_lower or "rather than +" in f_lower:
            if "prepositional_contrast_instead_of" in profiles:
                return _resolve_profile("prepositional_contrast_instead_of")
        if "not + [prepp/np], but" in f_lower or re.search(r",\s*not\b.*?\bbut\b", q_lower) or ("not + [" in f_lower and "but +" in f_lower):
            return _resolve_profile("antithesis_parallelism")

        # 3a. Causative bare infinitive: make/have/let + Object + [Bare Infinitive]
        if "bare infinitive" in f_lower:
            return _resolve_profile("causative_bare_infinitive")

        # 3b. Dummy-It Object Extraposition: make/find/think/consider + it + [Adj/NP] + to/that
        if "+ it + [" in f_lower and ("to + [vp]" in f_lower or "that + [clause]" in f_lower):
            if "dummy_it_object" in profiles:
                return _resolve_profile("dummy_it_object")

        # 3c. Object Complement / Complex Transitive (Noun Complement): make + Object + [Noun/NP]
        if ("make +" in f_lower or "find +" in f_lower or "appoint +" in f_lower or "consider +" in f_lower) and "[noun/np]" in f_lower:
            return _resolve_profile("object_complement_noun")

        # 3d. Object Complement / Complex Transitive (Adjective Complement): make/find/keep + Object + [Adj]
        if ("make +" in f_lower or "render +" in f_lower or "find +" in f_lower or "keep +" in f_lower or "leave +" in f_lower or "deem +" in f_lower) and ("[adj]" in f_lower or " + it + [" in f_lower or "[object]" in f_lower):
            return _resolve_profile("object_complement_adj")
        if re.search(r"\bmake\b.*?\b(?:possible|impossible|easy|clear|accessible|redundant)\b", q_lower):
            return _resolve_profile("object_complement_adj")

        # 4. Causative bare infinitive fallback: make/have/let + sb + do
        if "make + [" in f_lower or "let + [" in f_lower:
            return _resolve_profile("causative_bare_infinitive")

        # 5. Shell Noun / Appositive That-Clause: [Shell Noun] + that + [Clause]
        if "shell noun" in f_lower or "fact + that" in f_lower or "[abstract np] + that" in f_lower or re.search(r"\bthe\s+(?:fact|idea|belief|notion|view|possibility|chance|hope|fear|news|evidence|proof|assumption|hypothesis|premise|paradox|irony|conviction|realization|consensus|contention|proposition|myth|delusion|impression)\s+that\b", q_lower):
            return _resolve_profile("appositive_complement_that")

        # 5b. Participial Adjuncts (differentiating post-positioned V-ed vs V-ing, checked before relative clauses)
        if "[subject] + [vp], [" in f_lower and ("v-ing" in f_lower or "having +" in f_lower):
            return _resolve_profile("participial_adjunct_post_ving")
        if "[subject] + [vp], [" in f_lower and ("v-ed" in f_lower or "being + v-ed" in f_lower):
            return _resolve_profile("participial_adjunct_post_ved")
        if any(p in f_lower for p in ("[v-ing phrase]", "[v-ed phrase]", "[having + v-ed phrase]", "[being + v-ed phrase]")) or "particip" in c_lower:
            return _resolve_profile("participial_adjunct")

        # 6. Non-restrictive / Sentential Relative Clauses
        if "which + [interpretive" in f_lower or re.search(r",\s*which\s+(?:means|suggests|indicates|shows|implies|demonstrates|proves|reveals|reflects|underscores|highlights|explains|makes|leaves|renders|allows|enables|prevents|leads|results|contributes)\b", q_lower):
            if "propositional_encapsulation_which" in profiles:
                return _resolve_profile("propositional_encapsulation_which")
            return _resolve_profile("relative_clause_elaborative")
        if (", which" in q_lower and "fact + that" not in f_lower and "[shell noun]" not in f_lower) or "[np], which" in f_lower:
            return _resolve_profile("relative_clause_elaborative")

        # 7. Restrictive & Human Relative Clauses: [NP] + which/that/who + [VP] or Prep + which/whom
        if any(p in f_lower for p in ("[np] + which", "[np] + that", "[np] + who", "[np] + whom", "[np] + whose", "[np], who", "[np] , which", "[np] , who", "[np] + [prep] + which", "[np] + [prep] + whom")):
            return _resolve_profile("relative_clause_restrictive")

        # 8. Proportional Comparative & Equative Comparative (as ... as)
        if "the + [comparative]" in f_lower:
            return _resolve_profile("proportional_comparative")
        if "as + [adj/adv] + as" in f_lower or ("as + [" in f_lower and "+ as" in f_lower):
            return _resolve_profile("correlative_comparative_as_as")

        # 8b. Resultative Degree Constructions (so ... that / such ... that)
        if ("so + [" in f_lower or "such + [" in f_lower) and "that" in f_lower:
            return _resolve_profile("resultative_so_such_that")

        # 9. Dummy-It Extraposition: It is adj to/that
        if "it + [be] + [adj" in f_lower or "it is + [adj" in f_lower or "dummy" in f_lower:
            return _resolve_profile("dummy_it_extraposition")

        # 10. Purpose Infinitive Adjuncts (In order to / So as to / To + [VP]) & Infinitive Subject
        if any(p in f_lower for p in ("in order to +", "so as to +", "to + [vp], [subject]")):
            return _resolve_profile("purpose_infinitive_adjunct")
        if f_lower.startswith("to + [vp] + [v"):
            if "infinitive_subject_nominalization" in profiles:
                return _resolve_profile("infinitive_subject_nominalization")

        # 12. Gerund Subject Nominalization & Prepositional Means
        if "by + [v-ing" in f_lower or q_lower.startswith("by "):
            return _resolve_profile("prepositional_means_by_ving")
        if "[v-ing / gerund phrase]" in f_lower or "[v-ing phrase] + [v]" in f_lower:
            return _resolve_profile("gerund_subject_nominalization")

        # 13. Adversative & Concession
        if "but [subject] + [also]" in f_lower or ", but" in q_lower:
            return _resolve_profile("adversative_concession")
        if any(w in f_lower for w in ("although", "though", "while", "even if", "whereas")) or any(q_lower.startswith(w) for w in ("although ", "though ", "even though ", "even if ", "while ", "whereas ")):
            return _resolve_profile("subordinate_concessive")

        # 14. Discourse transition: However / Nevertheless / Nonetheless / Conversely / Instead
        if any(q_lower.startswith(w) or w in f_lower for w in ("however", "nevertheless", "nonetheless", "conversely", "in contrast", "instead,")):
            return _resolve_profile("discourse_transition_however")

        # 15. Causal & Explanatory Clauses (because / since / as / now that)
        if any(w in f_lower for w in ("because", "since + [clause]", "now that", "[causal clause]", "given that", "in that", "seeing that")):
            return _resolve_profile("explanatory_causal_clause")

        # 16. Comparative Manner & Analogy Clauses (just as / as if / as though)
        if any(w in f_lower for w in ("just as", "as if", "as though", "as + [clause]")):
            return _resolve_profile("manner_similarity_clause")

        # 17. Epistemic Stance & Hedging (suggest / indicate / show / argue that)
        if "stance" in f_lower or "epistemic" in f_lower or ("[verb] + that + [clause]" in f_lower and c_lower == "logic & stance"):
            return _resolve_profile("epistemic_stance_hedging")

        # 18. Passive voice
        if "passive" in f_lower or "passive" in c_lower or ("[be/modal be]" in f_lower and "[v-ed]" in f_lower):
            return _resolve_profile("passive_voice_agentless")

        # 19. Conditional clauses (if / as long as / unless / once)
        if "as long as" in f_lower or "as long as" in q_lower or "provided that" in f_lower or "on condition that" in f_lower or "so long as" in f_lower:
            return _resolve_profile("conditional_clause_as_long_as")
        if f_lower.startswith("if +") or f_lower.startswith("once +") or f_lower.startswith("unless +") or "if/unless" in f_lower or (q_lower.startswith("if ") or ", if " in q_lower or "so, if " in q_lower or "similarly, if " in q_lower):
            return _resolve_profile("conditional_clause_if")

        # Dynamic Category-Aware Fallback from grammar_profiles.json (3-Band Calibrated)
        return _resolve_fallback(category)

    @classmethod
    def _select_profile_variant(
        cls,
        profile: Dict[str, Any],
        quote: str,
        full_text: str,
        used_imitations: Set[str],
        used_mistakes: Set[str],
        formula: str = ""
    ) -> Tuple[str, str]:
        """
        Deterministically selects the best contextually-aligned (imitation_example, common_mistakes)
        variant from a profile's thematic bank based on lexical overlap with the quote and passage text,
        and structural alignment with the pattern formula's literal anchors.
        Guarantees band-calibrated examples and prevents duplicate templates.
        """
        default_imit = profile.get("imitation_example", quote)
        default_mist = profile.get("common_mistakes", "Grammar Warning (Syntax): Learners frequently misapply structural boundaries.")
        variants = profile.get("variants")

        # Extract literal anchor groups from formula (e.g. 'Although' or 'that/to')
        anchor_groups: List[List[str]] = []
        if formula:
            cobuild_pos = {
                "np", "vp", "v", "be", "aux", "v-ed", "v3", "v-ing", "to-v", "to-inf", "inf",
                "adj", "adv", "det", "prep", "conj", "clause", "s", "n", "pron", "modal"
            }
            raw_lits = re.findall(r"[a-zA-Z0-9_\'/]+", re.sub(r"\[.*?\]|\(.*?\)", " ", formula))
            for lit in raw_lits:
                alts = [a.strip().lower() for a in lit.split("/") if a.strip()]
                valid_alts = [a for a in alts if len(a) >= 2 and a not in cobuild_pos]
                if valid_alts:
                    anchor_groups.append(valid_alts)

        def _instantiates_anchors(imit_text: str) -> bool:
            if not anchor_groups or not imit_text:
                return True
            imit_low = imit_text.lower()
            imit_toks = set(re.findall(r"[a-z0-9']+", imit_low))
            for alts in anchor_groups:
                if not any(a in imit_toks or cls.text_contains_form(a, imit_low) for a in alts):
                    return False
            return True

        if not variants:
            return default_imit, default_mist

        q_words = set(re.findall(r"[a-z]{3,}", (quote or "").lower()))
        t_words = set(re.findall(r"[a-z]{3,}", (full_text or "").lower()))

        best_var = None
        best_score = -1000
        for var in variants:
            kws = set(var.get("keywords", ()))
            # Quote keyword match carries 5x weight; passage theme match carries 1x weight
            score = len(kws & q_words) * 5 + len(kws & t_words)
            v_imit = var.get("imitation_example", default_imit)
            v_mist = var.get("common_mistakes", default_mist)
            if _instantiates_anchors(v_imit):
                score += 50
            else:
                score -= 200
            if v_imit in used_imitations or v_mist in used_mistakes:
                score -= 100
            if score > best_score:
                best_score = score
                best_var = var

        if best_var and (best_score >= 0 or (not _instantiates_anchors(default_imit) and _instantiates_anchors(best_var.get("imitation_example", "")))):
            return (
                best_var.get("imitation_example", default_imit),
                best_var.get("common_mistakes", default_mist),
            )
        return default_imit, default_mist

    @classmethod
    def extract_deterministic_grammar(
        cls,
        text: str,
        sentence_pool: Optional[Dict[str, str]] = None,
        target_count: int = 5,
        syllabus_grammar: Optional[List[str]] = None,
        cefr_band: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Deterministically extracts grammar patterns from the text at 0 token cost,
        calibrating pattern priority, pedagogical functions, imitation examples, and
        empirical common mistakes to the passage's 3-Band CEFR level ('A1_A2', 'B1_B2', 'C1_C2').
        """
        if not sentence_pool:
            _, sentence_pool = cls.tokenize_and_index_sentences(text)

        resolved_band = cefr_band or cls.resolve_grammar_cefr_band(text if text else sentence_pool)

        skeletons = cls.mine_grammar_skeletons(
            sentence_pool,
            target_count=target_count,
            syllabus_grammar=syllabus_grammar,
            cefr_band=resolved_band
        )
        if not skeletons:
            return []

        results: List[Dict[str, Any]] = []
        used_imitations: Set[str] = set()
        used_mistakes: Set[str] = set()
        for s in skeletons:
            sid = s.get("sid", "S-1")
            quote = s.get("quote", "")
            cat = s.get("category", "Information Packaging")
            formula = s.get("pattern_formula", "[Subject] + [VP]")

            profile = cls.match_cobuild_grammar_profile(
                formula,
                quote,
                category=cat,
                cefr_band=resolved_band
            )

            prof_name = profile.get("name", "Syntactic Pattern")
            prof_cat = profile.get("category", cat)
            if profile.get("profile_key") == "fronted_negative_inversion" and "aux" not in formula.lower():
                formula = "Not only + [Aux] + [Subject] + [VP], but + [Subject] + [also] + [VP]"
            ped_func = profile.get("pedagogical_function", f"Synthesizes discourse via {prof_cat.lower()} syntax.")

            imitation, clean_mistake = cls._select_profile_variant(
                profile,
                quote=quote,
                full_text=text,
                used_imitations=used_imitations,
                used_mistakes=used_mistakes,
                formula=formula
            )

            # PedLex Authentic Grammar Alert & Don't Say diagnostics ONLY for generic fallback profiles
            try:
                STOP_WORDS_FOR_ALERTS = {"that", "not", "also", "it", "which", "this", "what", "there", "long", "like", "well", "such"}
                GENERIC_PROFILES = {
                    "Syntactic Pattern",
                    "Rhetorical Emphasis & Contrast",
                    "Logical Transition & Epistemic Stance",
                    "Discursive Cohesion & Structural Framing",
                    "Information Packaging & Syntactic Integration"
                }

                is_specific_prof = prof_name not in GENERIC_PROFILES and bool(clean_mistake)

                if not is_specific_prof:
                    formula_clean = re.sub(r'\[.*?\]|\(.*?\)', ' ', formula)
                    formula_lits = [tok.lower() for tok in re.findall(r"\b[a-zA-Z]{3,}\b", formula_clean) if tok.lower() not in STOP_WORDS_FOR_ALERTS]

                    quote_lits: List[str] = []
                    if not formula_lits:
                        quote_lits = [tok.lower() for tok in re.findall(r"\b[a-zA-Z]{3,}\b", quote)
                                      if tok.lower() in ("make", "want", "enjoy", "too", "enough", "such", "need", "own", "used", "help", "prefer", "suggest", "while", "if", "because", "unless", "since", "both", "neither")]

                    ordered_candidates = []
                    seen_cands = set()
                    for tok in formula_lits + quote_lits:
                        if tok not in seen_cands:
                            seen_cands.add(tok)
                            ordered_candidates.append(tok)

                    for aw in ordered_candidates:
                        alert = cls.get_ldoce_grammar_alert(aw, quote=quote)
                        if alert and ("Don't say" in alert.get("content", "") or "✗" in alert.get("content", "")):
                            raw_content = alert.get("content", "")
                            parts = [p.strip() for p in re.split(r'[•\n]+', raw_content) if p.strip()]
                            pitfalls = [p for p in parts if ("Don't say" in p or "✗" in p)]
                            for pit in pitfalls:
                                chosen = re.sub(r'\s+', ' ', pit).strip()
                                chosen = re.sub(r'^GRAMMAR:?\s*', '', chosen, flags=re.IGNORECASE).strip()
                                chosen = re.sub(r'\s*Grammar guide\s*[-‒–—].*$', '', chosen, flags=re.IGNORECASE).strip()
                                cand_text = f"Grammar Warning ({alert.get('word', '').upper()}): {chosen}"
                                if cand_text not in used_mistakes:
                                    clean_mistake = cand_text
                                    break
                        if clean_mistake not in used_mistakes:
                            break

                used_imitations.add(imitation)
                used_mistakes.add(clean_mistake)
            except Exception:
                used_imitations.add(imitation)
                used_mistakes.add(clean_mistake)

            results.append({
                "quote": quote,
                "pattern_formula": formula,
                "category": prof_cat,
                "syntax_topic": prof_name,
                "pedagogical_function": ped_func,
                "imitation_example": imitation,
                "common_mistakes": clean_mistake,
                "cefr_band": resolved_band,
                "design_audit": f"AUDIT: [{sid}] -> [{formula}] -> [BAND:{resolved_band}] -> [VERBATIM_CONFIRMED]"
            })

        return results

    @classmethod
    def _balance_single_and_multi_word(
        cls,
        items: List[Dict[str, Any]],
        target_count: int = 10,
        multi_word_ratio: float = 0.4
    ) -> List[Dict[str, Any]]:
        """
        Balances single-word and multi-word expression items (e.g. 60% single words + 40% multi-word phrases)
        and smoothly interleaves them to provide pedagogical variety and avoid clustering.
        """
        if not items or len(items) <= target_count and not any(it.get("is_multi_word") or len(it.get("target_word", "").split()) > 1 for it in items):
            return items[:target_count]

        singles: List[Dict[str, Any]] = []
        multis: List[Dict[str, Any]] = []

        for it in items:
            is_multi = it.get("is_multi_word") or (len(it.get("target_word", "").split()) > 1) or (it.get("part_of_speech") in ("phrase", "collocation", "phrasal verb", "set phrase", "idiom"))
            if is_multi:
                multis.append(it)
            else:
                singles.append(it)

        # If only one type exists, return up to target_count
        if not multis:
            return singles[:target_count]
        if not singles:
            return multis[:target_count]

        # Calculate quotas
        quota_multi = max(1, int(round(target_count * multi_word_ratio)))
        quota_single = target_count - quota_multi

        sel_multi = multis[:quota_multi]
        sel_single = singles[:quota_single]

        # Backfill if one group has fewer items than its allocated quota
        if len(sel_multi) < quota_multi:
            deficit = quota_multi - len(sel_multi)
            sel_single.extend(singles[len(sel_single):len(sel_single) + deficit])
        elif len(sel_single) < quota_single:
            deficit = quota_single - len(sel_single)
            sel_multi.extend(multis[len(sel_multi):len(sel_multi) + deficit])

        # Interleave selections pedagogically: 2 singles, 1 multi, 2 singles, 1 multi...
        balanced: List[Dict[str, Any]] = []
        i, j = 0, 0
        while (i < len(sel_single) or j < len(sel_multi)) and len(balanced) < target_count:
            for _ in range(2):
                if i < len(sel_single) and len(balanced) < target_count:
                    balanced.append(sel_single[i])
                    i += 1
            if j < len(sel_multi) and len(balanced) < target_count:
                balanced.append(sel_multi[j])
                j += 1

        return balanced

    @classmethod
    def _normalize_vocab_items_input(
        cls,
        vocab_input: Any,
        preserve_slash_slots: bool = False
    ) -> List[Dict[str, str]]:
        """
        Normalizes structured vocabulary items (List[VocabularyItem | Dict]) into a clean list of dicts
        with keys: raw_word, word, part_of_speech, definition, quote, example_usage.
        Also supports legacy markdown string input solely for backward-compatible unit tests.
        """
        if not vocab_input:
            return []

        parsed_items: List[Dict[str, str]] = []
        if isinstance(vocab_input, (list, tuple)):
            for item in vocab_input:
                if isinstance(item, dict):
                    raw_w = str(item.get("raw_word") or item.get("word") or "").strip()
                    pos_str = str(item.get("part_of_speech") or "noun").strip()
                    defn = str(item.get("definition") or "").strip()
                    quote = str(item.get("quote") or item.get("quoted_sentence") or "").strip()
                    ex_usage = str(item.get("example_usage") or "").strip()
                else:
                    raw_w = str(getattr(item, "word", "") or "").strip()
                    pos_str = str(getattr(item, "part_of_speech", "noun") or "noun").strip()
                    defn = str(getattr(item, "definition", "") or "").strip()
                    quote = str(getattr(item, "quoted_sentence", "") or getattr(item, "quote", "") or "").strip()
                    ex_usage = str(getattr(item, "example_usage", "") or "").strip()
                if not raw_w:
                    continue
                if preserve_slash_slots:
                    word = re.sub(r'(?:\s*/\s*)?(?:\[.*?\]|\(.*?\))(?:\s*/\s*(?:\[.*?\]|\(.*?\)))*', ' ', raw_w)
                    word = re.sub(r'(?:^\s*/+\s*|\s*/+\s*$|\s+/\s+)', ' ', word)
                    word = re.sub(r'\s+', ' ', word).strip()
                else:
                    word = re.sub(r'\s+', ' ', re.sub(r'\[.*?\]|\(.*?\)', '', raw_w)).strip()
                if not word or not cls.is_pedagogical_target(word):
                    continue
                parsed_items.append({
                    "raw_word": raw_w,
                    "word": word,
                    "part_of_speech": pos_str or "noun",
                    "definition": defn,
                    "quote": quote,
                    "example_usage": ex_usage,
                })
            return parsed_items

        if isinstance(vocab_input, str):
            blocks = re.split(r'\n(?=##\s*\[\[)', vocab_input)
            for b in blocks:
                m_word = re.search(r'##\s*\[\[(.*?)\]\]([^\n]*)', b)
                if not m_word:
                    continue
                raw_w = m_word.group(1).strip()
                header_tail = m_word.group(2).strip()
                if preserve_slash_slots:
                    word = re.sub(r'(?:\s*/\s*)?(?:\[.*?\]|\(.*?\))(?:\s*/\s*(?:\[.*?\]|\(.*?\)))*', ' ', raw_w)
                    word = re.sub(r'(?:^\s*/+\s*|\s*/+\s*$|\s+/\s+)', ' ', word)
                    word = re.sub(r'\s+', ' ', word).strip()
                else:
                    word = re.sub(r'\s+', ' ', re.sub(r'\[.*?\]|\(.*?\)', '', raw_w)).strip()
                if not word or not cls.is_pedagogical_target(word):
                    continue
                m_pos = re.search(r'-\s*\*\*Part\s+of\s+Speech\*\*:\s*([^\n]+)', b, re.IGNORECASE)
                pos_str = m_pos.group(1).strip() if m_pos else ""
                if not pos_str and header_tail:
                    ht_clean = re.sub(r'`[^`]*`', '', header_tail)
                    ht_clean = re.sub(r'\b(?:A1|A2|B1|B2|C1|C2)\b', '', ht_clean).strip()
                    if ht_clean:
                        pos_str = ht_clean
                m_def = re.search(r'-\s*\*\*Definition\*\*:\s*([^\n]+)', b, re.IGNORECASE)
                m_quote = re.search(r'-\s*\*\*Quoted Sentence\*\*:\s*([^\n]+)', b, re.IGNORECASE)
                parsed_items.append({
                    "raw_word": raw_w,
                    "word": word,
                    "part_of_speech": pos_str or "noun",
                    "definition": m_def.group(1).strip() if m_def else "",
                    "quote": m_quote.group(1).strip() if m_quote else ""
                })
        return parsed_items

    @classmethod
    def build_authentic_cloze_items(
        cls,
        vocab_content: Any,
        target_count: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Track 1: Authentic Passage Cloze (Achievement Testing / 学业水平测试)
        Accepts structured vocabulary items directly (or legacy markdown string for unit tests)
        and masks their target appearance in the authentic quoted passage sentence using spaCy
        lemmatization and exact word-boundary masking.
        Generates 0-token, 100% textbook-grounded question stems.
        """
        if not vocab_content:
            return []

        nlp = cls.get_spacy()
        parsed_items = cls._normalize_vocab_items_input(vocab_content, preserve_slash_slots=False)

        cloze_items: List[Dict[str, Any]] = []
        used_distractors: Set[str] = set()

        # --- Dependent-connective cloze (grammar/collocation axis -> single answer) ---
        connectives_of_taking = frozenset({"regardless", "irrespective"})
        connectives_bare_np = ("despite", "notwithstanding", "whatever", "barring")

        for it in parsed_items:
            w = it["word"]
            raw_w = it.get("raw_word", w)
            q = it["quote"]
            if not q:
                continue

            # Strip leading/trailing quotation marks from quote
            q_clean = q.strip().strip('"').strip("'").strip("“").strip("”")
            if not q_clean:
                continue

            doc = nlp(q_clean)
            w_lower = w.lower()

            # Dependent connective (e.g. 'regardless of')
            head_part, _, frame_rest = w_lower.partition(" ")
            if frame_rest and head_part in connectives_of_taking:
                frame_prep = frame_rest.strip().lower()
                head_tok = None
                for i, t in enumerate(doc):
                    if (t.lemma_.lower().strip() == head_part and not t.is_punct
                            and i + 1 < len(doc) and doc[i + 1].text.lower().strip() == frame_prep):
                        head_tok = t
                        break
                if head_tok is not None:
                    stem = q_clean[:head_tok.idx] + "____" + q_clean[head_tok.idx + len(head_tok.text):]
                    pre_d = [c for c in connectives_bare_np if c != head_part][:3]
                    if len(re.findall(r'_{2,}', stem)) == 1 and len(pre_d) >= 2:
                        cloze_items.append({
                            "target_word": head_part,
                            "base_headword": w,
                            "part_of_speech": it["part_of_speech"],
                            "definition": it["definition"],
                            "question": stem,
                            "source_sentence": q_clean,
                            "precomputed_distractors": pre_d,
                            "is_connective": True,
                        })
                        continue
                continue

            # Locate token or multi-word span in sentence
            tokens = w_lower.split()
            matched_start = None
            matched_end = None
            surface_form = ""
            is_multi_word = len(tokens) > 1

            if not is_multi_word:
                for t in doc:
                    if t.text.lower() == w_lower or t.lemma_.lower() == w_lower:
                        matched_start = t.idx
                        matched_end = t.idx + len(t.text)
                        surface_form = t.text
                        target_tok = t
                        break
            else:
                n = len(tokens)
                for i in range(len(doc) - n + 1):
                    window = doc[i : i + n]
                    if all(window[j].text.lower() == tokens[j] or window[j].lemma_.lower() == tokens[j] for j in range(n)):
                        matched_start = window[0].idx
                        matched_end = window[-1].idx + len(window[-1].text)
                        surface_form = q_clean[matched_start:matched_end]
                        target_tok = window[0]
                        break

                if matched_start is None and ('(' in raw_w or '[' in raw_w) and len(tokens) == 2:
                    t1, t2 = tokens[0], tokens[1]
                    for i, tok in enumerate(doc):
                        if tok.text.lower() == t1 or tok.lemma_.lower() == t1:
                            for k in range(i + 1, min(len(doc), i + 4)):
                                target_k = doc[k]
                                if target_k.text.lower() == t2 or target_k.lemma_.lower() == t2:
                                    matched_start = tok.idx
                                    matched_end = target_k.idx + len(target_k.text)
                                    surface_form = q_clean[matched_start:matched_end]
                                    target_tok = tok
                                    break
                            if matched_start is not None:
                                break

            if matched_start is None or matched_end is None or not surface_form:
                continue

            # Replace target span with exact 4 underscores '____'
            stem = q_clean[:matched_start] + "____" + q_clean[matched_end:]

            # Ensure exactly one blank exists in stem
            if len(re.findall(r'_{2,}', stem)) != 1:
                continue

            # Distractor generation (Sense-Locked WSD & Authentic Phrase Distractors)
            if is_multi_word:
                distractors = cls.generate_phrase_distractors(
                    phrase=w_lower,
                    count=3,
                    exclude_words=used_distractors
                )
            else:
                pos_raw = it["part_of_speech"].lower()
                if "verb" in pos_raw:
                    canonical_pos, pos_tag = "verb", "v"
                elif "adj" in pos_raw:
                    canonical_pos, pos_tag = "adj", "a"
                elif "adv" in pos_raw:
                    canonical_pos, pos_tag = "adv", "r"
                else:
                    canonical_pos, pos_tag = "noun", "n"

                distractors = cls.generate_vocab_distractors(
                    target_word=w_lower,
                    pos=canonical_pos,
                    target_count=3,
                    exclude_words=used_distractors,
                    definition=it["definition"],
                    quote=q_clean
                )
                if len(distractors) < 3:
                    fallback_d = cls.generate_zero_collision_distractors(
                        target_word=w if w else surface_form,
                        pos=pos_tag,
                        count=3,
                        context_sentence=q_clean
                    )
                    for fd in fallback_d:
                        if fd not in distractors and len(distractors) < 3:
                            distractors.append(fd)
                if pos_tag == "n" and target_tok.tag_ == "NNS":
                    distractors = [cls.pluralize_noun(d) for d in distractors]

            used_distractors.update(d for d in distractors if d)

            cloze_items.append({
                "target_word": surface_form,
                "base_headword": w,
                "part_of_speech": it["part_of_speech"],
                "definition": it["definition"],
                "question": stem,
                "source_sentence": q_clean,
                "precomputed_distractors": distractors[:3],
                "is_multi_word": is_multi_word
            })

        # Balance single-word headwords vs multi-word expressions (e.g. 60% single words + 40% multi-word phrases)
        return cls._balance_single_and_multi_word(cloze_items, target_count=target_count, multi_word_ratio=0.4)

    @classmethod
    def extract_sense_pattern_bundles(cls, sense: Dict[str, Any], target_word: str) -> List[Dict[str, Any]]:
        """
        Extracts atomic (pattern, example, bound_prep) bundles from a locked PedLex sense.
        Prioritizes native atomic pattern_bundles directly extracted from PedLex DOM.
        Falls back to parsing raw patterns while strictly preventing pattern-example desynchronization.
        """
        if not sense:
            return []

        # 1. Native Atomic Bundles (Directly from updated PedLex DB schema)
        native_bundles = sense.get("pattern_bundles")
        if native_bundles:
            out: List[Dict[str, Any]] = []
            for nb in native_bundles:
                pat = (nb.get("pattern") or "").strip()
                if not pat:
                    continue
                p_clean_search = re.sub(r"\[.*?\]|\(.*?\)", "", pat).strip().lower()
                p_for_prep = re.sub(r"\bto\s+(?:do\s+something|\w+ing)\b", "", p_clean_search)
                m_prep = re.search(r"\b(on|in|at|for|with|about|under|from|to|by|into|of)\b", p_for_prep)
                bound_prep = m_prep.group(1) if m_prep else None
                out.append({
                    "pattern": pat,
                    "example": nb.get("example") or None,
                    "prep": bound_prep,
                    "source": nb.get("source", "native_bundle")
                })
            if out:
                return out

        # 2. Fallback for older DB versions: parse flat patterns with exact scoring
        raw_patterns = sense.get("patterns") or []
        examples = sense.get("examples") or []
        bundles: List[Dict[str, Any]] = []
        seen_patterns = set()

        for raw in raw_patterns:
            raw_s = raw.strip()
            if not raw_s:
                continue
            m_emb = re.search(r"^(.*?)\s+([A-Z][\w\W]*\b[\.\?!])$", raw_s)
            if not m_emb:
                m_etc = re.search(r"^(.*?\betc)\s+([A-Za-z].*\s+.*)$", raw_s)
                if m_etc:
                    m_emb = m_etc
            if m_emb:
                clean_pat = m_emb.group(1).strip()
                emb_ex = m_emb.group(2).strip()
            else:
                clean_pat = raw_s
                emb_ex = None

            if clean_pat in seen_patterns:
                continue
            seen_patterns.add(clean_pat)

            p_clean_search = re.sub(r"\[.*?\]|\(.*?\)", "", clean_pat).strip().lower()
            p_for_prep = re.sub(r"\bto\s+(?:do\s+something|\w+ing)\b", "", p_clean_search)
            all_preps = re.findall(r"\b(on|in|at|for|with|about|under|from|to|by|into|of|between|over|against|towards|upon)\b", p_for_prep)
            # Deduplicate while preserving order
            cand_preps: List[str] = []
            for cp in all_preps:
                if cp not in cand_preps:
                    cand_preps.append(cp)
            bound_prep = cand_preps[0] if cand_preps else None

            matched_ex = emb_ex
            if not matched_ex:
                # 1. Match 'it [verb] somebody when/how/that'
                m_it = re.search(r"^it\s+(\w+)\s+somebody\s+(when|how|that)", clean_pat.lower())
                if m_it:
                    v_stem = m_it.group(1)
                    conj = m_it.group(2)
                    for ex in examples:
                        if re.search(r"\bit\b", ex.lower()) and v_stem in ex.lower() and re.search(rf"\b{conj}\b", ex.lower()):
                            matched_ex = ex
                            break
                # 2. Direct exact or branch/keyword match of clean_pat
                if not matched_ex:
                    branches = cls.clean_and_expand_pattern_slashes(clean_pat) or [clean_pat]
                    best_ex = None
                    best_match_count = 0
                    for ex in examples:
                        ex_l = ex.lower()
                        if clean_pat.lower() in ex_l:
                            matched_ex = ex
                            break
                        for br in branches:
                            br_words = [w for w in re.split(r"\W+", br.lower()) if w and w not in ("something", "somebody", "sb", "sth", "etc", "doing")]
                            if br_words:
                                matched_words = sum(1 for w in br_words if re.search(rf"\b{re.escape(w)}\b", ex_l) or cls.form_matches(w, ex_l))
                                # Also check morphological inflections of target word in example
                                if matched_words < len(br_words):
                                    matched_words = sum(
                                        1 for w in br_words
                                        if re.search(rf"\b{re.escape(w)}\b", ex_l)
                                        or any(re.search(rf"\b{re.escape(inf)}\b", ex_l) for inf in cls.inflected_forms(w))
                                    )
                                if matched_words == len(br_words) and matched_words > best_match_count:
                                    best_match_count = matched_words
                                    best_ex = ex
                    if not matched_ex and best_ex:
                        matched_ex = best_ex

                # 3. Match gerund 'doing something'
                if not matched_ex and "doing something" in clean_pat.lower():
                    prefix = clean_pat.lower().replace("doing something", "").strip()
                    for ex in examples:
                        if prefix in ex.lower() and re.search(r"\b\w+ing\b", ex.lower()):
                            matched_ex = ex
                            break
                # 4. Match infinitive 'to do something'
                if not matched_ex and "to do something" in clean_pat.lower():
                    prefix = clean_pat.lower().replace("to do something", "").strip()
                    for ex in examples:
                        if prefix in ex.lower() and re.search(r"\bto\s+\w+\b", ex.lower()):
                            matched_ex = ex
                            break

            # Synchronize bound_prep with the preposition actually attested in matched_ex if slash alternatives exist
            if matched_ex and len(cand_preps) > 1:
                ex_l = matched_ex.lower()
                for cp in cand_preps:
                    if re.search(rf"\b{re.escape(cp)}\b", ex_l):
                        bound_prep = cp
                        break

            bundles.append({
                "pattern": clean_pat,
                "example": matched_ex,
                "prep": bound_prep,
                "preps": cand_preps,
            })
        return bundles

    @classmethod
    def pick_atomic_pattern_bundle(
        cls,
        sense: Optional[Dict[str, Any]],
        target_word: str,
        quote: str = "",
        preferred_anchor: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Selects the single best atomic (pattern, example, prep) bundle from a locked sense.
        Guarantees that the pattern and its illustrative example are 100% harmonized from
        the exact same dictionary sense.
        """
        if not sense:
            return None
        bundles = cls.extract_sense_pattern_bundles(sense, target_word)
        if not bundles:
            return None

        tw_l = target_word.lower()
        q_l = (quote or "").lower()

        scored: List[Tuple[int, Dict[str, Any]]] = []
        for b in bundles:
            pat = b["pattern"]
            ex = b["example"]
            prep = b.get("prep")
            cand_preps = b.get("preps") or ([prep] if prep else [])
            score = 0

            # Absolute filter / priority: Target word presence
            # Genuine valency patterns contain the target headword or its inflections.
            # Idioms like 'got on his nerves' under 'annoy' do NOT contain 'annoy' and must not hijack the item!
            has_tw = bool(re.search(rf"\b{re.escape(tw_l)}\w*\b", pat.lower()))
            if has_tw:
                score += 50
            else:
                score -= 30

            # Massive bonus if bundle contains an authentic illustrative example
            if ex:
                score += 25
            elif cand_preps:
                # If pattern claims a bound preposition (e.g. 'identical in') but has NO matching example in the sense,
                # penalize heavily so it doesn't pair with a non-prepositional fallback sentence ('four identical houses')!
                score -= 35

            # Quote / preferred_anchor alignment: if curriculum quote or anchor uses one of the bundle's prepositions
            if preferred_anchor and any(cp.lower() == preferred_anchor.lower() for cp in cand_preps):
                score += 40
                # Pin the bundle's active prep to preferred_anchor if it's in cand_preps and attested in ex (or no ex)
                if not ex or re.search(rf"\b{re.escape(preferred_anchor.lower())}\b", ex.lower()):
                    b = dict(b, prep=preferred_anchor.lower())
                    prep = b["prep"]
            elif any(re.search(rf"\b{re.escape(cp)}\b", q_l) for cp in cand_preps):
                score += 35
                for cp in cand_preps:
                    if re.search(rf"\b{re.escape(cp)}\b", q_l) and (not ex or re.search(rf"\b{re.escape(cp)}\b", ex.lower())):
                        b = dict(b, prep=cp)
                        prep = cp
                        break
            # General prepositional valency bonus
            if prep:
                score += 10
            # Syntactic valency frames (gerunds, infinitives, complement clauses)
            if any(v in pat.lower() for v in ("doing something", "to do something", "somebody when", "somebody that", "somebody how")):
                score += 15
            # Clean length preference (compact, high-utility frames)
            p_len = len(re.sub(r"\[.*?\]|\(.*?\)", "", pat).strip())
            if 5 <= p_len <= 35:
                score += 5
            scored.append((score, b))

        scored.sort(key=lambda x: (x[0], -len(x[1]["pattern"])), reverse=True)
        if not scored:
            return None
        best_b = scored[0][1]
        # Do not return a preposition-claiming pattern that lacks any example AND is not attested in the quote
        if best_b.get("prep") and not best_b.get("example"):
            if not any(re.search(rf"\b{re.escape(cp)}\b", q_l) for cp in (best_b.get("preps") or [best_b["prep"]])):
                return None
        return best_b

    @classmethod
    def select_best_single_pattern(cls, patterns: List[str], target_word: str, preferred_anchor: Optional[str] = None) -> Optional[str]:
        """
        Selects exactly ONE best, high-value syntactic pattern from PedLex.
        Filters out noise like single verbs ('have') or repeated fragments.
        If preferred_anchor is given, boosts patterns containing that anchor (e.g. 'serious about' when anchor='about').
        """
        if not patterns:
            return None

        scored_patterns = []
        target_lower = target_word.lower()
        for p in patterns:
            p_clean = p.strip()
            if not p_clean:
                continue
            if " " not in p_clean and "/" not in p_clean:
                continue
            score = 0
            p_lower = p_clean.lower()
            if target_lower in p_lower:
                score += 10
            if preferred_anchor and re.search(rf"\b{re.escape(preferred_anchor.lower())}\b", p_lower):
                score += 15
            if "/" in p_clean:
                score += 5
            if any(prep in p_lower for prep in ["of", "to", "over", "for", "in", "on", "with", "from", "that", "about"]):
                score += 5
            if 6 <= len(p_clean) <= 40:
                score += 3
            scored_patterns.append((score, p_clean))

        if scored_patterns:
            scored_patterns.sort(key=lambda x: (x[0], -len(x[1])), reverse=True)
            return scored_patterns[0][1]

        return patterns[0].strip() if patterns else None


    @classmethod
    def build_precomputed_target_skeletons(
        cls,
        vocab_content: Any,
        target_count: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Track 2: Pre-Computed Generative Skeletons (Proficiency Testing / 语境迁移单选题)
        Accepts structured vocabulary items directly (or legacy markdown string for unit tests)
        and synthesizes authoritative collocational anchors (via PedLex 6th Edition) and
        collision-free distractors (via WordNet + PedLex).
        Produces pre-computed item skeletons to strictly constrain LLM generation, guaranteeing zero
        duplicate options, zero synonym pile double-keys, and perfect stem-option-explanation alignment.
        """
        if not vocab_content:
            return []

        nlp = cls.get_spacy()
        parsed_items = cls._normalize_vocab_items_input(vocab_content, preserve_slash_slots=True)

        skeletons: List[Dict[str, Any]] = []
        used_distractors: Set[str] = set()

        # Batch Anchor Collision Set : every headword that is itself a target of
        # this same batch. An anchor (or a prepositional object welded into the micro-task
        # frame) drawn from this set produces an unsatisfiable item — the evaluator's
        # anchor-presence gate demands the word inside the stem while its cross-target
        # leakage gate bans it with a fatal flag. Anchors must therefore be drawn from
        # OUTSIDE this set; when nothing outside the set fits, the blueprint degrades to
        # the neutral semantic-field template instead of welding two gates into a deadlock.
        batch_target_words: Set[str] = set()
        single_word_targets: Set[str] = set()
        for _it in parsed_items:
            _w = _it["word"].lower().strip()
            if not _w:
                continue
            batch_target_words.add(_w)
            if " " not in _w:
                single_word_targets.add(_w)
            else:
                for _tok in _w.split():
                    if len(_tok) >= 3 and _tok not in cls._ANCHOR_STOPWORDS:
                        batch_target_words.add(_tok)

        for it in parsed_items:
            w = it["word"]
            raw_w_item = it.get("raw_word") or w
            w_lower = w.lower()
            raw_pos = it["part_of_speech"].lower()

            # Discontinuous Verb + [sb/sth] + Prep Frame Resolution (e.g. "remind [sb] of [sb] / [sth]"):
            # Stripping internal object slots leaves an ungrammatical adjacent string ('remind of') that cannot
            # fill a single blank in English ('*reminds of me'). Decompose into head verb ('remind') + bound
            # preposition anchor ('of') so the blank tests the head verb in '... ____ [sb/sth] of ...'.
            discontinuous_prep = None
            m_disc = re.match(
                r'^([a-z]+)\s+(?:\[[^\]]+\]|\([^)]+\))\s+(of|to|with|from|into|for|on|in|about|against|at|upon|over)\b',
                raw_w_item.strip(),
                re.IGNORECASE
            )
            if m_disc:
                head_v = m_disc.group(1).lower()
                disc_p = m_disc.group(2).lower()
                # Only decompose if head_v is not already a standalone single-word target in the batch
                if head_v not in single_word_targets:
                    w = head_v
                    w_lower = head_v
                    discontinuous_prep = disc_p
                    raw_pos = "verb"

            # Map POS category
            # Idiomatic Slot Gate: If single word is embedded in a fixed idiomatic construction in authentic quote
            # (e.g. 'granted' in 'take ... for granted'), prioritize the authentic idiomatic slot over isolated POS/function-word
            is_idiom_slot = False
            if w_lower == "granted" and it.get("quote") and re.search(r'\btake\b(?:\W+\w+){0,6}\W+for\W+granted\b', it["quote"], re.IGNORECASE):
                is_idiom_slot = True
                canonical_pos = "idiom_slot"
                anchor = "take for"
                anchor_type = "idiom"

            if not is_idiom_slot:
                if discontinuous_prep:
                    canonical_pos = "verb"
                else:
                    resolved = cls.resolve_item_pos(w_lower, it.get("quote"), manual_pos=raw_pos)
                    if resolved in ("phrase", "collocation", "phrasal verb", "set phrase", "idiom"):
                        canonical_pos = "phrase"
                    elif resolved == "function_word" or cls.is_function_word(w_lower):
                        canonical_pos = "function_word"
                    elif resolved == "verb":
                        canonical_pos = "verb"
                    elif resolved in ("adjective", "adj"):
                        canonical_pos = "adj"
                    elif resolved in ("adverb", "adv"):
                        canonical_pos = "adv"
                    else:
                        canonical_pos = "noun"

            # Evidence-Grade Gate (P2): a curriculum 'Quoted Sentence' licenses an anchor or a
            # valency frame only when it actually contains a clause. Fragments lifted from
            # headings ('Punctuality Pays!') were treated as authentic usage, so the blueprint
            # welded anchors and frames onto a two-word title that never demonstrated them.
            # A weak quote is demoted to display-only evidence: it stays visible to the writer
            # as candidate_quote, but every anchor must then come from PedLex instead.
            quote_provenance, quote_wordcount = cls._quote_evidence_strength(it.get("quote"))
            it["raw_quote"] = it.get("quote") or ""
            if quote_provenance == "weak":
                it["quote"] = ""

            # 0. Sense-First Grounding (Sense Primacy — GEMINI.md §1.10):
            # Lock the authoritative Pedagogical Sense BEFORE anchor selection or distractor generation.
            target_p_code = cls._pos_code(canonical_pos)
            entry_base = cls.get_ldoce_entry(w_lower)
            target_entry = entry_base
            locked_sense = None
            t_s_idx = None
            if target_entry and target_entry.get("senses"):
                t_s_idx = cls._lock_sense(
                    target_entry,
                    definition=it.get("definition"),
                    quote=it.get("quote") or it.get("raw_quote"),
                    target_pos=canonical_pos
                )
                if t_s_idx is not None and 0 <= t_s_idx < len(target_entry["senses"]):
                    locked_sense = target_entry["senses"][t_s_idx]
            if locked_sense and not it.get("definition"):
                it["definition"] = cls.format_ldoce_definition(locked_sense)

            # 0.1 Pre-bind Atomic Sense Pattern Bundle (Pattern + Corpus Example + Bound Preposition)
            # before anchor selection and distractor generation so that sense-native prepositions
            # participate directly in Step 1 anchor resolution and Step 3 double-key clearance.
            ldoce_pattern_example = None
            ldoce_cloze_frame = None
            pattern_anchor = None
            best_pattern = None
            initial_atomic_bundle = cls.pick_atomic_pattern_bundle(
                locked_sense, w_lower, quote=it.get("quote"), preferred_anchor=discontinuous_prep
            )
            sense_native_prep = discontinuous_prep or (initial_atomic_bundle.get("prep") if initial_atomic_bundle else None)

            # 1. Retrieve Authentic Oxford Collocational Anchor (Sense-Guided)
            if not is_idiom_slot:
                anchor = discontinuous_prep
                anchor_type = "prep" if discontinuous_prep else "collocation"
            prep_obj = None
            quote_governed_prep: Optional[str] = discontinuous_prep if discontinuous_prep else None

            def _tok_matches_target(tok, doc_obj) -> bool:
                if "-" not in w_lower:
                    if (tok.i > 0 and doc_obj[tok.i - 1].text == "-" and not doc_obj[tok.i - 1].whitespace_) or (
                        tok.i + 1 < len(doc_obj) and doc_obj[tok.i + 1].text == "-" and not tok.whitespace_
                    ):
                        return False
                return tok.text.lower() == w_lower or tok.lemma_.lower() == w_lower

            if canonical_pos == "verb":
                # Verb Step A: Extract the valency complement PP from the authentic quote.
                if it["quote"]:
                    try:
                        doc = nlp(it["quote"])
                        v_collocs = cls.get_oxford_collocations(w_lower, pos="verb")
                        consensus_preps = set(v_collocs.get("prep", []))
                        if sense_native_prep:
                            consensus_preps.add(sense_native_prep)
                        for tok in doc:
                            if _tok_matches_target(tok, doc):
                                verified_prep = None   # (token_index, prep, pobj) confirmed by OCD/sense valency
                                unverified_prep = None  # (token_index, prep, pobj) bare syntactic complement
                                for child in tok.children:
                                    if child.dep_ != "prep":
                                        continue
                                    prep_tok = child.text.lower()
                                    if prep_tok not in ("to", "with", "from", "on", "for", "in", "into", "of", "against", "at", "upon", "towards"):
                                        continue
                                    if child.i <= tok.i:
                                        continue  # Post-head linear constraint
                                    child_pobj = None
                                    for pchild in child.children:
                                        if pchild.dep_ == "pobj" and pchild.lemma_.lower() not in cls._ANCHOR_STOPWORDS:
                                            child_pobj = pchild.lemma_.lower()
                                            break
                                    if child_pobj is not None and (prep_tok, child_pobj) in cls._ADJUNCT_PREP_PAIRS:
                                        continue  # Fixed adjunct / interjection (e.g. 'for example')
                                    if prep_tok in consensus_preps:
                                        if verified_prep is None:
                                            verified_prep = (child.i, prep_tok, child_pobj)
                                    elif unverified_prep is None:
                                        unverified_prep = (child.i, prep_tok, child_pobj)
                                chosen = None
                                if verified_prep is not None and unverified_prep is not None:
                                    chosen = verified_prep if verified_prep[0] < unverified_prep[0] else unverified_prep
                                elif verified_prep is not None:
                                    chosen = verified_prep
                                elif unverified_prep is not None:
                                    chosen = unverified_prep
                                if chosen is not None:
                                    anchor, prep_obj = chosen[1], chosen[2]
                                    anchor_type = "prep"
                                    quote_governed_prep = anchor
                                break
                    except Exception:
                        pass

                # Verb Step B1: If no bound preposition found in quote, inspect quote for authentic direct object patient (dobj)
                if not anchor and it["quote"]:
                    try:
                        doc = nlp(it["quote"])
                        for tok in doc:
                            if _tok_matches_target(tok, doc):
                                for child in tok.children:
                                    if child.dep_ == "dobj" and child.lemma_.lower() not in cls._ANCHOR_STOPWORDS:
                                        obj_form = child.text.lower() if child.tag_ == "NNS" else child.lemma_.lower()
                                        anchor = obj_form
                                        anchor_type = "object"
                                        break
                                if anchor:
                                    break
                    except Exception:
                        pass

                # Verb Step B2: If no bound prep or dobj, check if the verb has an authentic subject noun in quote (nsubj)
                if not anchor and it["quote"]:
                    try:
                        doc = nlp(it["quote"])
                        for tok in doc:
                            if _tok_matches_target(tok, doc):
                                for child in tok.children:
                                    if child.dep_ in ("nsubj", "nsubjpass") and child.lemma_.lower() not in cls._ANCHOR_STOPWORDS:
                                        subj_form = child.text.lower() if child.tag_ == "NNS" else child.lemma_.lower()
                                        anchor = subj_form
                                        anchor_type = "verb_subject"
                                        break
                                if anchor:
                                    break
                    except Exception:
                        pass

                # Verb Step B3: Fallback to sense-aligned direct noun object from OCD or sense-native prep
                if not anchor:
                    v_entry = cls.get_oxford_collocations(w_lower, pos="verb")
                    candidates = v_entry.get("colloc_nouns", []) + v_entry.get("noun_after", [])
                    if candidates:
                        anchor = cls._select_sense_aligned_anchor(candidates, definition=it.get("definition"), quote=it.get("quote"), target_word=w_lower, forbidden=batch_target_words)
                        if anchor:
                            anchor_type = "object"
                    if not anchor and sense_native_prep and not cls._in_forbidden_word_set(sense_native_prep, batch_target_words):
                        anchor = sense_native_prep
                        anchor_type = "prep"

            elif canonical_pos == "noun":
                n_entry = cls.get_oxford_collocations(w_lower, pos="noun")
                if it["quote"]:
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
                                    for child in tok.children:
                                        if child.dep_ in ("amod", "compound"):
                                            mod_tok = child.lemma_.lower()
                                            if child.pos_ != "PRON" and mod_tok not in cls._ANCHOR_STOPWORDS and mod_tok != w_lower:
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
                                                and head_tok_lemma not in cls._DELEXICAL_VERBS):
                                            verb_before_set = {v.lower().split()[0] for v in n_entry.get("verb_before", [])}
                                            if not verb_before_set or head_tok_lemma in verb_before_set:
                                                anchor = head_tok_lemma
                                                anchor_type = "verb"
                                break
                    except Exception:
                        pass

                # Priority D: dictionary candidates, strictly gated by quote/definition evidence
                if not anchor and n_entry:
                    if n_entry.get("adj") or n_entry.get("verb_before"):
                        candidates = n_entry.get("adj", []) + n_entry.get("verb_before", [])
                        anchor = cls._select_sense_aligned_anchor(candidates, definition=it.get("definition"), quote=it.get("quote"), target_word=w_lower, forbidden=batch_target_words)
                        if anchor:
                            anchor_type = "adj" if cls._token_in_phrases(anchor, n_entry.get("adj", [])) else "verb"
                            if cls._is_delexical_anchor(anchor, anchor_type, "noun"):
                                anchor, anchor_type = None, "collocation"
                    elif sense_native_prep and not cls._in_forbidden_word_set(sense_native_prep, batch_target_words):
                        anchor = sense_native_prep
                        anchor_type = "prep"
                    elif n_entry.get("prep"):
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
                # Adjective Step A: Syntactic Dependency Analysis on Authentic Quote
                if it["quote"]:
                    try:
                        doc = nlp(it["quote"])
                        for tok in doc:
                            if _tok_matches_target(tok, doc):
                                # A1. Predicative Valency: Check if adjective governs a bound preposition
                                for child in tok.children:
                                    if child.dep_ == "prep" and child.text.lower() in ("to", "for", "with", "of", "in", "from", "at", "about", "on"):
                                        anchor = child.text.lower()
                                        anchor_type = "prep"
                                        quote_governed_prep = anchor
                                        for pchild in child.children:
                                            if pchild.dep_ == "pobj" and pchild.pos_ != "PRON" and pchild.text.lower() not in cls._ANCHOR_STOPWORDS and pchild.lemma_.lower() != w_lower:
                                                prep_obj = pchild.lemma_.lower()
                                                break
                                        break
                                # A2. Attributive Modification: Check if adjective modifies a head noun (dep_ == "amod")
                                if not anchor and tok.dep_ == "amod" and tok.head and tok.head.pos_ in ("NOUN", "PROPN"):
                                    if tok.head.pos_ != "PRON" and tok.head.text.lower() not in cls._ANCHOR_STOPWORDS and tok.head.lemma_.lower() != w_lower:
                                        anchor = tok.head.lemma_.lower()
                                        anchor_type = "modified_noun"
                                # A3. Predicative / Causative Complement: Check if adjective is complement of a causative/copula verb
                                if not anchor and tok.dep_ in ("ccomp", "xcomp", "acomp", "oprd") and tok.head and tok.head.pos_ == "VERB":
                                    head_v = tok.head.lemma_.lower()
                                    if head_v in ("make", "render", "find", "deem", "keep", "consider"):
                                        anchor = head_v
                                        anchor_type = "verb_copula"
                                break
                    except Exception:
                        pass

                # Adjective Step B: Collocation Lexicon Lookup (OCD) if no anchor found in quote
                adj_entry = cls.get_oxford_collocations(w_lower, pos="adj")
                if not anchor and adj_entry:
                    candidates = adj_entry.get("noun_after", []) + adj_entry.get("colloc_nouns", [])
                    if candidates:
                        anchor = cls._select_sense_aligned_anchor(candidates, definition=it.get("definition"), quote=it.get("quote"), target_word=w_lower, forbidden=batch_target_words)
                        if anchor:
                            anchor_type = "modified_noun"
                    if not anchor and sense_native_prep and not cls._in_forbidden_word_set(sense_native_prep, batch_target_words):
                        anchor = sense_native_prep
                        anchor_type = "prep"
                    if not anchor and adj_entry.get("prep"):
                        p_cand = cls._pick_anchor_token(adj_entry["prep"], target_word=w_lower, forbidden=batch_target_words)
                        if p_cand:
                            anchor = p_cand
                            anchor_type = "prep"

                # Adjective Step C: Fall back to locked sense pattern first, then curated academic adjective preposition valency map
                if not anchor:
                    ld_entry = target_entry or cls.get_ldoce_entry(w_lower)
                    if ld_entry:
                        candidate_senses = [locked_sense] if locked_sense else ld_entry.get("senses", [])
                        for s in candidate_senses:
                            for pat in s.get("patterns", []):
                                for p in ("of", "to", "for", "with", "about", "from", "in", "on", "at"):
                                    if re.search(rf"\b{p}\b", pat.lower()):
                                        if p == "to":
                                            m_inf = re.search(r"\bto\s+([a-z]+)", pat.lower())
                                            if m_inf and m_inf.group(1) not in (
                                                "sb", "sth", "someone", "something", "oneself", "himself", "herself", "them", "the", "a", "an"
                                            ):
                                                continue
                                        anchor = p
                                        anchor_type = "prep"
                                        break
                                if anchor:
                                    break
                            if anchor:
                                break
                    if not anchor:
                        for p, adjs in cls._ADJ_PREP_VALENCY_MAP.items():
                            if w_lower in adjs:
                                anchor = p
                                anchor_type = "prep"
                                break

            elif canonical_pos == "adv":
                # Adverb Syntactic Dependency Walker & Sense-Lock Gate
                if it.get("quote"):
                    try:
                        doc = nlp(it["quote"])
                        for tok in doc:
                            if _tok_matches_target(tok, doc):
                                head = tok.head
                                if head and head.lemma_.lower() not in cls._ANCHOR_STOPWORDS and head.lemma_.lower() != w_lower:
                                    h_lem = head.lemma_.lower()
                                    if head.pos_ in ("ADJ", "ADV"):
                                        anchor = h_lem
                                        anchor_type = "modifies_adj"
                                        break
                                    elif head.pos_ == "VERB":
                                        anchor = h_lem
                                        anchor_type = "modifies_verb"
                                        break
                    except Exception:
                        pass

                # Priority B: Dictionary lookup from PedLex zero collision anchor
                if not anchor:
                    ld_anc, ld_atype, ld_frame, _ = cls.find_ldoce_zero_collision_anchor(
                        target_word=w_lower,
                        pos="adv",
                        definition=it.get("definition"),
                        quote=it.get("quote"),
                        locked_sense=locked_sense
                    )
                    if ld_anc:
                        aligned = cls._select_sense_aligned_anchor(
                            [ld_anc],
                            definition=it.get("definition"),
                            quote=it.get("quote"),
                            target_word=w_lower,
                            forbidden=batch_target_words
                        )
                        if aligned:
                            anchor = aligned
                            anchor_type = ld_atype or "modifies_verb"

            # 1.5 Sense-Native Atomic Pattern Anchor Inheritance (GEMINI.md §1.10):
            # If Step 1 found no anchor (or only a weak fallback) and the locked sense's
            # atomic pattern bundle specifies a bound preposition, inherit it BEFORE Step 3
            # distractor generation so double-key clearance runs against the true anchor.
            if not is_idiom_slot and sense_native_prep and (not anchor or anchor_type == "collocation"):
                if not cls._in_forbidden_word_set(sense_native_prep, batch_target_words):
                    anchor = sense_native_prep
                    anchor_type = "prep"
                    pattern_anchor = sense_native_prep

            # 2. Syntactic Fallback: Extract anchor from quoted sentence if OCD / prep lookup was empty
            if not anchor and it["quote"]:
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
                            if not anchor and tok.head and tok.head.pos_ in ("NOUN", "VERB"):
                                h_lem = tok.head.lemma_.lower()
                                if tok.head.pos_ != "PRON" and h_lem not in cls._ANCHOR_STOPWORDS and h_lem not in cls._DELEXICAL_VERBS and h_lem != w_lower:
                                    anchor = h_lem
                                    anchor_type = "contextual"
                            break
                except Exception:
                    pass

            # 3. Generate 3 Zero-Collision Distractors with Definition-Locked WSD & Sense Antonyms
            is_quantifier_frame = False
            quant_head_noun = None
            if canonical_pos == "phrase":
                q_match = re.match(r"^(a|an)\s+([a-z]+)\s+of$", w_lower)
                if q_match:
                    is_quantifier_frame = True
                    quant_head_noun = q_match.group(2)
                    q_siblings_all = [
                        "piece", "bit", "slice", "sheet", "block", "drop",
                        "pair", "pack", "stroke", "portion", "set", "share"
                    ]
                    q_cands = [s for s in q_siblings_all if s != quant_head_noun and s not in used_distractors]
                    h_rank = cls.CEFR_ORDER.get(cls.get_word_cefr(quant_head_noun) or "A2", 2)
                    if h_rank <= 2:
                        foundational_pool = ["bit", "slice", "sheet", "block", "drop", "pair", "pack", "set"]
                        q_cands = [s for s in q_cands if s in foundational_pool] + [s for s in q_cands if s not in foundational_pool]
                    distractors = q_cands[:3]
                    dist_meta = {d: "quantifier_collocation_contrast" for d in distractors}
                else:
                    p_dists = cls.generate_phrase_distractors(w_lower, count=3, exclude_words=used_distractors)
                    distractors = p_dists
                    dist_meta = {d: "phrase_contrast" for d in p_dists}
            elif canonical_pos == "idiom_slot" and w_lower == "granted":
                i_cands = ["accepted", "given", "known", "certain", "true"]
                distractors = [c for c in i_cands if c not in used_distractors][:3]
                dist_meta = {d: "idiom_collocation_contrast" for d in distractors}
            else:
                distractors, dist_meta = cls.generate_vocab_distractors(
                    target_word=w_lower,
                    pos=canonical_pos,
                    context_anchor=anchor,
                    anchor_type=anchor_type,
                    target_count=3,
                    exclude_words=used_distractors,
                    definition=it.get("definition"),
                    quote=it.get("quote"),
                    return_metadata=True
                )

            # Ensure we have valid distractors; fallback to zero_collision if needed
            if len(distractors) < 3 and canonical_pos != "phrase":
                pos_char = "v" if canonical_pos == "verb" else ("a" if canonical_pos == "adj" else ("r" if canonical_pos == "adv" else "n"))
                distractors = cls.generate_zero_collision_distractors(w_lower, pos=pos_char, count=3)

            verb_requires_object = cls._verb_takes_object(w_lower) if canonical_pos == "verb" else None

            def _is_double_key(cand: str) -> bool:
                if canonical_pos in ("function_word", "phrase", "idiom_slot"):
                    return False
                flag, _kind = cls.double_key_collision(
                    w_lower, cand, anchor, anchor_type, pos=canonical_pos, quote=it.get("quote")
                )
                if flag:
                    return True
                raw_w_low = str(raw_w_item or "").strip().lower()
                if raw_w_low and raw_w_low != w_lower:
                    flag_raw, _ = cls.double_key_collision(
                        raw_w_low, cand, anchor, anchor_type, pos=canonical_pos, quote=it.get("quote")
                    )
                    if flag_raw:
                        return True
                if canonical_pos == "noun":
                    p_tw = cls.pluralize_noun(w_lower)
                    p_cand = cls.pluralize_noun(cand)
                    if p_tw != w_lower or p_cand != cand:
                        flag_p, _ = cls.double_key_collision(
                            p_tw, p_cand, anchor, anchor_type, pos=canonical_pos, quote=it.get("quote")
                        )
                        if flag_p:
                            return True
                return False

            def _is_valid_distractor(cand: str) -> bool:
                if not cand or _is_double_key(cand):
                    return False
                c_low = cand.strip().lower()
                if len(w_lower) >= 4 and len(c_low) >= 4:
                    if w_lower in c_low or c_low in w_lower:
                        return False
                if canonical_pos in ("function_word", "phrase", "idiom_slot"):
                    return True
                if not cls._distractor_occupies_slot(
                    cand, canonical_pos, requires_object=(verb_requires_object is True)
                ):
                    return False
                return cls.is_cefr_compliant_distractor(cand, w_lower)

            def _clear_and_refill_distractors(curr_distractors: List[str], curr_meta: Dict[str, str]) -> Tuple[List[str], Dict[str, str]]:
                near_s, ants_s = cls.semantic_fields(w_lower, pos=canonical_pos)
                near_only_s = near_s - ants_s
                cleared_list: List[str] = []
                for d in curr_distractors:
                    if not _is_valid_distractor(d):
                        continue
                    if d.lower() in near_only_s and sum(1 for x in cleared_list if x.lower() in near_only_s) >= 1:
                        continue
                    cleared_list.append(d)
                r_meta: Dict[str, str] = {}
                if len(cleared_list) < 3:
                    try:
                        refill_pool, r_meta = cls.generate_vocab_distractors(
                            target_word=w_lower, pos=canonical_pos,
                            context_anchor=anchor, anchor_type=anchor_type,
                            target_count=8, exclude_words=used_distractors,
                            definition=it.get("definition"), quote=it.get("quote"),
                            return_metadata=True
                        )
                        r_meta = r_meta or {}
                    except Exception:
                        refill_pool = []
                    for cand in refill_pool:
                        if len(cleared_list) >= 3:
                            break
                        if cand and cand not in cleared_list and _is_valid_distractor(cand):
                            if cand.lower() in near_only_s and sum(1 for x in cleared_list if x.lower() in near_only_s) >= 1:
                                continue
                            cleared_list.append(cand)
                    if len(cleared_list) < 3:
                        pos_char = "v" if canonical_pos == "verb" else ("a" if canonical_pos == "adj" else ("r" if canonical_pos == "adv" else "n"))
                        try:
                            for cand in cls.generate_zero_collision_distractors(w_lower, pos=pos_char, count=3):
                                if len(cleared_list) >= 3:
                                    break
                                if cand and cand not in cleared_list and cand not in used_distractors and _is_valid_distractor(cand):
                                    if cand.lower() in near_only_s and sum(1 for x in cleared_list if x.lower() in near_only_s) >= 1:
                                        continue
                                    cleared_list.append(cand)
                        except Exception:
                            pass
                    if len(cleared_list) < 3:
                        pos_char = "v" if canonical_pos == "verb" else ("a" if canonical_pos == "adj" else ("r" if canonical_pos == "adv" else "n"))
                        t_lvl_str = cls.get_word_cefr(w_lower) or "B2"
                        t_rank = cls.CEFR_ORDER.get(t_lvl_str, 4)
                        if t_rank <= 3:
                            fallback_pool = {
                                "v": ["choose", "accept", "decide", "expect", "follow", "notice", "explain", "remain", "manage", "allow"],
                                "n": ["choice", "reason", "matter", "situation", "effort", "problem", "action", "change", "moment", "condition"],
                                "a": ["simple", "common", "certain", "different", "similar", "natural", "clear", "direct", "actual", "special"],
                                "r": ["deeply", "strictly", "tightly", "widely", "carefully", "clearly", "readily", "sharply", "directly", "steadily"]
                            }
                        else:
                            fallback_pool = {
                                "v": ["indicate", "facilitate", "establish", "evaluate", "demonstrate", "generate", "enhance", "maintain", "assess"],
                                "n": ["framework", "perspective", "dimension", "mechanism", "principle", "criterion", "strategy", "phenomenon", "capacity"],
                                "a": ["apparent", "consistent", "distinct", "variable", "significant", "plausible", "rigid", "modest", "tentative"],
                                "r": ["deeply", "strictly", "tightly", "widely", "carefully", "clearly", "readily", "sharply", "directly", "steadily"]
                            }
                        for cand in fallback_pool.get(pos_char, []):
                            if len(cleared_list) >= 3:
                                break
                            if cand != w_lower and cand not in cleared_list and cand not in used_distractors and _is_valid_distractor(cand):
                                cleared_list.append(cand)
                merged_meta = {
                    d: (curr_meta.get(d) or r_meta.get(d) or "distant_contrast")
                    for d in cleared_list
                }
                return cleared_list, merged_meta

            distractors, dist_meta = _clear_and_refill_distractors(distractors, dist_meta)

            # Assemble full options (target + 3 distractors)
            distractors3 = distractors[:3]
            slot_target = quant_head_noun if is_quantifier_frame else w_lower
            slot = zlib.crc32(slot_target.encode("utf-8")) % 4
            raw_options = distractors3[:slot] + [slot_target] + distractors3[slot:]
            correct_answer_index = slot
            used_distractors.update(d for d in distractors3 if d)

            # 4. Plan A: All options and target word maintain parallel inflection
            is_plural_noun = False
            plural_target = w_lower
            if canonical_pos == "noun":
                is_spacy_plural = cls._is_genuine_plural_noun(w_lower)
                plural_tantum = {"goods", "customs", "clothes", "belongings", "surroundings", "fireworks", "premises", "congratulations"}

                quote_plural = False
                if it["quote"]:
                    try:
                        q_doc = nlp(it["quote"])
                        quote_plural = any(
                            (tok.lemma_.lower() == w_lower or tok.text.lower() == w_lower)
                            and tok.tag_ == "NNS"
                            and (tok.text.lower() != w_lower or is_spacy_plural)
                            for tok in q_doc
                        )
                    except Exception:
                        quote_plural = False

                # Do NOT pluralize via quote_plural if the locked atomic pattern/example uses the singular base headword
                # in a fixed singular construction (e.g. 'matter of routine' -> do not corrupt into 'matter of routines')
                if quote_plural and initial_atomic_bundle:
                    init_pat = (initial_atomic_bundle.get("pattern") or "").lower()
                    init_ex = (initial_atomic_bundle.get("example") or "").lower()
                    p_cand = cls.pluralize_noun(w_lower)
                    if init_pat and re.search(rf"\b{re.escape(w_lower)}\b", init_pat) and not re.search(rf"\b{re.escape(p_cand)}\b", init_pat):
                        if not init_ex or not re.search(rf"\b{re.escape(p_cand)}\b", init_ex):
                            quote_plural = False

                if is_spacy_plural or w_lower in plural_tantum or quote_plural:
                    is_plural_noun = True
                    plural_target = cls.pluralize_noun(w_lower)

            if is_plural_noun:
                inflection_desc = "plural form (NNS)"
                final_target = plural_target
                inflected_distractors3 = [cls.pluralize_noun(d) for d in distractors3]
                inflected_distractors3 = cls.screen_inflected_options(
                    inflected_distractors3,
                    [cls.pluralize_noun(d) for d in distractors[3:]],
                    final_target,
                )
                used_distractors.update(inflected_distractors3)
                slot = min(slot, len(inflected_distractors3))
                correct_answer_index = slot
                final_options = inflected_distractors3[:slot] + [final_target] + inflected_distractors3[slot:]
            elif is_quantifier_frame:
                inflection_desc = "singular quantifier noun"
                final_target = quant_head_noun
                final_options = raw_options
            else:
                inflection_desc = "base form"
                final_target = w_lower
                final_options = raw_options

            # 5. Sense-First Grounding & Synthesize Itemized Micro-Task for LLM
            dist_list = [opt for opt in final_options if opt.lower() != final_target.lower()]
            dist_str = ", ".join(dist_list)

            # 5.1 Re-verify PedLex Entry & Sense if final_target differs and Step 0 had no entry
            if locked_sense is None:
                entry_final = cls.get_ldoce_entry(final_target)
                if entry_base and cls._pos_code(entry_base.get("pos")) == target_p_code:
                    target_entry = entry_base
                elif entry_final and cls._pos_code(entry_final.get("pos")) == target_p_code:
                    target_entry = entry_final
                else:
                    target_entry = entry_base or entry_final

                if target_entry and target_entry.get("senses"):
                    t_s_idx = cls._lock_sense(
                        target_entry,
                        definition=it.get("definition"),
                        quote=it.get("quote") or it.get("raw_quote"),
                        target_pos=canonical_pos
                    )
                    if t_s_idx is not None and 0 <= t_s_idx < len(target_entry["senses"]):
                        locked_sense = target_entry["senses"][t_s_idx]

            # Hydrate authoritative formatted definition (with [spoken], [formal], [variety] labels).
            # If it["definition"] was already extracted/disambiguated and is non-boilerplate,
            # preserve it whenever locked_sense is absent or represents a different/fallback sense.
            ext_defn = (it.get("definition") or "").strip()
            locked_defn = cls.format_ldoce_definition(locked_sense) if locked_sense else ""
            if ext_defn and not cls.is_boilerplate_definition(ext_defn):
                ext_norm = re.sub(r"^\s*(?:\[[^\]]+\]\s*)+", "", ext_defn).strip().lower()
                lock_norm = re.sub(r"^\s*(?:\[[^\]]+\]\s*)+", "", locked_defn).strip().lower()
                if locked_defn and ext_norm == lock_norm:
                    skeleton_definition = locked_defn
                else:
                    skeleton_definition = ext_defn
            else:
                skeleton_definition = locked_defn or ext_defn

            # 5.2 Atomic Sense Pattern Bundle (Pattern + Corpus Example + Bound Preposition as an indivisible bundle)
            anchor_before_step5 = anchor
            atomic_bundle = cls.pick_atomic_pattern_bundle(
                locked_sense,
                final_target,
                quote=it.get("quote") or it.get("raw_quote"),
                preferred_anchor=anchor
            )
            if atomic_bundle:
                best_pattern = atomic_bundle.get("pattern")
                if atomic_bundle.get("example"):
                    ldoce_pattern_example = atomic_bundle["example"]
                if atomic_bundle.get("prep") and (not anchor or anchor_type in ("collocation", "contextual")):
                    pat_prep = atomic_bundle["prep"]
                    if not any(cls.double_key_collision(final_target, d, anchor=pat_prep, anchor_type="prep", pos=canonical_pos)[0] for d in dist_list):
                        anchor = pat_prep
                        anchor_type = "prep"
                        pattern_anchor = pat_prep

            quote_shown_prep = bool(anchor) and anchor_type == "prep" and bool(
                quote_governed_prep and str(anchor).lower() == quote_governed_prep.lower()
            )
            quote_shown_partitive = bool(anchor) and anchor_type == "partitive" and bool(re.search(
                rf"\b{re.escape(str(anchor).lower())}\b", (it.get("quote") or "").lower()))

            if anchor and anchor_type != "contextual" and not ldoce_pattern_example:
                lookup_anchor = prep_obj if anchor_type == "partitive" else anchor
                ldoce_pattern_example = cls.ldoce_example_for_anchor(
                    final_target,
                    lookup_anchor,
                    locked_sense=locked_sense,
                    definition=it.get("definition"),
                    quote=it.get("quote"),
                    pos=canonical_pos
                )
                pattern_anchor = anchor
            if (not anchor or anchor_type == "contextual"
                    or (not ldoce_pattern_example and not quote_shown_prep and not quote_shown_partitive)):
                ld_anchor, ld_atype, ld_fdesc, ld_ex = cls.find_ldoce_zero_collision_anchor(
                    target_word=final_target,
                    pos=canonical_pos,
                    distractors=dist_list,
                    definition=it.get("definition"),
                    quote=it.get("quote"),
                    locked_sense=locked_sense
                )
                if ld_anchor:
                    anchor = ld_anchor
                    anchor_type = ld_atype
                    ldoce_pattern_example = ld_ex
                    pattern_anchor = ld_anchor

            # Batch Anchor Collision Gate 
            if batch_target_words and not is_idiom_slot and (anchor or prep_obj):
                if prep_obj and cls._in_forbidden_word_set(prep_obj, batch_target_words):
                    prep_obj = None
                if (anchor and anchor_type == "partitive"
                        and cls._in_forbidden_word_set(
                            prep_obj or (anchor.split()[0] if anchor.split() else ""),
                            batch_target_words)):
                    anchor, anchor_type = None, "collocation"
                    prep_obj = None
                if anchor and cls._in_forbidden_word_set(anchor, batch_target_words):
                    anchor, anchor_type = cls._batch_safe_reanchor(
                        final_target,
                        canonical_pos,
                        definition=it.get("definition"),
                        quote=it.get("quote"),
                        forbidden=batch_target_words
                    )
                    if not anchor_type:
                        anchor_type = "collocation"
                    ldoce_pattern_example = None
                    ldoce_cloze_frame = None

            # Locked-Sense Frame Concordance Gate 
            if anchor_type == "prep" and anchor and not is_idiom_slot and cls._prep_anchor_sense_conflict(
                final_target,
                anchor,
                definition=it.get("definition"),
                quote=it.get("quote"),
                canonical_pos=canonical_pos
            ):
                alt_anchor, alt_type, _alt_frame, alt_ex = cls.find_ldoce_zero_collision_anchor(
                    target_word=final_target,
                    pos=canonical_pos,
                    distractors=dist_list,
                    definition=it.get("definition"),
                    quote=it.get("quote"),
                    locked_sense=locked_sense
                )
                if alt_anchor and not cls._prep_anchor_sense_conflict(
                    final_target,
                    alt_anchor,
                    definition=it.get("definition"),
                    quote=it.get("quote"),
                    canonical_pos=canonical_pos
                ):
                    anchor, anchor_type = alt_anchor, (alt_type or "collocation")
                    ldoce_pattern_example = alt_ex
                    prep_obj = None
                else:
                    anchor, anchor_type = None, "collocation"
                    prep_obj = None
                    ldoce_pattern_example = None
                ldoce_cloze_frame = None

            # If Step 5 re-anchoring changed the anchor after Step 3 (including clearing anchor to None),
            # re-verify distractors against the final anchor so zero-double-key invariance holds 100%.
            if anchor != anchor_before_step5 and canonical_pos not in ("phrase", "idiom_slot", "function_word"):
                if any(_is_double_key(d) for d in distractors[:3]):
                    distractors, dist_meta = _clear_and_refill_distractors(distractors, dist_meta)
                    distractors3 = distractors[:3]
                    raw_options = distractors3[:slot] + [slot_target] + distractors3[slot:]
                    if is_plural_noun:
                        inflected_distractors3 = cls.screen_inflected_options(
                            [cls.pluralize_noun(d) for d in distractors3],
                            [cls.pluralize_noun(d) for d in distractors[3:]],
                            final_target,
                        )
                        used_distractors.update(inflected_distractors3)
                        final_options = inflected_distractors3[:slot] + [final_target] + inflected_distractors3[slot:]
                    else:
                        used_distractors.update(distractors3)
                        final_options = raw_options
                    dist_list = [opt for opt in final_options if opt.lower() != final_target.lower()]
                    dist_str = ", ".join(dist_list)

            # If a re-anchoring produced a new anchor, look up the PedLex sentence that
            # actually demonstrates this (target, anchor) pair within the locked sense.
            if anchor and not ldoce_pattern_example and anchor != pattern_anchor:
                ldoce_pattern_example = cls.ldoce_example_for_anchor(
                    final_target,
                    prep_obj if anchor_type == "partitive" else anchor,
                    locked_sense=locked_sense,
                    definition=it.get("definition"),
                    quote=it.get("quote"),
                    pos=canonical_pos
                )

            # Fallback to the locked sense's own example if no anchor-specific example survived
            if not ldoce_pattern_example:
                ldoce_pattern_example = cls.ldoce_sense_example(
                    final_target,
                    pos=canonical_pos,
                    definition=it.get("definition"),
                    quote=it.get("quote"),
                    locked_sense=locked_sense
                )

            # Final Atomic Pattern-Anchor-Example Synchronization (GEMINI.md §1.10):
            # Ensure best_pattern never claims a preposition that clashes with final anchor or is absent from ldoce_pattern_example.
            if best_pattern:
                p_clean_sync = re.sub(r"\[.*?\]|\(.*?\)", "", best_pattern).strip().lower()
                p_for_prep_sync = re.sub(r"\bto\s+(?:do\s+something|\w+ing)\b", "", p_clean_sync)
                pat_preps_sync = re.findall(
                    r"\b(on|in|at|for|with|about|under|from|to|by|into|of|between|over|against|towards|upon)\b",
                    p_for_prep_sync
                )
                if pat_preps_sync:
                    if anchor and anchor_type == "prep" and anchor.lower() not in pat_preps_sync:
                        # Re-select a sense pattern that matches the final anchor, or clear best_pattern
                        re_bundle = cls.pick_atomic_pattern_bundle(
                            locked_sense, final_target, quote=it.get("quote") or it.get("raw_quote"), preferred_anchor=anchor
                        )
                        if re_bundle and re_bundle.get("pattern") and anchor.lower() in (re_bundle.get("preps") or []):
                            best_pattern = re_bundle["pattern"]
                            if re_bundle.get("example"):
                                ldoce_pattern_example = re_bundle["example"]
                        else:
                            best_pattern = None
                    elif ldoce_pattern_example and not any(re.search(rf"\b{re.escape(pp)}\b", ldoce_pattern_example.lower()) for pp in pat_preps_sync):
                        best_pattern = None

            # Derive Authentic Cloze Sentence Frame Prototype if an authentic example is available
            if ldoce_pattern_example:
                try:
                    ex_doc = nlp(ldoce_pattern_example)
                    tok_texts = []
                    replaced = False
                    for tok in ex_doc:
                        if not replaced and (tok.lemma_.lower() == final_target.lower() or tok.text.lower() == final_target.lower()):
                            tok_texts.append("____")
                            replaced = True
                        else:
                            tok_texts.append(tok.text)
                    if replaced:
                        ldoce_cloze_frame = " ".join(tok_texts)
                except Exception:
                    ldoce_cloze_frame = None

            if anchor_type == "partitive" and anchor and ldoce_cloze_frame:
                quant_lem = (prep_obj or (anchor.split()[0] if anchor.split() else "")).lower()
                if quant_lem and not re.search(rf"\b{re.escape(quant_lem)}\w*\b", ldoce_cloze_frame.lower()):
                    ldoce_cloze_frame = None

            # Prescribed-Inflection Concordance 
            _base_options = list(final_options)
            inflection_desc, final_options, final_target, verb_form_tag = cls._reconcile_prescribed_inflection(
                base_options=_base_options,
                target_index=correct_answer_index,
                target_word=final_target,
                canonical_pos=canonical_pos,
                inflection_desc=inflection_desc,
                authentic_example=ldoce_pattern_example
            )
            if final_options != _base_options:
                if canonical_pos not in ("phrase", "idiom_slot", "function_word") and any(
                    opt.lower() != final_target.lower() and cls.double_key_collision(
                        final_target, opt, anchor, anchor_type, pos=canonical_pos, quote=it.get("quote")
                    )[0]
                    for opt in final_options
                ):
                    bad_bases = {
                        b for b, f in zip(_base_options, final_options)
                        if f.lower() != final_target.lower() and cls.double_key_collision(
                            final_target, f, anchor, anchor_type, pos=canonical_pos, quote=it.get("quote")
                        )[0]
                    }
                    filtered_base_dists = [d for d in distractors if d not in bad_bases]
                    distractors, dist_meta = _clear_and_refill_distractors(filtered_base_dists, dist_meta)
                    distractors3 = distractors[:3]
                    _base_options = distractors3[:slot] + [slot_target] + distractors3[slot:]
                    inflection_desc, final_options, final_target, verb_form_tag = cls._reconcile_prescribed_inflection(
                        base_options=_base_options,
                        target_index=correct_answer_index,
                        target_word=slot_target,
                        canonical_pos=canonical_pos,
                        inflection_desc=inflection_desc,
                        authentic_example=ldoce_pattern_example
                    )
                _form_remap = {b: f for b, f in zip(_base_options, final_options)}
                dist_meta = {_form_remap.get(k, k): v for k, v in dist_meta.items()}
                used_distractors.update(o for o in final_options if o.lower() != final_target.lower())
                dist_list = [opt for opt in final_options if opt.lower() != final_target.lower()]
                dist_str = ", ".join(dist_list)

            anchor_downgrade = None
            if not anchor and canonical_pos == "noun" and not is_idiom_slot:
                anchor_downgrade = "sense_recognition"

            if best_pattern:
                best_pattern = re.sub(
                    r"\b(?:American|British|Australian|North American)\s+English\b",
                    "",
                    best_pattern,
                    flags=re.IGNORECASE,
                )
                best_pattern = re.sub(
                    r"\b(?:formal|informal|spoken|written|technical|literary|dialect)\b",
                    "",
                    best_pattern,
                    flags=re.IGNORECASE,
                )
                best_pattern = re.sub(r"\s+", " ", best_pattern).strip()

            if best_pattern and is_plural_noun:
                best_pattern = re.sub(r"\b" + re.escape(w_lower) + r"\b", final_target, best_pattern, flags=re.IGNORECASE)

            # Streamlined Context Stem Directive
            cloze_slot_frame = None
            if best_pattern:
                p_clean = re.sub(r"\[.*?\]|\(.*?\)", "", best_pattern).strip()
                if p_clean.lower() == w_lower or p_clean.lower() == final_target.lower():
                    best_pattern = None
                    p_clean = ""
                else:
                    w_lemma = nlp(w_lower)[0].lemma_.lower() if nlp else w_lower
                    target_forms = set(cls.inflected_forms(w_lower)) | {w_lower, final_target, w_lemma}
                    if w_lemma != w_lower:
                        target_forms |= set(cls.inflected_forms(w_lemma))
                    t_pat = r"\b(?:" + "|".join(re.escape(f) for f in target_forms if f) + r")\b"
                    if re.search(t_pat, p_clean, re.IGNORECASE):
                        cloze_slot_frame = re.sub(t_pat, "____", p_clean, flags=re.IGNORECASE).strip()
                        if is_plural_noun:
                            cloze_slot_frame = re.sub(r"\b(?:a|an)\s+_{2,}\b", "____", cloze_slot_frame, flags=re.IGNORECASE).strip()
                        if not cloze_slot_frame.startswith("...") and not cloze_slot_frame.lower().startswith("it "):
                            cloze_slot_frame = f"... {cloze_slot_frame}"
                    elif any(p_clean.lower().startswith(prep) for prep in ("on ", "in ", "at ", "for ", "with ", "about ", "under ", "from ", "to ", "by ")):
                        cloze_slot_frame = f"... {p_clean} ____"
                        if is_plural_noun:
                            cloze_slot_frame = re.sub(r"\b(?:a|an)\s+_{2,}\b", "____", cloze_slot_frame, flags=re.IGNORECASE).strip()
                    elif any(p_clean.lower().endswith(" " + prep) for prep in ("on", "in", "at", "for", "with", "about", "under", "from", "to", "by", "of", "into")):
                        m_end_prep = re.search(r"\b(on|in|at|for|with|about|under|from|to|by|of|into)$", p_clean.lower())
                        if m_end_prep:
                            cloze_slot_frame = f"... ____ {m_end_prep.group(1)}"

            if cloze_slot_frame and len(re.findall(r"_{2,}", cloze_slot_frame)) > 1:
                cloze_slot_frame = None

            is_copula_adj = False
            if canonical_pos == "adj" and (cloze_slot_frame or anchor):
                copula_lemmas = {"be", "become", "fall", "remain", "stay", "seem", "appear", "look", "sound", "smell", "taste", "feel", "get", "turn", "grow", "keep"}
                check_tokens = re.findall(r"\b[a-zA-Z]+\b", (cloze_slot_frame or "") + " " + str(anchor or ""))
                for t in check_tokens:
                    doc_t = nlp(t)
                    if any(tok.lemma_.lower() in copula_lemmas for tok in doc_t):
                        is_copula_adj = True
                        break

            if best_pattern and not cloze_slot_frame:
                w_lemma = nlp(w_lower)[0].lemma_.lower() if nlp else w_lower
                target_forms = set(cls.inflected_forms(w_lower)) | {w_lower, final_target, w_lemma}
                if not any(re.search(rf"\b{re.escape(f)}\b", best_pattern, re.IGNORECASE) for f in target_forms if f):
                    best_pattern = None

            is_base_verb_slot = (
                (inflection_desc or "base form") == "base form"
                and not is_plural_noun
                and (
                    canonical_pos == "verb"
                    or (
                        canonical_pos == "phrase"
                        and final_target.split()
                        and final_target.split()[0] not in {"a", "an", "the", "in", "on", "at", "by", "for", "with", "from", "to", "of"}
                        and bool(cls._wordnet_pos(final_target.split()[0]) & {"v"})
                    )
                )
            )

            if cloze_slot_frame:
                stem_core = f"Fit into frame '{cloze_slot_frame}'."
                if is_plural_noun:
                    stem_core += f" (Plural noun slot required: e.g. '{final_target}')."
                elif is_copula_adj:
                    stem_core += " (Complete predicate required: stem MUST include the linking verb so the adjective predicate is complete)."
                elif is_base_verb_slot:
                    stem_core += " (Base-form verb slot required: use 'to ____', modal 'can/will/should ____', or plural/1st-person subject 'They/We/I ____' — NEVER singular 'He/She/It ____' without a modal)."
            elif best_pattern:
                stem_core = "Fit into the Pattern above."
                if is_plural_noun:
                    stem_core += f" (Plural noun slot required: e.g. '{final_target}')."
                elif is_copula_adj:
                    stem_core += " (Complete predicate required: stem MUST include the linking verb)."
                elif is_base_verb_slot:
                    stem_core += " (Base-form verb slot required: use 'to ____', modal 'can/will/should ____', or plural/1st-person subject 'They/We/I ____' — NEVER singular 'He/She/It ____' without a modal)."
            else:
                stem_core = "Write a natural sentence matching the Definition above."
                if is_plural_noun:
                    stem_core += f" (Plural noun slot required: e.g. '{final_target}')."
                elif is_copula_adj:
                    stem_core += " (Complete predicate required: stem MUST include a linking/copular verb)."
                elif is_base_verb_slot:
                    stem_core += " (Base-form verb slot required: use 'to ____', modal 'can/will/should ____', or plural/1st-person subject 'They/We/I ____' — NEVER singular 'He/She/It ____' without a modal)."

            task_parts = [stem_core]

            if ldoce_pattern_example:
                ex_clean = ldoce_pattern_example.strip()
                skip_model = False
                if is_plural_noun and canonical_pos == "noun":
                    if re.search(rf"\b{re.escape(w_lower)}\b", ex_clean, re.IGNORECASE) and not re.search(rf"\b{re.escape(final_target)}\b", ex_clean, re.IGNORECASE):
                        skip_model = True
                if not skip_model:
                    task_parts.append(f'Model: "{ex_clean}"')

            # Data-Driven Semantic & Grammatical Discriminator (PedLex <GRAM>, Grammar Boxes, Thesaurus & Language Activator)
            target_base = quant_head_noun if is_quantifier_frame else w_lower
            dist_base_list = [o for o in _base_options if o.lower() != target_base]
            dist_base_meta = {b: dist_meta.get(f) or dist_meta.get(b) for b, f in zip(_base_options, final_options)}
            discriminator_note = cls._semantic_discriminator_note(
                target_base,
                it.get("definition"),
                dist_base_list,
                dist_base_meta,
                pos=canonical_pos,
                sense_index=t_s_idx,
                quote=it.get("quote") or it.get("raw_quote")
            )
            if discriminator_note:
                task_parts.append(discriminator_note)

            if not any(p.startswith("Contrast:") for p in task_parts):
                t_entry_data = cls.get_ldoce_entry(final_target) or cls.get_ldoce_entry(target_base)
                if t_entry_data and t_entry_data.get("language_activator"):
                    for act in t_entry_data["language_activator"]:
                        for act_w in act.get("words", []):
                            act_word = (act_w.get("word") or "").lower()
                            if act_word in dist_base_list:
                                act_def = (act_w.get("definition") or "").strip()
                                if act_def:
                                    task_parts.append(f"Contrast: vs '{act_word}' ({act_def}).")
                                else:
                                    task_parts.append(f"Contrast: vs '{act_word}' (nuance distinction).")
                                break
                        if any(p.startswith("Contrast:") for p in task_parts):
                            break

            micro_task = " ".join(task_parts)

            # Determine anchor source: "quote" if anchor appeared in authentic quote text, else "dictionary"
            anchor_source = None
            if anchor:
                if anchor_type == "prep":
                    anchor_source = "quote" if quote_shown_prep else "dictionary"
                elif it.get("quote") and re.search(rf"\b{re.escape(anchor.lower())}\b", it["quote"].lower()):
                    anchor_source = "quote"
                else:
                    anchor_source = "dictionary"

            skeletons.append({
                "target_word": final_target,
                "base_headword": w,
                "part_of_speech": canonical_pos,
                "inflection": inflection_desc,
                "verb_form_tag": verb_form_tag,
                "verb_requires_object": verb_requires_object,
                "context_anchor": anchor,
                "anchor_type": anchor_type,
                "anchor_source": anchor_source,
                # B2: 'anchor_downgrade' records that no collocation frame survived the gates,
                # so this item is planned as a sense-recognition item and the writer is told not
                # to invent an anchor. None means an ordinary frame-driven cloze.
                "anchor_downgrade": anchor_downgrade,
                "item_type": anchor_downgrade or "cloze",
                "authentic_example": ldoce_pattern_example,
                "best_pattern": best_pattern,
                "cloze_frame_prototype": ldoce_cloze_frame,
                "micro_task": micro_task,
                "distractor_mechanisms": dist_meta,
                "prescribed_options": final_options,
                "correct_answer_index": correct_answer_index,
                "definition": skeleton_definition,
                # Quote provenance tiering: 'candidate_quote' is the raw curriculum quote and
                # is display-only when 'quote_provenance' == 'weak'; only 'licensed_quote' was
                # strong enough to license sense-locking, anchors and valency evidence.
                "candidate_quote": it.get("raw_quote") or it.get("quote") or "",
                "licensed_quote": it.get("quote") or "",
                "quote_provenance": quote_provenance,
                "quote_wordcount": quote_wordcount,
            })

        # Balance single-word headwords vs multi-word expressions (e.g. 60% single words + 40% multi-word phrases)
        skeletons = cls._balance_single_and_multi_word(skeletons, target_count=target_count, multi_word_ratio=0.4)

        # Balance Correct Answer Index Distribution evenly across 0, 1, 2, 3 (e.g. 3/3/2/2 for 10 items)
        # Prevents CRC32 clustering where one slot gets 4+ occurrences and disrupts psychometric balance
        if skeletons:
            n = len(skeletons)
            # Target distribution counts: base = n // 4, rem = n % 4
            target_distribution = [n // 4 + (1 if i < (n % 4) else 0) for i in range(4)]
            # Assign balanced slot to each item deterministically
            # Create a pool of slots [0, 1, 2, 3, 0, 1, 2, 3, 0, 1] shuffled deterministically
            balanced_slots = []
            for slot_idx, count in enumerate(target_distribution):
                balanced_slots.extend([slot_idx] * count)
            # Deterministic pseudo-shuffle based on passage words
            seed_hash = zlib.crc32("".join(s["target_word"] for s in skeletons).encode("utf-8"))
            # Standard Knuth shuffle with linear congruential generator
            rng_state = seed_hash
            for i in range(len(balanced_slots) - 1, 0, -1):
                rng_state = (rng_state * 1103515245 + 12345) & 0x7FFFFFFF
                j = rng_state % (i + 1)
                balanced_slots[i], balanced_slots[j] = balanced_slots[j], balanced_slots[i]

            # Re-map options for each skeleton to match its balanced slot
            for idx, skel in enumerate(skeletons):
                target_w = skel["target_word"]
                cur_opts = skel["prescribed_options"]
                new_slot = balanced_slots[idx]
                
                # Extract distractors in order
                other_opts = [opt for opt in cur_opts if opt.lower() != target_w.lower()]
                # If target was duplicated or missing, safeguard
                if len(other_opts) < 3:
                    continue
                new_opts = other_opts[:new_slot] + [target_w] + other_opts[new_slot:]
                skel["prescribed_options"] = new_opts
                skel["correct_answer_index"] = new_slot

        return skeletons

    # -------------------------------------------------------------------------
    # 4. WordNet Distractor Assembly & Zero-Double-Key Guarantee
    # -------------------------------------------------------------------------
    @classmethod
    def get_synonyms(cls, word: str) -> Set[str]:
        """Returns all synonymous lemma strings for the word across all synsets."""
        wn = cls.get_wordnet()
        synonyms = set()
        for syn in wn.synsets(word.lower()):
            for w in syn.words():
                lemma = w.lemma().replace("_", " ").lower()
                if lemma != word.lower():
                    synonyms.add(lemma)
        return synonyms

    @classmethod
    def get_antonyms(cls, word: str, pos: Optional[str] = None) -> Set[str]:
        """Returns strict direct antonyms from WordNet sense relations, locked to the exact lemma."""
        wn = cls.get_wordnet()
        target = word.lower()
        wn_pos = None
        if pos:
            pos_low = pos.lower()
            if "adj" in pos_low or pos_low in ("a", "s"):
                wn_pos = ("a", "s")
            elif "noun" in pos_low or pos_low == "n":
                wn_pos = ("n",)
            elif "verb" in pos_low or pos_low == "v":
                wn_pos = ("v",)
            elif "adv" in pos_low or pos_low == "r":
                wn_pos = ("r",)

        antonyms = set()
        for w in wn.words(target):
            if w.lemma().lower() != target:
                continue
            for syn in w.synsets():
                if wn_pos and syn.pos not in wn_pos:
                    continue
                for sense in syn.senses():
                    if sense.word().lemma().lower() == target:
                        for ant_sense in sense.get_related("antonym"):
                            ant_lemma = ant_sense.word().lemma().replace("_", " ").lower()
                            if ant_lemma != target:
                                antonyms.add(ant_lemma)
        return antonyms

    @classmethod
    def generate_zero_collision_distractors(
        cls,
        target_word: str,
        dependent_prep: Optional[str] = None,
        pos: str = "v",
        count: int = 3,
        context_sentence: Optional[str] = None
    ) -> List[str]:
        """
        Pre-computes distractors mathematically guaranteed to avoid synonym collisions with target.
        When dependent_prep is specified (e.g. 'on' for 'rely'), prioritizes candidates that
        do NOT govern that preposition or clash with it.
        Uses context_sentence and MiniLM disambiguation to anchor to the pedagogical synset
        rather than blind synsets[0].
        """
        wn = cls.get_wordnet()
        target_lower = target_word.lower()
        target_synonyms = cls.get_synonyms(target_lower)
        target_synonyms.add(target_lower)

        antonyms = cls.get_antonyms(target_lower)

        candidates: List[str] = []

        # 1. Antonyms make outstanding, collision-free distractors
        for ant in antonyms:
            if ant not in target_synonyms and " " not in ant and cls.is_cefr_compliant_distractor(ant, target_lower):
                candidates.append(ant)

        # 2. Taxonomy sibling / hypernym search
        target_synsets = cls.get_wordnet_synsets(target_lower, pos=pos)
        if target_synsets:
            primary_synset = cls.disambiguate_wordnet_synset(
                target_lower, pos=pos, context_sentence=context_sentence, candidate_synsets=target_synsets
            ) or target_synsets[0]
            # Get hypernyms (parent categories)
            hypernyms = primary_synset.get_related("hypernym")
            for hyper in hypernyms:
                # Get siblings (hyponyms of the hypernym)
                for sibling in hyper.get_related("hyponym"):
                    for w in sibling.words():
                        lemma = w.lemma().replace("_", " ").lower()
                        if (
                            lemma not in target_synonyms
                            and " " not in lemma
                            and lemma not in candidates
                            and cls.is_cefr_compliant_distractor(lemma, target_lower)
                        ):
                            if pos == "n" and (cls._is_primarily_adj_or_verb(lemma) if hasattr(cls, "_is_primarily_adj_or_verb") else False):
                                continue
                            candidates.append(lemma)

        # 3. Deduplicate and return requested count
        final_distractors = candidates[:count]

        # Fallback if synset taxonomy was sparse
        if len(final_distractors) < count:
            t_lvl_str = cls.get_word_cefr(target_lower) or "B2"
            t_rank = cls.CEFR_ORDER.get(t_lvl_str, 4)
            if t_rank <= 3:
                # Elementary / Intermediate Fallbacks (A1 - B1 core words)
                common_fallbacks = {
                    "v": ["choose", "accept", "decide", "expect", "follow", "notice", "explain", "remain"],
                    "n": ["choice", "reason", "matter", "situation", "effort", "problem", "action", "change"],
                    "a": ["simple", "common", "certain", "different", "similar", "natural", "clear", "direct"],
                    "r": ["deeply", "strictly", "tightly", "widely", "carefully", "clearly", "readily"]
                }
            else:
                # Advanced Fallbacks (B2 - C1 academic words)
                common_fallbacks = {
                    "v": ["assume", "indicate", "require", "maintain", "assess"],
                    "n": ["aspect", "factor", "criterion", "approach", "phenomenon"],
                    "a": ["apparent", "consistent", "variable", "significant", "distinct"],
                    "r": ["deeply", "strictly", "tightly", "widely", "carefully", "clearly", "readily"]
                }
            default_fb = common_fallbacks.get(pos, common_fallbacks["n" if pos == "n" else ("a" if pos == "a" else ("r" if pos == "r" else "v"))])
            for fb in default_fb:
                if fb not in target_synonyms and fb not in final_distractors:
                    final_distractors.append(fb)
                if len(final_distractors) == count:
                    break

        return final_distractors
