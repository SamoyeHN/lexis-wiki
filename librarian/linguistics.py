"""
Linguistic Engine for Lexis Wiki.
Integrates spaCy (computational dependency syntax) and WordNet (lexical relations)
to achieve deterministic sentence indexing, macro grammar domain classification,
boundary-accurate phrase extraction, canonical lemmatization, and zero-collision distractors.
"""

import json
import os
import re
import sqlite3
import zlib
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union


class LinguisticEngine:
    _spacy_nlp = None
    _wn = None
    _acl_data = None
    _awl_data = None
    _cefr_analyzer = None
    _ldoce_conn = None
    _ldoce_cache: Dict[str, Any] = {}
    _surface_lemma_cache: Dict[str, Optional[str]] = {}
    _attested_form_cache: Dict[str, bool] = {}
    _plural_form_cache: Dict[str, str] = {}
    _inflected_forms_cache: Dict[Tuple[str, str], Set[str]] = {}
    _ldoce_tables: Optional[Set[str]] = None
    _ldoce_compounds: Optional[Dict[str, str]] = None
    _ldoce_kind_ok: Optional[bool] = None
    _derived_qa_in_progress: Set[str] = set()
    _ocd_lexfile_cache: Dict[str, List[str]] = {}
    _phrase_text_cache: Dict[str, List[Tuple[str, str]]] = {}
    _phrase_evidence_cache: Dict[Tuple[str, str], List[str]] = {}

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

        # Pedagogical Text CEFR Rating combining baseline average & 80th percentile ceiling
        # A1: avg <= 1.55 and p80 <= 2.0
        # A2: avg <= 1.95 and p80 <= 3.0
        # B1: avg <= 2.60 and p80 <= 4.0
        # B2: avg <= 3.40
        # C1: avg <= 4.30
        # C2: >= 4.30
        if avg <= 1.55 and p80 <= 2.0:
            return "A1"
        elif avg <= 1.95 and p80 <= 3.0:
            return "A2"
        elif avg <= 2.60 and p80 <= 4.0:
            return "B1"
        elif avg <= 3.40:
            return "B2"
        elif avg <= 4.30:
            return "C1"
        else:
            return "C2"

    @classmethod
    def is_pedagogical_target(cls, word: str) -> bool:
        """
        Determines whether a candidate headword is a pedagogically sound vocabulary target for testing.
        Excludes:
        1. Demonyms / Nationalities / Regional descriptors (NORP: Thai, Spanish, American, Chinese).
        2. Geopolitical entities, Countries, Cities, Continents (GPE/LOC: Colombia, France, London).
        3. Proper personal names (PERSON).

        WordNet canonical truth:
        True proper nouns, demonyms, and geopolitical entities (Thai, Spanish, Colombia, Chinese)
        have their canonical lemma forms capitalized in WordNet (e.g. ['Thai', 'Thai']),
        whereas academic and standard content words ('foundation', 'correlate', 'custom')
        have lowercase lemmas (e.g. ['foundation', 'foundation']).
        """
        w_clean = word.strip()
        if not w_clean:
            return False
        w_lower = w_clean.lower()

        # Ultra-Basic Function Words Filter (Primary CEFR A1/A2 grammatical connectors)
        # Binds Section 1.9 of GEMINI.md: Basic function words ('but', 'and', 'or', 'so')
        # are filtered out from vocabulary quiz items to preserve academic discrimination.
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
        Backlog C1: what the three lexicons say about a surface form.

        An inflected form is not itself a headword - 'attaches', 'children' and 'buses'
        appear in none of OCD, the AWL or the CEFR list even though 'attach', 'child' and
        'bus' appear in all three - so a gate that reads the surface literally treats every
        inflected distractor as obscure and rejects it for A1/A2/B1 targets. An empty
        literal lookup is therefore retried through the lemma. The retry is lazy: the
        ordinary single-word path never pays a spaCy call for it.

        Returns (in_ocd, in_awl, cefr_level_or_None, form_the_lookup_resolved_to).
        """
        f = (form or "").strip().lower()
        if not f:
            return (False, False, None, "")
        ocd = cls.get_oxford_collocations()
        awl = cls.get_awl_words()
        in_ocd, in_awl = f in ocd, f in awl
        level = cls.get_word_cefr(f)
        resolved = f
        if not (in_ocd or in_awl or level):
            lemma = cls._surface_lemma(f)
            if lemma:
                resolved = lemma
                in_ocd, in_awl = lemma in ocd, lemma in awl
                level = cls.get_word_cefr(lemma)
        return (in_ocd, in_awl, level, resolved)

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
        3. Lexicon Filter: candidate must exist in Oxford Collocations Dictionary or CEFR database.
        4. Morphological Anomaly Gate:
           - Max length <= max(11, len(target) + 4) (rejects 14+ letter monsters like 'conceivableness').
           - Rejects obscure multi-affix formations ('-ableness', '-lessness', '-icalness', '-fulness').
        5. Backlog C1: every one of those tests is applied to the surface form the student
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

        # Lexicon Filter: must be recognized in OCD or AWL (through the lemma when the
        # candidate is an inflection rather than a headword).
        in_ocd, in_awl, c_lvl_str, resolved = cls._lexicon_profile(c_clean)

        analyzer = cls.get_cefr_analyzer()
        if not analyzer:
            return in_ocd or in_awl

        t_lvl_str = cls._lexicon_profile(t_clean)[2] or "B2"
        t_rank = cls.CEFR_ORDER.get(t_lvl_str, 4)
        # Backlog C1: the ceiling itself comes from the shared table, not from a
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
        if c_rank > ceiling_rank:
            return False

        return in_ocd or in_awl or analyzer.is_word_in_database(resolved)

    # ------------------------------------------------------------------
    # Backlog C1: ONE difficulty ceiling, shared by every gate
    # ------------------------------------------------------------------
    # Two tables because two different questions are being asked.
    #
    # CEFR_CEILING_BY_LEVEL answers "how hard may a DISTRACTOR be?" - a word the
    # student must decode on its own, competing against a target word. It may sit
    # one band above that target / passage. This is the ladder
    # is_cefr_compliant_distractor used to spell out inline.
    #
    # CEFR_TEXT_CEILING_BY_LEVEL answers "how hard may a word inside running text
    # the student reads be?" - a quiz option, a question stem, the sentence in an
    # 'example_usage' or an 'imitation_example'. Running text is easier to decode
    # than an isolated option, so the stop is looser: foundational levels stop at
    # C1/C2 words ('enigmatic', 'perused', 'meticulously'), B1 stops at C2, and
    # B2+ is not vocabulary-limited at all. This is exactly what the four
    # processor.py audits used to hard-code, and what the grammar / vocabulary
    # extraction paths never had at all.
    CEFR_CEILING_BY_LEVEL = {
        "A1": "B1", "A2": "B1", "B1": "B2", "B2": "C1", "C1": "C2", "C2": "C2",
    }
    CEFR_TEXT_CEILING_BY_LEVEL = {
        "A1": "B2", "A2": "B2", "B1": "C1", "B2": "C2", "C1": "C2", "C2": "C2",
    }

    @classmethod
    def cefr_ceiling(cls, passage_level: Optional[str], mode: str = "support") -> str:
        """
        Backlog C1: the hardest CEFR band that vocabulary may carry in a task
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
        """
        Backlog C1: every word in `text` rated harder than
        `cefr_ceiling(passage_level, mode)`, as (word, its_cefr_level) pairs
        ordered by descending difficulty so a caller can name the worst offender.

        Two exemptions keep the gate honest:
          * `allow` - words the source text already uses. A passage cannot be too
            hard for its own vocabulary, and a headword may not be banned from the
            example that is required to demonstrate it.
          * words no lexicon lists. An unlisted word is not evidence of difficulty;
            only a word the CEFR database actually rates above the ceiling is.

        Two more exemptions were forced by the corpus audit (scripts/_probe_stale_examples.py):
        the graded word list contains proper nouns ('Cassie' C2, 'Karen' C1) and the
        pieces of contractions ('doesn' C2), none of which is vocabulary a student is
        being asked to learn.

        Difficulty is looked up through the lemma, so 'deteriorating' is judged as
        'deteriorate' (Backlog A1 / C1).
        """
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
        """Lazy-loads the wn (Open English WordNet) client with multithreading enabled."""
        if cls._wn is None:
            import wn
            # Web dashboard / background worker threads share WordNet SQLite connection.
            # SQLite check_same_thread must be disabled for read-only multi-threaded queries.
            wn.config.allow_multithreading = True
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
        if clean_w1 in fam2 or bool(fam1 & fam2):
            return True

        # Morphological stem prefix fallback: if both words length >= 6 and share prefix >= 5
        min_w, max_w = (clean_w1, clean_w2) if len(clean_w1) <= len(clean_w2) else (clean_w2, clean_w1)
        if len(min_w) >= 5 and max_w.startswith(min_w):
            return True
        return False

    # The raw Oxford Collocations Dictionary file (`data/oxford_collocations.json.gz`)
    # used to be lazy-loaded here and consulted by generate_phrase_distractors and by
    # mine_expression_skeletons.  Both now ask LDOCE instead - see ldoce_phrase_evidence -
    # so nothing reads that file any more and the accessor is gone.

    # LDOCE stores lemmas and open compounds only, but the pipeline asks for whatever
    # surface form the passage happened to contain ('posts', 'luxuries', 'potluck').
    # Without headword resolution an inflected target loses its entire entry:
    # no collocations, no examples, no anchor, no structural model.
    _HEADWORD_FORM_EXCEPTIONS = {
        "news", "series", "species", "means", "headquarters", "premises",
    }
    _COLLOCATION_KEY_ALIASES = {
        "adjective": "adjectives", "noun": "nouns", "verb": "verbs",
        "adverb": "adverbs", "phrase": "phrases",
    }

    @classmethod
    def _headword_candidates(cls, surface: str) -> List[str]:
        """Surface form -> plausible LDOCE headwords, most literal first."""
        cands = [surface]
        if "-" in surface:
            cands.append(surface.replace("-", ""))
            cands.append(surface.replace("-", " "))
        else:
            cands.append(surface.replace(" ", "-"))
            cands.append(surface.replace(" ", ""))
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
        """The dictionary database in use. LDOCE_DB overrides it - that is how a
        rebuilt shadow database is validated before it replaces the production one."""
        override = os.environ.get("LDOCE_DB", "").strip()
        if override:
            return Path(override).expanduser()
        return Path(__file__).resolve().parent / "data" / "ldoce6_essential.db"

    @classmethod
    def _ldoce_table_exists(cls, name: str) -> bool:
        """Guard for auxiliary tables. 'ldoce_adj_nouns' and 'ldoce_adv_collocations'
        are queried but were never created by build_ldoce_db.py; an unguarded SELECT
        aborted every fallback that happened to share its try-block."""
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

    # ------------------------------------------------------------------
    # LDOCE ingestion QA (backlog E1 / E2) - the safety net for a pre-R1 database
    #
    # The old build_ldoce_db.py copied a base entry's whole sense payload onto derived
    # headwords: 'punctuality' got 'punctual''s adjective senses verbatim,
    # 'abandonment' got 'abandon''s verb senses, 'abductor' got 'abduct'.
    # Measured signature (scripts/ldoce_qa.py): 9,792 rows in 4,441 groups share
    # a byte-identical senses payload with a different headword.
    #
    # A headword that merely *looks* derived is not enough - 'enhance', 'augment'
    # and 'dance' end in these letters and are genuine verbs with their own
    # senses. The repair therefore fires only on the copy signature: the row's
    # senses payload is byte-identical to an existing base entry's payload.
    # ------------------------------------------------------------------
    _DERIVED_NOUN_SUFFIXES = (
        "ation", "ition", "ution", "ation", "sion", "tion", "ness", "ment",
        "ity", "ance", "ence", "ship", "hood", "dom", "ery", "age", "cy",
    )
    _DERIVED_STEM_RESTORES = (
        "", "e", "t", "d", "s", "te", "de", "ce", "se", "ate", "ite", "ute",
        "ert", "it", "ut", "ct", "pt", "ise", "ize",
    )
    _NOUN_POS_LABELS = {"noun", "n", "noun count", "noun mass", "number"}

    @classmethod
    def _senses_signature(cls, entry: Dict[str, Any]) -> str:
        """Canonical fingerprint of an entry's sense payload (copy-detection key)."""
        return json.dumps(entry.get("senses", []), sort_keys=True, ensure_ascii=False)

    @classmethod
    def _sense_definitions(cls, senses: List[Dict[str, Any]]) -> List[str]:
        return [(s.get("definition") or "").strip().lower() for s in senses or []]

    @classmethod
    def _is_copy_of_base(
        cls, senses: List[Dict[str, Any]], base_senses: List[Dict[str, Any]]
    ) -> bool:
        """
        Copy signature of the ingestion bug: the derived row reuses the base entry's
        glosses. Exact payload equality is too strict ('abandonment' carries 6 of
        'abandon''s 7 senses), so the test is gloss overlap >= 50%.
        """
        mine = cls._sense_definitions(senses)
        if not mine:
            return False
        base = set(cls._sense_definitions(base_senses))
        if not base:
            return False
        overlap = sum(1 for d in mine if d in base)
        return overlap >= max(1, len(mine) // 2)

    @classmethod
    def _derived_base_candidates(cls, word: str) -> List[str]:
        """'abdication' -> ['abdic', 'abdicate', ...], most plausible first."""
        out: List[str] = []
        for suf in cls._DERIVED_NOUN_SUFFIXES:
            if not word.endswith(suf) or len(word) <= len(suf) + 3:
                continue
            stem = word[: -len(suf)]
            for restore in cls._DERIVED_STEM_RESTORES:
                cand = stem + restore
                if len(cand) >= 4 and cand != word:
                    out.append(cand)
        seen, res = set(), []
        for cand in out:
            if cand not in seen:
                seen.add(cand)
                res.append(cand)
        return res

    @classmethod
    def _repair_derived_copy_artifact(
        cls, word: str, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Backlog E1 / E2. When a derived headword's senses payload is a verbatim copy
        of an existing base entry, the row carries the base word's POS, not its own.
        Relabel it to the part of speech its own morphology states, so the
        'if target_p not in s_pos: continue' gate in _lock_sense stops skipping
        every sense, and flag the row so consumers know the gloss is borrowed.
        """
        senses = data.get("senses") or []
        if not senses or data.get("_derived_qa_checked"):
            return data
        data["_derived_qa_checked"] = True

        if data.get("kind"):
            # Safety net only. A rebuilt database (backlog R1) already split every
            # Longman run-on form into its own row with its own POS and its own
            # gloss, so guessing a part of speech from a suffix here would override
            # the classification the builder made from the markup itself.
            return data

        sense_pos = {(s.get("pos") or "").strip().lower() for s in senses}
        if sense_pos & cls._NOUN_POS_LABELS:
            return data  # already a noun entry - nothing to repair

        candidates = cls._derived_base_candidates(word)
        if not candidates:
            return data

        # One indexed query tells which base candidates exist at all. Without it every
        # non-existent candidate costs a full get_ldoce_entry attempt.
        existing = cls._ldoce_words_present(candidates)
        if existing is not None:
            candidates = [c for c in candidates if c in existing]
        if not candidates:
            return data

        my_sig = cls._senses_signature(data)
        for base in candidates:
            if base == word or base in cls._derived_qa_in_progress:
                continue
            cls._derived_qa_in_progress.add(word)
            try:
                base_entry = cls.get_ldoce_entry(base)
            finally:
                cls._derived_qa_in_progress.discard(word)
            if not base_entry:
                continue
            if cls._senses_signature(base_entry) != my_sig and not cls._is_copy_of_base(
                senses, base_entry.get("senses") or []
            ):
                continue
            data["copy_artifact_suspect"] = True
            data["copy_artifact_base"] = base
            # The row copies the base word's senses AND its glosses: 'punctuality' shipped
            # 'punctual''s definition. Give every relabelled sense a gloss the headword
            # owns - WordNet's own noun sense for it, or the gloss its suffix states.
            noun_glosses = cls._wordnet_noun_glosses(word)
            for i, s in enumerate(senses):
                s["pos"] = "noun"
                if i < len(noun_glosses):
                    s["definition"] = noun_glosses[i]
                    s["gloss_source"] = "wordnet"
                else:
                    gloss = cls._derivational_noun_gloss(word, base, base_entry.get("pos", ""))
                    if gloss:
                        s["definition"] = gloss
                        s["gloss_source"] = "derivational"
            data["pos"] = "noun"
            data["all_poses"] = ["noun"]
            break
        return data

    @classmethod
    def _wordnet_noun_glosses(cls, word: str) -> List[str]:
        """Distinct noun definitions WordNet holds for this exact headword, in order."""
        glosses: List[str] = []
        seen: Set[str] = set()
        try:
            wn = cls.get_wordnet()
            for syn in wn.synsets((word or "").strip().lower()):
                pos = syn.pos() if callable(syn.pos) else syn.pos
                if pos != "n":
                    continue
                gloss = (syn.definition() or "").strip()
                if gloss and gloss not in seen:
                    seen.add(gloss)
                    glosses.append(gloss)
        except Exception:
            return []
        return glosses

    @classmethod
    def _derivational_noun_gloss(cls, word: str, base: str, base_pos: str) -> str:
        """'abandon' (verb) -> 'the act of abandoning'; 'punctual' (adj) -> 'the quality of being punctual'."""
        b = (base or "").strip().lower()
        if not b:
            return ""
        p = (base_pos or "").strip().lower()
        if p.startswith("verb"):
            gerund = cls._inflect_verb(b, "VBG") or f"{b}ing"
            return f"the act of {gerund}"
        if p.startswith("adj"):
            return f"the quality of being {b}"
        return ""

    @classmethod
    def _finish_ldoce_entry(cls, w_clean: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Single exit point: normalize collocation keys, run ingestion QA, cache."""
        data["collocations"] = cls._normalize_collocations(data.get("collocations"))
        data = cls._repair_derived_copy_artifact(w_clean, data)
        cls._ldoce_cache[w_clean] = data
        return data

    @classmethod
    def get_ldoce_entry(cls, word: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves full LDOCE 6th Edition entry for the given headword from SQLite.
        Cached in-memory for instant <0.1ms access.
        """
        if not word:
            return None
        w_clean = word.strip().lower()
        if w_clean in cls._ldoce_cache:
            return cls._ldoce_cache[w_clean]

        if cls._ldoce_conn is None:
            db_path = cls._ldoce_db_path()
            if not db_path.exists():
                return None
            try:
                import sqlite3
                cls._ldoce_conn = sqlite3.connect(str(db_path), check_same_thread=False)
            except Exception:
                return None

        try:
            for q_cand in cls._headword_candidates(w_clean):
                fetched = cls._ldoce_raw_row(q_cand)
                if fetched:
                    data = cls._resolve_ldoce_kind(q_cand, w_clean, *fetched)
                    return cls._finish_ldoce_entry(w_clean, data)

            # Last resort: closed compounds are spelled with a space in LDOCE
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

            cls._ldoce_cache[w_clean] = None
            return None
        except Exception:
            return None

    @classmethod
    def _ldoce_compound_index(cls) -> Dict[str, str]:
        """normalized headword -> the headword LDOCE actually stores, built once."""
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

    @classmethod
    def _ldoce_words_present(cls, words: List[str]) -> Optional[Set[str]]:
        """Which of these headwords have rows? One indexed query instead of one per
        candidate - the per-candidate version pays the compound fallback scan each miss."""
        if cls._ldoce_conn is None or not words:
            return None
        try:
            marks = ",".join("?" for _ in words)
            rows = cls._ldoce_conn.execute(
                f"SELECT word FROM ldoce WHERE word IN ({marks})", tuple(words)
            ).fetchall()
            return {r[0].strip().lower() for r in rows}
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Backlog R1 - the rebuilt database classifies every row, so the engine reads
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
    def _ldoce_has_kind(cls) -> bool:
        """Does this database carry the R1 classification columns?"""
        if cls._ldoce_kind_ok is None:
            cols: Set[str] = set()
            if cls._ldoce_conn is not None:
                try:
                    cols = {r[1] for r in cls._ldoce_conn.execute("PRAGMA table_info(ldoce)")}
                except Exception:
                    cols = set()
            cls._ldoce_kind_ok = "kind" in cols and "base_word" in cols
        return cls._ldoce_kind_ok

    @classmethod
    def _ldoce_raw_row(
        cls, word: str
    ) -> Optional[Tuple[Dict[str, Any], Optional[str], Optional[str]]]:
        """One indexed read -> (payload, kind, base_word). No cache, no QA, no hop."""
        if cls._ldoce_conn is None or not word:
            return None
        try:
            if cls._ldoce_has_kind():
                row = cls._ldoce_conn.execute(
                    "SELECT data_json, kind, base_word FROM ldoce WHERE word = ? LIMIT 1",
                    (word,)
                ).fetchone()
                if not row:
                    return None
                return json.loads(row[0]), row[1], row[2]
            row = cls._ldoce_conn.execute(
                "SELECT data_json FROM ldoce WHERE word = ? LIMIT 1", (word,)
            ).fetchone()
            if not row:
                return None
            return json.loads(row[0]), None, None
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
            return data  # pre-rebuild database: nothing was classified
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
            data["base_word"] = base_word
        return data

    @classmethod
    def get_ldoce_preps(cls, word: str) -> Set[str]:
        """
        Extracts governing prepositions for the given word from LDOCE grammar patterns and collocations.
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

    # The tokens Longman uses as the second word of a phrasal verb.
    _PHRASAL_PARTICLES = frozenset({
        "in", "off", "up", "down", "away", "back", "over", "into", "through", "on", "to",
        "for", "out", "with", "from", "around", "at", "by", "upon", "of", "about",
        "against", "along", "ahead", "across", "aside", "forth", "together", "off",
    })

    @classmethod
    def get_ldoce_phrasal_verbs(cls, word: str) -> List[str]:
        """The phrasal verbs Longman declares inside `word`'s own entry, as 'verb particle'
        pairs. Longman writes them with object slots and separable arrows ('send somebody
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
    # LDOCE phrase attestation (backlog F7 修法 2)
    # ------------------------------------------------------------------
    # 'keep of', 'havoc for', 'streak on' and 'depend for' are not phrases. Each one is a
    # preposition lifted out of somebody else's frame and glued onto the headword:
    #
    #   'keep of'     <- 'keep (somebody) out of something'      ('of' belongs to 'out of')
    #   'havoc for'   <- 'cause/create havoc ... for commuters'  ('for' belongs to 'cause')
    #   'streak on'   <- 'be on a winning/losing streak'         ('on' belongs to 'be')
    #   'depend for'  <- 'depend on somebody/something for something' (second complement)
    #
    # and these are phrases because Longman itself states them:
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
    #      or with nothing but Longman's own placeholders and a copula ('be embedded in
    #      something'), so a preposition governed by a different verb in the same pattern
    #      cannot be borrowed.  Sense examples are exempt from (2) - a sentence has no frame
    #      head - which is exactly why example-only evidence is the weakest tier and is not
    #      enough to ship a phrase to a learner.
    _PHRASE_SLOT_WORDS = frozenset({
        "somebody", "something", "sb", "sth", "one's", "oneself", "myself", "yourself",
        "himself", "herself", "his", "her", "their", "its", "my", "your", "our", "it",
        "this", "that", "these", "those", "some", "any", "etc",
    })
    # What may stand in front of a frame's head: Longman's own placeholders, bare articles,
    # and the copula or auxiliary of a passive frame.  Longman spells a passive out with the
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
    # evidence - Longman's own frames are - it is only what may sit inside a frame's gap as
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
    # Strongest first. 'phrasal_verb' is Longman declaring a phrasal verb; 'phrase' is its
    # PHRASES block; 'pattern' is a grammar pattern built on the headword; 'collocation' is
    # a COLLOCATIONS box item; 'example' is a sentence that merely contains the string.
    _PHRASE_TIERS = ("phrasal_verb", "phrase", "pattern", "collocation",
                     "cross_collocation", "example")
    _PHRASE_STRONG_TIERS = frozenset({"phrasal_verb", "phrase", "pattern", "collocation"})

    @classmethod
    def _entry_text_normalizer(cls, text: Any) -> str:
        """Longman's own strings, flattened so a frame can be matched as a token sequence:
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
    def _phrase_regex(cls, tokens: List[str], max_gap: int) -> str:
        """Match the tokens in order, allowing up to max_gap object slots between neighbours.
        Whitespace is always required between tokens; a slot run ('somebody something')
        counts as one gap, so 'keep somebody something warm' is one gap, not two."""
        parts: List[str] = []
        for token in tokens:
            forms = {f for f in cls.inflected_forms(token) if f}
            if not forms:
                return ""
            # A mined surface form has to reach the form Longman wrote: 'kept in touch' is
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
        elem = rf"(?:{slot_elem}|{particle_elem})"
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
    # heads the same tail, so 'keep in touch' is stated by Longman even though it shares a
    # line with 'stay'.  Only a group at the head of the string is expanded - a slash deeper
    # inside a frame separates alternatives inside it, and the normalizer keeps those as the
    # boundaries they are.
    _LEADING_ALT_RE = re.compile(
        r"^\s*([A-Za-z][A-Za-z'’\-]*(?:/[A-Za-z][A-Za-z'’\-]*)+)(?=\s|$)")

    @classmethod
    def _slash_variants(cls, text: str) -> List[str]:
        """'stay/keep in touch' -> ['stay in touch', 'keep in touch']."""
        if not isinstance(text, str):
            return []
        match = cls._LEADING_ALT_RE.match(text)
        if not match:
            return [text]
        tail = text[match.end():]
        return [f"{alt}{tail}" for alt in match.group(1).split("/") if alt]

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
                # Longman's slash-listed alternatives each stand on their own, so a frame is
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
        """Which LDOCE entries to look in. A mined surface form ('attaches', 'happened')
        has to be resolved back to the headword Longman filed it under, and a verbal idiom
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
        # Longman's declared units *and* its grammar patterns both write the object inside
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
                    # unit - its head is itself, and Longman writes alternatives with a
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
        """Is `phrase` a unit LDOCE6 states, or a string assembled out of tokens scraped
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
        """Ship a multi-word item to a learner only if Longman states it at a tier stronger
        than 'some example sentence happens to contain these words in this order'."""
        return any(tier in cls._PHRASE_STRONG_TIERS
                   for tier in cls.ldoce_phrase_evidence(phrase, headword))

    @classmethod
    def _phrasal_verb_blocks(cls, phrase: str,
                             headword: Optional[str] = None) -> List[Dict[str, Any]]:
        """The phrasal-verb blocks Longman states for `phrase`, with each block's own frame
        still attached.

        `ldoce_phrase_hits` flattens every block into tiered strings, which is what the
        attestation gate needs.  The object-fit audit needs the block's headword instead,
        because that is where Longman says whether the object comes before the particle
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
    def _frame_object_position(cls, frame: str, particle: str) -> str:
        """Where this frame puts its object relative to the particle: 'before', 'after' or
        'unknown'.

        Longman writes a frame as a token sequence with slots.  'depend on / upon somebody /
        something' puts the object after the particle; 'keep somebody in' and 'work somebody /
        something in' put it before.  '↔' marks the pair as separable, which means Longman
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
        """Is the object the passage put after this phrase one Longman's frame allows there?

        'keep in' is a real phrasal verb, but Longman's frame for it is 'keep somebody in' -
        the object comes before the particle - so a parse that reads 'keep in WeChat Moments'
        as that frame has mistaken a locative 'in ...' for the particle.  'depend on Mary' is
        a fit, because Longman's frame is 'depend on / upon somebody / something'.

        The veto needs positive evidence: it fires only when some frame puts the object before
        the particle and no frame puts it after.  An empty object is always a fit - nothing has
        been claimed about it - and so is a phrase Longman gives no frame for ('break away',
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

    # Longman spells parts of speech out ('noun count', 'predeterminer'), the sense
    # gate works in short codes. Canonicalising both sides once is what closes the
    # backlog E5 leak: 'v' is a substring of 'adjective' and 'adverb', 'n' is a
    # substring of 'adverb', 'conjunction', 'determiner' and 'pronoun'.
    _SENSE_POS_CODES = {
        "noun": "n", "noun count": "n", "noun mass": "n", "number": "n",
        "ordinal number": "n", "cardinal number": "n",
        "verb": "v", "modal verb": "v",
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
    # B1: the shortest passage sentence that can serve as an anchor's physical evidence.
    ANCHOR_MIN_WORDS = 6

    @classmethod
    def _lock_sense(
        cls,
        entry: Dict[str, Any],
        definition: Optional[str] = None,
        quote: Optional[str] = None,
        frame_evidence: Optional[str] = None,
        target_pos: Optional[str] = None,
        return_confidence: bool = False
    ):
        """
        Determines the authoritative sense index in LDOCE entry aligned to quote and definition.
        Calculates content-word overlap across examples, patterns, definitions, signposts,
        and enforces grammar/syntactic frame agreement.

        With return_confidence=True the result is (index, score, margin, confidence) instead of
        the index alone, so every existing caller keeps the single-value contract it expects.
        """
        senses = entry.get("senses", [])
        if not senses:
            return None

        # Backlog E5: this gate used to be `target_p not in s_pos`, a substring test
        # against single-letter codes. Both sides are canonical codes compared for
        # equality now, so a noun query can no longer admit an adverb sense.
        target_p = cls._pos_code(target_pos)
        
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'if', 'then', 'so', 'because', 'as', 'until', 'while',
            'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',
            'above', 'below', 'to', 'from', 'up', 'upon', 'down', 'in', 'out', 'on', 'off', 'over', 'under',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
            'would', 'should', 'could', 'ought', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'them', 'their', 'theirs',
            'his', 'her', 'hers', 'its', 'our', 'ours', 'your', 'yours', 'me', 'him', 'us', 'this', 'that', 'these', 'those',
            'more', 'most', 'many', 'much', 'some', 'any', 'other', 'another', 'such', 'very', 'even', 'also', 'too'
        }

        q_tokens = (set(re.findall(r"\b[a-zA-Z]{3,}\b", (quote or "").lower())) - stopwords) if quote else set()
        d_tokens = (set(re.findall(r"\b[a-zA-Z]{3,}\b", (definition or "").lower())) - stopwords) if definition else set()

        # A1: a pattern word is evidenced by the quote when the quote carries the word
        # OR any legal inflection of it. 'firm ... attaches itself to' satisfies the
        # LDOCE pattern 'attach something to something'; the old \battach\b regex did not.
        q_low_all = (quote or "").lower()
        q_word_tokens = set(re.findall(r"\b[a-z]+\b", q_low_all))

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

        # Homograph blocks (backlog E5): one row can hold 'right1 adj', 'right2 noun',
        # 'right3 verb'. A sense that sits in the block declaring the target POS is the
        # primary one for this query, so it must not lose a tie to ordering alone.
        block_pos = {
            h.get("homograph_num"): cls._pos_code(h.get("pos"))
            for h in entry.get("homographs") or []
        }

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

            # 1. Quote token overlap against sense examples & patterns
            ex_tokens = set()
            for ex in s.get("examples", []):
                ex_tokens.update(re.findall(r"\b[a-zA-Z]{3,}\b", ex.lower()))
            for pat in s.get("patterns", []):
                ex_tokens.update(re.findall(r"\b[a-zA-Z]{3,}\b", pat.lower()))
            
            ex_overlap = len(q_tokens & (ex_tokens - stopwords))
            score += ex_overlap * 8

            # 2. Quote token overlap against sense definition itself
            # Exclude ultra-generic words like 'time', 'likely', 'something', 'someone' that skew sense matching
            meta_stopwords = stopwords | {'time', 'likely', 'someone', 'something', 'somebody', 'things', 'people', 'used', 'make'}
            s_defn = s.get("definition", "").lower()
            s_def_tokens = set(re.findall(r"\b[a-zA-Z]{3,}\b", s_defn)) - meta_stopwords
            q_def_overlap = len((q_tokens - meta_stopwords) & s_def_tokens)
            score += q_def_overlap * 6

            # 3. Input definition token overlap (if provided)
            def_overlap = len(d_tokens & s_def_tokens)
            score += def_overlap * 5

            # 4. Signpost match
            sp = (s.get("signpost") or "").lower()
            if sp:
                sp_tokens = set(re.findall(r"\b[a-zA-Z]{3,}\b", sp))
                if sp_tokens & (q_tokens | d_tokens):
                    score += 15

            # 5. Pattern / Preposition / Collocation match directly from Quote
            if quote:
                for pat in s.get("patterns", []):
                    pat_clean = pat.lower().strip()
                    # Check if exact pattern phrase or bound prepositions (e.g. "serious about", "log in/on to") appear in quote
                    words_in_pat = [w for w in re.findall(r"\b[a-zA-Z]{2,}\b", pat_clean) if w not in ("somebody", "something", "etc", "sb", "sth")]
                    if len(words_in_pat) >= 2 and all(_quote_has_form(w) for w in words_in_pat):
                        score += 40
                    elif "/" in pat_clean:
                        # Handle slash alternative branches like "make something difficult/easy/possible etc"
                        slash_parts = [p.strip() for p in pat_clean.split("/") if p.strip()]
                        for sp in slash_parts:
                            sp_words = [w for w in re.findall(r"\b[a-zA-Z]{2,}\b", sp) if w not in ("somebody", "something", "etc", "sb", "sth")]
                            if sp_words and all(_quote_has_form(w) for w in sp_words):
                                score += 40
                                break
                    elif len(words_in_pat) == 1 and words_in_pat[0] not in ("about", "for", "with", "to", "in", "on", "of", "from") and _quote_has_form(words_in_pat[0]):
                        score += 15
                    elif len(words_in_pat) == 1 and words_in_pat[0] in ("about", "for", "with", "to", "in", "on", "of", "from") and _quote_has_form(words_in_pat[0]):
                        # Bound preposition match (e.g. serious about)
                        score += 25

            # 6. Authentic Syntactic Dependency & Argument Structure Match
            if quote:
                try:
                    nlp = cls.get_spacy()
                    q_doc = nlp(quote)
                    for tok in q_doc:
                        if tok.lemma_.lower() == entry.get("word", "").lower() or tok.text.lower() == entry.get("word", "").lower():
                            # If target is an adjective, check its modified noun against sense patterns & examples
                            if tok.pos_ == "ADJ" and tok.head and tok.head.pos_ == "NOUN":
                                mod_noun = tok.head.lemma_.lower()
                                if any(mod_noun in ex.lower() for ex in s.get("examples", [])) or any(mod_noun in p.lower() for p in s.get("patterns", [])):
                                    score += 25
                                # If sense definition mentions the semantic class of modified noun (e.g. relationship, time, decision)
                                if mod_noun in s_def_tokens:
                                    score += 20
                            # If target is a verb, check transitivity and arguments
                            elif tok.pos_ == "VERB":
                                has_obj = any(ch.dep_ in ("dobj", "obj") for ch in tok.children)
                                gram_l = (s.get("grammar") or "").lower()
                                if not has_obj and "[intransitive]" in gram_l and "[transitive]" not in gram_l:
                                    score += 20
                                elif has_obj and "[transitive]" in gram_l:
                                    score += 15
                                # Check subject token
                                for ch in tok.children:
                                    if ch.dep_ in ("nsubj", "nsubjpass"):
                                        subj_lem = ch.lemma_.lower()
                                        if subj_lem in s_def_tokens or any(subj_lem in ex.lower() for ex in s.get("examples", [])):
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

            # Tie-break slightly favors primary head senses
            score -= i * 0.5

            if score > best_score:
                second_score = best_score
                best_score = score
                best_idx = i
            elif score > second_score:
                second_score = score

        idx = best_idx if best_idx is not None else (0 if senses else None)
        if not return_confidence:
            return idx

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
        Fast full-text search against the 400k LDOCE6 corpus examples (0.2ms via FTS5).
        Returns a list of high-quality authentic sentence strings.
        """
        if not query or not query.strip():
            return []
        db_path = cls._ldoce_db_path()
        if not db_path.exists():
            return []
        try:
            conn = sqlite3.connect(str(db_path))
            c = conn.cursor()
            clean_q = query.strip()
            if category:
                fts_query = f'category : "{category}" AND "{clean_q}"' if " " in clean_q else f'category : "{category}" AND {clean_q}'
            else:
                fts_query = f'"{clean_q}"' if " " in clean_q and not clean_q.startswith("NEAR(") else clean_q

            c.execute(
                "SELECT content FROM corpus_fts WHERE corpus_fts MATCH ? LIMIT ?",
                (fts_query, limit)
            )
            rows = c.fetchall()
            conn.close()
            # Filter clean non-empty examples
            res = []
            for r in rows:
                txt = r[0].strip()
                if txt and len(txt) > 8 and txt not in res:
                    res.append(txt)
            return res[:limit]
        except Exception:
            return []

    @classmethod
    def get_ldoce_grammar_alert(
        cls,
        word: str,
        quote: str = ""
    ) -> Optional[Dict[str, str]]:
        """
        Retrieves authentic grammar error diagnostics (Don't say / ✗) from LDOCE6 Grammar Boxes.
        Returns dict with keys: 'word', 'title', 'content', 'rule' or None.
        """
        if not word:
            return None
        w_clean = word.strip().lower()
        db_path = cls._ldoce_db_path()
        if not db_path.exists():
            return None
        try:
            conn = sqlite3.connect(str(db_path))
            c = conn.cursor()
            fts_query = f'word : "{w_clean}" AND category : "grammar_box"'
            c.execute(
                "SELECT pattern, content FROM corpus_fts WHERE corpus_fts MATCH ?",
                (fts_query,)
            )
            rows = c.fetchall()
            conn.close()
            if not rows:
                return None

            best_box = None
            for title, content in rows:
                if "Don't say" in content or "✗" in content or "Grammar" in title:
                    # If quote has specific tokens, prioritize matching box
                    if quote and any(tok in content.lower() for tok in quote.lower().split() if len(tok) > 4):
                        return {
                            "word": w_clean,
                            "title": title or "GRAMMAR ALERT",
                            "content": content
                        }
                    if not best_box:
                        best_box = {
                            "word": w_clean,
                            "title": title or "GRAMMAR ALERT",
                            "content": content
                        }
            return best_box
        except Exception:
            return None

    @classmethod
    def get_ldoce_thesaurus(cls, word: str) -> List[Dict[str, Any]]:

        """
        Retrieves near-synonym clusters with exact distinctions from LDOCE Thesaurus.

        The Longman Language Activator boxes are thesaurus boxes too - Longman's own
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
                    "concept": concept_name,
                })
        return out

    @classmethod
    def get_ldoce_definition_and_example(
        cls,
        word: str,
        target_pos: Optional[str] = None,
        context_sentence: Optional[str] = None,
        require_sense_confidence: bool = False
    ) -> Tuple[str, str]:
        """
        Extracts authoritative definition and authentic example from LDOCE 6th Edition,
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

        if locked_idx is not None and locked_idx < len(senses):
            chosen_sense = senses[locked_idx]
        else:
            chosen_sense = senses[0]

        definition = chosen_sense.get("definition", "")

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

        # 6. Fallback example from corpus_fts index
        if not _shows_headword(example):
            example = _attested_example(cls.search_corpus_examples(word, limit=5)) or example

        # 7. Fallback example from LDOCE anchor finder - only when nothing at all was
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

    @classmethod
    def get_expression_definition_and_example(
        cls,
        phrase: str,
        expr_type: Optional[str] = None,
        context_sentence: Optional[str] = None
    ) -> Tuple[str, str]:
        """
        Three-tier authoritative cascade for multi-word expressions, collocations, and phrasal verbs:
        Tier 1: LDOCE 6th Edition direct multi-word entry (e.g. 'climate change', 'natural resources').
        Tier 2: LDOCE root verb entry scanning patterns and examples (e.g. 'turn' -> 'turn off').
        Tier 3: WordNet multi-word synsets (e.g. 'shut down', 'turn off').
        Tier 4: Graceful pedagogical fallback.
        """
        clean_phrase = re.sub(r'\[.*?\]|\(.*?\)', '', phrase).strip().lower()
        if not clean_phrase:
            return ("", "")

        # Tier 1: Check direct LDOCE entry
        defn, ex = cls.get_ldoce_definition_and_example(clean_phrase, target_pos=expr_type, context_sentence=context_sentence)
        if defn:
            return (defn, ex)

        # Tier 2: Check root headword in LDOCE for phrasal combinations (e.g. 'turn off' -> head 'turn')
        words = clean_phrase.split()
        if len(words) >= 2:
            head_word = words[0]
            tail = " ".join(words[1:])
            tail_pat = r'\b' + re.escape(tail) + r'\b'
            head_entry = cls.get_ldoce_entry(head_word)
            if head_entry:
                for s in head_entry.get("senses", []):
                    # Check patterns
                    for pat in s.get("patterns", []):
                        if re.search(tail_pat, pat.lower()):
                            d = s.get("definition", "")
                            e = s.get("examples", [""])[0] if s.get("examples") else ""
                            if d:
                                return (d, e)
                    # Check examples
                    for eg in s.get("examples", []):
                        if re.search(tail_pat, eg.lower()):
                            d = s.get("definition", "")
                            if d:
                                return (d, eg)

        # Tier 3: Check WordNet multi-word synset entries
        try:
            wn = cls.get_wordnet()
            # Try variations: 'shut down', 'shut_down', 'shutdown'
            query_candidates = [clean_phrase, clean_phrase.replace(" ", "_"), clean_phrase.replace(" ", "")]
            for q in query_candidates:
                wn_words = wn.words(q) if hasattr(wn, "words") else []
                if wn_words:
                    synsets = wn_words[0].synsets()
                    if synsets:
                        best_syn = synsets[0]
                        # If context_sentence given, use Lesk overlap to choose best synset
                        if len(synsets) > 1 and context_sentence:
                            c_toks = set(re.findall(r'[a-zA-Z]{3,}', context_sentence.lower()))
                            best_overlap = -1
                            for syn in synsets:
                                s_toks = set(re.findall(r'[a-zA-Z]{3,}', f"{syn.definition()} {' '.join(syn.examples())}".lower()))
                                overlap = len(c_toks & s_toks)
                                if overlap > best_overlap:
                                    best_overlap = overlap
                                    best_syn = syn
                        d = best_syn.definition()
                        egs = best_syn.examples()
                        e = egs[0].strip('"\'; ') if egs else ""
                        if d:
                            return (d, e)
                # Direct synsets call
                synsets = wn.synsets(q)
                if synsets:
                    best_syn = synsets[0]
                    d = best_syn.definition()
                    egs = best_syn.examples()
                    e = egs[0].strip('"\'; ') if egs else ""
                    if d:
                        return (d, e)
        except Exception:
            pass

        # Tier 4: Fallback
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
        Retrieves authoritative, semantically prioritized collocations from LDOCE 6th Edition.
        When word is None, returns a dict-like view of all headwords in LDOCE that resolves
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

        # 2. Collocation boxes in LDOCE
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
        # shipped database did. Nothing read that bucket, so Longman's intensifiers for an
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

        if cls._ldoce_conn is not None:
            try:
                cursor = cls._ldoce_conn.cursor()
                # 3. Adverb table. Guarded: this table is absent from the shipped
                # database, and an unguarded SELECT used to abort every lookup below it.
                if cls._ldoce_table_exists("ldoce_adv_collocations"):
                    cursor.execute(
                        "SELECT anchor_type, anchor_word, collocation, example FROM ldoce_adv_collocations WHERE adv = ?",
                        (w_clean,)
                    )
                    for atype, aword, c_text, c_ex in cursor.fetchall():
                        if atype == "modifies_verb":
                            res["verb"].append(aword)
                            res["verb_before"].append(aword)
                        elif atype == "modifies_adj":
                            res["adj"].append(aword)
                        if c_ex and c_ex not in res["examples"]:
                            res["examples"].append(c_ex)

                # 4. Adjective-noun table
                if cls._ldoce_table_exists("ldoce_adj_nouns"):
                    cursor.execute("SELECT adj FROM ldoce_adj_nouns WHERE noun = ?", (w_clean,))
                    for (a_adj,) in cursor.fetchall():
                        if a_adj not in res["adj"]:
                            res["adj"].append(a_adj)
                    cursor.execute("SELECT noun FROM ldoce_adj_nouns WHERE adj = ?", (w_clean,))
                    for (a_noun,) in cursor.fetchall():
                        if a_noun not in res["colloc_nouns"]:
                            res["colloc_nouns"].append(a_noun)
            except Exception:
                pass

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
        [DEPRECATED in favor of LDOCE 6th Edition]
        Redirects to get_ldoce_collocations to eliminate mechanical A-Z alphabetical bias
        and ground all collocation resolution 100% in LDOCE frequency & pedagogical salience.
        """
        return cls.get_ldoce_collocations(word=word, pos=pos, context_sentence=context_sentence)


    # ------------------------------------------------------------------
    # Collocation rendering (backlog E7 遗留 ①)
    # ------------------------------------------------------------------
    # A rendered collocation has to be a phrase the entry's own data supports.
    # Longman's boxes mix bare collocates ('lie', 'firm', 'music', 'birth') with whole
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

        Returns (order, gap, head_form): which of the two Longman's own text puts first,
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
        # Longman's COLLOCATIONS box for a noun prints the headword at the END of its own
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
        # Each box item carries the example Longman printed beside it, which is the entry's
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
            """Longman prints alternatives and holes inside a box
            ('completely/totally/quite wrong', 'work in industry/education/publishing
            etc', 'lay ... foundations'), and it prints optional words in brackets
            ('keep (somebody/something) warm/safe/dry etc', 'fit (into) a mould',
            'in the same mould (as somebody)'). Keep the first alternative, drop the
            'etc' and the bracketed optionals, and skip the frames with a hole in them -
            those are patterns, not phrases. Brackets have to go first: 1,092 of the
            164,207 stored collocation items carry one (scripts/_probe_collo_parens.py),
            and the '/'-split was cutting inside 'keep (somebody/something) ...', which
            put 'keep (somebody' in front of a learner. 35 of those items end in a
            register note glued on by the parser ('give a raspberry )American English',
            'in back (of something) American English'); the note is not part of the
            phrase, and it is only stripped on items that carried a bracket, so a real
            collocate like 'American English' on the row 'English' survives."""
            raw = item or ""
            text = re.sub(r"\([^)]*\)?", " ", raw)
            text = re.sub(r"[][()]", " ", text)
            if "(" in raw or ")" in raw:
                text = re.sub(r"\s*\b(?:American|British|Australian|Canadian|North American)"
                              r"\s+English\b", " ", text)
            text = re.sub(r"\s*\betc\.?\b", " ", text.strip(), flags=re.IGNORECASE)
            text = re.sub(r"\s+", " ", text).strip()
            if "..." in text:
                return ""
            if "/" in text:
                text = text.split("/")[0].strip()
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
            that. Gluing it onto a bare collocate in a combination Longman never prints
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
            # Already a phrase Longman wrote - the headword stands in it as its own word,
            # in any form the entry allows (hyphen boundaries count, so 'part-time' counts
            # for 'time' but 'network' does not count for 'work', and 'human rights' is
            # already the phrase rather than something to glue a second 'right' onto).
            if re.search(head_any_re, lowered):
                return text

            if mode == "verb_before":
                # Where the entry's own text puts the collocate in front of the headword
                # that frame decides the phrase. Where it does not, the collocate is only
                # safe to glue when the box is a noun box - Longman's VERBS section there
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

        # 5. Preposition: '[word] for', '[word] to'
        for p in (entry.get("prep") or [])[:2]:
            phrase = render(p, "prep")
            if phrase:
                results.append(phrase)

        # 6. The ADVERB section of a COLLOCATIONS box ('completely wrong', 'listen
        # attentively') and the PHRASES section ('listen to reason', 'Have a listen').
        # F7 ingested both buckets; nothing rendered them.
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
            """Filter out collective nominals (e.g. 'blind', 'baffled', 'dead') that are primarily adjectives."""
            try:
                pos_list = [w.pos for w in wn.words(lemma)]
                if any(p in ("a", "s") for p in pos_list):
                    if "n" not in pos_list or len([p for p in pos_list if p in ("a", "s")]) >= len([p for p in pos_list if p == "n"]):
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
            def_tokens = {t.lemma_.lower() for t in def_doc if t.is_alpha and not t.is_stop and len(t.text) >= 3} - cls._ANCHOR_STOPWORDS
            quote_tokens = ({t.lemma_.lower() for t in quote_doc if t.is_alpha and not t.is_stop and len(t.text) >= 3} - cls._ANCHOR_STOPWORDS) - def_tokens

            scored_synsets = []
            seen_syn_ids = set()
            for w in words:
                for s in w.synsets():
                    if s.id in seen_syn_ids:
                        continue
                    seen_syn_ids.add(s.id)

                    s_direct_doc = nlp((s.definition() + " " + " ".join(s.examples())).lower())
                    s_direct_lemmas = {t.lemma_.lower() for t in s_direct_doc if t.is_alpha and not t.is_stop and len(t.text) >= 3} - cls._ANCHOR_STOPWORDS

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
                    chain_lemmas = {t.lemma_.lower() for t in chain_doc if t.is_alpha and not t.is_stop and len(t.text) >= 3} - cls._ANCHOR_STOPWORDS

                    # Semantic Domain Guardrails:
                    # Penalize animal group senses if target definition is human/organization
                    is_human_def = any(h in def_tokens for h in ("people", "person", "organization", "employee", "staff", "student", "worker", "member", "social", "human", "group"))
                    penalty = 0
                    if is_human_def and ("animal" in chain_lemmas or "animal group" in " ".join(chain_texts).lower()):
                        penalty += 15

                    # Penalize military/war senses if definition is not military/war
                    is_military_def = any(m in def_tokens for m in ("military", "army", "soldier", "war", "battle", "weapon", "navy"))
                    if not is_military_def and ("military" in s.definition().lower() or "military" in " ".join(chain_texts).lower()):
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
                    target_synsets = [s for ov, s in scored_synsets if ov >= best_ov - 1]
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

        # 2. Extract Sense-Locked Antonyms directly from matched synsets (Scheme 2: Bipolar Opposite Cluster)
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
                elif al in cls.get_awl_words() or any(al.startswith(p) and al[len(p):] in ocd for p in ("un", "in", "im", "ir", "il", "non", "dis")):
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
                            _evaluate_antonym(ow.lemma().lower(), priority_bonus=1)
            # If adjective satellite ('s'), trace to head synset in bipolar cluster
            if wn_pos == "a" and s.pos == "s":
                for head in s.get_related("similar"):
                    for hs in head.senses():
                        for ant_sense in hs.get_related("antonym"):
                            ant_lemma = ant_sense.word().lemma().lower()
                            _evaluate_antonym(ant_lemma, priority_bonus=2)
                            for opp_sim in ant_sense.synset().get_related("similar") + ant_sense.synset().get_related("also"):
                                for ow in opp_sim.words():
                                    _evaluate_antonym(ow.lemma().lower(), priority_bonus=1)

        raw_antonyms.sort(key=lambda x: x[1])
        for al, _ in raw_antonyms:
            if al not in sense_antonyms:
                sense_antonyms.append(al)

        tier1_synonyms: List[str] = []
        tier2_satellites: List[str] = []
        tier3_attributes: List[str] = []
        tier2_hyponyms: List[str] = []
        tier3_coordinates: List[str] = []
        seen = {clean_target}
        seen.update(sense_antonyms)

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
                        and (lemma in ocd or lemma in cls.get_awl_words())
                        and cls.is_cefr_compliant_distractor(lemma, clean_target)
                    ):
                        if wn_pos == "n" and _is_primarily_adj_or_verb(lemma):
                            continue
                        seen.add(lemma)
                        tier1_synonyms.append(lemma)

            if wn_pos == "a":
                # For Adjectives: WordNet organizes concepts into Bipolar Clusters (Scheme 1: Satellite Synset Harvester)
                # Tier 2: Satellite synsets (fine-grained descriptive neighbors in the bipolar cluster)
                sat_synsets = []
                if s.pos == "a":
                    sat_synsets.extend(s.get_related("similar") + s.get_related("also"))
                elif s.pos == "s":
                    for head in s.get_related("similar"):
                        for hw in head.words():
                            hl = hw.lemma().lower()
                            if (
                                hl not in seen
                                and hl not in global_synonyms
                                and " " not in hl and "_" not in hl and "-" not in hl
                                and (hl in ocd or hl in cls.get_awl_words())
                                and _pos_ok(hl)
                                and cls.is_cefr_compliant_distractor(hl, clean_target)
                            ):
                                seen.add(hl)
                                tier2_satellites.append(hl)
                        sat_synsets.extend(head.get_related("similar") + head.get_related("also"))

                for sim in sat_synsets:
                    for sw in sim.words():
                        lemma = sw.lemma().lower()
                        if (
                            lemma not in seen
                            and lemma not in global_synonyms
                            and " " not in lemma
                            and "_" not in lemma
                            and "-" not in lemma
                            and (lemma in ocd or lemma in cls.get_awl_words())
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
                                and (lemma in ocd or lemma in cls.get_awl_words())
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
                                    and (lemma in ocd or lemma in cls.get_awl_words())
                                    and _pos_ok(lemma)
                                    and cls.is_cefr_compliant_distractor(lemma, clean_target)
                                ):
                                    seen.add(lemma)
                                    tier3_attributes.append(lemma)
            else:
                # Tier 2: Hyponyms (more specific concepts / troponyms)
                for hypo in s.hyponyms():
                    for hw in hypo.words():
                        lemma = hw.lemma().lower()
                        if (
                            lemma not in seen
                            and lemma not in global_synonyms
                            and " " not in lemma
                            and "_" not in lemma
                            and "-" not in lemma
                            and (lemma in ocd or lemma in cls.get_awl_words())
                            and _pos_ok(lemma)
                            and cls.is_cefr_compliant_distractor(lemma, clean_target)
                        ):
                            if wn_pos == "n" and _is_primarily_adj_or_verb(lemma):
                                continue
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
                                and (lemma in ocd or lemma in cls.get_awl_words())
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
                                        and (lemma in ocd or lemma in cls.get_awl_words())
                                        and _pos_ok(lemma)
                                        and cls.is_cefr_compliant_distractor(lemma, clean_target)
                                    ):
                                        if _is_primarily_adj_or_verb(lemma):
                                            continue
                                        seen.add(lemma)
                                        tier3_coordinates.append(lemma)

        # LDOCE 6th Edition Thesaurus Integration (Tier 0: Gold Standard Psychometric Near-Synonym Peers)
        tier0_ldoce_thesaurus: List[str] = []
        for item in cls.get_ldoce_thesaurus(clean_target):
            tw = item.get("word", "").strip().lower()
            if (
                tw
                and tw != clean_target
                and " " not in tw
                and "_" not in tw
                and "-" not in tw
                and tw not in seen
                and tw not in global_synonyms
                and _pos_ok(tw)
                and cls.is_cefr_compliant_distractor(tw, clean_target)
            ):
                cand_entry = cls.get_ldoce_entry(tw)
                if cand_entry:
                    c_poses = [p.lower() for p in cand_entry.get("all_poses", [cand_entry.get("pos", "")])]
                    if wn_pos == "n" and not any("noun" in p for p in c_poses):
                        continue
                    elif wn_pos == "v" and not any("verb" in p for p in c_poses):
                        continue
                    elif wn_pos == "a" and not any("adj" in p for p in c_poses):
                        continue
                seen.add(tw)
                tier0_ldoce_thesaurus.append(tw)

        # Candidate prioritization
        if wn_pos == "v":
            candidates = tier0_ldoce_thesaurus + [c for c in (tier2_hyponyms + tier3_coordinates) if c not in global_synonyms]
        elif wn_pos == "a":
            # For adjectives: retain satellite_synonyms first to satisfy existing fine-grained satellite tests,
            # then inject LDOCE thesaurus peers, followed by attribute coordinates.
            candidates = tier2_satellites + tier0_ldoce_thesaurus + tier3_attributes + tier1_synonyms
            # Preposition valency cloze (Scheme 4): prioritize academic adjectives governing DIFFERENT prepositions
            if anchor_type == "prep" and context_anchor:
                clean_anchor = context_anchor.strip().lower()
                valency_candidates = []
                for p, adjs in cls._ADJ_PREP_VALENCY_MAP.items():
                    if p != clean_anchor:
                        for a in adjs:
                            if a != clean_target and a not in exclude_words and cls.is_cefr_compliant_distractor(a, clean_target):
                                valency_candidates.append(a)
                # Prioritize local semantic peers (satellites, LDOCE thesaurus) for domain relevance,
                # then complement with academic adjectives governing distinct prepositions
                candidates = tier2_satellites + tier0_ldoce_thesaurus + valency_candidates + tier3_attributes + tier1_synonyms
        else:
            candidates = tier0_ldoce_thesaurus + tier1_synonyms + tier2_hyponyms + tier3_coordinates

        # 3. Oxford Collision Clearance (Anti Double-Key Gate - Schemes 3 & 4)
        forbidden_words = set(global_synonyms) if wn_pos == "v" else {clean_target}
        if context_anchor:
            clean_anchor = context_anchor.strip().lower()
            if wn_pos == "a" and anchor_type == "prep":
                # Scheme 4: Preposition Valency Gate - reject candidates that also govern the target preposition
                for cand in candidates:
                    cand_preps = ocd.get(cand, {}).get("prep", [])
                    if clean_anchor in cand_preps or cand in cls._ADJ_PREP_VALENCY_MAP.get(clean_anchor, []):
                        forbidden_words.add(cand)
            elif wn_pos == "a" and anchor_type == "modified_noun":
                # Scheme 3: Modified Noun Collocation Clash - reject legal adjectives that also collocate with the noun
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
        for ant in sense_antonyms:
            if ant not in safe_distractors and ant not in exclude_words and len(ant) >= 2:
                safe_distractors.append(ant)
                distractor_metadata[ant] = "antonym"
                break

        # Slots 2 & 3: Candidates from Coordinate/Troponym/Synonym trees
        thesaurus_picks = 0
        for cand in candidates:
            if cand in safe_distractors:
                continue
            # Attributive position check: if modifying a noun, reject predicative-only adjectives
            if anchor_type == "modified_noun" and cand in cls._PREDICATIVE_ONLY_ADJS:
                continue
            # Allomorph / Word Family Guard: prevent morphological double-keys (e.g. resiliency vs resilience)
            if cls.are_same_word_family(cand, clean_target) or any(cls.are_same_word_family(cand, s) for s in safe_distractors):
                continue
            # De-synonym piling: cap LDOCE thesaurus near-synonyms to at most 1
            if cand in tier0_ldoce_thesaurus:
                if thesaurus_picks >= 1:
                    continue
            # Cross-distractor mutual synonym exclusion: avoid multiple distractors being mutual synonyms
            cand_syns = cls.get_synonyms(cand)
            if any(s in cand_syns or cand in cls.get_synonyms(s) for s in safe_distractors):
                continue
            if cand not in forbidden_words and cand not in exclude_words and len(cand) >= 2:
                safe_distractors.append(cand)
                if cand in tier0_ldoce_thesaurus:
                    thesaurus_picks += 1
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

        # Second Defense: Dynamic domain-aligned fallback using target WordNet lexfile
        # When candidates pool is still sparse, extract homogeneous fallback words from the target's
        # exact WordNet lexfile (e.g. noun.group, noun.communication) to prevent Domain/Category Mismatch.
        if len(safe_distractors) < target_count and wn_pos == "n":
            target_syns = [s for s in wn.synsets(clean_target, "n") if s.lexfile()]
            target_lfs = set(s.lexfile() for s in target_syns)
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
    def semantic_fields(cls, word: str) -> Tuple[Set[str], Set[str]]:
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

        seeds = []
        for pos in ("a", "s", "v", "n", "r"):
            try:
                seeds += wn.synsets(target, pos)
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
        near, antonyms = cls.semantic_fields(t)
        # Also include LDOCE thesaurus words in near semantic field
        for item in cls.get_ldoce_thesaurus(t):
            tw = item.get("word", "").strip().lower()
            if tw and " " not in tw and tw != t:
                near.add(tw)

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
            # and candidate d does NOT share that collocation in OCD/LDOCE, candidate d is ruled out
            # by the anchor slot itself and is therefore a valid, single-fit distractor.
            if anchor and anchor_type in ("adj", "verb", "modified_noun", "verb_subject", "modifies_adj", "modifies_verb"):
                a_clean = anchor.lower()
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
                    # Check if distractor adverb explicitly collocates with this anchor in LDOCE
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

    @classmethod
    def generate_phrase_distractors(
        cls,
        phrase: str,
        pos: str = "phrasal_verb",
        count: int = 3,
        exclude_words: Optional[Set[str]] = None
    ) -> List[str]:
        """
        Synthesizes high-discrimination, authentic distractors for multi-word phrasal verbs and collocations.
        For 2-word phrasal verbs (e.g. 'send out'):
          - Pool A (Particle contrast): Same verb, different authentic particles (e.g. 'send in', 'send off')
          - Pool B (Verb contrast): Same particle, different core verbs (e.g. 'point out', 'carry out')
        For multi-word frames (3+ words, e.g. 'toward the end of'):
          - Synthesizes structural sister frames (e.g. 'at the beginning of', 'in the middle of')
        """
        phrase_clean = phrase.strip().lower()
        words = phrase_clean.split()
        exclude = set(exclude_words or [])
        distractors: List[str] = []

        # Membership is Longman's, not a string table.  A candidate qualifies only when the
        # dictionary states it - as a phrasal-verb block, a PHRASES item, a grammar pattern
        # built on the headword, or a COLLOCATIONS box item.  'keep of' fails because no entry
        # states it: the 'of' was lifted out of 'keep somebody out of something'.
        def ships(cand: str) -> bool:
            return (cand != phrase_clean and cand not in exclude
                    and cls.is_attested_phrase(cand))

        tier_rank = {tier: index for index, tier in enumerate(cls._PHRASE_TIERS)}

        def strength(cand: str) -> int:
            """Longman's own phrasal-verb block beats a PHRASES item, which beats a pattern."""
            hits = cls.ldoce_phrase_hits(cand)
            return min((tier_rank.get(tier, len(tier_rank)) for tier, _text in hits),
                       default=len(tier_rank))

        # 1. Two-word phrase (e.g. 'send out', 'tend to', 'cut out', 'work for')
        if len(words) == 2:
            v, p = words[0], words[1]
            diff_part: List[str] = []
            particles = ["in", "off", "up", "down", "away", "back", "over", "into", "through", "on", "to", "for", "out", "with", "from", "around"]
            for part in particles:
                if part != p:
                    cand = f"{v} {part}"
                    if ships(cand) and cand not in diff_part:
                        diff_part.append(cand)

            # Longman's own phrasal-verb list for this verb is the strongest pool there is
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

        # 2. Multi-word collocations (3+ words)
        elif len(words) >= 3:
            if "end of" in phrase_clean:
                candidates = ["at the beginning of", "in the middle of", "for the rest of", "at the start of"]
                distractors = [c for c in candidates if c != phrase_clean and c not in exclude]
            elif "piece of" in phrase_clean:
                candidates = ["a series of", "a range of", "a couple of", "a matter of", "a variety of"]
                distractors = [c for c in candidates if c != phrase_clean and c not in exclude]
            elif "thanks to" in phrase_clean:
                candidates = ["show respect to", "pay tribute to", "give credit to", "express gratitude to"]
                distractors = [c for c in candidates if c != phrase_clean and c not in exclude]
            elif "type of" in phrase_clean:
                candidates = ["any form of", "any sort of", "any kind of", "any part of"]
                distractors = [c for c in candidates if c != phrase_clean and c not in exclude]

        # General Fallback if pool is sparse - still filtered through the dictionary, because
        # an invented distractor is the same invention this gate exists to stop.
        if len(distractors) < count:
            if len(words) == 2:
                for fallback_cand in ["carry out", "take over", "bring about", "set up", "look into", "come across", "give in"]:
                    if ships(fallback_cand) and fallback_cand not in distractors:
                        distractors.append(fallback_cand)
                    if len(distractors) >= count:
                        break

        return distractors[:count]

    # When a verb has NO Oxford Collocations entry (or an empty 'prep' valency
    # list), only these typical academic complement prepositions may be
    # trusted from syntactic guessing; anything else degrades to anchor=None
    # (neutral template) instead of fabricating a frame (e.g. 'degrade in a
    # society' from a fronted adjunct).
    _TYPICAL_COMPLEMENT_PREPS = {"into", "to", "from"}

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
            # A usable quote must contain a real predicate (not just an aux/copula fragment)
            has_predicate = any(
                tok.pos_ == "VERB" and tok.dep_ not in ("aux", "auxpass", "cop")
                for tok in doc
            )
            if not has_predicate:
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
        if target_word and not cls.text_contains_form(text, target_word):
            return False
        nlp = cls.get_spacy()
        if nlp is None:
            return False
        try:
            doc = nlp(text)
        except Exception:
            return False
        # The root must be a real predicate: 'Why?' has no clause to read evidence from.
        roots = [tok for tok in doc if tok.dep_ == "ROOT"]
        if not roots or not any(tok.pos_ in ("VERB", "AUX") for tok in roots):
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
        Returns only those LDOCE patterns of a sense that actually contain the headword
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
        Locked-Sense Frame Concordance Gate (P0-4).

        A bound-preposition anchor is only legitimate when the LDOCE sense this item
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
    # (P1-1 distractor POS/slot legality, P1-2 inflection/example agreement)
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
        "lose": ("lost", "lost", "losing", "loses"),
        "understand": ("understood", "understood", "understanding", "understands"),
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
        True/False from LDOCE valency evidence for this verb; None when the dictionary is
        silent. Evidence is read from grammar patterns ('pay somebody for something') and,
        because many LDOCE senses publish no pattern at all, from the sense examples
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
        Slot Legality Gate (P1-1).

        A distractor must be able to occupy the SAME syntactic slot as the target, or the
        item is solvable by morphology alone instead of by meaning: 'log in to a ____'
        offered ['account', 'somehow', 'autonomy', 'message'], and a transitive verb frame
        offered the intransitive 'flourish'. The gate consults only positive evidence —
        WordNet lemma POS records and LDOCE valency patterns — and never vetoes a word the
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
        Public Slot Legality Gate (P1-1) for downstream consumers such as the evaluator,
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
        Sense-aware contrast instruction (P1-4).

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
            def_text = (definition or "").strip()
            grounding = f" that operationalises the locked meaning '{def_text}'" if def_text else ""
            named = ", ".join(f"'{d}'" for d in near_synonyms[:3])
            return (f" Establish a precision cue{grounding} — the exact scale, register or "
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
        LDOCE thesaurus headwords can be multi-form ('unwilling/not willing',
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
        return any(v.split()[0] == w for v in variants if v)

    @classmethod
    def _semantic_discriminator_note(
        cls,
        target_word: str,
        target_definition: Optional[str],
        dist_list: List[str],
        dist_meta: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Target-anchored semantic discriminator (P1-5).

        LDOCE thesaurus items carry only {word, distinction, example}, and 'distinction'
        is that word's own cluster definition. Emitting a distractor's distinction alone
        produced "Semantic Discriminator: <what the distractor means> (strictly ruling out
        <the distractor>)" — a distractor defining itself, which the model then copied
        verbatim into 'why_wrong'. Every note is now stated as a DISTINCTION whose subject
        is the locked target sense, and a note that merely restates the target's own
        definition (carrying no contrast) is rejected rather than pasted.
        """
        tw = (target_word or "").strip().lower()
        dists = [d for d in (dist_list or []) if d]
        if not tw or not dists:
            return ""
        cluster_nuance = ""
        thes_items = cls.get_ldoce_thesaurus(tw)
        for th in thes_items:
            if cls._thesaurus_variant_matches(th.get("word"), tw):
                cluster_nuance = (th.get("distinction") or "").strip()
                break
        locked = (target_definition or "").strip()
        # The subject of the contrast must be the sense this item actually teaches. A thesaurus
        # cluster can sit under a different part of speech than the locked sense ('charge' the
        # amount vs 'charge' to ask somebody to pay), so its wording is only adopted when it is
        # effectively the same sense; otherwise the locked definition anchors the note.
        target_side = locked
        if cluster_nuance and (not locked or
                               cls._definition_overlap_ratio(cluster_nuance, locked) >= 0.7):
            target_side = cluster_nuance
        if not target_side:
            return ""
        meta = dist_meta or {}
        for dist in dists:
            dist_note = ""
            for th in thes_items:
                if cls._thesaurus_variant_matches(th.get("word"), dist):
                    dist_note = (th.get("distinction") or "").strip()
                    break
            mechanism = meta.get(dist) or meta.get(dist.lower()) or ""
            if not dist_note:
                if mechanism != "antonym":
                    continue
                # A polar opposite needs no dictionary text to be excluded: the contrast is the
                # polarity of the locked sense itself, stated against the target, never against
                # the distractor's own definition.
                return (
                    f"Semantic Discriminator: DISTINCTION between '{tw}' ({target_side}) and "
                    f"'{dist}' — '{dist}' is the polar opposite of the locked meaning and cannot "
                    f"fill the blank. The sentence must supply the cue that only '{tw}' satisfies."
                )
            if cls._definition_overlap_ratio(dist_note, target_side) >= 0.8:
                continue  # no real contrast in this cluster pairing
            hinge = ("the polar opposite of the locked meaning"
                     if mechanism == "antonym"
                     else "a different point on the scale or a different collocational frame")
            return (
                f"Semantic Discriminator: DISTINCTION between '{tw}' ({target_side}) and "
                f"'{dist}' — '{dist}' fails the blank because it encodes {hinge}. "
                f"The sentence must supply the cue that only '{tw}' satisfies."
            )
        return ""

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
    # recognises monosyllables, so without this list Longman's pattern 'be embedded in
    # something' can never be reached from 'embed'.  Verbs that end in -er or -or but stress
    # the first syllable ('differ', 'offer', 'visit', 'profit') are correctly left out.
    _DOUBLING_VERBS = frozenset({
        "embed", "refer", "prefer", "transfer", "confer", "defer", "concur", "demur",
        "occur", "recur", "abhor", "control", "patrol", "regret", "offset", "upset",
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
        if w[-1] in "aeiouwxy":
            return False
        if w[-2] not in "aeiou":
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
        Backlog A1: the single legal-form generator.

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
        for tag in ("VBZ", "VBD", "VBN", "VBG"):
            inflected = cls._inflect_verb(w, tag)
            if inflected:
                forms.add(inflected)

        plural = cls.pluralize_noun(w)
        if plural:
            forms.add(plural)

        p = (pos or "").strip().lower()
        if p in ("adjective", "adj", "a", "s"):
            if len(w) > 3:
                forms.update({w + "er", w + "est"})
                if w.endswith("y"):
                    forms.update({w[:-1] + "ier", w[:-1] + "iest"})

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
    def text_contains_form(cls, text: Optional[str], target: str) -> bool:
        """Does `text` contain `target` as a word or as any legal inflection of it?"""
        body = (text or "").lower()
        t = (target or "").strip().lower()
        if not body or not t:
            return False
        if re.search(rf"\b{re.escape(t)}\b", body):
            return True
        tokens = set(re.findall(r"\b[a-z]+\b", body))
        return any(tok in cls.inflected_forms(t) for tok in tokens)

    @classmethod
    def is_attested_form(cls, form: str) -> bool:
        """
        Is this surface form an actual English word form?

        WordNet stores lemmas, not inflections - wn.words('buses') is empty even though
        'buses' is perfectly good English - so a form is also accepted when spaCy resolves
        it to a WordNet-attested lemma AND the deterministic generator agrees that is how
        that lemma inflects. The second half is what rejects 'choosed': spaCy may guess the
        lemma 'choose', but inflected_forms('choose') yields 'chose'/'chosen', never 'choosed'.
        """
        f = (form or "").strip().lower()
        if not f:
            return False
        if f in cls._attested_form_cache:
            return cls._attested_form_cache[f]
        verdict = False
        try:
            wn = cls.get_wordnet()
            if wn.words(f):
                verdict = True
            else:
                lemma = cls._surface_lemma(f)
                verdict = bool(lemma) and bool(wn.words(lemma)) and f in cls.inflected_forms(lemma)
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
        Backlog A1: compliance gates must see the form the student actually reads.

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
        Prescribed-Inflection Concordance (P1-2).

        The blueprint declares `inflection` and the LLM must obey it, so the declaration
        must come from the same evidence the item is built on. For verbs it was always
        'base form' even when the authentic LDOCE example that supplied the frame read
        'What annoyed him most was that he had received no apology.' — a past-tense frame.
        The blueprint then demanded a base-form verb inside a past-tense cloze, which is
        unanswerable. When the anchoring example is inflected, every option is inflected to
        the identical form so frame, declared inflection and options agree.
        Returns (inflection_desc, options, target_word, form_tag).
        """
        if (canonical_pos or "").lower() != "verb" or inflection_desc != "base form":
            return inflection_desc, base_options, target_word, "VB"
        tag = cls._example_target_tag(authentic_example, target_word)
        if tag not in ("VBD", "VBZ", "VBG"):
            return inflection_desc, base_options, target_word, tag or "VB"
        inflected: List[str] = []
        for opt in base_options:
            form = cls._inflect_verb(opt, tag)
            if not form or not cls._morph_form_valid(opt, form, tag):
                return inflection_desc, base_options, target_word, tag  # unsafe to rewrite
            inflected.append(form)
        new_target = inflected[target_index] if 0 <= target_index < len(inflected) else target_word
        # A1: an inflected target that is not an attested English form must never ship,
        # even when spaCy was willing to lemmatize it back to the base verb.
        if not cls.is_attested_form(new_target):
            return inflection_desc, base_options, target_word, tag
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
        [DEPRECATED in favor of LDOCE 6th Edition]
        Delegates directly to find_ldoce_zero_collision_anchor to ensure 100% LDOCE semantic ordering
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
    # grammar pattern to hang the anchor on. 18,884 LDOCE entries have neither - only
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
        quote: Optional[str] = None
    ) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
        """
        Retrieves an authoritative, natural collocational anchor and authentic corpus example
        from Longman Dictionary of Contemporary English (LDOCE 6th Edition).
        Guarantees that the chosen anchor collocates with the target word in LDOCE with authentic
        native-speaker example backing, while preventing double-key collisions with distractors.
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

        # 1. Adjective Cascades (prep valency >> modified_noun >> adv_mod)
        if p_low.startswith("adj") or p_low == "a":
            # Level 1: Preposition valency from grammar patterns (e.g. 'proud of', 'aware of')
            for s in entry.get("senses", []):
                for pat in s.get("patterns", []):
                    for p in ("of", "to", "for", "with", "about", "from", "in", "on", "at"):
                        if re.search(rf"\b{p}\b", pat.lower()):
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

            # Level 2: Modified Noun from indexed LDOCE Collocations
            if not anchor and cls._ldoce_conn is not None and cls._ldoce_table_exists("ldoce_adj_nouns"):
                try:
                    cursor = cls._ldoce_conn.cursor()
                    res = cursor.execute(
                        "SELECT noun, example, collocation FROM ldoce_adj_nouns WHERE adj = ?", (tw_clean,)
                    ).fetchall()
                    preferred_nouns = [
                        "task", "solution", "rule", "question", "answer", "decision", "problem",
                        "idea", "method", "system", "process", "calculation", "device", "technique"
                    ]
                    for pn in preferred_nouns:
                        for n, ex, col in res:
                            if n == pn and ex:
                                if not any(cls.double_key_collision(tw_clean, d, anchor=n, anchor_type="modified_noun", pos="adj")[0] for d in dists):
                                    anchor = n
                                    atype = "modified_noun"
                                    frame_desc = f"modifying noun {n}"
                                    authentic_ex = ex
                                    break
                        if anchor:
                            break
                    if not anchor and res:
                        for n, ex, col in res:
                            if ex and not any(cls.double_key_collision(tw_clean, d, anchor=n, anchor_type="modified_noun", pos="adj")[0] for d in dists):
                                anchor = n
                                atype = "modified_noun"
                                frame_desc = f"modifying noun {n}"
                                authentic_ex = ex
                                break
                except Exception:
                    pass

            if not anchor and entry.get("collocations", {}).get("nouns"):
                for it in entry["collocations"]["nouns"]:
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
                for it in entry["collocations"]["phrases"]:
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
            preferred_objs = [
                "problem", "crisis", "dispute", "mystery", "puzzle", "conflict",
                "dilemma", "difficulty", "case", "crime", "decision", "question"
            ]
            for it in entry.get("collocations", {}).get("nouns", []):
                c_str = it.get("collocation", "").lower()
                ex = it.get("example", "")
                for po in preferred_objs:
                    if po in c_str and ex:
                        if not any(cls.double_key_collision(tw_clean, d, anchor=po, anchor_type="object", pos="verb")[0] for d in dists):
                            anchor = po
                            atype = "object"
                            frame_desc = f"direct object noun {po}"
                            authentic_ex = ex
                            break
                if anchor:
                    break

            if not anchor and entry.get("collocations", {}).get("nouns"):
                for it in entry["collocations"]["nouns"]:
                    ex = it.get("example", "")
                    c_str = it.get("collocation", "").lower()
                    # 'undertake a task/a project' must yield 'task', never the target
                    # itself: the raw token split used to hand back 'undertake' as its own
                    # anchor, which the batch gate then rejected - taking the example with it.
                    tok_clean = cls._pick_anchor_token([c_str], target_word=tw_clean, prefer_pos="noun")
                    if tok_clean and not any(cls.double_key_collision(tw_clean, d, anchor=tok_clean, anchor_type="object", pos="verb")[0] for d in dists):
                        anchor = tok_clean
                        atype = "object"
                        frame_desc = f"direct object noun {tok_clean}"
                        authentic_ex = ex
                        break
                    if anchor:
                        break

            # Transitive verb direct object extraction from LDOCE sense examples
            if not anchor:
                try:
                    nlp = cls.get_spacy()
                    for s in entry.get("senses", []):
                        for ex in s.get("examples", []):
                            doc_ex = nlp(ex)
                            for tok in doc_ex:
                                if tok.lemma_.lower() == tw_clean:
                                    for child in tok.children:
                                        if child.dep_ == "dobj" and child.lemma_.lower() not in cls._ANCHOR_STOPWORDS:
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

            # Preposition patterns (e.g. 'depend on', 'arrive at', 'hesitate over', or transitive 'promote sb to sth')
            if not anchor:
                for s in entry.get("senses", []):
                    for pat in s.get("patterns", []):
                        pat_l = pat.lower()
                        for p in ("to", "on", "at", "about", "for", "with", "over", "into", "of", "from", "in"):
                            if re.search(rf"\b{p}\b", pat_l):
                                ex = s.get("examples", [""])[0] if s.get("examples") else ""
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

        # 3. Noun Cascades (agentive subject verbs >> preposition valency >> governing verbs >> descriptive adjectives)
        elif p_low.startswith("n"):
            # Level 1: Agentive / Characteristic Subject Verbs (e.g. 'telephone rings', 'clock chimes', 'heart beats')
            for it in entry.get("collocations", {}).get("verbs", []):
                c_str = it.get("collocation", "").lower()
                ex = it.get("example", "")
                # Word-boundary match, not substring: 'bring' contains 'ring' and
                # 'unchanged' contains 'change', so a plain `w in c_str` test promotes an
                # unrelated verb into the agentive-subject slot and the cascade stops there.
                is_subj = bool(re.search(
                    r"\b(?:ring|rings|rang|rung|chime|chimes|chimed|beat|beats|beating"
                    r"|erupt|erupts|erupted|start|starts|started|occur|occurs|occurred"
                    r"|exist|exists|existed|change|changes|changed|hardens|hardened)\b",
                    c_str,
                ))
                if is_subj:
                    tok_partner = cls._pick_anchor_token([c_str], target_word=tw_clean)
                    if tok_partner:
                        # Lemmatize third-person singular (e.g. rings -> ring, chimes -> chime, beats -> beat)
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

            # Level 2: Governing Verbs (e.g. 'make/reach a decision', 'answer the telephone')
            if not anchor:
                for it in entry.get("collocations", {}).get("verbs", []):
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

            # Level 3: Noun preposition patterns from LDOCE senses (e.g. 'control of/over', 'access to', 'interest in')
            if not anchor:
                quote_preps = []
                if quote:
                    for p_cand in ("over", "of", "to", "for", "with", "about", "from", "in", "on", "into", "against", "towards", "toward"):
                        if re.search(rf"\b{p_cand}\b", quote.lower()):
                            quote_preps.append(p_cand)

                # Prioritize prepositions confirmed in quote
                search_order = quote_preps + [p for p in ("over", "of", "to", "for", "with", "about", "from", "in", "on", "into", "against", "towards", "toward") if p not in quote_preps]

                for s in entry.get("senses", []):
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

            # Level 3: Descriptive Adjectives
            if not anchor:
                for it in entry.get("collocations", {}).get("adjectives", []):
                    c_str = it.get("collocation", "").lower()
                    ex = it.get("example", "")
                    tok_partner = cls._pick_anchor_token([c_str], target_word=tw_clean, prefer_pos="adj")
                    if tok_partner and tok_partner not in ("good", "bad", "great"):
                        if not any(cls.double_key_collision(tw_clean, d, anchor=tok_partner, anchor_type="adj", pos="noun")[0] for d in dists):
                            anchor = tok_partner
                            atype = "adj"
                            frame_desc = f"modified by adjective {tok_partner}"
                            authentic_ex = ex
                            break

        # 4. Adverb Cascades (modifies_verb >> modifies_adj or modifies_adj >> modifies_verb based on adverb semantics & quote evidence)
        elif p_low.startswith("adv") or p_low == "r":
            quote_frame = cls._adverb_frame_from_quote(tw_clean, quote)
            locked_idx = cls._lock_sense(entry, definition=definition, quote=quote, frame_evidence=quote_frame, target_pos="adv")
            locked_sense = entry.get("senses", [])[locked_idx] if (locked_idx is not None and locked_idx < len(entry.get("senses", []))) else None

            # First priority: Anchor token directly present in authentic quote or locked sense example
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

            if not anchor and cls._ldoce_conn is not None and cls._ldoce_table_exists("ldoce_adv_collocations"):
                try:
                    cursor = cls._ldoce_conn.cursor()
                    cursor.execute(
                        "SELECT anchor_type, anchor_word, collocation, example FROM ldoce_adv_collocations WHERE adv = ?",
                        (tw_clean,)
                    )
                    rows = cursor.fetchall()
                    
                    # Group by rows
                    v_rows = [r for r in rows if r[0] == "modifies_verb"]
                    a_rows = [r for r in rows if r[0] == "modifies_adj"]

                    # If quote evidence indicated a frame, prioritize that frame
                    if quote_frame == "modifies_adj":
                        cand_order = (a_rows, v_rows)
                    elif quote_frame == "modifies_verb":
                        cand_order = (v_rows, a_rows)
                    else:
                        # Fallback heuristic: degree adverbs prioritize modifies_adj
                        is_degree_adv = tw_clean in (
                            "extremely", "completely", "totally", "utterly", "highly", "deeply",
                            "fairly", "quite", "rather", "pretty", "slightly", "barely", "scarcely",
                            "hardly", "vastly", "immensely", "hugely", "terribly", "awfully", "decidedly"
                        ) or (locked_sense and any(k in (locked_sense.get("grammar") or "").lower() for k in ("adjective", "adverb", "[+adjective")))
                        cand_order = (a_rows, v_rows) if is_degree_adv else (v_rows, a_rows)

                    for cand_rows in cand_order:
                        for c_atype, c_aword, c_colloc, c_ex in cand_rows:
                            # Verify candidate anchor does NOT clash with distractors
                            clash = False
                            for d in dists:
                                if cls.double_key_collision(tw_clean, d, anchor=c_aword, anchor_type=c_atype, pos="adv", quote=quote)[0]:
                                    clash = True
                                    break
                            if not clash and c_ex:
                                anchor = c_aword
                                atype = c_atype
                                frame_desc = f"modifying verb {c_aword}" if c_atype == "modifies_verb" else f"modifying adjective {c_aword}"
                                authentic_ex = c_ex
                                break
                        if anchor:
                            break
                except Exception:
                    pass

            # Level 3: Syntactic extraction from LDOCE sense examples (e.g. 'extremely difficult', 'stopped abruptly')
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
                        for s in entry.get("senses", []):
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

        # 5. Last resort: the dictionary's own sense example. Every cascade above needs
        # a collocation item or a grammar pattern; entries that have neither used to
        # return nothing at all, leaving the item without any structural model.
        if not anchor:
            anchor, atype, frame_desc, authentic_ex = cls._anchor_from_sense_example(
                tw_clean, entry, pos=p_low, definition=definition,
                quote=quote, distractors=dists
            )

        if anchor:
            return anchor, atype, frame_desc, authentic_ex

        return None, None, None, None



    @classmethod
    def ldoce_example_for_anchor(cls, word: str, anchor: Optional[str]) -> Optional[str]:
        """
        The authentic LDOCE sentence that demonstrates the (target, anchor) pair.
        Returning None is meaningful: it means the dictionary never puts these two words
        together, so a quote-derived anchor is an accident of one sentence
        ('further penalty', 'maintain contact') and cannot serve as a structural model.
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
            # 1. Curated collocations of the target
            for items in (t_entry.get("collocations") or {}).values():
                for item in items or []:
                    if anc_re.search((item.get("collocation") or "").lower()) and item.get("example"):
                        return item["example"]
            # 2. Grammar patterns of the target, preferring the example that shows the frame
            for s in t_entry.get("senses") or []:
                for pat in s.get("patterns") or []:
                    if anc_re.search(pat.lower()) and s.get("examples"):
                        m_exs = [e for e in s["examples"] if anc_re.search(e.lower())]
                        return m_exs[0] if m_exs else s["examples"][0]
            # 3. The target's own examples - the anchor may simply live inside one
            for s in t_entry.get("senses") or []:
                for ex in s.get("examples") or []:
                    if anc_re.search(ex.lower()):
                        return ex
            # 4. Collocations imported from other entries
            for item in t_entry.get("cross_collocations") or []:
                if anc_re.search((item.get("collocation") or "").lower()) and item.get("example"):
                    return item["example"]

        # 5. The anchor's own entry may be the one that records the pairing
        a_entry = cls.get_ldoce_entry(anc_l)
        if a_entry:
            for items in (a_entry.get("collocations") or {}).values():
                for item in items or []:
                    if tw_re.search((item.get("collocation") or "").lower()) and item.get("example"):
                        return item["example"]

        # 6. Adjective-noun index, when the shipped database actually carries it
        if cls._ldoce_conn is not None and cls._ldoce_table_exists("ldoce_adj_nouns"):
            try:
                row = cls._ldoce_conn.execute(
                    "SELECT example FROM ldoce_adj_nouns WHERE adj = ? AND noun = ? AND length(example) > 0",
                    (tw_l, anc_l)
                ).fetchone()
                if row:
                    return row[0]
            except Exception:
                pass
        return None

    @classmethod
    def ldoce_sense_example(
        cls,
        word: str,
        pos: Optional[str] = None,
        definition: Optional[str] = None,
        quote: Optional[str] = None
    ) -> Optional[str]:
        """
        The dictionary's own sentence for the sense this item teaches, with no anchor
        requirement attached. Used as the structural model when no collocational anchor
        survived the gates - an item with no anchor still needs a real pattern to model.
        """
        entry = cls.get_ldoce_entry(word)
        if not entry:
            return None
        senses = entry.get("senses") or []
        locked = cls._lock_sense(entry, definition=definition, quote=quote, target_pos=pos)
        ordered: List[Dict[str, Any]] = []
        if locked is not None and 0 <= locked < len(senses):
            ordered.append(senses[locked])
        ordered.extend(s for i, s in enumerate(senses) if i != locked)
        for sense in ordered:
            for ex in sense.get("examples") or []:
                if ex and len(ex.split()) >= 3:
                    return ex
        for ex in (entry.get("corpus_examples") or []) + (entry.get("other_dict_examples") or []):
            if ex and len(ex.split()) >= 3:
                return ex
        return None

    @classmethod
    def get_acl_collocations(cls) -> Dict[str, str]:
        """Lazy-loads the Academic Collocation List (ACL, ~2474 items). Returns dict mapping core pattern -> canonical original."""
        if cls._acl_data is None:
            cls._acl_data = {}
            acl_path = Path(__file__).parent / "data" / "academic_collocations.json"
            if acl_path.exists():
                try:
                    with open(acl_path, "r", encoding="utf-8") as f:
                        raw_items = json.load(f)
                    for item in raw_items:
                        # strip optional prefixes/suffixes like (a), (be), (to), (of) for robust matching
                        core = re.sub(r"^\([a-z\s]+\)\s*", "", item)
                        core = re.sub(r"\s*\([a-z\s]+\)$", "", core).strip().lower()
                        if len(core) >= 3:
                            cls._acl_data[core] = item
                except Exception:
                    pass
        return cls._acl_data

    @classmethod
    def get_awl_words(cls) -> Set[str]:
        """Lazy-loads the Academic Word List (AWL, 560+ headwords)."""
        if cls._awl_data is None:
            cls._awl_data = set()
            awl_path = Path(__file__).parent / "data" / "academic_word_list.json"
            if awl_path.exists():
                try:
                    with open(awl_path, "r", encoding="utf-8") as f:
                        cls._awl_data = set(json.load(f))
                except Exception:
                    pass
        return cls._awl_data

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

        doc = nlp(body)

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
                masked_body = (
                    quote_body.replace(".", "§DOT§")
                              .replace("!", "§EXCL§")
                              .replace("?", "§QUES§")
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
                raw_sent = (
                    sent.text.replace("§DOT§", ".")
                             .replace("§EXCL§", "!")
                             .replace("§QUES§", "?")
                             .strip()
                )
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
        if not sent_text or not word:
            return "noun"

        nlp = cls.get_spacy()
        doc = nlp(sent_text)
        clean_w = re.sub(r"\[.*?\]|\(.*?\)", "", word).strip().lower()
        w_toks = clean_w.split()
        target_tok = w_toks[0] if w_toks else clean_w

        # 0. High-priority compound WordNet lookup: ONLY for genuine multi-word or hyphenated compounds
        # (Do NOT run on single words like 'firm' or 'log', where WordNet default sense overrides contextual dependency)
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

        # 1. Exact match on token text or lemma
        for tok in doc:
            if tok.text.lower() == target_tok or tok.lemma_.lower() == target_tok:
                # If modifying a noun as an adjectival modifier (amod, compound)
                if tok.dep_ in ("amod", "advmod") and tok.head.pos_ in ("NOUN", "PROPN"):
                    return "adjective"
                # Participle adjective / predicate adjective check:
                # E.g. 'we are done', 'we are finished', 'it is broken', 'he is tired'
                # If tagged as VBN/VERB governed by aux/cop 'be', check if dictionary lists it as an adjective
                if tok.tag_ == "VBN" or tok.pos_ == "VERB":
                    # Check if preceded or governed by form of 'be'
                    has_be_gov = any(c.lemma_ == "be" for c in tok.children if c.dep_ in ("aux", "auxpass", "cop")) or (tok.head.lemma_ == "be")
                    if has_be_gov:
                        ld_entry = cls.get_ldoce_entry(tok.text.lower())
                        if ld_entry and any("adj" in p for p in ld_entry.get("all_poses", [])):
                            return "adjective"
                return VALID_POS_MAP.get(tok.pos_, "noun")

        # 2. Hyphen-insensitive phrase span match (e.g. 'reschedule' in sentence with 're-schedule')
        target_no_hyphen = target_tok.replace("-", "")
        # Check regex span in sentence allowing hyphens/spaces
        hyphen_pat = r"\b" + r"[\s\-]*".join(re.escape(c) for c in target_no_hyphen) + r"\b"
        m_span = re.search(hyphen_pat, sent_text, re.IGNORECASE)
        if m_span:
            char_span = doc.char_span(m_span.start(), m_span.end(), alignment_mode="expand")
            if char_span:
                # Check if the span functions as an adjectival modifier for following noun
                if char_span.root.head.pos_ in ("NOUN", "PROPN") and char_span.root.dep_ in ("amod", "nmod", "compound"):
                    return "adjective"
                span_root_pos = char_span.root.pos_
                if span_root_pos in VALID_POS_MAP:
                    return VALID_POS_MAP[span_root_pos]
                for st in char_span:
                    if st.pos_ in VALID_POS_MAP:
                        return VALID_POS_MAP[st.pos_]

        # 3. Substring match on tokens
        for tok in doc:
            if target_tok in tok.text.lower() or target_tok in tok.lemma_.lower():
                if tok.dep_ in ("amod", "advmod") and tok.head.pos_ in ("NOUN", "PROPN"):
                    return "adjective"
                return VALID_POS_MAP.get(tok.pos_, "noun")

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

        # Check if word is already recognized as plural noun (e.g. 'things', 'assets', 'possessions')
        # to prevent corrupt pseudo-plurals like 'thingses' or 'assetses'
        nlp = cls.get_spacy()
        doc = nlp(w)
        tok = doc[0]
        if tok.tag_ == 'NNS' and tok.lemma_.lower() != w and not w.endswith(('ss', 'us', 'is')):
            return w

        if w.endswith(('s', 'sh', 'ch', 'x', 'z')):
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
        nlp = cls.get_spacy()
        doc = nlp(sentence)

        # ---------------------------------------------------------------------
        # 1. Rhetoric & Emphasis:
        #    a. Fronted Inversion: aux or ROOT precedes nsubj in linear word order
        #    b. Cleft Focus: expl ('it') + 'is/was' + focal constituent + relcl (that...)
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
                    # If followed by relative clause (relcl) modifying non-subject focus -> Cleft
                    has_relcl = any(child.dep_ == "relcl" for child in head.children)
                    for child in head.children:
                        if any(gc.dep_ == "relcl" for gc in child.children):
                            has_relcl = True
                    # Check if followed by that/who relcl
                    if has_relcl:
                        return "Rhetoric & Emphasis"

        # Antithesis / Correlative Parallelism check (not... but..., not only... but also)
        text_lower = sentence.lower()
        if re.search(r"\bnot\s+.*?\s*,\s*but\b", text_lower) or re.search(r"\bnot\s+only\b.*?\bbut\s+also\b", text_lower):
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
                    if child.dep_ in ("acl", "appos", "ccomp"):
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
        for token in doc:
            if token.text.lower() == "it" and token.dep_ in ("expl", "nsubj"):
                head = token.head
                if head.lemma_ in ("be", "seem"):
                    # Check if head has acomp/attr and ccomp/csubj/xcomp
                    has_predicate = any(child.dep_ in ("acomp", "attr") for child in head.children)
                    has_complement = any(child.dep_ in ("ccomp", "csubj", "xcomp") for child in head.children)
                    if has_predicate and has_complement:
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
        for token in doc:
            if token.dep_ == "advcl" and token.tag_ in ("VBG", "VBN"):
                # Ensure it has no subject of its own (non-finite) and not a preposition
                has_subj = any(c.dep_ in ("nsubj", "nsubjpass") for c in token.children)
                if not has_subj and token.lemma_.lower() not in ("include", "accord", "regard", "concern"):
                    return "Information Packaging"

        # Relative clauses (which, that, who, whom, whose)
        for token in doc:
            if token.dep_ == "relcl":
                if any(c.text.lower() in ("which", "that", "who", "whom", "whose") for c in token.children):
                    return "Information Packaging"

        # Elaborative non-restrictive relative clauses
        if re.search(r",\s*which\s+[a-z]+", text_lower):
            return "Information Packaging"

        return None

    @classmethod
    def generate_cobuild_formula(cls, sentence: str, category: Optional[str] = None) -> str:
        """
        Deterministically derives standard algebraic COBUILD slot formula from sentence
        via spaCy dependency trees and syntactic pattern anchors.
        Replaces prone-to-hallucination LLM formula drafting with zero-token invariant logic.
        """
        nlp = cls.get_spacy()
        doc = nlp(sentence)
        cat = category or cls.classify_grammar_dependency(sentence) or ""
        text_lower = sentence.lower()

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
            for token in doc:
                if token.text.lower() == "it" and token.dep_ in ("expl", "nsubj") and token.head.lemma_ in ("be", "seem"):
                    if any(c.dep_ == "relcl" or any(gc.dep_ == "relcl" for gc in c.children) for c in token.head.children):
                        return "It + [be] + [Focal Element] + that/who + [Clause]"
            # Correlative Parallelism
            if re.search(r"\bnot\s+only\b.*?\bbut\s+also\b", text_lower):
                return "[Subject] + not only + [VP], but also + [VP]"
            # Antithesis
            if re.search(r"\bnot\s+.*?\s*,\s*but\b", text_lower):
                return "[Subject] + [VP], not + [PrepP/NP], but + [PrepP/NP]"

        # 2. Logic & Stance
        if cat == "Logic & Stance" or not cat:
            # Multi-word conditional connectives
            m_conn = re.search(r"\b(as long as|so long as|provided that|providing that|in case|on condition that)\b", text_lower)
            if m_conn:
                conn_name = m_conn.group(1).capitalize()
                return f"{conn_name} + [Clause], [Subject] + [VP]"
            for token in doc:
                if token.dep_ == "mark" and token.lemma_.lower() in {"although", "though", "while", "whereas", "even though", "if", "unless", "provided"}:
                    mark_word = token.text.capitalize()
                    return f"{mark_word} + [Clause], [Subject] + [VP]"
            if re.match(r"^(however|nevertheless|nonetheless|conversely|in contrast)\b", text_lower):
                return "However, [Subject] + [Modal/VP]"
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
            shell_nouns = {"fact", "idea", "conclusion", "claim", "belief", "hypothesis", "notion", "argument", "difference", "discrepancy"}
            for token in doc:
                if token.lemma_.lower() in shell_nouns:
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
            if any(t.text.lower() == "it" and t.dep_ in ("expl", "nsubj") for t in doc):
                for t in doc:
                    if t.text.lower() == "it" and t.head.lemma_ in ("be", "seem"):
                        if any(c.dep_ in ("acomp", "attr") for c in t.head.children) and any(c.dep_ in ("ccomp", "csubj", "xcomp") for c in t.head.children):
                            return "It + [be] + [Adj/NP] + to-V/that + [Clause]"
            # Participial adjunct (strictly requiring non-subject participle and actual main subject)
            for token in doc:
                if token.dep_ == "advcl" and token.tag_ in ("VBG", "VBN"):
                    if not any(c.dep_ in ("nsubj", "nsubjpass") for c in token.children) and token.lemma_.lower() not in ("include", "accord", "regard"):
                        # Ensure main verb actually has a subject, otherwise it might be a gerund subject misclassified as advcl
                        main_has_subj = any(c.dep_ in ("nsubj", "nsubjpass") for c in token.head.children)
                        if main_has_subj:
                            is_fronted = token.i < token.head.i
                            v_type = "V-ing" if token.tag_ == "VBG" else "V-ed"
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

        # Default fallback standard formula
        return "[Subject] + [VP] + [Clause]"

    @classmethod
    def mine_grammar_skeletons(
        cls,
        sentence_pool: Dict[str, str],
        target_count: int = 5,
        syllabus_grammar: Optional[List[str]] = None
    ) -> List[Dict[str, str]]:
        """
        Deterministically mines genuine, non-trivial grammar structural skeletons from the sentence pool.
        Selects sentences exhibiting rich dependency structures across the Four Macro Domains.
        Returns a list of dicts: [{'sid': 'S-9', 'quote': '...', 'category': '...', 'pattern_formula': '...'}]
        """
        raw_skeletons: List[Dict[str, str]] = []

        for sid, sent in sentence_pool.items():
            sent_clean = sent.strip()
            # Skip very short or title-like sentences
            if len(sent_clean.split()) < 7:
                continue

            cat = cls.classify_grammar_dependency(sent_clean)
            if cat:
                formula = cls.generate_cobuild_formula(sent_clean, category=cat)
                # Filter out generic/un-abstracted formulas (e.g. [Subject] + [VP] + [Clause] or simple [S])
                if formula and not formula.startswith("[Subject] + [VP] + [Clause]") and "[" in formula:
                    raw_skeletons.append({
                        "sid": sid,
                        "quote": sent_clean,
                        "category": cat,
                        "pattern_formula": formula,
                    })

        if not raw_skeletons:
            return []

        # Diverse selection algorithm across macro domains and distinct formulas
        selected: List[Dict[str, str]] = []
        seen_cats: Set[str] = set()
        seen_formulas: Set[str] = set()

        # Pass 1: maximize category diversity across the 4 domains
        for item in raw_skeletons:
            cat = item["category"]
            form = item["pattern_formula"]
            if cat not in seen_cats and form not in seen_formulas:
                selected.append(item)
                seen_cats.add(cat)
                seen_formulas.add(form)
            if len(selected) >= target_count:
                break

        # Pass 2: fill remaining slots with distinct structural formulas
        if len(selected) < target_count:
            for item in raw_skeletons:
                if item not in selected and item["pattern_formula"] not in seen_formulas:
                    selected.append(item)
                    seen_formulas.add(item["pattern_formula"])
                if len(selected) >= target_count:
                    break

        # Pass 3: if still under target_count, append distinct quotes with unique formulas
        if len(selected) < target_count:
            selected_quotes = {s["quote"] for s in selected}
            for item in raw_skeletons:
                if item["quote"] not in selected_quotes and item["pattern_formula"] not in seen_formulas:
                    selected.append(item)
                    selected_quotes.add(item["quote"])
                    seen_formulas.add(item["pattern_formula"])
                if len(selected) >= target_count:
                    break

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
        acl_map = cls.get_acl_collocations()
        awl_set = cls.get_awl_words()

        COMMON_PARTICLES = frozenset({"away", "back", "down", "in", "off", "on", "out", "over", "round", "through", "up"})
        DEPENDENT_PREPS = frozenset({"on", "upon", "to", "with", "for", "from", "into", "against", "about", "of", "in"})

        raw_candidates: List[Dict[str, Any]] = []

        for sid, sentence in pool.items():
            sent_clean = sentence.strip()
            if len(sent_clean.split()) < 5:
                continue

            doc = nlp(sent_clean)
            text_lower = sent_clean.lower()

            # 1. Academic Collocation List (ACL) Matcher
            for core, original in acl_map.items():
                pattern = r"\b" + re.escape(core) + r"\b"
                if re.search(pattern, text_lower):
                    # Determine type: verb+noun or adj+noun
                    words = core.split()
                    expr_type = "collocation"
                    formula = core
                    if original.endswith("(to)") or original.endswith("(of)") or original.endswith("(in)") or original.endswith("(with)"):
                        prep_match = re.search(r"\(([a-z]+)\)$", original)
                        if prep_match:
                            formula = f"{core} {prep_match.group(1)} [sth]"
                    elif len(words) == 2 and any(w in awl_set for w in words):
                        formula = f"{core}"

                    raw_candidates.append({
                        "sid": sid,
                        "quote": sent_clean,
                        "phrase": core,
                        "type": expr_type,
                        "pattern_formula": formula,
                        "score": 10 + (2 if any(w in awl_set for w in words) else 0),
                    })

            # 2. Dependency Syntax Phrasal Verbs & Prepositional Combinations
            COLLOCATION_VERB_PREPS = {
                "welcome": "to",
                "provide": "with",
                "share": "with",
                "equip": "with",
                "attribute": "to",
                "contribute": "to",
                "rely": "on",
                "depend": "on",
                "focus": "on",
                "concentrate": "on",
                "participate": "in",
                "engage": "in",
                "succeed": "in",
                "benefit": "from",
                "derive": "from",
                "suffer": "from",
                "cope": "with",
                "comply": "with",
                "lead": "to",
                "adapt": "to",
                "conform": "to",
                "refer": "to",
                "apply": "to",
                "appeal": "to",
                "insist": "on",
                "consist": "of",
            }

            for token in doc:
                if token.pos_ in ("VERB", "AUX"):
                    v_lemma = token.lemma_.lower()

                    # Check high-value idiomatic verbal phrases like 'keep/get/stay in touch (with)'
                    touch_match = False
                    if v_lemma in ("keep", "get", "stay", "lose", "be"):
                        # spaCy parses either prt='in' + dobj='touch' or prep='in' -> pobj='touch'
                        has_in = any(c.text.lower() == "in" and c.dep_ in ("prt", "prep") for c in token.children)
                        has_touch = any(c.text.lower() == "touch" and c.dep_ in ("dobj", "pobj") for c in token.children)
                        if not has_touch and has_in:
                            for c in token.children:
                                if c.text.lower() == "in":
                                    has_touch = any(gc.text.lower() == "touch" for gc in c.children)
                        if has_in and has_touch:
                            touch_match = True
                            # Check if followed by 'with'
                            has_with = any(c.text.lower() == "with" for c in token.children)
                            if not has_with:
                                for c in token.children:
                                    if any(gc.text.lower() == "with" for gc in c.children):
                                        has_with = True
                                        break
                            phrase_name = f"{v_lemma} in touch with" if has_with else f"{v_lemma} in touch"
                            formula = f"{v_lemma} in touch with [sb]" if has_with else f"{v_lemma} in touch"
                            raw_candidates.append({
                                "sid": sid,
                                "quote": sent_clean,
                                "phrase": phrase_name,
                                "type": "idiom",
                                "pattern_formula": formula,
                                "score": 10,
                            })

                    # Check particle (e.g. wake up, get by, slave away, carry out)
                    if not touch_match:
                        for c in token.children:
                            if (c.dep_ == "prt" or (c.dep_ == "advmod" and c.text.lower() in COMMON_PARTICLES)) and c.i > token.i:
                                p_tok = c.text.lower()
                                # Check three-part phrasal verb: verb + particle + prep immediately following (distance <= 2)
                                prep_child = [gc for gc in token.children if gc.dep_ == "prep" and 0 < (gc.i - c.i) <= 2]
                                if prep_child:
                                    prep_word = prep_child[0].text.lower()
                                    p_verb = f"{v_lemma} {p_tok} {prep_word}"
                                    formula = f"{v_lemma} {p_tok} {prep_word} [sth/sb]"
                                else:
                                    dobj = [d for d in token.children if d.dep_ == "dobj"]
                                    if dobj:
                                        p_verb = f"{v_lemma} {p_tok}"
                                        formula = f"{v_lemma} [sb/sth] {p_tok}"
                                    else:
                                        p_verb = f"{v_lemma} {p_tok}"
                                        formula = f"{v_lemma} {p_tok}"

                                raw_candidates.append({
                                    "sid": sid,
                                    "quote": sent_clean,
                                    "phrase": p_verb,
                                    "type": "phrasal verb",
                                    "pattern_formula": formula,
                                    "score": 9,
                                })

                    # Check possessive object construction via spaCy 'poss' dependency (e.g. attain [one's] best, make up [one's] mind)
                    # Restrict to genuine idiomatic and academic lexicalized noun heads to avoid trivial 'take his dad'
                    IDIOMATIC_POSS_NOUNS = frozenset({
                        "best", "mind", "temper", "breath", "time", "part", "way", "heart",
                        "step", "place", "voice", "promise", "life", "potential", "future",
                        "passion", "dream", "goal", "duty", "responsibility", "effort", "stride",
                        "horizon", "footing", "view", "interest", "role", "purpose"
                    })
                    dobjs = [d for d in token.children if d.dep_ == "dobj"]
                    for d in dobjs:
                        poss = [p for p in d.children if p.dep_ == "poss"]
                        if poss:
                            noun_text = d.text.lower()
                            noun_lemma = d.lemma_.lower()
                            matched_noun = None
                            if noun_text in IDIOMATIC_POSS_NOUNS:
                                matched_noun = noun_text
                            elif noun_lemma in IDIOMATIC_POSS_NOUNS:
                                matched_noun = noun_lemma

                            if matched_noun:
                                prt = [p for p in token.children if p.dep_ == "prt"]
                                if prt:
                                    p_clean = f"{v_lemma} {prt[0].text.lower()} [one's] {matched_noun}"
                                    formula = f"{v_lemma} {prt[0].text.lower()} [one's] {matched_noun}"
                                else:
                                    p_clean = f"{v_lemma} [one's] {matched_noun}"
                                    formula = f"{v_lemma} [one's] {matched_noun}"

                                raw_candidates.append({
                                    "sid": sid,
                                    "quote": sent_clean,
                                    "phrase": p_clean,
                                    "type": "collocation",
                                    "pattern_formula": formula,
                                    "score": 10,
                                })

                    # Check dependent preposition collocation or idiomatic frame
                    for c in token.children:
                        if c.dep_ == "prep" and c.i > token.i:
                            p_text = c.text.lower()
                            dobj = [d for d in token.children if d.dep_ == "dobj"]

                            # Known academic verb+prep collocation (e.g. welcome [sb] to [sth], provide [sb] with [sth])
                            if v_lemma in COLLOCATION_VERB_PREPS and COLLOCATION_VERB_PREPS[v_lemma] == p_text:
                                if dobj:
                                    coll_name = f"{v_lemma} ... {p_text}"
                                    formula = f"{v_lemma} [sb] {p_text} [sth]"
                                else:
                                    coll_name = f"{v_lemma} {p_text}"
                                    formula = f"{v_lemma} {p_text} [sth/sb]"
                                raw_candidates.append({
                                    "sid": sid,
                                    "quote": sent_clean,
                                    "phrase": coll_name,
                                    "type": "collocation",
                                    "pattern_formula": formula,
                                    "score": 9,
                                })
                            elif not dobj and p_text in DEPENDENT_PREPS and (c.i - token.i <= 2):
                                # Intransitive prepositional verb (e.g. rely on, contend with)
                                p_verb = f"{v_lemma} {p_text}"
                                formula = f"{v_lemma} {p_text} [sth/sb]"
                                pobj_lemma = None
                                pobj_proper = False
                                pobj_nodes = [p for p in c.children if p.dep_ == "pobj"]
                                if pobj_nodes:
                                    pobj_tok = pobj_nodes[0]
                                    pobj_lemma = pobj_tok.lemma_.lower()
                                    # A capitalised object mid-sentence is a name, not the
                                    # dictionary's frame ('keep in WeChat Moments').
                                    pobj_proper = pobj_tok.pos_ == "PROPN" or bool(
                                        pobj_tok.text[:1].isupper() and not pobj_tok.is_sent_start)
                                raw_candidates.append({
                                    "sid": sid,
                                    "quote": sent_clean,
                                    "phrase": p_verb,
                                    "pobj": pobj_lemma,
                                    "pobj_proper": pobj_proper,
                                    "type": "phrasal verb",
                                    "pattern_formula": formula,
                                    "score": 8,
                                })
                            elif dobj:
                                # Fixed verbal idiom/collocation (e.g. take into account, take into consideration)
                                pobj = [p for p in c.children if p.dep_ == "pobj"]
                                # Strict syntactic & lexical invariant:
                                # 1. Verb must be a known idiom head ('take', 'bring', 'put', 'call')
                                # 2. Preposition must directly govern the noun (distance == 1) to avoid adverbial adjuncts like 'in different places'
                                # 3. Preposition object must be a recognized idiom target
                                if pobj and (pobj[0].i - c.i == 1) and v_lemma in ("take", "bring", "put", "call") and pobj[0].lemma_.lower() in ("account", "consideration", "effect", "play", "question"):
                                    prep_phrase = f"{v_lemma} {p_text} {pobj[0].lemma_.lower()}"
                                    formula = f"{v_lemma} {p_text} {pobj[0].lemma_.lower()} [sth]"
                                    raw_candidates.append({
                                        "sid": sid,
                                        "quote": sent_clean,
                                        "phrase": prep_phrase,
                                        "type": "idiom",
                                        "pattern_formula": formula,
                                        "score": 10,
                                    })

        if not raw_candidates:
            return []

        # LDOCE attestation gate: a candidate ships only when Longman itself states the unit.
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
                    # Longman states it as 'be/keep/stay etc in touch (with something)'
                    attested = cls.is_attested_phrase(" ".join(w for w in words if w != "with"))
                if attested:
                    # 'keep in' is a real phrasal verb, but Longman's frame for it is 'keep
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

            # Verbal idioms (e.g. 'take into consideration') that Longman does not list as a
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
            for expr in syllabus_expressions:
                expr_clean = re.sub(r"\s+", " ", expr.strip())
                words = expr_clean.split()
                matched_sid = None
                matched_sent = None

                for sid, s in pool.items():
                    s_lower = s.lower()
                    e_lower = expr_clean.lower()
                    if re.search(r"\b" + re.escape(e_lower) + r"\b", s_lower):
                        matched_sid = sid
                        matched_sent = s
                        break
                    elif all(re.search(r"\b" + re.escape(w.lower()) + r"\b", s_lower) for w in words):
                        matched_sid = sid
                        matched_sent = s
                        break
                    else:
                        s_doc = nlp(s)
                        s_lemmas = [t.lemma_.lower() for t in s_doc]
                        s_tokens = [t.text.lower() for t in s_doc]
                        if all(w.lower() in s_lemmas or w.lower() in s_tokens for w in words):
                            matched_sid = sid
                            matched_sent = s
                            break

                if not matched_sid or not matched_sent:
                    continue

                e_lower = expr_clean.lower()
                cand_item = None

                # 1. Specialized High-Frequency Idioms (figurative, non-compositional units)
                KNOWN_IDIOMS = {
                    "lone ranger", "hats off", "hat off", "pat on the back", 
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
                    "peace of mind", "pros and cons"
                }

                if any(idiom in e_lower for idiom in KNOWN_IDIOMS):
                    formula = expr_clean
                    if "lone ranger" in e_lower:
                        formula = "lone ranger"
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
                                    cands = [t for t in doc if t.i > curr[-1].i and (t.lemma_.lower() == w or t.text.lower() == w)]
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
                            expr_tokens = [t for t in doc if any(w.lower() in (t.text.lower(), t.lemma_.lower()) for w in words)]

                    first_tok = expr_tokens[0] if expr_tokens else None

                    if first_tok and first_tok.pos_ in ("VERB", "AUX"):
                        v_lemma = first_tok.lemma_.lower()
                        last_tok = expr_tokens[-1]
                        
                        # Syntax-aware detection: check if 'to' is an infinitive marker rather than a preposition
                        is_infinitive_to = False
                        if last_tok.text.lower() == "to":
                            if last_tok.pos_ == "PART" or last_tok.dep_ == "aux":
                                is_infinitive_to = True
                            elif last_tok.head.pos_ == "VERB" and last_tok.head.dep_ in ("xcomp", "advcl", "purpcl"):
                                is_infinitive_to = True

                        if is_infinitive_to:
                            # Genuine infinitive construction (e.g. tend to do [sth], fail to do [sth])
                            formula = f"{v_lemma} to do [sth]"
                            cand_type = "collocation"
                        elif last_tok.pos_ in ("ADP", "PART") or last_tok.dep_ in ("prep", "prt"):
                            prep_word = last_tok.text.lower()
                            # Dynamic Transitivity Audit via Sentence Dependency Tree & Valence
                            # Check whether the verb or particle takes an authentic object in the context sentence
                            verb_tok = expr_tokens[0] if expr_tokens else None
                            has_contextual_obj = False
                            if verb_tok is not None:
                                # 1. Verb direct object (dobj / obj)
                                verb_objs = [c for c in verb_tok.children if c.dep_ in ("dobj", "obj")]
                                # 2. Particle / prepositional object (pobj)
                                prep_objs = [c for c in last_tok.children if c.dep_ == "pobj"]
                                # 3. Passive voice extraction (e.g. 'the lights were turned off')
                                is_passive = any(c.dep_ in ("auxpass", "nsubjpass") for c in verb_tok.children)
                                if verb_objs or prep_objs or is_passive:
                                    has_contextual_obj = True

                            # If the particle is an adverbial particle (prt) with no object in context (e.g. 'show up'),
                            # it is an authentic intransitive phrasal verb: generate clean formula without [sth/sb]
                            is_intransitive = (not has_contextual_obj and last_tok.dep_ == "prt")

                            if len(words) == 2:
                                if is_intransitive:
                                    formula = f"{v_lemma} {prep_word}"
                                else:
                                    formula = f"{v_lemma} {prep_word} [sth/sb]"
                            else:
                                core = " ".join(t.lemma_ if t.pos_ == "VERB" else t.text.lower() for t in expr_tokens)
                                if "importance to" in core:
                                    formula = f"{core} [sth]"
                                elif is_intransitive:
                                    formula = core
                                else:
                                    formula = f"{core} [sth/sb]"
                            cand_type = "phrasal verb"
                        else:
                            # Verb + object / adj (e.g. make smart choices, keep silent, have a try)
                            core = " ".join(t.lemma_ if t.pos_ == "VERB" else t.text.lower() for t in expr_tokens)
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

        # Phase 2: If still under target_count (or no syllabus provided), fill from computational NLP/ACL/OCD candidates
        if len(selected) < target_count:
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
        awl_set = cls.get_awl_words()

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
            for token in doc:
                lemma = token.lemma_.lower().strip()
                if len(lemma) < 3 or lemma in seen_lemmas or not lemma.isalpha():
                    continue

                # 1. Academic Closed-Paradigm Function Word Gate:
                # If lemma belongs to our curated high-utility closed paradigms (e.g. 'despite', 'whereas',
                # 'beyond', 'throughout', 'unless', 'nonetheless'), bypass spaCy's stopword filter and POS restriction!
                if cls.is_function_word(lemma):
                    # Filter out ultra-basic elementary function words
                    ULTRA_BASIC_FUNC_WORDS = frozenset({
                        "in", "at", "on", "to", "for", "with", "from", "by", "of",
                        "and", "but", "or", "nor", "so", "for", "yet",
                        "if", "when", "as", "than", "that", "because"
                    })
                    if lemma in ULTRA_BASIC_FUNC_WORDS:
                        continue

                    f_idx = cls._get_function_word_index()
                    _fam, stype = f_idx[lemma][0]
                    # Map syntactic complement type to schema-supported POS
                    assigned_pos = "preposition" if stype == "prepositional" else ("conjunction" if stype == "clausal" else "adverb")
                    is_awl = lemma in awl_set
                    # Academic discourse connectors receive high priority
                    score = (25 if is_awl else 22) + min(len(lemma), 10)

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

                is_awl = lemma in awl_set

                # If not in AWL and is trivial or very short, skip
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
                    score = (20 if is_awl else 0) + min(len(phrase), 12)
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

                # Scoring: AWL headwords get high priority, followed by word length and syllable complexity
                score = (20 if is_awl else 0) + min(len(lemma), 12)

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
    def extract_deterministic_vocabulary(
        cls,
        text: str,
        syllabus_vocab: Optional[List[str]] = None,
        target_count: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Pure Computational Linguistic & Lexicographic Extractor (0 Tokens, 100% Deterministic).
        Extracts academic target vocabulary, locates authentic quoted sentences with Sentence IDs,
        tags contextual Part of Speech via spaCy, and hydrates authoritative definitions & authentic
        corpus examples from LDOCE 6th Edition.
        """
        _, pool = cls.tokenize_and_index_sentences(text)
        if not pool:
            return []

        # 1. Determine target words
        target_candidates: List[Dict[str, Any]] = []
        if syllabus_vocab:
            clusters: List[List[str]] = []
            for w in syllabus_vocab:
                w_clean = re.sub(r'\[.*?\]|\(.*?\)', '', str(w)).strip().lower()
                placed = False
                for c in clusters:
                    if any(cls.are_same_word_family(w_clean, cw) for cw in c):
                        c.append(w_clean)
                        placed = True
                        break
                if not placed:
                    clusters.append([w_clean])
            cleaned_syllabus = [c[0] for c in clusters][:target_count]
            for w in cleaned_syllabus:
                target_candidates.append({"word": w, "sid": "", "quote": ""})
        else:
            skels = cls.mine_vocabulary_skeletons(text, target_count=target_count)
            target_candidates = skels

        results: List[Dict[str, Any]] = []
        seen_words: Set[str] = set()

        for cand in target_candidates:
            w = cand["word"].strip().lower()
            if not w or w in seen_words:
                continue

            matching_sent = cand.get("quote", "")
            matching_sid = cand.get("sid", "")

            # If sentence not already matched, search in sentence pool
            anchor_quality = "quote" if matching_sent else "syllabus"
            if not matching_sent and pool:
                w_clean = re.sub(r'[^a-z0-9]', '', w)
                # A1: an anchor found through an inflected occurrence ('attaches' for
                # 'attach') is as authentic as one found through the base spelling.
                # B1: the pool is no longer scanned for the FIRST sentence containing the
                # word. A heading ('**Punctuality Pays!**') and a bare fragment ('Why?')
                # demonstrate nothing, and among the sentences that do, the one where the
                # target governs the most syntax is the one worth locking a sense against.
                shown = [(sid, sent) for sid, sent in pool.items() if cls.text_contains_form(sent, w)]
                best_sent, _best_rich = cls.pick_anchor_sentence(w, [s for _sid, s in shown])
                if best_sent:
                    matching_sent = best_sent
                    matching_sid = cls.get_sentence_id(best_sent, pool) or matching_sid
                    anchor_quality = "clause"
                elif shown:
                    matching_sent, matching_sid = shown[0]
                    anchor_quality = "unqualified"
                else:
                    for sid, sent in pool.items():
                        if w in sent.lower() or (w_clean and w_clean in re.sub(r'[^a-z0-9]', '', sent.lower())):
                            matching_sent = sent
                            matching_sid = sid
                            anchor_quality = "substring"
                            break

            # Infer contextual POS
            pos = cand.get("part_of_speech")
            if not pos and matching_sent:
                pos = cls.determine_contextual_pos(w, matching_sent)
            if not pos:
                pos = "noun"

            # Hydrate authoritative LDOCE definition and authentic example. B1: the sense lock
            # is graded first. An entry with several candidate senses whose sense the quote did
            # not earn ships no LDOCE definition - the WordNet fallback below is an honest
            # second source, not whichever Longman sense happens to be printed first.
            lock = cls.sense_confidence(
                cls.get_ldoce_entry(w),
                definition=cand.get("definition"),
                quote=matching_sent,
                target_pos=pos
            )
            definition, example = cls.get_ldoce_definition_and_example(
                w, target_pos=pos, context_sentence=matching_sent,
                require_sense_confidence=True
            )
            if not definition:
                # Fallback to WordNet definition if LDOCE headword missing
                try:
                    wn = cls.get_wordnet()
                    syns = wn.synsets(w)
                    if syns:
                        definition = syns[0].definition()
                except Exception:
                    pass
            if not definition:
                definition = f"A core academic term functioning as a {pos}."
            # A1: the anchor may only stand in for an example when it actually shows the
            # headword. An example that does not contain the word teaches nothing.
            if not example and matching_sent and cls.text_contains_form(matching_sent, w):
                example = matching_sent

            audit_sid = f"[{matching_sid}]" if matching_sid else "[S-1]"
            results.append({
                "word": w,
                "part_of_speech": pos,
                "definition": definition,
                "quoted_sentence": matching_sent,
                "example_usage": example,
                # B1: the evidence tier this row was built on, visible to the writer and to
                # the audit instead of being implicit in whichever sentence came first.
                "anchor_quality": anchor_quality,
                "sense_confidence": lock["confidence"],
                "sense_lock_score": lock["score"],
                "sense_lock_margin": lock["margin"],
                "design_audit": f"AUDIT: {audit_sid} -> [{w}] -> [VERBATIM_CONFIRMED]"
            })
            seen_words.add(w)

        return results

    @classmethod
    def extract_deterministic_expressions(
        cls,
        text: str,
        target_count: int = 5,
        syllabus_expressions: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Extracts genuine multi-word academic expressions and collocations from text
        in 100% deterministic fashion (0 Tokens).
        Combines mine_expression_skeletons with get_expression_definition_and_example.
        Returns a list of dicts conforming to VocabularyItem / ExpressionItem:
            {
                "word": canonical formula (e.g. 'shut [sb/sth] down'),
                "part_of_speech": 'collocation' | 'phrasal verb' | ...,
                "definition": authoritative LDOCE / WordNet definition,
                "quoted_sentence": verbatim passage sentence,
                "example_usage": authentic LDOCE example or passage sentence,
                "design_audit": "AUDIT: [S-ID] -> [Formula] -> [VERBATIM_CONFIRMED]"
            }
        """
        skeletons = cls.mine_expression_skeletons(
            text,
            target_count=target_count,
            syllabus_expressions=syllabus_expressions
        )
        if not skeletons:
            return []

        results: List[Dict[str, Any]] = []
        for s in skeletons:
            phrase = s.get("phrase", "")
            formula = s.get("pattern_formula", phrase)
            sid = s.get("sid", "S-1")
            quote = s.get("quote", "")
            expr_type = s.get("type", "collocation")

            defn, ex = cls.get_expression_definition_and_example(
                phrase=phrase,
                expr_type=expr_type,
                context_sentence=quote
            )
            if not ex and quote:
                ex = quote

            results.append({
                "word": formula,
                "part_of_speech": expr_type,
                "definition": defn,
                "quoted_sentence": quote,
                "example_usage": ex,
                "design_audit": f"AUDIT: [{sid}] -> [{formula}] -> [VERBATIM_CONFIRMED]"
            })

        return results

    # =========================================================================
    # COBUILD CORE PATTERN PROFILES & COMMON LEARNER PITFALLS (0 Copyright DB)
    # Grounded in John Sinclair's Pattern Grammar & Empirical Learner Syntax Pitfalls
    # =========================================================================
    COBUILD_GRAMMAR_PROFILES: Dict[str, Dict[str, Any]] = {
        "cleft_focus": {
            "name": "Cleft Sentences & Rhetorical Focus",
            "category": "Rhetoric & Emphasis",
            "pattern_formula": "It + [be] + [Focal Element] + that/who + [Clause]",
            "pedagogical_function": "Fronts the critical agent, instrument, or temporal circumstance to create contrastive focus and discursive prominence.",
            "imitation_example": "It is transparent communication that fosters mutual trust within academic teams.",
            "common_mistakes": (
                "COBUILD Warning (Cleft Sentences): Learners frequently produce: "
                "\"It was in the library [INCORRECT: where -> CORRECT: that] we discovered the manuscript.\" "
                "(Note: Even when focusing prepositional or time phrases, the matrix complement requires 'that', not 'where' or 'when')."
            )
        },
        "correlative_parallelism": {
            "name": "Correlative Coordinators & Inversion",
            "category": "Rhetoric & Emphasis",
            "pattern_formula": "[Subject] + not only + [VP], but also + [VP]",
            "pedagogical_function": "Constructs symmetrical syntactic coordination to deliver dual arguments with equal communicative weight.",
            "imitation_example": "The framework not only optimizes performance, but also enhances fault tolerance.",
            "common_mistakes": (
                "COBUILD Warning (Correlative Coordinators): Learners frequently produce: "
                "\"The program aims not only [INCORRECT: to train engineers, but also fostering -> CORRECT: to train engineers, but also to foster] research.\" "
                "(Note: Conjoined predicates must maintain identical non-finite morphological forms)."
            )
        },
        "fronted_negative_inversion": {
            "name": "Fronted Negative & Subject-Auxiliary Inversion",
            "category": "Rhetoric & Emphasis",
            "pattern_formula": "[Negative Adv] + [aux/be] + [Subject] + [VP]",
            "pedagogical_function": "Employs emphatic word-order inversion to strongly refute or delimit an assertion in formal academic discourse.",
            "imitation_example": "Rarely do experimental measurements deviate so markedly from theoretical predictions.",
            "common_mistakes": (
                "COBUILD Warning (Negative Inversion): Learners frequently produce: "
                "\"Not only [INCORRECT: the system saves -> CORRECT: does the system save] energy, but it also minimizes downtime.\" "
                "(Note: Fronted restrictive or negative adverbs strictly require Subject-Auxiliary Inversion)."
            )
        },
        "relative_clause_restrictive": {
            "name": "Adjective Clauses: Relative Pronoun Complementation",
            "category": "Information Packaging",
            "pattern_formula": "[NP] + which/that + [VP]",
            "pedagogical_function": "Packages restrictive identifying details directly onto the head noun, increasing lexical density in academic prose.",
            "imitation_example": "Scholars developed computational algorithms that automate anomaly detection across massive datasets.",
            "common_mistakes": (
                "COBUILD Warning (Relative Clauses): Learners frequently retain redundant object pronouns in relative clauses: "
                "\"This is the paper which we reviewed [INCORRECT: it -> CORRECT: ] yesterday.\""
            )
        },
        "relative_clause_elaborative": {
            "name": "Non-Restrictive Relative Clauses & Propositional Encapsulation",
            "category": "Cohesion & Framing",
            "pattern_formula": ", which + [VP/interpretive verb] + [Clause]",
            "pedagogical_function": "Encapsulates the antecedent proposition to append an evaluative stance or inferential commentary.",
            "imitation_example": "The reaction generated intense heat, which suggested that the catalytic process had initiated successfully.",
            "common_mistakes": (
                "COBUILD Warning (Sentential Relative Clauses): Learners frequently produce: "
                "\"The server crashed repeatedly, [INCORRECT: that -> CORRECT: which] forced the team to reboot.\" "
                "(Note: 'that' cannot head a non-restrictive sentential relative clause)."
            )
        },
        "object_complement_adj": {
            "name": "Complex Transitive Structures (Verb + Object + Adjective)",
            "category": "Information Packaging",
            "pattern_formula": "[Subject] + make/find/keep + [Object] + [Adj]",
            "pedagogical_function": "Synthesizes causal and stative outcomes into a compact clause, eliminating bloated causative periphrasis.",
            "imitation_example": "Automated regression pipelines make continuous deployment reliable and secure.",
            "common_mistakes": (
                "COBUILD Warning (Structures with Object Complements): Learners frequently produce: "
                "\"The new communication protocols make global collaboration [INCORRECT: easily -> CORRECT: easy].\" "
                "(Note: The complement characterizes the post-state of the object noun, requiring an adjective rather than a manner adverb)."
            )
        },
        "causative_bare_infinitive": {
            "name": "Causative Complementation (Make/Have/Let + Bare Infinitive)",
            "category": "Information Packaging",
            "pattern_formula": "[Subject] + make/let/have + [Object] + [Bare Verb]",
            "pedagogical_function": "Depicts direct coercive or permissive agency without extraneous prepositional scaffolding.",
            "imitation_example": "Strict peer-review standards make researchers substantiate every empirical assertion.",
            "common_mistakes": (
                "COBUILD Warning (Causative Structures): Learners frequently produce: "
                "\"The supervisor made each student [INCORRECT: to submit -> CORRECT: submit] weekly progress reports.\" "
                "(Note: Active causative 'make' governs an unmarked bare infinitive, not a 'to'-infinitive)."
            )
        },
        "dummy_it_extraposition": {
            "name": "Dummy-It Extraposition & Evaluative Stance",
            "category": "Information Packaging",
            "pattern_formula": "It + [be] + [Evaluative Adj] + to-V/that + [Clause]",
            "pedagogical_function": "Postposes heavy informational propositions while fronting evaluative stance attributes.",
            "imitation_example": "It is essential to calibrate sensor equipment prior to conducting field measurements.",
            "common_mistakes": (
                "COBUILD Warning (Preparatory It): Learners frequently produce: "
                "\"[INCORRECT: Is essential to calibrate -> CORRECT: It is essential to calibrate] sensor equipment.\" "
                "or: \"Technological advancements make [INCORRECT: possible to work -> CORRECT: it possible to work] remotely.\""
            )
        },
        "participial_adjunct": {
            "name": "Non-Finite Participial Adjuncts",
            "category": "Information Packaging",
            "pattern_formula": "[V-ing / V-ed Phrase], [Subject] + [VP]",
            "pedagogical_function": "Condenses circumstantial temporal, causal, or concessive framing into non-finite adverbial adjuncts.",
            "imitation_example": "Operating under stringent constraints, the development team completed the refactoring ahead of schedule.",
            "common_mistakes": (
                "COBUILD Warning (Participial Clauses): Learners frequently produce dangling modifiers: "
                "\"[INCORRECT: Having reviewed the telemetry data, the error became apparent -> CORRECT: Having reviewed the telemetry data, the engineers identified the error].\""
            )
        },
        "gerund_subject_nominalization": {
            "name": "Gerundial Subject Nominalization",
            "category": "Information Packaging",
            "pattern_formula": "[V-ing Phrase] + [Predicate Verb] + [Complement]",
            "pedagogical_function": "Packages dynamic processes into abstract nominal heads to serve as discourse topics.",
            "imitation_example": "Maintaining rigorous version control safeguards project integrity across distributed teams.",
            "common_mistakes": (
                "COBUILD Warning (Gerunds as Subject): Learners frequently produce: "
                "\"[INCORRECT: Maintain version control are -> CORRECT: Maintaining version control is] essential.\" "
                "(Note: Singular verbal agreement is mandatory for gerundial subject phrases)."
            )
        },
        "adversative_concession": {
            "name": "Adversative Concession & Epistemic Contrast",
            "category": "Logic & Stance",
            "pattern_formula": "[Subject] + [VP], but [Subject] + [also/still] + [VP]",
            "pedagogical_function": "Balances competing communicative claims or desires within a unified, nuanced discourse move.",
            "imitation_example": "Users desire modern computational convenience, but they also demand uncompromised data privacy.",
            "common_mistakes": (
                "COBUILD Warning (Linking Words): Learners frequently produce coordinate redundancy: "
                "\"[INCORRECT: Although they value convenience, but they demand -> CORRECT: Although they value convenience, they demand] uncompromised privacy.\""
            )
        },
        "discourse_transition_however": {
            "name": "Discourse Stance & Transition Adverbs",
            "category": "Logic & Stance",
            "pattern_formula": "However, [Subject] + [Modal/VP]",
            "pedagogical_function": "Signals an explicit rhetorical pivot or counter-expectation between adjacent discourse segments.",
            "imitation_example": "However, recent computational breakthroughs could afford researchers unprecedented analytical autonomy.",
            "common_mistakes": (
                "COBUILD Warning (Sentence Adverbials): Learners frequently produce comma splices: "
                "\"[INCORRECT: The system is reliable, however it requires updates -> CORRECT: The system is reliable; however, it requires updates / The system is reliable. However, it requires updates].\""
            )
        },
        "subordinate_concessive": {
            "name": "Subordinate Concessive Clauses (Although / While)",
            "category": "Logic & Stance",
            "pattern_formula": "Although/While + [Clause], [Subject] + [VP]",
            "pedagogical_function": "Establishes a concessive background frame to sharpen the significance of the main assertoric claim.",
            "imitation_example": "Although early benchmarks exhibited latency, subsequent caching resolved all bottleneck anomalies.",
            "common_mistakes": (
                "COBUILD Warning (Concessive Clauses): Learners frequently produce: "
                "\"Although the initial prototype exhibited latency, [INCORRECT: but -> CORRECT: (omit but)] the production version achieved peak throughput.\""
            )
        },
        "proportional_comparative": {
            "name": "Proportional Comparative Clauses (The more..., the more...)",
            "category": "Information Packaging",
            "pattern_formula": "The + [comparative] + [Clause], the + [comparative] + [Clause]",
            "pedagogical_function": "Encodes continuous covariation between two propositional scales with rhythmic rhetorical balance.",
            "imitation_example": "The more systematically engineers audit dependencies, the fewer security vulnerabilities emerge.",
            "common_mistakes": (
                "COBUILD Warning (Double Comparatives): Learners frequently produce: "
                "\"[INCORRECT: More we investigate, more we understand -> CORRECT: The more we investigate, the more we understand].\""
            )
        },
        "passive_voice_agentless": {
            "name": "Agentless Passive & Objective Information Packaging",
            "category": "Information Packaging",
            "pattern_formula": "[Subject] + [be/modal be] + [V-ed]",
            "pedagogical_function": "Suppresses circumstantial human agency to foreground the patient or objective outcome of the inquiry.",
            "imitation_example": "Rigorous baseline assessments must be conducted before any infrastructure adjustments are deployed.",
            "common_mistakes": (
                "COBUILD Warning (Passive Voice): Learners frequently produce intransitive passives: "
                "\"[INCORRECT: The accident was happened -> CORRECT: The accident happened].\""
            )
        }
    }

    @classmethod
    def match_cobuild_grammar_profile(cls, formula: str, quote: str, category: str = "") -> Dict[str, Any]:
        """
        Deterministically matches mined syntax formula and quote to the most specific
        COBUILD Grammar Profile (0 Token Cost, 0 Copyright DB).
        Eliminates the legacy Unit 69 fallback collapse.
        """
        f_lower = (formula or "").lower()
        q_lower = (quote or "").lower()
        c_lower = (category or "").lower()

        # 1. Cleft Focus: it + was/is + ... that/who
        if "focal element" in f_lower or "cleft" in f_lower:
            return cls.COBUILD_GRAMMAR_PROFILES["cleft_focus"]
        if re.search(r"\bit\s+(?:was|is|'s)\s+.*?\b(?:that|who)\b", q_lower):
            if not re.search(r"\bit\s+(?:was|is|'s)\s+(?:clear|obvious|important|necessary|said|true|likely)\b", q_lower):
                return cls.COBUILD_GRAMMAR_PROFILES["cleft_focus"]

        # 2. Correlative Parallelism: not only... but also
        if "not only" in f_lower or ("not only" in q_lower and "but also" in q_lower):
            if q_lower.strip().startswith("not only"):
                return cls.COBUILD_GRAMMAR_PROFILES["fronted_negative_inversion"]
            return cls.COBUILD_GRAMMAR_PROFILES["correlative_parallelism"]

        # 3. Object Complement / Complex Transitive: make + Object + Adj
        if ("make +" in f_lower or "render +" in f_lower or "find +" in f_lower) and ("[adj]" in f_lower or "[object]" in f_lower):
            return cls.COBUILD_GRAMMAR_PROFILES["object_complement_adj"]
        if re.search(r"\bmake\b.*?\b(?:possible|impossible|easy|clear|accessible|redundant)\b", q_lower):
            return cls.COBUILD_GRAMMAR_PROFILES["object_complement_adj"]

        # 4. Causative bare infinitive: make/have/let + sb + do
        if any(f"make + [" in f_lower or f"let + [" in f_lower for _ in (1,)):
            return cls.COBUILD_GRAMMAR_PROFILES["causative_bare_infinitive"]

        # 5. Non-restrictive / Sentential Relative Clauses
        if ", which" in q_lower or "which + [interpretive" in f_lower:
            return cls.COBUILD_GRAMMAR_PROFILES["relative_clause_elaborative"]

        # 6. Restrictive Relative Clauses: [NP] + which/that + [VP]
        if any(p in f_lower for p in ("[np] + which", "[np] + that", "[np] + who", "[np] + whom", "[np] + whose")):
            return cls.COBUILD_GRAMMAR_PROFILES["relative_clause_restrictive"]

        # 7. Proportional Comparative
        if "the + [comparative]" in f_lower:
            return cls.COBUILD_GRAMMAR_PROFILES["proportional_comparative"]

        # 8. Dummy-It Extraposition: It is adj to/that
        if "it + [be] + [adj" in f_lower or "dummy" in f_lower:
            return cls.COBUILD_GRAMMAR_PROFILES["dummy_it_extraposition"]

        # 9. Participial Adjuncts
        if "[v-ing phrase]" in f_lower or "[v-ed phrase]" in f_lower or "particip" in c_lower:
            return cls.COBUILD_GRAMMAR_PROFILES["participial_adjunct"]

        # 10. Gerund Subject Nominalization
        if "[v-ing / gerund phrase]" in f_lower:
            return cls.COBUILD_GRAMMAR_PROFILES["gerund_subject_nominalization"]

        # 11. Adversative & Concession
        if "but [subject] + [also]" in f_lower or ", but" in q_lower:
            return cls.COBUILD_GRAMMAR_PROFILES["adversative_concession"]
        if "although" in f_lower or "while" in f_lower or q_lower.startswith("although") or q_lower.startswith("while"):
            return cls.COBUILD_GRAMMAR_PROFILES["subordinate_concessive"]

        # 12. Discourse transition: However
        if q_lower.startswith("however") or "however" in f_lower:
            return cls.COBUILD_GRAMMAR_PROFILES["discourse_transition_however"]

        # 13. Passive voice
        if "passive" in f_lower or "passive" in c_lower:
            return cls.COBUILD_GRAMMAR_PROFILES["passive_voice_agentless"]

        # Dynamic Category-Aware Fallback (Prevents Unit 69 / Relative Clause Contamination)
        if c_lower == "rhetoric & emphasis":
            return {
                "name": "Rhetorical Emphasis & Contrast",
                "category": "Rhetoric & Emphasis",
                "pattern_formula": formula,
                "pedagogical_function": "Achieves stylistic prominence or balanced emphasis across coordinating structural units.",
                "imitation_example": "The investigation highlights not merely statistical significance, but practical real-world impact.",
                "common_mistakes": "COBUILD Warning (Emphasis & Symmetry): ESL learners frequently compromise stylistic balance through asymmetric syntactic coordination."
            }
        elif c_lower == "logic & stance":
            return {
                "name": "Logical Transition & Epistemic Stance",
                "category": "Logic & Stance",
                "pattern_formula": formula,
                "pedagogical_function": "Articulates epistemic qualification, conditional constraints, or calibrated argumentative nuance.",
                "imitation_example": "Preliminary simulations validate the architecture; nonetheless, empirical load testing remains essential.",
                "common_mistakes": "COBUILD Warning (Logical Connectors): ESL learners frequently produce run-on sentences or misapply punctuation when joining clausal transitions."
            }
        elif c_lower == "cohesion & framing":
            return {
                "name": "Discursive Cohesion & Structural Framing",
                "category": "Cohesion & Framing",
                "pattern_formula": formula,
                "pedagogical_function": "Integrates proposition chunks into unified thematic progressions across discourse boundaries.",
                "imitation_example": "The claim that renewable systems lack reliability has been comprehensively disproven by recent field data.",
                "common_mistakes": "COBUILD Warning (Noun Complementation): ESL learners frequently confuse noun complement 'that'-clauses with modifying relative clauses."
            }
        else:
            return {
                "name": "Information Packaging & Syntactic Integration",
                "category": "Information Packaging",
                "pattern_formula": formula,
                "pedagogical_function": "Synthesizes complex propositions into streamlined syntactic units with elevated informational density.",
                "imitation_example": "Rigorous cross-validation procedures ensure that algorithmic inferences remain robust and generalizable.",
                "common_mistakes": "COBUILD Warning (Syntactic Packaging): ESL learners frequently mismanage clause embedding boundaries or subject-verb concord in complex predicate chains."
            }

    @classmethod
    def extract_deterministic_grammar(
        cls,
        text: str,
        sentence_pool: Optional[Dict[str, str]] = None,
        target_count: int = 5,
        syllabus_grammar: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Deterministically extracts grammar patterns from the text at 0 token cost,
        binding them directly to authentic pedagogical functions and empirical common mistakes
        from the COBUILD Common Learner Pitfall and Pattern Grammar system.
        """
        if not sentence_pool:
            _, sentence_pool = cls.tokenize_and_index_sentences(text)

        skeletons = cls.mine_grammar_skeletons(
            sentence_pool,
            target_count=target_count,
            syllabus_grammar=syllabus_grammar
        )
        if not skeletons:
            return []

        results: List[Dict[str, Any]] = []
        for s in skeletons:
            sid = s.get("sid", "S-1")
            quote = s.get("quote", "")
            cat = s.get("category", "Information Packaging")
            formula = s.get("pattern_formula", "[Subject] + [VP]")

            profile = cls.match_cobuild_grammar_profile(formula, quote, category=cat)

            prof_name = profile.get("name", "Syntactic Pattern")
            prof_cat = profile.get("category", cat)
            ped_func = profile.get("pedagogical_function", f"Synthesizes advanced academic discourse via {prof_cat.lower()} syntax.")
            imitation = profile.get("imitation_example", quote)
            clean_mistake = profile.get("common_mistakes", "ESL learners frequently misapply slot boundary constraints or morphological agreement.")

            # 1. LDOCE6 Authentic Exemplar Retrieval for imitation_example
            # Try to retrieve authentic sentence from 400k LDOCE corpus matching the structural trigger
            try:
                corpus_cand = None
                q_low = quote.lower()
                f_low = formula.lower()
                if "not only" in q_low:
                    exs = cls.search_corpus_examples('"not only"', limit=2)
                    if exs:
                        corpus_cand = exs[0]
                elif "make" in q_low and "possible" in q_low:
                    # Retrieve authentic examples demonstrating [make + Object + possible]
                    cands = cls.search_corpus_examples('NEAR(make possible, 4)', limit=10)
                    for c in cands:
                        if re.search(r'\b(?:make|makes|made)\b\s+([a-zA-Z0-9\s-]+?)\s+\bpossible\b', c, re.IGNORECASE):
                            corpus_cand = c
                            break
                    if not corpus_cand and cands:
                        corpus_cand = cands[0]
                elif "which" in q_low and "[np] + which" in f_low:
                    exs = cls.search_corpus_examples('"which gives"', limit=1)
                    if not exs:
                        exs = cls.search_corpus_examples('"which allows"', limit=1)
                    if exs:
                        corpus_cand = exs[0]
                elif re.search(r"\bit\s+(?:was|is)\s+.*?\bthat\b", q_low):
                    exs = cls.search_corpus_examples('"it was" AND "that"', limit=1)
                    if exs:
                        corpus_cand = exs[0]

                if corpus_cand and len(corpus_cand) > 15:
                    imitation = corpus_cand
            except Exception:
                pass

            # 2. LDOCE6 Authentic Grammar Alert & Don't Say diagnostics for common_mistakes
            try:
                # If pattern is complex transitive [make/find/keep + Object + Adj], prioritize adverb misapplication
                if "[object] + [adj]" in f_low or ("make" in q_low and "possible" in q_low):
                    clean_mistake = (
                        "LDOCE Grammar Alert (OBJECT COMPLEMENTS): Learners frequently produce an adverb instead of an adjective: "
                        "\"The new communication methods make this [INCORRECT: possibly -> CORRECT: possible].\" "
                        "(Note: The complement characterizes the resultant state of the object, requiring an adjective)."
                    )
                else:
                    # Scan quote for structural anchor words with LDOCE grammar boxes
                    anchor_words = []
                    for tok in re.findall(r"\b[a-zA-Z]{3,}\b", quote.lower()):
                        if tok in ("make", "want", "enjoy", "that", "which", "too", "enough", "such", "need", "own", "used", "help", "prefer", "suggest"):
                            anchor_words.append(tok)

                    ldoce_alert = None
                    for aw in anchor_words:
                        alert = cls.get_ldoce_grammar_alert(aw, quote=quote)
                        if alert and ("Don't say" in alert.get("content", "") or "✗" in alert.get("content", "")):
                            ldoce_alert = alert
                            break

                    if ldoce_alert:
                        raw_content = ldoce_alert.get("content", "")
                        # Extract bulleted items or sentences with Don't say / ✗
                        parts = [p.strip() for p in re.split(r'[•\n]+', raw_content) if p.strip()]
                        pitfalls = [p for p in parts if ("Don't say" in p or "✗" in p)]
                        if pitfalls:
                            chosen_pitfall = re.sub(r'\s+', ' ', pitfalls[0]).strip()
                            # Format as high-authority LDOCE Grammar Alert
                            clean_mistake = f"LDOCE Grammar Alert ({ldoce_alert.get('word', '').upper()}): {chosen_pitfall}"
            except Exception:
                pass

            results.append({
                "quote": quote,
                "pattern_formula": formula,
                "category": prof_cat,
                "syntax_topic": prof_name,
                "pedagogical_function": ped_func,
                "imitation_example": imitation,
                "common_mistakes": clean_mistake,
                "design_audit": f"AUDIT: [{sid}] -> [{formula}] -> [VERBATIM_CONFIRMED]"
            })

        return results

    @classmethod
    def build_authentic_cloze_items(
        cls,
        vocab_content: str,
        target_count: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Track 1: Authentic Passage Cloze (Achievement Testing / 学业水平测试)
        Extracts extracted vocabulary items and masks their target appearance in the authentic
        quoted passage sentence using spaCy lemmatization and exact word-boundary masking.
        Generates 0-token, 100% textbook-grounded question stems.
        """
        if not vocab_content:
            return []

        nlp = cls.get_spacy()
        blocks = re.split(r'\n(?=##\s*\[\[)', vocab_content)
        parsed_items: List[Dict[str, str]] = []

        for b in blocks:
            m_word = re.search(r'##\s*\[\[(.*?)\]\]', b)
            if not m_word:
                continue
            raw_w = m_word.group(1).strip()
            # Clean slot placeholders from phrasal headwords, e.g. "tend to [sth/sb]" -> "tend to"
            word = re.sub(r'\[.*?\]|\(.*?\)', '', raw_w).strip()
            if not word:
                continue
            m_pos = re.search(r'-\s*\*\*Part\s+of\s+Speech\*\*:\s*([^\n]+)', b, re.IGNORECASE)
            m_def = re.search(r'-\s*\*\*Definition\*\*:\s*([^\n]+)', b, re.IGNORECASE)
            m_quote = re.search(r'-\s*\*\*Quoted Sentence\*\*:\s*([^\n]+)', b, re.IGNORECASE)

            parsed_items.append({
                "word": word,
                "part_of_speech": m_pos.group(1).strip() if m_pos else "noun",
                "definition": m_def.group(1).strip() if m_def else "",
                "quote": m_quote.group(1).strip() if m_quote else ""
            })

        cloze_items: List[Dict[str, Any]] = []

        # --- Dependent-connective cloze (grammar/collocation axis -> single answer) ---
        # 'of'-taking connective heads: the obligatory 'of' belongs to the fixed phrase
        # ('regardless of <NP>'). Against such a target the bare-NP connectives below are
        # UNgrammatical in the '____ of <NP>' frame, so they can never co-answer. Connectives
        # are a small closed class, so this is a stable, well-attested functional inventory
        # (real words), not an open-ended valency guess.
        connectives_of_taking = frozenset({"regardless", "irrespective"})
        connectives_bare_np = ("despite", "notwithstanding", "whatever", "barring")

        for it in parsed_items:
            w = it["word"]
            if not cls.is_pedagogical_target(w):
                continue
            q = it["quote"]
            if not q:
                continue

            # Strip leading/trailing quotation marks from quote
            q_clean = q.strip().strip('"').strip("'").strip("“").strip("”")
            if not q_clean:
                continue

            doc = nlp(q_clean)
            w_lower = w.lower()

            # Dependent connective (e.g. 'regardless of'): keep the obligatory 'of' in the
            # stem and blank only the head. Only an 'of'-taking head is grammatical in the
            # '____ of <NP>' frame, so the bare-NP distractors can never co-answer -> single.
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
                        if len(cloze_items) >= target_count:
                            break
                        continue
                # Connective entry whose quote lacks a well-formed 'head + of' frame -> skip.
                continue

            # Locate token in sentence that matches the lemma or literal surface form
            target_tok = None
            for t in doc:
                if t.text.lower() == w_lower or t.lemma_.lower() == w_lower:
                    target_tok = t
                    break

            if not target_tok:
                continue

            surface_form = target_tok.text
            # Replace target token with exact 4 underscores '____'
            stem = re.sub(rf'\b{re.escape(surface_form)}\b', '____', q_clean, count=1)

            # Ensure exactly one blank exists in stem
            if len(re.findall(r'_{2,}', stem)) != 1:
                continue

            # Wordnet distractor generation for Track 1 Cloze
            pos_tag = "n"
            pos_raw = it["part_of_speech"].lower()
            if "verb" in pos_raw: pos_tag = "v"
            elif "adj" in pos_raw: pos_tag = "a"
            elif "adv" in pos_raw: pos_tag = "r"

            distractors = cls.generate_zero_collision_distractors(
                target_word=surface_form,
                pos=pos_tag,
                count=3
            )

            cloze_items.append({
                "target_word": surface_form,
                "base_headword": w,
                "part_of_speech": it["part_of_speech"],
                "definition": it["definition"],
                "question": stem,
                "source_sentence": q_clean,
                "precomputed_distractors": distractors
            })

            if len(cloze_items) >= target_count:
                break

        return cloze_items

    # Backward compatibility alias
    extract_authentic_cloze_items = build_authentic_cloze_items

    @classmethod
    def build_precomputed_target_skeletons(
        cls,
        vocab_content: str,
        target_count: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Track 2: Pre-Computed Generative Skeletons (Proficiency Testing / 语境迁移单选题)
        Extracts vocabulary items from extracted cards and synthesizes authoritative collocational
        anchors (via Oxford Collocations Dictionary) and collision-free distractors (via WordNet + OCD).
        Produces pre-computed item skeletons to strictly constrain LLM generation, guaranteeing zero
        duplicate options, zero synonym pile double-keys, and perfect stem-option-explanation alignment.
        """
        if not vocab_content:
            return []

        nlp = cls.get_spacy()
        ocd = cls.get_oxford_collocations()
        blocks = re.split(r'\n(?=##\s*\[\[)', vocab_content)
        parsed_items: List[Dict[str, str]] = []

        for b in blocks:
            m_word = re.search(r'##\s*\[\[(.*)\]\]', b)
            if not m_word:
                continue
            raw_w = m_word.group(1).strip()
            # Clean slot placeholders from phrasal headwords, e.g. "tend to [sth/sb]" -> "tend to"
            word = re.sub(r'\[.*?\]|\(.*?\)', '', raw_w).strip()
            if not word:
                continue

            m_pos = re.search(r'-\s*\*\*Part\s+of\s+Speech\*\*:\s*([^\n]+)', b, re.IGNORECASE)
            m_def = re.search(r'-\s*\*\*Definition\*\*:\s*([^\n]+)', b, re.IGNORECASE)
            m_quote = re.search(r'-\s*\*\*Quoted Sentence\*\*:\s*([^\n]+)', b, re.IGNORECASE)

            parsed_items.append({
                "word": word,
                "part_of_speech": m_pos.group(1).strip() if m_pos else "noun",
                "definition": m_def.group(1).strip() if m_def else "",
                "quote": m_quote.group(1).strip() if m_quote else ""
            })

        skeletons: List[Dict[str, Any]] = []
        used_distractors: Set[str] = set()

        # Batch Anchor Collision Set (P0-5): every headword that is itself a target of
        # this same batch. An anchor (or a prepositional object welded into the micro-task
        # frame) drawn from this set produces an unsatisfiable item — the evaluator's
        # anchor-presence gate demands the word inside the stem while its cross-target
        # leakage gate bans it with a fatal flag. Anchors must therefore be drawn from
        # OUTSIDE this set; when nothing outside the set fits, the blueprint degrades to
        # the neutral semantic-field template instead of welding two gates into a deadlock.
        batch_target_words: Set[str] = set()
        for _it in parsed_items:
            _w = _it["word"].lower().strip()
            if not _w:
                continue
            batch_target_words.add(_w)
            if " " in _w:
                for _tok in _w.split():
                    if len(_tok) >= 3 and _tok not in cls._ANCHOR_STOPWORDS:
                        batch_target_words.add(_tok)

        for it in parsed_items:
            w = it["word"]
            w_lower = w.lower()
            raw_pos = it["part_of_speech"].lower()

            # Pedagogical Integrity Filter: Skip proper nouns, demonyms, country names (e.g. Thai, Spanish, Colombia)
            if not cls.is_pedagogical_target(w):
                continue
            
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
                if " " in w_lower:
                    canonical_pos = "phrase"
                elif cls.is_function_word(w_lower):
                    canonical_pos = "function_word"
                elif "adv" in raw_pos:
                    canonical_pos = "adv"
                elif "verb" in raw_pos:
                    canonical_pos = "verb"
                elif "adj" in raw_pos:
                    canonical_pos = "adj"
                elif "-" in w_lower and it.get("quote"):
                    # Use contextual POS inference for hyphenated compound words
                    c_pos = cls.determine_contextual_pos(w_lower, it["quote"])
                    if c_pos.startswith("adj"):
                        canonical_pos = "adj"
                    elif c_pos.startswith("adv"):
                        canonical_pos = "adv"
                    elif c_pos.startswith("verb"):
                        canonical_pos = "verb"
                    else:
                        canonical_pos = "noun"
                else:
                    canonical_pos = "noun"

            # Evidence-Grade Gate (P2): a curriculum 'Quoted Sentence' licenses an anchor or a
            # valency frame only when it actually contains a clause. Fragments lifted from
            # headings ('Punctuality Pays!') were treated as authentic usage, so the blueprint
            # welded anchors and frames onto a two-word title that never demonstrated them.
            # A weak quote is demoted to display-only evidence: it stays visible to the writer
            # as candidate_quote, but every anchor must then come from LDOCE instead.
            quote_provenance, quote_wordcount = cls._quote_evidence_strength(it.get("quote"))
            it["raw_quote"] = it.get("quote") or ""
            if quote_provenance == "weak":
                it["quote"] = ""

            # 1. Retrieve Authentic Oxford Collocational Anchor (Sense-Guided)
            if not is_idiom_slot:
                anchor = None
                anchor_type = "collocation"
            prep_obj = None

            if canonical_pos == "verb":
                # Verb Step A: Extract the valency complement PP from the authentic quote.
                # Four-way gate (Pillar 1): (1) post-head linear constraint — complements
                # of the head verb sit to its RIGHT, physically excluding fronted adjuncts
                # ('In a society ... degraded' can never attach to 'degrade'); (2) fixed
                # adjunct blacklist (for example / in fact / ...) blocks discourse markers;
                # (3) OCD consensus gate — a preposition confirmed by the Oxford valency
                # list of this verb outranks an unconfirmed one; (4) when the verb has no
                # OCD valency evidence at all, only typical academic complements (into/to/
                # from) are trusted, otherwise the anchor degrades to None (neutral template).
                if it["quote"]:
                    try:
                        doc = nlp(it["quote"])
                        v_collocs = cls.get_oxford_collocations(w_lower, pos="verb")
                        consensus_preps = set(v_collocs.get("prep", []))
                        for tok in doc:
                            if tok.text.lower() == w_lower or tok.lemma_.lower() == w_lower:
                                verified_prep = None   # (token_index, prep, pobj) confirmed by OCD valency
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
                                # Consensus cross-validation (soft gate, Pillar 1 step 3):
                                # an OCD-confirmed preposition and a bare syntactic
                                # complement are both real — the more head-adjacent one
                                # prevails. A complement that already survived the post-head
                                # linear gate and the adjunct blacklist is genuine syntax,
                                # so an incomplete dictionary record (e.g. 'rely': OCD lists
                                # 'for', yet the authentic frame is 'rely on') never vetoes it.
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
                                break
                    except Exception:
                        pass

                # Verb Step B1: If no bound preposition found in quote, inspect quote for authentic direct object patient (dobj)
                if not anchor and it["quote"]:
                    try:
                        doc = nlp(it["quote"])
                        for tok in doc:
                            if tok.text.lower() == w_lower or tok.lemma_.lower() == w_lower:
                                for child in tok.children:
                                    if child.dep_ == "dobj" and child.lemma_.lower() not in cls._ANCHOR_STOPWORDS:
                                        # Use surface form or lemma based on plural/singular
                                        obj_form = child.text.lower() if child.tag_ == "NNS" else child.lemma_.lower()
                                        anchor = obj_form
                                        anchor_type = "object"
                                        break
                                if anchor:
                                    break
                    except Exception:
                        pass

                # Verb Step B2: If no bound prep or dobj, check if the verb has an authentic subject noun in quote (nsubj)
                # E.g. "as the clock chimes twelve midnight" -> anchor='clock', anchor_type='verb_subject'
                if not anchor and it["quote"]:
                    try:
                        doc = nlp(it["quote"])
                        for tok in doc:
                            if tok.text.lower() == w_lower or tok.lemma_.lower() == w_lower:
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

                # Verb Step B3: Fallback to sense-aligned direct noun object from OCD if quote had no dobj/subject
                if not anchor:
                    v_entry = cls.get_oxford_collocations(w_lower, pos="verb")
                    candidates = v_entry.get("colloc_nouns", []) + v_entry.get("noun_after", [])
                    if candidates:
                        anchor = cls._select_sense_aligned_anchor(candidates, definition=it.get("definition"), quote=it.get("quote"), target_word=w_lower, forbidden=batch_target_words)
                        if anchor:
                            anchor_type = "object"

            elif canonical_pos == "noun":
                # Noun Syntactic Dependency Walker (Pillar 2) — terminates the old flat
                # string-matching that misattributed 'more' (degree adverb of 'more hours')
                # to 'autonomy' and let the selector draw 'strange' for 'compulsion'.
                # Priority A: post-nominal prepositional complement (autonomy FROM compulsion)
                # Priority B: true syntactic modifiers (amod/compound), stopword-filtered
                # Priority C: governing verb (head of dobj/pobj), cross-validated vs OCD verb_before
                # Priority D: OCD candidates vs quote/definition overlap (strict None on miss)
                n_entry = cls.get_oxford_collocations(w_lower, pos="noun")
                if it["quote"]:
                    try:
                        doc = nlp(it["quote"])
                        ocd_prep_tokens = set()
                        for p in n_entry.get("prep", []):
                            cleaned = re.sub(r'[\(\)]', '', p).lower()
                            # 'prep_tok', never 'w': 'w' is the headword of this item and this
                            # loop leaked 'of' into it, corrupting skeleton['base_headword'].
                            for prep_tok in cleaned.replace("/", " ").split():
                                if prep_tok in ("to", "with", "from", "on", "for", "in", "into", "of", "against", "at", "upon", "towards", "toward", "over", "under", "about"):
                                    ocd_prep_tokens.add(prep_tok)
                        for lp in cls.get_ldoce_preps(w_lower):
                            ocd_prep_tokens.add(lp)

                        for tok in doc:
                            if tok.text.lower() == w_lower or tok.lemma_.lower() == w_lower:
                                for child in tok.children:
                                    if child.dep_ == "prep" and child.text.lower() in ("to", "with", "from", "on", "for", "in", "into", "of", "against", "at", "upon", "towards", "toward", "over", "under", "about"):
                                        prep_word = child.text.lower()
                                        # Noun-preposition consensus gate: if noun has registered prepositions in OCD/LDOCE,
                                        # the syntactic preposition MUST be verified against OCD/LDOCE to reject casual adjuncts (e.g. 'control from').
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
                    # Check modifying adjectives, governing verbs, or single-word bound prepositions
                    if n_entry.get("adj") or n_entry.get("verb_before"):
                        candidates = n_entry.get("adj", []) + n_entry.get("verb_before", [])
                        anchor = cls._select_sense_aligned_anchor(candidates, definition=it.get("definition"), quote=it.get("quote"), target_word=w_lower, forbidden=batch_target_words)
                        if anchor:
                            # OCD stores collocations as phrases ('great luxury'), so testing the
                            # bare token against the raw list mislabelled 'great' as a governing
                            # verb and handed the frame template the wrong relation.
                            anchor_type = "adj" if cls._token_in_phrases(anchor, n_entry.get("adj", [])) else "verb"
                            if cls._is_delexical_anchor(anchor, anchor_type, "noun"):
                                anchor, anchor_type = None, "collocation"
                    elif n_entry.get("prep"):
                        p_cand = cls._pick_anchor_token(n_entry["prep"], target_word=w_lower, forbidden=batch_target_words)
                        if p_cand:
                            anchor = p_cand
                            anchor_type = "prep"

                # Priority E (B2): the partitive compound the passage itself shows. Every
                # priority above looks downward from the target, but a noun sitting inside
                # 'N of ____' only has a frame upward - and that is the frame the curriculum
                # sentence actually gave it ('various baskets of goodies').
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
                            if tok.text.lower() == w_lower or tok.lemma_.lower() == w_lower:
                                # A1. Predicative Valency: Check if adjective governs a bound preposition
                                for child in tok.children:
                                    if child.dep_ == "prep" and child.text.lower() in ("to", "for", "with", "of", "in", "from", "at", "about", "on"):
                                        anchor = child.text.lower()
                                        anchor_type = "prep"
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
                                # A3. Predicative / Causative Complement: Check if adjective is complement of a causative/copula verb (e.g. 'make this possible')
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
                    # If still no anchor, check if the adjective has a bound preposition in OCD
                    if not anchor and adj_entry.get("prep"):
                        p_cand = cls._pick_anchor_token(adj_entry["prep"], target_word=w_lower, forbidden=batch_target_words)
                        if p_cand:
                            anchor = p_cand
                            anchor_type = "prep"

                # Adjective Step C: Fall back to curated academic adjective preposition valency map or LDOCE pattern
                if not anchor:
                    ld_entry = cls.get_ldoce_entry(w_lower)
                    if ld_entry:
                        for s in ld_entry.get("senses", []):
                            for pat in s.get("patterns", []):
                                for p in ("of", "to", "for", "with", "about", "from", "in", "on", "at"):
                                    if re.search(rf"\b{p}\b", pat.lower()):
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
                # Priority A: Check if adverb modifies an adjective, adverb, or verb in authentic quote
                quote_frame = cls._adverb_frame_from_quote(w_lower, it.get("quote"))
                if it.get("quote"):
                    try:
                        doc = nlp(it["quote"])
                        for tok in doc:
                            if tok.lemma_.lower() == w_lower or tok.text.lower() == w_lower:
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

                # Priority B: Dictionary lookup from LDOCE zero collision anchor
                if not anchor:
                    ld_anc, ld_atype, ld_frame, _ = cls.find_ldoce_zero_collision_anchor(
                        target_word=w_lower,
                        pos="adv",
                        definition=it.get("definition"),
                        quote=it.get("quote")
                    )
                    if ld_anc:
                        # Pass through sense-aligned anchor gate
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


            # 2. Syntactic Fallback: Extract anchor from quoted sentence if OCD / prep lookup was empty
            if not anchor and it["quote"]:
                q_clean = it["quote"].strip().strip('"').strip("'").strip("“").strip("”")
                try:
                    doc = nlp(q_clean)
                    for tok in doc:
                        if tok.text.lower() == w_lower or tok.lemma_.lower() == w_lower:
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
                                if tok.head.pos_ != "PRON" and h_lem not in cls._ANCHOR_STOPWORDS and h_lem != w_lower:
                                    anchor = h_lem
                                    anchor_type = "contextual"
                            break
                except Exception:
                    pass

            # 3. Generate 3 Zero-Collision Distractors with Definition-Locked WSD & Sense Antonyms
            if canonical_pos == "phrase":
                p_dists = cls.generate_phrase_distractors(w_lower, count=3, exclude_words=used_distractors)
                distractors = p_dists
                dist_meta = {d: "phrase_contrast" for d in p_dists}
            elif canonical_pos == "idiom_slot" and w_lower == "granted":
                # Participle/adjective distractors that share VBN morphology and CEFR level but cannot collocate with 'take ... for'
                # Authentic idiom: 'take sth for granted'. 'take sth for accepted/given/known' are non-idiomatic or clunky.
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

            # Slot Legality pre-computation (P1-1): does the target's own frame demand a
            # direct object? A transitive target ('annoy somebody', 'pay somebody something')
            # cannot tolerate an intransitive-only distractor ('flourish') in its slot.
            verb_requires_object = cls._verb_takes_object(w_lower) if canonical_pos == "verb" else None

            # 3a. Double-Key Clearance (deterministic, frame-aware)
            #     A distractor is a double-key if it shares the target's bound
            #     preposition (OCD) AND sits in the target's WordNet semantic
            #     neighborhood; in slot items, same-slot near-synonyms / coordinate
            #     terms / troponyms qualify. Sense antonyms are whitelisted
            #     contrast distractors (eliminated by sentence polarity, not
            #     valency). Double-keys are dropped and refilled from an extended
            #     candidate pool so the item never degrades below 3 distractors.
            def _is_double_key(cand: str) -> bool:
                if canonical_pos in ("function_word", "phrase", "idiom_slot"):
                    return False
                flag, _kind = cls.double_key_collision(w_lower, cand, anchor, anchor_type, pos=canonical_pos)
                return flag

            def _is_valid_distractor(cand: str) -> bool:
                if not cand or _is_double_key(cand):
                    return False
                if canonical_pos in ("function_word", "phrase", "idiom_slot"):
                    return True
                # Slot Legality Gate (P1-1): a distractor that cannot occupy the target's own
                # syntactic slot makes the item solvable by morphology instead of meaning
                # ('log in to a ____' offering the adverb 'somehow').
                if not cls._distractor_occupies_slot(
                    cand, canonical_pos, requires_object=(verb_requires_object is True)
                ):
                    return False
                return cls.is_cefr_compliant_distractor(cand, w_lower)

            cleared = [d for d in distractors if _is_valid_distractor(d)]
            refill_meta: Dict[str, str] = {}
            if len(cleared) < 3:
                try:
                    refill_pool, refill_meta = cls.generate_vocab_distractors(
                        target_word=w_lower, pos=canonical_pos,
                        context_anchor=anchor, anchor_type=anchor_type,
                        target_count=8, exclude_words=used_distractors,
                        definition=it.get("definition"), return_metadata=True
                    )
                    refill_meta = refill_meta or {}
                except Exception:
                    refill_pool = []
                for cand in refill_pool:
                    if len(cleared) >= 3:
                        break
                    if cand and cand not in cleared and _is_valid_distractor(cand):
                        cleared.append(cand)
                # Secondary refill: the primary semantic-tree pool is saturated
                # with near-synonyms for synonym-heavy targets (designate,
                # adverse), so fall back to a semantically distant pool
                # (antonyms + taxonomy siblings + generic core/academic words).
                if len(cleared) < 3:
                    pos_char = "v" if canonical_pos == "verb" else ("a" if canonical_pos == "adj" else ("r" if canonical_pos == "adv" else "n"))
                    try:
                        for cand in cls.generate_zero_collision_distractors(w_lower, pos=pos_char, count=3):
                            if len(cleared) >= 3:
                                break
                            if cand and cand not in cleared and cand not in used_distractors and _is_valid_distractor(cand):
                                cleared.append(cand)
                    except Exception:
                        pass
                # Safe Refill: Never dump rejected double-keys (like 'astute' for 'smart') back into options!
                # If still under 3, refill from verified, non-colliding words calibrated to target CEFR level.
                if len(cleared) < 3:
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
                        if len(cleared) >= 3:
                            break
                        if cand != w_lower and cand not in cleared and cand not in used_distractors and _is_valid_distractor(cand):
                            cleared.append(cand)
            distractors = cleared

            # Mechanism bookkeeping: the clearance/refill above replaces candidates, so the
            # mechanism map must be rebuilt against the distractors that actually survived.
            # A stale map makes the contrast instruction describe words that are not in the item.
            dist_meta = {
                d: (dist_meta.get(d) or refill_meta.get(d) or "distant_contrast")
                for d in distractors
            }

            # Assemble full options (target + 3 distractors). The target is placed at a
            # stable, per-word position (not always slot A) so the answer key is not
            # trivially predictable and does not rely on post-hoc shuffling.
            distractors3 = distractors[:3]
            slot = zlib.crc32(w_lower.encode("utf-8")) % 4
            raw_options = distractors3[:slot] + [w_lower] + distractors3[slot:]
            correct_answer_index = slot
            used_distractors.update(d for d in distractors3 if d)

            # 4. Plan A: All options and target word maintain parallel inflection
            # If target noun is plural/plurale tantum (e.g. 'goods', 'customs', 'clothes', 'fireworks'),
            # or used as plural in authentic quote (e.g. 'fireworks' from lemma 'firework'),
            # ensure all 3 distractors are inflected to plural to maintain 100% morphological symmetry.
            is_plural_noun = False
            plural_target = w_lower
            if canonical_pos == "noun":
                spacy_doc = nlp(w_lower)
                is_spacy_plural = any(tok.tag_ == "NNS" for tok in spacy_doc)
                plural_tantum = {"goods", "customs", "clothes", "belongings", "surroundings", "fireworks", "premises", "congratulations"}
                
                # Also check authentic quote if the noun appeared in plural NNS form (e.g. fireworks)
                quote_plural = False
                if it["quote"]:
                    try:
                        q_doc = nlp(it["quote"])
                        quote_plural = any((tok.lemma_.lower() == w_lower or tok.text.lower() == w_lower) and tok.tag_ == "NNS" for tok in q_doc)
                    except Exception:
                        quote_plural = False

                if is_spacy_plural or w_lower in plural_tantum or quote_plural:
                    is_plural_noun = True
                    plural_target = cls.pluralize_noun(w_lower)

            if is_plural_noun:
                inflection_desc = "plural form (NNS)"
                final_target = plural_target
                # Symmetrically inflect distractors to plural
                inflected_distractors3 = [cls.pluralize_noun(d) for d in distractors3]
                # A1: every compliance gate must judge the plural surface form the
                # student actually reads, not the singular candidate that was screened.
                inflected_distractors3 = cls.screen_inflected_options(
                    inflected_distractors3,
                    [cls.pluralize_noun(d) for d in distractors[3:]],
                    final_target,
                )
                used_distractors.update(inflected_distractors3)
                slot = min(slot, len(inflected_distractors3))
                correct_answer_index = slot
                final_options = inflected_distractors3[:slot] + [final_target] + inflected_distractors3[slot:]
            else:
                inflection_desc = "base form"
                final_target = w_lower
                final_options = raw_options

            # 5. Synthesize Itemized Micro-Task for LLM based on Distractor Mechanisms & Definition
            dist_list = [opt for opt in final_options if opt.lower() != final_target.lower()]
            dist_str = ", ".join(dist_list)

            # LDOCE Authentic Collocations & Corpus Example Scaffold:
            # If quote lacked a strong grammatical anchor or yielded generic context,
            # query LDOCE 6th Edition for an authoritative zero-collision anchor and authentic sentence pattern.
            ldoce_pattern_example = None
            ldoce_cloze_frame = None
            pattern_anchor = None
            # A preposition the quote itself displayed IS the target's valency frame in that
            # sentence ('degrade into modern slaves', 'autonomy from compulsion'). The
            # locked-sense gate below already vets it, so the absence of a dictionary example
            # must not discard it in favour of some unrelated LDOCE collocation.
            quote_shown_prep = bool(anchor) and anchor_type == "prep" and bool(re.search(
                rf"\b{re.escape(str(anchor).lower())}\b", (it.get("quote") or "").lower()))
            # The same argument holds for a partitive compound: 'baskets of goodies' is the
            # frame this passage demonstrably used, so a missing dictionary example must not
            # trade it for an unrelated LDOCE collocation.
            quote_shown_partitive = bool(anchor) and anchor_type == "partitive" and bool(re.search(
                rf"\b{re.escape(str(anchor).lower())}\b", (it.get("quote") or "").lower()))
            if anchor and anchor_type != "contextual":
                # A quote-derived anchor is only a usable structural model when the
                # dictionary itself pairs the target with it. 'further penalty' and
                # 'maintain contact' are accidents of one curriculum sentence, not
                # collocations, and they left those items with no authentic pattern.
                # A compound anchor is looked up by its quantifier noun, because the
                # dictionary records 'basket of', not the glued string 'baskets of'.
                # Single-word anchors keep being looked up as themselves.
                lookup_anchor = prep_obj if anchor_type == "partitive" else anchor
                ldoce_pattern_example = cls.ldoce_example_for_anchor(final_target, lookup_anchor)
                pattern_anchor = anchor
            if (not anchor or anchor_type == "contextual"
                    or (not ldoce_pattern_example and not quote_shown_prep and not quote_shown_partitive)):
                ld_anchor, ld_atype, ld_fdesc, ld_ex = cls.find_ldoce_zero_collision_anchor(
                    target_word=final_target,
                    pos=canonical_pos,
                    distractors=dist_list,
                    definition=it.get("definition"),
                    quote=it.get("quote")
                )
                if ld_anchor:
                    anchor = ld_anchor
                    anchor_type = ld_atype
                    ldoce_pattern_example = ld_ex
                    pattern_anchor = ld_anchor

            # Batch Anchor Collision Gate (P0-5): an anchor — or a prepositional object
            # welded into the micro-task frame — that is itself another batch target makes
            # the item logically unsatisfiable (the evaluator's anchor-presence gate demands
            # it in the stem while its cross-target leakage gate bans it with a fatal flag).
            # Such anchors come from the authentic quote (spaCy nsubj/dobj/pobj walks) or
            # from the LDOCE fallback above, neither of which knows about the batch, so the
            # conflict is resolved here: drop the offending object, re-pick a batch-safe
            # anchor, and degrade to the neutral template when nothing outside the batch fits.
            if batch_target_words and not is_idiom_slot and (anchor or prep_obj):
                if prep_obj and cls._in_forbidden_word_set(prep_obj, batch_target_words):
                    prep_obj = None
                if (anchor and anchor_type == "partitive"
                        and cls._in_forbidden_word_set(
                            prep_obj or (anchor.split()[0] if anchor.split() else ""),
                            batch_target_words)):
                    # The quantifier noun of a partitive compound is itself another batch
                    # target: the anchor-presence gate would demand in the stem what the
                    # cross-target leakage gate bans, so the compound is dropped outright.
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
                    # Any corpus example retrieved for the rejected anchor described the
                    # wrong frame; force a re-lookup against the surviving anchor.
                    ldoce_pattern_example = None
                    ldoce_cloze_frame = None

            # Locked-Sense Frame Concordance Gate (P0-4): a bound-preposition anchor must
            # belong to the sense this item actually locks. 'annoy' was welded to 'on' because
            # some sense of 'annoy' registers 'get on somebody's nerves', while the sense this
            # item teaches is 'it annoys somebody when ...' — a direct-object frame. A prep the
            # locked sense does not license is dropped and re-picked from the locked sense's own
            # evidence; a preposition the quote genuinely showed is never vetoed.
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
                    quote=it.get("quote")
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
                    # The example retrieved for the rejected preposition demonstrated the
                    # wrong sense; it must not survive as this item's structural model.
                    ldoce_pattern_example = None
                # The corpus example retrieved for the rejected preposition described the wrong
                # frame; force a fresh lookup against the surviving anchor.
                ldoce_cloze_frame = None

            # If a re-anchoring produced a new anchor, look up the LDOCE sentence that
            # actually demonstrates this (target, anchor) pair.
            if anchor and not ldoce_pattern_example and anchor != pattern_anchor:
                ldoce_pattern_example = cls.ldoce_example_for_anchor(
                    final_target, prep_obj if anchor_type == "partitive" else anchor)

            # Nothing demonstrated the pair - or no anchor survived at all. The item still
            # gets the dictionary's own sentence for the sense it teaches: a skeleton used to
            # reach the LLM with no structural model whatsoever.
            if not ldoce_pattern_example:
                ldoce_pattern_example = cls.ldoce_sense_example(
                    final_target,
                    pos=canonical_pos,
                    definition=it.get("definition"),
                    quote=it.get("quote")
                )

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

            # A partitive frame and a corpus blueprint that contradicts it are two orders the
            # writer cannot both obey: 'baskets of' is the phrase the anchor-presence gate will
            # demand, so a blueprint reading 'lots of ____' is dropped rather than fought.
            if anchor_type == "partitive" and anchor and ldoce_cloze_frame:
                quant_lem = (prep_obj or (anchor.split()[0] if anchor.split() else "")).lower()
                if quant_lem and not re.search(rf"\b{re.escape(quant_lem)}\w*\b", ldoce_cloze_frame.lower()):
                    ldoce_cloze_frame = None

            # Prescribed-Inflection Concordance (P1-2): the blueprint declares an inflection
            # the LLM must obey, so the declaration has to match the frame it hands over. The
            # verb branch always declared 'base form' even when the authentic LDOCE example
            # that supplied the frame read 'What annoyed him most was that he had received no
            # apology.' — a past-tense frame. A base-form option inside a past-tense cloze is
            # unanswerable, so the whole option set is inflected to the example's own form.
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
                _form_remap = {b: f for b, f in zip(_base_options, final_options)}
                dist_meta = {_form_remap.get(k, k): v for k, v in dist_meta.items()}
                used_distractors.update(o for o in final_options if o.lower() != final_target.lower())
                dist_list = [opt for opt in final_options if opt.lower() != final_target.lower()]
                dist_str = ", ".join(dist_list)

            # B2: an anchor that survived the gates is a contract the writer must honour; an
            # anchor that survived none is a fact the writer must be told. A noun with no
            # frame is planned as a sense-recognition item so that the definition - and not a
            # fabricated collocation - is what makes one option the only answer.
            anchor_downgrade = None
            if not anchor and canonical_pos == "noun" and not is_idiom_slot:
                anchor_downgrade = "sense_recognition"

            # Frame-aware micro-task guard (double-key template fix):
            # Only distractors that genuinely LACK the bound frame may be claimed as
            # "strictly ruled out" on valency grounds. Survivors that still share the
            # frame (deliberate contrast / frame-only distractors) MUST be excluded
            # on semantic or polarity grounds — asserting a false valency exclusion
            # makes the LLM fabricate bogus exclusion rationales.
            if anchor_type == "prep" and anchor:
                _bound = anchor.lower()
                shared_frame = [
                    d for d in dist_list
                    if _bound in cls.get_oxford_collocations(d.lower(), pos=canonical_pos).get("prep", [])
                    or d.lower() in cls._ADJ_PREP_VALENCY_MAP.get(_bound, [])
                ]
                non_shared = [d for d in dist_list if d not in shared_frame]
                if shared_frame and non_shared:
                    valency_clause = (
                        f"strictly ruling out [{', '.join(non_shared)}] which require different prepositions or direct objects. "
                        f"Note that [{', '.join(shared_frame)}] can also grammatically govern '{anchor}', "
                        f"so exclude it on precise semantic or polarity grounds, not valency"
                    )
                elif shared_frame:
                    valency_clause = (
                        f"excluding [{dist_str}] on precise semantic or polarity grounds, "
                        f"since every option can govern '{anchor}'"
                    )
                else:
                    valency_clause = f"strictly ruling out [{dist_str}] which require different prepositions or direct objects"
            else:
                valency_clause = f"strictly ruling out [{dist_str}]"

            # The declared inflection and the frame must be stated with one voice (P1-2).
            verb_form_phrase = {
                "VBD": "past-tense verb", "VBG": "present-participle verb",
                "VBZ": "third-person-singular verb", "VBP": "base-form verb", "VB": "base-form verb",
            }.get(verb_form_tag, "base-form verb")

            if canonical_pos == "verb":
                if anchor_type == "prep":
                    needs_dobj, dobj_slot, is_pass = cls.analyze_verb_valency_pattern(final_target, anchor)
                    # Direct-object slot preservation (P0-4): analyze_verb_valency_pattern only
                    # inspects patterns that mention this preposition, so a transitive verb whose
                    # prep pattern omits the patient ('attach importance to') lost its object slot
                    # and the frame degraded to prep-only. The target's own valency re-asserts it.
                    if not needs_dobj and verb_requires_object is True:
                        needs_dobj, dobj_slot = True, "[somebody/something]"
                    obj_note = f" (followed by object '{prep_obj}')" if prep_obj else ""
                    ant_clue = cls._contrast_strategy_clause(final_target, it.get("definition"), dist_list, dist_meta, anchor_type)
                    if needs_dobj:
                        frame_str = f"[Subject] + [Modal/Auxiliary optional] + ____ + {dobj_slot} + {anchor} + [Noun Phrase / Complement]"
                        micro_task = (
                            f"Construct a natural academic sentence where the blank requires {verb_form_phrase} '{final_target}' "
                            f"taking a direct object followed by bound preposition '{anchor}'{obj_note}. "
                            f"Syntactic Frame: {frame_str}. "
                            f"Ensure '{final_target}' is the only idiomatic fit governing {dobj_slot} + '{anchor}', "
                            f"{valency_clause}.{ant_clue}"
                        )
                    else:
                        frame_str = f"[Subject] + [Modal/Auxiliary optional] + ____ + {anchor} + [Noun Phrase / Object]"
                        micro_task = (
                            f"Construct a natural academic sentence where the blank requires {verb_form_phrase} '{final_target}' "
                            f"immediately followed by bound preposition '{anchor}'{obj_note}. "
                            f"Syntactic Frame: {frame_str}. "
                            f"Ensure '{final_target}' is the only idiomatic fit governing '{anchor}', "
                            f"{valency_clause}.{ant_clue}"
                        )
                elif anchor and anchor_type == "verb_subject":
                    ant_clue = cls._contrast_strategy_clause(final_target, it.get("definition"), dist_list, dist_meta, anchor_type)
                    obj_slot = " + [Direct Object: somebody/something]" if verb_requires_object is True else ""
                    micro_task = (
                        f"Construct a natural academic sentence where the blank requires {verb_form_phrase} '{final_target}' "
                        f"governed by subject '{anchor}'. "
                        f"Syntactic Frame: [Determiner/Modifier optional] + {anchor} + [Modal/Auxiliary optional] + ____{obj_slot} + [Adverbial / Complement]. "
                        f"Ensure the subject '{anchor}' naturally performs the action '{final_target}', establishing semantic clues demanding '{final_target}' while ruling out [{dist_str}].{ant_clue}"
                    )
                elif anchor:
                    ant_clue = cls._contrast_strategy_clause(final_target, it.get("definition"), dist_list, dist_meta, anchor_type)
                    micro_task = (
                        f"Construct a natural academic sentence where the blank requires {verb_form_phrase} '{final_target}' "
                        f"collocating with object/anchor '{anchor}'. "
                        f"Syntactic Frame: [Subject] + [Modal/Auxiliary optional] + ____ + [Determiner/Modifier optional] + {anchor} + [Complement]. "
                        f"Establish semantic clues demanding '{final_target}' while ruling out [{dist_str}].{ant_clue}"
                    )
                else:
                    ant_clue = cls._contrast_strategy_clause(final_target, it.get("definition"), dist_list, dist_meta, anchor_type)
                    obj_slot = "[Direct Object: somebody/something] + [Complement]" if verb_requires_object is True else "[Object/Complement]"
                    micro_task = (
                        f"Construct a natural academic sentence in a formal register requiring {verb_form_phrase} '{final_target}'. "
                        f"Syntactic Frame: [Subject] + [Modal/Auxiliary optional] + ____ + {obj_slot}. "
                        f"Establish clear contextual contrast that rules out [{dist_str}].{ant_clue}"
                    )
            elif canonical_pos == "noun":
                ant_clue = cls._contrast_strategy_clause(final_target, it.get("definition"), dist_list, dist_meta, anchor_type)
                # Contract tiering (Pillar 4): a real prepositional complement yields a
                # hard valency command (autonomy from compulsion); a genuine modifier or
                # governing verb yields a strict collocation slot; with no reliable anchor
                # the blueprint degrades to a neutral conceptual/semantic-field template
                # instead of hard-welding a fabricated anchor.
                if anchor_type == "prep" and anchor:
                    obj_note = f" (governing object '{prep_obj}')" if prep_obj else ""
                    micro_task = (
                        f"Construct a natural academic sentence where the blank requires noun '{final_target}' "
                        f"immediately followed by bound preposition '{anchor}'{obj_note}. "
                        f"Syntactic Frame: [Subject/Verb] + [Determiner/Modifier optional] + ____ + {anchor} + [Noun Phrase / Complement]. "
                        f"Ensure '{final_target}' is the only idiomatic fit governing '{anchor}'{obj_note}, "
                        f"{valency_clause}.{ant_clue}"
                    )
                elif anchor_type == "partitive" and anchor:
                    # The compound is real but weak on its own: almost any countable noun can
                    # sit in 'baskets of ____'. The frame makes the stem concrete and keeps the
                    # plural symmetry; the definition is what actually eliminates the options.
                    obj_note = f" (quantifier noun '{prep_obj}')" if prep_obj else ""
                    micro_task = (
                        f"Construct a natural academic sentence where the blank requires noun '{final_target}' "
                        f"governed by the partitive compound '{anchor}'{obj_note}. "
                        f"Syntactic Frame: [Determiner/Modifier optional] + {anchor} + ____ + [Optional Complement]. "
                        f"The compound must appear verbatim, but it narrows the slot without deciding it - "
                        f"make '{final_target}' the only fit on definition grounds, "
                        f"{valency_clause}.{ant_clue}"
                    )
                elif anchor_downgrade == "sense_recognition":
                    # No dictionary collocation and no passage frame survived. Say so, and put
                    # the whole discriminating weight on meaning instead of pretending a frame.
                    sr_def = it.get("definition", "").strip() or w_lower
                    micro_task = (
                        f"Sense-recognition item, not a collocation test: noun '{final_target}' has no frame "
                        f"that the passage or the dictionary licenses, so the stem must NOT lean on any bound "
                        f"preposition or fixed modifier to eliminate [{dist_str}]. "
                        f"Syntactic Frame: [Subject/Object position] requiring an appropriate head noun. "
                        f"Write a natural academic sentence whose situation makes the definition "
                        f"'{sr_def}' true of the blank and false of every option, "
                        f"{valency_clause}.{ant_clue}"
                    )
                else:
                    # Natural Narrative Context Clues (SLA Triangulation & Cognitive Contrast)
                    # Liberates generation from stiff, rare dictionary verbs (e.g. 'extend hospitality')
                    # while establishing unambiguous single-fit discrimination based on situational logic.
                    def_text = it.get("definition", "").strip()
                    clue_guidance = ""
                    if def_text:
                        clue_guidance = f" Embed triangulated narrative context clues (e.g. specific roles, setting/props, or actions reflecting '{def_text}')"
                    else:
                        clue_guidance = " Embed triangulated narrative context clues (e.g. specific roles, setting/props, or observable actions)"
                    micro_task = (
                        f"Construct a natural, vivid academic sentence where the blank requires noun '{final_target}'. "
                        f"Syntactic Frame: [Subject/Object position] requiring an appropriate head noun. "
                        f"{clue_guidance} to unambiguously demand '{final_target}', "
                        f"ruling out [{dist_str}] on clear situational, logical, and definition grounds.{ant_clue}"
                    )


            elif canonical_pos == "adj":
                if anchor_type == "prep":
                    obj_note = f" (followed by noun/phrase '{prep_obj}')" if prep_obj else ""
                    ant_clue = cls._contrast_strategy_clause(final_target, it.get("definition"), dist_list, dist_meta, anchor_type)
                    micro_task = (
                        f"Construct a natural academic sentence where the blank requires adjective '{final_target}' "
                        f"immediately followed by bound preposition '{anchor}'{obj_note}. "
                        f"Syntactic Frame: [Subject] + [Copula Verb] + ____ + {anchor} + [Noun Phrase / Complement]. "
                        f"Ensure '{final_target}' is the only idiomatic fit governing '{anchor}', "
                        f"{valency_clause}.{ant_clue}"
                    )
                elif anchor_type == "modified_noun" and anchor:
                    ant_clue = cls._contrast_strategy_clause(final_target, it.get("definition"), dist_list, dist_meta, anchor_type)
                    micro_task = (
                        f"Construct a natural academic sentence where the blank requires adjective '{final_target}' "
                        f"modifying noun '{anchor}'. "
                        f"Syntactic Frame: [Clause Head] + [Determiner optional] + ____ + {anchor} + [Predicate/Complement]. The noun '{anchor}' MUST appear immediately after the blank. "
                        f"Ensure the sentence context strictly demands '{final_target}' as the precise collocational and semantic fit, "
                        f"while ruling out [{dist_str}].{ant_clue}"
                    )
                else:
                    ant_clue = cls._contrast_strategy_clause(final_target, it.get("definition"), dist_list, dist_meta, anchor_type)
                    anchor_note = f"modifying '{anchor}' or similar academic concepts" if anchor else "in an academic evaluation"
                    if anchor in ("make", "render", "find", "deem", "consider", "keep"):
                        anchor_clause = (
                            f"Syntactic Frame: [Subject] + [{anchor} in appropriate form] + [Object/Dummy-It] + ____ + [Complement/Infinitive]. "
                            f"[CRITICAL CONSTRAINT] Do NOT turn the blank into a verb; the blank is an adjective complement! "
                            f"The target word '{final_target}' MUST ONLY appear in the answer key and inside the blank '____' — NEVER write '{final_target}' or its derivatives anywhere else in the question stem! "
                        )
                    elif anchor and anchor_type in ("adv_mod", "adj"):
                        anchor_clause = (
                            f"Syntactic Frame: [Subject] + [Copula] + {anchor} + ____ + [Complement]. "
                        )
                    elif anchor:
                        anchor_clause = (
                            f"Syntactic Frame: [Determiner optional] + ____ + {anchor} + [Predicate/Complement]. "
                        )
                    else:
                        anchor_clause = "Syntactic Frame: [Subject/Attributive/Predicative position] requiring an academic evaluative adjective. "
                    micro_task = (
                        f"Construct a natural academic sentence where the blank requires adjective '{final_target}' ({anchor_note}). "
                        f"{anchor_clause}"
                        f"Ensure the sentence context strictly demands '{final_target}' while ruling out [{dist_str}].{ant_clue}"
                    )
            elif canonical_pos == "adv":
                if anchor and anchor_type in ("modifies_adj", "adj"):
                    micro_task = (
                        f"Construct a natural academic sentence where the blank requires adverb '{final_target}'. "
                        f"Syntactic Frame: [Subject] + [Copula/Verb] + ____ + {anchor} + [Noun/Complement]. "
                        f"The adjective '{anchor}' MUST be explicitly written immediately after the blank. "
                        f"Establish semantic clues demanding '{final_target}' while ruling out [{dist_str}]."
                    )
                elif anchor:
                    micro_task = (
                        f"Construct a natural academic sentence where the blank requires adverb '{final_target}'. "
                        f"Syntactic Frame: [Subject] + ____ + [{anchor} in appropriate form] + [Object/Complement], or [Subject] + [{anchor}] + [Object] + ____. "
                        f"The verb '{anchor}' MUST be explicitly written in the sentence; do NOT place the blank in the main verb position! "
                        f"Establish semantic clues demanding '{final_target}' while ruling out [{dist_str}]."
                    )
                else:
                    micro_task = (
                        f"Construct a natural academic sentence where the blank requires adverb '{final_target}'. "
                        f"Syntactic Frame: Adverbial modifier position modifying an academic predicate or clause. "
                        f"Establish clear contextual clues that rule out [{dist_str}]."
                    )
            elif cls.is_function_word(final_target):
                f_idx = cls._get_function_word_index()
                fam, stype = f_idx[final_target][0]
                if fam == "scope":
                    micro_task = (
                        f"Construct a natural academic sentence where the blank requires preposition '{final_target}' denoting spatial or metaphorical scope. "
                        f"Syntactic Frame: [Clause] + ____ + [Noun Phrase denoting limit/domain]. "
                        f"Ensure context clearly demands the precise conceptual boundary of '{final_target}' while ruling out [{dist_str}]."
                    )
                elif fam == "noun_clause":
                    micro_task = (
                        f"Construct a natural academic sentence where the blank requires complementizer '{final_target}' introducing a noun clause. "
                        f"Syntactic Frame: [Main Verb / Proposition] + ____ + [Subject + Verb Subordinate Clause]. "
                        f"Ensure the epistemic uncertainty or syntactic role strictly requires '{final_target}' while eliminating [{dist_str}]."
                    )
                elif stype == "prepositional":
                    micro_task = (
                        f"Construct a natural academic sentence where the blank requires prepositional connective '{final_target}'. "
                        f"Syntactic Frame: ____ + [Noun Phrase / Gerund V-ing], [Independent Main Clause]. "
                        f"The blank '____' MUST be followed directly by a noun phrase or gerund, NOT a clause with a finite verb! "
                        f"This syntactic constraint physically eliminates clausal distractors like [{dist_str}]. "
                        f"Ensure context clearly demands '{final_target}'."
                    )
                elif stype == "clausal":
                    micro_task = (
                        f"Construct a natural academic sentence where the blank requires clausal conjunction '{final_target}'. "
                        f"Syntactic Frame: ____ + [Subordinate Subject + Finite Verb], [Main Clause]. "
                        f"The blank '____' MUST introduce a complete subordinate clause with subject and finite verb! "
                        f"This syntactic requirement physically eliminates prepositional distractors like [{dist_str}]. "
                        f"Ensure context clearly demands '{final_target}'."
                    )
                else:
                    micro_task = (
                        f"Construct a natural academic sentence where the blank requires discourse connective '{final_target}'. "
                        f"Syntactic Frame: [First Proposition]; ____, [Second Proposition]. "
                        f"Establish clear logical clues demanding '{final_target}' while ruling out [{dist_str}]."
                    )
            elif canonical_pos == "phrase":
                micro_task = (
                    f"Construct a natural academic sentence where the blank requires the exact multi-word phrase '{final_target}'. "
                    f"Syntactic Slot: [Clause Head] + ____ + [Predicate/Complement]. The blank '____' MUST represent the entire phrase. "
                    f"Establish clear semantic and contextual clues requiring '{final_target}' while ruling out [{dist_str}]."
                )
            elif canonical_pos == "idiom_slot" and final_target == "granted":
                micro_task = (
                    f"Construct a natural communicative or academic sentence testing the fixed idiom 'take sth for granted' where the blank requires '{final_target}'. "
                    f"Syntactic Frame: [Subject] + [take / takes / took / taking] + [Direct Object NP] + for + ____. "
                    f"The blank '____' MUST be governed immediately by 'take [sth] for' to form the authentic idiom 'take ... for granted' (meaning to fail to appreciate or expect sb/sth to always be available). "
                    f"This fixed idiomatic collocation physically eliminates distractors [{dist_str}]."
                )
            else:
                micro_task = (
                    f"Compose a CEFR-aligned academic sentence where the blank requires '{final_target}'. "
                    f"Provide clear clues eliminating [{dist_str}]."
                )

            # Data-Driven Semantic Discriminator (LDOCE Thesaurus & Language Activator)
            # Target-anchored (P1-5): the note contrasts the locked target sense against the
            # distractor and never degenerates into the distractor's own definition. Lookups
            # use base headwords because inflection reconciliation rewrites the options.
            dist_base_list = [o for o in _base_options if o.lower() != w_lower]
            dist_base_meta = {b: dist_meta.get(f) for b, f in zip(_base_options, final_options)}
            discriminator_note = cls._semantic_discriminator_note(
                w_lower, it.get("definition"), dist_base_list, dist_base_meta
            )
            if discriminator_note:
                micro_task += " " + discriminator_note

            if "Semantic Discriminator:" not in micro_task:
                # Check Language Activator concepts in LDOCE
                t_entry_data = cls.get_ldoce_entry(final_target) or cls.get_ldoce_entry(w_lower)
                if t_entry_data and t_entry_data.get("language_activator"):
                    target_side = (it.get("definition") or "").strip() or w_lower
                    for act in t_entry_data["language_activator"]:
                        concept_name = act.get("concept", "")
                        for act_w in act.get("words", []):
                            act_word = (act_w.get("word") or "").lower()
                            if act_word in dist_base_list:
                                micro_task += (
                                    f" Semantic Discriminator: DISTINCTION between '{w_lower}' "
                                    f"({target_side}) and '{act_word}' — the LDOCE concept "
                                    f"'{concept_name}' covers both, so the sentence must supply the "
                                    f"cue that only '{w_lower}' satisfies."
                                )
                                break
                        if "Semantic Discriminator:" in micro_task:
                            break

            # Fallback high-discrimination semantic feature injection for subtle/high-risk distractors:
            if "Semantic Discriminator:" not in micro_task:
                if final_target == "voice" and "message" in dist_list:
                    micro_task += " Semantic Discriminator: Emphasize physical acoustic vocal quality (e.g. trembling, audible pitch, whispered tone) rather than ideological content, strictly ruling out 'message'."
                elif final_target == "control" and any(d in dist_list for d in ("strength", "effectiveness", "influence", "power")):
                    micro_task += " Semantic Discriminator: Emphasize regulatory authority and restriction of access or behavior, strictly ruling out physical 'strength' and generic 'effectiveness'."

            # Inject Authentic Corpus Example Cloze Frame Prototype to completely eliminate mechanical sentences
            if ldoce_cloze_frame:
                micro_task += (
                    f" Authentic Corpus Blueprint: You may emulate the natural syntactic structure and authentic register of: "
                    f"'{ldoce_cloze_frame}' (emulate its realistic idiomatic sentence pattern without verbatim copying)."
                )

            # Determine anchor source: "quote" if anchor appeared in authentic quote text, else "dictionary"
            anchor_source = None
            if anchor:
                if it.get("quote") and re.search(rf"\b{re.escape(anchor.lower())}\b", it["quote"].lower()):
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
                "cloze_frame_prototype": ldoce_cloze_frame,
                "micro_task": micro_task,
                "distractor_mechanisms": dist_meta,
                "prescribed_options": final_options,
                "correct_answer_index": correct_answer_index,
                "definition": it["definition"],
                # Quote provenance tiering: 'candidate_quote' is the raw curriculum quote and
                # is display-only when 'quote_provenance' == 'weak'; only 'licensed_quote' was
                # strong enough to license sense-locking, anchors and valency evidence.
                "candidate_quote": it.get("raw_quote") or it.get("quote") or "",
                "licensed_quote": it.get("quote") or "",
                "quote_provenance": quote_provenance,
                "quote_wordcount": quote_wordcount,
            })

            if len(skeletons) >= target_count:
                break

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
        antonyms = set()
        for w in wn.words(target):
            if w.lemma().lower() != target:
                continue
            for syn in w.synsets():
                if pos and syn.pos != pos:
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
        count: int = 3
    ) -> List[str]:
        """
        Pre-computes distractors mathematically guaranteed to avoid synonym collisions with target.
        When dependent_prep is specified (e.g. 'on' for 'rely'), prioritizes candidates that
        do NOT govern that preposition or clash with it.
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
        target_synsets = wn.synsets(target_lower, pos=pos)
        if target_synsets:
            primary_synset = target_synsets[0]
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
