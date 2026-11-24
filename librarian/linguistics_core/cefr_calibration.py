"""
Deterministic CEFR difficulty calibration, Zipf frequency gates, and ceiling calculations.
"""

import logging
import math
import os
import re
import sqlite3
import threading
from contextvars import ContextVar
from functools import lru_cache
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union

logger = logging.getLogger("librarian.linguistics")

# Thread-safe and async-safe context variable for the currently active passage CEFR level.
_CURRENT_PASSAGE_CEFR: ContextVar[Optional[str]] = ContextVar("_CURRENT_PASSAGE_CEFR", default=None)


class CefrCalibrationMixin:
    _cefr_analyzer = None
    CEFR_ORDER = {"A1": 1, "A2": 2, "B1": 3, "B2": 4, "C1": 5, "C2": 6}

    @classmethod
    @lru_cache(maxsize=32768)
    def _zipf_freq(cls, word: str) -> float:
        if not word:
            return 0.0
        try:
            from wordfreq import zipf_frequency
            return float(zipf_frequency(word.strip().lower(), "en"))
        except Exception:
            return 0.0

    @classmethod
    def word_zipf(cls, word: str) -> float:
        return cls._zipf_freq(word)

    # Authoritative Closed Paradigms for Grammatical / Function Words
    # Organized by Functional Discourse Family and Syntactic Complement Requirements:
    # - clausal: Governs finite clause (Subject + Finite Verb)
    # - prepositional: Governs noun phrase / gerund (NP / V-ing)
    # - adverbial: Independent discourse adverbial adjunct
    _FUNCTION_WORD_PARADIGMS = {
        "concession": {
            "clausal": ["although", "though", "even though", "while", "whereas", "albeit"],
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

    # Closed-class function words / grammar stopwords excluded from passage CEFR lexical density
    # so function-word dilution (~55-60% of running tokens at A1=1.0) never artificially caps
    # technical/academic C1 passages (e.g. IELTS reading) at B2.
    _CEFR_FUNCTION_WORDS = {
        "a", "an", "the", "this", "that", "these", "those", "my", "your", "his", "her", "its",
        "our", "their", "mine", "yours", "hers", "ours", "theirs", "i", "you", "he", "she",
        "it", "we", "they", "me", "him", "us", "them", "myself", "yourself", "himself",
        "herself", "itself", "ourselves", "themselves", "who", "whom", "whose", "which", "what",
        "where", "when", "why", "how", "be", "am", "is", "are", "was", "were", "been", "being",
        "have", "has", "had", "having", "do", "does", "did", "doing", "done", "will", "would",
        "shall", "should", "can", "cannot", "could", "may", "might", "must", "ought", "in", "on", "at",
        "to", "for", "of", "with", "by", "from", "up", "down", "out", "off", "over", "under",
        "into", "onto", "upon", "about", "above", "below", "between", "among", "through",
        "during", "before", "after", "since", "until", "till", "around", "along", "across",
        "behind", "beyond", "within", "without", "against", "toward", "towards", "near",
        "and", "but", "or", "nor", "so", "yet", "if", "because", "as", "than", "then",
        "while", "whilst", "although", "though", "whether", "unless", "not", "no", "yes", "very", "too",
        "also", "just", "only", "even", "still", "already", "always", "never", "often",
        "sometimes", "usually", "again", "here", "there", "now", "well", "much", "many",
        "more", "most", "some", "any", "all", "both", "each", "every", "other", "another",
        "such", "same", "own", "one", "ones", "two", "three", "four", "five", "six", "seven",
        "eight", "nine", "ten", "first", "second", "third",
    }

    # Measurement units, era markers, and common initialisms that carry arbitrary C1/C2 noise in cefrpy
    _CEFR_NOISE_ABBREVS = {
        "cm", "mm", "km", "kg", "mg", "ml", "bc", "ad", "am", "pm", "uv", "led", "cfl",
        "ioc", "dna", "uk", "us", "tv", "pc", "id", "ok", "mr", "mrs", "ms", "dr", "st", "etc",
    }

    # Cached reverse map from irregular verb forms ('went', 'taken', 'led') -> base lemma
    _IRREGULAR_SURFACE_TO_BASE: Optional[Dict[str, str]] = None

    # High-register academic subordination, pied-piped relative, and discourse framing markers
    _CEFR_ACADEMIC_SYNTACTIC_MARKERS = re.compile(
        r"\b(?:whilst|whereas|whereby|wherein|in\s+which|on\s+which|to\s+which|by\s+which|for\s+which|"
        r"prior\s+to|so\s+as\s+to|not\s+only|in\s+this\s+respect|provided\s+that|as\s+well\s+as|"
        r"despite\s+this|rather\s+than|accompanied\s+by|confronted\s+with|facilitated\s+by|"
        r"classified\s+under|characterized\s+by|attributed\s+to|associated\s+with|regardless\s+of|"
        r"notwithstanding|subsequent\s+to|albeit|thereby|moreover|furthermore|nevertheless|nonetheless|"
        r"consequently|on\s+the\s+other\s+hand|in\s+particular)\b",
        re.I,
    )

    @classmethod
    def _fast_cefr_lemma_and_score(cls, w: str, analyzer: Any) -> Tuple[str, Optional[float]]:
        """
        Zero-spaCy (<0.01ms) morphological de-inflection for passage CEFR calculation.
        Maps inflected surface forms ('electrodes', 'diodes', 'emitting', 'compared') to their
        canonical base lemma without invoking spaCy pipelines.
        """
        if cls._IRREGULAR_SURFACE_TO_BASE is None:
            cls._IRREGULAR_SURFACE_TO_BASE = {
                f: base
                for base, forms in cls._IRREGULAR_VERB_FORMS.items()
                for f in forms
                if f
            }
        if w in cls._IRREGULAR_SURFACE_TO_BASE:
            base = cls._IRREGULAR_SURFACE_TO_BASE[w]
            return base, analyzer.get_average_word_level_float(base)

        cands = cls._headword_candidates(w)
        w_lvl = analyzer.get_average_word_level_float(w)
        if len(cands) > 1:
            best_lem: Optional[str] = None
            best_lvl: Optional[float] = None
            for c in cands[1:]:
                if len(c) < 3:
                    continue
                if c in cls._CEFR_FUNCTION_WORDS:
                    return c, 1.0
                clvl = analyzer.get_average_word_level_float(c)
                if clvl is not None and (best_lvl is None or clvl < best_lvl):
                    best_lem = c
                    best_lvl = clvl

            if best_lem is not None and best_lvl is not None:
                if w_lvl is None or best_lvl <= w_lvl:
                    w, w_lvl = best_lem, best_lvl
                elif w.endswith("s") and not w.endswith(("ss", "us", "is", "as", "os")) and best_lvl <= 5.95:
                    w, w_lvl = best_lem, best_lvl

        # Compositional derivational prefix-base monotonicity guard:
        # Productive negative/degree prefixes ('dis-', 'under-', 'over-', 'mis-', 'counter-', 'non-')
        # attached to an attested B1+ stem ('dissimilarity' <- 'similarity', 'disproportionate' <-
        # 'proportionate', 'underperforming' <- 'performing') cannot have a lower CEFR level than
        # their simpler morphological base.
        for pref in ("dis", "under", "over", "mis", "counter", "non"):
            if w.startswith(pref) and len(w) - len(pref) >= 6:
                stem = w[len(pref):]
                b_lvl = analyzer.get_average_word_level_float(stem)
                if b_lvl is None and stem.endswith("ing") and len(stem) >= 6:
                    b_lvl = analyzer.get_average_word_level_float(stem[:-3])
                if b_lvl is not None and b_lvl <= 5.5 and (
                    (w_lvl is None and b_lvl >= 1.5)
                    or (b_lvl >= 3.0 and w_lvl is not None and w_lvl <= b_lvl)
                ):
                    bonus = 1.5 if w.endswith(("ity", "tion", "sion", "ment", "ance", "ence", "ate")) else 1.0
                    w_lvl = min(5.3, max(4.0 if w_lvl is None else w_lvl, b_lvl + bonus))
                    break

        return w, w_lvl

    @classmethod
    def _compute_syntactic_complexity_bonus(cls, text: str) -> float:
        """
        Computes a zero-spaCy (<0.5ms) sentence & syntactic complexity bonus (0.00..0.35)
        for passage CEFR calibration, combining:
        1. Mean sentence length (`avg_sent_len >= 18.0`)
        2. Multi-clause long-sentence tail ratio (`% of sentences >= 22 words`)
        3. Formal academic subordination / discourse framing density (`_CEFR_ACADEMIC_SYNTACTIC_MARKERS`)
        4. Intra-sentence clausal punctuation embedding depth (commas, semicolons, colons, parentheses, dashes)
        """
        prose_body = re.split(r"\n#+\s*(?:Word\s*List|Vocabulary|New\s*Words)", text, flags=re.I)[0]
        sents = [
            s.strip()
            for s in re.split(r"[.!?]+(?:\s+|$)", prose_body)
            if len(re.findall(r"[A-Za-z]+", s)) >= 4
        ]
        if len(sents) < 3:
            return 0.0

        sent_lens = [len(re.findall(r"[A-Za-z]+", s)) for s in sents]
        n_sents = len(sents)
        avg_sl = sum(sent_lens) / n_sents
        long22_ratio = sum(1 for ln in sent_lens if ln >= 22) / n_sents
        punct_per_sent = sum(len(re.findall(r"[,;:()\u2014\u2013-]", s)) for s in sents) / n_sents
        marker_rate = len(cls._CEFR_ACADEMIC_SYNTACTIC_MARKERS.findall(prose_body)) / n_sents * 10.0

        bonus = 0.0
        if avg_sl >= 18.0:
            bonus += min(0.10, (avg_sl - 18.0) * 0.015)
        if long22_ratio >= 0.25:
            bonus += min(0.08, (long22_ratio - 0.25) * 0.25)
        if marker_rate >= 0.8:
            bonus += min(0.12, (marker_rate - 0.5) * 0.04)
        if punct_per_sent >= 1.25:
            bonus += min(0.05, (punct_per_sent - 1.20) * 0.06)
        return bonus

    @classmethod
    def calculate_text_cefr(cls, text_or_words: Union[str, List[str]], default: str = "B1") -> str:
        """
        Statistically computes the authentic, objective CEFR level ('A1'..'C2') of a passage
        or word collection using offline CEFR-J + Cambridge frequency scores combined with
        sentence and syntactic complexity (<5ms, zero token cost).

        Key Psychometric & Corpus Calibration Principles:
        1. Function-Word Exclusion (`_CEFR_FUNCTION_WORDS`):
           Excludes closed-class grammatical stopwords ('the', 'is', 'of', 'to', 'and', pronouns,
           prepositions, auxiliaries) which account for ~55-60% of English running text at score 1.0
           and would otherwise mathematically crush simple-syntax/high-terminology C1 passages (e.g. IELTS
           science/history texts) down to ~1.9-2.0.
        2. Proper-Noun & Unit Filtering:
           Excludes mid-sentence capitalized proper nouns (person/place names like 'Beijing', 'Rosetta',
           'Bonaparte'), contractions, and measurement units/initialisms ('cm', 'kg', 'bc', 'led', 'cfl')
           so beginner texts are never falsely inflated by recurring proper names.
        3. Lemma-Normalized Content-Word Type Distribution (`typ_vals`) + Syntactic Complexity (`syn_bonus`):
           Evaluates unique content-word lemmas via mean (`typ_avg`), 80th/90th percentiles (`p80`, `p90`),
           advanced lexical type ratios (`B2+ >= 3.5`, `C1+ >= 4.5`, `C2 >= 5.3`), and sentence/syntactic
           complexity (`_compute_syntactic_complexity_bonus`). When an academic passage combines sustained
           multi-clause syntactic complexity (`syn_bonus >= 0.14`) with a dense C2 lexical spike
           (`c2_cnt >= 14`, `c2_typ >= 5%`) and `eff_avg >= 2.68`, it resolves to C1 even when concrete
           illustrative examples keep the raw 90th percentile around `4.0`.
        """
        analyzer = cls.get_cefr_analyzer()
        if not analyzer:
            return default

        type_scores: Dict[str, List[float]] = {}
        fallback_scores: List[float] = []
        syn_bonus = 0.0

        if isinstance(text_or_words, str):
            syn_bonus = cls._compute_syntactic_complexity_bonus(text_or_words)
            for m in re.finditer(r"[A-Za-z]+(?:['\u2019][A-Za-z]+)*", text_or_words):
                raw_tok = m.group(0)
                start = m.start()
                if raw_tok.lower().endswith(("'s", "\u2019s")):
                    raw_tok = raw_tok[:-2]
                if not raw_tok or "'" in raw_tok or "\u2019" in raw_tok:
                    continue
                w = raw_tok.lower()
                lvl = analyzer.get_average_word_level_float(w)
                if lvl is not None:
                    fallback_scores.append(lvl)
                if len(w) <= 2 or w in cls._CEFR_FUNCTION_WORDS or w in cls._CEFR_NOISE_ABBREVS:
                    continue

                prefix = text_or_words[max(0, start - 4):start]
                is_sent_start = (start == 0) or bool(re.search(r"[.!?:\n\r\"'\u201c\u2018(\[]\s*$", prefix))
                if not is_sent_start and raw_tok[0].isupper():
                    continue

                lem, score = cls._fast_cefr_lemma_and_score(w, analyzer)
                if lem in cls._CEFR_FUNCTION_WORDS or lem in cls._CEFR_NOISE_ABBREVS or score is None:
                    continue
                type_scores.setdefault(lem, []).append(score)
        else:
            for item in text_or_words:
                if not item:
                    continue
                w = re.sub(r"[^a-zA-Z]", "", str(item)).lower()
                if not w:
                    continue
                lvl = analyzer.get_average_word_level_float(w)
                if lvl is not None:
                    fallback_scores.append(lvl)
                if len(w) <= 2 or w in cls._CEFR_FUNCTION_WORDS or w in cls._CEFR_NOISE_ABBREVS:
                    continue
                lem, score = cls._fast_cefr_lemma_and_score(w, analyzer)
                if lem in cls._CEFR_FUNCTION_WORDS or lem in cls._CEFR_NOISE_ABBREVS or score is None:
                    continue
                type_scores.setdefault(lem, []).append(score)

        if not type_scores:
            if not fallback_scores:
                return default
            avg_fb = sum(fallback_scores) / len(fallback_scores)
            if avg_fb <= 1.60:
                return "A1"
            elif avg_fb <= 2.10:
                return "A2"
            elif avg_fb <= 2.75:
                return "B1"
            elif avg_fb <= 3.50:
                return "B2"
            elif avg_fb <= 4.35:
                return "C1"
            return "C2"

        typ_vals = sorted(sum(v) / len(v) for v in type_scores.values())
        n_typ = len(typ_vals)
        typ_avg = sum(typ_vals) / n_typ

        # Short sentence / small wordlist fast path (< 8 unique content lemmas)
        if n_typ < 8:
            if typ_avg <= 1.80:
                return "A1"
            elif typ_avg <= 2.25:
                return "A2"
            elif typ_avg <= 2.85:
                return "B1"
            elif typ_avg <= 3.60:
                return "B2"
            elif typ_avg <= 4.50:
                return "C1"
            return "C2"

        p80 = typ_vals[int(n_typ * 0.80)]
        p90 = typ_vals[int(n_typ * 0.90)]
        b2_typ = sum(1 for s in typ_vals if s >= 3.5) / n_typ
        c1_typ = sum(1 for s in typ_vals if s >= 4.5) / n_typ
        c2_cnt = sum(1 for s in typ_vals if s >= 5.3)
        c2_typ = c2_cnt / n_typ

        eff_avg = typ_avg + syn_bonus
        c1_syntactic_bridge = (
            syn_bonus >= 0.14
            and eff_avg >= 2.68
            and p90 >= 4.0
            and c2_cnt >= 14
            and c2_typ >= 0.05
        )

        # Content-Word Type + Sentence Complexity Calibration across CEFR A1..C2:
        if typ_avg <= 1.82 and p80 <= 2.5 and b2_typ < 0.05:
            return "A1"
        elif typ_avg <= 2.14 and p80 <= 3.0 and p90 <= 3.6 and b2_typ < 0.13 and c1_typ < 0.04:
            return "A2"
        elif (typ_avg <= 2.47 and eff_avg <= 2.58) and p80 <= 3.5 and p90 <= 4.1 and c1_typ < 0.082:
            return "B1"
        elif not c1_syntactic_bridge and (typ_avg < 2.55 or p90 < 4.35 or c1_typ < 0.095):
            return "B2"
        elif eff_avg <= 3.40 and c1_typ < 0.25:
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
        # Exclude numbers, digits, decades, years, or temporal numerical tokens
        if re.search(r"\d", w_clean):
            return False
        w_lower = w_clean.lower()

        # Cardinal number words
        _CARDINAL_NUMBERS = {
            "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
            "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
            "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
            "hundred", "thousand", "million", "billion", "trillion",
        }
        if w_lower in _CARDINAL_NUMBERS or w_lower.rstrip("s") in _CARDINAL_NUMBERS:
            return False

        # Ultra-Basic Function Words & Low-SLA Trivia / Commodity / Mineral Filter
        ultra_basic_or_non_pedagogical_words = {
            "but", "and", "or", "so", "if", "because", "then", "than", "too",
            "very", "also", "just", "now", "not", "no", "yes", "oh", "ah",
            "he", "she", "it", "they", "we", "you", "i", "me", "him", "her", "us", "them",
            # Everyday food / toiletry / household trivia nouns (inflated by low academic Zipf frequency)
            "toothpaste", "toothbrush", "shampoo", "soap", "lotion", "cosmetics", "toiletries",
            "popcorn", "chocolate", "candy", "cookie", "biscuit", "pizza", "burger", "soda",
            "groceries", "refreshments", "goody", "goodies", "trash", "housewares",
            # Pure encyclopedic geological rocks / chemical elements / compounds (no SLA collocation value)
            "basalt", "granite", "limestone", "sandstone", "tungsten", "dioxide", "monoxide",
        }
        if w_lower in ultra_basic_or_non_pedagogical_words:
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
    def is_pedagogical_expression(cls, phrase: str, definition: Optional[str] = None) -> bool:
        """
        Gating function to ensure candidate multi-word expressions, collocations, phrasal verbs,
        and idioms are pedagogically sound SLA vocabulary targets.
        Excludes:
        1. Any phrase containing digits, decades, or years (e.g. 'the early 1980s', 'in 1985', '2 percent').
        2. Transparent quantifier/partitive phrases of quantity or currency (e.g. 'millions of dollars',
           'thousands of people', 'hundreds of years').
        3. Phrases mapped to pure cardinal number definitions (e.g. 'the number 1,000,000').
        """
        p_clean = re.sub(r'\[.*?\]|\(.*?\)', '', str(phrase or '')).strip()
        if not p_clean:
            return False
        p_low = p_clean.lower()

        # 1. Reject digits or temporal decade/year patterns
        if re.search(r"\d", p_clean):
            return False

        # 2. Reject transparent quantifier / numeral + of + noun expressions
        if re.search(
            r"^(?:millions?|billions?|thousands?|hundreds?|dozens?|scores?)\s+of\s+(?:dollars?|pounds?|euros?|cents?|money|cash|currency|people|person|man|men|woman|women|child|children|students?|workers?|years?|months?|weeks?|days?|hours?|minutes?|seconds?|miles?|tons?)\b",
            p_low,
        ):
            return False

        # 3. Reject if definition indicates a cardinal number
        if definition:
            d_low = str(definition).strip().lower()
            if re.match(r"^the number [\d,]+", d_low) or d_low.startswith("the number "):
                return False

        return True

    @classmethod
    def _lexicon_profile(cls, form: str) -> Tuple[bool, bool, Optional[str], str]:
        """
        What the PedLex lexicon and CEFR analyzer say about a surface form.

        An inflected form is not itself a headword - 'attaches', 'children' and 'buses'
        appear in neither PedLex nor the CEFR headword list even though 'attach', 'child' and
        'bus' appear in both - so a gate that reads the surface literally treats every
        inflected distractor as obscure and rejects it for A1/A2/B1 targets. An empty
        literal lookup is therefore retried through the lemma. The retry is lazy: the
        ordinary single-word path never pays a spaCy call for it.

        Returns (in_pedlex, is_academic_cefr, cefr_level_or_None, form_the_lookup_resolved_to).
        """
        f = (form or "").strip().lower()
        if not f:
            return (False, False, None, "")
        in_pedlex = cls.is_exact_pedlex_headword(f)
        level = cls.get_word_cefr(f)
        resolved = f
        if not (in_pedlex or level):
            lemma = cls._surface_lemma(f)
            if lemma:
                resolved = lemma
                in_pedlex = cls.is_exact_pedlex_headword(lemma)
                level = cls.get_word_cefr(lemma)
        is_acad = level in ("B2", "C1")
        return (in_pedlex, is_acad, level, resolved)

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
    def get_current_passage_cefr(cls) -> Optional[str]:
        """Returns the currently active passage CEFR level from context."""
        return _CURRENT_PASSAGE_CEFR.get()

    @classmethod
    def set_current_passage_cefr(cls, level: Optional[str]) -> Any:
        """Sets the active passage CEFR level in context. Returns reset token."""
        lvl = str(level).strip().upper() if level else None
        if lvl and lvl not in cls.CEFR_ORDER:
            lvl = None
        return _CURRENT_PASSAGE_CEFR.set(lvl)

    @classmethod
    def reset_current_passage_cefr(cls, token: Any) -> None:
        """Resets the active passage CEFR level using the token returned by set_current_passage_cefr."""
        if token is not None:
            _CURRENT_PASSAGE_CEFR.reset(token)

    @classmethod
    def is_cefr_compliant_distractor(
        cls,
        candidate: str,
        target_word: str,
        passage_level: Optional[str] = None,
    ) -> bool:
        """
        Enforces the SLA Lexical Accessibility Matrix & Dual-Track Distractor Ceiling Principle:
        1. Distractor must NOT be significantly harder than either the target word or the passage context:
           effective_ceiling = min(Ceiling(Target_CEFR), Global_Ceiling(Passage_CEFR))
           - For A1/A2 context: distractor hard ceiling is B1 (never B2, C1, C2, or obscure unlisted).
           - For B1 context: distractor hard ceiling is B2 (never C1, C2, or obscure unlisted).
           - For B2 context: distractor hard ceiling is C1.
        2. SLA Foundational Gates across CEFR tiers:
           - A1 contexts: Distractor must belong to LDOCE core 1000/2000/3000 defining lexicon or high-frequency
             (core_level in (1, 2, 3) or freq_tags). Academic Word List (AWL) terms strictly banned.
             B1+ distractors must have high spoken/written frequency tags.
           - A2 contexts: Distractor must belong to core defining lexicon or high-frequency; AWL terms require high frequency.
           - B1 contexts: Distractor must belong to core 3000, high-frequency, or recognized AWL vocabulary.
             Rejects rare/obscure non-core non-AWL intruders.
        3. Lexicon Filter: candidate must exist in PedLex or CEFR database.
        4. Morphological Anomaly Gate:
           - Max length <= max(11, len(target) + 4) (rejects 14+ letter monsters like 'conceivableness').
           - Rejects obscure multi-affix formations ('-ableness', '-lessness', '-icalness', '-fulness').
        5. Applied to the surface form the student reads.
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

        # Lexicon Filter: must be recognized in PedLex or CEFR (through the lemma when the
        # candidate is an inflection rather than a headword).
        in_pedlex, is_acad, c_lvl_str, resolved = cls._lexicon_profile(c_clean)

        analyzer = cls.get_cefr_analyzer()
        if not analyzer:
            return in_pedlex or is_acad

        t_lvl_str = cls._lexicon_profile(t_clean)[2] or "B2"
        t_rank = cls.CEFR_ORDER.get(t_lvl_str, 4)

        # Dual-Track Ceiling Resolution:
        # Reconciles micro target word CEFR with macro passage context CEFR.
        effective_passage = (passage_level or cls.get_current_passage_cefr() or "").strip().upper()
        p_rank = cls.CEFR_ORDER.get(effective_passage) if effective_passage in cls.CEFR_ORDER else None

        t_ceiling_rank = cls._ceiling_rank(t_rank, "support")
        if p_rank is not None:
            p_ceiling_rank = cls._ceiling_rank(p_rank, "support")
            ceiling_rank = min(t_ceiling_rank, p_ceiling_rank)
            effective_base_rank = min(t_rank, p_rank)
        else:
            ceiling_rank = t_ceiling_rank
            effective_base_rank = t_rank

        if not c_lvl_str:
            # Candidate is unlisted in CEFR database!
            # If target or passage is A1, A2, or B1, completely reject unlisted candidates (e.g. 'conceivableness')
            if effective_base_rank <= 3:
                return False
            # For B2/C1 targets, only allow if verified in PedLex
            return in_pedlex

        c_rank = cls.CEFR_ORDER.get(c_lvl_str, 4)

        # Distractor Ceiling Gate: a distractor may never sit above the ceiling
        # its target/passage band licenses (A1/A2 -> B1, B1 -> B2, B2 -> C1, C1+ -> C2).
        # Exception: Direct morphological negative antonyms (e.g. unpunctual from punctual, imperfect from perfect)
        # share the exact base lemma and are transparently understood by learners, exempting them from CEFR ceiling.
        _polar_duals = (
            ("on", "off"), ("off", "on"),
            ("in", "out"), ("out", "in"),
            ("in", "de"), ("de", "in"),
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

        # SLA & Psychometric Ceiling Gate for Foundational Levels (A1, A2, B1):
        # A distractor must come from the foundational core vocabulary:
        # (LDOCE 3000 defining lexicon: core_level >= 1 or non-empty freq_tags, or is_awl, or inflection of such).
        # Rejects obscure/C1/C2 words like 'subjugation', 'pretext', 'commiserate', 'impasse', 'salvation'
        # which cefrpy heuristically/statistically misclassifies as A2 or B1.
        if effective_base_rank <= 3 and not is_direct_negation and not cls._is_ldoce_disabled():
            c_entry = cls.get_ldoce_entry(c_clean)
            c_core = (c_entry.get("core_level") or 0) if c_entry else 0
            c_freq = bool(c_entry.get("freq_tags")) if c_entry else False
            c_awl = bool(c_entry.get("is_awl")) if c_entry else False

            # If surface form does not have core/frequency tags (e.g. inflected forms like 'children', 'boxes', 'attaches'),
            # inspect candidate lemmas/headwords to inherit defining core properties.
            if not (c_core >= 1 or c_freq):
                lem_cands = set(cls._headword_candidates(c_clean))
                s_lem = cls._surface_lemma(c_clean)
                if s_lem:
                    lem_cands.add(s_lem)
                if resolved and resolved != c_clean:
                    lem_cands.add(resolved)
                for lem in lem_cands:
                    if not lem or lem == c_clean:
                        continue
                    lem_entry = cls.get_ldoce_entry(lem)
                    if lem_entry:
                        lem_core = lem_entry.get("core_level") or 0
                        lem_freq = bool(lem_entry.get("freq_tags"))
                        lem_awl = bool(lem_entry.get("is_awl"))
                        if lem_core or lem_freq:
                            c_core = max(c_core, lem_core)
                            c_freq = c_freq or lem_freq
                            c_awl = lem_awl
                            break

            if effective_base_rank == 1:
                # A1 foundational gate:
                # 1. Distractor must belong to LDOCE core defining lexicon (1000/2000/3000) or high-frequency
                if not (c_core >= 1 or c_freq):
                    return False
                # 2. Academic Word List (AWL) abstract terms strictly barred in A1
                if c_awl:
                    return False
                # 3. B1+ distractors barred unless bearing high spoken/written frequency tags
                if c_rank >= 3 and not c_freq:
                    return False
            elif effective_base_rank == 2:
                # A2 foundational gate:
                if not (c_core >= 1 or c_freq):
                    return False
                if c_awl and not c_freq:
                    return False
                # +1 Band Frequency Lock: B1+ distractors in A2 must bear everyday spoken/written frequency tags
                # (blocks low-frequency cultural/domain words like 'choir'/'orchestra' lacking S1..S3/W1..W3).
                if c_rank > 2 and not c_freq:
                    return False
            else:
                # B1 intermediate gate:
                # If distractor sits above B1 (i.e. c_rank > 3, B2 ceiling), it must be grounded in
                # everyday frequency, AWL, or LDOCE 1000/2000 core vocabulary (core_level >= 2).
                if c_rank > 3 and not (c_freq or c_awl or c_core >= 2):
                    return False

        return in_pedlex or is_acad or analyzer.is_word_in_database(resolved) or is_direct_negation

    @classmethod
    def distractor_familiarity_key(
        cls,
        word: str,
        target_word: str = "",
        passage_level: Optional[str] = None,
    ) -> Tuple[int, int, int, int]:
        """Sort key (ascending = more familiar) prioritizing base-level, high-frequency, core-1000/2000 words."""
        w_clean = (word or "").strip().lower()
        t_clean = (target_word or "").strip().lower()
        _, _, c_lvl_str, _ = cls._lexicon_profile(w_clean)
        c_rank = cls.CEFR_ORDER.get(c_lvl_str or "", 4)

        t_lvl_str = cls._lexicon_profile(t_clean)[2] if t_clean else None
        t_rank = cls.CEFR_ORDER.get(t_lvl_str or "", 3)
        eff_p = (passage_level or cls.get_current_passage_cefr() or "").strip().upper()
        p_rank = cls.CEFR_ORDER.get(eff_p) if eff_p in cls.CEFR_ORDER else None
        base_rank = min(t_rank, p_rank) if p_rank is not None else t_rank

        entry = cls.get_ldoce_entry(w_clean) if not cls._is_ldoce_disabled() else None
        core_lvl = (entry.get("core_level") or 0) if entry else 0
        freq_tags = (entry.get("freq_tags") or []) if entry else []
        # Score S1/W1=3, S2/W2=2, S3/W3=1, none=0 (summed across spoken & written)
        freq_score = 0
        for tag in freq_tags:
            t_s = str(tag).upper()
            if t_s in ("S1", "W1"):
                freq_score += 3
            elif t_s in ("S2", "W2"):
                freq_score += 2
            elif t_s in ("S3", "W3"):
                freq_score += 1

        is_above_base = 1 if c_rank > base_rank else 0
        return (is_above_base, -freq_score, -core_lvl, c_rank)

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
            if (
                len(token) < floor
                or token in allowed
                or token in cls._CEFR_FUNCTION_WORDS
                or token in cls._CEFR_NOISE_ABBREVS
                or token in seen
            ):
                continue
            seen.add(token)
            if any(mark in token for mark in ("'", "\u2019")):
                continue
            _in_pedlex, _in_awl, level, resolved = cls._lexicon_profile(token)
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

