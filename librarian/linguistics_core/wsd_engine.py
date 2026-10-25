"""
Two-stage Neuro-Symbolic Word Sense Disambiguation (_lock_sense + 1-token LLM arbitration) and expression sense resolution.
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
import logging
import os
import re
import sqlite3
import threading
import zlib
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

logger = logging.getLogger("librarian.linguistics")



class WsdEngineMixin:

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

    @classmethod
    def _sense_pos_codes(cls, sense: Dict[str, Any]) -> Set[str]:
        """Return all normalized POS codes declared on a sense (`pos` + `all_poses`)."""
        codes: Set[str] = set()
        if not isinstance(sense, dict):
            return codes
        p0 = cls._pos_code(sense.get("pos"))
        if p0:
            codes.add(p0)
        for p in (sense.get("all_poses") or []):
            pc = cls._pos_code(p)
            if pc:
                codes.add(pc)
        return codes

    @classmethod
    def _sense_matches_pos(cls, sense: Dict[str, Any], target_p: str) -> bool:
        """True if `sense` supports normalized POS code `target_p` via `pos` or `all_poses`."""
        if not target_p:
            return True
        return target_p in cls._sense_pos_codes(sense)

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
        pat_clean = re.sub(
            r"\b(?:(?:especially|mainly|chiefly|usually|mostly)\s+)?(?:american|british|australian)\s+english\b",
            "",
            (pattern_fragment or "").lower()
        )
        raw = re.findall(r"\b[a-zA-Z]{2,}\b", pat_clean)
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
                WHERE host_word = ? AND sense_idx >= 0 AND cue_word IN ({placeholders})
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
        pat_clean = re.sub(
            r"\b(?:(?:especially|mainly|chiefly|usually|mostly)\s+)?(?:american|british|australian)\s+english\b",
            "",
            pat.lower()
        )
        pat_clean = re.sub(r"\betc\.?\b", "", pat_clean).strip()
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
                    if target_p and not cls._sense_matches_pos(s, target_p):
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
        block_poses: Dict[Any, Set[str]] = {}
        for h in entry.get("homographs") or []:
            h_num = h.get("homograph_num")
            h_codes = {cls._pos_code(h.get("pos"))} if h.get("pos") else set()
            for hp in (h.get("all_poses") or []):
                hpc = cls._pos_code(hp)
                if hpc:
                    h_codes.add(hpc)
            block_poses[h_num] = {c for c in h_codes if c}

        target_w_clean = (entry.get("word") or "").lower()
        headword_tokens = set(re.findall(r"\b[a-z]+\b", target_w_clean))
        target_inflections = set(cls.inflected_forms(target_w_clean)) | headword_tokens | {target_w_clean}
        for hw_t in headword_tokens:
            target_inflections.update(cls.inflected_forms(hw_t))

        has_pron_obj = False
        target_tok_morph_number = None
        target_tok_is_predicative = False
        target_tok_is_attributive = False
        doc_q = None
        if quote and len(senses) > 1:
            try:
                nlp = cls.get_spacy()
                doc_q = nlp(cls._strip_lexical_quotes(quote))
                doc = doc_q
                matching_toks = [
                    t for t in doc
                    if t.lemma_.lower() == target_w_clean or t.text.lower() == target_w_clean or
                    cls.form_matches(t.text.lower(), target_w_clean) or
                    cls.form_matches(t.lemma_.lower(), target_w_clean) or
                    (headword_tokens and (t.lemma_.lower() in headword_tokens or t.text.lower() in headword_tokens))
                ]
                if matching_toks:
                    mt0 = matching_toks[0]
                    has_pron_obj = any(ch.dep_ in ('dobj', 'obj') and ch.pos_ == 'PRON' for ch in mt0.children)
                    if mt0.pos_ == 'NOUN':
                        target_tok_morph_number = mt0.morph.get('Number')
                    if mt0.dep_ in ('acomp', 'oprd') or (
                        mt0.head != mt0
                        and mt0.head.lemma_.lower() in ('be', 'become', 'seem', 'feel', 'get', 'remain', 'stay', 'grow', 'turn', 'appear', 'look', 'sound')
                        and mt0.dep_ != 'amod'
                    ):
                        target_tok_is_predicative = True
                    elif mt0.dep_ == 'amod' and mt0.head.pos_ in ('NOUN', 'PROPN') and mt0.i < mt0.head.i:
                        target_tok_is_attributive = True
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
        if quote and target_w_clean and not entry.get("_is_synthetic_pool"):
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
                                for bio_m in re.findall(r"\b(plant|plants|animal|animals|crop|crops|disease|diseases|organism|organisms|bacteria|virus|computer|computers|electronic|screen|screens|device|devices|phone|telephone|mobile|digital|internet|software|wireless)\b", cdef):
                                    second_order_domain_tokens.add(bio_m)
                                    b_lem = cls._surface_lemma(bio_m)
                                    if b_lem:
                                        second_order_domain_tokens.add(b_lem)
                except Exception:
                    pass

        # Map phrasal verb senses to their mandatory phrasal particles/words (e.g. 'pay back' -> ['back'])
        _pv_slot_stop = {"something", "somebody", "someone", "sth", "sb", "<->", "\u2194", "oneself", "one's", "etc"}
        pv_sense_map: Dict[int, List[str]] = {}
        for pv in entry.get("phrasal_verbs") or []:
            phr = re.sub(r"[\(\)\[\]]", " ", pv.get("phrase", ""))
            p_words = [
                w for w in phr.lower().split()
                if w != target_w_clean and not cls.form_matches(w, target_w_clean) and w not in _pv_slot_stop
            ]
            if p_words:
                for pvs in pv.get("senses") or []:
                    p_defn = pvs.get("definition")
                    for s_idx, s_cand in enumerate(senses):
                        if (
                            s_cand.get("definition") == p_defn
                            and not s_cand.get("sensenum")
                            and s_cand.get("homograph_num") in (None, 0)
                        ):
                            pv_sense_map[s_idx] = p_words

        for i, s in enumerate(senses):
            s_pos = cls._pos_code(s.get("pos"))
            if target_p and not cls._sense_matches_pos(s, target_p):
                continue
            eff_pos = target_pos or s.get("pos")
            eff_p = target_p or s_pos

            eligible_senses += 1
            score = 0

            # 0. Homograph block agreement: the sense that lives in the block which
            # declares the target POS outranks a same-POS sense borrowed from another
            # block, instead of losing to whichever sense happens to come first.
            if target_p and target_p in block_poses.get(s.get("homograph_num"), set()):
                score += 10
            elif target_p and s_pos == target_p:
                score += 4

            # 0a. Phrasal Verb Particle Gate:
            # If this sense belongs to a phrasal verb block with a mandatory particle (e.g. 'pay back' -> 'back', 'pay off' -> 'off', 'pay out' -> 'out'),
            # the quote MUST contain the particle within a local window of the verb; otherwise this phrasal sense cannot hijack base verb usages.
            if i in pv_sense_map:
                req_particles = pv_sense_map[i]
                if all(
                    any(cls._quote_has_frame(q_seq, [target_w_clean, alt], window=4) for alt in rp.split("/") if alt)
                    for rp in req_particles
                ):
                    score += 45
                else:
                    score -= 60
            elif eff_p == "v":
                # Also guard verb senses whose definition starts with 'if ... <verb>s <particle>' when the particle is absent
                m_pv_def = re.match(
                    rf"^if\s+[^,]+?\s+(?:{re.escape(target_w_clean)}[a-z]*)\s+(up|down|out|off|away|back|over|through|around|round|apart|aside|ahead|along|about)\b",
                    (s.get("definition") or "").strip().lower(),
                )
                if m_pv_def and not any(cls.form_matches(tok, m_pv_def.group(1)) for tok in q_seq):
                    score -= 60

            # 0b. Morphological Number / Countability & Syntactic Slot Gate:
            # If target noun in quote is physically plural ('causes', 'customs'),
            # it cannot match an exclusively [uncountable] sense if a [countable] sense is available.
            s_gram = (s.get("grammar") or "").lower()
            if target_tok_morph_number and "Plur" in target_tok_morph_number:
                if "[uncountable]" in s_gram and "[countable]" not in s_gram:
                    score -= 60
                if "[singular]" in s_gram and "[countable]" not in s_gram and "[plural]" not in s_gram:
                    score -= 60
            elif target_tok_morph_number and "Sing" in target_tok_morph_number:
                if "[plural]" in s_gram and "[singular]" not in s_gram and "[countable]" not in s_gram:
                    score -= 60
            if target_tok_is_predicative and "[only before noun]" in s_gram:
                score -= 60
            elif target_tok_is_attributive and "[not before noun]" in s_gram:
                score -= 60

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
                s_reg in ("chemistry", "physics", "biology", "computing")
                or (s_reg == "technical" and i > 0)
                or any(tag in s_gram_str for tag in ("[chemistry]", "[physics]", "[computing]"))
                or ("[technical]" in s_gram_str and i > 0)
                or s_signpost in ("chemistry", "physics", "technical", "chemical")
            )
            if is_tech_sense:
                has_tech_trigger = any(w in q_low_all for w in (
                    "chemical", "acid", "substance", "atom", "molecule", "reaction", "physics",
                    "laboratory", "voltage", "circuit", "alkali", "gas", "metal", "compound",
                    "technology", "technological", "scientific", "science", "engineering"
                ))
                if not has_tech_trigger:
                    score -= 50

            is_grammar_sense = (
                s_reg in ("grammar", "linguistics")
                or "[grammar]" in s_gram_str
                or s_signpost in ("grammar", "linguistics")
            )
            if is_grammar_sense:
                has_grammar_trigger = any(w in q_low_all for w in (
                    "grammar", "grammatical", "sentence", "verb", "noun", "adjective", "adverb",
                    "pronoun", "preposition", "clause", "phrase", "tense", "plural", "singular",
                    "syntax", "linguistic", "linguistics", "conjugat", "inflect"
                ))
                if not has_grammar_trigger:
                    score -= 60

            # 0e. Pejorative / Disapproval Stance Gate:
            # Senses explicitly marked 'used to show disapproval' (e.g. 'commercialize' sense 0: 'to be more concerned
            # with making money from something than about its quality – used to show disapproval') require a critical/pejorative
            # context or default absence of constructive enabling framing ('help ... design, produce, and commercialize their ideas').
            s_defn_raw_low = (s.get("definition") or "").lower()
            if "show disapproval" in s_defn_raw_low or "disapproving" in s_reg:
                has_pejorative_cue = any(w in q_low_all for w in (
                    "too", "overly", "merely", "corrupt", "ruin", "ruined", "spoil", "spoiled", "spoilt",
                    "cheap", "greed", "greedy", "profit", "money", "exploit", "bad", "poor", "excessive", "shameless"
                ))
                has_constructive_frame = any(w in q_low_all for w in (
                    "help", "helps", "helped", "helping", "enable", "enables", "enabled", "allow", "allows",
                    "support", "encourage", "develop", "design", "produce", "innovate", "innovation", "launch"
                ))
                if has_constructive_frame and not has_pejorative_cue:
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
            sp = (s.get("signpost") or "").lower().strip()
            if sp:
                sp_tokens = set(re.findall(r"\b[a-zA-Z]{3,}\b", sp)) - meta_stopwords
                sp_lemmas = {cls._surface_lemma(t) or t for t in sp_tokens}
                all_sp_tokens = (sp_tokens | sp_lemmas) - meta_stopwords - generic_signpost_stopwords
                # If the entire signpost is a single domain word like 'time' or 'place', preserve it
                if not all_sp_tokens and sp in ("time", "place", "state", "form", "number", "amount", "part"):
                    all_sp_tokens = {sp}
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
            if (
                second_order_domain_tokens
                and s_def_tokens
                and eff_p == "n"
                and (
                    bool(m_genus)
                    or (sp and sp not in ("controlled by a computer", "computer", "machine"))
                    or bool(convergent_domain_tokens)
                )
            ):
                _tech_bio_dom = second_order_domain_tokens & {
                    "computer", "computers", "electronic", "screen", "screens", "device", "devices",
                    "phone", "telephone", "mobile", "digital", "internet", "software", "wireless",
                    "plant", "plants", "animal", "animals", "crop", "crops", "disease", "diseases",
                    "space", "astronaut", "astronauts", "float", "floating", "gravity", "orbit", "spacecraft",
                }
                if _tech_bio_dom & s_def_tokens:
                    score += 14 * len(_tech_bio_dom & s_def_tokens)

            # 5. Pattern / Preposition / Collocation match directly from Quote
            # A pattern states a frame, so it is tested as one. As a bag of words it accepts
            # any sentence that merely contains the headword plus the pattern's other words
            # somewhere, which is not the same claim at all.
            if quote:
                target_w_clean = entry.get("word", "").lower()
                any_sense_pat_matched = False
                for pat in s.get("patterns", []):
                    pat_clean = pat.lower().strip()
                    branches = cls.clean_and_expand_pattern_slashes(pat)

                    matched_pat = False
                    for br in branches:
                        # Check separable/transitive phrasal patterns: if pattern specifies a mandatory (non-parenthesized)
                        # 'somebody ↔ ...' or 'something ↔ ...', it requires an explicit transitive direct object governed by the target verb.
                        br_non_opt = re.sub(r"\(.*?\)", "", br)
                        is_separable_transitive = "\u2194" in br or "<->" in br or (eff_p == "v" and ("somebody" in br_non_opt or "something" in br_non_opt or " sb " in f" {br_non_opt} " or " sth " in f" {br_non_opt} "))
                        if is_separable_transitive and quote and eff_p == "v":
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
                                    has_passive_or_rel_obj = any(c.dep_ in ("nsubjpass", "nsubj:pass", "relcl") for v in v_toks for c in v.children) or any(v.dep_ in ("relcl", "acl") for v in v_toks)
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
                        if has_to_do_in_pat and quote and eff_p == "v":
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

                        # First test via Typed Slot Pattern Matcher (handles (doing) something, in a ... fashion, etc.)
                        if cls.match_typed_slot_pattern(br, quote, doc=doc_q, headword=target_w_clean):
                            score += 40
                            matched_pat = True
                            break

                        words_in_pat = cls._pattern_words(br)
                        # If pattern is a slotless preposition + headword frame (e.g. 'in fashion', 'out of fashion')
                        # with no wildcard/infix slot, require contiguous adjacency (window=1) so 'in a predictably irrational fashion'
                        # does not falsely match 'in fashion'.
                        has_slot_in_br = bool(re.search(r"(\.\.\.|somebody|something|someone|\bsb\b|\bsth\b|\bdoing\b|\bto\s+do\b|\()", br.lower()))
                        frame_win = cls.PATTERN_FRAME_WINDOW if has_slot_in_br else (1 if words_in_pat and words_in_pat[0] in cls._PATTERN_PREPS else cls.PATTERN_FRAME_WINDOW)
                        if len(words_in_pat) >= 2 and cls._quote_has_frame(q_seq, words_in_pat, window=frame_win):
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
                        if prep_words and frame_words and all(cls._quote_has_frame(q_seq, frame_words + [pw]) for pw in prep_words):
                            score += 40
                            matched_pat = True
                            break

                    if matched_pat:
                        any_sense_pat_matched = True
                        break

                    if not matched_pat and "/" not in pat_clean:
                        words_in_pat = cls._pattern_words(pat_clean)
                        non_hw_words = [
                            w for w in words_in_pat
                            if w not in cls.inflected_forms(target_w_clean)
                        ]
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
                        elif (len(non_hw_words) == 1
                                and non_hw_words[0] in cls._PATTERN_PREPS + cls._PATTERN_PARTICLES
                                and cls._quote_has_frame(q_seq, [target_w_clean, non_hw_words[0]])):
                            # Bound preposition/particle match (e.g. serious about, line ... along)
                            score += 25

                # 5a. Example-Attested Preposition Frame Parity:
                # When a sense omits `patterns` in the dictionary (e.g. 'paradigm' Sense 0 has patterns=[]
                # while its own dictionary example is 'the basic paradigm of the family tree', whereas Sense 1
                # declares patterns=['paradigm of']), grant frame parity if both the quote and this sense's
                # own examples attest the exact same contiguous '<headword> <prep>' frame declared by a sibling sense.
                if not any_sense_pat_matched and not s.get("units") and s.get("examples"):
                    sibling_prep_frames: Set[str] = set()
                    for sib_s in senses:
                        if sib_s is s or cls._pos_code(sib_s.get("pos")) != s_pos:
                            continue
                        for sib_pat in (sib_s.get("patterns") or []):
                            sib_words = cls._pattern_words(sib_pat.lower().strip())
                            if len(sib_words) == 2 and sib_words[0] == target_w_clean and sib_words[1] in cls._PATTERN_PREPS:
                                sibling_prep_frames.add(sib_words[1])
                    for prep_f in sibling_prep_frames:
                        if cls._quote_has_frame(q_seq, [target_w_clean, prep_f], window=1):
                            if any(
                                cls._quote_has_frame(re.findall(r"\b[a-z]+\b", ex_s.lower()), [target_w_clean, prep_f], window=1)
                                for ex_s in (s.get("examples") or [])
                            ):
                                score += 40
                                break

            # 5b. Units Matching & Idiom Constraint Enforcement
            if quote:
                any_unit_demanded = False
                any_unit_matched = False
                best_unit_bonus = 0

                # Political/institutional domain restriction guard (e.g. 'an event that causes a government, political organization etc problems')
                if "a government, political organization" in s_defn:
                    _pol_triggers = {"government", "political", "minister", "president", "party", "administration", "parliament", "state", "official", "election", "policy", "senator", "congress", "scandal"}
                    _ctx_tokens = set(q_seq) | set(re.findall(r"\b[a-z]+\b", (discourse_context or "").lower()))
                    if not (_ctx_tokens & _pol_triggers):
                        score -= 45

                for u in s.get("units", []):
                    # Capitalized Proper-Noun Unit Guard (e.g. 'the Passion' vs lowercase 'the passion')
                    _u_words = re.findall(r"[A-Za-z]+", str(u))
                    _cap_non_initial = [w for w in _u_words[1:] if w[0].isupper()]
                    if _cap_non_initial and not all(cw in (quote or "") for cw in _cap_non_initial):
                        any_unit_demanded = True
                        continue

                    u_clean = u.lower().replace("\u2019", "'")
                    s_gram_low = (s.get("grammar") or "").lower()

                    # First test via Typed Slot Pattern Matcher before stripping slots!
                    # This preserves (doing) something, to do something, and infix slots like 'in a ... fashion'.
                    if cls.match_typed_slot_pattern(u, quote, doc=doc_q, headword=target_w_clean):
                        if not (" " not in target_w_clean and u_clean.startswith(("not ", "no ", "never ")) and s_defn.startswith(("no ", "not ", "never "))):
                            if not (eff_p == "v" and re.search(r"(?<!in)transitive", s_gram_low)):
                                any_unit_demanded = True
                            best_unit_bonus = max(best_unit_bonus, 45)
                            any_unit_matched = True
                            continue

                    has_infix_slot = bool(re.search(r"(\.\.\.|somebody|something|someone|\bsb\b|\bsth\b)", u_clean))
                    u_no_opt = re.sub(r"\(.*?\)", "", u_clean).strip()
                    _has_non_poss_obj_slot = bool(
                        eff_p == "v"
                        and re.search(r"\b(?:somebody|something|someone|sb|sth)\b(?!'s\b)", u_no_opt)
                    )
                    if _has_non_poss_obj_slot and doc_q is not None:
                        head_cand_lemmas = {target_w_clean.split()[0]} if target_w_clean else set()
                        v_toks_u_chk = [t for t in doc_q if t.lemma_.lower() in head_cand_lemmas or t.text.lower() in head_cand_lemmas]
                        if not v_toks_u_chk:
                            v_toks_u_chk = [t for t in doc_q if t.pos_ in ("VERB", "AUX")]
                        has_u_gov_obj = any(
                            c.dep_ in ("dobj", "obj", "nsubjpass", "nsubj:pass", "relcl", "ccomp", "xcomp")
                            or (c.dep_ == "prep" and any(gc.dep_ in ("pobj", "pcomp") for gc in c.children))
                            for v in v_toks_u_chk for c in v.children
                        ) or any(v.dep_ in ("relcl", "acl") for v in v_toks_u_chk)
                        if not has_u_gov_obj:
                            any_unit_demanded = True
                            continue
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
                        # Note: Only strip attributive determiners ('your', 'her', 'my', 'their', 'our', 'its'), NEVER strip
                        # predicative possessive pronouns ('yours', 'hers', 'mine', 'theirs', 'ours') nor 'his' when part of a possessive pronoun slash group.
                        _is_poss_pron_unit = any(p in u_clean for p in ("yours", "hers", "mine", "theirs", "ours"))
                        _unit_stop = ("the", "an", "somebody", "something", "someone", "your", "her", "their", "my", "its", "one", "oneself", "sb", "sth", "etc", "eg", "ie", "doing") + (() if _is_poss_pron_unit else ("his",))
                        words = [w for w in re.findall(r"\b[a-z]+\b", br) if len(w) >= 2 and w not in _unit_stop]
                        if len(words) >= 2:
                            # If extracting a bare single word without negator, a unit sense that prepends an external negator
                            # ('not a single' -> 'no people or things at all') defines the negative phrase, not the bare headword
                            if " " not in target_w_clean and words[0] in ("not", "no", "never") and s_defn.startswith(("no ", "not ", "never ")):
                                score -= 50
                                continue
                            # If a verb sense explicitly declares [transitive], a prepositional unit (e.g. 'tend to sb/sth')
                            # is an optional prepositional variant, not an obligatory idiom requirement.
                            # Similarly, an intensifier + headword GramExa (e.g. 'all alone', 'quite right', 'long before')
                            # is an optional degree collocation, not an obligatory fixed idiom.
                            other_words = [w for w in words if not cls.form_matches(w, target_w_clean) and w not in headword_tokens]
                            _is_optional_intensifier_unit = (
                                bool(other_words)
                                and all(ow in ("all", "quite", "very", "completely", "entirely", "totally", "just", "right", "far", "much", "well", "long", "soon", "ever") for ow in other_words)
                                and not s_defn.startswith(tuple(other_words))
                            )
                            if not (eff_p == "v" and re.search(r"(?<!in)transitive", s_gram_low)) and not _is_optional_intensifier_unit:
                                any_unit_demanded = True
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
                                # the figurative idiom 'get the message' — UNLESS the unit explicitly specifies an infix slot ('in a ... fashion')!
                                has_intervening_modifier = False
                                if not has_infix_slot and target_w_clean in words and len(words) == 2:
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
                            elif (
                                not is_formulaic_greeting
                                and eff_p == "v"
                                and re.search(r"\betc\b", u_clean)
                                and doc_q is not None
                                and other_words
                            ):
                                # Open-ended verb collocation unit (e.g. 'defy logic/the odds etc', 'defy description/analysis/belief etc'):
                                # Check if the verb in quote takes a direct object in the same WordNet semantic supersense (e.g. 'common sense' -> noun.cognition)
                                unit_supers: Set[str] = set()
                                for uw in other_words:
                                    unit_supers |= (cls.get_wordnet_supersenses(uw, pos="n") or set())
                                if unit_supers:
                                    v_toks_u = [t for t in doc_q if cls.form_matches(t.text.lower(), target_w_clean) or cls.form_matches(t.lemma_.lower(), target_w_clean)]
                                    for vt in v_toks_u:
                                        for vc in vt.children:
                                            if vc.dep_ in ("dobj", "obj", "attr"):
                                                cand_nouns = [vc.lemma_.lower()]
                                                amods = [gc.lemma_.lower() for gc in vc.children if gc.dep_ in ("amod", "compound")]
                                                if amods:
                                                    cand_nouns.append(f"{amods[0]} {vc.lemma_.lower()}")
                                                obj_supers: Set[str] = set()
                                                for cn in cand_nouns:
                                                    obj_supers |= (cls.get_wordnet_supersenses(cn, pos="n") or set())
                                                if obj_supers & unit_supers:
                                                    best_unit_bonus = max(best_unit_bonus, 35)
                                                    any_unit_matched = True
                                                    break
                                        if any_unit_matched:
                                            break
                                if any_unit_matched:
                                    break
                # Single-word non-verb headword guard against unattested verbal phrase definitions (e.g. 'be in attendance' -> 'to be at...'):
                # Only penalize when the sense's unit frame was NOT matched in the quote (so attested predicative frames like
                # 'people are accustomed to' matching 'be accustomed to (doing) something' are never penalized).
                if " " not in target_w_clean and eff_p != "v" and not any_unit_matched:
                    if re.match(r"^(?:\[.*?\]\s*)*to\s+[a-z]+\b", s_defn) or s_defn.startswith("used to say that"):
                        score -= 80
                if any_unit_matched:
                    score += best_unit_bonus
                elif any_unit_demanded and not (eff_p == "v" and any(re.search(r"\betc\b", str(u_item).lower()) for u_item in s.get("units", []))):
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
                if eff_pos == "verb" and target_w_clean == "post":
                    if re.search(r"\bposted\s+on\b.*\bby\b", q_low_all) or re.search(r"\bposted\s+on\s+\d+", q_low_all):
                        if any(k in s_defn for k in ("internet", "message", "website", "online", "document")):
                            score += 50

                # Internet domain boost for post noun
                if eff_pos == "noun" and target_w_clean == "post":
                    if any(k in q_low_all for k in ("online", "internet", "website", "web", "blog")):
                        if any(k in s_defn or k in sp for k in ("internet", "website", "online", "message", "discussion")):
                            score += 40

                # Passage A sense guards: keep the deterministic lock on the sense the
                # passage actually uses, not on a nearby PedLex sense with shared words.
                if eff_pos == "noun" and target_w_clean == "stamp":
                    if any(k in q_low_all for k in ("letter", "envelope", "post", "postage", "mail", "keep in touch")):
                        if "mail" in sp or "postage stamp" in s.get("units", []) or "envelope" in s_defn:
                            score += 50

                if eff_pos == "adverb" and target_w_clean == "closely":
                    if re.search(r"\b(follow|follows|followed|following)\b.*\b(time|times|trend|trends|development|developments|news|events)\b.*\bclosely\b", q_low_all):
                        if "carefully" in s_defn:
                            score += 50

                if eff_pos == "adjective" and target_w_clean == "simple":
                    if re.search(r"\bsimple\b[^\"'“”]{0,20}[\"'“”]", q_low_all):
                        if "not difficult" in s_defn or "complicated" in s_defn:
                            score += 50

                if eff_pos == "noun" and target_w_clean == "line":
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

                if eff_pos == "noun" and target_w_clean == "message":
                    if any(k in q_low_all for k in ("voice message", "text message", "phone message", "left me a message", "leave a message", "send a message")):
                        if i == 0 or "spoken or written piece of information" in s_defn:
                            score += 50

                if eff_pos == "noun" and target_w_clean == "freedom":
                    if re.search(r"\bfreedom\b.*\b(chance|opportunity|choice|choices|choose)\b", q_low_all):
                        if "right to do what you want" in s_defn:
                            score += 50

                # Passage A specific sense guards:
                # 'common' as happening often / widely found vs shared by society
                if eff_pos == "adjective" and target_w_clean == "common":
                    if any(k in q_low_all for k in ("now", "today", "scooter", "scooters", "everywhere", "popular", "widespread")):
                        if "happening often" in s_defn:
                            score += 55
                        elif "shared by everyone in a society" in s_defn:
                            score -= 50

                # 'tip' as touching/raising hat as a greeting
                if eff_pos == "verb" and target_w_clean == "tip":
                    if any(k in q_low_all for k in ("hat", "hats", "cap", "caps")):
                        if re.search(r"\b(hat|hats|cap|caps|greeting)\b", s_defn):
                            score += 80
                        elif any(k in s_defn for k in ("money", "waiter", "likely to succeed")):
                            score -= 60

                # 'come from' as obtained from a place, thing, or person / starting somewhere vs traveling
                if eff_pos in ("verb", "phrasal verb") and (target_w_clean in ("come", "come from") or "come from" in target_w_clean):
                    if "from" in q_low_all:
                        if any(k in q_low_all for k in ("gas", "gases", "greenhouse", "thing", "things", "pollution", "money", "heat", "power", "word", "idea", "product")):
                            if "obtained from a place" in s_defn or "start or be made somewhere" in s_defn:
                                score += 80
                            elif "travel to or reach a place" in s_defn:
                                score -= 60

                # 'swell' as increasing in amount/number
                if eff_pos == "verb" and target_w_clean == "swell":
                    if any(k in q_low_all for k in ("membership", "member", "members", "rank", "ranks", "number", "numbers", "crowd")):
                        if "increase in amount or number" in s_defn:
                            score += 65
                        elif any(k in s_defn for k in ("upwards", "move suddenly", "curve")):
                            score -= 40

                # 'present' as formally giving something at an occasion
                if eff_pos == "verb" and target_w_clean == "present":
                    if re.search(r"\bpresent(?:ed|ing)?\b.*\bto\b", q_low_all) and any(k in q_low_all for k in ("ioc", "committee", "award", "gift", "medal", "unveil", "unveiled", "official", "formal")):
                        if "give something to someone" in s_defn and "formal or official occasion" in s_defn:
                            score += 70
                        elif any(k in s_defn for k in ("talk and behave", "performance", "symptom")):
                            score -= 50

                # 'give' as behaving towards someone with praise/respect/loyalty
                if eff_pos == "verb" and target_w_clean == "give":
                    if any(k in q_low_all for k in ("praise", "respect", "loyalty", "obedience")):
                        if "behave towards someone" in s_defn:
                            score += 70
                        elif any(k in s_defn for k in ("let someone have something as a present", "pay a particular amount")):
                            score -= 50

                # 'turn' in phrasal/contextual 'turn around': physical movement vs situation/business
                if eff_pos == "verb" and (target_w_clean == "turn" or target_w_clean == "turn around"):
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
                if eff_pos == "verb" and target_w_clean == "represent":
                    is_symbolic_subj = any(k in q_low_all for k in ("color", "colour", "red", "white", "blue", "green", "black", "symbol", "emblem", "flag", "sign", "logo", "mark", "image", "painting"))
                    is_abstract_dobj = any(k in q_low_all for k in ("passion", "friendship", "peace", "hope", "freedom", "love", "unity", "power", "strength", "joy", "purity", "courage", "danger", "ease"))
                    if is_symbolic_subj or is_abstract_dobj:
                        if any(k in s_defn for k in ("symbol of something", "sign or mark that means something", "painting, statue")):
                            score += 85
                        elif any(k in s_defn for k in ("speak or take action for another person", "court of law", "elected to a parliament", "describe someone or something")):
                            score -= 60

                # 'trap' as capturing/preventing gas, energy, heat, water from escaping
                if eff_pos == "verb" and target_w_clean == "trap":
                    if any(k in q_low_all for k in ("gas", "gases", "greenhouse", "energy", "heat", "sun", "atmosphere", "water", "air")):
                        if "prevent something such as gas" in s_defn or "prevent energy" in s_defn:
                            score += 85
                        elif any(k in s_defn for k in ("bad situation from which you cannot escape", "catch an animal", "trick someone", "crushed between")):
                            score -= 60

                # 'simple' as plain/ordinary/not complicated vs 'the simple fact/truth/reason' emphasis sense
                if eff_pos == "adjective" and target_w_clean == "simple":
                    if "used to emphasize that only one thing is involved" in s_defn:
                        if not re.search(r"\bthe\s+simple\s+(?:fact|truth|reason|matter|act|question)\b", q_low_all):
                            score -= 60
                    elif "plain and without anything extra" in s_defn or "ordinary and not special" in s_defn:
                        score += 45
                    elif "not difficult or complicated" in s_defn:
                        score += 35

                # 'appetite' as physical desire for food vs metaphorical 'appetite for [activity]'
                if eff_pos == "noun" and target_w_clean == "appetite":
                    has_appetite_for = bool(re.search(r"\bappetites?\s+for\b", q_low_all))
                    if not has_appetite_for:
                        if "desire for food" in s_defn:
                            score += 55
                        elif "desire or liking for a particular activity" in s_defn:
                            score -= 55

                # 'community' as local residents/neighborhood vs shared interest/religion/race group
                if eff_pos == "noun" and target_w_clean == "community":
                    has_group_mod = any(k in q_low_all for k in ("scientific", "religious", "ethnic", "international", "business", "academic", "online", "medical", "intelligence", "immigrant", "asian", "black", "gay", "jewish", "christian", "muslim", "community of"))
                    if not has_group_mod:
                        if "people who live in the same area" in s_defn:
                            score += 45
                        elif any(k in s_defn for k in ("same religion, race, job", "same interests, religion, race", "group of countries")):
                            score -= 45

                # 'ride' on a single-wheel/personal device/bike/horse (active control) vs passive bus/train passenger
                if eff_pos == "verb" and target_w_clean == "ride":
                    if re.search(r"\brid(?:e|es|ing|den)|rode\b", q_low_all) and any(k in q_low_all for k in ("riding on", "ride on", "rode on", "wheel", "self-balance", "bike", "bicycle", "motorcycle", "horse", "scooter", "skateboard")):
                        if "control it as it moves" in s_defn or "sit on a horse, bicycle" in s_defn:
                            score += 60
                        elif "vehicle that you are not driving" in s_defn:
                            score -= 50

                # 'critic' as general fault-finder/opponent vs professional art/film/music reviewer
                if eff_pos == "noun" and target_w_clean == "critic":
                    has_arts_ctx = any(k in q_low_all for k in ("art", "music", "film", "films", "movie", "movies", "theatre", "theater", "book", "books", "literary", "play", "concert", "opera", "dance", "painting", "novel", "performance", "review"))
                    if not has_arts_ctx:
                        if "does not approve" in s_defn or "talks about their faults" in s_defn:
                            score += 55
                        elif "good and bad qualities of art, music, films" in s_defn:
                            score -= 55

                # 'cut out' physical cutting with scissors/knife vs stopping something from happening
                if eff_pos in ("verb", "phrasal verb") and target_w_clean in ("cut", "cut out"):
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
                        if (tok.pos_ == "ADJ" and eff_p != "v") or is_hyphen_adj or eff_p == "adj":
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
                        elif (tok.pos_ == "NOUN" and eff_p != "v") or eff_p == "n":
                            gram_l = (s.get("grammar") or "").lower()
                            has_count_det = any(
                                ch.dep_ in ("det", "nummod")
                                and ch.lemma_.lower() in ("a", "an", "one", "two", "three", "many", "few", "several", "each", "every", "another", "both")
                                for ch in tok.children
                            )
                            if tok.tag_ == "NNS" or has_count_det:
                                if ("[uncountable]" in gram_l or (tok.tag_ == "NNS" and "[singular]" in gram_l)) and "countable" not in gram_l:
                                    score -= 60 if has_count_det else 50
                                elif "countable" in gram_l or (has_count_det and "[singular]" in gram_l):
                                    score += 25
                            if tok.tag_ == "NN" or (target_tok_morph_number and "Sing" in target_tok_morph_number):
                                if "[plural]" in gram_l and "[singular]" not in gram_l and "countable" not in gram_l:
                                    score -= 60

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
                        elif tok.pos_ == "VERB" or (eff_p == "v" and (tok.dep_ in ("ROOT", "conj", "advcl", "relcl", "ccomp", "xcomp", "pobj", "dobj", "obj", "amod", "compound") or any(ch.dep_ in ("nsubj", "nsubjpass", "csubj") for ch in tok.children))):
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
                            if not surface_dobjs and eff_p == "v":
                                for conj_v in tok.children:
                                    if conj_v.dep_ == "conj" and conj_v.pos_ == "VERB" and conj_v.i > tok.i:
                                        conj_dobjs = [ch for ch in conj_v.children if ch.dep_ in ("dobj", "obj")]
                                        if not conj_dobjs:
                                            for prep_ch in conj_v.children:
                                                if prep_ch.dep_ == "prep" and prep_ch.lower_ in ("about", "into", "of", "on"):
                                                    conj_dobjs.extend(p_ch for p_ch in prep_ch.children if p_ch.dep_ == "pobj")
                                        if conj_dobjs:
                                            surface_dobjs = conj_dobjs
                                            break
                            if not surface_dobjs and eff_p == "v" and tok.dep_ in ("amod", "compound") and tok.head.pos_ in ("NOUN", "PROPN") and tok.head.i > tok.i:
                                surface_dobjs = [tok.head]
                            if not surface_dobjs and eff_p == "v" and tok.dep_ == "conj":
                                _r_head = (
                                    tok.head if tok.head.dep_ == "ROOT"
                                    else (
                                        tok.head.head.head
                                        if tok.head.dep_ == "pobj" and tok.head.head.dep_ == "prep" and tok.head.head.head.dep_ == "ROOT"
                                        else None
                                    )
                                )
                                if _r_head is not None and tok.i + 1 < len(q_doc) and q_doc[tok.i + 1].text != ",":
                                    surface_dobjs = [ch for ch in _r_head.children if ch.dep_ == "appos" and ch.i > tok.i]
                            passive_dobjs = [ch for ch in tok.children if ch.dep_ in ("nsubjpass", "nsubj:pass")] if is_passive else []
                            rel_dobjs = [tok.head] if (
                                tok.dep_ == "relcl"
                                and tok.head.pos_ in ("NOUN", "PROPN", "PRON")
                                and tok.head.lemma_.lower() not in ("way", "time", "day", "year", "moment", "reason", "place", "manner", "case")
                                and not surface_dobjs
                                and not any(t.text == "," for t in q_doc[tok.head.i : tok.i])
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
                            if not surface_nsubjs and eff_p == "v" and tok.dep_ == "xcomp":
                                surface_nsubjs = [ch for ch in tok.head.children if ch.dep_ in ("nsubj", "csubj", "dobj", "obj")]
                            if not surface_nsubjs and eff_p == "v":
                                left_nom_children = [
                                    ch for ch in tok.children
                                    if ch.dep_ in ("compound", "nmod", "npadvmod")
                                    and ch.pos_ in ("NOUN", "PROPN", "PRON")
                                    and ch.i < tok.i
                                ]
                                for lnc in list(left_nom_children):
                                    for gc in lnc.children:
                                        if gc.dep_ == "conj" and gc.pos_ in ("NOUN", "PROPN", "PRON") and gc.i < tok.i:
                                            left_nom_children.append(gc)
                                surface_nsubjs = left_nom_children
                                if tok.dep_ == "conj" and tok.i > 0 and q_doc[tok.i - 1].pos_ in ("NOUN", "PROPN", "PRON") and q_doc[tok.i - 1].dep_ == "conj" and q_doc[tok.i - 1].head == tok.head:
                                    surface_nsubjs.append(q_doc[tok.i - 1])
                                elif tok.dep_ == "conj" and tok.head.pos_ in ("NOUN", "PROPN", "PRON") and tok.head.i < tok.i:
                                    surface_nsubjs.append(tok.head)
                                    if tok.head.dep_ == "pobj" and tok.head.head.dep_ == "prep" and tok.head.head.head.pos_ in ("NOUN", "PROPN", "PRON"):
                                        surface_nsubjs.append(tok.head.head.head)
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

                            is_coord_predicate_ellipsis = (
                                not has_effective_obj
                                and (any(c.dep_ == "conj" for c in tok.children) or tok.dep_ == "conj")
                                and any(
                                    sn.lemma_.lower() in ("relationship", "divorce", "failure", "defeat", "rejection", "criticism", "remark", "comment", "word", "experience", "life")
                                    for sn in deep_nsubj_tokens
                                )
                                and any(k in s_defn for k in ("affect someone badly", "feel less confident", "make them feel", "upset", "emotional"))
                            )

                            if not has_effective_obj:
                                if is_pure_intransitive and not requires_obj:
                                    score += 25
                                elif is_coord_predicate_ellipsis:
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
                                _all_pat_preps = set(cls._PATTERN_PREPS) | {
                                    "above", "before", "after", "below", "under", "over", "between",
                                    "among", "into", "onto", "against", "through", "across", "towards",
                                    "toward", "upon", "by", "without", "behind", "beside", "beyond", "around"
                                }
                                def _pat_preps_satisfied(pat_str: str) -> bool:
                                    p_toks = re.findall(r"\b[a-z]+\b", pat_str.lower())
                                    req_preps = [t for t in p_toks if t in _all_pat_preps]
                                    return (not req_preps) or any(rp in q_seq for rp in req_preps)

                                has_trans_pat = any(_requires_direct_object(p) and _pat_preps_satisfied(p) for p in s_pats)
                                has_strictly_intrans_prep_pat = any(
                                    not _requires_direct_object(p)
                                    and len(p.lower().split()) >= 2
                                    and p.lower().split()[1] in cls._PATTERN_PREPS
                                    for p in s_pats
                                )
                                if has_trans_pat:
                                    score += 35
                                elif has_strictly_intrans_prep_pat and not has_trans_pat:
                                    score -= 30

                            # Mandatory Passive / Predicative Frame Gate (e.g. 'be well/ideally placed to do something'):
                            if all_pat_units and all(re.match(r"^be\b", p.strip()) for p in all_pat_units):
                                if not is_passive and tok.tag_ not in ("VBN", "JJ"):
                                    score -= 60

                            # Mandatory Locative / Prepositional Complement Gate:
                            # Senses marked '[... always + adverb/preposition]' (e.g. 'conduct sb to/into sth', 'place sth on/under sth')
                            # require a prepositional or directional adverbial complement governed by the verb (or attached to its direct object).
                            if "always + adverb/preposition" in gram_l:
                                has_adv_or_prep_comp = any(
                                    c.dep_ in ("prep", "prt", "npadvmod")
                                    or (c.dep_ == "advmod" and c.i > tok.i)
                                    for c in tok.children
                                ) or any(
                                    gc.dep_ in ("prep", "prt", "npadvmod") or (gc.dep_ == "advmod" and gc.i > tok.i)
                                    for dobj_c in surface_dobjs
                                    for gc in dobj_c.children
                                )
                                if has_adv_or_prep_comp:
                                    score += 35
                                elif not (has_direct_obj and has_trans_gram):
                                    score -= 70

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

                                # Abstract temporal/informational direct objects (e.g. 'follow the time closely', 'follow the news/events/schedule'):
                                # Penalize physical locomotion, event-sequence, or instruction-compliance senses and boost attention/tracking senses.
                                if obj_lem in ("time", "schedule", "timetable", "news", "event", "events", "story", "progress", "development", "trend", "change", "situation"):
                                    if any(k in s_defn for k in ("walk", "drive", "behind or after someone", "chase", "happen or come after", "told or advised you")):
                                        score -= 55
                                    elif any(k in s_defn for k in ("pay close attention", "very interested in", "understand", "watch", "listen")):
                                        score += 55

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
                                        has_etc_locative = (
                                            "etc" in pat.lower().split()
                                            and any(lp in pat.lower() for lp in ("through", "around", "across", "along", "into", "over", "down"))
                                        )
                                        for br in branches:
                                            br_tokens = br.split()
                                            prep_hit = (
                                                f" {p_txt} " in f" {br} " or f" {p_lem} " in f" {br} "
                                                or p_txt in br_tokens or p_lem in br_tokens
                                            )
                                            if not prep_hit and has_etc_locative and p_lem in ("in", "inside", "throughout", "across", "through", "around", "over", "along"):
                                                p_objs_loc = [pc for pc in ch.children if pc.dep_ in ("pobj", "pcomp")]
                                                if any(
                                                    any(ss in cls.get_wordnet_supersenses(pc.lemma_.lower(), pos="n") for ss in ("noun.location", "noun.artifact", "noun.object"))
                                                    for pc in p_objs_loc
                                                ):
                                                    prep_hit = True
                                            if prep_hit:
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
                                                content_pw = [w for w in pw if w not in cls._PATTERN_PREPS and w not in cls._PATTERN_PARTICLES and w not in cls._PATTERN_FUNCTION_WORDS and w not in ("be", "become", "get", "grow") and w not in target_inflections]
                                                req_prep_prt = [w for w in pw if w in cls._PATTERN_PREPS or w in cls._PATTERN_PARTICLES]
                                                if req_prep_prt and not all(rp in q_seq for rp in req_prep_prt):
                                                    continue
                                                if _requires_direct_object(br) and not has_effective_obj:
                                                    continue
                                                if not content_pw or any(any(cls.form_matches(tok_q, cpw) for tok_q in q_seq) for cpw in content_pw):
                                                    # When particle/preposition matches pattern (or open-ended locative pattern)
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
                                        # Institutional / organization prepositional complement ('admitted to Jilin University / the club / the society')
                                        p_ch_supers = cls.get_wordnet_supersenses(p_child_lem, pos="n")
                                        if (
                                            p_child.ent_type_ == "ORG"
                                            or p_child_lem in ("university", "college", "school", "academy", "institute", "club", "organization", "organisation", "society", "party", "union", "association")
                                            or (p_ch_supers and "noun.group" in p_ch_supers)
                                        ):
                                            if any(k in s_defn for k in ("organization", "organisation", "club", "school", "university", "college", "member", "join")):
                                                score += 35

                                    # Specific alignment for intransitive verbs of appearing/existence with locative/conceptual preposition (in myths, in dreams, etc.)
                                    if p_lem in ("in", "at", "on", "from") and any(k in s_defn for k in ("exist in a place", "start to be seen", "arrive", "become visible")):
                                        # When appearing in concepts/settings (myths, dreams, place, air, sky) rather than film/court/publication
                                        if not any(k in s_defn for k in ("film", "play", "court", "television", "print", "newspaper")):
                                            score += 30

                                    # Universal verb-preposition / verb-adverb valency frame agreement:
                                    # If the verb in context governs a preposition (e.g. 'function as', 'look into', 'depend on', 'succeed in')
                                    # or an open-class manner/degree adverb (e.g. 'follow ... closely', 'listen ... carefully'),
                                    # check if this frame is explicitly articulated in the sense definition or examples.
                                    target_lemmas = {target_w_clean} | cls.inflected_forms(target_w_clean) | {tok.lemma_.lower()}
                                    v_prep_pat = r"\b(?:" + "|".join(re.escape(f) for f in target_lemmas) + r")(?:\s+\w+)?\s+(?:" + re.escape(p_lem) + r"|" + re.escape(p_txt) + r")\b"
                                    if ch.dep_ == "advmod" and len(p_lem) >= 4 and p_lem not in stopwords and p_lem not in (
                                        "very", "quite", "rather", "pretty", "really", "always", "never", "often", "sometimes",
                                        "usually", "already", "still", "just", "even", "also", "only", "almost", "nearly",
                                        "hardly", "barely", "merely", "simply", "actually", "probably", "perhaps", "maybe"
                                    ):
                                        v_adv_pat = r"\b(?:" + "|".join(re.escape(f) for f in target_lemmas) + r")(?:\s+\S+){0,4}\s+(?:" + re.escape(p_lem) + r"|" + re.escape(p_txt) + r")\b"
                                        if p_lem in s_def_tokens or re.search(v_adv_pat, s_defn) or any(re.search(v_adv_pat, ex.lower()) for ex in s.get("examples", [])):
                                            score += 45
                                            pat_matched = True
                                        elif any(re.search(rf"\b(?:{re.escape(p_lem)}|{re.escape(p_txt)})\b", ex.lower()) for ex in s.get("examples", [])):
                                            score += 30
                                    if not pat_matched:
                                        if re.search(v_prep_pat, s_defn):
                                            score += 35
                                        elif any(re.search(v_prep_pat, ex.lower()) for ex in s.get("examples", [])):
                                            # Guard incidental locative prepositions ('in', 'at', 'on') in examples when the sense has no explicit patterns
                                            # and the quote's prepositional object is a physical room/building/place while the sense is purely abstract/folklore.
                                            if p_lem in ("in", "at", "on") and not s.get("patterns") and any(
                                                any(ss in cls.get_wordnet_supersenses(pc.lemma_.lower(), pos="n") for ss in ("noun.location", "noun.artifact"))
                                                and not any(ss in cls.get_wordnet_supersenses(pc.lemma_.lower(), pos="n") for ss in ("noun.communication", "noun.cognition"))
                                                for pc in p_objs_expanded
                                            ) and any(k in s_defn for k in ("mentioned", "talked about", "folklore", "history", "memory")):
                                                score -= 25
                                            else:
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
                                # Direct subject-signpost selectional match (e.g. "As time went on" -> signpost [time])
                                if subj_lem and sp and (subj_lem == sp or subj_lem in re.findall(r"\b[a-z]+\b", sp)):
                                    score += 35
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
                                    is_sound_subj = subj_lem in ("music", "sound", "noise", "applause", "cheer", "chorus", "song", "shout", "cry", "word", "voice", "laughter", "echo", "bell", "horn", "roar", "note", "tone")
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

                                    # Quantitative / Financial / Abstract Metric Subject vs. Physical Gravity / Spatial Relocation:
                                    is_metric_or_abstract_subj = (
                                        subj_lem in (
                                            "price", "prices", "stock", "stocks", "gold", "rate", "rates", "wage", "wages",
                                            "tax", "taxes", "demand", "supply", "export", "exports", "import", "imports",
                                            "credit", "trust", "wealth", "power", "value", "cost", "costs", "revenue",
                                            "income", "profit", "profits", "temperature", "temperatures", "heat", "pressure",
                                            "level", "levels", "standard", "standards", "quality", "culture", "management",
                                            "policy", "policies", "law", "laws", "attitude", "attitudes", "habit", "habits",
                                        )
                                        or (
                                            bool(subj_supersenses & {"noun.possession", "noun.quantity", "noun.attribute", "noun.cognition"})
                                            and not bool(subj_supersenses & {"noun.artifact", "noun.body", "noun.person", "noun.animal", "noun.plant", "noun.object"})
                                        )
                                    )
                                    if is_metric_or_abstract_subj:
                                        if any(k in s_defn or k in sp for k in (
                                            "lower level or amount", "lower level, amount", "become less", "decrease",
                                            "opinions, beliefs", "become different", "go upwards", "increase in",
                                        )):
                                            score += 18
                                        elif any(k in s_defn for k in (
                                            "onto the ground", "from one place or position to another",
                                            "from a higher position to a lower position", "take off your clothes",
                                            "take back to a shop",
                                        )):
                                            score -= 18

                                    # Universal Conditional-Subject Ontology Gate ('if a <Subject-Class> <verb>s'):
                                    # Parse LDOCE's canonical conditional subject clause (e.g. 'if a piece of furniture or equipment collapses',
                                    # 'if a piece of fruit bruises', 'if a system, idea, or organization collapses') and verify ontological
                                    # compatibility between the passage subject (subj_lem + WordNet supersenses) and the condition's nouns.
                                    m_cond_subj = re.match(
                                        r"^if\s+(?:a\s+|an\s+|the\s+|your\s+)?([^,]{2,65}?)\s+"
                                        + rf"(?:{re.escape(target_w_clean)}[a-z]*|{re.escape(tok.lemma_.lower())}[a-z]*)\b",
                                        s_defn,
                                    )
                                    if m_cond_subj:
                                        cond_np = m_cond_subj.group(1).lower()
                                        _cond_stop = {
                                            "piece", "pieces", "kind", "kinds", "type", "types", "sort", "sorts",
                                            "part", "parts", "amount", "number", "group", "set", "someone", "somebody",
                                            "something", "person", "people", "thing", "things", "one", "two", "other",
                                            "particular", "certain", "large", "small", "new", "old", "etc"
                                        }
                                        cond_nouns = [
                                            w_c for w_c in re.findall(r"[a-z]{3,}", cond_np)
                                            if w_c not in stopwords and w_c not in _cond_stop
                                        ]
                                        if cond_nouns:
                                            _open_class_anchors = {
                                                "system", "organization", "situation", "process", "idea", "plan",
                                                "institution", "company", "business", "relationship", "activity",
                                                "opinion", "opinions", "belief", "beliefs", "price", "prices", "level", "levels",
                                            }
                                            is_open_class_cond = (
                                                len(cond_nouns) >= 3
                                                or "etc" in cond_np
                                                or bool(set(cond_nouns) & _open_class_anchors)
                                            )
                                            cond_supers: Set[str] = set()
                                            for cn in cond_nouns:
                                                cond_supers.update(cls.get_wordnet_supersenses(cn, pos="n"))
                                            # Check primary WordNet hypernym closure for narrow conditions
                                            is_hypernym_of_cond = False
                                            wn_inst = cls.get_wordnet()
                                            if wn_inst and subj_lem:
                                                try:
                                                    for syn in (wn_inst.synsets(subj_lem, pos="n") or [])[:2]:
                                                        for h_path in syn.hypernym_paths():
                                                            for h_syn in h_path:
                                                                h_lems = {l.lower() for l in h_syn.lemmas()}
                                                                if h_lems & set(cond_nouns):
                                                                    is_hypernym_of_cond = True
                                                                    break
                                                            if is_hypernym_of_cond:
                                                                break
                                                        if is_hypernym_of_cond:
                                                            break
                                                except Exception:
                                                    pass
                                            # Direct lexical or hypernym/supersense match with passage subject
                                            if subj_lem in cond_nouns or any(cls.form_matches(subj_lem, cn) for cn in cond_nouns) or is_hypernym_of_cond:
                                                score += 45
                                            elif is_open_class_cond and subj_supersenses and cond_supers and (subj_supersenses & cond_supers):
                                                score += 15
                                            elif any(cn in q_low_all for cn in cond_nouns):
                                                score += 20
                                            elif not is_open_class_cond:
                                                # Narrow closed-class condition (e.g. 'if the wind changes', 'if a leader or a government falls',
                                                # 'if someone's voice or a sound falls', 'if a piece of furniture or equipment collapses')
                                                # is penalized when the passage subject is neither the restricted noun nor its hyponym.
                                                score -= 25
                                            else:
                                                # Narrow concrete/artifact/substance conditional subjects should be penalized when
                                                # the passage subject belongs to a disjoint ontological domain (e.g. abstract/state/location)
                                                _narrow_concrete_supers = {
                                                    "noun.food", "noun.plant", "noun.animal", "noun.substance", "noun.body"
                                                }
                                                _narrow_artifact_nouns = {
                                                    "furniture", "equipment", "chair", "table", "bed", "tent", "vehicle",
                                                    "car", "aircraft", "plane", "ship", "boat", "train", "machine", "engine",
                                                    "garment", "cloth", "clothing", "shoe", "containers", "container", "bottle",
                                                    "pipe", "wire", "coin", "note", "cheque", "check", "wound", "bone", "tooth"
                                                }
                                                is_narrow_concrete_cond = (
                                                    bool(cond_supers & _narrow_concrete_supers)
                                                    or any(cn in _narrow_artifact_nouns for cn in cond_nouns)
                                                )
                                                if is_narrow_concrete_cond and not (subj_supersenses & (cond_supers | {"noun.Tops"})):
                                                    score -= 35

                                    if "piece of furniture or equipment" in s_defn:
                                        if not any(fw in q_low_all for fw in ("furniture", "equipment", "chair", "table", "bed", "tent", "umbrella", "stroller", "tripod", "fold")):
                                            score -= 35
                                    if "piece of fruit" in s_defn:
                                        if not any(fw in q_low_all for fw in ("fruit", "fruits", "apple", "peach", "pear", "banana", "plum", "berry", "tomato", "produce", "peel", "ripe")):
                                            score -= 45
                                    if subj_lem in ("relationship", "divorce", "failure", "defeat", "rejection", "criticism", "remark", "comment", "word", "experience"):
                                        if any(k in s_defn for k in ("affect someone badly", "feel less confident", "make them feel", "upset", "hurt")):
                                            score += 55
                                    if subj_lem in ("system", "economy", "society", "environment", "ecosystem", "land", "ocean", "atmosphere", "nature", "climate", "civilization", "empire", "regime", "marriage", "relationship", "talk", "negotiation"):
                                        if "system, idea, or organization" in s_defn:
                                            score += 35
                                        elif "building, wall" in s_defn:
                                            score -= 20

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

                        # If target is a preposition / conjunction, check complement & clause aspect
                        elif tok.pos_ in ("ADP", "PREP", "SCONJ") or tok.dep_ in ("prep", "mark") or target_p in ("prep", "conj"):
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

                            # Temporal vs. Causal Conjunction Disambiguation (e.g. 'since', 'as'):
                            # 1) Temporal origin sense ('from a particular time in the past until a later past time, or until now'):
                            #    governed by perfect aspect in matrix clause ('has/have/had + VBN'), 'ever since', or past-tense point-in-time subordinate clause.
                            # 2) Causal reason sense ('used to give the reason why something happens'):
                            #    sentence-initial or present/modal subordinate clause without 'ever since' and without matrix temporal duration.
                            if target_p == "conj" or tok.dep_ == "mark":
                                is_temporal_origin_sense = (
                                    "from a particular time in the past" in s_defn
                                    or str(s.get("signpost") or "").strip().lower() == "time"
                                )
                                is_causal_reason_sense = (
                                    "give the reason" in s_defn
                                    or "because" in s_defn
                                    or str(s.get("signpost") or "").strip().lower() == "reason"
                                )
                                if is_temporal_origin_sense or is_causal_reason_sense:
                                    sub_verb = tok.head if tok.dep_ == "mark" else None
                                    matrix_verb = sub_verb.head if (sub_verb is not None and sub_verb.head != sub_verb) else None
                                    has_ever_since = tok.i > 0 and q_doc[tok.i - 1].lemma_.lower() == "ever"
                                    sub_is_past_event = bool(
                                        sub_verb is not None
                                        and (
                                            sub_verb.tag_ == "VBD"
                                            or any(ch.tag_ == "VBD" for ch in sub_verb.children if ch.dep_ in ("aux", "auxpass"))
                                        )
                                    )
                                    sub_is_present_or_modal = bool(
                                        sub_verb is not None
                                        and (
                                            sub_verb.tag_ in ("VBP", "VBZ")
                                            or any(ch.tag_ in ("VBP", "VBZ", "MD") for ch in sub_verb.children if ch.dep_ in ("aux", "auxpass"))
                                        )
                                    )
                                    matrix_has_perfect_or_time = bool(
                                        matrix_verb is not None
                                        and (
                                            any(ch.lemma_.lower() == "have" and ch.dep_ == "aux" for ch in matrix_verb.children)
                                            or any(t.lemma_.lower() in time_nouns or t.lemma_.lower() in ("pass", "elapse", "ago") for t in q_doc)
                                        )
                                    )
                                    if has_ever_since or (sub_is_past_event and matrix_has_perfect_or_time):
                                        if is_temporal_origin_sense:
                                            score += 45
                                        if is_causal_reason_sense:
                                            score -= 35
                                    elif sub_is_present_or_modal and not has_ever_since and target_w_clean in ("since", "as"):
                                        if is_causal_reason_sense:
                                            score += 45
                                        if is_temporal_origin_sense:
                                            score -= 35
                                if tok.dep_ == "mark" and target_w_clean == "after":
                                    if "when a particular event or time has happened" in s_defn:
                                        score += 45
                                    elif "because of something that happened earlier" in s_defn:
                                        score -= 30

                        # If target is an adverb, check focus attachment, degree-infinitive frame, and motion/space vs dimension/time
                        if tok.pos_ == "ADV" or target_p == "adv":
                            # 1) Post-nominal focus adverb ('Money alone cannot solve this', 'in legal fees alone')
                            if (
                                tok.i > 0
                                and q_doc[tok.i - 1].pos_ in ("NOUN", "PROPN", "PRON")
                                and (tok.head == q_doc[tok.i - 1] or q_doc[tok.i - 1].dep_ in ("nsubj", "nsubjpass", "pobj", "dobj"))
                            ):
                                if any(k in s_defn for k in ("without including anything else", "emphasize that there is only one")):
                                    score += 45
                                elif any(k in s_defn for k in ("there is no one with you", "without any friends")):
                                    score -= 35
                            # 2) Degree adverb with 'to'-infinitive / 'for' complement ('fast enough to catch', 'old enough for school')
                            if (
                                (tok.i + 1 < len(q_doc) and q_doc[tok.i + 1].lower_ in ("to", "for"))
                                or any(c.dep_ in ("xcomp", "advcl") for c in tok.head.children)
                            ):
                                if "to the degree that is necessary or wanted" in s_defn:
                                    score += 45
                                elif "fairly but not very" in s_defn:
                                    score -= 35
                            # 3) Directional motion adverb vs. dimensional width measurement ('swam across' vs '2 km across')
                            has_measure_np = tok.i > 0 and (
                                q_doc[tok.i - 1].pos_ == "NUM"
                                or (
                                    q_doc[tok.i - 1].pos_ == "NOUN"
                                    and any(c.pos_ == "NUM" or c.dep_ == "nummod" for c in q_doc[tok.i - 1].children)
                                )
                            )
                            if not has_measure_np and tok.head.pos_ == "VERB":
                                if any(k in s_defn for k in ("from one side of something to the other", "towards someone or something on the other side")):
                                    score += 40
                                elif "used to show how wide something is" in s_defn:
                                    score -= 40
                            # 4) Spatial vs. Temporal separation ('in between' two houses vs two times/events)
                            _adv_time_nouns = {"day", "hour", "minute", "second", "week", "month", "year", "decade", "century", "time", "period", "meal", "job", "event", "monday", "sunday", "christmas", "summer", "winter"}
                            _has_adv_time = any(t.lemma_.lower() in _adv_time_nouns or t.ent_type_ in ("DATE", "TIME") for t in q_doc)
                            if not _has_adv_time:
                                if "in or through the space that separates" in s_defn:
                                    score += 35
                                elif "in the time that separates" in s_defn:
                                    score -= 35
                except Exception:
                    pass

            # 6b. Adverbial Governing Verb Valency & Selectional Alignment
            if quote and target_p in ("adv", "r"):
                if s_defn.startswith("(of the elements)") and not any(
                    w in q_low_all for w in ("wind", "storm", "rain", "sea", "wave", "waves", "weather", "gale", "snow", "tempest", "thunder", "lightning")
                ):
                    score -= 45
                try:
                    nlp_r = cls.get_spacy()
                    doc_r = nlp_r(quote) if nlp_r else None
                    if doc_r is not None:
                        for tok_r in doc_r:
                            if cls.form_matches(tok_r.text.lower(), target_w_clean) or cls.form_matches(tok_r.lemma_.lower(), target_w_clean):
                                if tok_r.head and tok_r.head.pos_ == "VERB":
                                    gov_v = tok_r.head.lemma_.lower()
                                    if gov_v in s_def_tokens or any(re.search(rf"\b{re.escape(gov_v)}\b", ex.lower()) for ex in s.get("examples", [])):
                                        score += 35
                                    v_supers = cls.get_wordnet_supersenses(gov_v, pos="v")
                                    is_effort_motion_v = (
                                        gov_v in ("work", "type", "write", "run", "row", "pedal", "scrub", "dig", "fight", "struggle", "think", "kick", "swim", "drive", "ride", "climb", "dance", "spin", "beat", "scribble", "search")
                                        or bool(v_supers & {"verb.motion", "verb.creation", "verb.competition", "verb.contact"})
                                    ) and gov_v not in ("shout", "yell", "scream", "argue", "complain", "protest", "react", "glare", "stare")
                                    if is_effort_motion_v:
                                        if any(k in s_defn for k in ("energy", "effort", "speed", "fast", "hard", "rapidly", "quickly")):
                                            score += 35
                                        elif any(k in s_defn for k in ("angry", "anger", "annoyed", "upset", "sad")):
                                            score -= 20
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
            idx = next((i for i, s in enumerate(senses) if cls._sense_matches_pos(s, target_p)), None)
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
                    if not (target_p and not cls._sense_matches_pos(s, target_p))]
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
            raw_register = re.sub(r"\bliterary\s+humorous\b", "literary", raw_register, flags=re.IGNORECASE)
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
            entry_poses = {pc for s in senses for pc in cls._sense_pos_codes(s)}
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
                if not pos_tag or cls._sense_matches_pos(s, pos_tag) or not s.get("pos")
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
        # British/American particle equivalence ('turn around' <-> 'turn round')
        if "around" in tokens or "round" in tokens:
            swapped_key = (
                re.sub(r"\baround\b", "round", key)
                if "around" in tokens
                else re.sub(r"\bround\b", "around", key)
            )
            if swapped_key != key:
                swapped_tokens = cls._phrase_tokens(swapped_key)
                if len(swapped_tokens) >= 2:
                    for allow_mod in ([False, True] if allow_modifier else [False]):
                        pat_sw = cls._phrase_regex(swapped_tokens, 2, allow_modifier=allow_mod)
                        if pat_sw:
                            rx_sw = re.compile(pat_sw)
                            for cand in candidates:
                                text = str(cand or "").strip()
                                if text and rx_sw.search(cls._entry_text_normalizer(text)) and _is_usable(text):
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
        key_variants = {key}
        if re.search(r"\baround\b", key):
            key_variants.add(re.sub(r"\baround\b", "round", key))
        elif re.search(r"\bround\b", key):
            key_variants.add(re.sub(r"\bround\b", "around", key))
        for host in cls._phrase_headwords(key, headword):
            entry = cls.get_ldoce_entry(host)
            if not entry:
                continue
            for block in entry.get("phrasal_verbs") or []:
                if not isinstance(block, dict):
                    continue
                names = [block.get("phrase"), block.get("headword")] \
                    + list(block.get("variants") or []) + list(block.get("alternates") or [])
                if not any(kv in cls._declared_unit_spellings(name) for kv in key_variants for name in names):
                    continue
                # If key only matches via variants/alternates (not primary phrase or headword),
                # and this block contains no example showing this key, while the host entry's
                # own senses contain an example showing the key, skip this block so the host sense can claim it.
                primary = [block.get("phrase"), block.get("headword")]
                if not any(kv in cls._declared_unit_spellings(p) for kv in key_variants for p in primary if p):
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
        key_variants = {key}
        if re.search(r"\baround\b", key):
            key_variants.add(re.sub(r"\baround\b", "round", key))
        elif re.search(r"\bround\b", key):
            key_variants.add(re.sub(r"\bround\b", "around", key))
        senses = [s for s in (block.get("senses") or [])
                  if isinstance(s, dict) and str(s.get("definition") or "").strip()]
        for s in senses:
            s.setdefault("pos", "verb")
        owned = [s for s in senses
                 if any(cls._unit_key(unit) in key_variants
                        for pattern in (s.get("patterns") or [])
                        for unit in cls._slash_variants(pattern))]
        return owned + [s for s in senses if s not in owned]

    @classmethod
    def _unit_phrase_rows(cls, key: str,
                          headword: Optional[str] = None) -> List[Tuple[str, Dict[str, Any]]]:
        """(host, row) for every PHRASES or CROSSREFS row PedLex names `key`."""
        out: List[Tuple[str, Dict[str, Any]]] = []
        keys = {key} | cls._declared_unit_spellings(key)
        for host in cls._phrase_headwords(key, headword):
            entry = cls.get_ldoce_entry(host)
            if not entry:
                continue
            for row in (entry.get("phrases") or []) + (entry.get("crossrefs") or []):
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
        key = cls._unit_key(key) or key
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
            key_toks_0 = key.split()
            is_passive_k = len(key_toks_0) == 3 and key_toks_0[0] in ("be", "feel", "get") and key_toks_0[1].endswith("ed")
            all_hits = cls.ldoce_phrase_hits(key, headword)
            licensed = {
                text for tier, text in all_hits
                if tier in ("phrase", "pattern", "collocation", "cross_collocation")
                and not ("hospital" in str(text).lower() and context_sentence and not any(hw in context_sentence.lower() for hw in ("hospital", "patient", "ward", "doctor", "ill", "sick", "injured", "emergency")))
                and not (is_passive_k and re.search(rf"\b{re.escape(key_toks_0[-1])}\s+(?:doing\s+|somebody\b|sb\b)", re.sub(r"[()\[\]]+", " ", str(text).lower())))
            }
            is_caus_adj_k = len(key_toks_0) == 2 and key_toks_0[0] in ("make", "keep", "find", "leave", "remain", "stay", "fall")
            ordered_hosts_0 = list(reversed(hosts)) if is_caus_adj_k else list(hosts)
            keys_0 = {key} | cls._declared_unit_spellings(key)
            for host in ordered_hosts_0:
                entry = cls.get_ldoce_entry(host)
                if not entry:
                    continue
                for sense in entry.get("senses") or []:
                    if not isinstance(sense, dict) or not str(sense.get("definition") or "").strip():
                        continue
                    for u in sense.get("units") or []:
                        u_str = u.get("phrase") if isinstance(u, dict) else str(u or "")
                        if u_str and bool(keys_0 & cls._declared_unit_spellings(u_str)):
                            return (host, sense, "unit_sense")
            matching_cands = []
            for host in ordered_hosts_0:
                entry = cls.get_ldoce_entry(host)
                if not entry:
                    continue
                for sense in entry.get("senses") or []:
                    if not isinstance(sense, dict) or not str(sense.get("definition") or "").strip():
                        continue
                    if "if a chemical substance reacts" in str(sense.get("definition") or "").lower() and not has_chem_ctx:
                        continue
                    for pattern in sense.get("patterns") or []:
                        p_low_0 = str(pattern).lower()
                        p_flat_0 = re.sub(r"[()\[\]]+", " ", p_low_0)
                        if "hospital" in p_low_0 and context_sentence and not any(hw in context_sentence.lower() for hw in ("hospital", "patient", "ward", "doctor", "ill", "sick", "injured", "emergency")):
                            continue
                        if is_passive_k and re.search(rf"\b{re.escape(key_toks_0[-1])}\s+(?:doing\s+|somebody\b|sb\b)", p_flat_0):
                            continue
                        if is_caus_adj_k and host == key_toks_0[1] and re.search(
                            rf"\b{re.escape(key_toks_0[0])}s?\s+(?:it\s+|something\s+|somebody\s+|sb\s+|sth\s+)?{re.escape(key_toks_0[1])}\b",
                            p_low_0
                        ):
                            matching_cands.append((host, sense, "sense_pattern"))
                            break
                        for unit in cls._slash_variants(pattern):
                            if cls._entry_text_normalizer(unit) in licensed:
                                matching_cands.append((host, sense, "sense_pattern"))
                                break
                if is_caus_adj_k and host == key_toks_0[1] and matching_cands:
                    break
            if matching_cands:
                if is_passive_k:
                    # Also include sibling senses of the same host whose examples explicitly attest the passive frame
                    # (e.g. 'admit' sense #3 'be admitted to the club' alongside sense #2 'be admitted to/into something')
                    existing_s_ids = {id(c[1]) for c in matching_cands}
                    for host in hosts:
                        entry = cls.get_ldoce_entry(host)
                        if not entry:
                            continue
                        for sense in entry.get("senses") or []:
                            if id(sense) in existing_s_ids or not isinstance(sense, dict) or not str(sense.get("definition") or "").strip():
                                continue
                            if any("hospital" in str(p).lower() for p in (sense.get("patterns") or [])) and context_sentence and not any(hw in context_sentence.lower() for hw in ("hospital", "patient", "ward", "doctor", "ill", "sick", "injured", "emergency")):
                                continue
                            if any(re.search(rf"\b{re.escape(key_toks_0[-1])}\s+(?:doing\s+|somebody\b|sb\b)", re.sub(r"[()\[\]]+", " ", str(p).lower())) for p in (sense.get("patterns") or [])):
                                continue
                            if cls._unit_example(key, list(sense.get("examples") or [])):
                                matching_cands.append((host, sense, "sense_pattern"))
                                existing_s_ids.add(id(sense))
                if len(matching_cands) >= 2 and context_sentence:
                    mock_entry = {"word": hosts[0], "senses": [c[1] for c in matching_cands]}
                    u_idx, _u_sc, _u_mar, _u_conf = cls._lock_sense(
                        mock_entry,
                        quote=context_sentence,
                        target_pos="verb" if is_passive_k else None,
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
                        if any("hospital" in str(p).lower() for p in (sense.get("patterns") or [])) and context_sentence and not any(hw in context_sentence.lower() for hw in ("hospital", "patient", "ward", "doctor", "ill", "sick", "injured", "emergency")):
                            continue
                        if cls._unit_example(key, list(sense.get("examples") or [])):
                            return (host, sense, "sense_pattern")

            # Tier 2c: Attested in PedLex collocation box (e.g. 'a family member / a member of the family'
            # under 'member', or 'a passenger plane / jet' under 'passenger').
            colloc_hits = [
                text for tier, text in all_hits
                if tier in ("collocation", "cross_collocation")
                and not ("hospital" in str(text).lower() and context_sentence and not any(hw in context_sentence.lower() for hw in ("hospital", "patient", "ward", "doctor", "ill", "sick", "injured", "emergency")))
            ]
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
                        and not ("hospital" in str(t).lower() and context_sentence and not any(hw in context_sentence.lower() for hw in ("hospital", "patient", "ward", "doctor", "ill", "sick", "injured", "emergency")))
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
                # Case A: Copula 'be'/'feel'/'get' + Adj/Participle + Prep (e.g. 'be smart about', 'be exhausted from', 'be featured in', 'be measured in')
                if tokens_k[0] in ("be", "is", "was", "were", "are", "been", "feel", "get") and len(tokens_k) == 3:
                    adj_host, prep_tok = tokens_k[1], tokens_k[2]
                    adj_entry = cls.get_ldoce_entry(adj_host)
                    if adj_entry:
                        adj_senses = [
                            s for s in (adj_entry.get("senses") or [])
                            if isinstance(s, dict) and str(s.get("definition") or "").strip()
                            and "adj" in str(s.get("pos") or adj_entry.get("pos") or "").lower()
                        ]
                        # Verify the adjective is attested with this preposition in examples/patterns/units
                        for s in adj_senses:
                            s_exs = [str(e).lower() for e in (s.get("examples") or [])]
                            s_pats = []
                            for p in (s.get("patterns") or []):
                                s_pats.extend(v.lower() for v in cls._slash_variants(str(p)))
                            for u_obj in (s.get("units") or []):
                                u_raw = u_obj.get("phrase") if isinstance(u_obj, dict) else str(u_obj or "")
                                s_pats.extend(v.lower() for v in cls._slash_variants(str(u_raw)))
                            if any(
                                f"{adj_host} {prep_tok}" in x or re.search(rf"\b{re.escape(adj_host)}\b.*\b{re.escape(prep_tok)}\b", x)
                                for x in s_exs + s_pats
                            ):
                                return (adj_host, s, "sense_pattern")
                        # Also check base verb entry when adj_host is a -ed participle (e.g. 'measured' -> 'measure', 'featured' -> 'feature')
                        v_bases = [
                            c for c in ([cls._surface_lemma(adj_host)] + cls._headword_candidates(adj_host))
                            if c and c != adj_host and cls.get_ldoce_entry(c)
                        ] if adj_host.endswith("ed") else []
                        v_base = v_bases[0] if v_bases else ""
                        if not v_base:
                            if adj_senses and context_sentence and re.search(rf"\b{re.escape(adj_host)}\s+{re.escape(prep_tok)}\b", context_sentence.lower()):
                                if len(adj_senses) >= 2:
                                    mock_e = {"word": adj_host, "senses": adj_senses}
                                    u_idx = cls._lock_sense(mock_e, quote=context_sentence, target_pos="adjective")
                                    if u_idx is not None and 0 <= u_idx < len(adj_senses):
                                        return (adj_host, adj_senses[u_idx], "sense_pattern")
                                return (adj_senses[0] and (adj_host, adj_senses[0], "sense_pattern"))
                    # Base verb entry lookup for passive/predicative participles ('be obsessed with', 'be featured in', 'be measured in', 'be admitted to', 'be based on', 'be modeled on')
                    v_bases = [
                        c for c in ([cls._surface_lemma(adj_host)] + cls._headword_candidates(adj_host))
                        if c and c != adj_host and cls.get_ldoce_entry(c)
                    ] if adj_host.endswith("ed") else []
                    v_base = v_bases[0] if v_bases else ""
                    if v_base and v_base != adj_host:
                        v_entry = cls.get_ldoce_entry(v_base)
                        if v_entry:
                            v_senses = [
                                s for s in (v_entry.get("senses") or [])
                                if isinstance(s, dict) and str(s.get("definition") or "").strip()
                                and "verb" in str(s.get("pos") or v_entry.get("pos") or "").lower()
                            ]
                            # Include British/American doubled-consonant participle variants ('modeled' <-> 'modelled')
                            part_forms = {adj_host}
                            if adj_host.endswith("led") and not adj_host.endswith("lled"):
                                part_forms.add(adj_host[:-3] + "lled")
                            elif adj_host.endswith("lled"):
                                part_forms.add(adj_host[:-4] + "led")
                            part_alt = "|".join(re.escape(pf) for pf in sorted(part_forms))
                            matching_v_senses = []
                            for s in v_senses:
                                s_pats = []
                                for p in (s.get("patterns") or []):
                                    s_pats.extend(v.lower() for v in cls._slash_variants(str(p)))
                                for pb in (s.get("pattern_bundles") or []):
                                    if isinstance(pb, dict) and pb.get("pattern"):
                                        s_pats.extend(v.lower() for v in cls._slash_variants(str(pb["pattern"])))
                                for u_obj in (s.get("units") or []):
                                    u_raw = u_obj.get("phrase") if isinstance(u_obj, dict) else str(u_obj or "")
                                    s_pats.extend(v.lower() for v in cls._slash_variants(str(u_raw)))
                                s_exs = [str(e).lower() for e in (s.get("examples") or [])]
                                # Skip domain-narrow 'be admitted to hospital' when context has no hospital/medical clue
                                if any("hospital" in p for p in s_pats) and context_sentence and not any(hw in context_sentence.lower() for hw in ("hospital", "patient", "ward", "doctor", "ill", "sick", "injured", "emergency")):
                                    continue
                                valid_passive_pats = [
                                    p for p in s_pats
                                    if re.search(rf"\b(?:{re.escape(v_base)}|{part_alt})\b.*\b{re.escape(prep_tok)}\b", p)
                                    and not re.search(rf"\b{re.escape(v_base)}\s+{re.escape(prep_tok)}\s+(?:somebody|sb|doing)\b", p)
                                ]
                                if valid_passive_pats or any(
                                    re.search(rf"\b(?:{part_alt})\s+{re.escape(prep_tok)}\b", ex_s)
                                    for ex_s in s_exs
                                ):
                                    matching_v_senses.append(s)
                            if matching_v_senses:
                                if len(matching_v_senses) >= 2 and context_sentence:
                                    mock_ve = {"word": v_base, "senses": matching_v_senses}
                                    v_idx = cls._lock_sense(mock_ve, quote=context_sentence, target_pos="verb")
                                    if v_idx is not None and 0 <= v_idx < len(matching_v_senses):
                                        return (v_base, matching_v_senses[v_idx], "sense_pattern")
                                return (v_base, matching_v_senses[0], "sense_pattern")

                # Case B: Light-verb + Possessive/Slot + Head Noun (e.g. 'do part' from 'do one's part')
                light_verbs = {"do", "make", "take", "give", "show", "have", "play", "pay", "put", "keep", "get"}
                if tokens_k[0] in light_verbs:
                    # Sub-frame 1: Tail 2-token prepositional frame (e.g. 'show thanks to' -> 'thanks to')
                    # or Light-verb + [Attributive Adj] + Head Noun (e.g. 'make smart choices' -> 'make choices' on 'choice')
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
                        # Check Light-verb + Head Noun stripping intervening attributive modifier (e.g. 'make smart choices' -> 'make choice/choices')
                        v_tok_3, n_tok_3 = tokens_k[0], tokens_k[2]
                        n_lem_3 = cls._surface_lemma(n_tok_3) or n_tok_3
                        for cand_nh in dict.fromkeys([n_lem_3, n_tok_3]):
                            vn_owner = cls._unit_owning_sense(f"{v_tok_3} {cand_nh}", headword=cand_nh, context_sentence=context_sentence)
                            if vn_owner:
                                return vn_owner
                            nh_entry = cls.get_ldoce_entry(cand_nh)
                            if nh_entry:
                                for s in (nh_entry.get("senses") or []):
                                    if not isinstance(s, dict) or not str(s.get("definition") or "").strip():
                                        continue
                                    if cls._unit_example(f"{v_tok_3} {cand_nh}", list(s.get("examples") or [])) or cls._unit_example(f"{v_tok_3} {n_tok_3}", list(s.get("examples") or [])):
                                        return (cand_nh, s, "sense_pattern")

                    # Sub-frame 2: Light-verb + Head Noun with role/partitive pattern (e.g. 'do part' -> 'part')
                    # or Causative/Copular Verb + Adjective Complement (e.g. 'make possible' -> 'possible', 'keep silent' -> 'silent')
                    if len(tokens_k) == 2:
                        v_tok, n_tok = tokens_k[0], tokens_k[1]
                        n_entry = cls.get_ldoce_entry(n_tok)
                        if n_entry:
                            role_senses = []
                            for s in (n_entry.get("senses") or []):
                                if not isinstance(s, dict) or not str(s.get("definition") or "").strip():
                                    continue
                                pats = [str(p).lower().strip() for p in (s.get("patterns") or [])]
                                pb_pats = [
                                    str(pb.get("pattern") or "").lower().strip()
                                    for pb in (s.get("pattern_bundles") or [])
                                    if isinstance(pb, dict) and pb.get("pattern")
                                ]
                                units = [str(u).lower().strip() for u in (s.get("units") or [])]
                                if any(
                                    p.startswith(f"have a {n_tok}")
                                    or p.startswith(f"play a {n_tok}")
                                    or p.startswith(f"{v_tok} a {n_tok}")
                                    or p.startswith(f"somebody's {n_tok}")
                                    or p.startswith(f"one's {n_tok}")
                                    or (
                                        v_tok in ("make", "keep", "find", "leave", "remain", "stay", "fall")
                                        and re.search(rf"\b{re.escape(v_tok)}s?\s+(?:it\s+|something\s+|somebody\s+|sb\s+|sth\s+)?{re.escape(n_tok)}\b", p)
                                    )
                                    for p in pats + pb_pats + units
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

        # 1. A sense that explicitly declares this exact unit in sense['units'] across any constituent host
        for host in ordered + [h for h in hosts if h not in ordered]:
            entry = cls.get_ldoce_entry(host)
            if not entry:
                continue
            senses = [s for s in (entry.get("senses") or [])
                      if isinstance(s, dict) and str(s.get("definition") or "").strip()]
            for sense in senses:
                for u in sense.get("units") or []:
                    u_str = u.get("phrase") if isinstance(u, dict) else str(u or "")
                    if u_str and bool(keys & cls._declared_unit_spellings(u_str)):
                        return (host, sense, "phrase_row")

        # 2. Pre-resolved (resolved_word, resolved_sense_idx) on PHRASES / CROSSREFS rows in pedlex.db
        # (excluding weak 'sense_example' matches until after pattern checks)
        resolved_row_cands: List[Tuple[str, Dict[str, Any]]] = []
        for _r_host, row in rows:
            if not isinstance(row, dict) or row.get("resolved_via") == "sense_example":
                continue
            rw = row.get("resolved_word") or row.get("target_word")
            rs_idx = row.get("resolved_sense_idx")
            if rw and rs_idx is not None:
                r_entry = cls.get_ldoce_entry(str(rw))
                if r_entry:
                    r_senses = r_entry.get("senses") or []
                    if 0 <= int(rs_idx) < len(r_senses):
                        rs_obj = r_senses[int(rs_idx)]
                        if isinstance(rs_obj, dict) and str(rs_obj.get("definition") or "").strip():
                            if (str(rw), rs_obj) not in resolved_row_cands:
                                resolved_row_cands.append((str(rw), rs_obj))
        if len(resolved_row_cands) == 1:
            return (resolved_row_cands[0][0], resolved_row_cands[0][1], "phrase_row")
        elif len(resolved_row_cands) > 1:
            non_first = [rc for rc in resolved_row_cands if rc[0] != tokens[0]]
            chosen_rc = non_first[0] if non_first else resolved_row_cands[0]
            return (chosen_rc[0], chosen_rc[1], "phrase_row")
        for host in ordered:
            entry = cls.get_ldoce_entry(host)
            if not entry:
                continue
            senses = [s for s in (entry.get("senses") or [])
                      if isinstance(s, dict) and str(s.get("definition") or "").strip()]
            pat_matches: List[Dict[str, Any]] = []
            for sense in senses:
                matched_this = False
                for pattern in sense.get("patterns") or []:
                    for unit in cls._slash_variants(pattern):
                        frame = cls._phrase_tokens(unit)
                        if len(frame) > 1 and frame[1:] == tail:
                            pat_matches.append(sense)
                            matched_this = True
                            break
                    if matched_this:
                        break
            if len(pat_matches) == 1:
                return (host, pat_matches[0], "phrase_row")
            elif len(pat_matches) > 1:
                if context_sentence:
                    mock_entry = {"word": key, "senses": pat_matches}
                    u_idx = cls._lock_sense(mock_entry, quote=context_sentence, target_pos="verb" if host == tokens[0] else None)
                    if u_idx is not None and 0 <= u_idx < len(pat_matches):
                        return (host, pat_matches[u_idx], "phrase_row")
                return (host, pat_matches[0], "phrase_row")
            for sense in senses:
                if cls._unit_example(key, list(sense.get("examples") or [])):
                    return (host, sense, "phrase_row")

        # Direct index lookup fallback via ldoce_phrase_index without LIMIT 1 truncation
        candidates = cls.lookup_ldoce_phrase_candidates(key, headword)
        if candidates:
            # Collect valid candidate (host, sense) pairs
            valid_cands = []
            for c_host, c_sense_idx, c_tier, _c_raw in candidates:
                if "hospital" in str(_c_raw).lower() and context_sentence and not any(hw in context_sentence.lower() for hw in ("hospital", "patient", "ward", "doctor", "ill", "sick", "injured", "emergency")):
                    continue
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
        k_parts = key.lower().split()
        if (
            len(k_parts) == 4
            and k_parts[0] in {"be", "keep", "stay", "get", "lose", "remain", "fall", "come", "go", "put", "make", "take"}
            and k_parts[1] in {"in", "on", "at", "out", "off", "under", "beyond", "within"}
            and k_parts[3] in {"with", "to", "of", "from", "for", "in", "on", "at"}
        ):
            for sub_k in (f"{k_parts[0]} {k_parts[1]} {k_parts[2]}", f"{k_parts[1]} {k_parts[2]} {k_parts[3]}", f"{k_parts[1]} {k_parts[2]}"):
                keys.add(sub_k)
                keys |= cls._declared_unit_spellings(sub_k)
        unit_sibling_senses = [
            s for s in senses
            if any(
                bool(keys & cls._declared_unit_spellings(u.get("phrase") if isinstance(u, dict) else str(u or "")))
                for u in (s.get("units") or [])
            )
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

        alt_pats = alt.get("patterns") or []
        if any("hospital" in str(p).lower() for p in alt_pats) and context_sentence and not any(hw in context_sentence.lower() for hw in ("hospital", "patient", "ward", "doctor", "ill", "sick", "injured", "emergency")):
            return None
        # Syntactic unit compatibility: the override candidate must support the key's prepositions/particles and non-host content words.
        # It cannot override a specific multiword frame (e.g. 'start with') with an incompatible sense (e.g. 'start at/from'),
        # nor override a lexical collocation (e.g. 'attach importance to') with an incompatible literal sense (e.g. 'attach sth to sth').
        key_tokens = key.lower().split()
        if len(key_tokens) == 3 and key_tokens[0] in ("be", "feel", "get") and key_tokens[1].endswith("ed"):
            if any(re.search(rf"\b{re.escape(key_tokens[-1])}\s+(?:doing\s+|somebody\b|sb\b)", re.sub(r"[()\[\]]+", " ", str(p).lower())) for p in alt_pats):
                return None
        if len(key_tokens) >= 2:
            key_particles = [w for w in key_tokens[1:] if w in cls._PATTERN_PREPS or w in cls._PATTERN_PARTICLES]
            key_content_words = [w for w in key_tokens[1:] if w not in cls._PATTERN_PREPS and w not in cls._PATTERN_PARTICLES]
            alt_units = [u.get("phrase") if isinstance(u, dict) else str(u or "") for u in (alt.get("units") or [])]
            alt_examples = list(alt.get("examples") or [])
            for u_obj in (alt.get("units") or []):
                if isinstance(u_obj, dict) and u_obj.get("examples"):
                    alt_examples.extend(str(e) for e in u_obj["examples"] if e)
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
        for pb in (sense.get("pattern_bundles") or []):
            if isinstance(pb, dict) and pb.get("example"):
                pb_ex = str(pb["example"]).strip()
                if pb_ex and pb_ex not in pool:
                    pool.append(pb_ex)
        entry = cls.get_ldoce_entry(host) or {}
        senses_list = entry.get("senses") or []
        sense_pos_idx = next((idx for idx, s_item in enumerate(senses_list) if s_item is sense), None)
        rows = [r for r in (entry.get("phrases") or []) if isinstance(r, dict)]
        for unit in sense.get("units") or []:
            u_str = unit.get("phrase") if isinstance(unit, dict) else str(unit or "")
            if isinstance(unit, dict) and unit.get("examples"):
                pool.extend(str(e) for e in unit["examples"] if e and str(e) not in pool)
            unit_key = cls._unit_key(u_str)
            if not unit_key:
                continue
            for row in rows:
                if cls._unit_key(row.get("phrase")) == unit_key:
                    pool.extend(str(e) for e in (row.get("examples") or []) if e and str(e) not in pool)
        if sense_pos_idx is not None:
            for row in rows:
                r_s_idx = row.get("sense_idx")
                if r_s_idx is not None and int(r_s_idx) == sense_pos_idx:
                    examples = [str(e) for e in (row.get("examples") or []) if e]
                    if examples and cls._unit_example(key, examples, allow_modifier=True):
                        pool.extend(e for e in examples if e not in pool)
        # Sense-isolation guard: if the sense's own examples/units/linked rows already attest `key`,
        # return immediately without mixing in unlinked entry-level phrases or collocations from other senses.
        if pool and cls._unit_example(key, pool, allow_modifier=True):
            return pool
        for row in rows:
            r_s_idx = row.get("sense_idx")
            if sense_pos_idx is not None and r_s_idx is not None and int(r_s_idx) >= 0 and int(r_s_idx) != sense_pos_idx:
                continue
            examples = [str(e) for e in (row.get("examples") or []) if e]
            if examples and cls._unit_example(key, examples, allow_modifier=True):
                pool.extend(e for e in examples if e not in pool)
        # Entry collocations: authoritative collocation examples matching this unit
        for _cat, c_items in (entry.get("collocations") or {}).items():
            for it in (c_items or []):
                if isinstance(it, dict) and it.get("example"):
                    ex_str = str(it["example"]).strip()
                    if ex_str and cls._unit_example(key, [ex_str]) and ex_str not in pool:
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
        if host:
            h_ent = cls.get_ldoce_entry(host)
            if h_ent:
                for _sec, c_items in (h_ent.get("collocations") or {}).items():
                    for c_it in (c_items or []):
                        if isinstance(c_it, dict) and c_it.get("gloss") and (c_it.get("collocation") or c_it.get("phrase")):
                            c_raw_p = str(c_it.get("collocation") or c_it.get("phrase") or "")
                            c_col = cls._entry_text_normalizer(c_raw_p)
                            if c_col and (c_col == k_low or k_low in cls._declared_unit_spellings(c_raw_p)):
                                g_clean = re.sub(r"^[\(\s=]+|[\)\s]+$", "", str(c_it["gloss"])).strip()
                                if g_clean and len(g_clean) >= 6:
                                    return g_clean
        if "late, early, or at the right time" in defn_clean.lower():
            if k_low.endswith(" late"):
                defn_clean = re.sub(
                    r"(?i)\blate,\s*early,\s*or\s*at\s*the\s*right\s*time\b",
                    "late (behind schedule or later than expected)",
                    defn_clean,
                )
            elif k_low.endswith(" early"):
                defn_clean = re.sub(
                    r"(?i)\blate,\s*early,\s*or\s*at\s*the\s*right\s*time\b",
                    "early (ahead of schedule)",
                    defn_clean,
                )
            elif k_low.endswith(" on time"):
                defn_clean = re.sub(
                    r"(?i)\blate,\s*early,\s*or\s*at\s*the\s*right\s*time\b",
                    "on time (according to schedule)",
                    defn_clean,
                )
        m_pref = re.match(r"^((?:\[[^\]]+\]\s*)+)", defn_clean)
        prefix_str = m_pref.group(1) if m_pref else ""
        defn_body = re.sub(r"^(?:\[[^\]]+\]\s*)+", "", defn_clean).strip()
        if defn_body and not re.match(r"^(?:to|if|used)\s+", defn_body, flags=re.IGNORECASE):
            body_low = defn_body.lower()
            if k_low.startswith(("keep in ", "stay in ", "remain in ")) and body_low.startswith(("talking ", "writing ", "communicating ")):
                return f"{prefix_str}to continue {defn_body}; to stay in contact"
            if k_low.startswith("get in ") and body_low.startswith(("talking ", "writing ", "communicating ")):
                return f"{prefix_str}to start {defn_body}; to make contact"
            if k_low.startswith("be in ") and body_low.startswith(("talking ", "writing ", "communicating ")):
                return f"{prefix_str}to be {defn_body}; to be in contact"
            if k_low.startswith(("come to ", "rise to ", "achieve ")) and body_low.startswith(("the fact of being ", "a quality of being ", "the quality of being ", "the state of being ")):
                for prefix_pat in ("the fact of being ", "a quality of being ", "the quality of being ", "the state of being "):
                    if body_low.startswith(prefix_pat):
                        return f"{prefix_str}to become " + defn_body[len(prefix_pat):]
            if k_low.startswith(("bring ", "put ", "restore ")) and (" to " in k_low or " into " in k_low) and body_low.startswith(("a quality of being ", "the quality of being ", "the state of being ")):
                for prefix_pat in ("a quality of being ", "the quality of being ", "the state of being "):
                    if body_low.startswith(prefix_pat):
                        return f"{prefix_str}to make something " + defn_body[len(prefix_pat):]
            if k_low.startswith(("do ", "try ")) and body_low.startswith("the most that can be done"):
                return f"{prefix_str}to do " + defn_body + "; to try as hard as possible"
            if k_low.startswith("do ") and body_low.startswith(("the actions and activities ", "the effort contributed ")):
                return f"{prefix_str}to perform " + defn_body
            if (
                expr_type == "collocation"
                and len(k_low.split()) == 2
                and host.lower() == k_low.split()[0]
                and "noun" in s_pos
                and body_low.startswith(("someone who ", "a person who "))
            ):
                head_n = k_low.split()[-1]
                head_ent = cls.get_ldoce_entry(head_n) or cls.get_ldoce_entry(cls._surface_lemma(head_n) or "")
                for hs in (head_ent.get("senses") or []) if head_ent else []:
                    if cls._pos_code(hs.get("pos")) == "n":
                        hd = (cls.format_ldoce_definition(hs) or str(hs.get("definition") or "")).strip()
                        hd_body = re.sub(r"^(?:\[[^\]]+\]\s*)+", "", hd).strip()
                        if hd_body and not hd_body.lower().startswith(("someone who ", "a person who ")):
                            return f"{prefix_str}{k_low}: {hd_body}"
            m_sth_you_do = re.match(r"^something\s+(?:that\s+)?you\s+do\s+(.+)$", defn_body, flags=re.IGNORECASE)
            if m_sth_you_do and k_low.split()[0] in ("get", "take", "make", "have", "do", "give", "seek", "find"):
                return f"{prefix_str}to do something " + m_sth_you_do.group(1)
            if k_low.startswith("take ") and body_low.startswith("the process of doing something"):
                return f"{prefix_str}to do something" + defn_body[len("the process of doing something"):]
            if k_low.startswith(("pose ", "present ")) and body_low.startswith("someone or something that is "):
                return f"{prefix_str}to be " + defn_body
            if k_low.startswith(("make ", "leave ", "find ", "clear ", "create ")) and (
                host in ("room", "space", "way", "place")
                or body_low.startswith(("space ", "empty space ", "an area "))
            ):
                return f"{prefix_str}to provide or create " + defn_body
            if k_low.startswith(("fall ", "keep ", "remain ", "stay ", "feel ", "get ")) and "adj" in s_pos:
                v_head = k_low.split()[0]
                if v_head in ("keep", "remain", "stay"):
                    return f"{prefix_str}to continue to be " + defn_body
                return f"{prefix_str}to become or feel " + defn_body
            if k_low.startswith("make ") and "adj" in s_pos:
                return f"{prefix_str}to make something " + defn_body
            if host and host.lower() == "member" and "a person or country that belongs to" in body_low and not any(g in k_low for g in ("un ", "eu ", "nato ", "state", "union", "alliance", "treaty")):
                return re.sub(r"(?i)\ba person or country that belongs to a group or organization\b", "a person who belongs to a particular family, group, or organization", defn_clean)
        elif k_low.startswith("make ") and "adj" in s_pos:
            m_if_can_be = re.match(rf"^if\s+something\s+is\s+{re.escape(host.lower())},\s*it\s+can\s+be\s+(.+)$", defn_body, flags=re.IGNORECASE)
            if m_if_can_be:
                return "to make something able to happen, be " + m_if_can_be.group(1)
        elif host and k_low.split()[0] in ("make", "take"):
            m_if_have_a = re.match(rf"^if\s+you\s+have\s+an?\s+{re.escape(host.lower())},\s*you\s+(?:can\s+)?(.+)$", defn_body, flags=re.IGNORECASE)
            if m_if_have_a:
                return f"{prefix_str}to " + m_if_have_a.group(1)
        elif defn_clean.lower() == "to have or experience a particular feeling" and k_low.startswith("take "):
            toks = [t for t in k_low.split()[1:] if len(t) >= 3 and t not in cls._PHRASE_PARTICLE_WORDS and t not in cls._PHRASE_SLOT_WORDS]
            if toks:
                n_entry = cls.get_ldoce_entry(toks[0])
                for ns in (n_entry.get("senses") or []) if n_entry else []:
                    nd = str(ns.get("definition") or "").strip()
                    if nd.lower().startswith("the feeling of "):
                        return "to feel or experience " + nd[len("the feeling of "):]
        return defn_clean or raw_defn

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
            (
                expr_type not in cls.EXPRESSION_TYPE_LABELS
                and cls.determine_contextual_pos(key.split()[0], context_sentence) == "verb"
            )
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
                        ctx_words_low = set(re.findall(r"[a-z]+", context_sentence.lower()))
                        _pat_slot_meta = {
                            "something", "somebody", "someone", "sth", "sb", "oneself",
                            "one's", "etc", "to", "for", "with", "from", "in", "on", "at",
                            "by", "of", "into", "out", "up", "down", "off", "away", "back",
                            "over", "through", "around", "round", "about", "along", "ahead",
                            "a", "an", "the", "doing", "do", "be", "being",
                        }
                        for s in v_entry.get("senses", []):
                            for p in (s.get("patterns") or []):
                                norm_p = cls._entry_text_normalizer(p)
                                p_toks = norm_p.split()
                                if not p_toks or not cls.form_matches(p_toks[0], base_verb):
                                    continue
                                if all(re.search(r'(?<!\w)' + re.escape(tok) + r'(?!\w)', norm_p) for tok in other_tokens):
                                    # If pattern requires specific lexical nouns beyond verb + particle + slots
                                    # (e.g. 'go on a trip/tour/cruise'), require at least one to appear in context_sentence
                                    extra_lex = [
                                        wt for wt in p_toks[1:]
                                        if wt not in other_tokens and not any( sub in _pat_slot_meta for sub in wt.split("/") )
                                        and all(len(sub) >= 3 and sub.isalpha() for sub in wt.split("/"))
                                    ]
                                    if extra_lex and not any(
                                        any(cls.form_matches(cw, alt) for cw in ctx_words_low for alt in wt.split("/"))
                                        for wt in extra_lex
                                    ):
                                        continue
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

                    # Detect separable direct object between verb and particle (e.g. 'take the giraffe's body back to their village')
                    has_sep_dobj_or_dest = False
                    if len(other_tokens) == 1 and context_sentence:
                        v_forms_re = "|".join(re.escape(f) for f in (cls.inflected_forms(base_verb) | {base_verb}))
                        pt_re = re.escape(other_tokens[0])
                        if re.search(rf"\b(?:{v_forms_re})\s+(?:the|a|an|his|her|their|our|my|your|its|this|that|these|those|[a-z]+'s)\s+[^.,;!?]{{1,35}}\b{pt_re}\b", ctx_low) or re.search(rf"\b{pt_re}\s+to\s+(?:the|a|an|his|her|their|our|my|your|its)\b", ctx_low):
                            has_sep_dobj_or_dest = True

                    prepared_b_senses = []
                    for s in b_senses:
                        s_def_l = str(s.get("definition") or "").lower()
                        if is_passive_remind and "to seem similar to" in s_def_l:
                            continue
                        if "if a chemical substance reacts" in s_def_l and not has_chem_ctx:
                            continue
                        if has_sep_dobj_or_dest and any(k in s_def_l for k in ("admit that you were wrong", "something that you said")):
                            continue
                        s_copy = dict(s)
                        if not s_copy.get("pos"):
                            s_copy["pos"] = "verb"
                        s_pats = [
                            p for p in (s_copy.get("patterns") or [])
                            if str(p).lower().strip() != key
                        ]
                        s_copy["patterns"] = s_pats
                        prepared_b_senses.append(s_copy)
                    if not prepared_b_senses and b_senses:
                        prepared_b_senses = [
                            dict(
                                s,
                                pos=s.get("pos") or "verb",
                                patterns=[p for p in (s.get("patterns") or []) if str(p).lower().strip() != key],
                            )
                            for s in b_senses
                        ]

                    candidates = []
                    seen_cand_defs = set()
                    # Add phrasal_verb_block senses first or deduplicate against them so no sense is ever duplicated
                    b_def_keys = {str(s.get("definition") or "").strip().lower() for s in prepared_b_senses if s.get("definition")}
                    for s in cand_base_senses:
                        s_def_l = str(s.get("definition") or "").strip().lower()
                        if not s_def_l or s_def_l in b_def_keys or s_def_l in seen_cand_defs:
                            continue
                        if "if a chemical substance reacts" in s_def_l and not has_chem_ctx:
                            continue
                        seen_cand_defs.add(s_def_l)
                        candidates.append((s, "sense_pattern"))
                    for s in prepared_b_senses:
                        s_def_l = str(s.get("definition") or "").strip().lower()
                        if s_def_l and s_def_l in seen_cand_defs:
                            continue
                        if s_def_l:
                            seen_cand_defs.add(s_def_l)
                        candidates.append((s, "phrasal_verb_block"))

                    mock_entry = {"word": base_verb, "senses": [c[0] for c in candidates], "_is_synthetic_pool": True}
                    res = cls._lock_sense(
                        mock_entry, quote=context_sentence, target_pos="verb", return_scores=True
                    )
                    idx = None
                    if res:
                        _idx, _best_score, _margin, _conf, score_trace = res
                        pos_trace = [(s_i, sc) for s_i, sc in score_trace if sc >= 0] or score_trace
                        sorted_scores = sorted(pos_trace, key=lambda x: -x[1])
                        top_indices = [s_i for s_i, _ in sorted_scores[:4]]
                        if _idx is not None and _idx in {s_i for s_i, _ in pos_trace} and _idx not in top_indices:
                            top_indices = [_idx] + top_indices[:3]
                        if len(top_indices) > 1 and _conf != "high" and context_sentence and enable_llm_disambiguation:
                            locked_et = expr_type if (expr_type in cls.EXPRESSION_TYPE_LABELS) else "phrasal verb"
                            top_cand_dicts = [
                                {
                                    "definition": cls.format_ldoce_definition(candidates[s_i][0]) or candidates[s_i][0].get("definition", ""),
                                    "pos": locked_et,
                                    "signpost": candidates[s_i][0].get("signpost", ""),
                                    "patterns": [
                                        p for p in (candidates[s_i][0].get("patterns") or [])
                                        if str(p).lower().strip() != key
                                    ],
                                    "examples": candidates[s_i][0].get("examples") or [],
                                    "raw_sense": candidates[s_i][0],
                                }
                                for s_i in top_indices
                            ]
                            chosen_sub = cls.disambiguate_via_llm(key, context_sentence, top_cand_dicts)
                            idx = top_indices[chosen_sub] if 0 <= chosen_sub < len(top_indices) else None
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
            t_pos_lookup = "verb" if is_phrasal_v else expr_type
            defn, ex = cls.get_ldoce_definition_and_example(
                clean_phrase, target_pos=t_pos_lookup, context_sentence=context_sentence)
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
                        if (
                            v_host != _host
                            and not (v_host in ("make", "keep", "find", "leave", "get", "have") and _host == tokens[-1])
                            and cls.determine_contextual_pos(v_host, context_sentence) == "verb"
                        ):
                            v_entry = cls.get_ldoce_entry(v_host)
                            if not v_entry:
                                continue
                            v_senses_all = v_entry.get("senses", [])
                            v_idx = cls._lock_sense(v_entry, quote=context_sentence, target_pos="verb")
                            cand_v_indices = ([v_idx] if v_idx is not None and 0 <= v_idx < len(v_senses_all) else []) + [
                                idx for idx, vs in enumerate(v_senses_all)
                                if idx != v_idx and cls._pos_code(vs.get("pos")) == "v"
                            ]
                            is_passive_frame = len(tokens) >= 3 and tokens[0] in ("be", "feel", "get") and tokens[1].endswith("ed")
                            other_tokens = [
                                tok for tok in tokens
                                if tok != v_host and not (is_passive_frame and tok in ("be", "feel", "get", tokens[1]))
                            ]
                            other_content_toks = [
                                tok for tok in other_tokens
                                if len(tok) >= 3 and tok not in ("the", "for", "with", "from", "into", "onto", "upon", "about", "over", "under")
                            ]
                            for cand_vi in cand_v_indices:
                                v_sense = v_senses_all[cand_vi]
                                v_defn = cls.format_ldoce_definition(v_sense) or str(v_sense.get("definition") or "").strip()
                                if not v_defn:
                                    continue
                                v_pats = v_sense.get("patterns") or []
                                if any("hospital" in str(p).lower() for p in v_pats) and context_sentence and not any(hw in context_sentence.lower() for hw in ("hospital", "patient", "ward", "doctor", "ill", "sick", "injured", "emergency")):
                                    continue
                                if is_passive_frame and v_pats and all(
                                    re.search(rf"\b{re.escape(tokens[-1])}\s+(?:somebody|sb|doing)\b", str(p).lower())
                                    for p in v_pats if tokens[-1] in str(p).lower()
                                ):
                                    continue
                                norm_v_pats = [cls._entry_text_normalizer(p) for p in v_pats]
                                matches_pat = bool(other_tokens) and (
                                    any(
                                        all(re.search(r'(?<!\w)' + re.escape(tok) + r'(?!\w)', np) for tok in other_tokens)
                                        for np in norm_v_pats
                                    )
                                    or (
                                        bool(other_content_toks)
                                        and all(
                                            any(re.search(r'(?<!\w)' + re.escape(tok) + r'(?!\w)', np) for np in norm_v_pats)
                                            for tok in other_tokens if tok not in ("a", "an", "the")
                                        )
                                    )
                                )
                                v_ex = cls._unit_example(key, list(v_sense.get("examples") or []))
                                if not v_ex and matches_pat:
                                    v_ex = cls._relaxed_unit_example(
                                        key, cls._arbitrated_example_pool(v_host, v_sense, key))
                                if (matches_pat or v_ex) and (v_ex or not cls._unit_example(key, list(sense.get("examples") or []))):
                                    v_defn = cls._compose_collocation_definition(key, expr_type, v_host, v_sense, v_defn, context_sentence)
                                    if any(w in key.lower() for w in ("push up", "push (sb/sth) up", "push sth up")) and "increase or decrease" in v_defn.lower():
                                        v_defn = re.sub(r"(?i)\bto increase or decrease\b", "to make something higher or larger; to increase", v_defn)
                                    return (v_defn, v_ex, "sense_pattern")

                sense_pool: List[str] = []
                pb_matched_ex: List[str] = []
                key_toks = key.split()
                is_be_frame = len(key_toks) == 3 and key_toks[0] == "be"
                for pb in sense.get("pattern_bundles", []):
                    if isinstance(pb, dict) and pb.get("example"):
                        pb_pat = cls._entry_text_normalizer(pb.get("pattern") or "")
                        pb_words = pb_pat.split()
                        if all(tok in pb_words for tok in key_toks):
                            pb_matched_ex.append(str(pb["example"]).strip())
                        elif is_be_frame and key_toks[-1] in pb_words and (key_toks[1] in pb_words or (_host and _host in pb_words)):
                            pb_matched_ex.append(str(pb["example"]).strip())
                        elif any(tok in pb_pat for tok in key_toks):
                            sense_pool.append(str(pb["example"]).strip())
                sense_pool += [str(e) for e in (sense.get("examples") or []) if e]
                found_ex = pb_matched_ex[0] if pb_matched_ex else cls._unit_example(key, sense_pool)
                if not found_ex:
                    pool = list(sense_pool)
                    for _h, row in cls._unit_phrase_rows(key):
                        for e in (row.get("examples") or []):
                            if e:
                                pool.append(str(e))
                        rw = (row.get("resolved_word") or row.get("target_word")) if isinstance(row, dict) else None
                        rs_idx = row.get("resolved_sense_idx") if isinstance(row, dict) else None
                        if rw and rs_idx is not None:
                            r_ent = cls.get_ldoce_entry(str(rw))
                            if r_ent and 0 <= int(rs_idx) < len(r_ent.get("senses") or []):
                                for e in (r_ent["senses"][int(rs_idx)].get("examples") or []):
                                    if e:
                                        pool.append(str(e))
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
                    if not found_ex and len(key_toks) == 3 and key_toks[0] in ("do", "make", "take", "give", "show", "have", "play", "pay", "put", "keep", "get"):
                        for vn_key in (f"{key_toks[0]} {key_toks[2]}", f"{key_toks[0]} {cls._surface_lemma(key_toks[2]) or key_toks[2]}"):
                            found_ex = cls._unit_example(vn_key, pool)
                            if found_ex:
                                break
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
                    mock_entry = {"word": key, "senses": cand_senses, "_is_synthetic_pool": True}
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
    @contextmanager
    def wsd_log_session(cls, unit_name: str = "unit"):
        """
        Context manager that aggregates all Top-K LLM WSD arbitration calls within a unit
        into a single consolidated log file (`logs/<timestamp>_extract_wsd_<unit>_TEXT_MODE.log`).
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

        letters = [chr(ord("A") + i) for i in range(min(len(candidates), 20))]
        unique_poses = {
            str(c.get("pos") or "").strip()
            for c in candidates[:len(letters)]
            if str(c.get("pos") or "").strip()
        }
        shared_pos = next(iter(unique_poses)) if len(unique_poses) == 1 else ""
        options_text = []
        cand_sig_parts = []
        for idx, cand in enumerate(candidates[:len(letters)]):
            pos_str = f"({cand.get('pos', '')}) " if (cand.get("pos") and not shared_pos) else ""
            raw_s = cand.get("raw_sense") if isinstance(cand.get("raw_sense"), dict) else {}
            defn_str = cand.get('definition', '').strip()
            signpost_raw = str(cand.get("signpost") or raw_s.get("signpost") or "").strip()
            signpost_str = (
                f"[{signpost_raw.upper()}] "
                if signpost_raw and f"[{signpost_raw.lower()}]" not in defn_str.lower()
                else ""
            )
            gram_raw = str(raw_s.get("grammar") or "").strip()
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
                        "disapproving",
                        "approving",
                        "figurative",
                        "technical",
                        "literary",
                        "humorous",
                    )
                )
                and gram_raw.lower() not in defn_str.lower()
                else ""
            )
            pats = list(cand.get('patterns') or ([cand['pattern']] if cand.get('pattern') else []))
            if not pats and raw_s.get("collocations"):
                for col_item in (raw_s.get("collocations") or [])[:2]:
                    col_p = col_item.get("pattern") if isinstance(col_item, dict) else str(col_item or "")
                    if col_p:
                        pats.append(str(col_p))
            pat_str = f" [Pattern: {', '.join(str(p) for p in pats[:2] if p)}]" if pats else ""
            exs = cand.get('examples') or []
            ex_str = f" (e.g. \"{str(exs[0]).strip()}\")" if exs and str(exs[0]).strip() else ""
            options_text.append(f"{letters[idx]}. {pos_str}{signpost_str}{gram_str}{defn_str}{pat_str}{ex_str}")
            cand_sig_parts.append(f"{cand.get('pos', '')}:{signpost_str}{gram_str}{defn_str}:{pat_str}")

        ctx_clean = str(discourse_context or "").strip()
        pos_header = f" ({shared_pos})" if shared_pos else ""
        if ctx_clean and ">>" in ctx_clean and "<<" in ctx_clean:
            context_section = (
                f"Select the single best dictionary sense for '{item_name}'{pos_header}.\n\n"
                f"Passage Context (target sentence in >>...<<):\n"
                f"\"{ctx_clean}\"\n\n"
            )
        else:
            ctx_block = f"Surrounding Context:\n\"{ctx_clean}\"\n\n" if (ctx_clean and ctx_clean != sentence.strip()) else ""
            context_section = (
                f"Select the single best dictionary sense for '{item_name}'{pos_header} as used in this sentence:\n"
                f"\"{sentence}\"\n\n"
                f"{ctx_block}"
            )

        prompt = (
            f"{context_section}"
            f"Candidate Senses:\n" + "\n".join(options_text) + "\n\n"
            f"Instructions:\n"
            f"- Match the candidate whose definition, [SIGNPOST], and [Pattern: ...] best fit how '{item_name}' is used in the target sentence.\n"
            f"- Respect selectional restrictions (e.g., 'if a [subject] ...', animate vs. inanimate arguments) and register/domain tags ([formal], [informal], [spoken], [technical], [disapproving]).\n"
            f"- Prefer core general senses over narrow technical, legal, medical, or [spoken]/[informal] senses unless the context clearly requires them.\n"
            f"Reply with ONLY the single capital letter ({letters[0]}-{letters[-1]}):"
        )

        cache_raw = f"{active_model}|v4|{item_name.strip().lower()}|{sentence.strip()}|{ctx_clean}|{'||'.join(cand_sig_parts)}"
        cache_key = hashlib.sha1(cache_raw.encode("utf-8")).hexdigest()

        active_session = getattr(cls._wsd_session_local, "session", None)
        if cache_key in cls._wsd_llm_cache:
            cached_idx, cached_raw = cls._wsd_llm_cache[cache_key]
            if active_session is not None:
                active_session["model"] = active_session.get("model") or active_model
                active_session["records"].append({
                    "item_name": item_name,
                    "sentence": sentence,
                    "discourse_context": ctx_clean,
                    "candidates": candidates,
                    "options_text": options_text,
                    "prompt": prompt,
                    "chosen_idx": cached_idx,
                    "raw_reply": cached_raw,
                    "error": None,
                    "duration": 0.0,
                    "cached": True,
                })
            return cached_idx

        chosen_idx: Optional[int] = None
        raw = ""
        err_msg = None
        call_dur = 0.0
        t0 = datetime.datetime.now()
        try:
            from librarian.llm import llm, LLMError
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
                    m = re.search(r'\b(?:OPTION\s+)?([A-T])\b', raw_upper)
                    if m and m.group(1) in letters:
                        parsed_idx = letters.index(m.group(1))
                    else:
                        for l in letters:
                            if raw_upper.startswith(l):
                                parsed_idx = letters.index(l)
                                break

            if parsed_idx is None:
                raise LLMError(
                    f"WSD LLM arbitration returned unparseable option '{raw[:80]}' for '{item_name}' (valid options: {', '.join(letters)})"
                )
            chosen_idx = parsed_idx
        except Exception as exc:
            call_dur = (datetime.datetime.now() - t0).total_seconds()
            err_msg = str(exc)

        if err_msg is None and chosen_idx is not None:
            cls._wsd_llm_cache[cache_key] = (chosen_idx, raw)
        if active_session is not None:
            active_session["model"] = active_session.get("model") or active_model
            active_session["records"].append({
                "item_name": item_name,
                "sentence": sentence,
                "discourse_context": ctx_clean,
                "candidates": candidates,
                "options_text": options_text,
                "prompt": prompt,
                "chosen_idx": chosen_idx,
                "raw_reply": raw,
                "error": err_msg,
                "duration": call_dur,
                "cached": False,
            })
        if err_msg is not None or chosen_idx is None:
            logger.warning(
                f"WSD LLM arbitration rejected/failed for '{item_name}' (skipping candidate): {err_msg or 'No valid option selected'}"
            )
            return -1
        return chosen_idx
