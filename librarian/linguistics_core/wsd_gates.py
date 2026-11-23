"""WSD Part-of-Speech and syntactic frame gate mixin."""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Set


class WsdGatesMixin:
    """POS canonicalization, short codes, pattern frame matching, and slash expansion."""

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
        "det": "det", "pron": "pron", "excl": "excl", "interj": "excl",
    }

    _CANONICAL_POS_MAP = {
        "n": "noun", "noun": "noun", "noun count": "noun", "noun mass": "noun", "number": "noun",
        "v": "verb", "verb": "verb", "modal verb": "verb",
        "a": "adjective", "s": "adjective", "adj": "adjective", "adjective": "adjective",
        "r": "adverb", "adv": "adverb", "adverb": "adverb",
        "prep": "preposition", "preposition": "preposition",
        "conj": "conjunction", "conjunction": "conjunction",
        "pron": "pronoun", "pronoun": "pronoun",
        "det": "determiner", "determiner": "determiner", "predeterminer": "determiner", "article": "determiner",
        "excl": "interjection", "interj": "interjection", "interjection": "interjection", "exclamation": "interjection",
        "phrasal verb": "phrasal verb", "phrasal_verb": "phrasal verb",
        "collocation": "collocation",
        "set phrase": "set phrase", "set_phrase": "set phrase",
        "idiom": "idiom",
        "phrase": "phrase",
    }

    @classmethod
    def normalize_pos(cls, label: Optional[str]) -> str:
        """Canonicalize any POS tag or short code ('adj' -> 'adjective', 'prep' -> 'preposition',
        'n' -> 'noun', 'v' -> 'verb', etc.) into the project-wide full POS label."""
        raw = re.sub(r"\s+", " ", re.sub(r"\[.*?\]", " ", (label or ""))).strip().lower()
        if not raw:
            return ""
        if "/" in raw and raw not in cls._CANONICAL_POS_MAP:
            first_part = raw.split("/")[0].strip()
            if first_part in cls._CANONICAL_POS_MAP:
                return cls._CANONICAL_POS_MAP[first_part]
        return cls._CANONICAL_POS_MAP.get(raw, raw)

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
        """Return all normalized POS codes declared on a sense (`pos` + `all_poses` + `licensed_poses`)."""
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
        for p in (sense.get("licensed_poses") or []):
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



__all__ = ["WsdGatesMixin"]
