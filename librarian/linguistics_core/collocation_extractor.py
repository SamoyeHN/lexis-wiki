"""
Collocation extraction from PedLex (LDOCE 6th Edition).
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


class CollocationExtractorMixin:

    @classmethod
    def get_pedlex_headwords(cls) -> List[str]:
        """Returns all distinct headwords from the pedlex (ldoce) table."""
        if cls._ldoce_conn is None:
            cls.get_ldoce_entry("the")
        conn = cls._ldoce_conn
        if conn is not None:
            try:
                c = conn.cursor()
                c.execute("SELECT DISTINCT word FROM ldoce")
                return [r[0] for r in c.fetchall()]
            except Exception:
                pass
        return []

    @classmethod
    def get_pedlex_collocations(
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
        return cls.get_ldoce_collocations(word=word, pos=pos, context_sentence=context_sentence)

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
                    return cls.get_ldoce_collocations(key)
                def __contains__(self, key: object) -> bool:
                    return bool(cls.get_ldoce_entry(str(key)))
                def get(self, key: Any, default: Any = None) -> Any:
                    res = cls.get_ldoce_collocations(str(key))
                    return res if res else (default if default is not None else {})
                def keys(self) -> List[str]:
                    return cls.get_pedlex_headwords()

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

        # 1. Prepositions from patterns, senses, phrasal_verbs, and units
        norm_pos = str(pos or "").strip().lower()
        if norm_pos.startswith("v"):
            norm_pos = "verb"
        elif norm_pos.startswith("n"):
            norm_pos = "noun"
        elif norm_pos.startswith("adj") or norm_pos == "a":
            norm_pos = "adjective"
        elif norm_pos.startswith("adv") or norm_pos == "r":
            norm_pos = "adverb"

        raw_pats = []
        for s in d.get("senses", []):
            if norm_pos in {"verb", "noun", "adjective", "adverb"}:
                s_pos = str(s.get("pos") or d.get("pos") or "").strip().lower()
                if s_pos.startswith("v"):
                    s_pos = "verb"
                elif s_pos.startswith("n"):
                    s_pos = "noun"
                elif s_pos.startswith("adj") or s_pos == "a":
                    s_pos = "adjective"
                elif s_pos.startswith("adv") or s_pos == "r":
                    s_pos = "adverb"
                if s_pos in {"verb", "noun", "adjective", "adverb"} and s_pos != norm_pos:
                    continue
            raw_pats.extend(s.get("patterns", []))
        if norm_pos in ("", "verb"):
            for pv in d.get("phrasal_verbs", []):
                if isinstance(pv, dict) and pv.get("phrase"):
                    raw_pats.append(pv["phrase"])
                for pv_s in (pv.get("senses") or []) if isinstance(pv, dict) else []:
                    if isinstance(pv_s, dict):
                        raw_pats.extend(pv_s.get("patterns", []))
        for u in d.get("units", []):
            if isinstance(u, dict) and u.get("phrase"):
                raw_pats.append(u["phrase"])

        for pat in raw_pats:
            pat_l_raw = str(pat).lower()
            if norm_pos == "adjective" and re.search(rf"\bit\s+is\s+{re.escape(w_clean)}\b.*\bto\b", pat_l_raw):
                continue
            pat_clean = re.sub(r"\bto\s+[a-z]+(?:\s+something|\s+somebody|\s+sb|\s+sth)?\b", "", pat_l_raw)
            pat_clean = re.sub(r"^(?:be\s+[a-z]+(?:ed|en)|be\s+pleased|be\s+likely)\s+to\b", "", pat_clean)
            for p in ("of", "to", "for", "with", "about", "from", "in", "on", "at", "into", "against", "over", "towards", "under", "upon"):
                if norm_pos == "adjective":
                    has_p = bool(
                        re.search(
                            rf"\b{re.escape(w_clean)}(?:\s*/\s*[a-z]+)*\s+(?:[a-z]+\s*/\s*)*{p}\b",
                            pat_clean,
                        )
                    )
                else:
                    has_p = bool(re.search(rf"\b{p}\b", pat_clean))
                if has_p and p not in res["prep"]:
                    res["prep"].append(p)

        # 2. Collocation boxes in PedLex (with strict POS / homograph isolation)
        def _colloc_pos_matches(item: Dict[str, Any]) -> bool:
            if not norm_pos:
                return True
            it_p = str(item.get("pos") or "").strip().lower()
            if not it_p:
                return True
            return cls._pos_code(it_p) == cls._pos_code(norm_pos)

        for it in colls.get("verbs", []):
            if not _colloc_pos_matches(it):
                continue
            c_str = it.get("collocation", "")
            if c_str:
                res["verb_before"].append(c_str)
            if it.get("example"):
                res["examples"].append(it["example"])
        for it in colls.get("adjectives", []):
            if not _colloc_pos_matches(it):
                continue
            c_str = it.get("collocation", "")
            if c_str:
                res["adj"].append(c_str)
            if it.get("example"):
                res["examples"].append(it["example"])
        for it in colls.get("nouns", []):
            if not _colloc_pos_matches(it):
                continue
            c_str = it.get("collocation", "")
            if c_str:
                res["colloc_nouns"].append(c_str)
            if it.get("example"):
                res["examples"].append(it["example"])
        for it in colls.get("phrases", []):
            if not _colloc_pos_matches(it):
                continue
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
            if not _colloc_pos_matches(it):
                continue
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
        Retrieves top authoritative, natural collocations for a word from PedLex.
        Respects target POS to eliminate cross-POS anomalies.
        Synthesizes phrases like 'lay the foundation', 'firm foundation', 'comprehensive guide'.
        """
        if not word:
            return []
        entry = cls.get_pedlex_collocations(word, pos=pos, context_sentence=context_sentence)
        if not entry:
            return []

        results: List[str] = []

        ldoce = cls.get_ldoce_entry(word) or {}
        head = word.strip().lower()
        sense_pos = {(s.get("pos") or "").strip().lower() for s in (ldoce.get("senses") or [])}
        sense_pos.discard("")
        explicit_pos = (pos or "").strip().lower()
        if explicit_pos:
            if explicit_pos.startswith("n"):
                explicit_pos = "noun"
            elif explicit_pos.startswith("v"):
                explicit_pos = "verb"
            elif explicit_pos.startswith("adj") or explicit_pos == "a":
                explicit_pos = "adjective"
            elif explicit_pos.startswith("adv") or explicit_pos == "r":
                explicit_pos = "adverb"
        strict_pos = bool(
            explicit_pos
            and (
                not sense_pos
                or explicit_pos in sense_pos
                or (explicit_pos == "adjective" and "adj" in sense_pos)
            )
        )
        declared = (explicit_pos if strict_pos else (ldoce.get("pos") or explicit_pos or "")).strip().lower()
        if strict_pos:
            has_noun = declared == "noun"
            has_verb = declared == "verb"
            has_adj = declared in ("adjective", "adj")
        else:
            has_noun = "noun" in sense_pos or declared == "noun"
            has_verb = "verb" in sense_pos or declared == "verb"
            has_adj = bool({"adjective", "adj"} & sense_pos) or declared in ("adjective", "adj")
        texts = cls._collocation_texts(ldoce)
        preps = {p.strip().lower() for p in cls.get_ldoce_preps(word, pos=declared if strict_pos else None)}
        verb_only_preps = (
            {p.strip().lower() for p in cls.get_ldoce_preps(word, pos="verb")} - preps
            if (strict_pos and has_noun and "verb" in sense_pos)
            else set()
        )
        noun_only_preps = (
            {p.strip().lower() for p in cls.get_ldoce_preps(word, pos="noun")} - preps
            if (strict_pos and has_verb and "noun" in sense_pos)
            else set()
        )
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

        noun_box = False if (strict_pos and not has_noun) else (
            declared == "noun" or any(
                ends_with_headword(s)
                for bucket in ("adj", "colloc_nouns")
                for s in (entry.get(bucket) or [])
            )
        )
        # Each box item carries the example PedLex printed beside it, which is the entry's
        # own statement of what the pairing means.
        box_examples: dict = {}
        for bucket_items in (ldoce.get("collocations") or {}).values():
            for it in bucket_items or []:
                if strict_pos and it.get("pos"):
                    it_p = str(it.get("pos")).strip().lower()
                    if it_p and cls._pos_code(it_p) != cls._pos_code(declared):
                        continue
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
            text = re.sub(
                r"\s*\b(?:(?:especially|mainly|chiefly|mostly|usually)\s+)?(?:American|British|Australian|Canadian|North American|Scottish|Irish)\s+English\b[,;:\.]*",
                " ",
                text,
                flags=re.IGNORECASE,
            )
            text = re.sub(r"\s*\betc\.?\b", " ", text.strip(), flags=re.IGNORECASE)
            text = re.sub(r"\s+", " ", text).strip(" ,;:")
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
                if mode == "adj" and not has_noun:
                    return None
                if mode == "verb_before" and not (has_noun or has_adj):
                    return None
                if strict_pos and has_verb and not has_noun:
                    if " " not in lowered:
                        return None
                    if mode in ("noun", "phrase") and ends_with_headword(lowered):
                        w_parts = lowered.split()
                        if len(w_parts) != 2 or not (w_parts[0].endswith("ly") or w_parts[0] in ("to", "not", "never", "always", "often")):
                            return None
                    if mode == "phrase" and any(
                        re.search(rf"\b(?:a|an|the|this|that|my|your|his|her|its|our|their)\s+{re.escape(lowered)}\b", ex)
                        for ex in texts
                    ):
                        return None
                    if re.search(
                        rf"\b(?:a|an|the|this|that|some|any|no|my|your|his|her|its|our|their|public|good|firm|solid|on|under|in|of|into|beyond|out of|outside|within|without|have|has|had|take|took|lose|lost|gain|gained|regain|regained|keep|kept|maintain|maintained)\s+(?:[a-z-]+\s+)?{head_alt}\b",
                        lowered,
                    ):
                        return None
                    for m_np in re.finditer(rf"\b{head_alt}\s+([a-z]+)\b", lowered):
                        if m_np.group(1) in noun_only_preps:
                            return None
                if strict_pos and has_noun and not has_verb and mode == "phrase":
                    if re.match(rf"^{head_alt}\s+(?:to\s+be|to\s+[a-z]+|part-time|full-time|late|hard|round\s+the\s+clock)\b", lowered):
                        return None
                    m_vp = re.match(rf"^{head_alt}\s+([a-z]+)\b", lowered)
                    if m_vp and (m_vp.group(1) in verb_only_preps or "/" in (item or "")):
                        return None
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
                if order == "item" and not has_noun:
                    return None
                if order == "headword" and not governed(gap) and not (has_noun or has_adj):
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
                    if strict_pos and (lowered in noun_only_preps or lowered in cls._COLLOCATION_GAP_WORDS):
                        return None
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
        if has_noun or has_adj:
            added = 0
            for v in (entry.get("verb_before") or []):
                if "(" not in v:
                    phrase = render(v, "verb_before")
                    if phrase:
                        results.append(phrase)
                        added += 1
                        if added >= 3:
                            break

        # 2. Adj + Noun: 'firm [word]', 'solid [word]'
        if has_noun:
            added = 0
            for adj in (entry.get("adj") or []):
                phrase = render(adj, "adj")
                if phrase:
                    results.append(phrase)
                    added += 1
                    if added >= 3:
                        break

        # 3. Noun collocations: '[word] guide', 'birth [word]'
        added = 0
        for noun in (entry.get("colloc_nouns") or []):
            phrase = render(noun, "noun")
            if phrase:
                results.append(phrase)
                added += 1
                if added >= 3:
                    break

        # 4. Adv + Verb: '[word] heavily', '[word] directly'
        added = 0
        for adv in (entry.get("verb") or []):
            phrase = render(adv, "adverb")
            if phrase:
                results.append(phrase)
                added += 1
                if added >= 2:
                    break

        # 5. The ADVERB section of a COLLOCATIONS box ('completely wrong', 'listen
        # attentively') and the PHRASES section ('listen to reason', 'Have a listen').
        # (F7 遗留 ①: bare prepositions like 'director of' or 'streak on' are syntactic patterns,
        # not complete phrases, and are excluded from rich phrase outputs).
        added = 0
        for adv in (entry.get("adverb") or []):
            phrase = render(adv, "adverb")
            if phrase:
                results.append(phrase)
                added += 1
                if added >= 3:
                    break
        added = 0
        for phrase_text in (entry.get("noun_after") or []):
            phrase = render(phrase_text, "phrase")
            if phrase:
                results.append(phrase)
                added += 1
                if added >= 3:
                    break

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

