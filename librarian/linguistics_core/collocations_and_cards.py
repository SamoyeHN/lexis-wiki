"""
Collocation extraction, zero-collision collocational anchor mining, and zero-token vocabulary card enrichment.
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



class CollocationsAndCardsMixin:

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
                    conn = cls._ldoce_conn
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
            pat_clean = re.sub(r"\bto\s+[a-z]+(?:\s+something|\s+somebody|\s+sb|\s+sth)?\b", "", str(pat).lower())
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
        _DEF_STOP = {
            "a", "an", "the", "to", "of", "in", "on", "at", "by", "for", "with", "from",
            "into", "onto", "upon", "about", "as", "or", "and", "that", "which", "who",
            "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
            "do", "does", "did", "can", "could", "will", "would", "way", "used", "especially",
            "someone", "something", "somebody", "things", "person", "people", "your", "you",
            "their", "them", "they", "it", "its", "not", "very", "more", "most", "much", "many"
        }
        raw_a = set(re.findall(r"[a-z]+", (text_a or "").lower()))
        raw_b = set(re.findall(r"[a-z]+", (text_b or "").lower()))
        if not raw_a or not raw_b:
            return 0.0
        ta = raw_a - _DEF_STOP
        tb = raw_b - _DEF_STOP
        if not ta or not tb:
            ta, tb = raw_a, raw_b
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
        raw_w = str(word or "").strip()
        if " " in raw_w:
            return out
        w_clean = re.sub(r"\[.*?\]|\(.*?\)", "", raw_w).strip().lower()
        if not w_clean or " " in w_clean:
            return out

        entry = cls.get_ldoce_entry(w_clean)
        if not entry:
            return out

        out["core_dots"] = str(entry.get("core_dots") or "")
        out["freq_tags"] = " ".join(entry.get("freq_tags") or [])

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
            s = re.sub(r"\b(sth|sb|something|somebody)\s+[A-Z][a-z]+.*$", r"\1", s).strip()
            s = re.sub(r"\bsth\b", "something", s)
            s = re.sub(r"\bsb\b", "somebody", s)
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
            if raw_gram and raw_gram.strip("[]` ").lower() not in ("noun", "verb", "adjective", "adverb", "preposition", "conjunction", "interjection"):
                gram_tag_str = raw_gram.strip("`").strip()
            if p_code == "a" and "not before noun" in gram_tag_str.lower():
                att_texts = [quote] if quote else []
                att_texts.extend(str(ex) for ex in (locked_sense.get("examples") or [])[:3] if ex)
                has_attr_use = False
                nlp_inst = cls.get_spacy()
                for atxt in att_texts:
                    adoc = nlp_inst(atxt)
                    if any(t.lower_ == w_clean and t.dep_ == "amod" and t.head.pos_ in ("NOUN", "PROPN") for t in adoc):
                        has_attr_use = True
                        break
                if has_attr_use:
                    gram_tag_str = re.sub(r"\[?\bnot before noun\b\]?", "", gram_tag_str, flags=re.IGNORECASE).strip(" ,;[]")
            raw_pats_and_units = list(dict.fromkeys(
                list(locked_sense.get("syntactic_patterns") or locked_sense.get("patterns") or [])
                + [
                    u for u in (locked_sense.get("units") or [])
                    if not (p_code == "v" and re.search(rf"\b(?:a|an|the|to a)\s+{re.escape(w_clean)}\b", str(u).lower()))
                ]
            ))
            pats = [
                _clean_pattern_text(p) for p in raw_pats_and_units
                if _clean_pattern_text(p) and _clean_pattern_text(p).lower() != w_clean
            ]
            if pats:
                pattern_parts.append(f"*{pats[0]}*")

        gb_alert = cls.get_ldoce_grammar_alert(w_clean, quote=quote, sense_index=t_s_idx, pos=p_code)
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
        locked_def_kws: Set[str] = set()
        other_sense_kws: Set[str] = set()
        if locked_sense and len(senses) > 1:
            l_def_str = str(definition or cls.format_ldoce_definition(locked_sense) or "").lower()
            locked_def_kws = {w for w in re.findall(r"[a-z]{4,}", l_def_str) if w != w_clean}
            for s_other in senses:
                if s_other is locked_sense:
                    continue
                o_def_str = str(cls.format_ldoce_definition(s_other) or "").lower()
                for ow in re.findall(r"[a-z]{4,}", o_def_str):
                    if ow != w_clean and ow not in locked_def_kws:
                        other_sense_kws.add(ow)

        entry_p_code = cls._pos_code(entry.get("pos"))
        all_poses = [cls._pos_code(p) for p in (entry.get("all_poses") or []) if cls._pos_code(p)]
        is_multi_pos = len(set(all_poses)) > 1
        verb_only_preps: Set[str] = set()
        noun_only_preps: Set[str] = set()
        if is_multi_pos and "v" in all_poses and "n" in all_poses:
            v_preps = set(cls.get_ldoce_preps(w_clean, pos="verb"))
            n_preps = set(cls.get_ldoce_preps(w_clean, pos="noun"))
            verb_only_preps = v_preps - n_preps
            noun_only_preps = n_preps - v_preps

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

        def _add_colloc(cand_str: str, check_cross_sense: bool = False) -> None:
            c_s = _clean_pattern_text(cand_str)
            if not c_s or " " not in c_s or "..." in c_s:
                return
            c_low = c_s.lower()
            if c_low == w_clean or c_low in (f"the {w_clean}", f"a {w_clean}", f"an {w_clean}"):
                return
            if p_code == "v" and re.search(rf"\b(?:a|an|the|to a|under|out of|in|beyond|within|without)\s+{re.escape(w_clean)}\b", c_low):
                return
            c_words = c_low.split()
            if p_code == "n" and len(c_words) == 2 and c_words[1] == w_clean and re.match(r"^[a-z]+ing$", c_words[0]):
                return
            if is_multi_pos and p_code == "n" and "v" in all_poses and c_words and c_words[0] == w_clean:
                # Guard against verb-initial frames leaking onto noun cards (e.g., "study to be...", "study law/business/...", "study for an exam")
                if len(c_words) >= 2 and (
                    c_words[1] == "to"
                    or "/" in c_words[1]
                    or c_words[1] in verb_only_preps
                ):
                    return
            if is_multi_pos and p_code == "v" and "n" in all_poses and c_words and c_words[0] == w_clean:
                if len(c_words) >= 2 and c_words[1] in noun_only_preps:
                    return
            if check_cross_sense and (other_sense_kws or (len(senses) > 1 and t_s_idx is not None and t_s_idx > 0)):
                c_toks = {w for w in re.findall(r"[a-z]{4,}", c_low) if w != w_clean and not cls.form_matches(w, w_clean)}
                if c_toks & other_sense_kws and not (c_toks & locked_def_kws):
                    return
                if len(senses) > 1 and t_s_idx is not None and t_s_idx > 0 and not (c_toks & locked_def_kws):
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
        colloc_box_pos_ok = (
            not entry_p_code
            or not p_code
            or entry_p_code == p_code
            or not is_multi_pos
        )
        if len(colloc_items) < 5:
            raw_collocs = entry.get("collocations")
            if isinstance(raw_collocs, dict):
                allowed_cats = ("phrases", "verbs", "adjectives", "nouns", "adverbs")
                if p_code == "v":
                    allowed_cats = ("adverbs", "phrases", "nouns")
                elif p_code == "n":
                    allowed_cats = ("verbs", "adjectives", "phrases", "nouns")
                elif p_code == "adj":
                    allowed_cats = ("adverbs", "nouns", "phrases")
                elif p_code == "adv":
                    allowed_cats = ("verbs", "adjectives", "phrases")
                for cat_k in allowed_cats:
                    for it_c in (raw_collocs.get(cat_k) or []):
                        if not isinstance(it_c, dict) or it_c.get("source") == "corpus":
                            continue
                        it_p_code = cls._pos_code(str(it_c.get("pos") or ""))
                        if p_code and it_p_code in ("n", "v", "adj", "adv"):
                            if it_p_code != p_code:
                                continue
                        elif not colloc_box_pos_ok:
                            continue
                        it_s_indices = it_c.get("sense_indices")
                        if isinstance(it_s_indices, list) and it_s_indices and t_s_idx is not None:
                            if t_s_idx not in it_s_indices:
                                continue
                            has_direct_sense_link = True
                        else:
                            has_direct_sense_link = False
                        c_val = _clean_pattern_text(str(it_c.get("collocation") or ""))
                        if not c_val or " " not in c_val:
                            continue
                        _add_colloc(c_val, check_cross_sense=not has_direct_sense_link)
                        if len(colloc_items) >= 5:
                            break
                    if len(colloc_items) >= 5:
                        break

        # Tier C: Same-POS/same-homograph sibling senses' patterns and lexical units (only when not conflicting with locked sense)
        if len(colloc_items) < 5 and len(senses) <= 1:
            for s_obj in senses:
                if s_obj is locked_sense or not _sense_pos_matches(s_obj):
                    continue
                for pb in s_obj.get("pattern_bundles") or []:
                    _add_colloc(str(pb.get("pattern") or ""), check_cross_sense=True)
                for pat in s_obj.get("patterns") or []:
                    _add_colloc(str(pat), check_cross_sense=True)
                for u in s_obj.get("units") or []:
                    _add_colloc(str(u), check_cross_sense=True)
                if len(colloc_items) >= 5:
                    break

        # Tier D: Entry-level fixed phrases & cross-referenced dictionary collocations (when单POS or needed)
        is_ed_adj = (p_code == "adj" and w_clean.endswith("ed"))
        if len(colloc_items) < 5 and colloc_box_pos_ok and not is_ed_adj:
            for ph_obj in (entry.get("phrases") or []):
                ph_str = ph_obj.get("phrase") or ph_obj.get("title") if isinstance(ph_obj, dict) else str(ph_obj or "")
                _add_colloc(str(ph_str or ""), check_cross_sense=True)
                if len(colloc_items) >= 5:
                    break
        if len(colloc_items) < 5 and not is_multi_pos and not is_ed_adj:
            for cr_obj in (entry.get("cross_collocations") or []):
                if isinstance(cr_obj, dict):
                    _add_colloc(str(cr_obj.get("collocation") or ""), check_cross_sense=True)
                    if len(colloc_items) >= 5:
                        break

        if colloc_items:
            out["collocations"] = "; ".join(colloc_items[:5])

        # 3. Sense-Aligned THESAURUS Note (exclude target word itself; rank peers by semantic overlap & proximity)
        thes_list = entry.get("thesaurus") or []
        if thes_list:
            def _thes_item_matches_pos(t_obj: Dict[str, Any]) -> bool:
                if not is_multi_pos or not p_code:
                    return True
                t_pos_code = cls._pos_code(str(t_obj.get("pos") or ""))
                if t_pos_code in ("n", "v", "adj", "adv") and t_pos_code != p_code:
                    return False
                tw_low = str(t_obj.get("word") or "").strip().lower()
                tg_low = str(t_obj.get("grammar") or "").lower()
                td_low = re.sub(r"^[:\s]+", "", str(t_obj.get("distinction") or "")).strip().lower()
                has_n_gram = any(k in tg_low for k in ("countable", "uncountable", "singular", "plural", "noun"))
                has_v_gram = any(k in tg_low for k in ("transitive", "intransitive", "phrasal verb"))
                starts_v_def = bool(re.match(r"^(?:to\s+[a-z]+|if\s+you\s+[a-z]+)\b", td_low))
                starts_n_def = bool(
                    re.match(r"^(?:a|an|the|one\s+of\s+the|something\s+that|information\s+about|someone\s+who)\b", td_low)
                )
                is_reflexive_vp = any(r in tw_low for r in ("yourself", "himself", "herself", "oneself", "themselves"))
                if p_code == "v":
                    if has_n_gram or starts_n_def or (not has_v_gram and not starts_v_def):
                        return False
                elif p_code == "n":
                    if has_v_gram or is_reflexive_vp or (starts_v_def and not has_n_gram):
                        return False
                elif p_code in ("adj", "adv"):
                    if has_n_gram or has_v_gram or starts_v_def or is_reflexive_vp:
                        return False
                return True

            locked_def = (definition or (cls.format_ldoce_definition(locked_sense) if locked_sense else "")).strip()
            self_indices = [
                idx_t for idx_t, t in enumerate(thes_list)
                if cls._thesaurus_variant_matches(t.get("word"), w_clean) and _thes_item_matches_pos(t)
            ]
            self_exp = thes_list[self_indices[0]] if self_indices else None
            # Only surface THESAURUS box when its cluster aligns with the locked sense (or entry has 1 POS)
            box_matches_sense = True
            if is_multi_pos and not self_indices:
                # Verify that at least one peer matches the requested POS
                if not any(_thes_item_matches_pos(t) for t in thes_list):
                    box_matches_sense = False
            if locked_def and len(senses) > 1:
                if self_exp:
                    self_dist = re.sub(r"^[:\s]+", "", str(self_exp.get("distinction") or "")).strip()
                    if self_dist and cls._definition_overlap_ratio(self_dist, locked_def) < 0.25:
                        box_matches_sense = False
                thes_blob = " ".join(
                    re.sub(r"^[:\s]+", "", str(t.get("distinction") or "")).strip()
                    for t in thes_list if _thes_item_matches_pos(t)
                ).strip()
                if thes_blob:
                    locked_ov = cls._definition_overlap_ratio(thes_blob, locked_def)
                    if not self_exp and locked_ov < 0.20:
                        box_matches_sense = False
                    elif t_s_idx is not None:
                        for s_i_o, s_other in enumerate(senses):
                            if s_i_o == t_s_idx or not _sense_pos_matches(s_other):
                                continue
                            o_def = (cls.format_ldoce_definition(s_other) or "").strip()
                            if o_def and cls._definition_overlap_ratio(thes_blob, o_def) > locked_ov + 0.08:
                                box_matches_sense = False
                                break
            if box_matches_sense:
                # Reference text combines the locked definition and the target's own thesaurus distinction(s)
                ref_texts = [locked_def] if locked_def else []
                for s_idx in self_indices:
                    s_dist = re.sub(r"^[:\s]+", "", str(thes_list[s_idx].get("distinction") or "")).strip()
                    if s_dist:
                        ref_texts.append(s_dist)
                ref_combined = " ".join(ref_texts)

                ld_low = locked_def.lower()
                is_nonhuman_object_def = (
                    ("for something" in ld_low or "in something" in ld_low or "about something" in ld_low)
                    and not any(k in ld_low for k in ("someone", "somebody", "person", "people", "sexual", "romantic"))
                )

                scored_peers: List[Tuple[float, int, str]] = []
                seen_peer_words: Set[str] = set()
                for idx_t, t in enumerate(thes_list):
                    if not _thes_item_matches_pos(t):
                        continue
                    ew = str(t.get("word") or "").strip()
                    ed = re.sub(r"^[:\s]+", "", str(t.get("distinction") or "")).strip()
                    eg = str(t.get("grammar") or "").strip()
                    er = str(t.get("register") or "").strip()
                    if not ew or not ed:
                        continue
                    # Never list the target headword itself in Thesaurus Note (already defined above)
                    if cls._thesaurus_variant_matches(ew, w_clean) or ew.lower() == w_clean:
                        continue
                    if is_nonhuman_object_def:
                        ed_low = ed.lower()
                        if any(k in ed_low for k in ("sexually attracted", "wife, husband", "boyfriend", "girlfriend", "romantic")):
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
        gb_alert = cls.get_ldoce_grammar_alert(tw, quote=quote or "", sense_index=sense_index, pos=pos or "")
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
        locked_s_idx: Optional[int] = None
        if locked_sense is not None and senses:
            for idx_s, s_cand in enumerate(senses):
                if s_cand is locked_sense or (
                    s_cand.get("definition") == locked_sense.get("definition")
                    and s_cand.get("pos") == locked_sense.get("pos")
                ):
                    locked_s_idx = idx_s
                    break
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
            it_s_indices = it_c.get("sense_indices")
            if isinstance(it_s_indices, list) and it_s_indices and locked_s_idx is not None:
                return 100 if locked_s_idx in it_s_indices else 0
            if not sense_lemmas:
                return 0
            try:
                nlp = cls.get_spacy()
                self_forms = {tw_clean} | {t.lemma_.lower() for t in nlp(tw_clean)}
                c_text = (
                    (it_c.get("collocation") or "")
                    + " "
                    + (it_c.get("gloss") or "")
                    + " "
                    + (it_c.get("example") or "")
                ).lower()
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
                                    if child.dep_ in ("dobj", "obj") and child.is_alpha and len(child.lemma_) >= 3 and child.lemma_.lower() not in cls._ANCHOR_STOPWORDS and child.pos_ in ("NOUN", "PROPN"):
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
                            is_prep_obj = any(
                                n_tok.dep_ == "pobj" and n_tok.head.dep_ == "prep" and (
                                    n_tok.head.head.lemma_.lower() == tw_clean or n_tok.head.head.text.lower().startswith(tw_clean)
                                )
                                for n_tok in doc_ex
                                if n_tok.lemma_.lower() == tok_clean or n_tok.text.lower() == tok_clean
                            )
                            if is_noun_compound or not has_verb_usage or is_prep_obj:
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
