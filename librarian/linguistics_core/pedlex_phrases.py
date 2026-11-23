"""
PedLex multi-word phrase index, attestation check, candidate retrieval, and phrasal frame fitting.
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
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Set, Tuple, Union

from .pedlex_patterns import PedlexPatternsMixin

logger = logging.getLogger("librarian.linguistics")


class PedlexPhrasesMixin:
    _PHRASE_SLOT_WORDS = PedlexPatternsMixin._PHRASE_SLOT_WORDS
    _PHRASAL_PARTICLES = PedlexPatternsMixin._PHRASAL_PARTICLES
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
    _PHRASE_TIERS = (
        "phrasal_verb", "phrase", "phrase_row", "pattern", "sense_unit",
        "collocation", "colloc_phrase", "cross_collocation", "crossref_phrase", "example"
    )
    _PHRASE_STRONG_TIERS = frozenset({
        "phrasal_verb", "phrase", "phrase_row", "pattern", "sense_unit", "collocation", "colloc_phrase"
    })

    @classmethod
    def _entry_text_normalizer(cls, text: Any) -> str:
        """PedLex's own strings, flattened so a frame can be matched as a token sequence:
        brackets, the separable-particle arrow, alternative slashes and '...' holes all
        become spaces ('keep something <-> up' -> 'keep something up')."""
        if not isinstance(text, str):
            return ""
        flat = text.lower()
        # Normalize 'to do something/sth' so 'do' as an action placeholder does not force literal 'do',
        # while keeping the infinitive/preposition 'to' intact.
        flat = re.sub(r"\(\s*to\s+do(?:\s+(?:something|somebody|someone|sth|sb))?\s*\)", " ", flat)
        flat = re.sub(r"\bto\s+do\s+(?:something|somebody|someone|sth|sb)\b", "to something", flat)
        flat = re.sub(r"[()\[\]{}|]+", " ", flat)
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
        is_prep_noun_pair = (
            len(tokens) == 2
            and tokens[0] in ("in", "on", "at", "by", "for", "out", "under", "without", "beyond", "within")
        )
        for t_idx, token in enumerate(tokens):
            if is_prep_noun_pair and t_idx == 1:
                forms = {token}
            else:
                forms = {f for f in cls.inflected_forms(token) if f}
            if token == "a":
                forms |= {"a", "an"}
            if not forms:
                return ""
            # A mined surface form has to reach the form PedLex wrote: 'kept in touch' is
            # stated by the entry as 'keep in touch', so the token's own lemma is inflected
            # too.  Only a real word gets that treatment, so a typo cannot lemmatize its way
            # into a frame it was never part of.
            lemma = cls._surface_lemma(token) if (not (is_prep_noun_pair and t_idx == 1) and cls.is_attested_form(token)) else None
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
            # Particles and gerunds (-ing) are strictly excluded so a phrasal verb particle or
            # gerund complement (e.g. 'making' in 'in making advances') cannot masquerade as an intervening modifier.
            mod_word = rf"(?!(?:{particles})\b)(?![a-z]+ing\b)[a-z]{{2,15}}"
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
        for item in entry.get("crossrefs") or []:
            add("phrase", item)
        for sense in entry.get("senses") or []:
            for value in sense.get("units") or []:
                add("phrase", value)
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

    _phrase_hits_cache: Dict[Tuple[str, str, bool], List[Tuple[str, str]]] = {}

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
        cache_key = (
            (phrase or "").strip().lower(),
            (headword or "").strip().lower(),
            bool(include_examples),
        )
        cached = cls._phrase_hits_cache.get(cache_key)
        if cached is not None:
            return list(cached)
        tokens = cls._phrase_tokens(phrase)
        if len(tokens) < 2:
            cls._phrase_hits_cache[cache_key] = []
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
        cls._phrase_hits_cache[cache_key] = list(hits)
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
    def _ensure_in_memory_phrase_index(cls) -> bool:
        """
        Loads the entire ldoce_phrase_index (~100,952 rows) into process memory in a single
        pass (~170ms) and builds four O(1) lookup structures:
        - _phrase_by_key_cache: {phrase_key: [(host_word, sense_idx, tier, raw_phrase), ...]}
        - _phrase_by_raw_cache: {raw_phrase_lower: [(host_word, sense_idx, tier, raw_phrase), ...]}
        - _PHRASE_INDEX_CACHE: {phrase_key: best_tier}
        - _PHRASE_SYNONYMS_CACHE: {phrase_key: {synonym_phrase_keys...}}
        """
        if not cls._ensure_ldoce_conn() or cls._ldoce_conn is None:
            return False
        if (
            cls._phrase_by_key_cache is not None
            and cls._phrase_by_raw_cache is not None
            and cls._PHRASE_INDEX_CACHE is not None
            and cls._PHRASE_SYNONYMS_CACHE is not None
        ):
            return True

        by_key: Dict[str, List[Tuple[str, Optional[int], str, str]]] = {}
        by_raw: Dict[str, List[Tuple[str, Optional[int], str, str]]] = {}
        tier_map: Dict[str, str] = {}
        slash_groups: Dict[str, Set[str]] = {}

        try:
            cur = cls._ldoce_conn.cursor()
            cur.execute("SELECT phrase_key, tier, host_word, sense_idx, raw_phrase FROM ldoce_phrase_index")
            for p_key, tier, host_word, sense_idx, raw_phrase in cur.fetchall():
                item = (host_word, sense_idx, tier, raw_phrase)
                if p_key in by_key:
                    by_key[p_key].append(item)
                else:
                    by_key[p_key] = [item]

                if raw_phrase:
                    r_low = raw_phrase.lower()
                    if r_low in by_raw:
                        by_raw[r_low].append(item)
                    else:
                        by_raw[r_low] = [item]
                    if "/" in raw_phrase:
                        if raw_phrase in slash_groups:
                            slash_groups[raw_phrase].add(p_key)
                        else:
                            slash_groups[raw_phrase] = {p_key}

                if p_key not in tier_map or tier == "phrasal_verb":
                    tier_map[p_key] = tier

            def _tier_rank(row: Tuple[str, Optional[int], str, str]) -> int:
                t = row[2]
                return 1 if t == "sense_unit" else (2 if t == "phrasal_verb" else 3)

            for rows in by_key.values():
                if len(rows) > 1:
                    rows.sort(key=_tier_rank)
            for rows in by_raw.values():
                if len(rows) > 1:
                    rows.sort(key=_tier_rank)

            syn_map: Dict[str, Set[str]] = {}
            for keys in slash_groups.values():
                if len(keys) > 1:
                    for k in keys:
                        syn_map.setdefault(k, set()).update(keys)
        except Exception:
            return False

        cls._phrase_by_key_cache = by_key
        cls._phrase_by_raw_cache = by_raw
        cls._PHRASE_INDEX_CACHE = tier_map
        cls._PHRASE_SYNONYMS_CACHE = syn_map
        return True

    @classmethod
    def lookup_ldoce_phrase(
        cls,
        phrase: str,
        headword: Optional[str] = None
    ) -> Optional[Tuple[str, Optional[int], str, str]]:
        """
        Dual-track progressive phrase lookup via in-memory ldoce_phrase_index:
        1. Exact literal match (respecting apostrophes/contractions, e.g. "don't be a stranger")
        2. Canonical slot/token match (via _unit_tokens, e.g. 'hello stranger')
        3. Lemmatized verb fallback
        Returns (host_word, sense_idx, tier, raw_phrase) or None.
        """
        if not phrase or not cls._ensure_in_memory_phrase_index():
            return None

        raw = str(phrase).strip()
        if not raw:
            return None

        by_raw = cls._phrase_by_raw_cache or {}
        by_key = cls._phrase_by_key_cache or {}

        # Track 1: Exact literal match (preserving contraction apostrophes and punctuation stripped)
        exact_clean = raw.lower().replace("\u2019", "'").replace("\u2018", "'")
        exact_clean = re.sub(r"[!?,.]+$", "", exact_clean).strip()
        if exact_clean:
            r1 = by_raw.get(exact_clean)
            r2 = by_raw.get(exact_clean + "!")
            if r1 and r2:
                combined = sorted(
                    r1 + r2,
                    key=lambda r: 1 if r[2] == "sense_unit" else (2 if r[2] == "phrasal_verb" else 3),
                )
                return combined[0]
            if r1:
                return r1[0]
            if r2:
                return r2[0]

        # Track 2: Canonical slot/token match via _unit_tokens
        canon_key = " ".join(cls._unit_tokens(raw))
        if canon_key:
            rows = by_key.get(canon_key)
            if rows:
                return rows[0]

        # Track 2b: Possessive slot normalization ('a roof over one\'s head' / 'a roof over their heads' -> 'a roof over your head')
        poss_norm = re.sub(
            r"\b(?:\[one's\]|one's|ones|somebody's|someone's|sb's|my|his|her|their|our|its)\b",
            "your",
            raw.lower().replace("\u2019", "'").replace("\u2018", "'"),
        )
        poss_key = " ".join(cls._unit_tokens(poss_norm))
        if poss_key and poss_key != canon_key:
            rows = by_key.get(poss_key)
            if rows:
                return rows[0]
            p_toks = poss_key.split()
            if len(p_toks) >= 3:
                last_lem = cls._surface_lemma(p_toks[-1]) or cls.lemmatize_headword(p_toks[-1])
                if last_lem and last_lem != p_toks[-1]:
                    sing_key = " ".join(p_toks[:-1] + [last_lem])
                    rows = by_key.get(sing_key)
                    if rows:
                        return rows[0]

        # Track 3: Lemmatized headword/verb fallback (e.g. "banked up" -> "bank up")
        words = raw.split()
        if words:
            lemmatized_head = cls._surface_lemma(words[0]) or cls.lemmatize_headword(words[0])
            if lemmatized_head and lemmatized_head != words[0]:
                lemmatized_phrase = " ".join([lemmatized_head] + words[1:])
                lemmatized_key = " ".join(cls._unit_tokens(lemmatized_phrase))
                if lemmatized_key and lemmatized_key != canon_key:
                    rows = by_key.get(lemmatized_key)
                    if rows:
                        return rows[0]

        # Track 4: Orthographic variant fallback (e.g. "in favor of" -> "in favour of", "enroll in" -> "enrol in")
        for ortho_p in cls.orthographic_variants(raw)[1:]:
            ortho_key = " ".join(cls._unit_tokens(ortho_p))
            if ortho_key and ortho_key != canon_key:
                rows = by_key.get(ortho_key)
                if rows:
                    return rows[0]

        return None

    @classmethod
    def lookup_ldoce_phrase_candidates(
        cls,
        phrase: str,
        headword: Optional[str] = None
    ) -> List[Tuple[str, Optional[int], str, str]]:
        """
        Retrieves all attested candidate matches from in-memory ldoce_phrase_index without LIMIT 1 truncation.
        Returns list of (host_word, sense_idx, tier, raw_phrase) tuples.
        """
        if not phrase or not cls._ensure_in_memory_phrase_index():
            return []

        raw = str(phrase).strip()
        if not raw:
            return []

        by_raw = cls._phrase_by_raw_cache or {}
        by_key = cls._phrase_by_key_cache or {}
        results: List[Tuple[str, Optional[int], str, str]] = []
        seen = set()

        # Track 1: Exact literal match
        exact_clean = raw.lower().replace("\u2019", "'").replace("\u2018", "'")
        exact_clean = re.sub(r"[!?,.]+$", "", exact_clean).strip()
        if exact_clean:
            r1 = by_raw.get(exact_clean, [])
            r2 = by_raw.get(exact_clean + "!", [])
            rows = (
                sorted(
                    r1 + r2,
                    key=lambda r: 1 if r[2] == "sense_unit" else (2 if r[2] == "phrasal_verb" else 3),
                )
                if (r1 and r2)
                else (r1 or r2)
            )
            for r in rows:
                sig = (r[0], r[1], r[2])
                if sig not in seen:
                    seen.add(sig)
                    results.append(r)

        # Track 2: Canonical slot/token match via _unit_tokens
        canon_key = " ".join(cls._unit_tokens(raw))
        if canon_key:
            for r in by_key.get(canon_key, ()):
                sig = (r[0], r[1], r[2])
                if sig not in seen:
                    seen.add(sig)
                    results.append(r)

        # Track 3: Lemmatized headword/verb fallback
        words = raw.split()
        if words:
            lemmatized_head = cls._surface_lemma(words[0]) or cls.lemmatize_headword(words[0])
            if lemmatized_head and lemmatized_head != words[0]:
                lemmatized_phrase = " ".join([lemmatized_head] + words[1:])
                lemmatized_key = " ".join(cls._unit_tokens(lemmatized_phrase))
                if lemmatized_key and lemmatized_key != canon_key:
                    for r in by_key.get(lemmatized_key, ()):
                        sig = (r[0], r[1], r[2])
                        if sig not in seen:
                            seen.add(sig)
                            results.append(r)

        # Track 4: Orthographic variant fallback
        for ortho_p in cls.orthographic_variants(raw)[1:]:
            ortho_key = " ".join(cls._unit_tokens(ortho_p))
            if ortho_key and ortho_key != canon_key:
                for r in by_key.get(ortho_key, ()):
                    sig = (r[0], r[1], r[2])
                    if sig not in seen:
                        seen.add(sig)
                        results.append(r)

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

