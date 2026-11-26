"""
PedLex governing preposition extraction, phrasal verb cataloging, and typed slot matching.
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

logger = logging.getLogger("librarian.linguistics")


class PedlexPatternsMixin:
    @classmethod
    def get_ldoce_preps(cls, word: str, pos: Optional[str] = None) -> Set[str]:
        """
        Extracts governing prepositions for the given word from PedLex grammar patterns and collocations.
        When `pos` is provided, only inspects senses matching that part of speech so noun prepositions
        (e.g. 'notice of/to') do not leak into verb valency checks ('notice' verb).
        """
        entry = cls.get_ldoce_entry(word)
        if not entry:
            return set()
        norm_pos = str(pos or "").strip().lower()
        if norm_pos.startswith("v"):
            norm_pos = "verb"
        elif norm_pos.startswith("n"):
            norm_pos = "noun"
        elif norm_pos.startswith("adj") or norm_pos == "a":
            norm_pos = "adjective"
        elif norm_pos.startswith("adv") or norm_pos == "r":
            norm_pos = "adverb"

        preps: Set[str] = set()
        std_preps = ("to", "with", "from", "on", "for", "in", "into", "of", "against", "at", "upon", "towards", "toward", "over", "under", "about", "among")
        # 1. Inspect senses grammar patterns (e.g. 'attitude to/towards', 'depend on/upon')
        for s in entry.get("senses", []):
            if norm_pos in {"verb", "noun", "adjective", "adverb"}:
                s_pos = str(s.get("pos") or entry.get("pos") or "").strip().lower()
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
        sense_poses = {
            cls._pos_code(s.get("pos") or entry.get("pos"))
            for s in (entry.get("senses") or [])
            if cls._pos_code(s.get("pos") or entry.get("pos"))
        }
        is_multi_pos_entry = len(sense_poses) > 1
        collocs = entry.get("collocations", {})
        head_variants = cls.orthographic_variants((entry.get("base_word") or word).lower()) or [(entry.get("base_word") or word).lower()]
        head_alt_pat = "|".join(re.escape(hv) for hv in head_variants)
        head_prep_re = re.compile(
            rf"\b(?:{head_alt_pat})(?:s|es|ed|d|ing)?(?:\s+(?:somebody|something|someone|sb|sth|oneself|/|\.\.\.))*\s+"
            rf"({'|'.join(std_preps)})\b"
        )
        if not (norm_pos and is_multi_pos_entry):
            for ph in collocs.get("phrases", []):
                p_text = ph.get("collocation", "").lower() if isinstance(ph, dict) else str(ph).lower()
                p_text = re.sub(r"\bto\s+[a-z]+(?:\s+something|\s+somebody|\s+sb|\s+sth)?\b", "", p_text)
                for m in head_prep_re.finditer(p_text):
                    preps.add(m.group(1))
        if not norm_pos or norm_pos == "verb":
            for pv in entry.get("phrasal_verbs", []):
                if isinstance(pv, dict):
                    for m in head_prep_re.finditer(str(pv.get("phrase") or "").lower()):
                        preps.add(m.group(1))
                    for pv_s in (pv.get("senses") or []):
                        if isinstance(pv_s, dict):
                            for pat in (pv_s.get("patterns") or []):
                                pat_c = re.sub(r"\bto\s+[a-z]+(?:\s+something|\s+somebody|\s+sb|\s+sth)?\b", "", str(pat).lower())
                                for m in head_prep_re.finditer(pat_c):
                                    preps.add(m.group(1))
        if not (norm_pos and is_multi_pos_entry):
            for u in entry.get("units", []):
                if isinstance(u, dict):
                    u_c = re.sub(r"\bto\s+[a-z]+(?:\s+something|\s+somebody|\s+sb|\s+sth)?\b", "", str(u.get("phrase") or "").lower())
                    for m in head_prep_re.finditer(u_c):
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
        "somebody", "something", "someone", "somewhere", "anything", "anyone", "anybody",
        "sb", "sth", "swh", "doing", "one's", "oneself", "myself", "yourself",
        "himself", "herself", "ourselves", "yourselves", "themselves",
        "his", "her", "their", "its", "my", "your", "our", "it",
        "this", "that", "these", "those", "some", "any", "etc",
    })

    @classmethod
    def _strip_lexical_quotes(cls, text: Optional[str]) -> str:
        """
        Strips quotation noise ("" '' “” ‘’) wrapping individual lexical words or phrases
        before spaCy NLP tagging/parsing (e.g. `more like a "lunatic"` -> `more like a lunatic`),
        while strictly preserving internal contraction apostrophes (`don't`, `it's`) and
        plural possessive apostrophes (`students' books`).
        """
        if not text:
            return ""
        s = str(text)
        # Ensure space when a quote is glued between two word characters (e.g. an"upright"message)
        s = re.sub(r'(\w)(["“”])(\w)', r'\1 \2 \3', s)
        # Strip double quotes and directional curly quotes unconditionally
        s = re.sub(r'["“”«»`]+', '', s)
        # Strip opening single quotes (preceded by start-of-string, whitespace, or opening punctuation)
        s = re.sub(r"(?:^|(?<=[\s(\[{<]))['‘’]+(?=\w)", "", s)
        # Strip closing single quotes wrapped around a word that had an opening quote or non-s-plural ending
        # (Preserve genuine plural possessive s' followed by a noun/modifier, e.g. "parents' expectations",
        #  unless the word itself was single-quoted like 'lunatic' or 'intellectual').
        s = re.sub(r"(?<=[a-rt-zA-RT-Z])['’]+(?=$|[\s.,!?;:)\]\}>])", "", s)
        return re.sub(r"\s+", " ", s).strip()

    @classmethod
    def compile_typed_slot_pattern(cls, template: str) -> List[Dict[str, Any]]:
        """
        Compiles a dictionary unit or grammar pattern (e.g. 'be accustomed to (doing) something',
        'in a ... fashion', 'remember doing something', 'be able to do something') into an ordered
        sequence of Anchor tokens and Typed Wildcard Slots (CQL-style pattern representation).
        Can be used both for forward sentence matching and reverse Quiz pattern generation.

        Slot types:
          - ANCHOR: {'type': 'ANCHOR', 'word': str, 'optional': bool}
          - OPT_VBG_OR_NP: {'type': 'OPT_VBG_OR_NP', 'min_tokens': 1, 'max_tokens': 4, 'optional': True}
          - REQ_VBG: {'type': 'REQ_VBG', 'optional': bool}
          - INF_TO_VB: {'type': 'INF_TO_VB', 'optional': bool}
          - INFIX_SLOT: {'type': 'INFIX_SLOT', 'min_tokens': 1, 'max_tokens': 4, 'optional': bool}
          - BOUNDARY_SLOT: {'type': 'BOUNDARY_SLOT', 'role': str, 'optional': True}
        """
        if not template or not isinstance(template, str):
            return []
        raw = template.strip().lower().replace("\u2019", "'").replace("\u2018", "'")
        raw = raw.replace("\u2194", " ").replace("<->", " ").replace("\u2026", "...")
        raw = re.sub(r"\betc\.?\b", "", raw)

        # Normalize composite verbal/nominal slot placeholders into atomic markers before tokenizing
        # 1. (doing) something / (doing) sth / (doing)
        raw = re.sub(
            r"\(\s*doing\s*\)\s*(?:something|somebody|someone|sth|sb)?\.?",
            " __SLOT_OPT_VBG_NP__ ",
            raw,
        )
        # 2. (to do something) / (to do sth) / (to do)
        raw = re.sub(
            r"\(\s*to\s+do(?:\s+(?:something|somebody|someone|sth|sb)\.?)?\s*\)",
            " __SLOT_OPT_INF_TO_VB__ ",
            raw,
        )
        # 3. Unparenthesized 'to do something/sth' or trailing 'to do'
        # Note: in noun/adj valency templates like 'chance to do something', 'do something' is a placeholder
        # for any complement (infinitive OR noun phrase after 'to'), whereas 'to' is the required anchor!
        raw = re.sub(
            r"\bto\s+do(?:\s+(?:something|somebody|someone|sth|sb)\.?)?\b",
            " to __SLOT_NP__ ",
            raw,
        )
        # 4. Unparenthesized 'doing something/sth'
        raw = re.sub(
            r"\bdoing\s+(?:something|somebody|someone|sth|sb)\b\.?",
            " __SLOT_REQ_VBG__ ",
            raw,
        )
        # 5. Bracketed/parenthesized slot placeholders like (sth), [sth], (somebody), (something)
        raw = re.sub(
            r"[\(\[]\s*(?:somebody|something|someone|somewhere|sb|sth|swh)(?:/(?:somebody|something|someone|sb|sth))?\s*[\)\]]",
            " __SLOT_NP__ ",
            raw,
        )
        # 6. Remaining parenthesized optional lexical words: mark each word inside (...) as optional
        def _mark_opt(m: re.Match) -> str:
            inner = m.group(1).strip()
            if not inner:
                return " "
            return " " + " ".join(f"__OPT_{w}__" for w in inner.split()) + " "

        raw = re.sub(r"\(([^()]+)\)", _mark_opt, raw)
        raw = re.sub(r"\[([^\[\]]+)\]", _mark_opt, raw)

        # 7. Possessive placeholders (somebody's, one's, your, his, her, their) -> NP/POSS slot
        raw = re.sub(
            r"\b(?:somebody's|someone's|something's|sb's|sth's|one's|your|his|her|their|my|our)\b",
            " __SLOT_POSS__ ",
            raw,
        )
        # 8. Bare nominal placeholders and ellipsis (...)
        raw = re.sub(r"\.{2,3}", " __SLOT_ELLIPSIS__ ", raw)
        raw = re.sub(
            r"\b(?:somebody|something|someone|somewhere|anything|anyone|anybody|sb|sth|swh|oneself|yourself|myself|himself|herself|themselves)\b\.?",
            " __SLOT_NP__ ",
            raw,
        )

        raw_toks = [t for t in raw.split() if t]
        # Determine first and last non-slot anchor positions to classify BOUNDARY_SLOT vs INFIX_SLOT
        is_slot_tok = [t.startswith("__SLOT_") for t in raw_toks]
        anchor_indices = [i for i, is_s in enumerate(is_slot_tok) if not is_s]
        first_anchor = anchor_indices[0] if anchor_indices else 999
        last_anchor = anchor_indices[-1] if anchor_indices else -1

        compiled: List[Dict[str, Any]] = []
        for idx, tok in enumerate(raw_toks):
            if tok == "__SLOT_OPT_VBG_NP__":
                compiled.append({"type": "OPT_VBG_OR_NP", "min_tokens": 1, "max_tokens": 4, "optional": True})
            elif tok == "__SLOT_OPT_INF_TO_VB__":
                compiled.append({"type": "INF_TO_VB", "optional": True})
            elif tok == "__SLOT_REQ_INF_TO_VB__":
                compiled.append({"type": "INF_TO_VB", "optional": False})
            elif tok == "__SLOT_REQ_VBG__":
                compiled.append({"type": "REQ_VBG", "optional": False})
            elif tok in ("__SLOT_ELLIPSIS__", "__SLOT_NP__", "__SLOT_POSS__"):
                is_infix = (first_anchor < idx < last_anchor)
                if is_infix:
                    # Merge consecutive infix slots
                    if compiled and compiled[-1]["type"] == "INFIX_SLOT":
                        compiled[-1]["max_tokens"] = min(6, compiled[-1]["max_tokens"] + 2)
                    else:
                        compiled.append({
                            "type": "INFIX_SLOT",
                            "slot_kind": "ellipsis" if tok == "__SLOT_ELLIPSIS__" else ("poss" if tok == "__SLOT_POSS__" else "np"),
                            "min_tokens": 1,
                            "max_tokens": 3 if tok == "__SLOT_POSS__" else 4,
                            "optional": False,
                        })
                else:
                    role = "leading" if idx < first_anchor else "trailing"
                    if not (compiled and compiled[-1]["type"] == "BOUNDARY_SLOT" and compiled[-1].get("role") == role):
                        compiled.append({"type": "BOUNDARY_SLOT", "role": role, "optional": True})
            else:
                opt = False
                w = tok
                if w.startswith("__OPT_") and w.endswith("__"):
                    opt = True
                    w = w[6:-2]
                w = re.sub(r"^[^a-z0-9']+|[^a-z0-9']+$", "", w)
                if w:
                    compiled.append({"type": "ANCHOR", "word": w, "optional": opt})
        return compiled

    @classmethod
    def match_typed_slot_pattern(
        cls,
        template: str,
        quote: str,
        doc: Optional[Any] = None,
        headword: Optional[str] = None,
    ) -> bool:
        """
        Matches a dictionary unit/pattern against a sentence using Typed Wildcard Slots.
        - ANCHOR('be') matches any copula form (am/is/are/was/were/been/being) or passive auxiliary.
        - ANCHOR('a'/'an') matches either 'a' or 'an'.
        - INFIX_SLOT('in a ... fashion') requires 1..4 intervening modifier/argument tokens between 'a' and 'fashion'.
        - OPT_VBG_OR_NP('(doing) something') matches a V-ing token or a noun/pronoun complement.
        - REQ_VBG('doing something') requires a V-ing token (e.g. 'using', 'working').
        - INF_TO_VB('to do something') requires 'to' + a verb base/infinitive token.
        """
        if not template or not quote:
            return False

        # Expand slash variants first so each alternative branch is tested cleanly
        branches = cls._slash_variants(template)
        if len(branches) > 1:
            return any(cls.match_typed_slot_pattern(br, quote, doc=doc, headword=headword) for br in branches)

        compiled = cls.compile_typed_slot_pattern(template)
        if not compiled:
            return False

        if doc is None:
            nlp = cls.get_spacy()
            q_clean = cls._strip_lexical_quotes(quote)
            doc = nlp(q_clean) if nlp else None

        if doc is not None:
            word_toks = [t for t in doc if not t.is_punct and not t.is_space]
        else:
            q_clean = cls._strip_lexical_quotes(quote).lower()
            word_toks = re.findall(r"\b[a-z']+\b", q_clean)

        if not word_toks:
            return False

        # Extract core sequence (strip BOUNDARY_SLOTs since leading/trailing sb/sth are open external arguments)
        core = [item for item in compiled if item["type"] != "BOUNDARY_SLOT"]
        if not core:
            return False

        # Require at least 2 core elements (or 2 anchors, or 1 anchor + 1 non-boundary slot)
        # so a single-word pattern like 'along' or 'smart' does not match as a full typed frame.
        anchor_nodes = [item for item in core if item["type"] == "ANCHOR" and not item.get("optional")]
        if len(core) < 2 or len(anchor_nodes) < 1:
            return False
        if len(anchor_nodes) == 1 and all(item["type"] == "ANCHOR" for item in core):
            return False

        # When headword is provided (e.g. 'line' with pattern 'along'), if the headword is absent from
        # the template's anchors, prepend it as a required anchor so valency patterns stay anchored to the headword.
        if headword:
            hw_clean = headword.strip().lower()
            if hw_clean and " " not in hw_clean:
                has_hw_in_core = any(
                    item["type"] == "ANCHOR" and (item["word"] == hw_clean or cls.form_matches(item["word"], hw_clean))
                    for item in core
                )
                if not has_hw_in_core:
                    core = [{"type": "ANCHOR", "word": hw_clean, "optional": False}] + core

        def _tok_text(t: Any) -> str:
            return t.text.lower() if hasattr(t, "text") else str(t).lower()

        def _tok_lemma(t: Any) -> str:
            return t.lemma_.lower() if hasattr(t, "lemma_") else str(t).lower()

        def _tok_tag(t: Any) -> str:
            return getattr(t, "tag_", "")

        def _tok_pos(t: Any) -> str:
            return getattr(t, "pos_", "")

        def _spans_punctuation_boundary(t_start: int, t_end: int) -> bool:
            """True if doc has punctuation (comma, semicolon, period, dash) between word_toks[t_start-1] and word_toks[t_end]."""
            if doc is None or t_start <= 0 or t_end >= len(word_toks):
                return False
            prev_tok = word_toks[t_start - 1]
            next_tok = word_toks[t_end]
            if not hasattr(prev_tok, "i") or not hasattr(next_tok, "i"):
                return False
            for d_idx in range(prev_tok.i + 1, next_tok.i):
                dt = doc[d_idx]
                if dt.is_punct and dt.text not in ("-", "'", "\u2019"):
                    return True
            return False

        def _anchor_matches(t: Any, anchor_w: str) -> bool:
            txt = _tok_text(t)
            lem = _tok_lemma(t)
            if anchor_w in ("a", "an"):
                return txt in ("a", "an")
            if anchor_w == "be":
                return lem == "be" or txt in ("be", "am", "is", "are", "was", "were", "been", "being", "'m", "'s", "'re")
            if anchor_w == "have":
                return lem == "have" or txt in ("have", "has", "had", "having", "'ve", "'d")
            if anchor_w == "get":
                return lem == "get" or txt in ("get", "gets", "got", "gotten", "getting")
            if txt == anchor_w or lem == anchor_w:
                return True
            return cls.form_matches(txt, anchor_w) or cls.form_matches(lem, anchor_w)

        n_toks = len(word_toks)

        def _match_step(c_idx: int, t_idx: int) -> bool:
            if c_idx >= len(core):
                return True
            node = core[c_idx]
            ntype = node["type"]
            is_opt = node.get("optional", False)

            if ntype == "ANCHOR":
                w = node["word"]
                # Allow a single intervening adverb/negator before a predicate/verb/adjective anchor
                # (e.g., 'are already accustomed to', 'was deeply absorbed in')
                if t_idx < n_toks and _anchor_matches(word_toks[t_idx], w):
                    if not (c_idx > 0 and _spans_punctuation_boundary(t_idx, t_idx)):
                        if _match_step(c_idx + 1, t_idx + 1):
                            return True
                if (
                    c_idx > 0
                    and t_idx + 1 < n_toks
                    and not _spans_punctuation_boundary(t_idx, t_idx + 1)
                    and (_tok_pos(word_toks[t_idx]) in ("ADV", "PART") or _tok_text(word_toks[t_idx]) in ("not", "never", "ever", "also", "already", "still", "well", "quite", "very", "more", "most", "so", "too"))
                    and _anchor_matches(word_toks[t_idx + 1], w)
                ):
                    if _match_step(c_idx + 1, t_idx + 2):
                        return True
                # If leading copula 'be' is omitted in a participial/attributive reduction (e.g. 'accustomed to'),
                # allow skipping leading 'be' only when the headword + following preposition still match
                if c_idx == 0 and w == "be" and sum(1 for x in core if x["type"] == "ANCHOR") >= 3:
                    if _match_step(c_idx + 1, t_idx):
                        return True
                if is_opt and _match_step(c_idx + 1, t_idx):
                    return True
                return False

            if ntype == "INFIX_SLOT":
                min_k = node.get("min_tokens", 1)
                max_k = node.get("max_tokens", 4)
                slot_kind = node.get("slot_kind", "np")
                for span_len in range(min_k, max_k + 1):
                    if t_idx + span_len <= n_toks:
                        if _spans_punctuation_boundary(t_idx, t_idx + span_len):
                            continue
                        span_toks = word_toks[t_idx : t_idx + span_len]
                        # A possessive slot ('in your dreams', 'on one's own') requires a possessive determiner/pronoun or genitive 's
                        if slot_kind == "poss":
                            has_poss = any(
                                _tok_tag(st) in ("PRP$", "WP$", "POS")
                                or _tok_text(st) in ("my", "your", "his", "her", "its", "our", "their", "one's", "'s", "whose")
                                or _tok_text(st).endswith("'s")
                                for st in span_toks
                            )
                            if not has_poss:
                                continue
                        # Do not allow prepositions or finite verbs inside a short nominal/modifier infix slot
                        if any(_tok_pos(st) in ("ADP", "SCONJ", "CCONJ") or (_tok_pos(st) in ("VERB", "AUX") and _tok_tag(st) not in ("VBN", "VBG")) for st in span_toks):
                            continue
                        if _match_step(c_idx + 1, t_idx + span_len):
                            return True
                if is_opt and min_k > 0 and _match_step(c_idx + 1, t_idx):
                    return True
                return False

            if ntype == "OPT_VBG_OR_NP":
                # Matches either a VBG verb ('using') or a nominal/pronoun/determiner phrase
                if t_idx < n_toks:
                    tk = word_toks[t_idx]
                    if _tok_tag(tk) == "VBG" or _tok_text(tk).endswith("ing") or _tok_pos(tk) in ("NOUN", "PROPN", "PRON", "DET", "ADJ", "NUM", "ADV"):
                        if _match_step(c_idx + 1, t_idx + 1):
                            return True
                if is_opt and _match_step(c_idx + 1, t_idx):
                    return True
                return False

            if ntype == "REQ_VBG":
                if t_idx < n_toks:
                    tk = word_toks[t_idx]
                    if _tok_tag(tk) == "VBG" or _tok_text(tk).endswith("ing"):
                        if _match_step(c_idx + 1, t_idx + 1):
                            return True
                    # Allow 1 adverb before VBG ('carefully doing')
                    if t_idx + 1 < n_toks and _tok_pos(tk) == "ADV" and (_tok_tag(word_toks[t_idx + 1]) == "VBG" or _tok_text(word_toks[t_idx + 1]).endswith("ing")):
                        if _match_step(c_idx + 1, t_idx + 2):
                            return True
                return False

            if ntype == "INF_TO_VB":
                if t_idx + 1 < n_toks and _tok_text(word_toks[t_idx]) == "to":
                    nxt = word_toks[t_idx + 1]
                    if _tok_tag(nxt) == "VB" or _tok_pos(nxt) in ("VERB", "AUX"):
                        if _match_step(c_idx + 1, t_idx + 2):
                            return True
                    # Split infinitive ('to Boldly go')
                    if t_idx + 2 < n_toks and _tok_pos(nxt) == "ADV" and (_tok_tag(word_toks[t_idx + 2]) == "VB" or _tok_pos(word_toks[t_idx + 2]) in ("VERB", "AUX")):
                        if _match_step(c_idx + 1, t_idx + 3):
                            return True
                if is_opt and _match_step(c_idx + 1, t_idx):
                    return True
                return False

            return False

        return any(_match_step(0, start_i) for start_i in range(n_toks))

