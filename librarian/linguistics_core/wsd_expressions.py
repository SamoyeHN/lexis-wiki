"""Multi-word unit, collocation, and phrasal expression definition mixin."""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("librarian.linguistics")


class WsdExpressionsMixin:
    """Multi-word unit expression cascade, unit senses, and collocation definitions."""

    EXPRESSION_SOURCE_TIERS = ("ldoce_entry", "phrasal_verb_block", "sense_pattern",
                               "phrase_row", "wordnet", "")

    # Quantifier frame regex (e.g. 'a piece of', 'a kind of', 'a sort of', 'a bottle of')
    _QUANTIFIER_OF_RE = re.compile(r'^(?:a|an)\s+([a-z]+)\s+of$', re.IGNORECASE)


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
                            return (host, sense, "sense_pattern")
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

        has_internal_obj_slot = bool(
            re.search(
                r"^[a-z]+\s+(?:\[[^\]]+\]|\([^)]+\)|\.\.\.)\s+[a-z]+",
                str(phrase or "").strip(),
                re.IGNORECASE,
            )
        )
        key = cls._unit_key(clean_phrase)
        is_multiword = len(key.split()) >= 2

        # When expr_type is explicitly a phrasal verb or verbal multiword (or has an internal object slot V + [sth/sb] + Prep),
        # check PedLex's phrasal-verb block alongside base verb senses in a unified candidate pool.
        is_phrasal_v = is_multiword and (
            expr_type in ("phrasal verb", "v", "verb")
            or has_internal_obj_slot
            or (
                expr_type not in ("noun", "idiom", "set phrase")
                and bool(cls._unit_blocks(key))
                and cls.determine_contextual_pos(key.split()[0], context_sentence) == "verb"
            )
            or (
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
                    ctx_low = (context_sentence or "").lower()
                    v_forms_re = "|".join(re.escape(f) for f in (cls.inflected_forms(base_verb) | {base_verb}))
                    pt_re = re.escape(other_tokens[0]) if len(other_tokens) == 1 else ""
                    is_passive_v_prep = bool(
                        pt_re
                        and context_sentence
                        and re.search(
                            rf"\b(?:am|is|are|was|were|be|been|being|get|gets|got)\s+(?:[a-z]+ly\s+)?(?:{v_forms_re})\s+{pt_re}\b",
                            ctx_low,
                        )
                    )
                    # Detect separable direct object between verb and particle/preposition
                    has_sep_dobj_or_dest = False
                    if len(other_tokens) == 1 and context_sentence:
                        if re.search(rf"\b(?:{v_forms_re})\s+(?:the|a|an|his|her|their|our|my|your|its|this|that|these|those|[a-z]+'s)\s+[^.,;!?]{{1,35}}\b{pt_re}\b", ctx_low) or re.search(rf"\b{pt_re}\s+to\s+(?:the|a|an|his|her|their|our|my|your|its)\b", ctx_low):
                            has_sep_dobj_or_dest = True

                    _relational_preps = {
                        "to", "with", "into", "from", "of", "upon", "for", "at", "by", "about", "as", "against"
                    }
                    is_trivalent_prep_frame = bool(
                        len(other_tokens) == 1
                        and other_tokens[0] in _relational_preps
                        and (has_internal_obj_slot or is_passive_v_prep or has_sep_dobj_or_dest)
                    )

                    cand_base_senses = []
                    v_entry = cls.get_ldoce_entry(base_verb)
                    if v_entry and (context_sentence or is_trivalent_prep_frame):
                        ctx_words_low = set(re.findall(r"[a-z]+", ctx_low))
                        _pat_slot_meta = {
                            "something", "somebody", "someone", "sth", "sb", "oneself",
                            "one's", "etc", "to", "for", "with", "from", "in", "on", "at",
                            "by", "of", "into", "out", "up", "down", "off", "away", "back",
                            "over", "through", "around", "round", "about", "along", "ahead",
                            "a", "an", "the", "doing", "do", "be", "being",
                        }
                        for s in v_entry.get("senses", []):
                            matched_exact_pat = False
                            for p in (s.get("patterns") or []):
                                norm_p = cls._entry_text_normalizer(p)
                                p_toks = norm_p.split()
                                if not p_toks or not cls.form_matches(p_toks[0], base_verb):
                                    continue
                                if all(re.search(r'(?<!\w)' + re.escape(tok) + r'(?!\w)', norm_p) for tok in other_tokens):
                                    extra_lex = [
                                        wt for wt in p_toks[1:]
                                        if wt not in other_tokens and not any(sub in _pat_slot_meta for sub in wt.split("/"))
                                        and all(len(sub) >= 3 and sub.isalpha() for sub in wt.split("/"))
                                    ]
                                    if extra_lex and not any(
                                        any(cls.form_matches(cw, alt) for cw in ctx_words_low for alt in wt.split("/"))
                                        for wt in extra_lex
                                    ):
                                        continue
                                    cand_base_senses.append(s)
                                    matched_exact_pat = True
                                    break
                            # When the item/sentence uses a transitive or passive trivalent frame (e.g. 'equate (sth) to (sth)' /
                            # 'is equated to') and the host verb's main transitive sense lists a sister relational preposition
                            # (e.g. 'equate something with something'), include that transitive sense in cand_base_senses.
                            if not matched_exact_pat and is_trivalent_prep_frame:
                                s_gram_l = str(s.get("grammar") or "").lower()
                                s_pats = list(s.get("patterns") or [])
                                has_trans_obj = (
                                    ("transitive" in s_gram_l and "intransitive" not in s_gram_l)
                                    or any(
                                        re.search(
                                            rf"\b{re.escape(base_verb)}\s+(?:something|somebody|someone|sth|sb)\b",
                                            cls._entry_text_normalizer(p),
                                        )
                                        for p in s_pats
                                    )
                                )
                                if has_trans_obj:
                                    s_triv = dict(s)
                                    synth_pat = f"{base_verb} something {other_tokens[0]} something"
                                    s_triv["patterns"] = list(s_pats) + [synth_pat]
                                    cand_base_senses.append(s_triv)

                    # Construct unified candidate pool: base senses + phrasal verb block senses
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
                        if has_sep_dobj_or_dest and any(k in s_def_l for k in ("admit that you were wrong", "something that you said")):
                            continue
                        # Filter out strictly intransitive / no-passive phrasal_verbs senses (e.g. 'equate to sth' [intransitive, no passive]:
                        # 'to be equal to something') when the phrase/context is transitive or passive and a transitive base sense exists.
                        if is_trivalent_prep_frame and cand_base_senses:
                            b_gram_l = (str(s.get("grammar") or "") + " " + " ".join(str(x) for x in (s.get("syntax_codes") or []))).lower()
                            b_pats_norm = [cls._entry_text_normalizer(p) for p in (s.get("patterns") or [])]
                            b_has_internal_obj = any(
                                re.search(rf"\b{re.escape(base_verb)}\s+(?:something|somebody|someone|sth|sb)\s+{re.escape(other_tokens[0])}\b", bp)
                                for bp in b_pats_norm
                            )
                            if not b_has_internal_obj and ("intransitive" in b_gram_l or "no passive" in b_gram_l):
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
                    if not prepared_b_senses and b_senses and not (is_trivalent_prep_frame and cand_base_senses):
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
                                if not ex and is_trivalent_prep_frame:
                                    for cand_ex in (chosen_sense.get("examples") or []):
                                        if cls.text_contains_form(cand_ex, base_verb):
                                            ex = cand_ex
                                            break
                            if (not ex or (not cls.text_contains_form(ex, key) and not (is_trivalent_prep_frame and cls.text_contains_form(ex, base_verb)))) and context_sentence and cls._unit_example(key, [context_sentence]):
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



__all__ = ["WsdExpressionsMixin"]
