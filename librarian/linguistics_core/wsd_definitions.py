"""LDOCE and PedLex definition formatting and boilerplate detection mixin."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


class WsdDefinitionsMixin:
    """Definition formatting, boilerplate detection, and authentic example extraction."""

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



__all__ = ["WsdDefinitionsMixin"]
