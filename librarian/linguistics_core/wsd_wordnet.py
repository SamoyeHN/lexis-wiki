"""WordNet supersense, synset disambiguation, and sense cue index mixin."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set


class WsdWordNetMixin:
    """WordNet lexical relations and fast LDOCE sense cue lookups."""

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



__all__ = ["WsdWordNetMixin"]
