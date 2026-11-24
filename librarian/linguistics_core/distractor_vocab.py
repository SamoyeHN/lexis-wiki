"""Single-word vocabulary distractor generation and zero-collision orchestrator."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from .distractor_vocab_wsd import DistractorVocabWsdMixin
from .distractor_vocab_harvest import DistractorVocabHarvestMixin
from .distractor_vocab_assembly import DistractorVocabAssemblyMixin

logger = logging.getLogger("librarian.linguistics.vocab_distractors")

__all__ = [
    "DistractorVocabAssemblyMixin",
    "DistractorVocabHarvestMixin",
    "DistractorVocabMixin",
    "DistractorVocabWsdMixin",
]


class DistractorVocabMixin(
    DistractorVocabWsdMixin,
    DistractorVocabHarvestMixin,
    DistractorVocabAssemblyMixin,
):

    @classmethod
    def generate_vocab_distractors(
        cls,
        target_word: str,
        pos: str = "noun",
        context_anchor: str = None,
        anchor_type: str = "collocation",
        target_count: int = 3,
        exclude_words: Optional[Set[str]] = None,
        definition: Optional[str] = None,
        quote: Optional[str] = None,
        return_metadata: bool = False,
        passage_level: Optional[str] = None,
    ) -> Union[List[str], Tuple[List[str], Dict[str, str]]]:
        """
        Synthesizes high-discrimination, collision-free distractors for a target vocabulary word,
        grounded in the target's authentic curriculum definition (Sense-Specific Primacy).
        """
        token = cls.set_current_passage_cefr(passage_level) if passage_level else None
        try:
            clean_target = target_word.strip().lower()

            # Closed Grammatical Paradigms Gate:
            # If target word is an indexed function word (connective, clausal conjunction, preposition),
            # retrieve psychometrically calibrated distractors directly from closed paradigms.
            if cls.is_function_word(clean_target):
                f_dists, f_meta = cls.get_function_word_distractors(
                    clean_target,
                    target_count=target_count,
                    exclude_words=exclude_words,
                )
                if len(f_dists) >= target_count:
                    if return_metadata:
                        return f_dists, f_meta
                    return f_dists

            wsd_ctx = cls._prepare_vocab_wsd_and_antonyms(
                clean_target=clean_target,
                pos=pos,
                context_anchor=context_anchor,
                anchor_type=anchor_type,
                target_count=target_count,
                exclude_words=exclude_words,
                definition=definition,
                quote=quote,
            )

            harvest_ctx = cls._harvest_vocab_candidates(wsd_ctx)

            return cls._assemble_vocab_distractors(
                wsd_ctx=wsd_ctx,
                harvest_ctx=harvest_ctx,
                return_metadata=return_metadata,
            )
        finally:
            if token is not None:
                cls.reset_current_passage_cefr(token)

    @classmethod
    def generate_zero_collision_distractors(
        cls,
        target_word: str,
        dependent_prep: Optional[str] = None,
        pos: str = "v",
        count: int = 3,
        context_sentence: Optional[str] = None
    ) -> List[str]:
        """
        Pre-computes distractors mathematically guaranteed to avoid synonym collisions with target.
        When dependent_prep is specified (e.g. 'on' for 'rely'), prioritizes candidates that
        do NOT govern that preposition or clash with it.
        Uses context_sentence and MiniLM disambiguation to anchor to the pedagogical synset
        rather than blind synsets[0].
        """
        wn = cls.get_wordnet()
        target_lower = target_word.lower()
        target_synonyms = cls.get_synonyms(target_lower)
        target_synonyms.add(target_lower)

        antonyms = cls.get_antonyms(target_lower)

        candidates: List[str] = []

        # Antonyms make outstanding, collision-free distractors
        for ant in antonyms:
            if ant not in target_synonyms and " " not in ant and cls.is_cefr_compliant_distractor(ant, target_lower):
                candidates.append(ant)

        # 2. Taxonomy sibling / hypernym search
        target_synsets = cls.get_wordnet_synsets(target_lower, pos=pos)
        if target_synsets:
            primary_synset = cls.disambiguate_wordnet_synset(
                target_lower, pos=pos, context_sentence=context_sentence, candidate_synsets=target_synsets
            ) or target_synsets[0]
            # Get hypernyms (parent categories)
            hypernyms = primary_synset.get_related("hypernym")
            for hyper in hypernyms:
                # Get siblings (hyponyms of the hypernym)
                for sibling in hyper.get_related("hyponym"):
                    for w in sibling.words():
                        lemma = w.lemma().replace("_", " ").lower()
                        if (
                            lemma not in target_synonyms
                            and " " not in lemma
                            and lemma not in candidates
                            and cls.is_cefr_compliant_distractor(lemma, target_lower)
                        ):
                            if pos == "n" and (cls._is_primarily_adj_or_verb(lemma) if hasattr(cls, "_is_primarily_adj_or_verb") else False):
                                continue
                            candidates.append(lemma)

        # 3. Deduplicate and return requested count
        final_distractors = candidates[:count]

        # Fallback if synset taxonomy was sparse
        if len(final_distractors) < count:
            needed = count - len(final_distractors)
            excl = set(target_synonyms) | set(final_distractors) | {target_lower}
            pref_vowel: Optional[bool] = None
            if pos == "n" and not target_lower.endswith("s") and target_lower[:1].isalpha():
                pref_vowel = target_lower[0].lower() in "aeiou"
            fb_cands = cls.get_corpus_fallback_candidates(
                target_word=target_lower,
                pos=pos,
                target_count=needed,
                exclude_words=excl,
                prefer_initial_vowel=pref_vowel,
            )
            for fb in fb_cands:
                if fb not in target_synonyms and fb not in final_distractors:
                    final_distractors.append(fb)
                if len(final_distractors) >= count:
                    break

        return final_distractors
