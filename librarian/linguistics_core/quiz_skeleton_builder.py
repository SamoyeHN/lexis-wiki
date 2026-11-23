"""
Generative vocabulary quiz skeletons pipeline orchestrator.
Coordinates single-item blueprint stages and executes batch slot distribution balancing.
"""

import logging
import zlib
from typing import Any, Dict, List, Optional, Set

from .quiz_skeleton_context import SkeletonContext


class QuizSkeletonBuilderMixin:
    @classmethod
    def _build_single_item_skeleton(
        cls,
        it: Dict[str, Any],
        single_word_targets: Set[str],
        batch_target_words: Set[str],
        used_distractors: Set[str],
        nlp: Any,
        has_multiple_candidates: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Coordinates the 4 stages for a single vocabulary target item:
        1. Target Preprocessing & Construction Decomposition
        2. Collocational Anchor Resolution
        3. Distractor Coordination & Collision Screening
        4. Authentic Example Selection & Micro-Task Assembly
        """
        ctx = cls._preprocess_skeleton_target(
            it,
            single_word_targets=single_word_targets,
            batch_target_words=batch_target_words,
            nlp=nlp,
            has_multiple_candidates=has_multiple_candidates,
        )
        if not ctx:
            return None

        cls._resolve_skeleton_anchors(ctx, batch_target_words=batch_target_words, nlp=nlp)
        cls._coordinate_skeleton_distractors(
            ctx,
            used_distractors=used_distractors,
            batch_target_words=batch_target_words,
            nlp=nlp,
        )
        return cls._assemble_skeleton_micro_task(
            ctx,
            used_distractors=used_distractors,
            batch_target_words=batch_target_words,
            nlp=nlp,
        )

    @classmethod
    def build_precomputed_target_skeletons(
        cls,
        vocab_content: Any,
        target_count: int = 10,
        multi_word_ratio: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Track 2: Pre-Computed Generative Skeletons (Proficiency Testing / 语境迁移单选题)
        Accepts structured vocabulary items directly (or legacy markdown string for unit tests)
        and synthesizes authoritative collocational anchors (via PedLex 6th Edition) and
        collision-free distractors (via WordNet + PedLex).
        Produces pre-computed item skeletons to strictly constrain LLM generation, guaranteeing zero
        duplicate options, zero synonym pile double-keys, and perfect stem-option-explanation alignment.
        """
        if not vocab_content:
            return []

        nlp = cls.get_spacy()
        parsed_items = cls._normalize_vocab_items_input(vocab_content, preserve_slash_slots=True)
        if multi_word_ratio <= 0.0 and len(parsed_items) > target_count:
            # Vocabulary Quiz evaluates single lexical items exclusively; exclude multi-word collocations/phrases
            single_candidates = [
                it for it in parsed_items
                if not it.get("is_multi_word")
                and len(str(it.get("word") or "").split()) == 1
                and it.get("part_of_speech", "").lower() not in ("phrase", "collocation", "phrasal verb", "set phrase", "idiom")
            ]
            if len(single_candidates) >= target_count:
                parsed_items = single_candidates
            elif single_candidates:
                rem_needed = max(0, target_count - len(single_candidates))
                multi_candidates = [it for it in parsed_items if it not in single_candidates]
                parsed_items = single_candidates + multi_candidates[:rem_needed]

        if len(parsed_items) > target_count + 4:
            parsed_items = cls._balance_single_and_multi_word(
                parsed_items, target_count=target_count + 4, multi_word_ratio=multi_word_ratio
            )

        skeletons: List[Dict[str, Any]] = []
        used_distractors: Set[str] = set()

        batch_target_words: Set[str] = set()
        single_word_targets: Set[str] = set()
        for _it in parsed_items:
            _w = _it["word"].lower().strip()
            if not _w:
                continue
            batch_target_words.add(_w)
            if " " not in _w:
                single_word_targets.add(_w)
            else:
                for _tok in _w.split():
                    if len(_tok) >= 3 and _tok not in cls._ANCHOR_STOPWORDS:
                        batch_target_words.add(_tok)

        has_multiple_candidates = len(parsed_items) > 1
        for it in parsed_items:
            skel = cls._build_single_item_skeleton(
                it,
                single_word_targets=single_word_targets,
                batch_target_words=batch_target_words,
                used_distractors=used_distractors,
                nlp=nlp,
                has_multiple_candidates=has_multiple_candidates,
            )
            if skel:
                skeletons.append(skel)

        # Balance single-word headwords vs multi-word expressions (e.g. 100% single words by default)
        skeletons = cls._balance_single_and_multi_word(skeletons, target_count=target_count, multi_word_ratio=multi_word_ratio)

        # Balance Correct Answer Index Distribution evenly across 0, 1, 2, 3 (e.g. 3/3/2/2 for 10 items)
        if skeletons:
            n = len(skeletons)
            target_distribution = [n // 4 + (1 if i < (n % 4) else 0) for i in range(4)]
            balanced_slots = []
            for slot_idx, count in enumerate(target_distribution):
                balanced_slots.extend([slot_idx] * count)
            seed_hash = zlib.crc32("".join(s["target_word"] for s in skeletons).encode("utf-8"))
            rng_state = seed_hash
            for i in range(len(balanced_slots) - 1, 0, -1):
                rng_state = (rng_state * 1103515245 + 12345) & 0x7FFFFFFF
                j = rng_state % (i + 1)
                balanced_slots[i], balanced_slots[j] = balanced_slots[j], balanced_slots[i]

            for idx, skel in enumerate(skeletons):
                target_w = skel["target_word"]
                cur_opts = skel["prescribed_options"]
                new_slot = balanced_slots[idx]

                other_opts = [opt for opt in cur_opts if opt.lower() != target_w.lower()]
                if len(other_opts) < 3:
                    continue
                new_opts = other_opts[:new_slot] + [target_w] + other_opts[new_slot:]
                skel["prescribed_options"] = new_opts
                skel["correct_answer_index"] = new_slot

        return skeletons
