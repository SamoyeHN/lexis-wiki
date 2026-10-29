"""
Level 1 deterministic code gates for vocabulary, reading, translation, listening, and video quizzes.
"""

import random
import os
import json
import re
import dataclasses
import base64
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any, Set, Optional
from concurrent.futures import ThreadPoolExecutor
from ..config import config
from ..llm import llm
from ..prompts import Prompts
from ..linguistics import LinguisticEngine

logger = logging.getLogger("librarian.processor")

from ..schemas import (
    VocabularyExtraction, GrammarExtraction, SummaryExtraction,
    VocabularyItem, GrammarItem,
    VocabularyQuiz, ReadingQuiz, TranslationQuiz, ListeningQuiz,
    RoutingResult, MindMapExtraction,
    validate_and_map
)

class QuizIntegrityGatesMixin:

    @staticmethod
    def _max_consecutive_word_overlap(s1: str, s2: str) -> int:
        """Calculates the maximum number of consecutive words shared between two strings."""
        if not s1 or not s2:
            return 0
        w1 = [w.lower() for w in re.findall(r'\b\w+\b', s1)]
        w2 = [w.lower() for w in re.findall(r'\b\w+\b', s2)]
        if not w1 or not w2:
            return 0
        w2_str = " " + " ".join(w2) + " "
        max_k = 0
        for i in range(len(w1)):
            for j in range(i + 1, min(len(w1) + 1, i + 30)):
                gram = " " + " ".join(w1[i:j]) + " "
                if gram in w2_str:
                    max_k = max(max_k, j - i)
                else:
                    break
        return max_k

    @classmethod
    def audit_quiz_integrity(
        cls,
        quiz_obj: Any,
        banned_sentences: List[str] = None,
        unit_headwords: List[str] = None,
        strict_distractor_recycling: bool = True,
        allow_authentic_cloze: bool = False
    ) -> Tuple[List[int], List[str]]:
        """
        Level 1 Deterministic Code Gate for Stem Copying & In-List Distractor Recycling.
        Returns:
            (flagged_indices, defect_messages)
        - Stem Copying Gate: Flags items where the question stem copies >= 7 consecutive words
          from any quoted sentence or example usage.
        - In-List Recycling Gate: Flags items where distractors are recycled from other headwords
          in the unit wordlist.
        """
        if not quiz_obj:
            return [], []

        questions = quiz_obj.get("questions") if isinstance(quiz_obj, dict) else getattr(quiz_obj, "questions", None)
        if not questions or not isinstance(questions, list):
            return [], []

        banned = banned_sentences or []
        headword_set = {h.strip().lower() for h in (unit_headwords or []) if h.strip()}

        flagged_indices = set()
        defect_messages = []
        seen_targets = {}

        for idx, q in enumerate(questions):
            q_dict = q if isinstance(q, dict) else (dataclasses.asdict(q) if dataclasses.is_dataclass(q) else {})
            stem = q_dict.get("question") or q_dict.get("translated_sentence") or ""
            target = (q_dict.get("target_word") or q_dict.get("target_keyword") or q_dict.get("word") or "").strip().lower()
            options = q_dict.get("options") or []
            correct_idx = q_dict.get("correct_answer_index", 0)

            # -1. Target Word Uniqueness Gate
            if target and ("target_word" in q_dict or "word" in q_dict):
                if target in seen_targets:
                    flagged_indices.add(idx)
                    defect_messages.append(
                        f"Item #{idx + 1} ('{target}'): Duplicate target word detected! Already tested in Item #{seen_targets[target] + 1}."
                    )
                else:
                    seen_targets[target] = idx

            # 0. Strict Single Blank Gate & Target Word Leakage Prevention
            if stem and ("target_word" in q_dict or "word" in q_dict):
                # Auto-heal accidental parenthetical leakage: "____ (target)" -> "____"
                if target:
                    esc_target = re.escape(target)
                    cleaned_stem = re.sub(rf"_{{2,}}\s*[\(\[\{{]\s*{esc_target}\s*[\)\]\}}]", "____", stem, flags=re.IGNORECASE)
                    cleaned_stem = re.sub(rf"[\(\[\{{]\s*{esc_target}\s*[\)\]\}}]\s*_{{2,}}", "____", cleaned_stem, flags=re.IGNORECASE)
                    cleaned_stem = re.sub(rf"_{{2,}}\s+{esc_target}\b", "____", cleaned_stem, flags=re.IGNORECASE)
                    cleaned_stem = re.sub(rf"\b{esc_target}\s+_{{2,}}", "____", cleaned_stem, flags=re.IGNORECASE)
                    if cleaned_stem != stem:
                        stem = cleaned_stem
                        if isinstance(q, dict):
                            if "question" in q: q["question"] = stem
                            elif "translated_sentence" in q: q["translated_sentence"] = stem
                        elif dataclasses.is_dataclass(q):
                            if hasattr(q, "question"): setattr(q, "question", stem)
                            elif hasattr(q, "translated_sentence"): setattr(q, "translated_sentence", stem)

                # Auto-heal article leakage: "a ____" / "an ____" -> "a(n) ____"
                # If options mix vowel and consonant initials, an unhedged "a" or "an" directly before
                # the blank leaks the answer or renders distractors grammatically impossible.
                if options and re.search(r"\b(?:a|an)\s+_{2,}\b", stem, flags=re.IGNORECASE):
                    initials = {str(opt).strip()[:1].lower() for opt in options if str(opt).strip()}
                    has_vowel_init = any(init in 'aeiou' for init in initials)
                    has_cons_init = any(init.isalpha() and init not in 'aeiou' for init in initials)
                    if has_vowel_init and has_cons_init:
                        healed_stem = re.sub(r"\b(?:a|an)\s+(_{2,}\b)", r"a(n) \1", stem, flags=re.IGNORECASE)
                        if healed_stem != stem:
                            stem = healed_stem
                            if isinstance(q, dict):
                                if "question" in q: q["question"] = stem
                                elif "translated_sentence" in q: q["translated_sentence"] = stem
                            elif dataclasses.is_dataclass(q):
                                if hasattr(q, "question"): setattr(q, "question", stem)
                                elif hasattr(q, "translated_sentence"): setattr(q, "translated_sentence", stem)

                # Auto-heal redundant finite copula before bare-infinitive 'be ...' phrase options
                # (e.g., 'was ____' + 'be filled with' -> 'seemed to ____' + 'be filled with')
                if options and all(str(opt).strip().lower().startswith("be ") for opt in options if str(opt).strip()):
                    def _repl_copula_gate(m: re.Match) -> str:
                        cop = m.group(1).lower()
                        adv_mid = m.group(2) or ""
                        blank_tok = m.group(3)
                        is_cap = m.group(1)[0].isupper()
                        is_past = cop in ("was", "were") or cop.startswith("had ")
                        if adv_mid.strip():
                            aux = ("Would" if is_cap else "would") if is_past else ("Can" if is_cap else "can")
                        elif is_past:
                            aux = "Seemed to" if is_cap else "seemed to"
                        elif cop == "is" or cop.startswith("has "):
                            aux = "Tends to" if is_cap else "tends to"
                        else:
                            aux = "Tend to" if is_cap else "tend to"
                        return f"{aux} {adv_mid}{blank_tok}"

                    healed_cop_stem = re.sub(
                        r"\b(am|is|are|was|were|(?:has|have|had)\s+been)\s+((?:(?:not|never|always|often|also|still|deeply|completely|fully|truly|quite|very|\w+ly)\s+)*)(_{2,})\b",
                        _repl_copula_gate,
                        stem,
                        flags=re.IGNORECASE,
                    )
                    if healed_cop_stem != stem:
                        stem = healed_cop_stem
                        if isinstance(q, dict):
                            if "question" in q: q["question"] = stem
                            elif "translated_sentence" in q: q["translated_sentence"] = stem
                        elif dataclasses.is_dataclass(q):
                            if hasattr(q, "question"): setattr(q, "question", stem)
                            elif hasattr(q, "translated_sentence"): setattr(q, "translated_sentence", stem)

                # Auto-heal redundant catenative 'promised/agreed/decided/sought/refused to ____' before past-tense '-ed' verb options
                # (e.g. 'The minister promised to ____ that...' + [denied, assured, affirmed, testified] -> 'The minister ____ that...')
                if options and len(options) == 4 and all(
                    str(opt).strip().lower().endswith("ed") and " " not in str(opt).strip()
                    for opt in options if str(opt).strip()
                ):
                    healed_vbd_stem = re.sub(
                        r"\b(?:promised|agreed|decided|chose|sought|tended|seemed|appeared|managed|attempted|tried|refused|claimed|offered|hoped|wanted)\s+to\s+(_{2,}\b)",
                        r"\1",
                        stem,
                        flags=re.IGNORECASE,
                    )
                    if healed_vbd_stem != stem:
                        stem = healed_vbd_stem
                        if isinstance(q, dict):
                            if "question" in q: q["question"] = stem
                            elif "translated_sentence" in q: q["translated_sentence"] = stem
                        elif dataclasses.is_dataclass(q):
                            if hasattr(q, "question"): setattr(q, "question", stem)
                            elif hasattr(q, "translated_sentence"): setattr(q, "translated_sentence", stem)

                blank_matches = re.findall(r'_{2,}', stem)
                if len(blank_matches) > 1:
                    flagged_indices.add(idx)
                    defect_messages.append(
                        f"Item #{idx + 1} ('{target}'): Multiple blanks ({len(blank_matches)}) detected in stem. Only exactly ONE blank '____' is permitted."
                    )
                elif len(blank_matches) == 0:
                    flagged_indices.add(idx)
                    defect_messages.append(
                        f"Item #{idx + 1} ('{target}'): Missing blank '____' in stem."
                    )

                # Flag any remaining target leak (verbatim, inflectional root, or prefix-stripped derivational root) in the stem outside the blank
                if target:
                    esc_target = re.escape(target)
                    stem_no_blank = re.sub(r'_{2,}', ' ', stem)
                    stem_words = re.findall(r'\b[a-zA-Z]+\b', stem_no_blank)
                    target_root = re.sub(r'(?:ed|ing|s|es|ly|tion|ment)$', '', target.lower())
                    leaked_in_stem = [
                        w for w in stem_words
                        if w.lower() == target.lower()
                        or (len(target_root) >= 4 and re.sub(r'(?:ed|ing|s|es|ly|tion|ment)$', '', w.lower()) == target_root)
                        or (" " not in target and len(w) >= 4 and LinguisticEngine.shares_derivational_root(w.lower(), target.lower()))
                    ]
                    if leaked_in_stem:
                        flagged_indices.add(idx)
                        defect_messages.append(
                            f"Item #{idx + 1} ('{target}'): Target word leaks into stem text outside blank: {leaked_in_stem}"
                        )

            # 1. Stem Copying Gate (Zero-Tolerance for synthetic items, permitted for Track 1 Cloze when enabled)
            if stem and banned:
                clean_stem = re.sub(r'_{2,}', ' ', stem)
                for b_sent in banned:
                    overlap_len = cls._max_consecutive_word_overlap(clean_stem, b_sent)
                    if overlap_len >= 7:
                        # If this stem is an exact cloze mask of an authentic passage sentence, check if Track 1 cloze mode is enabled
                        is_authentic_cloze = False
                        if allow_authentic_cloze and target:
                            b_masked = re.sub(rf'\b{re.escape(target)}\b', '____', b_sent, flags=re.IGNORECASE)
                            norm_stem = re.sub(r'[^\w\s]', '', stem).strip().lower()
                            norm_b_masked = re.sub(r'[^\w\s]', '', b_masked).strip().lower()
                            if norm_stem == norm_b_masked or overlap_len >= min(len(clean_stem.split()) - 2, 10):
                                is_authentic_cloze = True

                        if not is_authentic_cloze:
                            flagged_indices.add(idx)
                            defect_messages.append(
                                f"Item #{idx + 1} ('{target}'): Stem copies {overlap_len} consecutive words "
                                f"from input example/source: \"{b_sent[:60]}...\""
                            )
                            break

            # 2. In-List Distractor Recycling Gate (used when verifying L2 LLM candidate cures or legacy tests)
            if options and headword_set and strict_distractor_recycling:
                recycled_distractors = []
                for o_idx, opt in enumerate(options):
                    if o_idx == correct_idx:
                        continue
                    opt_clean = str(opt).strip().lower()
                    if re.search(r'\((?:synonym|antonym|distractor|correct|key|target|meaning|i\.e\.|e\.g\.)\b[^)]*\)', opt_clean):
                        flagged_indices.add(idx)
                        defect_messages.append(
                            f"Item #{idx + 1} ('{target}'): Distractor '{opt}' contains parenthetical meta-annotation."
                        )
                        continue
                    if target and opt_clean != target and re.search(rf'\b{re.escape(target)}\b', opt_clean):
                        flagged_indices.add(idx)
                        defect_messages.append(
                            f"Item #{idx + 1} ('{target}'): Distractor '{opt}' leaks the target word."
                        )
                        continue
                    # If distractor is an inflectional variant of this item's own single-word target, exempt it
                    if target and (
                        opt_clean == target or
                        (" " not in opt_clean and len(target) >= 4 and opt_clean.startswith(target[:4]))
                    ):
                        continue

                    # If distractor is in the unit's headword list (exact or inflectional derivative) and differs from target
                    is_recycled = False
                    if opt_clean in headword_set:
                        is_recycled = True
                    else:
                        # Check inflectional variants (e.g., overwhelm -> overwhelmed, pledge -> pledging)
                        opt_stem = re.sub(r'(?:ed|ing|s|es|ly|tion|ment)$', '', opt_clean)
                        for hw in headword_set:
                            if hw == target:
                                continue
                            # Prefix matching only applies to single-word items (never across multi-word phrases sharing a light verb like 'take ')
                            if " " in hw or " " in opt_clean:
                                continue
                            hw_stem = re.sub(r'(?:ed|ing|s|es|ly|tion|ment)$', '', hw)
                            if (len(hw_stem) >= 4 and len(opt_stem) >= 4 and hw_stem == opt_stem) or \
                               (len(hw) >= 5 and opt_clean.startswith(hw[:5])) or \
                               (len(opt_clean) >= 5 and hw.startswith(opt_clean[:5])):
                                is_recycled = True
                                break
                    if is_recycled:
                        recycled_distractors.append(opt_clean)
                if recycled_distractors:
                    flagged_indices.add(idx)
                    defect_messages.append(
                        f"Item #{idx + 1} ('{target}'): Distractors recycle headwords from current unit: {recycled_distractors}"
                    )

            # 3. Degraded-Options & Key-Target Alignment Gate (structural option cardinality & key invariant)
            if "target_word" in q_dict and options:
                if len(options) < 4:
                    flagged_indices.add(idx)
                    defect_messages.append(
                        f"Item #{idx + 1} ('{target}'): Degraded item — only {len(options)} options "
                        f"(exactly 4 required). Clear double-keys by refilling, not by truncation."
                    )
                elif isinstance(correct_idx, int) and 0 <= correct_idx < len(options) and target:
                    key_opt = str(options[correct_idx]).strip().lower()
                    target_forms = {target} | set(LinguisticEngine.inflected_forms(target))
                    key_forms = {key_opt} | set(LinguisticEngine.inflected_forms(key_opt))
                    if not (target_forms & key_forms) and key_opt not in target and target not in key_opt:
                        flagged_indices.add(idx)
                        defect_messages.append(
                            f"Item #{idx + 1} ('{target}'): Declared key option '{options[correct_idx]}' "
                            f"at index {correct_idx} does not match target_word '{target}'."
                        )

        return sorted(list(flagged_indices)), defect_messages

    @classmethod
    def audit_reading_integrity(
        cls,
        quiz_obj: Any,
        passage_text: str = "",
        cefr_level: str = "B2"
    ) -> Tuple[List[int], List[str]]:
        """
        Level 1 Deterministic Code Gate for Reading Comprehension Quiz.
        Enforces physical ground-truth invariants:
        1. Skill Diversity: Balanced mix of Main Idea, Detail/Recall, Inference, Author's Tone/Purpose.
        2. Structural Option Bounds: Exactly 4 distinct, parallel, non-empty options.
        3. Option Length & Complexity Bounds: Calibrated to CEFR level (A1/A2 <= 16 words, B1 <= 20 words).
        4. CEFR Difficulty Ceiling Filter: For A1/A2 foundational texts, options/stems cannot contain C1/C2 words not in passage.
        5. Answer Index Integrity: Bound [0, 3] check.
        6. Verbatim Text Anchoring & Anti-Hallucination: context_sentence must physically exist in passage.
        7. Trivia / Option Echo Filter: Prevent trivially verbatim options or stem-option duplication.
        Returns:
            (flagged_indices, defect_messages)
        """
        if not quiz_obj:
            return [], []

        questions = quiz_obj.get("questions") if isinstance(quiz_obj, dict) else getattr(quiz_obj, "questions", None)
        if not questions or not isinstance(questions, list):
            return [], ["Missing or invalid 'questions' array in Reading Quiz."]

        flagged_indices = set()
        defect_messages = []
        passage_norm = re.sub(r'\s+', ' ', passage_text.lower()) if passage_text else ""
        clean_passage_words = set(re.findall(r"\b[a-zA-Z]{3,}\b", passage_text.lower())) if passage_text else set()
        cefr_upper = str(cefr_level or "B2").upper()
        meta_whitelist = {
            "according", "paragraph", "passage", "infer", "inferred", "inference",
            "author", "statement", "summarize", "suggests", "purpose", "following",
            "attitude", "tone", "mainly", "primary", "central", "illustrate", "express"
        }

        # 1. Inspect questions
        categories_seen = set()
        for idx, q in enumerate(questions):
            q_dict = q if isinstance(q, dict) else (dataclasses.asdict(q) if dataclasses.is_dataclass(q) else {})
            stem = str(q_dict.get("question", "")).strip()
            cat = str(q_dict.get("category", "")).strip()
            options = q_dict.get("options") or []
            correct_idx = q_dict.get("correct_answer_index")

            if cat:
                categories_seen.add(cat.lower())

            # 1.1 Check Option Count
            if len(options) != 4:
                flagged_indices.add(idx)
                defect_messages.append(f"Reading Item #{idx + 1}: Must contain exactly 4 options (found {len(options)})")
            else:
                # 1.2 Check Option Uniqueness
                cleaned_opts = [str(o).strip().lower() for o in options]
                if len(set(cleaned_opts)) < 4:
                    flagged_indices.add(idx)
                    defect_messages.append(f"Reading Item #{idx + 1}: Options contain duplicate choices: {options}")

                # 1.3 Check Option Echo (option identical to stem)
                stem_lower = stem.lower()
                for opt in cleaned_opts:
                    if len(opt) > 15 and opt in stem_lower:
                        flagged_indices.add(idx)
                        defect_messages.append(f"Reading Item #{idx + 1}: Option is an echo of the question stem: \"{opt[:40]}\"")
                        break

                # 1.4 Check Option Complexity & Length based on CEFR
                max_opt_len = 16 if cefr_upper in ("A1", "A2") else (20 if cefr_upper == "B1" else 32)
                for opt in options:
                    opt_words = str(opt).split()
                    if len(opt_words) > max_opt_len:
                        flagged_indices.add(idx)
                        defect_messages.append(
                            f"Reading Item #{idx + 1}: Option is overly long for CEFR {cefr_upper} ({len(opt_words)} words > {max_opt_len}): \"{opt[:40]}...\""
                        )
                        break

                # 1.5 Check Difficulty Ceiling (shared C1 helper: over_ceiling_tokens)
                for opt in options:
                    over = LinguisticEngine.over_ceiling_tokens(
                        str(opt), cefr_upper, mode="text",
                        allow=clean_passage_words | meta_whitelist,
                    )
                    if over:
                        flagged_indices.add(idx)
                        tok, tok_lvl = over[0]
                        defect_messages.append(
                            f"Reading Item #{idx + 1}: Option contains obscure/super-advanced word '{tok}' ({tok_lvl}) exceeding CEFR {cefr_upper} ceiling: \"{opt[:40]}\""
                        )
                        break

            # 1.6 Check Key Index Bounds
            if not isinstance(correct_idx, int) or correct_idx not in (0, 1, 2, 3):
                flagged_indices.add(idx)
                defect_messages.append(f"Reading Item #{idx + 1}: Invalid correct_answer_index ({correct_idx})")

            # 1.7 Check Stem Quality & CEFR Ceiling
            if len(stem.split()) < 4:
                flagged_indices.add(idx)
                defect_messages.append(f"Reading Item #{idx + 1}: Question stem is too short or empty: \"{stem}\"")
            else:
                over = LinguisticEngine.over_ceiling_tokens(
                    stem, cefr_upper, mode="text",
                    allow=clean_passage_words | meta_whitelist,
                )
                if over:
                    flagged_indices.add(idx)
                    tok, tok_lvl = over[0]
                    defect_messages.append(
                        f"Reading Item #{idx + 1}: Question stem contains obscure/super-advanced word '{tok}' ({tok_lvl}) exceeding CEFR {cefr_upper} ceiling: \"{stem[:40]}\""
                    )

        # 2. Skill Diversity Gate
        if len(questions) >= 4 and len(categories_seen) < 2:
            defect_messages.append(
                f"Reading Skill Gate: Insufficient skill diversity (found only {list(categories_seen)}). "
                "Must distribute across Main Idea, Detail/Recall, Inference, and Author's Tone/Purpose."
            )
            for i in range(1, min(3, len(questions))):
                flagged_indices.add(len(questions) - i)

        # 3. Vocabulary Physical Grounding Gate
        vocab_items = quiz_obj.get("vocabulary") if isinstance(quiz_obj, dict) else getattr(quiz_obj, "vocabulary", None)
        if passage_norm and vocab_items and isinstance(vocab_items, list):
            for v_idx, v in enumerate(vocab_items):
                v_dict = v if isinstance(v, dict) else (dataclasses.asdict(v) if dataclasses.is_dataclass(v) else {})
                word = str(v_dict.get("word", "")).strip().lower()
                c_sent = str(v_dict.get("context_sentence", "")).strip()

                if not word:
                    defect_messages.append(f"Reading Vocabulary Item #{v_idx + 1}: Missing target word")
                    continue

                if c_sent:
                    # Clean punctuation and normalize whitespace
                    clean_c_sent = re.sub(r'[^\w\s]', '', c_sent.lower())
                    clean_c_sent = re.sub(r'\s+', ' ', clean_c_sent).strip()
                    clean_passage = re.sub(r'[^\w\s]', '', passage_norm)
                    clean_passage = re.sub(r'\s+', ' ', clean_passage).strip()

                    c_words = clean_c_sent.split()
                    matched = False
                    if len(c_words) >= 4:
                        # Check windows of 4 words
                        for i in range(len(c_words) - 3):
                            test_chunk = " ".join(c_words[i:i+4])
                            if test_chunk in clean_passage:
                                matched = True
                                break
                    elif clean_c_sent in clean_passage:
                        matched = True

                    if not matched:
                        defect_messages.append(
                            f"Reading Vocabulary Item #{v_idx + 1} ('{word}'): context sentence "
                            f"does not match verbatim text in passage: \"{c_sent[:50]}...\""
                        )
                else:
                    defect_messages.append(f"Reading Vocabulary Item #{v_idx + 1} ('{word}'): Empty context_sentence")

        return sorted(list(flagged_indices)), defect_messages

    @classmethod
    def audit_video_integrity(
        cls,
        quiz_obj: Any,
        transcript_text: str = "",
        cefr_level: str = "B2"
    ) -> Tuple[List[int], List[str]]:
        """
        Level 1 Deterministic Code Gate for Video Comprehension Quiz.
        Enforces 5 physical ground-truth invariants:
        1. Structural Option Bounds: Exactly 4 distinct, non-empty options.
        2. Answer Index Integrity: Bound [0, 3] check.
        3. Option Complexity & Length Bounds based on CEFR.
        4. CEFR Vocabulary Difficulty Ceiling (cefrpy) for Foundational CEFR (A1, A2).
        5. Timestamp Format & Grounding: Valid [MM:SS] or [HH:MM:SS] and physically anchored in transcript.
        Returns:
            (flagged_indices, defect_messages)
        """
        if not quiz_obj:
            return [], []

        questions = quiz_obj.get("questions") if isinstance(quiz_obj, dict) else getattr(quiz_obj, "questions", None)
        if not questions or not isinstance(questions, list):
            return [], ["Missing or invalid 'questions' array in Video Quiz."]

        flagged_indices = set()
        defect_messages = []
        transcript_norm = transcript_text.lower() if transcript_text else ""
        cefr_upper = (cefr_level or "B2").upper()

        clean_transcript_words = set(re.findall(r"\b[a-zA-Z]{4,}\b", transcript_norm))
        meta_whitelist = {
            "speaker", "transcript", "statement", "mention", "discuss", "indicate", "imply", "conclude",
            "section", "passage", "reason", "purpose", "detail", "meaning", "option", "correct", "video",
            "timestamp", "demonstrate", "illustrate", "according", "paragraph", "author", "true", "false"
        }

        for idx, q in enumerate(questions):
            q_dict = q if isinstance(q, dict) else (dataclasses.asdict(q) if dataclasses.is_dataclass(q) else {})
            stem = str(q_dict.get("question", "")).strip()
            options = q_dict.get("options") or []
            correct_idx = q_dict.get("correct_answer_index")
            ts = str(q_dict.get("timestamp", "")).strip()

            # 1. Option Bounds
            if len(options) != 4:
                flagged_indices.add(idx)
                defect_messages.append(f"Video Item #{idx + 1}: Must contain exactly 4 options (found {len(options)})")
            else:
                cleaned_opts = [str(o).strip().lower() for o in options]
                if len(set(cleaned_opts)) < 4:
                    flagged_indices.add(idx)
                    defect_messages.append(f"Video Item #{idx + 1}: Options contain duplicate choices: {options}")

                # 1.1 Option Complexity & Length based on CEFR
                max_opt_len = 14 if cefr_upper in ("A1", "A2") else (18 if cefr_upper == "B1" else 30)
                for opt in options:
                    opt_words = str(opt).split()
                    if len(opt_words) > max_opt_len:
                        flagged_indices.add(idx)
                        defect_messages.append(
                            f"Video Item #{idx + 1}: Option is overly long for CEFR {cefr_upper} ({len(opt_words)} words > {max_opt_len}): \"{opt[:40]}...\""
                        )
                        break

                # 1.2 Difficulty Ceiling (shared C1 helper: over_ceiling_tokens)
                for opt in options:
                    over = LinguisticEngine.over_ceiling_tokens(
                        str(opt), cefr_upper, mode="text",
                        allow=clean_transcript_words | meta_whitelist,
                    )
                    if over:
                        flagged_indices.add(idx)
                        tok, tok_lvl = over[0]
                        defect_messages.append(
                            f"Video Item #{idx + 1}: Option contains obscure/super-advanced word '{tok}' ({tok_lvl}) exceeding CEFR {cefr_upper} ceiling: \"{opt[:40]}\""
                        )
                        break

            # 2. Key Index Bounds
            if not isinstance(correct_idx, int) or correct_idx not in (0, 1, 2, 3):
                flagged_indices.add(idx)
                defect_messages.append(f"Video Item #{idx + 1}: Invalid correct_answer_index ({correct_idx})")

            # 3. Stem Quality & CEFR Ceiling
            if len(stem.split()) < 4:
                flagged_indices.add(idx)
                defect_messages.append(f"Video Item #{idx + 1}: Question stem is too short or empty: \"{stem}\"")
            else:
                over = LinguisticEngine.over_ceiling_tokens(
                    stem, cefr_upper, mode="text",
                    allow=clean_transcript_words | meta_whitelist,
                )
                if over:
                    flagged_indices.add(idx)
                    tok, tok_lvl = over[0]
                    defect_messages.append(
                        f"Video Item #{idx + 1}: Question stem contains obscure/super-advanced word '{tok}' ({tok_lvl}) exceeding CEFR {cefr_upper} ceiling: \"{stem[:40]}\""
                    )

            # 4. Timestamp Validation
            if not ts:
                flagged_indices.add(idx)
                defect_messages.append(f"Video Item #{idx + 1}: Missing timestamp anchor.")
            else:
                # Check timestamp format like [01:23] or 01:23
                ts_clean = re.sub(r'[\[\]]', '', ts).strip()
                if not re.match(r'^\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?$', ts_clean):
                    flagged_indices.add(idx)
                    defect_messages.append(f"Video Item #{idx + 1}: Malformed timestamp format \"{ts}\". Expected [MM:SS] or [HH:MM:SS].")
                elif transcript_norm:
                    # Check if timestamp prefix exists in transcript
                    # e.g., if ts_clean is "01:23", match "01:23" or "[01:23"
                    ts_min_sec = ts_clean.split(".")[0]
                    if ts_min_sec not in transcript_norm:
                        # Soft warning/check: timestamp not found in transcript
                        defect_messages.append(f"Video Item #{idx + 1}: Timestamp [{ts_clean}] not found in video transcript.")

        return sorted(list(flagged_indices)), defect_messages

    @classmethod
    def _expand_headword_whitelist(cls, unit_headwords: Optional[List[str]]) -> Set[str]:
        """Expands unit headwords (including multi-word expressions and inflections) into a lowercase token whitelist."""
        allow_set: Set[str] = set()
        if not unit_headwords:
            return allow_set
        for hw in unit_headwords:
            if not hw:
                continue
            hw_clean = re.sub(r"[\(\[\{].*?[\)\]\}]", " ", str(hw).lower())
            for tok in re.findall(r"[a-zA-Z]{3,}", hw_clean):
                if tok in {"sth", "somebody", "something", "someone", "oneself"}:
                    continue
                allow_set.add(tok)
                try:
                    allow_set.update(LinguisticEngine.inflected_forms(tok))
                except Exception:
                    pass
        return allow_set

    @classmethod
    def audit_listening_integrity(
        cls,
        quiz_obj: Any,
        script_text: str = "",
        cefr_level: str = "B2",
        unit_headwords: List[str] = None
    ) -> Tuple[List[int], List[str]]:
        """
        Level 1 Deterministic Code Gate for Listening Comprehension Quiz.
        Enforces 5 physical ground-truth invariants:
        1. Dialogue Script Integrity: At least 4 speaker turns in script.
        2. Structural Option Bounds: Exactly 4 distinct, non-empty options.
        3. Option Complexity & Length Bounds based on CEFR.
        4. CEFR Vocabulary Difficulty Ceiling (cefrpy) for Foundational CEFR (A1, A2).
        5. Answer Index Integrity: Bound [0, 3] check.
        6. Category Integrity: Detail, Main Idea, or Inference.
        Returns:
            (flagged_indices, defect_messages)
        """
        if not quiz_obj:
            return [], []

        questions = quiz_obj.get("questions") if isinstance(quiz_obj, dict) else getattr(quiz_obj, "questions", None)
        if not questions or not isinstance(questions, list):
            return [], ["Missing or invalid 'questions' array in Listening Quiz."]

        flagged_indices = set()
        defect_messages = []
        cefr_upper = (cefr_level or "B2").upper()

        script_norm = script_text.lower() if script_text else ""
        clean_script_words = set(re.findall(r"\b[a-zA-Z]{4,}\b", script_norm))
        headword_whitelist = cls._expand_headword_whitelist(unit_headwords)
        meta_whitelist = {
            "speaker", "transcript", "statement", "mention", "discuss", "indicate", "imply", "conclude",
            "dialogue", "conversation", "reason", "purpose", "detail", "meaning", "option", "correct",
            "opinion", "suggest", "agree", "disagree", "talk", "true", "false", "about"
        }

        # 1. Script checks
        script_items = quiz_obj.get("script") if isinstance(quiz_obj, dict) else getattr(quiz_obj, "script", None)
        if script_items is not None and isinstance(script_items, list):
            if len(script_items) < 4:
                defect_messages.append(f"Listening Script Gate: Dialogue script contains fewer than 4 turns ({len(script_items)} turns).")
            # Always merge dialogue turn words into clean_script_words so script vocabulary is licensed
            all_turn_text = " ".join([str(item.get("text", "") if isinstance(item, dict) else getattr(item, "text", "")) for item in script_items])
            clean_script_words.update(re.findall(r"\b[a-zA-Z]{4,}\b", all_turn_text.lower()))

        combined_allow = clean_script_words | headword_whitelist | meta_whitelist

        # 2. Inspect questions
        valid_cats = {"detail", "main idea", "inference"}
        for idx, q in enumerate(questions):
            q_dict = q if isinstance(q, dict) else (dataclasses.asdict(q) if dataclasses.is_dataclass(q) else {})
            stem = str(q_dict.get("question", "")).strip()
            options = q_dict.get("options") or []
            correct_idx = q_dict.get("correct_answer_index")
            cat = str(q_dict.get("category", "")).strip().lower()

            # 2.1 Option Bounds
            if len(options) != 4:
                flagged_indices.add(idx)
                defect_messages.append(f"Listening Item #{idx + 1}: Must contain exactly 4 options (found {len(options)})")
            else:
                cleaned_opts = [str(o).strip().lower() for o in options]
                if len(set(cleaned_opts)) < 4:
                    flagged_indices.add(idx)
                    defect_messages.append(f"Listening Item #{idx + 1}: Options contain duplicate choices: {options}")

                # 2.1.1 Option Complexity & Length based on CEFR
                max_opt_len = 12 if cefr_upper in ("A1", "A2") else (16 if cefr_upper == "B1" else 28)
                for opt in options:
                    opt_words = str(opt).split()
                    if len(opt_words) > max_opt_len:
                        flagged_indices.add(idx)
                        defect_messages.append(
                            f"Listening Item #{idx + 1}: Option is overly long for CEFR {cefr_upper} ({len(opt_words)} words > {max_opt_len}): \"{opt[:40]}...\""
                        )
                        break

                # 2.1.2 Difficulty Ceiling (shared C1 helper: over_ceiling_tokens)
                for opt in options:
                    over = LinguisticEngine.over_ceiling_tokens(
                        str(opt), cefr_upper, mode="text",
                        allow=combined_allow,
                    )
                    if over:
                        flagged_indices.add(idx)
                        tok, tok_lvl = over[0]
                        defect_messages.append(
                            f"Listening Item #{idx + 1}: Option contains obscure/super-advanced word '{tok}' ({tok_lvl}) exceeding CEFR {cefr_upper} ceiling: \"{opt[:40]}\""
                        )
                        break

            # 2.2 Key Index Bounds
            if not isinstance(correct_idx, int) or correct_idx not in (0, 1, 2, 3):
                flagged_indices.add(idx)
                defect_messages.append(f"Listening Item #{idx + 1}: Invalid correct_answer_index ({correct_idx})")

            # 2.3 Stem Quality & CEFR Ceiling
            if len(stem.split()) < 4:
                flagged_indices.add(idx)
                defect_messages.append(f"Listening Item #{idx + 1}: Question stem is too short or empty: \"{stem}\"")
            else:
                over = LinguisticEngine.over_ceiling_tokens(
                    stem, cefr_upper, mode="text",
                    allow=combined_allow,
                )
                if over:
                    flagged_indices.add(idx)
                    tok, tok_lvl = over[0]
                    defect_messages.append(
                        f"Listening Item #{idx + 1}: Question stem contains obscure/super-advanced word '{tok}' ({tok_lvl}) exceeding CEFR {cefr_upper} ceiling: \"{stem[:40]}\""
                    )

            # 2.4 Category check
            if cat and cat not in valid_cats:
                defect_messages.append(f"Listening Item #{idx + 1}: Category '{cat}' is not one of 'Detail', 'Main Idea', 'Inference'.")

        return sorted(list(flagged_indices)), defect_messages

    @classmethod
    def audit_translation_integrity(
        cls,
        quiz_obj: Any,
        unit_headwords: List[str] = None,
        unit_grammar_patterns: List[str] = None,
        target_language: str = "Chinese",
        cefr_level: str = "B2",
        enforce_minimal_pair: bool = False,
        passage_text: str = "",
    ) -> Tuple[List[int], List[str]]:
        """
        Level 1 Deterministic Code Gate for Target-to-English Translation Quiz.
        Enforces 6 physical ground-truth invariants:
        1. Language Purity Gate:
           - {target_language} prompt sentence must contain valid {target_language} characters.
           - Options must be 100% English sentences, strictly free of {target_language} characters.
        2. Option Complexity & Length Bounds based on CEFR.
        3. CEFR Vocabulary Difficulty Ceiling (cefrpy) for Foundational CEFR (A1, A2).
        4. Target Vocabulary Presence Gate:
           - The declared target_word (from design_audit or unit wordlist) must physically appear
             verbatim or with inflection in correct_english_answer / idiomatic_translation.
        5. Structural Parallelism Gate:
           - Exactly 2 options (comparative appraisal) or 4 options (anchored skeleton).
           - Key index within bounds.
        6. Option Echo / Giveaway Filter.
        Returns:
            (flagged_indices, defect_messages)
        """
        if not quiz_obj:
            return [], []

        questions = quiz_obj.get("questions") if isinstance(quiz_obj, dict) else getattr(quiz_obj, "questions", None)
        if not questions or not isinstance(questions, list):
            return [], ["Missing or invalid 'questions' array in Translation Quiz."]

        flagged_indices = set()
        defect_messages = []
        cefr_upper = (cefr_level or "B2").upper()

        headword_whitelist = cls._expand_headword_whitelist(unit_headwords)
        clean_passage_words = set(re.findall(r"\b[a-zA-Z]{3,}\b", passage_text.lower())) if passage_text else set()
        meta_whitelist = {
            "speaker", "transcript", "statement", "mention", "discuss", "indicate", "imply", "conclude",
            "sentence", "translation", "english", "chinese", "option", "correct", "flawed", "idiomatic"
        } | headword_whitelist | clean_passage_words

        # Determine target language regex pattern
        lang_str = (target_language or "Chinese").lower()
        if any(w in lang_str for w in ["chinese", "mandarin", "cjk", "中文", "汉语", "漢語"]):
            target_char_pattern = r'[\u4e00-\u9fff]'
        elif any(w in lang_str for w in ["japanese", "日"]):
            target_char_pattern = r'[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]'
        elif any(w in lang_str for w in ["korean", "韩", "韓"]):
            target_char_pattern = r'[\uac00-\ud7af\u1100-\u11ff]'
        elif any(w in lang_str for w in ["russian", "cyrillic", "俄"]):
            target_char_pattern = r'[\u0400-\u04ff]'
        elif any(w in lang_str for w in ["arabic", "阿"]):
            target_char_pattern = r'[\u0600-\u06ff]'
        elif any(w in lang_str for w in ["thai", "泰"]):
            target_char_pattern = r'[\u0e00-\u0e7f]'
        elif any(w in lang_str for w in ["spanish", "french", "german", "italian", "portuguese"]):
            # European target languages (Latin with accented / non-ASCII Latin characters)
            target_char_pattern = r'[A-Za-z\u00C0-\u024F]'
            option_forbidden_char_pattern = r'[\u00C0-\u024F\u4e00-\u9fff]'
        else:
            # Default fallback: check for CJK or non-ASCII characters
            target_char_pattern = r'[\u4e00-\u9fff]|[^\x00-\x7F]'
            option_forbidden_char_pattern = r'[\u4e00-\u9fff]'

        if "option_forbidden_char_pattern" not in locals():
            option_forbidden_char_pattern = target_char_pattern

        # Inflectional / Stemming helper for target word matching
        def _word_in_text(word: str, text: str) -> bool:
            w = word.strip().lower()
            t = text.lower()
            if not w or not t:
                return False
            # Clean placeholders like "sth", "sb", "something", "someone" (e.g. "impose sth on sb" -> "impose on")
            w_clean = re.sub(r'\b(sth|sb|something|someone|somebody|oneself)\b', '', w)
            w_clean = re.sub(r'[^\w\s\-]', ' ', w_clean)
            w_clean = ' '.join(w_clean.split())
            if w_clean:
                w = w_clean
            # Direct inflection-aware check via LinguisticEngine
            try:
                if LinguisticEngine.text_contains_form(t, w):
                    return True
            except Exception:
                pass
            # Multi-word phrase check (handling inflected verbs/nouns inside the phrase, e.g. "turn up" -> "turned up")
            if " " in w:
                if w in t:
                    return True
                raw_parts = [p for p in w.split() if len(p) > 1]
                if len(raw_parts) >= 2:
                    # Allow each content token in the phrase to match via LinguisticEngine.text_contains_form
                    if all(LinguisticEngine.text_contains_form(t, p) for p in raw_parts):
                        return True
                    # Regex fallback allowing optional inflectional suffixes on each part with flexible gap
                    part_patterns = []
                    for p in raw_parts:
                        p_stem = re.sub(r'(?:ed|ing|es|s|d)$', '', p) if len(p) >= 5 else p
                        part_patterns.append(rf"{re.escape(p_stem)}\w*")
                    pattern = r'\b' + r'\b.*?\b'.join(part_patterns) + r'\b'
                    if re.search(pattern, t):
                        return True
            # Exact word boundary
            if re.search(rf"\b{re.escape(w)}\b", t):
                return True
            # Stage 6 Cross-POS / Derived-Form Guard: if target is an -ly derived adverb
            # (e.g., 'effectively'), do NOT accept its base adjective ('effective').
            if " " not in w and len(w) >= 5 and w.endswith("ly"):
                return False
            # Irregular past tense / inflection pairs
            irregulars = {
                "go": ["went", "gone", "goes", "going"],
                "went": ["go", "gone"],
                "impose": ["imposed", "imposing", "imposes"],
                "be": ["is", "am", "are", "was", "were", "been", "being"]
            }
            for root, forms in irregulars.items():
                if w == root or root in w.split():
                    for f in forms:
                        if re.search(rf"\b{re.escape(f)}\b", t):
                            return True
            # Inflectional variants check (strictly inflectional suffixes only, never derivational -ly/-tion/-ive)
            w_stem = re.sub(r'(?:ed|ing|es|s|d|er|est)$', '', w)
            if len(w_stem) >= 4:
                if re.search(rf"\b{re.escape(w_stem)}(?:s|es|d|ed|ing|er|est)?\b", t):
                    return True
            return False

        for idx, q in enumerate(questions):
            q_dict = q if isinstance(q, dict) else (dataclasses.asdict(q) if dataclasses.is_dataclass(q) else {})
            stem = str(q_dict.get("translated_sentence", "")).strip()
            options = q_dict.get("options") or []
            correct_idx = q_dict.get("correct_answer_index")
            correct_eng = str(q_dict.get("correct_english_answer", "")).strip()
            audit_str = str(q_dict.get("design_audit", "")).strip()

            idiomatic_trans = str(q_dict.get("idiomatic_translation", "")).strip()
            flawed_trans = str(q_dict.get("flawed_translation", "")).strip()
            is_comparative = bool(idiomatic_trans and flawed_trans)

            # Auto-synthesize 2 options (Version A vs Version B) if model output Scheme B fields but options not yet built
            if is_comparative and len(options) != 2:
                # Default canonical layout before shuffling: [idiomatic, flawed]
                options = [idiomatic_trans, flawed_trans]
                correct_idx = 0
                if isinstance(q, dict):
                    q["options"] = options
                    q["correct_answer_index"] = correct_idx
                elif hasattr(q, "options"):
                    q.options = options
                    q.correct_answer_index = correct_idx

            expected_opt_count = 2 if is_comparative else 4

            # 1. Language Purity Gate
            # 1.1 Stem must contain target language characters
            if not re.search(target_char_pattern, stem):
                flagged_indices.add(idx)
                defect_messages.append(
                    f"Translation Item #{idx + 1}: 'translated_sentence' must contain {target_language} characters."
                )

            # 1.2 Options must be purely in English (no target language characters)
            for o_i, opt in enumerate(options):
                opt_str = str(opt)
                if re.search(option_forbidden_char_pattern, opt_str):
                    flagged_indices.add(idx)
                    defect_messages.append(
                        f"Translation Item #{idx + 1}: Option [{chr(65 + o_i)}] contains {target_language} characters: \"{opt_str[:40]}...\""
                    )

            # 2. Structural & Parallelism Gate
            if len(options) != expected_opt_count:
                flagged_indices.add(idx)
                defect_messages.append(
                    f"Translation Item #{idx + 1}: Must contain exactly {expected_opt_count} options (found {len(options)})"
                )
            else:
                # 2.1 Option uniqueness
                cleaned_opts = [str(o).strip().lower() for o in options]
                if len(set(cleaned_opts)) < expected_opt_count:
                    flagged_indices.add(idx)
                    defect_messages.append(
                        f"Translation Item #{idx + 1}: Options contain duplicate choices."
                    )

                # 2.2 Parallelism / Length balance (only check for >= 3 options)
                if expected_opt_count >= 3:
                    opt_lens = [len(o.split()) for o in cleaned_opts]
                    min_len = min(opt_lens) if opt_lens else 0
                    max_len = max(opt_lens) if opt_lens else 0
                    if min_len > 0 and max_len > 4 and (max_len / min_len) > 3.0:
                        flagged_indices.add(idx)
                        defect_messages.append(
                            f"Translation Item #{idx + 1}: Extreme length disparity across options (shortest={min_len} words, longest={max_len} words)."
                        )

                # 2.3 Option Complexity & Length based on CEFR
                max_trans_len = 16 if cefr_upper in ("A1", "A2") else (24 if cefr_upper == "B1" else 36)
                for opt in options:
                    opt_words = str(opt).split()
                    if len(opt_words) > max_trans_len:
                        flagged_indices.add(idx)
                        defect_messages.append(
                            f"Translation Item #{idx + 1}: Translation option is overly long for CEFR {cefr_upper} ({len(opt_words)} words > {max_trans_len}): \"{opt[:40]}...\""
                        )
                        break

                # 2.4 Difficulty Ceiling (shared C1 helper: over_ceiling_tokens)
                for opt in options:
                    over = LinguisticEngine.over_ceiling_tokens(
                        str(opt), cefr_upper, mode="text", allow=meta_whitelist,
                    )
                    if over:
                        flagged_indices.add(idx)
                        tok, tok_lvl = over[0]
                        defect_messages.append(
                            f"Translation Item #{idx + 1}: Translation option contains obscure/super-advanced word '{tok}' ({tok_lvl}) exceeding CEFR {cefr_upper} ceiling: \"{opt[:40]}\""
                        )
                        break

                # 2.4b Single-Point Minimal-Pair Contract Gate (Comparative Appraisal)
                if is_comparative:
                    mp_sim = LinguisticEngine.calculate_minimal_pair_similarity(idiomatic_trans, flawed_trans)
                    min_sim_threshold = 0.85 if enforce_minimal_pair else 0.50
                    if mp_sim < min_sim_threshold:
                        flagged_indices.add(idx)
                        defect_messages.append(
                            f"Translation Item #{idx + 1}: Minimal-Pair contract violation — idiomatic_translation and "
                            f"flawed_translation share only {mp_sim:.0%} wording (< {min_sim_threshold:.0%} required)."
                        )

            # 2.5 Answer index range
            if not isinstance(correct_idx, int) or correct_idx not in range(expected_opt_count):
                flagged_indices.add(idx)
                defect_messages.append(
                    f"Translation Item #{idx + 1}: Invalid correct_answer_index ({correct_idx})"
                )

            skeleton = str(q_dict.get("english_skeleton", "")).strip()
            keyword = str(q_dict.get("target_keyword", "")).strip()

            # 2.4 Skeleton Slot Validation (if anchored skeleton mode)
            if skeleton and not is_comparative:
                if not re.search(r'\[\s*_{2,}\s*\]|_{3,}', skeleton):
                    flagged_indices.add(idx)
                    defect_messages.append(
                        f"Translation Item #{idx + 1}: 'english_skeleton' must contain exactly one '[ ____ ]' slot."
                    )

            # 3. Target Vocabulary Presence Gate (Active Usage Mandate)
            # Find declared target word from target_keyword, design_audit, or unit_headwords
            declared_word = keyword or None
            if not declared_word and audit_str:
                m_word = re.search(r'(?:Target Vocab(?:ulary)?|Target Word|Target Keyword)[\:\s\-]+([A-Za-z\s\-]+?)(?:\]|\+|\-\>|\n|$)', audit_str, re.IGNORECASE)
                if m_word:
                    declared_word = m_word.group(1).strip()
                else:
                    m_bracket = re.search(r'\[.*?->\s*([A-Za-z\s\-]+?)(?:\+|\,|\])', audit_str)
                    if m_bracket:
                        declared_word = m_bracket.group(1).strip()

            # If not in audit_str, search which unit_headword matches this item
            if not declared_word and unit_headwords:
                target_search_corpus = f"{correct_eng} {idiomatic_trans} {options[correct_idx]}" if (options and isinstance(correct_idx, int) and 0 <= correct_idx < len(options)) else f"{correct_eng} {idiomatic_trans}"
                for hw in unit_headwords:
                    if _word_in_text(hw, target_search_corpus):
                        declared_word = hw
                        break

            # If we know the target word, verify it is physically present in the correct translation
            if declared_word:
                clean_target = re.sub(r'[\(\[\{].*?[\)\]\}]', '', declared_word).strip()
                if clean_target and len(clean_target) > 2:
                    check_text = idiomatic_trans if is_comparative else (
                        f"{correct_eng} {options[correct_idx]}" if (options and isinstance(correct_idx, int) and 0 <= correct_idx < len(options)) else correct_eng
                    )
                    if not _word_in_text(clean_target, check_text):
                        flagged_indices.add(idx)
                        defect_messages.append(
                            f"Translation Item #{idx + 1}: Target vocabulary '{clean_target}' is declared "
                            f"but missing in correct translation/option: \"{check_text}\""
                        )

            # 4. Correct Answer Synchronization Gate
            if options and isinstance(correct_idx, int) and 0 <= correct_idx < len(options):
                chosen_opt = str(options[correct_idx]).strip()
                if is_comparative:
                    # In comparative appraisal, chosen_opt MUST match idiomatic_translation
                    if chosen_opt.lower() != idiomatic_trans.lower():
                        flagged_indices.add(idx)
                        defect_messages.append(
                            f"Translation Item #{idx + 1}: Correct option [{chr(65 + correct_idx)}] does not match idiomatic_translation."
                        )
                elif skeleton:
                    # In anchored skeleton mode:
                    if correct_eng and (chosen_opt.lower() not in correct_eng.lower()) and (chosen_opt.lower() != correct_eng.lower()):
                        filled_skeleton = re.sub(r'\[\s*_{2,}\s*\]|_{3,}', chosen_opt, skeleton)
                        if filled_skeleton.replace(" ", "").lower() != correct_eng.replace(" ", "").lower():
                            flagged_indices.add(idx)
                            defect_messages.append(
                                f"Translation Item #{idx + 1}: Correct option [{chr(65 + correct_idx)}] does not align with correct_english_answer."
                            )

                    # 4.1 Slot Stitching Duplication & Stutter Gate
                    stitched = re.sub(r'\[\s*_{2,}\s*\]|_{3,}', chosen_opt, skeleton)
                    clean_stitched = re.sub(r'[,\.\"\';:\?!]', ' ', stitched)
                    m_repeat = re.search(r'\b([a-zA-Z]+(?:\s+[a-zA-Z]+){1,3})\s+\1\b', clean_stitched, re.IGNORECASE)
                    if m_repeat:
                        flagged_indices.add(idx)
                        defect_messages.append(
                            f"Translation Item #{idx + 1}: Slot stitching duplication detected: repeated sequence '{m_repeat.group(0)}' in filled skeleton."
                        )
                else:
                    # Legacy full-sentence mode
                    if correct_eng and chosen_opt.lower() != correct_eng.lower():
                        match_found = False
                        for o_idx, opt in enumerate(options):
                            if str(opt).strip().lower() == correct_eng.lower():
                                match_found = True
                                break
                        if not match_found:
                            flagged_indices.add(idx)
                            defect_messages.append(
                                f"Translation Item #{idx + 1}: Declared correct_english_answer does not match any of the 4 options."
                            )

        return sorted(list(flagged_indices)), defect_messages
