"""
Multi-modal quiz generation (generate_quiz), Fisher-Yates option shuffling, and HTML handout rendering.
"""

import random
import os
import json
import re
import dataclasses
import base64
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any, Set
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

class QuizGeneratorMixin:

    def _shuffle_quiz_options(self, quiz_obj, preserve_blueprint_order: bool = False):
        """Randomizes the order of options for each question and updates the correct index."""
        return self.shuffle_quiz_options(quiz_obj, preserve_blueprint_order=preserve_blueprint_order)

    @staticmethod
    def _remap_explanation_labels(explanation: str, old_to_new: Dict[int, int]) -> str:
        """
        Safely remaps 'Option A/B/C/D', 'Choice 1/2/3/4', '(A)', '[B]' etc. in explanations
        when options are shuffled, avoiding swap collisions via atomic single-pass regex substitution.
        """
        if not explanation or not old_to_new:
            return explanation

        num_to_let = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
        word_to_num = {'one': 0, 'two': 1, 'three': 2, 'four': 3, 'first': 0, 'second': 1, 'third': 2, 'fourth': 3}
        let_to_num = {'a': 0, 'b': 1, 'c': 2, 'd': 3}

        # 1. Matches: Option A, Choice B, Option 1, Option One, etc.
        def replace_prefixed(m):
            prefix = m.group(1)
            raw_target = m.group(2).lower()
            old_idx = None
            if raw_target.isdigit():
                val = int(raw_target)
                if 1 <= val <= 4: old_idx = val - 1
                elif 0 <= val <= 3: old_idx = val
            elif raw_target in word_to_num:
                old_idx = word_to_num[raw_target]
            elif raw_target in let_to_num:
                old_idx = let_to_num[raw_target]

            if old_idx is not None and old_idx in old_to_new:
                return f"{prefix} {num_to_let[old_to_new[old_idx]]}"
            return m.group(0)

        # 2. Matches bracketed or parenthesized letters: (A), (B), [A], [B]
        def replace_bracketed(m):
            full_match = m.group(0)
            open_b = full_match[0]
            let = full_match[1].lower()
            close_b = full_match[2]
            if let in let_to_num and let_to_num[let] in old_to_new:
                return f"{open_b}{num_to_let[old_to_new[let_to_num[let]]]}{close_b}"
            return full_match

        res = re.sub(
            r'\b(Option|Choice)\s+([A-Da-d\d]|One|Two|Three|Four|First|Second|Third|Fourth)\b',
            replace_prefixed,
            explanation,
            flags=re.IGNORECASE
        )
        res = re.sub(r'(\([A-D]\)|\[[A-D]\])', replace_bracketed, res)
        return res

    @staticmethod
    def _sanitize_vocab_for_quiz(vocab_content: Any) -> Tuple[str, List[str], List[str]]:
        """
        Extracts authorized unit headwords and banned sentences for Level 1/Level 2 code gates:
        1. Accepts structured List[VocabularyItem | Dict] directly (or legacy markdown string for unit tests).
        2. Returns:
           - sanitized_content: Empty string for structured list inputs (blueprints format the prompt), or legacy markdown summary.
           - headwords: List of lowercase headword strings in this unit.
           - banned_sentences: List of the raw quoted and example sentences to audit stems against.
        """
        if not vocab_content:
            return "", [], []

        if isinstance(vocab_content, (list, tuple)):
            headwords: List[str] = []
            banned_sentences: List[str] = []
            for item in vocab_content:
                if isinstance(item, dict):
                    raw_w = str(item.get("word") or "").strip()
                    quote = str(item.get("quoted_sentence") or item.get("quote") or "").strip()
                    ex_usage = str(item.get("example_usage") or "").strip()
                else:
                    raw_w = str(getattr(item, "word", "") or "").strip()
                    quote = str(getattr(item, "quoted_sentence", "") or "").strip()
                    ex_usage = str(getattr(item, "example_usage", "") or "").strip()
                if not raw_w:
                    continue
                hw_clean = raw_w.lower()
                headwords.append(hw_clean)
                hw_stripped = re.sub(r'\[.*?\]|\(.*?\)', '', hw_clean)
                hw_stripped = re.sub(r'\s+', ' ', hw_stripped).strip()
                if hw_stripped and hw_stripped != hw_clean:
                    headwords.append(hw_stripped)
                if quote:
                    banned_sentences.append(quote)
                if ex_usage:
                    banned_sentences.append(ex_usage)
            return "", headwords, banned_sentences

        headwords = []
        for hw in re.findall(r'##\s*\[\[(.*?)\]\]', vocab_content):
            hw_clean = hw.strip().lower()
            if hw_clean:
                headwords.append(hw_clean)
                # Strip parenthetical or bracketed slot placeholders, e.g. "make (sth) possible" -> "make possible"
                hw_stripped = re.sub(r'\[.*?\]|\(.*?\)', '', hw_clean)
                hw_stripped = re.sub(r'\s+', ' ', hw_stripped).strip()
                if hw_stripped and hw_stripped != hw_clean:
                    headwords.append(hw_stripped)

        banned_sentences = []
        clean_lines = []
        for line in vocab_content.splitlines():
            m = re.match(r'^\s*-\s*\*\*(?:Quoted Sentence|Example Usage)\*\*:\s*(.*)', line, re.IGNORECASE)
            if m:
                sent = m.group(1).strip()
                if sent:
                    banned_sentences.append(sent)
            elif re.match(r'^\s*-\s*\*\*(?:Collocations|Grammar & Usage|Pattern|Thesaurus Note)\*\*:\s*(.*)', line, re.IGNORECASE):
                continue
            elif re.match(r'^\s{2,}\*\*', line):
                continue
            else:
                clean_lines.append(line)

        sanitized_content = "\n".join(clean_lines).strip()
        return sanitized_content, headwords, banned_sentences

    @classmethod
    def _sanitize_grammar_for_quiz(cls, grammar_content: Any) -> Tuple[str, List[str]]:
        """
        Physically isolates the quiz generator from grammar example sentences:
        Accepts structured List[GrammarItem | Dict] directly (or legacy markdown string for unit tests),
        preserving Pattern Formula, Pedagogical Function, and Common Mistakes while excluding
        Quote and Imitation Example.
        Also extracts a concise, high-density summary list of grammar patterns & common mistakes.
        """
        if not grammar_content:
            return "", []

        if isinstance(grammar_content, (list, tuple)):
            blocks: List[str] = []
            pattern_summaries: List[str] = []
            for item in grammar_content:
                if isinstance(item, dict):
                    raw_formula = str(item.get("pattern_formula") or "").strip()
                    ped_func = str(item.get("pedagogical_function") or "").strip()
                    mistakes = str(item.get("common_mistakes") or "").strip()
                    topic_name = str(item.get("syntax_topic") or item.get("name") or "").strip()
                else:
                    raw_formula = str(getattr(item, "pattern_formula", "") or "").strip()
                    ped_func = str(getattr(item, "pedagogical_function", "") or "").strip()
                    mistakes = str(getattr(item, "common_mistakes", "") or "").strip()
                    topic_name = str(getattr(item, "syntax_topic", "") or getattr(item, "name", "") or "").strip()
                norm_formula = cls.normalize_grammar_formula(raw_formula)
                name = topic_name or norm_formula or raw_formula or "Grammar Pattern"
                if ":" in name and not topic_name:
                    name = name.split(":", 1)[0].strip()
                blocks.append(
                    f"## [[{name}]]\n"
                    f"- **Pattern Formula**: {norm_formula}\n"
                    f"- **Pedagogical Function**: {ped_func}\n"
                    f"- **Common Mistakes**: {mistakes}"
                )
                summary = f"* {name}"
                if norm_formula:
                    summary += f": {norm_formula}"
                if mistakes:
                    summary += f" (Common Mistake: {mistakes})"
                pattern_summaries.append(summary)
            return "\n\n".join(blocks).strip(), pattern_summaries

        clean_lines = []
        pattern_summaries = []
        curr_name = ""
        curr_formula = ""
        curr_mistakes = ""

        def flush_pattern():
            nonlocal curr_name, curr_formula, curr_mistakes
            if curr_name:
                summary = f"* {curr_name}"
                if curr_formula:
                    summary += f": {curr_formula}"
                if curr_mistakes:
                    summary += f" (Common Mistake: {curr_mistakes})"
                pattern_summaries.append(summary)
            curr_name = ""
            curr_formula = ""
            curr_mistakes = ""

        for line in grammar_content.splitlines():
            m = re.match(r'^\s*-\s*\*\*(?:Quote|Imitation Example)\*\*:\s*(.*)', line, re.IGNORECASE)
            if not m:
                clean_lines.append(line)

            # Parse pattern header: ## [[Pattern Name]]
            m_header = re.match(r'^\s*##\s*\[\[([^\]]+)\]\]', line)
            if m_header:
                flush_pattern()
                curr_name = m_header.group(1).strip()
                continue

            # Parse Pattern Formula
            m_formula = re.match(r'^\s*-\s*\*\*Pattern Formula\*\*:\s*(.*)', line, re.IGNORECASE)
            if m_formula:
                raw_formula = m_formula.group(1).strip()
                curr_formula = cls.normalize_grammar_formula(raw_formula)
                continue

            # Parse Common Mistakes
            m_mistakes = re.match(r'^\s*-\s*\*\*Common Mistakes\*\*:\s*(.*)', line, re.IGNORECASE)
            if m_mistakes:
                curr_mistakes = m_mistakes.group(1).strip()
                continue
            elif curr_mistakes and (line.startswith("  ") or line.startswith("\t")):
                curr_mistakes += " " + line.strip()
                continue

        flush_pattern()

        return "\n".join(clean_lines).strip(), pattern_summaries

    @classmethod
    def hydrate_translation_blueprint_items(
        cls,
        quiz_obj: Any,
        blueprint_specs: List[Dict[str, Any]],
    ) -> Any:
        """
        Deterministically hydrates blueprint-owned fields on TranslationQuiz items:
        - Binds `target_keyword`, `target_grammar`, and `flaw_type` from `blueprint_specs`.
        - Constructs `options` from `[idiomatic_translation, flawed_translation]` ordered according
          to `prescribed_correct_index` (CRC32-locked 0 or 1).
        - Sets `correct_answer_index`, `correct_english_answer`, and hydrates `explanation` from
          `diagnostic_critique`.
        """
        if not quiz_obj or not blueprint_specs:
            return quiz_obj

        q_list = (
            quiz_obj.get("questions", [])
            if isinstance(quiz_obj, dict)
            else getattr(quiz_obj, "questions", [])
        )
        if not isinstance(q_list, list):
            return quiz_obj

        for q_i, q_item in enumerate(q_list):
            if q_i >= len(blueprint_specs):
                break
            spec = blueprint_specs[q_i] or {}
            is_dict = isinstance(q_item, dict)

            def _get(field: str, default: Any = "") -> Any:
                return q_item.get(field, default) if is_dict else getattr(q_item, field, default)

            def _set(field: str, val: Any) -> None:
                if is_dict:
                    q_item[field] = val
                else:
                    setattr(q_item, field, val)

            if spec.get("target_keyword"):
                _set("target_keyword", spec["target_keyword"])
            if spec.get("target_grammar"):
                _set("target_grammar", spec["target_grammar"])
            if spec.get("flaw_type"):
                _set("flaw_type", spec["flaw_type"])

            idiomatic = str(_get("idiomatic_translation", "") or "").strip()
            flawed = str(_get("flawed_translation", "") or "").strip()
            critique = str(_get("diagnostic_critique", "") or "").strip()

            if idiomatic and flawed:
                prescribed_idx = spec.get("prescribed_correct_index")
                if prescribed_idx is None:
                    prescribed_idx = spec.get("correct_answer_index")
                if prescribed_idx not in (0, 1):
                    import zlib
                    seed = f"{spec.get('target_keyword', '')}:{q_i}".encode("utf-8")
                    prescribed_idx = zlib.crc32(seed) % 2

                if prescribed_idx == 1:
                    opts = [flawed, idiomatic]
                else:
                    opts = [idiomatic, flawed]

                _set("options", opts)
                _set("correct_answer_index", prescribed_idx)
                _set("correct_english_answer", idiomatic)

            if critique:
                _set("explanation", critique)

        return quiz_obj

    @classmethod
    def shuffle_quiz_options(cls, quiz_obj: Any, preserve_blueprint_order: bool = False) -> Any:
        """
        Sanitizes options and ensures robust answer key distribution.
        When preserve_blueprint_order=True (for blueprint-backed vocabulary quizzes), CRC32-balanced
        option positions and indices are preserved intact without random re-shuffling or regex guessing.
        Otherwise, if answers are heavily biased (e.g. all 0s or >50% on a single index), options are shuffled
        and any references like 'Option A/B/C/D' in explanations are atomically remapped to the new indices.
        """
        import dataclasses
        from collections import Counter
        
        if not hasattr(quiz_obj, "questions") and not isinstance(quiz_obj, dict):
            return quiz_obj
        
        # Access questions (handle both dict and dataclass)
        questions = quiz_obj["questions"] if isinstance(quiz_obj, dict) else getattr(quiz_obj, "questions", None)
        if not questions or not isinstance(questions, list):
            return quiz_obj

        # Auto-flatten nested questions (e.g. if small models put a "questions": [...] array inside an item)
        flattened_questions = []
        for q_item in questions:
            if isinstance(q_item, dict):
                if "questions" in q_item and isinstance(q_item["questions"], list):
                    nested = q_item.pop("questions")
                    flattened_questions.append(q_item)
                    for nq in nested:
                        if isinstance(nq, dict):
                            flattened_questions.append(nq)
                else:
                    flattened_questions.append(q_item)
            else:
                flattened_questions.append(q_item)
        questions = flattened_questions

        # Filter out malformed items (e.g. naked strings, items missing options or with empty options)
        valid_questions = []
        for q in questions:
            if isinstance(q, dict):
                idiom_t = str(q.get("idiomatic_translation") or "").strip()
                flawed_t = str(q.get("flawed_translation") or "").strip()
                if idiom_t and flawed_t and len(q.get("options") or []) != 2:
                    q["options"] = [idiom_t, flawed_t]
                    q["correct_answer_index"] = 0
                opts = q.get("options")
                has_stem = q.get("question") or q.get("translated_sentence")
                if isinstance(opts, list) and len(opts) > 0 and has_stem:
                    valid_questions.append(q)
            else:
                idiom_t = str(getattr(q, "idiomatic_translation", "") or "").strip()
                flawed_t = str(getattr(q, "flawed_translation", "") or "").strip()
                if idiom_t and flawed_t and len(getattr(q, "options", None) or []) != 2:
                    q.options = [idiom_t, flawed_t]
                    q.correct_answer_index = 0
                opts = getattr(q, "options", None)
                has_stem = getattr(q, "question", None) or getattr(q, "translated_sentence", None)
                if isinstance(opts, list) and len(opts) > 0 and has_stem:
                    valid_questions.append(q)

        if not valid_questions:
            return quiz_obj
        if isinstance(quiz_obj, dict):
            quiz_obj["questions"] = valid_questions
        else:
            quiz_obj.questions = valid_questions
        questions = valid_questions

        # 1. Check answer index distribution across the entire quiz
        raw_indices = []
        for q in questions:
            q_dict = q if isinstance(q, dict) else (dataclasses.asdict(q) if dataclasses.is_dataclass(q) else {})
            idx = q_dict.get("correct_answer_index")
            try:
                raw_indices.append(int(idx))
            except (TypeError, ValueError):
                raw_indices.append(0)

        # Determine if answer distribution is biased:
        # For 4-option quizzes: biased if any answer accounts for > 50% or >= 2 choices unused.
        # For 2-option quizzes: biased if any single answer accounts for > 80% (extreme lopsidedness).
        is_biased = False
        if not preserve_blueprint_order:
            n_q = len(raw_indices)
            if n_q >= 3:
                counts = Counter(raw_indices)
                most_common_freq = counts.most_common(1)[0][1]
                unique_indices = set(raw_indices)
                max_choices = max([len(getattr(q, "options", []) or (q.get("options", []) if isinstance(q, dict) else [])) for q in questions] or [4])
                if max_choices <= 2:
                    if (most_common_freq / n_q) > 0.80:
                        is_biased = True
                else:
                    if (most_common_freq / n_q) > 0.50 or len(unique_indices) <= 2:
                        is_biased = True

        for q in questions:
            # Handle dict, dataclass, and generic object
            if isinstance(q, dict):
                q_dict = q
            elif dataclasses.is_dataclass(q):
                q_dict = dataclasses.asdict(q)
            elif hasattr(q, "__dict__"):
                q_dict = q.__dict__
            else:
                q_dict = {}
            
            # Sanitize string fields (options, target_word, word, correct_english_answer)
            quote_strip_pattern = r'^[«»"\'\u201c\u201d\u2018\u2019\s]+|[«»"\'\u201c\u201d\u2018\u2019\s]+$'
            label_strip_pattern = r'^(?:[A-Da-d\d][\.\)\:\-]\s*)'
            raw_options = q_dict.get("options", [])
            options = []
            for opt in raw_options:
                cleaned = re.sub(quote_strip_pattern, '', str(opt or ''))
                cleaned = re.sub(label_strip_pattern, '', cleaned).strip()
                cleaned = re.sub(quote_strip_pattern, '', cleaned)
                # Strip markdown bold/italic formatting e.g. **option** or *option*
                cleaned = re.sub(r'^\*+|\*+$', '', cleaned).strip()
                options.append(cleaned)
            
            for str_field in ["target_word", "word", "correct_english_answer"]:
                if str_field in q_dict and q_dict[str_field]:
                    clean_val = re.sub(quote_strip_pattern, '', str(q_dict[str_field]))
                    clean_val = re.sub(r'^\*+|\*+$', '', clean_val).strip()
                    # Strip accidental parenthesized or bracketed POS tags e.g. "tend (verb)" -> "tend"
                    if str_field in ("target_word", "word"):
                        clean_val = re.sub(r'[\(\[\{].*?[\)\]\}]', '', clean_val).strip()
                    if isinstance(q, dict):
                        q[str_field] = clean_val
                    else:
                        setattr(q, str_field, clean_val)

            # Sanitize explanation, definition, pedagogical_rationale fields: strip markdown bold/backtick artifacts like **Option B ("...")**
            for text_field in ["explanation", "definition", "pedagogical_rationale"]:
                if text_field in q_dict and q_dict[text_field]:
                    t_val = str(q_dict[text_field])
                    # Remove markdown asterisks and backticks from inline labels like **Option A ("...")**
                    clean_text = re.sub(r'[*`]', '', t_val)
                    # Strip LLM internal thinking leakage, author notes, or parenthetical remarks from definitions/explanations
                    # e.g., 'To fail to notice or consider. (Note: Using "overbook"...)' -> 'To fail to notice or consider.'
                    if text_field == "definition":
                        clean_text = re.sub(
                            r'\s*[\(\[\{]?(?:note|thinking|thought|author\'?s?\s*note|target\s*note)\s*:.*',
                            '',
                            clean_text,
                            flags=re.IGNORECASE
                        ).strip()
                        clean_text = clean_text.rstrip(')]};,. ')
                        if clean_text and clean_text[-1] not in ('.', '!', '?'):
                            clean_text += '.'
                    # Normalize double/triple spaces introduced by stripping
                    clean_text = re.sub(r'[ \t]+', ' ', clean_text).strip()
                    if isinstance(q, dict):
                        q[text_field] = clean_text
                    else:
                        setattr(q, text_field, clean_text)

            # Sanitize design_audit: strip accidental leading Note:/Thinking: prefixes
            if "design_audit" in q_dict and q_dict["design_audit"]:
                da_val = str(q_dict["design_audit"])
                da_clean = re.sub(
                    r'^(?:note|thinking|thought|audit\s*note)\s*:\s*',
                    '',
                    da_val,
                    flags=re.IGNORECASE
                ).strip()
                if isinstance(q, dict):
                    q["design_audit"] = da_clean
                else:
                    setattr(q, "design_audit", da_clean)

            # Resolve the ground-truth answer text when the schema exposes one, so we can
            # guarantee the declared correct index actually points at the right option.
            expected_answer = None
            for truth_field in ("idiomatic_translation", "correct_english_answer", "target_word", "word"):
                if truth_field in q_dict and q_dict[truth_field]:
                    expected_answer = q_dict[truth_field]
                    break

            # Anti-leak / Auto-masking: Ensure fill-in-the-blank questions actually contain a clean blank (____)
            # 1. Clean up models that output bolded blanks like **____** or *____*
            q_stem = q_dict.get("question") or q_dict.get("translated_sentence") or ""
            if q_stem:
                q_stem = re.sub(r'\*+(_{2,})\*+', r'\1', q_stem)
                if "question" in q_dict:
                    if isinstance(q, dict): q["question"] = q_stem
                    else: setattr(q, "question", q_stem)
                elif "translated_sentence" in q_dict:
                    if isinstance(q, dict): q["translated_sentence"] = q_stem
                    else: setattr(q, "translated_sentence", q_stem)

            # 2. If the LLM forgot to create a blank and directly wrote the full sentence containing the target word,
            # auto-mask the target word into '____'.
            if q_stem and expected_answer:
                has_blank = bool(re.search(r"_{2,}", q_stem))
                if not has_blank:
                    # Pattern matching target word as whole word, case-insensitive
                    esc_target = re.escape(expected_answer)
                    pattern = rf"\b{esc_target}\b"
                    if re.search(pattern, q_stem, re.IGNORECASE):
                        new_stem = re.sub(pattern, "____", q_stem, count=1, flags=re.IGNORECASE)
                        if "question" in q_dict:
                            if isinstance(q, dict):
                                q["question"] = new_stem
                            else:
                                setattr(q, "question", new_stem)
                        elif "translated_sentence" in q_dict:
                            if isinstance(q, dict):
                                q["translated_sentence"] = new_stem
                            else:
                                setattr(q, "translated_sentence", new_stem)
                        q_stem = new_stem

            # 3. Indefinite Article Leakage Auto-Healing:
            # If the stem has 'a ____' or 'an ____' immediately before the blank,
            # and the options have mixed vowel/consonant onsets (or would reveal/leak the answer),
            # auto-heal to neutral 'a(n) ____'.
            if q_stem and options and re.search(r"\b(?:a|an)\s+_{2,}\b", q_stem, flags=re.IGNORECASE):
                initials = {str(opt).strip()[:1].lower() for opt in options if str(opt).strip()}
                has_vowel_init = any(init in 'aeiou' for init in initials)
                has_cons_init = any(init.isalpha() and init not in 'aeiou' for init in initials)
                if has_vowel_init and has_cons_init:
                    # Auto-heal 'a ____' or 'an ____' to 'a(n) ____'
                    def _heal_art(m: re.Match) -> str:
                        prefix = m.group(1)
                        blank = m.group(2)
                        return f"A(n) {blank}" if prefix[0].isupper() else f"a(n) {blank}"

                    healed_stem = re.sub(r"\b(a|an)\s+(_{2,})\b", _heal_art, q_stem, flags=re.IGNORECASE)
                    if "question" in q_dict:
                        if isinstance(q, dict):
                            q["question"] = healed_stem
                        else:
                            setattr(q, "question", healed_stem)
                    elif "translated_sentence" in q_dict:
                        if isinstance(q, dict):
                            q["translated_sentence"] = healed_stem
                        else:
                            setattr(q, "translated_sentence", healed_stem)
                    q_stem = healed_stem

            declared_idx = q_dict.get("correct_answer_index")
            try:
                declared_idx = int(declared_idx)
            except (TypeError, ValueError):
                declared_idx = 0
            if options:
                declared_idx = max(0, min(declared_idx, len(options) - 1))

            # Enforce answer synchronization. LLMs occasionally desync the index from the
            # option it should point at; repair it here so a graded quiz is never wrong.
            index_was_repaired = False
            explanation_was_already_aligned = False
            matching_idx = None

            if expected_answer is not None and options:
                clean_exp = expected_answer.strip().lower()
                # 1. Direct option equality (vocabulary or full sentence options)
                for o_i, opt in enumerate(options):
                    if opt.strip().lower() == clean_exp:
                        matching_idx = o_i
                        break

                # 2. Anchored Skeleton Slot Completion matching:
                # If options are slot fragments and expected_answer is the full English sentence
                if matching_idx is None and "english_skeleton" in q_dict and q_dict["english_skeleton"]:
                    skel = str(q_dict["english_skeleton"]).strip()
                    norm_target = re.sub(r'[^\w]', '', clean_exp)
                    for o_i, opt in enumerate(options):
                        filled = re.sub(r'\[\s*_{2,}\s*\]|_{3,}', opt.strip(), skel)
                        if re.sub(r'[^\w]', '', filled.lower()) == norm_target:
                            matching_idx = o_i
                            break

                if matching_idx is not None:
                    correct_idx = matching_idx
                    if correct_idx != declared_idx:
                        index_was_repaired = True
                        logger.info(
                            f"Auto-healed index desync via expected answer/skeleton: repaired correct_answer_index "
                            f"from {declared_idx} to {matching_idx} ('{options[matching_idx]}')."
                        )

            # 3. Fallback heuristic: For non-blueprint quizzes (reading/listening/video), if expected answer
            # did not resolve matching_idx, detect if the explanation/audit declares a specific option letter.
            if not preserve_blueprint_order and matching_idx is None and options:
                cur_exp = q_dict.get("explanation", "")
                cur_audit = q_dict.get("design_audit", "")
                full_text = f"{cur_audit}\n{cur_exp}"
                
                inferred_letter = None
                # Check patterns like "Option D is the only form..." or "Option A correctly supplies..."
                m_lead_opt = re.search(
                    r'\bOption\s+([A-D])\b[^\.\n]*?\b(?:is\s+the\s+only|correctly\s+(?:supplies|matches|forms|completes|renders)|is\s+(?:the\s+)?correct\b)',
                    cur_exp,
                    re.IGNORECASE
                )
                if m_lead_opt:
                    inferred_letter = m_lead_opt.group(1).upper()
                else:
                    m_opt = re.search(
                        r'(?:(?:correct\s+answer\s+is|supporting|matches|aligned\s+with|only)\s+Option\s+([A-D])\b)',
                        full_text,
                        re.IGNORECASE
                    )
                    if m_opt:
                        inferred_letter = m_opt.group(1).upper()
                    else:
                        m_quote = re.search(r'correct\s+answer\s+is\s*[‘"\'“](.+?)[’”\'"]', cur_exp, re.IGNORECASE)
                        if m_quote:
                            quoted_txt = m_quote.group(1).strip().lower()
                            for o_i, opt in enumerate(options):
                                if opt.strip().lower() == quoted_txt or (len(opt) > 10 and opt.strip().lower() in quoted_txt):
                                    inferred_letter = chr(65 + o_i)
                                    break
                
                if inferred_letter:
                    inferred_idx = ord(inferred_letter) - 65
                    if 0 <= inferred_idx < len(options) and inferred_idx != declared_idx:
                        dec_letter = chr(65 + declared_idx)
                        disqualifies_declared = bool(re.search(
                            rf'Option\s+{dec_letter}\b.*?(?:contradict|absent|never|not\s+mentioned|fails|claim|incorrect|introduces|misuse|lacks|incompatible|noun\s+form)',
                            cur_exp,
                            re.IGNORECASE
                        ))
                        if disqualifies_declared or m_lead_opt:
                            correct_idx = inferred_idx
                            index_was_repaired = True
                            explanation_was_already_aligned = True
                            logger.info(
                                f"Auto-healed index desync in quiz item: repaired correct_answer_index "
                                f"from {declared_idx} ({dec_letter}) to {inferred_idx} ({inferred_letter}) "
                                f"based on explanation truth."
                            )
                        else:
                            correct_idx = declared_idx
                    else:
                        correct_idx = declared_idx
                else:
                    correct_idx = declared_idx
            elif matching_idx is None:
                correct_idx = declared_idx

            if not options:
                continue

            if is_biased:
                # Shuffle options and record old_idx -> new_idx mapping
                combined = list(zip(options, range(len(options))))
                random.shuffle(combined)
                
                new_options = [opt for opt, old_idx in combined]
                new_correct_idx = next((i for i, (opt, old_idx) in enumerate(combined) if old_idx == correct_idx), 0)
                old_to_new = {old_idx: new_idx for new_idx, (opt, old_idx) in enumerate(combined)}

                # Update explanation if present
                cur_exp = q_dict.get("explanation", "")
                if cur_exp:
                    remapped_exp = cls._remap_explanation_labels(cur_exp, old_to_new)
                    if isinstance(q, dict):
                        q["explanation"] = remapped_exp
                    else:
                        q.explanation = remapped_exp

                # Update options and correct_answer_index
                if isinstance(q, dict):
                    q["options"] = new_options
                    q["correct_answer_index"] = new_correct_idx
                else:
                    q.options = new_options
                    q.correct_answer_index = new_correct_idx
            else:
                # If options are not shuffled, but the index was repaired because LLM pointed to wrong slot,
                # remap the explanation to align with the repaired index (unless explanation was already aligned with correct_idx)
                if index_was_repaired and not explanation_was_already_aligned:
                    cur_exp = q_dict.get("explanation", "")
                    if cur_exp:
                        # Map old declared_idx to actual correct_idx bidirectionally to prevent label collision
                        repair_map = {declared_idx: correct_idx, correct_idx: declared_idx}
                        remapped_exp = cls._remap_explanation_labels(cur_exp, repair_map)
                        if isinstance(q, dict):
                            q["explanation"] = remapped_exp
                        else:
                            q.explanation = remapped_exp

                if isinstance(q, dict):
                    q["options"] = options
                    q["correct_answer_index"] = correct_idx
                else:
                    q.options = options
                    q.correct_answer_index = correct_idx
                
        return quiz_obj

    def generate_quiz(self, unit_name, count=10, template_name="vocabulary", audit_callback=None):
        """
        Generates a quiz handout based on extracted wiki data.
        Optional audit_callback(stage, progress, message, extra=None) for live dashboard progress.
        """
        # 1. Normalize unit_name to extract core unit folder name robustly
        parts = Path(unit_name).parts
        core_name = None
        if "wiki" in parts:
            idx = parts.index("wiki")
            if idx + 1 < len(parts):
                core_name = parts[idx + 1]
        elif "raw" in parts:
            idx = parts.index("raw")
            if idx + 1 < len(parts):
                core_name = parts[idx + 1]
        else:
            core_name = parts[0] if parts else unit_name
            
        if core_name.endswith(".md"): core_name = core_name[:-3]
        elif core_name.endswith(".txt"): core_name = core_name[:-4]
            
        for suffix in ["_vocabulary", "_grammar", "_concepts"]:
            if core_name.endswith(suffix):
                core_name = core_name[:-len(suffix)]

        # 2. Resolve Data
        data = self._load_wiki_data(core_name, template_name)
        if not data:
            return f"Error: Could not find extracted {template_name} data for {core_name}. If this is a fresh setup, please verify that Ollama or your LLM API is running, and check if the extraction pipeline ran successfully."

        # 3. Build Prompt
        prompt_template, schema_cls = Prompts.get(f"{template_name}_quiz")
        
        kwargs = {"count": count}
        unit_headwords: List[str] = []
        unit_grammar_patterns: List[str] = []
        banned_quiz_sentences: List[str] = []
        vocab_sense_map: Dict[str, str] = {}
        has_prescribed_options: bool = False
        blueprint_specs: List[Dict[str, Any]] = []

        if template_name == "vocabulary":
            raw_vocab = data.get("content", "")
            # Extract authorized unit headwords and banned authentic sentences for L1/L2 code gates
            _, unit_headwords, banned_quiz_sentences = self._sanitize_vocab_for_quiz(raw_vocab)

            # Achievement MCQ vs Proficiency MCQ Switch:
            # - Mode 'cloze' or enable_authentic_cloze: True -> Achievement MCQ (学业测试 / 0-token 真实原句完形)
            # - Mode 'generative' or enable_authentic_cloze: False -> Proficiency MCQ (能力测试 / LLM 语境迁移单选题)
            vocab_mode = self.config.get_quiz_config("vocabulary", "mode", "cloze")
            enable_cloze = (vocab_mode == "cloze") if isinstance(vocab_mode, str) else bool(vocab_mode)
            if enable_cloze:
                # Track 1: Authentic Passage Cloze (Achievement MCQ Mode)
                raw_cloze_items = LinguisticEngine.build_authentic_cloze_items(raw_vocab, target_count=count)
                effective_count = min(count, len(raw_cloze_items)) if raw_cloze_items else 0
                kwargs["count"] = effective_count
                logger.info(f"🎯 Built {len(raw_cloze_items)} authentic passage cloze skeletons (Achievement MCQ Mode, count capped at {effective_count}).")
                cloze_bullets = []
                for idx, c in enumerate(raw_cloze_items[:effective_count], 1):
                    tw_key = str(c.get("target_word") or "").strip().lower()
                    if tw_key and c.get("definition"):
                        vocab_sense_map[tw_key] = str(c["definition"]).strip()
                    pre_opts = c.get("precomputed_options", [])
                    if pre_opts:
                        has_prescribed_options = True
                    blueprint_specs.append({
                        "target_word": c.get("target_word"),
                        "definition": c.get("definition"),
                        "options": list(pre_opts) if pre_opts else None,
                        "correct_answer_index": c.get("correct_answer_index") if pre_opts else None,
                    })
                    opts_str = ", ".join(c.get("precomputed_distractors", []) or pre_opts)
                    cloze_bullets.append(
                        f"### Item {idx} ###\n"
                        f"- Target Word: {c['target_word']}\n"
                        f"- Part of Speech: {c['part_of_speech']}\n"
                        f"- Authentic Question Stem: {c['question']}\n"
                        f"- Candidate Distractors: [{opts_str}]\n"
                        f"- Contextual Definition: {c['definition']}"
                    )
                kwargs["vocabulary_content"] = (
                    "### AUTHENTIC PASSAGE CLOZE TARGETS (ACHIEVEMENT MCQ MODE) ###\n"
                    "For EACH item below, adopt its EXACT Target, Part of Speech, and Authentic Question Stem (do NOT invent new sentences or alter the stem). Formulate 3 plausible academic distractors (or use the candidate distractors) and craft the pedagogical explanation and design audit:\n\n"
                    + "\n\n".join(cloze_bullets)
                )
            else:
                # Track 2: Pre-Computed Generative Skeletons (Proficiency MCQ Mode)
                skeletons = LinguisticEngine.build_precomputed_target_skeletons(raw_vocab, target_count=count)
                effective_count = min(count, len(skeletons)) if skeletons else 0
                kwargs["count"] = effective_count
                has_prescribed_options = True
                logger.info(f"🎯 Built {len(skeletons)} pre-computed target skeletons (Proficiency MCQ Mode, count capped at {effective_count}).")
                skeleton_bullets = []
                for idx, s in enumerate(skeletons[:effective_count], 1):
                    tw_key = str(s.get("target_word") or "").strip().lower()
                    if tw_key and s.get("definition"):
                        vocab_sense_map[tw_key] = f"({s.get('part_of_speech', '')}) {s['definition']}".strip()
                    prescribed_opts = s.get("prescribed_options", [])
                    blueprint_specs.append({
                        "target_word": s.get("target_word"),
                        "definition": s.get("definition"),
                        "options": list(prescribed_opts) if prescribed_opts else None,
                        "correct_answer_index": s.get("correct_answer_index", 0) if prescribed_opts else None,
                    })
                    opts_str = ", ".join(prescribed_opts)
                    micro_task_str = s.get("micro_task", "")
                    pattern_line = f"- Pattern: {s['best_pattern']}\n" if s.get("best_pattern") else ""
                    skeleton_bullets.append(
                        f"### Item {idx} ###\n"
                        f"- Target Word: {s['target_word']}\n"
                        f"- Part of Speech: {s['part_of_speech']}\n"
                        f"- Contextual Definition: {s['definition']}\n"
                        + pattern_line
                        + f"- Prescribed Options: [{opts_str}]\n"
                        f"- Correct Answer Index: {s.get('correct_answer_index', 0)}\n"
                        f"- 🎯 Micro-Task for LLM: {micro_task_str}"
                    )

                kwargs["vocabulary_content"] = (
                    f"### TARGET SPECIFICATIONS ({len(skeleton_bullets)} ITEMS) ###\n\n"
                    + "\n\n".join(skeleton_bullets)
                )

            kwargs["cefr_level"] = data.get("cefr_level", "B2")
        elif template_name == "reading":
            kwargs["passage_content"] = data["passage"]
            cefr = str(data.get("cefr_level", "B2")).upper()
            kwargs["cefr_level"] = cefr
            is_foundational = cefr in ("A1", "A2", "B1")
            kwargs["is_foundational"] = is_foundational
            if cefr in ("A1", "A2"):
                kwargs["cefr_descriptor"] = f"Foundational English (CEFR {cefr})"
                kwargs["question_stem_guidance"] = (
                    "Use clear, direct, and accessible question stems (e.g., 'What did people use...', 'Why does the author mention...', 'According to paragraph 1...'). "
                    "Avoid dense academic vocabulary, complex passive inversions, or convoluted hypothetical framing in the question stems."
                )
                kwargs["option_complexity_guidance"] = (
                    "Keep each option concise (ideally 4 to 12 words) and structurally simple. "
                    "Options MUST NOT introduce obscure, advanced words (C1/C2) not found in the passage."
                )
                kwargs["skill_distribution_guidance"] = (
                    "     * `Detail/Recall`: 2 to 3 questions assessing key facts, explicit statements, or simple causes.\n"
                    "     * `Main Idea`: 1 to 2 questions assessing the overall topic or central message.\n"
                    "     * `Inference`: 0 to 1 question assessing a straightforward, obvious conclusion directly supported by the text."
                )
                kwargs["vocab_target_guidance"] = "5 to 8 key functional or topical vocabulary items essential for understanding the passage (e.g. core verbs, key nouns, useful everyday expressions)"
            elif cefr == "B1":
                kwargs["cefr_descriptor"] = "Intermediate English (CEFR B1)"
                kwargs["question_stem_guidance"] = (
                    "Use standard, clear question stems assessing paragraph connections, cause-and-effect, and key claims. "
                    "Maintain natural, accessible syntax."
                )
                kwargs["option_complexity_guidance"] = (
                    "Keep options standard and clear (ideally 6 to 15 words). Do not use obscure, hyper-specialized vocabulary in options."
                )
                kwargs["skill_distribution_guidance"] = (
                    "     * `Detail/Recall`: 2 questions targeting key factual statements or causal links.\n"
                    "     * `Main Idea`: 1 to 2 questions assessing paragraph or text-level main ideas.\n"
                    "     * `Inference`: 1 question assessing logical deductions warranted by the text.\n"
                    "     * `Author's Tone/Purpose`: 0 to 1 question assessing speaker attitude or purpose."
                )
                kwargs["vocab_target_guidance"] = "5 to 8 intermediate vocabulary items from the passage that expand students' communicative competence"
            else:
                kwargs["cefr_descriptor"] = f"Advanced Academic English (CEFR {cefr})"
                kwargs["question_stem_guidance"] = (
                    f"Use rigorous, standardized CEFR {cefr} reading assessment stems assessing synthesis, global discourse organization, and implicit logic."
                )
                kwargs["option_complexity_guidance"] = (
                    "Craft intellectually mature options with accurate paraphrasing, precise lexical substitutions, and nuanced distractors."
                )
                kwargs["skill_distribution_guidance"] = (
                    "     * `Main Idea`: 1 to 2 questions assessing global gist, central thesis, or primary communicative purpose.\n"
                    "     * `Detail/Recall`: 2 to 4 questions targeting key factual statements, causal links, or explicit mechanisms.\n"
                    "     * `Inference`: 1 to 3 questions assessing logical implications, unstated assumptions, or deductions fully warranted by the text.\n"
                    "     * `Author's Tone/Purpose`: 1 question assessing stance, attitude, rhetorical strategy, or underlying perspective."
                )
                kwargs["vocab_target_guidance"] = "5 to 8 challenging academic vocabulary items directly from the passage"
        elif template_name == "translation":
            raw_v = data.get("vocab_list", "")
            sanitized_v, unit_headwords, banned_quiz_sentences = self._sanitize_vocab_for_quiz(raw_v)
            raw_g = data.get("grammar_list", "")
            sanitized_g, unit_grammar_patterns = self._sanitize_grammar_for_quiz(raw_g)
            target_lang = self.config.get("target_language") or "Chinese"
            cefr = (data.get("cefr_level") or "B2").upper()
            kwargs["target_language"] = target_lang
            kwargs["cefr_level"] = cefr

            # Build Symbolic Translation Micro-Task Blueprints (Register-Aligned Pairing + Minimal-Pair Corruption Spec)
            translation_blueprints = LinguisticEngine.build_translation_micro_task_blueprints(
                raw_v,
                grammar_content=raw_g,
                target_count=count,
                target_language=target_lang,
                cefr_level=cefr,
            )
            if translation_blueprints:
                effective_count = min(count, len(translation_blueprints))
                kwargs["count"] = effective_count
                has_prescribed_options = True
                blueprint_specs = list(translation_blueprints[:effective_count])
                kwargs["vocabulary_content"] = "\n\n".join(
                    bp["micro_task_block"] for bp in blueprint_specs if bp.get("micro_task_block")
                )
                kwargs["grammar_content"] = sanitized_g or "Included in each Item Specification above."
                logger.info(
                    f"🎯 Built {len(blueprint_specs)} symbolic translation micro-task blueprints "
                    f"(Register-Aligned Pairing + Minimal-Pair Corruption Operators)."
                )
            else:
                kwargs["vocabulary_content"] = sanitized_v
                kwargs["grammar_content"] = sanitized_g
                if unit_headwords and unit_grammar_patterns:
                    kwargs["count"] = min(count, len(unit_headwords), len(unit_grammar_patterns))
                elif unit_headwords:
                    kwargs["count"] = min(count, len(unit_headwords))

            if cefr in ("A1", "A2"):
                kwargs["sentence_complexity_guidance"] = (
                    "Simple, clear, and direct everyday or classroom situations (8 to 14 English words). "
                    "Use basic coordination (and, but) or single simple clauses. Strictly avoid dense subordinate embedding or inversion."
                )
                kwargs["target_grammar_guidance"] = (
                    "Focus strictly on foundational morphosyntax: singular/plural concord, basic tense forms (simple present/past), "
                    "pronoun cases, and essential prepositions (in, at, on, with, for)."
                )
                kwargs["flaw_taxonomy_guidance"] = (
                    "     * *Trap 1 (L1 Negative Transfer & Missing Dummy Subjects)*: Direct literal word-for-word translation, missing 'There is/are' or 'It is' dummy subjects, or stacking multiple verbs without conjunctions.\n"
                    "     * *Trap 2 (Preposition & Function Word Loss)*: Omitting or confusing everyday prepositions (e.g. *listen music* instead of *listen to music*, *arrive to* instead of *arrive at*).\n"
                    "     * *Trap 3 (Basic Agreement & Inflection Breakdown)*: Missing third-person singular '-s', bare infinitive after preposition, or plural suffix disagreement."
                )
            elif cefr == "B1":
                kwargs["sentence_complexity_guidance"] = (
                    "Standard compound/complex sentences (12 to 20 English words) describing common social, educational, or work situations. "
                    "Incorporate common adverbial clauses (because, although, if, when) or direct relative clauses."
                )
                kwargs["target_grammar_guidance"] = (
                    "Intermediate grammatical patterns: modal verbs (should, must, might), passive voice, "
                    "gerunds vs infinitives, and straightforward coordinating/subordinating clause links."
                )
                kwargs["flaw_taxonomy_guidance"] = (
                    "     * *Trap 1 (L1 Negative Transfer & Conjunction Doubling)*: Using correlative doubles like 'Although... but...' or misplaced adverbial adjuncts.\n"
                    "     * *Trap 2 (Collocation & Dependent Prepositions)*: Dependent preposition errors (e.g. *depend of* instead of *depend on*, *comply to* instead of *comply with*).\n"
                    "     * *Trap 3 (Voice & Aspect Inflection Errors)*: Incomplete passive voice formulas or incorrect participle forms (e.g. *is break* instead of *is broken*)."
                )
            else:
                kwargs["sentence_complexity_guidance"] = (
                    "Intellectually mature, formal compound-complex sentences (18 to 30 English words) involving academic analysis, "
                    "cause-and-effect reasoning, or professional discourse."
                )
                kwargs["target_grammar_guidance"] = (
                    "Advanced academic structures: fronted inversion, non-finite participial clauses, cleft sentences, "
                    "abstract shell noun complements, or concessive refutations."
                )
                kwargs["flaw_taxonomy_guidance"] = (
                    "     * *Trap 1 (L1 Negative Transfer & Chinglish Syntax)*: Mechanically translating word order, resulting in missing dummy subjects, awkward topic-prominent structures, or verb stacking.\n"
                    "     * *Trap 2 (Collocation & Preposition Clash)*: Misusing formal dependent prepositions or verb-noun collocations (e.g. *comply to* instead of *comply with*, *make a damage*).\n"
                    "     * *Trap 3 (Morpho-syntactic & Formula Breakdown)*: Distorting the target formula or non-finite verb morphology (e.g. failed subject-auxiliary inversion, dangling participle)."
                )

        elif template_name == "listening":
            raw_v = data.get("vocab_list", "")
            sanitized_v, unit_headwords, banned_quiz_sentences = self._sanitize_vocab_for_quiz(raw_v)
            kwargs["vocabulary_content"] = sanitized_v
            cefr = (data.get("cefr_level") or "B2").upper()
            kwargs["cefr_level"] = cefr

            if cefr in ("A1", "A2"):
                kwargs["dialogue_style_guidance"] = (
                    "Conversational, everyday campus or life interactions (e.g. discussing schedules, course plans, or collaborative assignments). "
                    "Keep speaker turns short and clear (8 to 15 words per turn), using accessible phrasing and concrete dialogue markers."
                )
                kwargs["question_stem_guidance"] = (
                    "Use direct, unambiguous question stems (e.g. 'What does Speaker 1 want to do?', 'Why is Speaker 2 calling?'). "
                    "Avoid complex speculative or hypothetical framing."
                )
                kwargs["option_complexity_guidance"] = (
                    "Keep options concise, simple, and direct (ideally 3 to 8 words). Never use obscure academic vocabulary."
                )
                kwargs["skill_distribution_guidance"] = (
                    "     * `Detail`: 2 to 3 questions assessing concrete facts, explicitly stated intentions, or direct causes.\n"
                    "     * `Main Idea`: 1 to 2 questions assessing the overall topic or mutual agreement.\n"
                    "     * `Inference`: 0 to 1 question assessing a straightforward, obvious conclusion."
                )
            elif cefr == "B1":
                kwargs["dialogue_style_guidance"] = (
                    "Standard academic or semi-formal exchange (e.g. tutorial discussion, study group analysis). "
                    "Turns should be moderately detailed (14 to 22 words per turn), presenting reasoning and practical opinions."
                )
                kwargs["question_stem_guidance"] = (
                    "Use clear question stems assessing explicit facts, speaker reasoning, and immediate deductions."
                )
                kwargs["option_complexity_guidance"] = (
                    "Keep options standard and clear (ideally 5 to 12 words), with natural everyday vocabulary."
                )
                kwargs["skill_distribution_guidance"] = (
                    "     * `Detail`: 2 questions assessing specific rationale or facts stated by a speaker.\n"
                    "     * `Main Idea`: 1 to 2 questions assessing the core theme or conclusion.\n"
                    "     * `Inference`: 1 question assessing speaker attitude or logical takeaway."
                )
            else:
                kwargs["dialogue_style_guidance"] = (
                    "Intellectually engaging academic debate or seminar discussion. "
                    "Turns are detailed and articulated (18 to 32 words per turn), featuring subtle distinctions, counter-arguments, and nuanced concessions."
                )
                kwargs["question_stem_guidance"] = (
                    "Use rigorous assessment stems testing speaker stance, implicit qualifications, and analytical deductions."
                )
                kwargs["option_complexity_guidance"] = (
                    "Craft nuanced, intellectually mature options (8 to 16 words) with precise paraphrasing."
                )
                kwargs["skill_distribution_guidance"] = (
                    "     * `Detail`: 1 to 2 questions on key nuances, qualifying conditions, or exact rationales.\n"
                    "     * `Inference`: 2 to 3 questions drawing deductions warranted by tone, stance, or logic.\n"
                    "     * `Main Idea`: 1 question assessing the overarching academic consensus or disagreement."
                )

            # Extract only Core Concepts 1, 2, 3 from Summary
            summary_txt = data.get("summary_content", "")
            core_unit = data.get("unit_core_name", core_name)
            
            thematic_concepts = []
            if summary_txt:
                concept_matches = re.findall(r'###\s*\[\[(.*?)\]\]', summary_txt)
                if not concept_matches:
                    concept_matches = re.findall(r'###\s*([^\n]+)', summary_txt)
                if concept_matches:
                    clean_concepts = [
                        c.strip() for c in concept_matches 
                        if c.strip() and not c.strip().lower().startswith("narrative")
                    ]
                    for idx, c in enumerate(clean_concepts[:3], 1):
                        thematic_concepts.append(f"{idx}. {c}")

            if not thematic_concepts:
                thematic_concepts.append(f"1. {core_unit.replace('_', ' ')}")

            thematic_topic_str = "\n".join(thematic_concepts)
            kwargs["thematic_topic"] = thematic_topic_str
            
            # Retrieve tts configurations to get genders & accents for Schema-First Live Injection
            from ..tts import tts_service
            tts_service._refresh()
            
            def get_voice_gender(voice_name):
                v = str(voice_name or "").lower()
                if any(v.startswith(p) for p in ["af_", "bf_"]):
                    return "female"
                if any(v.startswith(p) for p in ["am_", "bm_"]):
                    return "male"
                if any(x in v for x in ["aria", "jenny", "sonia", "libby"]):
                    return "female"
                if any(x in v for x in ["guy", "christopher", "ryan", "thomas"]):
                    return "male"
                return "female"
                
            def get_voice_accent(voice_name):
                v_lower = str(voice_name or "").lower()
                if "gb" in v_lower or "bf_" in v_lower or "bm_" in v_lower or "sonia" in v_lower:
                    return "Accent: British"
                if "us" in v_lower or "af_" in v_lower or "am_" in v_lower or "aria" in v_lower or "michael" in v_lower or "guy" in v_lower:
                    return "Accent: American"
                if "cn" in v_lower or "zh" in v_lower or "xiaoxiao" in v_lower or "yunxi" in v_lower:
                    return "Accent: Chinese (Mandarin)"
                return "Accent: Standard"

            g1 = get_voice_gender(tts_service.voice_a)
            g2 = get_voice_gender(tts_service.voice_b)
            a1 = get_voice_accent(tts_service.voice_a)
            a2 = get_voice_accent(tts_service.voice_b)

            kwargs["speaker_1_gender"] = g1
            kwargs["speaker_2_gender"] = g2
            kwargs["speaker_1_accent"] = a1
            kwargs["speaker_2_accent"] = a2

        elif template_name == "video":
            kwargs["transcript_content"] = data["transcript"]
            kwargs["video_url"] = data["video_url"]
            kwargs["video_type"] = data["video_type"]
            cefr = (data.get("cefr_level") or "B2").upper()
            kwargs["cefr_level"] = cefr

            if cefr in ("A1", "A2"):
                kwargs["question_depth_guidance"] = (
                    "Focus on direct, concrete observations and simple facts explicitly described in the video at the timestamp "
                    "(e.g. 'What is shown at [MM:SS]?', 'What problem does the speaker describe?'). Avoid complex synthesis across long segments."
                )
                kwargs["option_complexity_guidance"] = (
                    "Keep options short and straightforward (ideally 4 to 10 words). Avoid dense academic terminology in options."
                )
            elif cefr == "B1":
                kwargs["question_depth_guidance"] = (
                    "Focus on clear causal relationships, operational mechanisms, and primary challenges discussed at the timestamp "
                    "(e.g. 'Why was this method chosen?', 'How does the process function?')."
                )
                kwargs["option_complexity_guidance"] = (
                    "Keep options clear and standard (ideally 6 to 14 words), using accessible technical vocabulary."
                )
            else:
                kwargs["question_depth_guidance"] = (
                    "Focus on technical mechanisms, comparative evaluations, underlying controversies, or future projections discussed at the timestamp."
                )
                kwargs["option_complexity_guidance"] = (
                    "Craft intellectually mature options (8 to 18 words) with nuanced distractors and academic phrasing."
                )

        prompt = prompt_template.format(**kwargs)

        # 4. Call LLM (Direct One-Shot Structured Output)
        try:
            interpolated_schema = self._interpolate_schema(schema_cls, kwargs)
            quiz_obj = llm.chat(
                [{"role": "user", "content": prompt}],
                schema=interpolated_schema,
                task_name=f"quiz_{template_name}_{core_name}"
            )

            if not quiz_obj:
                return "Error: LLM returned empty quiz data."

            # 4.4. Deterministic Blueprint Hydration: bind blueprint-owned fields
            # (target_word/target_keyword, definition, target_grammar, flaw_type, options, correct_answer_index) directly from code
            if blueprint_specs:
                if template_name == "translation":
                    quiz_obj = self.hydrate_translation_blueprint_items(quiz_obj, blueprint_specs)
                else:
                    q_list = quiz_obj.get("questions", []) if isinstance(quiz_obj, dict) else getattr(quiz_obj, "questions", [])
                    if isinstance(q_list, list):
                        for q_i, q_item in enumerate(q_list):
                            if q_i >= len(blueprint_specs):
                                break
                            spec = blueprint_specs[q_i]
                            if isinstance(q_item, dict):
                                if spec.get("target_word"):
                                    q_item["target_word"] = spec["target_word"]
                                if spec.get("definition"):
                                    q_item["definition"] = spec["definition"]
                                if spec.get("options"):
                                    q_item["options"] = list(spec["options"])
                                if spec.get("correct_answer_index") is not None:
                                    q_item["correct_answer_index"] = spec["correct_answer_index"]
                            else:
                                if spec.get("target_word"):
                                    setattr(q_item, "target_word", spec["target_word"])
                                if spec.get("definition"):
                                    setattr(q_item, "definition", spec["definition"])
                                if spec.get("options"):
                                    setattr(q_item, "options", list(spec["options"]))
                                if spec.get("correct_answer_index") is not None:
                                    setattr(q_item, "correct_answer_index", spec["correct_answer_index"])

            # 4.5. Randomize options (skip when blueprint prescribes CRC32-balanced options)
            quiz_obj = self._shuffle_quiz_options(quiz_obj, preserve_blueprint_order=has_prescribed_options)

            # Deterministically attach physical CEFR level to quiz_obj
            cefr_val = str(data.get("cefr_level") or kwargs.get("cefr_level") or "B2").upper()
            if isinstance(quiz_obj, dict):
                quiz_obj["cefr_level"] = cefr_val
            else:
                try:
                    setattr(quiz_obj, "cefr_level", cefr_val)
                except Exception:
                    pass

            if template_name == "reading":
                if isinstance(quiz_obj, dict):
                    quiz_obj["passage"] = data["passage"]
                else:
                    quiz_obj.passage = data["passage"]

            if template_name == "video":
                if isinstance(quiz_obj, dict):
                    quiz_obj["video_url"] = data["video_url"]
                    quiz_obj["video_type"] = data["video_type"]
                    quiz_obj["transcript"] = data["transcript"]
                else:
                    quiz_obj.video_url = data["video_url"]
                    quiz_obj.video_type = data["video_type"]
                    quiz_obj.transcript = data["transcript"]

            # 4.6. Two-Level Auditing: Pre-Audit L1 Code Gate & L2 Expert In-Place Surgical Cure
            if self.config.get("enable_expert_audit", False):
                try:
                    from ..expert_auditor import ExpertAuditor
                    source_context = ""
                    if template_name == "reading":
                        source_context = data.get("passage", "")
                    elif template_name == "video":
                        source_context = data.get("transcript", "")
                    elif template_name == "translation":
                        # For translation, syllabus pools (vocabulary list & grammar patterns with common mistakes)
                        # are passed directly into the Expert Auditor's structured gates.
                        # We do NOT dump redundant raw dictionary definitions into source_context.
                        source_context = ""
                    elif template_name == "listening":
                        script_turns = []
                        raw_script = quiz_obj.get("script") if isinstance(quiz_obj, dict) else getattr(quiz_obj, "script", [])
                        if isinstance(raw_script, list):
                            for turn in raw_script:
                                if isinstance(turn, dict):
                                    spk = turn.get("speaker", "Speaker")
                                    txt = turn.get("text", "")
                                    script_turns.append(f"{spk}: {txt}")
                                elif hasattr(turn, "speaker") and hasattr(turn, "text"):
                                    script_turns.append(f"{turn.speaker}: {turn.text}")
                        source_context = "\n".join(script_turns)
                    elif template_name == "vocabulary":
                        # For vocabulary quizzes, each item must be a standalone sentence test.
                        # We intentionally DO NOT supply the vocabulary dictionary definitions/quoted sentences
                        # to the blind solver, ensuring the judge evaluates pure sentence-level single-fit validity without key-leaks.
                        source_context = ""

                    quiz_dict_eval = dataclasses.asdict(quiz_obj) if dataclasses.is_dataclass(quiz_obj) else quiz_obj
                    if template_name == "vocabulary" and vocab_sense_map and isinstance(quiz_dict_eval, dict):
                        for q_item in quiz_dict_eval.get("questions", []):
                            if isinstance(q_item, dict):
                                tw_q = str(q_item.get("target_word") or "").strip().lower()
                                if tw_q in vocab_sense_map and not q_item.get("contextual_sense"):
                                    q_item["contextual_sense"] = vocab_sense_map[tw_q]

                    # ---------------------------------------------------------
                    # Step 1: Pre-Audit Level 1 Code Gate (Sanitize & Surface Flaws)
                    # ---------------------------------------------------------
                    curr_tgt_lang = kwargs.get("target_language") or self.config.get("target_language") or "Chinese"
                    if template_name == "reading":
                        l1_defective, l1_defects = self.audit_reading_integrity(
                            quiz_dict_eval,
                            passage_text=source_context,
                            cefr_level=kwargs.get("cefr_level", "B2")
                        )
                    elif template_name == "video":
                        l1_defective, l1_defects = self.audit_video_integrity(
                            quiz_dict_eval,
                            transcript_text=source_context,
                            cefr_level=kwargs.get("cefr_level", "B2")
                        )
                    elif template_name == "listening":
                        l1_defective, l1_defects = self.audit_listening_integrity(
                            quiz_dict_eval,
                            script_text=source_context,
                            cefr_level=kwargs.get("cefr_level", "B2"),
                            unit_headwords=unit_headwords
                        )
                    elif template_name == "translation":
                        l1_defective, l1_defects = self.audit_translation_integrity(
                            quiz_dict_eval,
                            unit_headwords=unit_headwords,
                            target_language=curr_tgt_lang,
                            cefr_level=kwargs.get("cefr_level", "B2"),
                            enforce_minimal_pair=bool(blueprint_specs),
                        )
                    else:
                        is_cloze_mode = (template_name == "vocabulary" and self.config.get_quiz_config("vocabulary", "mode", "cloze") == "cloze")
                        l1_defective, l1_defects = self.audit_quiz_integrity(
                            quiz_dict_eval,
                            banned_sentences=banned_quiz_sentences,
                            unit_headwords=unit_headwords,
                            allow_authentic_cloze=is_cloze_mode,
                            strict_distractor_recycling=not has_prescribed_options,
                        )

                    if l1_defects:
                        logger.warning(
                            f"Pre-Audit Level 1 Code Gate flagged {len(l1_defects)} issue(s) before Level 2: {l1_defects}"
                        )

                    # ---------------------------------------------------------
                    # Step 2: Level 2 Expert Model Quality Audit & Triage
                    # ---------------------------------------------------------
                    if audit_callback:
                        try:
                            audit_callback(
                                stage="auditing",
                                audit_progress=30,
                                message="🔍 Level 2 Expert Audit: Running psychometric blind evaluation & surgical triage...",
                                extra={"attempt": 1, "max_retries": 1}
                            )
                        except Exception:
                            pass

                    judge_model_to_use = ExpertAuditor.get_judge_model()
                    active_gen_model = llm.model

                    # If generator model and judge model are distinct, unload generator from Ollama to protect VRAM
                    if judge_model_to_use and active_gen_model and judge_model_to_use != active_gen_model:
                        llm.unload_model(active_gen_model)

                    audit_report = ExpertAuditor.audit_quiz(
                        source_context,
                        quiz_dict_eval,
                        judge_model=judge_model_to_use,
                        quiz_type=template_name,
                        target_language=curr_tgt_lang,
                        unit_headwords=unit_headwords,
                        unit_grammar_patterns=unit_grammar_patterns,
                        cefr_level=kwargs.get("cefr_level", "B2")
                    )

                    # If distinct models, unload judge model to free memory
                    if judge_model_to_use and active_gen_model and judge_model_to_use != active_gen_model:
                        llm.unload_model(judge_model_to_use)

                    if audit_report:
                        # Record raw generation score before cure
                        raw_gen_score = audit_report.get("overall_quality_score", 100)
                        audit_report["raw_generation_score"] = raw_gen_score

                        orig_questions = quiz_dict_eval.get("questions", [])
                        audit_items = audit_report.get("questions", [])
                        cured_questions = list(orig_questions)

                        cure_stats = {"pass": 0, "repair": 0, "rewrite": 0, "discard": 0}

                        # ---------------------------------------------------------
                        # Step 3: In-Place Surgical Cure & Post-Cure L1 Verification
                        # ---------------------------------------------------------
                        for a_idx, qa in enumerate(audit_items):
                            item_slot = qa.get("item_index", a_idx + 1) - 1
                            if not (0 <= item_slot < len(cured_questions)):
                                item_slot = a_idx
                            if not (0 <= item_slot < len(cured_questions)):
                                continue

                            triage = str(qa.get("triage_action", "PASS")).upper()
                            candidate_cure = qa.get("cured_question")

                            if triage in ("REPAIR", "REWRITE") and isinstance(candidate_cure, dict) and candidate_cure:
                                # Modality-specific and defensive pre-normalization for candidate cure
                                for q_alias in ("stem", "revised_question", "new_question", "cured_stem", "revised_stem"):
                                    if q_alias in candidate_cure and not candidate_cure.get("question"):
                                        candidate_cure["question"] = candidate_cure.pop(q_alias)
                                for opt_alias in ("revised_options", "new_options", "cured_options"):
                                    if opt_alias in candidate_cure and not candidate_cure.get("options"):
                                        candidate_cure["options"] = candidate_cure.pop(opt_alias)
                                for idx_alias in ("revised_correct_answer_index", "new_correct_answer_index"):
                                    if idx_alias in candidate_cure and candidate_cure.get("correct_answer_index") is None:
                                        candidate_cure["correct_answer_index"] = candidate_cure.pop(idx_alias)
                                for exp_alias in ("revised_explanation", "new_explanation"):
                                    if exp_alias in candidate_cure and not candidate_cure.get("explanation"):
                                        candidate_cure["explanation"] = candidate_cure.pop(exp_alias)

                                # Normalize options if LLM outputted list of dicts e.g. [{"letter": "A", "text": "..."}]
                                raw_cand_opts = candidate_cure.get("options")
                                if isinstance(raw_cand_opts, list):
                                    normalized_opts = []
                                    cand_corr_idx = candidate_cure.get("correct_answer_index")
                                    for o_i, o_val in enumerate(raw_cand_opts):
                                        if isinstance(o_val, dict):
                                            txt = o_val.get("text") or o_val.get("option_text") or o_val.get("sentence") or ""
                                            if o_val.get("is_correct") is True and cand_corr_idx is None:
                                                cand_corr_idx = o_i
                                        else:
                                            txt = str(o_val)
                                        if template_name == "vocabulary":
                                            txt = re.sub(r'\s*\([^)]*\)', '', str(txt)).strip()
                                        else:
                                            txt = re.sub(
                                                r'\s*\((?:synonym|antonym|distractor|correct|key|target|meaning|i\.e\.|e\.g\.)\b[^)]*\)',
                                                '',
                                                str(txt),
                                                flags=re.IGNORECASE
                                            ).strip()
                                        normalized_opts.append(txt)
                                    candidate_cure["options"] = normalized_opts
                                    if cand_corr_idx is not None:
                                        try:
                                            candidate_cure["correct_answer_index"] = int(cand_corr_idx)
                                        except (ValueError, TypeError):
                                            pass

                                if isinstance(candidate_cure.get("explanation"), str):
                                    candidate_cure["explanation"] = re.sub(
                                        r'\s*(?:This\s+(?:rewrite|revision|item|stem|change)\s+(?:ensures|eliminates|prevents|fixes|resolves)[^.\n]*\.?)+\s*$',
                                        '',
                                        candidate_cure["explanation"],
                                        flags=re.IGNORECASE
                                    ).strip()

                                orig_item = orig_questions[item_slot] if 0 <= item_slot < len(orig_questions) else {}
                                orig_dict = orig_item if isinstance(orig_item, dict) else (dataclasses.asdict(orig_item) if dataclasses.is_dataclass(orig_item) else {})

                                # Vocabulary fallback: inherit question / target_word / definition / explanation if omitted in cure
                                if template_name == "vocabulary":
                                    if not candidate_cure.get("question") and orig_dict.get("question") and triage == "REPAIR":
                                        candidate_cure["question"] = orig_dict["question"]
                                    if not candidate_cure.get("explanation") and orig_dict.get("explanation"):
                                        candidate_cure["explanation"] = orig_dict["explanation"]
                                    # For REPAIR, target word was valid, so inheritance is safe.
                                    # For REWRITE, only inherit if target_word was omitted AND original target belongs to authorized unit headwords.
                                    if not candidate_cure.get("target_word"):
                                        orig_tw = orig_dict.get("target_word")
                                        if orig_tw:
                                            if triage == "REPAIR":
                                                candidate_cure["target_word"] = orig_tw
                                            elif unit_headwords:
                                                hw_set = {h.strip().lower() for h in unit_headwords if h.strip()}
                                                if orig_tw.strip().lower() in hw_set:
                                                    candidate_cure["target_word"] = orig_tw
                                    if not candidate_cure.get("definition") and orig_dict.get("definition"):
                                        candidate_cure["definition"] = orig_dict["definition"]

                                # Reading fallback: inherit category if omitted in cure
                                if template_name == "reading":
                                    if not candidate_cure.get("category") and orig_dict.get("category"):
                                        candidate_cure["category"] = orig_dict["category"]

                                # Video fallback: inherit timestamp if omitted in cure
                                if template_name == "video":
                                    if not candidate_cure.get("timestamp") and orig_dict.get("timestamp"):
                                        candidate_cure["timestamp"] = orig_dict["timestamp"]

                                # Listening fallback: inherit category if omitted in cure
                                if template_name == "listening":
                                    if not candidate_cure.get("category") and orig_dict.get("category"):
                                        candidate_cure["category"] = orig_dict["category"]

                                if template_name == "translation":
                                    cand_opts = candidate_cure.get("options")
                                    corr_idx = candidate_cure.get("correct_answer_index", 0)
                                    extracted_opts = []
                                    if isinstance(cand_opts, list) and len(cand_opts) == 2:
                                        extracted_opts = [str(o) for o in cand_opts]
                                    elif candidate_cure.get("idiomatic_translation") and candidate_cure.get("flawed_translation"):
                                        extracted_opts = [
                                            candidate_cure["idiomatic_translation"],
                                            candidate_cure["flawed_translation"]
                                        ]
                                        corr_idx = 0

                                    if len(extracted_opts) == 2:
                                        # Clean option prefixes e.g. "A) " or "B) "
                                        label_strip = r'^(?:[A-Da-d\d][\.\)\:\-]\s*)'
                                        clean_cand_opts = [re.sub(label_strip, '', str(o)).strip() for o in extracted_opts]
                                        candidate_cure["options"] = clean_cand_opts
                                        if not isinstance(corr_idx, int) or corr_idx not in (0, 1):
                                            corr_idx = 0
                                        candidate_cure["correct_answer_index"] = corr_idx
                                        # Always synchronize idiomatic_translation and flawed_translation with clean options
                                        candidate_cure["idiomatic_translation"] = clean_cand_opts[corr_idx]
                                        candidate_cure["flawed_translation"] = clean_cand_opts[1 - corr_idx]

                                    # Preserve original target language stem if cured item accidentally provided English translation as stem
                                    orig_stem = orig_dict.get("translated_sentence", "")
                                    curr_stem = candidate_cure.get("translated_sentence", "")
                                    # If target language is Chinese and curr_stem lacks Chinese chars, restore original stem
                                    if orig_stem:
                                        target_lang_str = str(curr_tgt_lang or "Chinese").lower()
                                        if any(w in target_lang_str for w in ["chinese", "mandarin", "cjk", "中文", "汉语", "漢語"]):
                                            if not re.search(r'[\u4e00-\u9fff]', curr_stem) and re.search(r'[\u4e00-\u9fff]', orig_stem):
                                                candidate_cure["translated_sentence"] = orig_stem

                                    for bp_field in ("target_keyword", "target_grammar", "flaw_type"):
                                        if not candidate_cure.get(bp_field) and orig_dict.get(bp_field):
                                            candidate_cure[bp_field] = orig_dict[bp_field]
                                    if not candidate_cure.get("explanation") and candidate_cure.get("diagnostic_critique"):
                                        candidate_cure["explanation"] = candidate_cure["diagnostic_critique"]

                                # Validate candidate cure through Level 1 Code Gate
                                temp_wrap = {"questions": [candidate_cure]}
                                temp_wrap = self._shuffle_quiz_options(temp_wrap, preserve_blueprint_order=has_prescribed_options)
                                healed_item = temp_wrap["questions"][0]

                                # Modality-specific Level 1 gate on candidate cure
                                cure_l1_defects = []
                                cure_cefr = kwargs.get("cefr_level", "B2")
                                if template_name == "reading":
                                    cure_l1_flagged, cure_l1_defects = self.audit_reading_integrity(
                                        {"questions": [healed_item]},
                                        passage_text=source_context,
                                        cefr_level=cure_cefr
                                    )
                                    cure_item_failed = bool(cure_l1_flagged)
                                elif template_name == "video":
                                    cure_l1_flagged, cure_l1_defects = self.audit_video_integrity(
                                        {"questions": [healed_item]},
                                        transcript_text=source_context,
                                        cefr_level=cure_cefr
                                    )
                                    cure_item_failed = bool(cure_l1_flagged)
                                elif template_name == "listening":
                                    cure_l1_flagged, cure_l1_defects = self.audit_listening_integrity(
                                        {"questions": [healed_item]},
                                        script_text=source_context,
                                        cefr_level=cure_cefr,
                                        unit_headwords=unit_headwords
                                    )
                                    cure_item_failed = bool(cure_l1_flagged)
                                elif template_name == "translation":
                                    cure_l1_flagged, cure_l1_defects = self.audit_translation_integrity(
                                        {"questions": [healed_item]},
                                        unit_headwords=unit_headwords,
                                        target_language=curr_tgt_lang,
                                        cefr_level=cure_cefr,
                                        enforce_minimal_pair=bool(blueprint_specs),
                                    )
                                    cure_item_failed = bool(cure_l1_flagged)
                                else:
                                    candidate_batch = list(cured_questions)
                                    candidate_batch[item_slot] = healed_item
                                    cure_l1_flagged, cure_l1_defects = self.audit_quiz_integrity(
                                        {"questions": candidate_batch},
                                        banned_sentences=banned_quiz_sentences,
                                        unit_headwords=unit_headwords,
                                        strict_distractor_recycling=True,
                                        allow_authentic_cloze=is_cloze_mode
                                    )
                                    cure_item_failed = item_slot in cure_l1_flagged

                                if not cure_item_failed:
                                    # Candidate passed L1 gate: splice in-place and assign deterministic score
                                    cured_questions[item_slot] = healed_item
                                    qa["single_fit_valid"] = True
                                    if triage == "REPAIR":
                                        qa["pedagogical_score"] = 90
                                        cure_stats["repair"] += 1
                                        logger.info(f"Surgically REPAIRED Item #{item_slot + 1} with in-place expert cure (Score: 90).")
                                    else:
                                        qa["pedagogical_score"] = 95
                                        cure_stats["rewrite"] += 1
                                        logger.info(f"Surgically REWROTE Item #{item_slot + 1} with in-place expert cure (Score: 95).")
                                else:
                                    # Candidate failed L1 gate: reject cure
                                    logger.warning(
                                        f"Candidate expert cure for Item #{item_slot + 1} failed Level 1 gate ({cure_l1_defects}). Keeping original item penalized."
                                    )
                                    cure_stats["discard"] += 1
                            else:
                                if qa.get("single_fit_valid", True):
                                    cure_stats["pass"] += 1
                                else:
                                    cure_stats["discard"] += 1

                        # Prune uncured defective items to protect pedagogical output
                        final_clean_questions = []
                        for idx, q_candidate in enumerate(cured_questions):
                            corresponding_audit = audit_items[idx] if idx < len(audit_items) and isinstance(audit_items[idx], dict) else {}
                            # Keep item if it is single_fit_valid
                            if corresponding_audit.get("single_fit_valid", True):
                                final_clean_questions.append(q_candidate)
                            else:
                                logger.warning(
                                    f"Pruned uncurable defective Item #{idx + 1} from final handout (single_fit_valid=False)."
                                )

                        # Apply cured & pruned questions to quiz_obj
                        if isinstance(quiz_obj, dict):
                            quiz_obj["questions"] = final_clean_questions
                        else:
                            try:
                                quiz_obj.questions = final_clean_questions
                            except Exception:
                                pass
                        quiz_obj = self._shuffle_quiz_options(quiz_obj)

                        # Re-bind modality-specific properties and deterministic CEFR level
                        if isinstance(quiz_obj, dict):
                            quiz_obj["cefr_level"] = cefr_val
                        else:
                            try:
                                setattr(quiz_obj, "cefr_level", cefr_val)
                            except Exception:
                                pass

                        if template_name == "reading":
                            if isinstance(quiz_obj, dict): quiz_obj["passage"] = data["passage"]
                            else: quiz_obj.passage = data["passage"]
                        elif template_name == "video":
                            if isinstance(quiz_obj, dict):
                                quiz_obj["video_url"] = data["video_url"]
                                quiz_obj["video_type"] = data["video_type"]
                                quiz_obj["transcript"] = data["transcript"]
                            else:
                                quiz_obj.video_url = data["video_url"]
                                quiz_obj.video_type = data["video_type"]
                                quiz_obj.transcript = data["transcript"]

                        # ---------------------------------------------------------
                        # Step 4: Deterministic Post-Cure Scoring
                        # ---------------------------------------------------------
                        total_items_count = len(orig_questions) or len(audit_items) or 1
                        missing_audit_items = max(0, len(orig_questions) - len(audit_items)) if orig_questions else 0
                        audit_report["missing_audit_items"] = missing_audit_items

                        item_scores = [qa.get("pedagogical_score", 90) for qa in audit_items if isinstance(qa, dict)]
                        score_denom = max(len(item_scores), total_items_count)
                        final_avg = round(sum(item_scores) / score_denom) if item_scores else 100
                        audit_report["base_quality_score"] = final_avg

                        # Preserve judge's cap and fail verdict if uncured defects or truncated items remain
                        has_fatal_residual = any(
                            qa.get("single_fit_valid") is False 
                            for qa in audit_items if isinstance(qa, dict)
                        ) or (cure_stats["discard"] > 0) or (missing_audit_items > 0)

                        if has_fatal_residual:
                            audit_report["pass_audit"] = False
                            cap_limit = 59 if missing_audit_items > 0 else 70
                            audit_report["overall_quality_score"] = min(final_avg, cap_limit)
                        else:
                            audit_report["pass_audit"] = (final_avg >= 80)
                            audit_report["overall_quality_score"] = final_avg

                        final_overall_score = audit_report["overall_quality_score"]
                        audit_report["cure_stats"] = cure_stats

                        # Post-cure accuracy updates: successfully cured items now have a unique, verified key
                        valid_items_count = cure_stats["pass"] + cure_stats["repair"] + cure_stats["rewrite"]
                        audit_report["post_cure_accuracy"] = round(valid_items_count / total_items_count, 2)
                        audit_report["blind_solve_accuracy"] = audit_report["post_cure_accuracy"]

                        cured_count = cure_stats["repair"] + cure_stats["rewrite"]
                        # Update summary verdict to disclose in-place surgical cure when actual cures occurred
                        if cured_count > 0:
                            cure_summary = (
                                f"[SURGICAL CURE APPLIED: Pre-Cure {raw_gen_score}% -> Post-Cure {final_overall_score}%. "
                                f"Pass: {cure_stats['pass']}, Repaired: {cure_stats['repair']}, "
                                f"Rewritten: {cure_stats['rewrite']}, Discard/Flawed: {cure_stats['discard']}]."
                            )
                            orig_verdict = str(audit_report.get("summary_verdict", ""))
                            audit_report["summary_verdict"] = f"{cure_summary} {orig_verdict}".strip()

                        # Synchronize Pre-Cure vs Post-Cure scores to the expert audit log file
                        try:
                            logs_dir = Path(self.config.get("project_root", ".")).resolve() / "logs"
                            task_prefix = f"expert_audit_{quiz_dict_eval.get('title', 'quiz')[:20]}"
                            safe_prefix = re.sub(r'[\\/:*?"<>|\r\n]+', '_', task_prefix).strip('_')
                            matching_logs = sorted(logs_dir.glob(f"*_{safe_prefix}*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
                            if matching_logs:
                                latest_log = matching_logs[0]
                                log_txt = latest_log.read_text(encoding="utf-8")
                                cured_tag = " (CURED)" if cured_count > 0 else ""
                                score_block = (
                                    f"=== L2_QUALITY_SCORE: {final_overall_score}.0%{cured_tag} ===\n"
                                    f"=== COMPOSITE_SCORE: {final_overall_score}.0%{cured_tag} ===\n"
                                    f"=== PRE_CURE_SCORE: {raw_gen_score}.0% ===\n"
                                    f"=== POST_CURE_SCORE: {final_overall_score}.0% ===\n"
                                    f"=== SURGICAL_CURE: Repaired={cure_stats['repair']}, Rewritten={cure_stats['rewrite']}, Passed={cure_stats['pass']}, Flawed={cure_stats['discard']} ==="
                                )
                                if "=== COMPOSITE_SCORE:" in log_txt:
                                    log_txt = re.sub(
                                        r"(?:=== L2_QUALITY_SCORE:[^\n]+\n)?=== COMPOSITE_SCORE:\s*[\d\.]+%\s*===(?:\n=== L2_PENALTY:[^\n]+===)?",
                                        score_block,
                                        log_txt
                                    )
                                    latest_log.write_text(log_txt, encoding="utf-8")
                        except Exception as log_err:
                            logger.debug(f"Could not update audit log with post-cure score: {log_err}")

                        # Attach final audit report to quiz_obj
                        if isinstance(quiz_obj, dict):
                            quiz_obj["_expert_audit"] = audit_report
                        else:
                            try:
                                setattr(quiz_obj, "_expert_audit", audit_report)
                            except Exception:
                                pass

                        final_accuracy = audit_report.get("blind_solve_accuracy", 1.0)
                        final_passed = audit_report.get("pass_audit", True)

                        logger.info(
                            f"Level 2 In-Place Surgical Cure Complete for {core_name} ({template_name}): "
                            f"Raw Gen: {raw_gen_score}% -> Final Post-Cure: {final_avg}%, "
                            f"Pass: {final_passed}, Blind Acc: {final_accuracy * 100:.1f}%, Stats: {cure_stats}"
                        )

                        if audit_callback:
                            try:
                                if final_passed:
                                    audit_callback(
                                        stage="passed",
                                        audit_progress=100,
                                        message=f"✅ Level 2 Quality Audit PASSED (Final: {final_avg}%, Blind Acc: {final_accuracy*100:.0f}%)",
                                        extra={"score": final_avg, "accuracy": final_accuracy, "passed": True, "cure_stats": cure_stats}
                                    )
                                else:
                                    audit_callback(
                                        stage="completed_with_warnings",
                                        audit_progress=100,
                                        message=f"⚠️ Audit finished with warnings (Score: {final_avg}%). Proceeding with best attempt.",
                                        extra={"score": final_avg, "accuracy": final_accuracy, "passed": False, "cure_stats": cure_stats}
                                    )
                            except Exception:
                                pass

                except Exception as audit_err:
                    logger.warning(f"Level 2 Expert Quality Audit encountered non-fatal error: {audit_err}", exc_info=True)



            # 5. TTS for Listening Quiz (Base64 Embedding)
            audio_url = None
            if template_name == "listening":
                # Detect dialogue data (dataclass or dict)
                is_dialogue = isinstance(quiz_obj, ListeningQuiz) or (isinstance(quiz_obj, dict) and "script" in quiz_obj)

                if is_dialogue:
                    from ..tts import tts_service
                    tts_service._refresh()

                    # Use pre-calculated genders, roles, and accents directly from kwargs or quiz_obj
                    if isinstance(quiz_obj, dict):
                        g1 = quiz_obj.get("speaker_1_gender") or kwargs["speaker_1_gender"]
                        g2 = quiz_obj.get("speaker_2_gender") or kwargs["speaker_2_gender"]
                        a1 = quiz_obj.get("speaker_1_accent") or kwargs["speaker_1_accent"]
                        a2 = quiz_obj.get("speaker_2_accent") or kwargs["speaker_2_accent"]
                        
                        role1 = "Creative Director" if g1 == "female" else "Technical Lead"
                        role2 = "Technical Lead" if g2 == "male" else "Creative Director"
                        if role1 == role2:
                            role2 = "Product Manager" if g2 == "female" else "Lead Analyst"
                            
                        quiz_obj["speaker_1_gender"] = g1
                        quiz_obj["speaker_2_gender"] = g2
                        quiz_obj["speaker_1_role"] = role1
                        quiz_obj["speaker_2_role"] = role2
                        quiz_obj["speaker_1_accent"] = a1
                        quiz_obj["speaker_2_accent"] = a2
                        script_dicts = quiz_obj["script"]
                        s1 = quiz_obj.get("speaker_1")
                        s2 = quiz_obj.get("speaker_2")
                    else:
                        g1 = quiz_obj.speaker_1_gender or kwargs["speaker_1_gender"]
                        g2 = quiz_obj.speaker_2_gender or kwargs["speaker_2_gender"]
                        a1 = quiz_obj.speaker_1_accent or kwargs["speaker_1_accent"]
                        a2 = quiz_obj.speaker_2_accent or kwargs["speaker_2_accent"]
                        
                        role1 = "Creative Director" if g1 == "female" else "Technical Lead"
                        role2 = "Technical Lead" if g2 == "male" else "Creative Director"
                        if role1 == role2:
                            role2 = "Product Manager" if g2 == "female" else "Lead Analyst"
                            
                        quiz_obj.speaker_1_gender = g1
                        quiz_obj.speaker_2_gender = g2
                        quiz_obj.speaker_1_role = role1
                        quiz_obj.speaker_2_role = role2
                        quiz_obj.speaker_1_accent = a1
                        quiz_obj.speaker_2_accent = a2
                        script_dicts = [dataclasses.asdict(t) for t in quiz_obj.script]
                        s1 = quiz_obj.speaker_1
                        s2 = quiz_obj.speaker_2

                    audio_binary = tts_service.process_script(
                        script_dicts, 
                        return_binary=True,
                        speaker_1=s1,
                        speaker_2=s2,
                        speaker_1_gender=g1,
                        speaker_2_gender=g2
                    )
                    
                    if audio_binary:
                        b64_str = base64.b64encode(audio_binary).decode("utf-8")
                        audio_url = f"data:audio/mp3;base64,{b64_str}"

            # 6. Render and Save
            language = self.config.get("target_language") or "Chinese"
            html_content = self._render_handout(quiz_obj, template_name, audio_url, language=language)
            asset_stem = data.get("source_stem", core_name) if isinstance(data, dict) else core_name
            handout_filename = f"{asset_stem}_{template_name}_quiz.html"
            handout_dir = self.config.wiki_content_path / core_name / "handouts"
            handout_dir.mkdir(parents=True, exist_ok=True)
            handout_path = handout_dir / handout_filename

            # -------------------------------------------------------------
            # Transparent Delivery: Handouts always ship to handouts/
            # Log audit recommendations if items were cured or discarded
            # -------------------------------------------------------------
            quiz_audit = None
            if hasattr(quiz_obj, "_expert_audit") and getattr(quiz_obj, "_expert_audit"):
                quiz_audit = getattr(quiz_obj, "_expert_audit")
            elif hasattr(quiz_obj, "audit") and getattr(quiz_obj, "audit"):
                quiz_audit = getattr(quiz_obj, "audit")
            elif isinstance(quiz_obj, dict):
                quiz_audit = quiz_obj.get("_expert_audit") or quiz_obj.get("audit")

            final_questions = getattr(quiz_obj, "questions", []) or (quiz_obj.get("questions", []) if isinstance(quiz_obj, dict) else [])
            if quiz_audit:
                pass_audit = quiz_audit.get("pass_audit", True)
                discarded_count = (quiz_audit.get("cure_stats") or {}).get("discard", 0)
                available_count = len(final_questions)
                if not pass_audit:
                    logger.warning(
                        f"⚠️ Handout {handout_filename} shipped with audit advisory: {pass_audit=}, "
                        f"{available_count=} questions available, {discarded_count=} discarded."
                    )

            with open(handout_path, "w", encoding="utf-8") as f:
                f.write(html_content)

            return str(handout_path)

        except Exception as e:
            return f"Quiz Generation Error: {e}"

    def _render_handout(self, quiz_obj, template_name, audio_url=None, language=None):
        """Renders HTML templates with injected data."""
        template_path = Path(__file__).resolve().parent.parent / "templates" / f"{template_name}.html"
        
        # 1. Convert to dict if it's a dataclass
        if dataclasses.is_dataclass(quiz_obj):
            data_dict = dataclasses.asdict(quiz_obj)
            if hasattr(quiz_obj, "_expert_audit") and getattr(quiz_obj, "_expert_audit"):
                data_dict["_expert_audit"] = getattr(quiz_obj, "_expert_audit")
        else:
            data_dict = quiz_obj # It's already a dict from a JSON-file schema
            
        if not template_path.exists():
            return f"<html><body><pre>{json.dumps(data_dict, indent=2)}</pre></body></html>"

        with open(template_path, "r", encoding="utf-8") as f: html = f.read()
        
        if audio_url: data_dict["audio_url"] = audio_url
        if language: data_dict["target_language"] = language
        
        # Strip design_audit from questions/root if present so internal drafts do not leak to client HTML
        def _strip_design_audit(obj):
            if isinstance(obj, dict):
                return {k: _strip_design_audit(v) for k, v in obj.items() if k != "design_audit"}
            elif isinstance(obj, list):
                return [_strip_design_audit(item) for item in obj]
            return obj

        cleaned_data = _strip_design_audit(data_dict)
        json_data = json.dumps(cleaned_data, ensure_ascii=False)
        

        if "const quizData =" in html:
            return re.sub(r'const quizData = .*?;', lambda _: f'const quizData = {json_data};', html, flags=re.DOTALL)
        return html.replace("</body>", f"<script>const quizData = {json_data};</script></body>")

    def _interpolate_schema(self, schema, mapping):
        """Recursively interpolates placeholders in a JSON schema dict."""
        if not isinstance(schema, dict):
            return schema
        
        def _recurse(obj):
            if isinstance(obj, str):
                val = self._safe_format(obj, mapping)
                # If the resulting value is a pure integer string, convert to int
                # This is critical for JSON Schema fields like maxItems, minLength, etc.
                if re.match(r'^-?\d+$', val):
                    return int(val)
                return val
            elif isinstance(obj, dict):
                return {k: _recurse(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [_recurse(v) for v in obj]
            return obj
            
        return _recurse(schema)

    def _safe_format(self, text, mapping):
        """Formats string with mapping, ignoring missing keys and non-stringable values."""
        for k, v in mapping.items():
            if isinstance(v, (str, int, float)):
                text = text.replace(f"{{{k}}}", str(v))
        return text
