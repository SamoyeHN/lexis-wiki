"""Fisher-Yates option shuffling, deterministic seed locking, and explanation label remapping."""

from __future__ import annotations

import logging
import random
import re
from typing import Any, Dict, List, Tuple

from ..linguistics import LinguisticEngine

logger = logging.getLogger("librarian.processor.shuffler")


class QuizShufflerMixin:
    """Mixin providing deterministic Fisher-Yates option shuffling and explanation label synchronization."""

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

            # 3b. Redundant Finite Copula Auto-Healing before bare-infinitive 'be ...' phrase options:
            if q_stem and options and all(str(opt).strip().lower().startswith("be ") for opt in options if str(opt).strip()):
                def _heal_copula(m: re.Match) -> str:
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
                    _heal_copula,
                    q_stem,
                    flags=re.IGNORECASE,
                )
                if healed_cop_stem != q_stem:
                    if "question" in q_dict:
                        if isinstance(q, dict):
                            q["question"] = healed_cop_stem
                        else:
                            setattr(q, "question", healed_cop_stem)
                    elif "translated_sentence" in q_dict:
                        if isinstance(q, dict):
                            q["translated_sentence"] = healed_cop_stem
                        else:
                            setattr(q, "translated_sentence", healed_cop_stem)
                    q_stem = healed_cop_stem

            # 3c. Verb Inflection Agreement Auto-Healing (动词同型自愈门禁):
            # If target word is an inflected verb (-ed / -ing / -s form), ensure options do not clash
            # with base forms (e.g. 'protested' with ['arise', 'protect', 'profess']).
            tw_cand = str(q_dict.get("target_word") or q_dict.get("word") or expected_answer or "").strip()
            if tw_cand and options and len(options) >= 2:
                from librarian.linguistics import LinguisticEngine
                tw_parts = tw_cand.split()
                tw_head = tw_parts[0].lower()
                tw_head_base = LinguisticEngine.lemma_of(tw_head) or tw_head
                pos_str = str(q_dict.get("part_of_speech") or "").lower()
                is_adj = any(a in pos_str for a in ("adj", "adjective"))
                is_pv = len(tw_parts) >= 2 and LinguisticEngine._is_base_verb_lemma(tw_head_base)
                is_v = not is_adj and len(tw_parts) == 1 and (
                    "verb" in pos_str
                    or (not pos_str and LinguisticEngine._is_base_verb_lemma(tw_head_base))
                )
                if is_v or is_pv:
                    tw_verb_tag = None
                    if tw_head != tw_head_base:
                        for cand_tag in ("VBD", "VBG", "VBZ", "VBN"):
                            if (LinguisticEngine._inflect_verb(tw_head_base, cand_tag) or "").lower() == tw_head:
                                tw_verb_tag = cand_tag
                                break
                    if not tw_verb_tag and q_stem:
                        tw_verb_tag = LinguisticEngine._example_target_tag(q_stem, tw_head)
                    if tw_verb_tag in ("VBD", "VBN", "VBG", "VBZ"):
                        healed_opts = []
                        opts_were_healed = False
                        opt_replacements = {}
                        for o_i, opt in enumerate(options):
                            opt_str = str(opt).strip()
                            if not opt_str:
                                healed_opts.append(opt_str)
                                continue
                            if is_pv:
                                o_parts = opt_str.split()
                                if not o_parts:
                                    healed_opts.append(opt_str)
                                    continue
                                o_head = o_parts[0].lower()
                                o_base = o_head if LinguisticEngine._is_base_verb_lemma(o_head) else (LinguisticEngine.lemma_of(o_head) or o_head)
                                o_can_be_verb = (
                                    LinguisticEngine._is_base_verb_lemma(o_base)
                                    or bool(LinguisticEngine._wordnet_pos(o_base) & {"v"})
                                )
                                if o_can_be_verb and o_head == o_base and o_head != tw_head:
                                    o_inf = LinguisticEngine._inflect_verb(o_base, tw_verb_tag)
                                    if (
                                        o_inf
                                        and LinguisticEngine._morph_form_valid(o_base, o_inf, tw_verb_tag)
                                        and LinguisticEngine.is_attested_form(o_inf)
                                    ):
                                        healed_opt = " ".join([o_inf] + o_parts[1:])
                                        healed_opts.append(healed_opt)
                                        opts_were_healed = True
                                        opt_replacements[opt_str] = healed_opt
                                        continue
                                healed_opts.append(opt_str)
                            else:
                                o_base = opt_str if LinguisticEngine._is_base_verb_lemma(opt_str) else (LinguisticEngine.lemma_of(opt_str) or opt_str)
                                o_can_be_verb = (
                                    LinguisticEngine._is_base_verb_lemma(o_base)
                                    or bool(LinguisticEngine._wordnet_pos(o_base) & {"v"})
                                )
                                if o_can_be_verb and o_base == opt_str.lower() and opt_str.lower() != tw_head:
                                    o_inf = LinguisticEngine._inflect_verb(o_base, tw_verb_tag)
                                    if (
                                        o_inf
                                        and LinguisticEngine._morph_form_valid(o_base, o_inf, tw_verb_tag)
                                        and LinguisticEngine.is_attested_form(o_inf)
                                    ):
                                        healed_opts.append(o_inf)
                                        opts_were_healed = True
                                        opt_replacements[opt_str] = o_inf
                                        continue
                                healed_opts.append(opt_str)
                        if opts_were_healed and len(healed_opts) == len(options):
                            options = healed_opts
                            cur_exp = str(q_dict.get("explanation") or "")
                            if cur_exp and opt_replacements:
                                for old_opt, new_opt in opt_replacements.items():
                                    cur_exp = re.sub(rf"\b{re.escape(old_opt)}\b", new_opt, cur_exp, flags=re.IGNORECASE)
                                if isinstance(q, dict):
                                    q["explanation"] = cur_exp
                                else:
                                    setattr(q, "explanation", cur_exp)

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

