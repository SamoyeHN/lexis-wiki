import random
import os
import json
import re
import dataclasses
import base64
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from .config import config
from .llm import llm
from .prompts import Prompts

logger = logging.getLogger("librarian.processor")

from .schemas import (
    VocabularyExtraction, GrammarExtraction, SummaryExtraction,
    VocabularyQuiz, ReadingQuiz, TranslationQuiz, ListeningQuiz,
    RoutingResult, MindMapExtraction,
    validate_and_map
)

class WikiProcessor:
    def __init__(self):
        self.config = config

    def _shuffle_quiz_options(self, quiz_obj):
        """Randomizes the order of options for each question and updates the correct index."""
        return self.shuffle_quiz_options(quiz_obj)

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
            open_b = m.group(1)
            let = m.group(2).lower()
            close_b = m.group(3)
            if let in let_to_num and let_to_num[let] in old_to_new:
                return f"{open_b}{num_to_let[old_to_new[let_to_num[let]]]}{close_b}"
            return m.group(0)

        res = re.sub(
            r'\b(Option|Choice)\s+([A-Da-d\d]|One|Two|Three|Four|First|Second|Third|Fourth)\b',
            replace_prefixed,
            explanation,
            flags=re.IGNORECASE
        )
        res = re.sub(r'(\(|\b\[)([A-Da-d])(\]|\))', replace_bracketed, res)
        return res

    @staticmethod
    def parse_raw_headwords(raw_input: Any) -> List[str]:
        """
        Smart parser for messy textbook vocabulary lists.
        Handles dirty pastes with numbers, IPA, PoS abbreviations, Chinese translations,
        bullet points, colons, commas, semicolons, and newlines.
        """
        if not raw_input:
            return []

        if isinstance(raw_input, list):
            lines = [str(x) for x in raw_input]
        else:
            lines = str(raw_input).strip().split('\n')

        headwords = []
        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue

            # Remove leading Markdown bullets, numbers, colons
            line_clean = re.sub(r'^(?:[\d\.\-\*\•\–\—\)\s]+)', '', line_str).strip()
            if not line_clean:
                continue

            # If separated by commas or semicolons
            chunks = re.split(r'[,;]+', line_clean)
            for chunk in chunks:
                chunk = chunk.strip()
                if not chunk:
                    continue

                # Remove IPA pronunciations like /'bɜːɡləri/ or [ˈfæsɪneɪtɪŋ]
                chunk = re.sub(r'\/[^\/]+\/|\[[^\]]+\]', ' ', chunk)

                # Split out Chinese translations or English definitions following colons, dashes, tabs, or spaces
                chunk = re.split(r'[\u4e00-\u9fa5]|(?<!\w)[—–\-:]+(?!\w)|\t', chunk)[0].strip()

                # Clean common PoS abbreviations
                chunk = re.sub(r'\b(?:n|v|vt|vi|adj|adv|prep|conj|pron|art|num|phr|idiom)\.?\b', ' ', chunk, flags=re.IGNORECASE)

                # Normalize whitespace
                chunk = re.sub(r'\s+', ' ', chunk).strip()

                # Filter out numbers and punctuation
                if chunk and re.search(r'[A-Za-z]', chunk):
                    cleaned_hw = re.sub(r'[^\w\s\-\'\[\]\(\)\/]', '', chunk).strip()
                    if cleaned_hw and len(cleaned_hw) >= 2:
                        headwords.append(cleaned_hw)

        # Deduplicate while preserving order
        seen = set()
        final_list = []
        for w in headwords:
            w_lower = w.lower()
            if w_lower not in seen:
                seen.add(w_lower)
                final_list.append(w)

        return final_list

    @classmethod
    def parse_syllabus_sections(cls, content: str) -> Tuple[str, List[str], List[str], List[str]]:
        """
        Parses source Markdown for syllabus sections (Plan 1: single-source architecture).
        Extracts:
        - clean_body: Text excluding syllabus sections (avoids polluting standard extraction)
        - syllabus_vocab: Extracted mandatory vocabulary/phrases list
        - syllabus_grammar: Extracted mandatory grammar patterns list
        - syllabus_expressions: Extracted mandatory expressions/phrases (explicit or multi-word items)
        """
        if not content:
            return "", [], [], []

        # Find any ## Syllabus Vocabulary, ## Syllabus Grammar, or ## Syllabus Expressions sections
        vocab_matches = re.search(
            r'##+\s*(?:Syllabus\s+Vocabulary|Vocabulary\s+List|Target\s+Words|Word\s*List|词汇表|生词表)[^\n]*\n([\s\S]*?)(?=\n##+|\Z)',
            content,
            re.IGNORECASE
        )
        grammar_matches = re.search(
            r'##+\s*(?:Syllabus\s+Grammar|Grammar\s+Topics|Grammar\s+List|Target\s+Grammar|语法点|语法表)[^\n]*\n([\s\S]*?)(?=\n##+|\Z)',
            content,
            re.IGNORECASE
        )
        expr_matches = re.search(
            r'##+\s*(?:Syllabus\s+Expressions|Syllabus\s+Phrases|Expressions\s+List|Phrases\s+List|短语表|词组表)[^\n]*\n([\s\S]*?)(?=\n##+|\Z)',
            content,
            re.IGNORECASE
        )

        raw_vocab = []
        if vocab_matches:
            raw_vocab_text = vocab_matches.group(1).strip()
            raw_vocab = cls.parse_raw_headwords(raw_vocab_text)

        syllabus_grammar = []
        if grammar_matches:
            raw_grammar_text = grammar_matches.group(1).strip()
            # Split grammar items by line / bullets / numbers
            for line in raw_grammar_text.split('\n'):
                line = re.sub(r'^(?:[\d\.\-\*\•\–\—\)\s]+)', '', line).strip()
                if line and len(line) >= 2:
                    syllabus_grammar.append(line)

        # Multi-word indicator test: spaces, slashes, placeholders (sb/sth/one's), parentheses
        def is_multiword(item: str) -> bool:
            clean_item = item.strip()
            return (
                " " in clean_item
                or "/" in clean_item
                or "sb." in clean_item.lower()
                or "sb " in clean_item.lower()
                or "sth." in clean_item.lower()
                or "sth " in clean_item.lower()
                or "one's" in clean_item.lower()
                or "(" in clean_item
            )

        # Partition raw_vocab into pure single words vs detected expressions
        pure_words = []
        detected_exprs = []
        for item in raw_vocab:
            if is_multiword(item):
                detected_exprs.append(item)
            else:
                pure_words.append(item)

        # 1. Parse explicit expressions if section present
        syllabus_expressions = []
        if expr_matches:
            raw_expr_text = expr_matches.group(1).strip()
            syllabus_expressions = cls.parse_raw_headwords(raw_expr_text)
            # Merge any extra detected expressions from raw_vocab without duplicates
            seen_exprs = {e.lower() for e in syllabus_expressions}
            for e in detected_exprs:
                if e.lower() not in seen_exprs:
                    syllabus_expressions.append(e)
                    seen_exprs.add(e.lower())
        else:
            syllabus_expressions = detected_exprs

        # 2. Pure single words strictly for vocabulary extraction (no multi-word overlap)
        syllabus_vocab = pure_words

        # Strip syllabus sections from body text to avoid confusing general prompt
        clean_body = re.sub(
            r'##+\s*(?:Syllabus\s+Vocabulary|Vocabulary\s+List|Target\s+Words|Word\s*List|词汇表|生词表)[^\n]*\n[\s\S]*?(?=\n##+|\Z)',
            '',
            content,
            flags=re.IGNORECASE
        )
        clean_body = re.sub(
            r'##+\s*(?:Syllabus\s+Grammar|Grammar\s+Topics|Grammar\s+List|Target\s+Grammar|语法点|语法表)[^\n]*\n[\s\S]*?(?=\n##+|\Z)',
            '',
            clean_body,
            flags=re.IGNORECASE
        )
        clean_body = re.sub(
            r'##+\s*(?:Syllabus\s+Expressions|Syllabus\s+Phrases|Expressions\s+List|Phrases\s+List|短语表|词组表)[^\n]*\n[\s\S]*?(?=\n##+|\Z)',
            '',
            clean_body,
            flags=re.IGNORECASE
        ).strip()

        return clean_body, syllabus_vocab, syllabus_grammar, syllabus_expressions

    # Normalization mapping from legacy/verbose grammar slots to standard COBUILD tokens
    GRAMMAR_SLOT_NORMALIZATION: Dict[str, str] = {
        "clause": "[S]",
        "main clause": "[S]",
        "subordinate clause": "[S]",
        "dependent clause": "[S]",
        "subject": "[S]",
        "predicate": "[S]",
        "noun phrase": "[NP]",
        "verb phrase": "[VP]",
        "focus element": "[NP]",
        "object": "[NP]",
        "complement": "[NP]",
        "copula": "[be]",
        "auxiliary": "[aux]",
        "modal": "[aux]",
        "past participle": "[V3]",
        "v-ed": "[V3]",
        "verb-ed": "[V3]",
        "present participle": "[V-ing]",
        "gerund": "[V-ing]",
        "verb-ing": "[V-ing]",
        "base verb": "[V]",
        "finite verb": "[V]",
        "infinitive": "[to-V]",
        "adjective": "[adj]",
        "evaluative adjective": "[adj]",
        "comparative": "[adj]",
        "adverb": "[adv]",
        "preposition": "[prep]",
        "prepositional phrase": "[prep] + [NP]",
    }

    # Standard literal functional words/connectors commonly wrapped in brackets by LLMs
    LITERAL_FUNCTIONAL_WORDS = frozenset({
        "the", "a", "an", "that", "this", "these", "those", "such", "it", "there",
        "if", "then", "when", "while", "as", "since", "because", "although", "though", "even though",
        "to", "for", "of", "with", "by", "at", "in", "on", "from", "into", "onto",
        "have", "has", "had", "have to", "has to", "had to", "do", "does", "did",
        "not", "no", "never", "only", "also", "too", "either", "neither", "nor", "both",
        "so", "such that", "so that", "more", "most", "less", "least", "than"
    })

    # Canonical casing for standard COBUILD slots
    CANONICAL_SLOT_CASING = {
        "s": "[S]", "np": "[NP]", "vp": "[VP]", "v": "[V]", "be": "[be]", "aux": "[aux]",
        "v-ed": "[V3]", "v3": "[V3]", "v-ing": "[V-ing]", "to-v": "[to-V]",
        "adj": "[adj]", "adv": "[adv]", "prep": "[prep]", "det": "[det]", "conj": "[conj]"
    }

    @classmethod
    def unwrap_literal_brackets(cls, formula: str) -> str:
        """
        Unwraps accidental square brackets around literal words/connectors in formulas.
        E.g.:
        - '[If] [S] [to-V], [then] [S] [have to] [V]' -> 'If [S] [to-V], then [S] have to [V]'
        - '[Such] [NP] [V] [that] [S]' -> 'Such [NP] [V] that [S]'
        - '[The] [adj] [NP] of [NP] [that] [be] [adv] [S]' -> 'The [adj] [NP] of [NP] that [be] [adv] [S]'
        - '[V-ing] [it] [adj] [to-V]' -> '[V-ing] it [adj] [to-V]'
        """
        if not formula or "[" not in formula:
            return formula

        from .evaluator import ALLOWED_GRAMMAR_SLOTS

        def _repl(match: re.Match) -> str:
            raw_slot = match.group(1).strip()
            slot_lower = raw_slot.lower()
            if slot_lower in cls.GRAMMAR_SLOT_NORMALIZATION:
                return cls.GRAMMAR_SLOT_NORMALIZATION[slot_lower]
            if "/" in slot_lower:
                sub_parts = [p.strip() for p in slot_lower.split("/") if p.strip()]
                if all(p in ALLOWED_GRAMMAR_SLOTS or p in cls.GRAMMAR_SLOT_NORMALIZATION for p in sub_parts):
                    norm_parts = [
                        cls.GRAMMAR_SLOT_NORMALIZATION.get(p, cls.CANONICAL_SLOT_CASING.get(p, f"[{p}]")).strip("[]")
                        for p in sub_parts
                    ]
                    return f"[{'/'.join(norm_parts)}]"
            if slot_lower in cls.CANONICAL_SLOT_CASING:
                return cls.CANONICAL_SLOT_CASING[slot_lower]
            if slot_lower in ALLOWED_GRAMMAR_SLOTS:
                return f"[{raw_slot}]"
            if slot_lower in cls.LITERAL_FUNCTIONAL_WORDS:
                # It is a literal word or connective phrase (e.g. that, it, if, then, the, such, to, have to)
                # Unwrap bracket so it serves as natural functional anchor
                return raw_slot
            # Keep original bracket for unrecognized slot so evaluator catches it
            return f"[{raw_slot}]"

        unwrapped = re.sub(r'\[(.*?)\]', _repl, formula)
        return re.sub(r'\s+', ' ', unwrapped).strip()

    @classmethod
    def normalize_grammar_formula(cls, formula: str) -> str:
        """
        Normalizes legacy or verbose bracketed slots into concise, standard COBUILD tokens,
        and unwraps literal words from square brackets.
        """
        return cls.unwrap_literal_brackets(formula)

    @staticmethod
    def _sanitize_vocab_for_quiz(vocab_content: str) -> Tuple[str, List[str], List[str]]:
        """
        Physically isolates the quiz generator from example sentences:
        1. Strips all '- **Quoted Sentence**:' and '- **Example Usage**:' lines.
        2. Returns:
           - sanitized_content: Clean markdown with only headword, part of speech, definition, and CEFR level.
           - headwords: List of lowercase headword strings in this unit.
           - banned_sentences: List of the raw quoted and example sentences to audit stems against.
        """
        if not vocab_content:
            return "", [], []

        headwords = []
        for hw in re.findall(r'##\s*\[\[(.*?)\]\]', vocab_content):
            hw_clean = hw.strip().lower()
            if hw_clean:
                headwords.append(hw_clean)

        banned_sentences = []
        clean_lines = []
        for line in vocab_content.splitlines():
            m = re.match(r'^\s*-\s*\*\*(?:Quoted Sentence|Example Usage)\*\*:\s*(.*)', line, re.IGNORECASE)
            if m:
                sent = m.group(1).strip()
                if sent:
                    banned_sentences.append(sent)
            else:
                clean_lines.append(line)

        sanitized_content = "\n".join(clean_lines).strip()
        return sanitized_content, headwords, banned_sentences

    @classmethod
    def _sanitize_grammar_for_quiz(cls, grammar_content: str) -> Tuple[str, List[str]]:
        """
        Physically isolates the quiz generator from grammar example sentences:
        Strips '- **Quote**:' and '- **Imitation Example**:' lines,
        preserving Pattern Formula, Pedagogical Function, Common Mistakes, and CEFR level.
        Also extracts a concise, high-density summary list of grammar patterns & common mistakes.
        """
        if not grammar_content:
            return "", []

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

        flush_pattern()

        return "\n".join(clean_lines).strip(), pattern_summaries

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
    def shuffle_quiz_options(cls, quiz_obj: Any) -> Any:
        """
        Sanitizes options and ensures robust answer key distribution.
        If the LLM already naturally randomized the correct_answer_index across options (balanced spread),
        the natural options order and explanations are preserved intact.
        If answers are heavily biased (e.g. all 0s or >50% on a single index), options are shuffled
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
            elif dataclasses.is_dataclass(q_item):
                flattened_questions.append(q_item)
        questions = flattened_questions

        # Filter out malformed items (e.g. naked strings, items missing options or with empty options)
        valid_questions = []
        for q in questions:
            if isinstance(q, dict):
                opts = q.get("options")
                has_stem = q.get("question") or q.get("translated_sentence")
                if isinstance(opts, list) and len(opts) > 0 and has_stem:
                    valid_questions.append(q)
            elif dataclasses.is_dataclass(q):
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
            # Handle both dict and dataclass
            q_dict = q if isinstance(q, dict) else (dataclasses.asdict(q) if dataclasses.is_dataclass(q) else {})
            
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

            # 3. Fallback heuristic: If expected answer did not resolve matching_idx,
            # detect if the explanation/audit unambiguously declares a specific option letter as the correct one.
            if matching_idx is None and options:
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
                        # Map old declared_idx to actual correct_idx
                        repair_map = {declared_idx: correct_idx}
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

    @classmethod
    def audit_quiz_integrity(
        cls,
        quiz_obj: Any,
        banned_sentences: List[str] = None,
        unit_headwords: List[str] = None,
        strict_distractor_recycling: bool = True
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

            # -1. Target Word Uniqueness & Unit Glossary Membership Gate
            if target and ("target_word" in q_dict or "word" in q_dict):
                if target in seen_targets:
                    flagged_indices.add(idx)
                    defect_messages.append(
                        f"Item #{idx + 1} ('{target}'): Duplicate target word detected! Already tested in Item #{seen_targets[target] + 1}."
                    )
                else:
                    seen_targets[target] = idx

                if headword_set and target not in headword_set:
                    # Check inflectional stem
                    t_stem = re.sub(r'(?:ed|ing|s|es|ly|tion|ment)$', '', target)
                    in_set = any((hw == target or (len(t_stem) >= 4 and hw.startswith(t_stem))) for hw in headword_set)
                    if not in_set:
                        flagged_indices.add(idx)
                        defect_messages.append(
                            f"Item #{idx + 1} ('{target}'): Target word is not in the authorized unit vocabulary list."
                        )

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

                # Flag any remaining verbatim target leak in the stem outside the blank
                if target:
                    esc_target = re.escape(target)
                    stem_no_blank = re.sub(r'_{2,}', '', stem)
                    if re.search(rf"\b{esc_target}\b", stem_no_blank, flags=re.IGNORECASE):
                        flagged_indices.add(idx)
                        defect_messages.append(
                            f"Item #{idx + 1} ('{target}'): Target word leaks verbatim into stem text outside blank."
                        )

            # 1. Stem Copying Gate (Zero-Tolerance)
            if stem and banned:
                clean_stem = re.sub(r'_{2,}', ' ', stem)
                for b_sent in banned:
                    overlap_len = cls._max_consecutive_word_overlap(clean_stem, b_sent)
                    if overlap_len >= 7:
                        flagged_indices.add(idx)
                        defect_messages.append(
                            f"Item #{idx + 1} ('{target}'): Stem copies {overlap_len} consecutive words "
                            f"from input example/source: \"{b_sent[:60]}...\""
                        )
                        break

            # 2. In-List Distractor Recycling Gate
            if options and headword_set and strict_distractor_recycling:
                recycled_distractors = []
                for o_idx, opt in enumerate(options):
                    if o_idx == correct_idx:
                        continue
                    opt_clean = str(opt).strip().lower()
                    # If distractor is derived from this item's own target keyword, exempt it
                    if target and (
                        opt_clean == target or
                        target in opt_clean or
                        (len(target) >= 4 and opt_clean.startswith(target[:4]))
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

        return sorted(list(flagged_indices)), defect_messages

    @classmethod
    def audit_reading_integrity(
        cls,
        quiz_obj: Any,
        passage_text: str = ""
    ) -> Tuple[List[int], List[str]]:
        """
        Level 1 Deterministic Code Gate for Reading Comprehension Quiz.
        Enforces 5 physical ground-truth invariants:
        1. Skill Diversity: Balanced mix of Main Idea, Detail/Recall, Inference, Author's Tone/Purpose.
        2. Structural Option Bounds: Exactly 4 distinct, parallel, non-empty options.
        3. Answer Index Integrity: Bound [0, 3] check.
        4. Verbatim Text Anchoring & Anti-Hallucination: context_sentence must physically exist in passage.
        5. Trivia / Option Echo Filter: Prevent trivially verbatim options or stem-option duplication.
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

            # 1.4 Check Key Index Bounds
            if not isinstance(correct_idx, int) or correct_idx not in (0, 1, 2, 3):
                flagged_indices.add(idx)
                defect_messages.append(f"Reading Item #{idx + 1}: Invalid correct_answer_index ({correct_idx})")

            # 1.5 Check Stem Quality
            if len(stem.split()) < 4:
                flagged_indices.add(idx)
                defect_messages.append(f"Reading Item #{idx + 1}: Question stem is too short or empty: \"{stem}\"")

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
        transcript_text: str = ""
    ) -> Tuple[List[int], List[str]]:
        """
        Level 1 Deterministic Code Gate for Video Comprehension Quiz.
        Enforces 4 physical ground-truth invariants:
        1. Structural Option Bounds: Exactly 4 distinct, non-empty options.
        2. Answer Index Integrity: Bound [0, 3] check.
        3. Timestamp Format & Grounding: Valid [MM:SS] or [HH:MM:SS] and physically anchored in transcript.
        4. Trivia / Option Echo Filter: Prevent trivially verbatim options or stem-option duplication.
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

            # 2. Key Index Bounds
            if not isinstance(correct_idx, int) or correct_idx not in (0, 1, 2, 3):
                flagged_indices.add(idx)
                defect_messages.append(f"Video Item #{idx + 1}: Invalid correct_answer_index ({correct_idx})")

            # 3. Stem Quality
            if len(stem.split()) < 4:
                flagged_indices.add(idx)
                defect_messages.append(f"Video Item #{idx + 1}: Question stem is too short or empty: \"{stem}\"")

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
    def audit_listening_integrity(
        cls,
        quiz_obj: Any,
        script_text: str = ""
    ) -> Tuple[List[int], List[str]]:
        """
        Level 1 Deterministic Code Gate for Listening Comprehension Quiz.
        Enforces 4 physical ground-truth invariants:
        1. Dialogue Script Integrity: At least 4 speaker turns in script.
        2. Structural Option Bounds: Exactly 4 distinct, non-empty options.
        3. Answer Index Integrity: Bound [0, 3] check.
        4. Category Integrity: Detail, Main Idea, or Inference.
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

        # 1. Script checks
        script_items = quiz_obj.get("script") if isinstance(quiz_obj, dict) else getattr(quiz_obj, "script", None)
        if script_items is not None and isinstance(script_items, list):
            if len(script_items) < 4:
                defect_messages.append(f"Listening Script Gate: Dialogue script contains fewer than 4 turns ({len(script_items)} turns).")

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

            # 2.2 Key Index Bounds
            if not isinstance(correct_idx, int) or correct_idx not in (0, 1, 2, 3):
                flagged_indices.add(idx)
                defect_messages.append(f"Listening Item #{idx + 1}: Invalid correct_answer_index ({correct_idx})")

            # 2.3 Stem Quality
            if len(stem.split()) < 4:
                flagged_indices.add(idx)
                defect_messages.append(f"Listening Item #{idx + 1}: Question stem is too short or empty: \"{stem}\"")

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
        target_language: str = "Chinese"
    ) -> Tuple[List[int], List[str]]:
        """
        Level 1 Deterministic Code Gate for Target-to-English Translation Quiz.
        Enforces 5 physical ground-truth invariants:
        1. Language Purity Gate:
           - {target_language} prompt sentence must contain valid {target_language} characters.
           - Options must be 100% English sentences, strictly free of {target_language} characters.
        2. Target Vocabulary Presence Gate:
           - The declared target_word (from design_audit or unit wordlist) must physically appear
             verbatim or with inflection in correct_english_answer.
        3. Grammar Formula Anchor Gate:
           - Checks that the grammatical formula/keywords declared in design_audit exist in correct_english_answer.
        4. Structural Parallelism Gate:
           - Exactly 4 options, non-empty, distinct (no duplicates).
           - Key index within bounds [0, 3].
           - Option lengths balanced (longest option not > 2.5x shortest option).
        5. Option Echo / Giveaway Filter:
           - No option is a trivial copy of the {target_language} prompt or design audit leak.
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
        else:
            # Default fallback: check for CJK or non-ASCII characters
            target_char_pattern = r'[\u4e00-\u9fff]|[^\x00-\x7F]'

        # Inflectional / Stemming helper for target word matching
        def _word_in_text(word: str, text: str) -> bool:
            w = word.strip().lower()
            t = text.lower()
            if not w or not t:
                return False
            # Clean placeholders like "sth", "sb", "something", "someone" (e.g. "impose sth on sb" -> "impose on")
            w_clean = re.sub(r'\b(sth|sb|something|someone|somebody|oneself)\b', '', w)
            w_clean = ' '.join(w_clean.split())
            if w_clean != w:
                w = w_clean
            # Multi-word phrase check
            if " " in w:
                # If exact phrase is present
                if w in t:
                    return True
                # Check with flexible gap between words (e.g. "impose ... on")
                parts = [re.escape(p) for p in w.split() if len(p) > 1]
                if len(parts) >= 2:
                    pattern = r'\b' + r'\b.*?\b'.join(parts) + r'\b'
                    if re.search(pattern, t):
                        return True
            # Exact word boundary
            if re.search(rf"\b{re.escape(w)}\b", t):
                return True
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
            # Inflectional variants check (e.g. s, es, ed, ing, d, ly)
            w_stem = re.sub(r'(?:ed|ing|s|es|ly|tion|ment|ness|ity|ive|able|al)$', '', w)
            if len(w_stem) >= 4:
                if re.search(rf"\b{re.escape(w_stem)}\w*\b", t):
                    return True
            return False

        for idx, q in enumerate(questions):
            q_dict = q if isinstance(q, dict) else (dataclasses.asdict(q) if dataclasses.is_dataclass(q) else {})
            stem = str(q_dict.get("translated_sentence", "")).strip()
            options = q_dict.get("options") or []
            correct_idx = q_dict.get("correct_answer_index")
            correct_eng = str(q_dict.get("correct_english_answer", "")).strip()
            audit_str = str(q_dict.get("design_audit", "")).strip()

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
                if re.search(target_char_pattern, opt_str):
                    flagged_indices.add(idx)
                    defect_messages.append(
                        f"Translation Item #{idx + 1}: Option [{chr(65 + o_i)}] contains {target_language} characters: \"{opt_str[:40]}...\""
                    )

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
                if re.search(target_char_pattern, opt_str):
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

            # 2.3 Answer index range
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

    def run_pipeline(self, source_filename: str, categories: List[str] = None) -> Tuple[str, List[str]]:
        """
        Modular pipeline for processing a raw unit file.
        Runs Vocabulary, Grammar, and Concept extractions in parallel for efficiency.
        """
        from .config import normalize_name
        import shutil

        input_path = Path(source_filename)
        wiki_dir = self.config.wiki_content_path

        if input_path.exists() and input_path.is_file():
            # Explicit file path on disk (either absolute or relative)
            input_stem = input_path.stem
            normalized_input = normalize_name(input_stem)

            # Determine the canonical unit folder name
            unit_folder_name = None
            try:
                resolved_input = input_path.resolve()
                resolved_wiki = wiki_dir.resolve()
                if resolved_input.is_relative_to(resolved_wiki):
                    rel_parts = resolved_input.relative_to(resolved_wiki).parts
                    if len(rel_parts) >= 3 and rel_parts[1] == "sources":
                        unit_folder_name = rel_parts[0]
            except Exception:
                pass

            if not unit_folder_name:
                unit_folder_name = input_stem.replace(" ", "_")
                if wiki_dir.exists():
                    for child in wiki_dir.iterdir():
                        if child.is_dir() and normalize_name(child.name) == normalized_input:
                            unit_folder_name = child.name
                            break

            # Define paths
            unit_dir = wiki_dir / unit_folder_name
            sources_dir = unit_dir / "sources"
            extractions_dir = unit_dir / "extractions"
            handouts_dir = unit_dir / "handouts"

            sources_dir.mkdir(parents=True, exist_ok=True)
            extractions_dir.mkdir(parents=True, exist_ok=True)
            handouts_dir.mkdir(parents=True, exist_ok=True)

            # Compile directly from the original explicit file path without copying to sources_dir
            source_path = input_path
        else:
            # Treated as a unit name or filename under existing units
            input_stem = Path(source_filename).stem
            normalized_input = normalize_name(source_filename)

            unit_folder_name = None
            if wiki_dir.exists():
                for child in wiki_dir.iterdir():
                    if child.is_dir() and (normalize_name(child.name) == normalized_input or normalize_name(child.name) == normalize_name(input_stem)):
                        unit_folder_name = child.name
                        break

            if not unit_folder_name:
                return f"Error: Source file or unit folder '{source_filename}' not found.", []

            unit_dir = wiki_dir / unit_folder_name
            sources_dir = unit_dir / "sources"
            extractions_dir = unit_dir / "extractions"
            handouts_dir = unit_dir / "handouts"

            sources_dir.mkdir(parents=True, exist_ok=True)
            extractions_dir.mkdir(parents=True, exist_ok=True)
            handouts_dir.mkdir(parents=True, exist_ok=True)

            # Locate the source file in sources_dir
            text_sources = [f for f in sources_dir.iterdir() if f.is_file() and f.suffix in [".md", ".txt"]]
            if len(text_sources) == 1:
                source_path = text_sources[0]
            elif len(text_sources) > 1:
                matching = [f for f in text_sources if normalize_name(f.stem) == normalized_input]
                source_path = matching[0] if matching else text_sources[0]
            else:
                return f"Error: No source markdown or text file found under unit sources for {unit_folder_name}.", []

        with open(source_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Plan 1 (Single Source): Parse optional syllabus sections directly from source markdown
        clean_content, syllabus_vocab, syllabus_grammar, syllabus_expressions = self.parse_syllabus_sections(content)
        self._last_source_content = clean_content or content
        file_stem = unit_folder_name
        all_saved = []
        
        # Determine extraction counts from compile_defaults config (priority: wiki_config.json -> config.py DEFAULT_CONFIG -> fallback)
        compile_defaults = self.config.get("compile_defaults") or {}
        v_count = compile_defaults.get("vocabulary", 20)
        e_count = compile_defaults.get("expressions", 5)
        g_count = compile_defaults.get("grammar", 5)
        c_count = compile_defaults.get("concepts", 3)
        max_p = compile_defaults.get("max_parallel", 3)

        # 1. Prepare Extraction Tasks
        v_prompt_template, v_schema = Prompts.get("extract_vocabulary")
        e_prompt_template, e_schema = Prompts.get("extract_expressions")
        g_prompt_template, g_schema = Prompts.get("extract_grammar")
        s_prompt_template, s_schema = Prompts.get("extract_summary")
        m_prompt_template, m_schema = Prompts.get("extract_mindmap")

        v_syllabus_sec = ""
        if syllabus_vocab:
            logger.info(f"📋 Detected {len(syllabus_vocab)} syllabus vocabulary/phrase item(s) in source markdown.")
            vocab_bullets = "\n".join([f"- {w}" for w in syllabus_vocab])
            v_syllabus_sec = (
                f"\n### TARGET VOCABULARY LIST ###\n"
                f"The text has {len(syllabus_vocab)} syllabus candidate items:\n"
                f"{vocab_bullets}\n\n"
                f"From this syllabus list, prioritize and select the most essential, pedagogically significant academic vocabulary (up to {v_count} words total) that appear in the passage below. For each selected word, find its authentic verbatim sentence in the passage.\n\n"
            )

        e_syllabus_sec = ""
        if syllabus_expressions:
            logger.info(f"📋 Detected {len(syllabus_expressions)} syllabus expression/phrase item(s) in source markdown.")
            expr_bullets = "\n".join([f"- {e}" for e in syllabus_expressions])
            e_syllabus_sec = (
                f"\n### TARGET EXPRESSIONS LIST ###\n"
                f"The text has {len(syllabus_expressions)} syllabus multi-word candidate items:\n"
                f"{expr_bullets}\n\n"
                f"From these syllabus expressions, prioritize and extract genuine expressions (up to {e_count} expressions total) that appear in the passage below. For each selected expression, derive its canonical slotted base form in design_audit and copy it to 'word'.\n\n"
            )

        g_syllabus_sec = ""
        if syllabus_grammar:
            logger.info(f"📋 Detected {len(syllabus_grammar)} syllabus grammar pattern(s) in source markdown.")
            grammar_bullets = "\n".join([f"- {g}" for g in syllabus_grammar])
            g_syllabus_sec = (
                f"\n### TARGET GRAMMAR TOPICS ###\n"
                f"The text has {len(syllabus_grammar)} syllabus grammar pattern candidates:\n"
                f"{grammar_bullets}\n\n"
                f"From these syllabus topics, prioritize and extract the most prominent advanced grammar patterns (up to {g_count} patterns total) from the text. For each pattern, find its exact verbatim quote in the passage and formulate its structural blueprint.\n\n"
            )

        v_kwargs = {"content": clean_content or content, "count": v_count, "syllabus_section": v_syllabus_sec}
        e_kwargs = {"content": clean_content or content, "count": e_count, "syllabus_section": e_syllabus_sec}
        g_kwargs = {"content": clean_content or content, "count": g_count, "syllabus_section": g_syllabus_sec}
        s_kwargs = {"content": clean_content or content, "count": c_count}
        m_kwargs = {"content": clean_content or content}

        # Format prompts directly via explicit placeholders
        v_prompt_formatted = v_prompt_template.format(**v_kwargs)
        e_prompt_formatted = e_prompt_template.format(**e_kwargs)
        g_prompt_formatted = g_prompt_template.format(**g_kwargs)

        tasks = [
            ("vocabulary", v_prompt_formatted, self._interpolate_schema(v_schema, v_kwargs)),
            ("expressions", e_prompt_formatted, self._interpolate_schema(e_schema, e_kwargs)),
            ("grammar", g_prompt_formatted, self._interpolate_schema(g_schema, g_kwargs)),
            ("summary", s_prompt_template.format(**s_kwargs), self._interpolate_schema(s_schema, s_kwargs)),
            ("mindmap", m_prompt_template.format(**m_kwargs), self._interpolate_schema(m_schema, m_kwargs))
        ]

        if categories:
            normalized_cats = [c.lower() for c in categories]
            task_names_to_run = []
            for cat in normalized_cats:
                if cat == "vocabulary":
                    task_names_to_run.extend(["vocabulary", "expressions"])
                else:
                    task_names_to_run.append(cat)
            tasks = [t for t in tasks if t[0] in task_names_to_run]

        # Route extraction through the Prose-to-JSON pipeline.
        # Turn 1 = free natural-language selection (preserves native reasoning);
        # Turn 2 = deterministic packaging. Other extractions stay one-shot JSON.
        use_prose_vocab = self.config.get("enable_vocab_prose", False)
        vocab_requested = any(name == "vocabulary" for name, _, _ in tasks)
        if use_prose_vocab and vocab_requested:
            tasks = [t for t in tasks if t[0] != "vocabulary"]

        use_prose_grammar = self.config.get("enable_grammar_prose", False)
        grammar_requested = any(name == "grammar" for name, _, _ in tasks)
        if use_prose_grammar and grammar_requested:
            tasks = [t for t in tasks if t[0] != "grammar"]

        # 2. Run extractions in parallel
        results = []
        try:
            with ThreadPoolExecutor(max_workers=max_p) as executor:
                futures = {
                    executor.submit(llm.chat, [{"role": "user", "content": prompt}], schema=schema, task_name=f"extract_{name}_{file_stem}"): name 
                    for name, prompt, schema in tasks
                }
                failed_tasks = []
                for future in futures:
                    name = futures[future]
                    try:
                        data = future.result()
                        if data:
                            results.append((name, data))
                    except Exception as task_err:
                        failed_tasks.append((name, str(task_err)))
                        logger.error(f"Extraction task '{name}' failed for {file_stem}: {task_err}", exc_info=True)

                # Vocabulary runs its own 2-turn Prose-to-JSON pipeline
                if use_prose_vocab and vocab_requested:
                    try:
                        logger.info(f"🚀 Vocabulary extraction via Prose-to-JSON pipeline ({file_stem})...")
                        vocab_data = self._run_vocab_prose_pipeline(
                            v_prompt_template.format(**v_kwargs),
                            v_kwargs,
                            self._interpolate_schema(v_schema, v_kwargs),
                            file_stem,
                        )
                        if vocab_data:
                            results.append(("vocabulary", vocab_data))
                    except Exception as vocab_err:
                        failed_tasks.append(("vocabulary", str(vocab_err)))
                        logger.error(f"Vocabulary prose pipeline failed for {file_stem}: {vocab_err}", exc_info=True)

                # Grammar runs its own 2-turn Prose-to-JSON pipeline
                if use_prose_grammar and grammar_requested:
                    try:
                        logger.info(f"🚀 Grammar extraction via Prose-to-JSON pipeline ({file_stem})...")
                        grammar_data = self._run_grammar_prose_pipeline(
                            g_prompt_template.format(**g_kwargs),
                            g_kwargs,
                            self._interpolate_schema(g_schema, g_kwargs),
                            file_stem,
                        )
                        if grammar_data:
                            results.append(("grammar", grammar_data))
                    except Exception as grammar_err:
                        failed_tasks.append(("grammar", str(grammar_err)))
                        logger.error(f"Grammar prose pipeline failed for {file_stem}: {grammar_err}", exc_info=True)

            # 3. Process, Merge, and Save Results
            vocab_data = None
            expressions_data = None
            other_results = []

            for category, data in results:
                if category == "vocabulary":
                    vocab_data = data
                elif category == "expressions":
                    expressions_data = data
                else:
                    other_results.append((category, data))

            # Merge expressions into vocabulary if both exist
            if vocab_data:
                merged_vocab = []
                if dataclasses.is_dataclass(vocab_data):
                    merged_vocab.extend(getattr(vocab_data, "vocabulary", []))
                elif isinstance(vocab_data, dict):
                    merged_vocab.extend(vocab_data.get("vocabulary", []))

                if expressions_data:
                    def _resolve_slotted_word(expr_obj):
                        w = expr_obj.get("word", "") if isinstance(expr_obj, dict) else getattr(expr_obj, "word", "")
                        aud = expr_obj.get("design_audit", "") if isinstance(expr_obj, dict) else getattr(expr_obj, "design_audit", "")
                        valid_slot_pattern = r'\[(something|somebody|someone|one\'s|one|entity|domain|factor|doing something|clause|[a-z_]+)\]|\bone\'s\b'
                        if ("[" not in w and "one's" not in w) and ("[" in aud or "one's" in aud):
                            parts = [p.strip() for p in aud.replace("->", "➔").split("➔")]
                            candidate_steps = parts[1:] if len(parts) > 1 else parts
                            for p in candidate_steps:
                                if ("[" in p or "one's" in p):
                                    cand = re.sub(r'^(?:AUDIT|DRAFT|STEP\s*\d*)\s*:\s*', '', p, flags=re.IGNORECASE).strip()
                                    candidate = cand.split(" -")[0].split(" (")[0].strip()
                                    if candidate.startswith("[") and candidate.endswith("]"):
                                        inner = candidate[1:-1].strip()
                                        if not re.search(valid_slot_pattern, inner, re.IGNORECASE):
                                            continue
                                    if not re.search(valid_slot_pattern, candidate, re.IGNORECASE):
                                        continue
                                    cand_tokens = re.findall(r'[a-zA-Z]+', candidate.replace("[", "").replace("]", ""))
                                    w_tokens = re.findall(r'[a-zA-Z]+', w)
                                    if cand_tokens and w_tokens and cand_tokens[0].lower() == w_tokens[0].lower():
                                        return candidate
                        return w

                    expr_list = getattr(expressions_data, "expressions", []) if dataclasses.is_dataclass(expressions_data) else expressions_data.get("expressions", [])
                    for expr in expr_list:
                        resolved_word = _resolve_slotted_word(expr)
                        while resolved_word.startswith("[") and resolved_word.endswith("]"):
                            depth = 0
                            matched_end = False
                            for idx, char in enumerate(resolved_word):
                                if char == "[": depth += 1
                                elif char == "]":
                                    depth -= 1
                                    if depth == 0:
                                        if idx == len(resolved_word) - 1: matched_end = True
                                        break
                            if matched_end:
                                resolved_word = resolved_word[1:-1].strip()
                            else:
                                break
                        if isinstance(expr, dict):
                            mapped_item = {
                                "design_audit": expr.get("design_audit", ""),
                                "word": resolved_word,
                                "part_of_speech": expr.get("part_of_speech", "phrasal verb"),
                                "definition": expr.get("definition", ""),
                                "word_cefr_level": expr.get("word_cefr_level", "B2"),
                                "quoted_sentence": expr.get("quoted_sentence", ""),
                                "example_usage": expr.get("example_usage", ""),
                            }
                        else:
                            from .schemas import VocabularyItem
                            mapped_item = VocabularyItem(
                                design_audit=getattr(expr, "design_audit", ""),
                                word=resolved_word,
                                part_of_speech=getattr(expr, "part_of_speech", "phrasal verb"),
                                definition=getattr(expr, "definition", ""),
                                word_cefr_level=getattr(expr, "word_cefr_level", "B2"),
                                quoted_sentence=getattr(expr, "quoted_sentence", ""),
                                example_usage=getattr(expr, "example_usage", ""),
                            )
                        merged_vocab.append(mapped_item)

                if dataclasses.is_dataclass(vocab_data):
                    vocab_data.vocabulary = merged_vocab
                elif isinstance(vocab_data, dict):
                    vocab_data["vocabulary"] = merged_vocab

                all_saved.extend(self._save_extraction_results(vocab_data, source_path.name, category_override="vocabulary"))
            elif expressions_data:
                from .schemas import VocabularyExtraction, VocabularyItem
                vocab_list = []
                expr_list = getattr(expressions_data, "expressions", []) if dataclasses.is_dataclass(expressions_data) else expressions_data.get("expressions", [])
                for expr in expr_list:
                    resolved_word = _resolve_slotted_word(expr) if '_resolve_slotted_word' in locals() else getattr(expr, "word", "")
                    vocab_list.append(VocabularyItem(
                        design_audit=getattr(expr, "design_audit", ""),
                        word=resolved_word,
                        part_of_speech=getattr(expr, "part_of_speech", "phrasal verb"),
                        definition=getattr(expr, "definition", ""),
                        word_cefr_level=getattr(expr, "word_cefr_level", "B2"),
                        quoted_sentence=getattr(expr, "quoted_sentence", ""),
                        example_usage=getattr(expr, "example_usage", ""),
                    ))
                v_extracted = VocabularyExtraction(
                    title=f"{file_stem.replace('_', ' ')} Vocabulary",
                    overall_cefr_level="B2",
                    vocabulary=vocab_list
                )
                all_saved.extend(self._save_extraction_results(v_extracted, source_path.name, category_override="vocabulary"))

            # Save other results (grammar, summary)
            for category, data in other_results:
                all_saved.extend(self._save_extraction_results(data, source_path.name, category_override=category))

            if failed_tasks:
                failed_names = ", ".join(name for name, _ in failed_tasks)
                if all_saved:
                    return f"Pipeline completed with warnings for {source_filename}. Generated {len(all_saved)} files (failed tasks: {failed_names}).", all_saved
                else:
                    return f"Pipeline failed for {source_filename}. Failed tasks: {failed_names}.", all_saved

            return f"Pipeline completed for {source_filename}. Generated {len(all_saved)} files.", all_saved

        except Exception as e:
            return f"Pipeline Error: {e}", all_saved

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

        if template_name == "vocabulary":
            raw_vocab = data.get("content", "")
            sanitized_vocab, unit_headwords, banned_quiz_sentences = self._sanitize_vocab_for_quiz(raw_vocab)
            kwargs["vocabulary_content"] = sanitized_vocab
            kwargs["cefr_level"] = data.get("cefr_level", "B2")
        elif template_name == "reading":
            kwargs["passage_content"] = data["passage"]
            kwargs["cefr_level"] = data.get("cefr_level", "B2")
        elif template_name == "translation":
            raw_v = data.get("vocab_list", "")
            sanitized_v, unit_headwords, banned_quiz_sentences = self._sanitize_vocab_for_quiz(raw_v)
            raw_g = data.get("grammar_list", "")
            sanitized_g, unit_grammar_patterns = self._sanitize_grammar_for_quiz(raw_g)
            kwargs["vocabulary_content"] = sanitized_v
            kwargs["grammar_content"] = sanitized_g
            kwargs["target_language"] = self.config.get("target_language") or "Chinese"
            kwargs["cefr_level"] = data.get("cefr_level", "B2")
        elif template_name == "listening":
            raw_v = data.get("vocab_list", "")
            sanitized_v, unit_headwords, banned_quiz_sentences = self._sanitize_vocab_for_quiz(raw_v)
            kwargs["vocabulary_content"] = sanitized_v
            kwargs["cefr_level"] = data.get("cefr_level", "B2")

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
            from .tts import tts_service
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
            kwargs["cefr_level"] = data.get("cefr_level", "B2")

        prompt = prompt_template.format(**kwargs)

        # 4. Call LLM (Turn 1 Prose Drafting -> Turn 2 Schema Packaging or direct JSON)
        try:
            enable_prose = self.config.get("enable_prose_pipeline", True)
            interpolated_schema = self._interpolate_schema(schema_cls, kwargs)

            if enable_prose and template_name in ("vocabulary", "reading", "translation"):
                logger.info(f"🚀 Executing Prose-to-JSON Pipeline for {template_name} ({core_name})...")
                expected_count = kwargs.get("count", 5)
                # Turn 1: Standard natural language assessment generation
                if template_name == "reading":
                    prose_instructions = (
                        f"{prompt}\n\n"
                        "### GENERATION FORMAT MANDATE:\n"
                        "Write out the reading assessment strictly using this clean, structured text format:\n\n"
                        "VOCABULARY LIST (5 to 8 challenging academic words directly from the passage):\n"
                        "- Word: [target headword]\n"
                        "- Context Sentence: \"[exact verbatim sentence from the passage]\"\n"
                        "- Part of Speech: [noun/verb/adjective/adverb/preposition/conjunction/interjection]\n"
                        "- Definition: [concise contextual meaning]\n"
                        "- Example Usage: [original academic illustrative example]\n\n"
                        "COMPREHENSION QUESTIONS:\n"
                        "Item 1:\n"
                        "- Category: [Main Idea | Detail/Recall | Inference | Author's Tone/Purpose]\n"
                        "- Question: [clear, intellectually mature reading question stem]\n"
                        "- Options:\n"
                        "  A. option text\n"
                        "  B. option text\n"
                        "  C. option text\n"
                        "  D. option text\n"
                        "- Correct Answer: [A, B, C, or D - distribute keys evenly across items]\n"
                        "- Text Anchor: [exact paragraph and quoted statement supporting the answer]\n"
                        "- Explanation: [objective reason why the key is correct and why each distractor fails]\n"
                        "- Design Audit: AUDIT: [Category] -> [Textual Anchor] -> [Trap 1 (Literal Match): ...] [Trap 2 (Scope Shift): ...] [Trap 3 (Distortion): ...] -> [Why Distractors Fail]\n\n"
                        "Item 2:\n"
                        "...\n\n"
                        "Ensure all questions follow this exact item layout consecutively."
                    )
                elif template_name == "translation":
                    tgt_lang = kwargs.get("target_language") or "Chinese"
                    prose_instructions = (
                        f"{prompt}\n\n"
                        f"### GENERATION FORMAT MANDATE ({tgt_lang}-to-English COMPARATIVE TRANSLATION APPRAISAL DRAFT):\n"
                        f"Output all {expected_count} items directly and consecutively using this clean, structured format.\n"
                        "🚫 DO NOT include pre-analysis, stream-of-consciousness deliberations, self-correction dialogues, or repetitive drafts. Begin immediately with 'Item 1:' and write out the items cleanly:\n\n"
                        "Item 1:\n"
                        "- Target Keyword: [exact vocabulary headword directly from the VOCABULARY list]\n"
                        "- Target Grammar Pattern: [grammar pattern formula from list]\n"
                        f"- {tgt_lang} Sentence: [formal, natural, polished source sentence]\n"
                        "- Idiomatic Translation: [complete, flawless, publishable academic English translation featuring the target keyword]\n"
                        "- Flawed Translation: [typical plausible learner/machine translation containing a specific Chinglish or structural error]\n"
                        "- Flaw Type: [concise defect label, e.g., Chinglish literal syntax, wrong dependent preposition, formula breakdown]\n"
                        "- Diagnostic Critique: [contrastive pedagogical explanation comparing why the idiomatic version works and identifying the exact rule violated by the flawed translation]\n"
                        f"- Design Audit: AUDIT: [{tgt_lang} Anchor -> Target Keyword: [word] + Grammar Formula] -> [Idiomatic Core: ...] -> [Flaw Type: ...] -> [Pedagogical Takeaway]\n\n"
                        "Item 2:\n"
                        "...\n\n"
                        f"Ensure all {expected_count} questions follow this exact item layout consecutively."
                    )
                else:
                    prose_instructions = (
                        f"{prompt}\n\n"
                        "### GENERATION FORMAT MANDATE:\n"
                        "Write out the assessment items strictly using this clean, structured format (do NOT use markdown tables or repetitive outlines):\n\n"
                        "Item 1:\n"
                        "- Target: [exact word or phrase from the list, without part of speech in parentheses]\n"
                        "- Question: [academic sentence with strictly four underscores '____' for the blank]\n"
                        "- Options:\n"
                        "  A. option text\n"
                        "  B. option text\n"
                        "  C. option text\n"
                        "  D. option text\n"
                        "- Correct Answer: [A, B, C, or D - distribute keys evenly across items]\n"
                        "- Definition: [concise definition in this context - do NOT append any notes or remarks]\n"
                        "- Explanation: [objective reason why the key fits and why each distractor fails]\n"
                        "- Design Audit: AUDIT: [Target] -> [Sentence Clues & Syntactic Slot] -> [Trap 1: ...] [Trap 2: ...] [Trap 3: ...] -> [Why Distractors Fail]\n\n"
                        "Item 2:\n"
                        "...\n\n"
                        "Ensure all questions follow this exact item layout consecutively."
                    )

                prose_draft = llm.chat(
                    [{"role": "user", "content": prose_instructions}],
                    json_format=False,
                    task_name=f"quiz_{template_name}_{core_name}_turn1_prose"
                )

                if prose_draft:
                    logger.info(f"📦 Packaging Turn 1 prose draft into structured JSON Schema...")
                    expected_count = kwargs.get("count", 5)
                    if template_name == "reading":
                        packaging_prompt = (
                            "You are a deterministic assessment data converter.\n"
                            "Faithfully convert the following reading assessment draft into the required ReadingQuiz JSON schema format.\n"
                            "MANDATES:\n"
                            f"- ⚠️ CRITICAL FULL COMPLETION MANDATE: You MUST convert ALL {expected_count} questions (Item 1 through Item {expected_count}) provided below consecutively into the 'questions' list. NEVER omit items or stop early!\n"
                            "- Convert 'Correct Answer: A/B/C/D' into the 0-based integer 'correct_answer_index' (0 for A, 1 for B, 2 for C, 3 for D).\n"
                            "- Ensure 'options' contains the 4 choices as plain text strings without 'A.', 'B.', 'C.', 'D.' prefixes or quotes.\n"
                            "- Ensure 'category' is strictly one of: 'Main Idea', 'Detail/Recall', 'Inference', \"Author's Tone/Purpose\".\n"
                            "- Populate the 'vocabulary' array with 5 to 8 ReadingVocabItem objects extracted in the draft. Each 'context_sentence' MUST contain only the pure verbatim sentence quoted from the passage, without any leading prefixes like 'ASSESSMENT ITEMS:' or headings.\n"
                            "- Clean any accidental author notes, bracketed remarks like '(Note: ...)', or thinking process from definitions and explanations.\n"
                            "- Preserve all questions, stems, explanations, and design audits exactly as written.\n\n"
                            f"ASSESSMENT ITEMS:\n{prose_draft}"
                        )
                    elif template_name == "translation":
                        tgt_lang = kwargs.get("target_language") or "Chinese"
                        packaging_prompt = (
                            f"You are a deterministic {tgt_lang}-to-English comparative translation assessment data converter.\n"
                            "Faithfully convert the following translation assessment draft into the required TranslationQuiz JSON schema format.\n"
                            "MANDATES:\n"
                            f"- ⚠️ CRITICAL FULL COMPLETION MANDATE: You MUST convert ALL {expected_count} translation items (Item 1 through Item {expected_count}) provided below consecutively into the 'questions' list. NEVER omit items or stop early after Item 1!\n"
                            "- Map 'Target Keyword' to 'target_keyword' (the exact English vocabulary word tested).\n"
                            "- Map 'Target Grammar Pattern' to 'target_grammar' (the grammar pattern tested).\n"
                            f"- Map '{tgt_lang} Sentence' to 'translated_sentence' (the pure source sentence without field prefix).\n"
                            "- Map 'Idiomatic Translation' to 'idiomatic_translation' (the complete, pristine English translation).\n"
                            "- Map 'Flawed Translation' to 'flawed_translation' (the contrastive translation containing the defect).\n"
                            "- Map 'Flaw Type' to 'flaw_type'.\n"
                            "- Map 'Diagnostic Critique' to 'diagnostic_critique'.\n"
                            "- Map 'Diagnostic Critique' ALSO to 'explanation' for compatibility.\n"
                            "- Map 'Idiomatic Translation' ALSO to 'correct_english_answer' for compatibility.\n"
                            "- Set 'options' to an array of 2 strings: [\"<Idiomatic Translation>\", \"<Flawed Translation>\"].\n"
                            "- Set 'correct_answer_index' to 0.\n"
                            "- Map 'Design Audit' to 'design_audit'.\n"
                            "- Clean any accidental author notes, bracketed remarks like '(Note: ...)', or thinking process from diagnostic critiques.\n\n"
                            f"ASSESSMENT ITEMS:\n{prose_draft}"
                        )
                    else:
                        packaging_prompt = (
                            "You are a deterministic data converter.\n"
                            "Faithfully convert the following assessment items into the required JSON schema format.\n"
                            "MANDATES:\n"
                            f"- ⚠️ CRITICAL FULL COMPLETION MANDATE: You MUST convert ALL {expected_count} assessment items (Item 1 through Item {expected_count}) provided below consecutively into the 'questions' list. NEVER omit items or stop early!\n"
                            "- Convert 'Correct Answer: A/B/C/D' into the 0-based integer 'correct_answer_index' (0 for A, 1 for B, 2 for C, 3 for D).\n"
                            "- Ensure 'options' contains the 4 choices as plain text strings without 'A.', 'B.', 'C.', 'D.' prefixes or markdown bolding.\n"
                            "- Ensure 'target_word' contains only the pure target word or phrase, without part-of-speech annotations in parentheses.\n"
                            "- Clean any accidental author notes, bracketed remarks like '(Note: ...)', or thinking process from 'definition' and 'explanation'.\n"
                            "- Preserve all questions, stems, definitions, explanations, and design audits exactly as written.\n\n"
                            f"ASSESSMENT ITEMS:\n{prose_draft}"
                        )
                    quiz_obj = llm.chat(
                        [{"role": "user", "content": packaging_prompt}],
                        schema=interpolated_schema,
                        temperature=0.0,
                        task_name=f"quiz_{template_name}_{core_name}_turn2_package",
                        _disable_qa_retry=True
                    )

                    # Packaging completeness guard: verify all draft items were packaged
                    if quiz_obj:
                        q_items = quiz_obj.get("questions", []) if isinstance(quiz_obj, dict) else getattr(quiz_obj, "questions", [])
                        if len(q_items) < expected_count:
                            logger.warning(
                                f"⚠️ Packaging incomplete: expected {expected_count} questions, but converter only packaged {len(q_items)}. "
                                f"Triggering strict full-completion packaging retry..."
                            )
                            # Multi-turn compact recovery prompt: refer to previous draft and ask for all items
                            retry_messages = [
                                {"role": "user", "content": packaging_prompt},
                                {"role": "assistant", "content": json.dumps(quiz_obj, ensure_ascii=False) if isinstance(quiz_obj, dict) else str(quiz_obj)},
                                {
                                    "role": "user",
                                    "content": (
                                        f"### 🚨 PACKAGING INCOMPLETE DEFECT\n"
                                        f"- [ERROR]: Packaged only {len(q_items)} of {expected_count} items (omitted items {len(q_items) + 1} to {expected_count}).\n"
                                        f"- [LOOKUP]: Refer to `ASSESSMENT ITEMS:` in Turn 1.\n\n"
                                        f"🛑 MANDATE: Convert ALL {expected_count} items into the 'questions' array consecutively. Return ONLY the complete JSON object."
                                    )
                                }
                            ]
                            retry_obj = llm.chat(
                                retry_messages,
                                schema=interpolated_schema,
                                temperature=0.0,
                                task_name=f"quiz_{template_name}_{core_name}_turn2_package_retry",
                                _disable_qa_retry=True
                            )
                            if retry_obj:
                                retry_items = retry_obj.get("questions", []) if isinstance(retry_obj, dict) else getattr(retry_obj, "questions", [])
                                if len(retry_items) > len(q_items):
                                    quiz_obj = retry_obj
                else:
                    quiz_obj = None
            else:
                quiz_obj = llm.chat(
                    [{"role": "user", "content": prompt}],
                    schema=interpolated_schema,
                    task_name=f"quiz_{template_name}_{core_name}"
                )

            if not quiz_obj:
                return "Error: LLM returned empty quiz data."
            
            # 4.5. Randomize options
            quiz_obj = self._shuffle_quiz_options(quiz_obj)

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
                    from .expert_auditor import ExpertAuditor
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

                    # ---------------------------------------------------------
                    # Step 1: Pre-Audit Level 1 Code Gate (Sanitize & Surface Flaws)
                    # ---------------------------------------------------------
                    curr_tgt_lang = kwargs.get("target_language") or self.config.get("target_language") or "Chinese"
                    if template_name == "reading":
                        l1_defective, l1_defects = self.audit_reading_integrity(
                            quiz_dict_eval,
                            passage_text=source_context
                        )
                    elif template_name == "video":
                        l1_defective, l1_defects = self.audit_video_integrity(
                            quiz_dict_eval,
                            transcript_text=source_context
                        )
                    elif template_name == "listening":
                        l1_defective, l1_defects = self.audit_listening_integrity(
                            quiz_dict_eval,
                            script_text=source_context
                        )
                    elif template_name == "translation":
                        l1_defective, l1_defects = self.audit_translation_integrity(
                            quiz_dict_eval,
                            unit_headwords=unit_headwords,
                            target_language=curr_tgt_lang
                        )
                    else:
                        l1_defective, l1_defects = self.audit_quiz_integrity(
                            quiz_dict_eval,
                            banned_sentences=banned_quiz_sentences,
                            unit_headwords=unit_headwords
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
                        unit_grammar_patterns=unit_grammar_patterns
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
                                if "stem" in candidate_cure and "question" not in candidate_cure:
                                    candidate_cure["question"] = candidate_cure.pop("stem")

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
                                            normalized_opts.append(txt)
                                        else:
                                            normalized_opts.append(str(o_val))
                                    candidate_cure["options"] = normalized_opts
                                    if cand_corr_idx is not None:
                                        try:
                                            candidate_cure["correct_answer_index"] = int(cand_corr_idx)
                                        except (ValueError, TypeError):
                                            pass

                                orig_item = orig_questions[item_slot] if 0 <= item_slot < len(orig_questions) else {}
                                orig_dict = orig_item if isinstance(orig_item, dict) else (dataclasses.asdict(orig_item) if dataclasses.is_dataclass(orig_item) else {})

                                # Vocabulary fallback: inherit target_word / definition if omitted in cure
                                if template_name == "vocabulary":
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

                                # Validate candidate cure through Level 1 Code Gate
                                temp_wrap = {"questions": [candidate_cure]}
                                temp_wrap = self._shuffle_quiz_options(temp_wrap)
                                healed_item = temp_wrap["questions"][0]

                                # Modality-specific Level 1 gate on candidate cure
                                cure_l1_defects = []
                                if template_name == "reading":
                                    cure_l1_flagged, cure_l1_defects = self.audit_reading_integrity(
                                        {"questions": [healed_item]},
                                        passage_text=source_context
                                    )
                                elif template_name == "video":
                                    cure_l1_flagged, cure_l1_defects = self.audit_video_integrity(
                                        {"questions": [healed_item]},
                                        transcript_text=source_context
                                    )
                                elif template_name == "listening":
                                    cure_l1_flagged, cure_l1_defects = self.audit_listening_integrity(
                                        {"questions": [healed_item]},
                                        script_text=source_context
                                    )
                                elif template_name == "translation":
                                    cure_l1_flagged, cure_l1_defects = self.audit_translation_integrity(
                                        {"questions": [healed_item]},
                                        unit_headwords=unit_headwords,
                                        target_language=curr_tgt_lang
                                    )
                                else:
                                    cure_l1_flagged, cure_l1_defects = self.audit_quiz_integrity(
                                        {"questions": [healed_item]},
                                        banned_sentences=banned_quiz_sentences,
                                        unit_headwords=unit_headwords,
                                        strict_distractor_recycling=False
                                    )

                                if not cure_l1_flagged:
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

                        # Re-bind modality-specific properties
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
                        item_scores = [qa.get("pedagogical_score", 90) for qa in audit_items if isinstance(qa, dict)]
                        final_avg = round(sum(item_scores) / len(item_scores)) if item_scores else 100
                        audit_report["base_quality_score"] = final_avg

                        # Preserve judge's cap and fail verdict if uncured defects remain
                        has_fatal_residual = any(
                            qa.get("single_fit_valid") is False 
                            for qa in audit_items if isinstance(qa, dict)
                        ) or (cure_stats["discard"] > 0)

                        if has_fatal_residual:
                            audit_report["pass_audit"] = False
                            audit_report["overall_quality_score"] = min(final_avg, 70)
                        else:
                            audit_report["pass_audit"] = (final_avg >= 80)
                            audit_report["overall_quality_score"] = final_avg

                        audit_report["cure_stats"] = cure_stats

                        # Post-cure accuracy updates: successfully cured items now have a unique, verified key
                        total_items_count = len(orig_questions) or 1
                        valid_items_count = cure_stats["pass"] + cure_stats["repair"] + cure_stats["rewrite"]
                        audit_report["post_cure_accuracy"] = round(valid_items_count / total_items_count, 2)
                        audit_report["blind_solve_accuracy"] = audit_report["post_cure_accuracy"]

                        # Update summary verdict to disclose in-place surgical cure
                        cure_summary = (
                            f"[SURGICAL CURE APPLIED: Pre-Cure {raw_gen_score}% -> Post-Cure {audit_report['overall_quality_score']}%. "
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
                                score_block = (
                                    f"=== COMPOSITE_SCORE: {final_avg}.0% (CURED) ===\n"
                                    f"=== PRE_CURE_SCORE: {raw_gen_score}.0% ===\n"
                                    f"=== POST_CURE_SCORE: {final_avg}.0% ===\n"
                                    f"=== SURGICAL_CURE: Repaired={cure_stats['repair']}, Rewritten={cure_stats['rewrite']}, Passed={cure_stats['pass']}, Flawed={cure_stats['discard']} ==="
                                )
                                if "=== COMPOSITE_SCORE:" in log_txt:
                                    log_txt = re.sub(
                                        r"=== COMPOSITE_SCORE:\s*[\d\.]+%\s*===(?:\n=== L2_PENALTY:[^\n]+===)?",
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
                    from .tts import tts_service
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

    def ask_wiki(self, query):
        """RAG-lite for querying wiki content."""
        inventory = self._get_wiki_inventory()
        inventory_str = "\n".join([f"- {i['name']} ({i['type']})" for i in inventory])
        
        routing_prompt = f"""You are a Librarian. Select up to 5 relevant files for: {query}\nINVENTORY:\n{inventory_str}"""

        try:
            route = llm.chat([{"role": "user", "content": routing_prompt}], schema=RoutingResult, task_name="wiki_routing")
            selected_names = route.selected_files if route else []
            
            # Helper to normalize names for comparison (ignores spaces, underscores, and extension)
            def normalize_name(n):
                if n.lower().endswith('.md'):
                    n = n[:-3]
                return re.sub(r'[^a-zA-Z0-9]', '', n).lower()

            context_parts = []
            for name in selected_names:
                norm_name = normalize_name(name)
                item = next((i for i in inventory if normalize_name(i['name']) == norm_name), None)
                if item:
                    with open(item['path'], "r", encoding="utf-8") as f:
                        context_parts.append(f"--- FILE: {item['name']} ---\n{f.read()}")

            # Robust Fallback: If LLM selected nothing, or all selected names failed to match any inventory file,
            # use a smart local keyword matcher
            if not context_parts:
                # Pre-process query to separate digits (e.g. "4unit" -> "4 unit", "book4" -> "book 4")
                clean_query = re.sub(r'([0-9]+)', r' \1 ', query).lower()
                tokens = []
                # Keep words length >= 2, or any length if they are unit digits
                for word in re.findall(r'[a-zA-Z0-9]+', clean_query):
                    if len(word) >= 2 or word.isdigit():
                        tokens.append(word)
                # Keep Chinese characters
                for char in re.findall(r'[\u4e00-\u9fff]', query):
                    tokens.append(char)

                scored_items = []
                for item in inventory:
                    name_lower = item['name'].lower()
                    score = 0
                    
                    # Category indicators boost matched types
                    if "vocabulary" in tokens and item['type'] == "wiki_vocabulary":
                        score += 5
                    if "grammar" in tokens and item['type'] == "wiki_grammar":
                        score += 5
                    if "summary" in tokens and item['type'] == "wiki_summaries":
                        score += 5
                    
                    for t in tokens:
                        if t in name_lower:
                            score += 10
                            if t.isdigit():
                                score += 10  # heavy priority match on unit/book numbers
                    
                    if score > 15:
                        scored_items.append((item['name'], score))
                
                if scored_items:
                    scored_items.sort(key=lambda x: x[1], reverse=True)
                    # Take top 3 matching items to avoid bloating prompt context
                    top_matches = [name for name, _ in scored_items[:3]]
                    for name in top_matches:
                        item = next((i for i in inventory if i['name'] == name), None)
                        if item:
                            with open(item['path'], "r", encoding="utf-8") as f:
                                context_parts.append(f"--- FILE: {item['name']} ---\n{f.read()}")

            # Smart fallback for source/original text queries not yet resolved
            if not context_parts:
                query_lower = query.lower()
                is_source_query = any(
                    kw in query_lower
                    for kw in ["original text", "full text", "source text", "complete text",
                                "原文", "全文", "完整内容", "课文", "文章"]
                )

                if is_source_query:
                    # Extract unit identifier from query (e.g. "Book 4 Unit 5" -> "book_4_unit_5")
                    source_tokens = []
                    for word in re.findall(r'[a-zA-Z0-9]+', clean_query):
                        if len(word) >= 2 or word.isdigit():
                            source_tokens.append(word.lower())

                    # Collect names already used as context to avoid duplicates
                    existing_names = {re.sub(r'[^a-z0-9]', '', n).lower() for n in inventory}

                    seen_units = set()
                    for item in inventory:
                        if item['type'] != 'raw_source':
                            continue
                        # Check path and name for unit identifier tokens
                        item_path = str(item['path']).replace('\\', '/')
                        item_name = (item['name'] + " " + item_path).lower()
                        matched_tokens = [t for t in source_tokens if t in item_name]
                        if not matched_tokens:
                            continue

                        # Derive unit folder name from path (wiki/<unit>/sources/...)
                        wiki_idx = item_path.find('/wiki/')
                        if wiki_idx >= 0:
                            after_wiki = item_path[wiki_idx + 6:]
                            parts = after_wiki.split('/')
                            if len(parts) >= 3 and parts[1] == 'sources':
                                unit_folder = parts[0]
                            else:
                                continue
                        else:
                            # Bare filename like "Book_4_Unit_5.md" -> use stem as unit name
                            unit_folder = Path(item['name']).stem

                        if unit_folder in seen_units:
                            continue
                        norm_key = re.sub(r'[^a-z0-9]', '', unit_folder).lower()
                        if norm_key in existing_names:
                            continue
                        seen_units.add(unit_folder)
                        existing_names.add(norm_key)

                        with open(item['path'], "r", encoding="utf-8") as f:
                            context_parts.append(
                                f"--- SOURCE FILE ({unit_folder}): {item['name']} ---\n{f.read()}"
                            )

            full_context = "\n\n".join(context_parts)
            answer_prompt = f"""Answer based on context:\n{full_context}\n\nQUERY: {query}"""
            return llm.chat([{"role": "user", "content": answer_prompt}], json_format=False)

        except Exception as e:
            return f"Wiki Query Error: {e}"

    def _save_extraction_results(self, data, source_filename, category_override=None):
        """Categorizes and saves extraction results."""
        path_obj = Path(source_filename)
        if path_obj.stem.lower() in ["subtitle", "transcript"] and path_obj.parent.name:
            filename_stem = path_obj.parent.name
        else:
            filename_stem = path_obj.stem
        saved_paths = []
        
        # Use explicit override if provided (most robust)
        category = category_override
        
        # Fallback to dynamic category detection for Virtual Schemas
        if not category:
            category = getattr(data, "_category", None)
        
        # Fallback to class-based detection for legacy/internal schemas
        if not category:
            if isinstance(data, VocabularyExtraction): category = "vocabulary"
            elif isinstance(data, GrammarExtraction): category = "grammar"
            elif isinstance(data, SummaryExtraction): category = "summary"
            elif isinstance(data, MindMapExtraction): category = "mindmap"

        unit_dir = self.config.wiki_content_path / filename_stem
        extractions_dir = unit_dir / "extractions"
        extractions_dir.mkdir(parents=True, exist_ok=True)

        # -------------------------------------------------------------
        # Transparent Delivery: Extractions always ship to extractions/
        # QA status & review flag are recorded in frontmatter for teacher visibility
        # -------------------------------------------------------------
        qa_audit = getattr(data, "_qa_audit", None) if dataclasses.is_dataclass(data) else (data.get("_qa_audit") if isinstance(data, dict) else None)
        has_fatal_flags = False
        if qa_audit and isinstance(qa_audit, dict):
            composite = qa_audit.get("composite_score")
            flags = qa_audit.get("flags", [])
            from .evaluator import FATAL_QA_FLAGS
            has_fatal_flags = any(
                any(fatal in f for fatal in FATAL_QA_FLAGS)
                for f in flags
            )
            if (composite is not None and composite < 80.0) or has_fatal_flags:
                logger.warning(
                    f"⚠️ {category} extraction for {filename_stem} scored {composite}/100 (review recommended). "
                    f"Delivering directly to extractions/ with qa_status: 'review_needed'."
                )

        if category == "vocabulary":
            # Deduplicate vocabulary list based on 'word' field (case-insensitive & stripped)
            seen = set()
            deduped_vocabulary = []
            if dataclasses.is_dataclass(data):
                vocab_list = getattr(data, "vocabulary", [])
                for item in vocab_list:
                    word_val = getattr(item, "word", "")
                    if isinstance(word_val, str):
                        w_clean = word_val.strip().lower().replace("[[", "").replace("]]", "")
                        if w_clean and w_clean not in seen:
                            seen.add(w_clean)
                            deduped_vocabulary.append(item)
                data.vocabulary = deduped_vocabulary
            elif isinstance(data, dict):
                vocab_list = data.get("vocabulary", [])
                for item in vocab_list:
                    word_val = item.get("word", "") if isinstance(item, dict) else getattr(item, "word", "")
                    if isinstance(word_val, str):
                        w_clean = word_val.strip().lower().replace("[[", "").replace("]]", "")
                        if w_clean and w_clean not in seen:
                            seen.add(w_clean)
                            deduped_vocabulary.append(item)
                data["vocabulary"] = deduped_vocabulary

            path = extractions_dir / f"{filename_stem}_vocabulary.md"
            content = self._format_as_markdown(data, "vocabulary", source_filename)
            with open(path, "w", encoding="utf-8") as f: f.write(content)
            saved_paths.append(str(path))
        
        elif category == "grammar":
            path = extractions_dir / f"{filename_stem}_grammar.md"
            content = self._format_as_markdown(data, "grammar", source_filename)
            with open(path, "w", encoding="utf-8") as f: f.write(content)
            saved_paths.append(str(path))

        elif category == "summary":
            path = extractions_dir / f"{filename_stem}_summary.md"
            content = self._format_as_markdown(data, "summary", source_filename)
            with open(path, "w", encoding="utf-8") as f: f.write(content)
            saved_paths.append(str(path))

        elif category == "mindmap":
            # 1. Save raw JSON file
            json_path = extractions_dir / f"{filename_stem}_mindmap.json"
            if dataclasses.is_dataclass(data):
                mindmap_dict = dataclasses.asdict(data)
            else:
                mindmap_dict = data
            nodes = mindmap_dict.get("nodes", []) if isinstance(mindmap_dict, dict) else []
            mindmap_dict["item_count"] = len(nodes)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(mindmap_dict, f, indent=2, ensure_ascii=False)
            saved_paths.append(str(json_path))

            # 2. Render and save standalone HTML to extractions directory
            html_content = self._render_mindmap(mindmap_dict)
            html_path = extractions_dir / f"{filename_stem}_mindmap.html"
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            saved_paths.append(str(html_path))

        return saved_paths

    def _format_as_markdown(self, data, category, source_filename=None):
        """
        Converts an extraction object to a markdown string dynamically.
        Iterates over dataclass fields or dict keys to remain field-agnostic.
        """
        source_link = f"[[{source_filename}]]" if source_filename else "None"
        display_title = Path(source_filename).stem.replace("_", " ") if source_filename else "Unit"
        
        # 1. Handle Collection Extractions (Vocabulary, Grammar)
        if category in ["vocabulary", "grammar"]:
            # Identify top-level fields (like overall_cefr_level)
            if dataclasses.is_dataclass(data):
                fields = [(f.name, getattr(data, f.name)) for f in dataclasses.fields(data)]
            else:
                fields = list(data.items())

            # Identify the collection list & count
            items = []
            if dataclasses.is_dataclass(data):
                list_field = next((f for f in dataclasses.fields(data) if isinstance(getattr(data, f.name), list)), None)
                if list_field:
                    items = getattr(data, list_field.name)
            else:
                list_field_name = next((k for k, v in fields if isinstance(v, list)), None)
                if list_field_name: items = data.get(list_field_name, [])

            # Automatic Deduplication: filter out repeated entries with identical primary key (e.g. word / headword)
            seen_keys = set()
            unique_items = []
            for item in items:
                k_val = None
                if dataclasses.is_dataclass(item):
                    for attr in ["word", "name", "concept_name", "quote"]:
                        if hasattr(item, attr) and getattr(item, attr):
                            k_val = str(getattr(item, attr)).strip().lower()
                            break
                elif isinstance(item, dict):
                    for attr in ["word", "name", "concept_name", "quote"]:
                        if attr in item and item[attr]:
                            k_val = str(item[attr]).strip().lower()
                            break
                
                if k_val:
                    if k_val in seen_keys:
                        continue
                    seen_keys.add(k_val)
                unique_items.append(item)
            
            items = unique_items
            item_count = len(items)

            lines = [
                "---",
                f"title: \"{display_title}\"",
                f"source: \"{source_link}\"",
                f"category: [\"{category}\", \"extraction\"]",
                f"item_count: {item_count}"
            ]

            # Inject QA Audit Score if available
            qa_audit = getattr(data, "_qa_audit", None) if dataclasses.is_dataclass(data) else (data.get("_qa_audit") if isinstance(data, dict) else None)
            if qa_audit and isinstance(qa_audit, dict):
                qa_composite = qa_audit.get("composite_score")
                if qa_composite is not None:
                    lines.append(f"qa_score: {round(float(qa_composite))}")
                    from .evaluator import FATAL_QA_FLAGS
                    flags = qa_audit.get("flags", [])
                    has_fatal = any(any(fatal in f for fatal in FATAL_QA_FLAGS) for f in flags)
                    qa_status = "passed" if (qa_composite >= 80.0 and not has_fatal) else "review_needed"
                    lines.append(f"qa_status: \"{qa_status}\"")

            for name, val in fields:
                if name not in ["title", "grammar_patterns", "vocabulary", "concepts", "_category", "_qa_audit"]:
                    if val: lines.append(f"{name}: \"{val}\"")
            
            lines.extend(["---", "", f"# {category.title()}: {display_title}", ""])

            for item in items:
                if dataclasses.is_dataclass(item):
                    item_fields = [(f.name, getattr(item, f.name)) for f in dataclasses.fields(item) if f.name != "design_audit"]
                else:
                    item_fields = [(k, v) for k, v in item.items() if k != "design_audit"]

                if not item_fields:
                    continue

                # Locate canonical primary header field
                primary_key = None
                for hk in ["word", "category", "name", "concept_name", "title"]:
                    if any(k == hk for k, v in item_fields):
                        primary_key = hk
                        break

                if primary_key:
                    header_entry = next((k, v) for k, v in item_fields if k == primary_key)
                    body_entries = [entry for entry in item_fields if entry[0] != primary_key]
                else:
                    header_entry = item_fields[0]
                    body_entries = item_fields[1:]

                header_name, header_val = header_entry
                if not (str(header_val).startswith("[[") and str(header_val).endswith("]]")):
                    header_val = f"[[{header_val}]]"
                
                lines.append(f"## {header_val}")

                # Iterate remaining fields as bullet points with canonical field ordering
                CANONICAL_FIELD_ORDERS = {
                    "grammar": [
                        "quote",
                        "pattern_formula",
                        "pedagogical_function",
                        "imitation_example",
                        "common_mistakes",
                        "cefr_level"
                    ],
                    "vocabulary": [
                        "part_of_speech",
                        "word_cefr_level",
                        "definition",
                        "quoted_sentence",
                        "example_usage"
                    ],
                    "expressions": [
                        "part_of_speech",
                        "word_cefr_level",
                        "definition",
                        "quoted_sentence",
                        "example_usage"
                    ]
                }

                preferred_order = CANONICAL_FIELD_ORDERS.get(category, [])
                def _field_sort_key(entry):
                    fname = entry[0].lower()
                    if fname in preferred_order:
                        return (0, preferred_order.index(fname))
                    return (1, fname)

                sorted_body_entries = sorted(body_entries, key=_field_sort_key)

                for fname, fval in sorted_body_entries:
                    label = fname.replace("_", " ").title()
                    if fval:
                        if category == "grammar" and fname == "pattern_formula":
                            fval = self.normalize_grammar_formula(str(fval))
                        lines.append(f"- **{label}**: {fval}")
                lines.append("")

            return "\n".join(lines)

        # 2. Handle Summary Extractions (with nested Concepts)
        elif category == "summary":
            if dataclasses.is_dataclass(data):
                title = getattr(data, "title", display_title)
                summary_text = getattr(data, "text_summary_or_plot", "")
                reading_time = getattr(data, "estimated_reading_time", "")
                questions = getattr(data, "essential_questions", [])
                lesson_hook = getattr(data, "lesson_hook", "")
                concepts = getattr(data, "concepts", [])
                overall_cefr = getattr(data, "overall_cefr_level", "B2")
            else:
                title = data.get("title", display_title)
                summary_text = data.get("text_summary_or_plot", "")
                reading_time = data.get("estimated_reading_time", "")
                questions = data.get("essential_questions", [])
                lesson_hook = data.get("lesson_hook", "")
                concepts = data.get("concepts", [])
                overall_cefr = data.get("overall_cefr_level", "B2")

            # Deterministic word count & reading time calculation (approx. 180-200 WPM)
            if hasattr(self, "_last_source_content") and self._last_source_content:
                words = len(re.findall(r'\b\w+\b', self._last_source_content))
                minutes = max(1, round(words / 180))
                reading_time = f"{words} words, approx. {minutes} min{'s' if minutes > 1 else ''} reading time"

            item_count = len(concepts)

            lines = [
                "---",
                f"title: \"{title}\"",
                f"source: \"{source_link}\"",
                "category: [\"summary\", \"extraction\"]",
                f"overall_cefr_level: \"{overall_cefr}\"",
                f"item_count: {item_count}",
                f"estimated_reading_time: \"{reading_time}\"",
            ]

            # Inject QA Audit Score if available
            qa_audit = getattr(data, "_qa_audit", None) if dataclasses.is_dataclass(data) else (data.get("_qa_audit") if isinstance(data, dict) else None)
            if qa_audit and isinstance(qa_audit, dict):
                qa_composite = qa_audit.get("composite_score")
                if qa_composite is not None:
                    lines.append(f"qa_score: {round(float(qa_composite))}")
                    from .evaluator import FATAL_QA_FLAGS
                    flags = qa_audit.get("flags", [])
                    has_fatal = any(any(fatal in f for fatal in FATAL_QA_FLAGS) for f in flags)
                    qa_status = "passed" if (qa_composite >= 80.0 and not has_fatal) else "review_needed"
                    lines.append(f"qa_status: \"{qa_status}\"")

            lines.extend([
                "---",
                "",
                f"# Summary: {title}",
                "",
                "## Narrative Overview",
                summary_text,
                "",
                "## Lesson Hook",
                lesson_hook,
                "",
                "## Essential Questions",
            ])
            for q in questions:
                lines.append(f"- {q}")
            lines.append("")
            
            lines.append("## Core Concepts")
            lines.append("")
            for concept in concepts:
                if dataclasses.is_dataclass(concept):
                    c_name = getattr(concept, "concept_name", "")
                    c_sig = getattr(concept, "educational_significance", "")
                    c_details = getattr(concept, "key_details", [])
                    c_conn = getattr(concept, "related_connections", [])
                else:
                    c_name = concept.get("concept_name", "")
                    c_sig = concept.get("educational_significance", "")
                    c_details = concept.get("key_details", [])
                    c_conn = concept.get("related_connections", [])

                # Add double-brackets around concept name for wikilinks
                c_header = c_name
                if not (c_header.startswith("[[") and c_header.endswith("]]")):
                    c_header = f"[[{c_header}]]"

                lines.append(f"### {c_header}")
                if c_sig:
                    lines.append(f"- **Educational Significance**: {c_sig}")
                if c_details:
                    lines.append("- **Key Details**:")
                    for detail in c_details:
                        lines.append(f"  - {detail}")
                if c_conn:
                    conn_links = []
                    for conn in c_conn:
                        clean_conn = str(conn or "").strip()
                        if "Connect to" in clean_conn or ":" in clean_conn:
                            m = re.search(r'(?:Connect to\s*)?\*?\*?([A-Za-z0-9\s/&#\-]+?)\*?\*?(?:\s*:|\s*$)', clean_conn, re.IGNORECASE)
                            if m and len(m.group(1).strip()) > 1:
                                clean_conn = m.group(1).strip()
                            elif ":" in clean_conn:
                                clean_conn = re.sub(r'^(?:Connect to\s*)?\*?\*?|\*?\*?$', '', clean_conn.split(":")[0]).strip()
                        if clean_conn:
                            if not (clean_conn.startswith("[[") and clean_conn.endswith("]]")):
                                conn_links.append(f"[[{clean_conn}]]")
                            else:
                                conn_links.append(clean_conn)
                    if conn_links:
                        lines.append(f"- **Related Connections**: {', '.join(conn_links)}")
                lines.append("")
                lines.append("&nbsp;")
                lines.append("")

            return "\n".join(lines)

        return str(data)

    def _load_wiki_data(self, core_name, quiz_type):
        """Loads data from the wiki for quiz generation."""
        parts = Path(core_name).parts
        if "wiki" in parts:
            idx = parts.index("wiki")
            if idx + 1 < len(parts):
                core_name = parts[idx + 1]
        elif "raw" in parts:
            idx = parts.index("raw")
            if idx + 1 < len(parts):
                core_name = parts[idx + 1]
        else:
            core_name = parts[0] if parts else core_name
            
        if core_name.endswith(".md"): core_name = core_name[:-3]
        elif core_name.endswith(".txt"): core_name = core_name[:-4]

        # 1. Resolve unit directory case-insensitively using normalize_name
        from .config import normalize_name
        norm_core = normalize_name(core_name)
        unit_dir = self.config.wiki_content_path / core_name
        if not unit_dir.exists() and self.config.wiki_content_path.exists():
            for child in self.config.wiki_content_path.iterdir():
                if child.is_dir() and normalize_name(child.name) == norm_core:
                    unit_dir = child
                    core_name = child.name
                    break

        # Check standard location: wiki/<core_name>/extractions/<core_name>_vocabulary.md
        vocab_path = unit_dir / "extractions" / f"{core_name}_vocabulary.md"
        if not vocab_path.exists():
            vocab_paths = list(self.config.wiki_content_path.rglob(f"{core_name}_vocabulary.md"))
            vocab_path = vocab_paths[0] if vocab_paths else (unit_dir / f"{core_name}_vocabulary.md")

        cefr = None
        vocab_content = ""
        
        if vocab_path.exists():
            with open(vocab_path, "r", encoding="utf-8") as f:
                vocab_content = f.read()
                # Match both quoted and unquoted frontmatter: overall_cefr_level: "B2" or overall_cefr_level: B2
                m = re.search(r'overall_cefr_level:\s*["\']?([A-C][1-2])["\']?', vocab_content, re.IGNORECASE)
                if m:
                    cefr = m.group(1).upper()

        # Multi-source fallback: Check summary, grammar, or mindmap if vocabulary doesn't have it or isn't generated yet
        if not cefr:
            for ext_candidate in [f"{core_name}_summary.md", f"{core_name}_grammar.md", f"{core_name}_mindmap.json"]:
                candidate_path = unit_dir / "extractions" / ext_candidate
                if candidate_path.exists():
                    try:
                        cand_txt = candidate_path.read_text(encoding="utf-8")
                        m = re.search(r'overall_cefr_level["\']?\s*[:=]\s*["\']?([A-C][1-2])["\']?', cand_txt, re.IGNORECASE)
                        if m:
                            cefr = m.group(1).upper()
                            break
                    except Exception:
                        pass

        if not cefr:
            import logging
            logging.getLogger("librarian").warning(f"Could not resolve overall_cefr_level for {core_name}; falling back to B2.")
            cefr = "B2"

        if quiz_type == "vocabulary":
            if not vocab_path.exists(): return None
            return {"content": vocab_content, "cefr_level": cefr}

        elif quiz_type == "reading":
            # Check standard location: wiki/<core_name>/sources/<core_name>.md or .txt
            source_path = self.config.wiki_content_path / core_name / "sources" / f"{core_name}.md"
            if not source_path.exists():
                source_path = self.config.wiki_content_path / core_name / "sources" / f"{core_name}.txt"
            if not source_path.exists():
                sources_dir = self.config.wiki_content_path / core_name / "sources"
                if sources_dir.exists() and sources_dir.is_dir():
                    candidates = [f for f in sources_dir.iterdir() if f.is_file() and f.suffix in [".md", ".txt"]]
                    if candidates:
                        source_path = candidates[0]
            if not source_path or not source_path.exists(): return None
            with open(source_path, "r", encoding="utf-8") as f: passage = f.read()
            # Clean syllabus sections (e.g. ## Syllabus Vocabulary, ## Syllabus Grammar) so only pure passage text is supplied
            clean_passage, _, _, _ = self.parse_syllabus_sections(passage)
            return {"passage": clean_passage.strip() if clean_passage else passage, "cefr_level": cefr}

        elif quiz_type == "translation":
            if not vocab_path.exists(): return None
            grammar_path = self.config.wiki_content_path / core_name / "extractions" / f"{core_name}_grammar.md"
            if not grammar_path.exists():
                grammar_paths = list(self.config.wiki_content_path.rglob(f"{core_name}_grammar.md"))
                grammar_path = grammar_paths[0] if grammar_paths else (self.config.wiki_content_path / core_name / f"{core_name}_grammar.md")
            grammar = ""
            if grammar_path.exists():
                with open(grammar_path, "r", encoding="utf-8") as f: grammar = f.read()
            return {"vocab_list": vocab_content, "grammar_list": grammar, "cefr_level": cefr}

        elif quiz_type == "listening":
            if not vocab_path.exists(): return None
            # Retrieve unit summary to anchor dialogue topic to Core Concepts
            summary_content = ""
            summary_path = unit_dir / "extractions" / f"{core_name}_summary.md"
            if not summary_path.exists():
                summary_candidates = list(self.config.wiki_content_path.rglob(f"{core_name}_summary.md"))
                if summary_candidates:
                    summary_path = summary_candidates[0]
            if summary_path and summary_path.exists():
                try:
                    summary_content = summary_path.read_text(encoding="utf-8")
                except Exception:
                    pass
            # Fallback to source article if summary does not exist
            source_content = ""
            if not summary_content:
                source_path = unit_dir / "sources" / f"{core_name}.md"
                if not source_path.exists():
                    source_path = unit_dir / "sources" / f"{core_name}.txt"
                if source_path.exists():
                    try:
                        source_content = source_path.read_text(encoding="utf-8")
                    except Exception:
                        pass
            return {
                "vocab_list": vocab_content,
                "summary_content": summary_content,
                "source_content": source_content,
                "unit_core_name": core_name,
                "cefr_level": cefr
            }

        elif quiz_type in ["video", "listening"]:
            # For listening quiz, we check if there's an active media file first
            from .config import normalize_name
            normalized_core = normalize_name(core_name)
            unit_wiki_dir = self.config.wiki_content_path / core_name
            
            # Resolve actual unit_wiki_dir case-insensitively
            if self.config.wiki_content_path.exists():
                for child in self.config.wiki_content_path.iterdir():
                    if child.is_dir() and normalize_name(child.name) == normalized_core:
                        unit_wiki_dir = child
                        core_name = child.name  # Sync core_name
                        break

            source_path = None
            active_media_name = None
            if unit_wiki_dir.exists() and unit_wiki_dir.is_dir():
                # Dynamically resolve active media by scanning sources/media (most recently modified first)
                media_dir = unit_wiki_dir / "sources" / "media"
                if media_dir.exists() and media_dir.is_dir():
                    candidates = [f for f in media_dir.iterdir() if f.is_file() and f.suffix in [".md", ".txt"]]
                    if candidates:
                        candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                        source_path = candidates[0]
                        active_media_name = source_path.name

            # Fallback if no active_media or active_media not found (only for video, or if listening needs it)
            if not source_path and quiz_type == "video":
                if unit_wiki_dir.exists() and unit_wiki_dir.is_dir():
                    # Check sources/media/ first for any md/txt transcript files
                    media_dir = unit_wiki_dir / "sources" / "media"
                    if media_dir.exists() and media_dir.is_dir():
                        for f in media_dir.iterdir():
                            if f.is_file() and f.suffix in [".md", ".txt"]:
                                source_path = f
                                active_media_name = f.name
                                break
                    
                    if not source_path:
                        # Look for explicit video_transcript files first
                        for f in unit_wiki_dir.rglob("*.md"):
                            if f.is_file():
                                try:
                                    with open(f, "r", encoding="utf-8") as f_read:
                                        test_content = f_read.read()
                                    if 'category: "video_transcript"' in test_content or 'video_type:' in test_content or 'video_url:' in test_content or 'source_url:' in test_content:
                                        source_path = f
                                        break
                                except Exception:
                                    pass

                # If no explicit video transcript found, try standard source paths
                if not source_path or not source_path.exists():
                    source_path = self.config.wiki_content_path / core_name / "sources" / f"{core_name}.md"
                    if not source_path.exists():
                        source_path = self.config.wiki_content_path / core_name / "sources" / f"{core_name}.txt"
                    
                    # Fallback to any md file in the unit directory (excluding compiled node outputs)
                    if not source_path.exists() and unit_wiki_dir.exists() and unit_wiki_dir.is_dir():
                        for f in unit_wiki_dir.glob("*.md"):
                            if f.is_file() and not any(suffix in f.name for suffix in ["_vocabulary", "_grammar", "_summary"]):
                                source_path = f
                                break
                        if not source_path.exists():
                            for f in unit_wiki_dir.rglob("*.md"):
                                if f.is_file() and not any(suffix in f.name for suffix in ["_vocabulary", "_grammar", "_summary"]):
                                    source_path = f
                                    break

            if quiz_type == "listening" and not source_path:
                # Normal listening quiz defaults to using vocab list (not a specific transcript file)
                if not vocab_path.exists(): return None
                return {"vocab_list": vocab_content, "cefr_level": cefr}

            if not source_path or not source_path.exists(): return None
            with open(source_path, "r", encoding="utf-8") as f: transcript = f.read()
            # Try multiple common keys for video URL (quoted or unquoted)
            for key in ["source_url", "video_url", "url", "source"]:
                m_url = re.search(fr'{key}:\s*["\']?([^\n"\']+)["\']?', transcript)
                if m_url and m_url.group(1).strip():
                    video_url = m_url.group(1).strip()
                    break
                    
            m_type = re.search(r'video_type:\s*["\']?([^\n"\']+)["\']?', transcript)
            if m_type:
                video_type = m_type.group(1).strip()
            elif video_url:
                # Auto-detect type based on URL
                url_lower = video_url.lower()
                if "youtube" in url_lower or "youtu.be" in url_lower:
                    video_type = "youtube"
                elif url_lower.endswith((".mp4", ".webm", ".ogg", ".mp3", ".wav")) or "/media/" in url_lower:
                    video_type = "local"
                else:
                    video_type = "youtube"  # default fallback if URL exists

            # If video_url is still empty and file is inside the media folder, resolve local companion or fallback
            if not video_url and ("media" in source_path.parts or (active_media_name and source_path.name == active_media_name)):
                parent_dir = source_path.parent
                found_companion = False
                for ext in [".mp4", ".mp3", ".webm", ".ogg", ".wav"]:
                    companion = parent_dir / f"{source_path.stem}{ext}"
                    if companion.exists():
                        video_url = f"/wiki/{core_name}/sources/media/{companion.name}"
                        video_type = "local"
                        found_companion = True
                        break
                if not found_companion:
                    # Fallback default local video url pointing to standard stem
                    video_url = f"/wiki/{core_name}/sources/media/{source_path.stem}.mp4"
                    video_type = "local"
            
            # Rewrite local video URL to point to /wiki/<UnitName>/sources/media/
            if video_type == "local" and video_url:
                import os
                video_filename = os.path.basename(video_url)
                video_url = f"/wiki/{core_name}/sources/media/{video_filename}"
                
            return {"transcript": transcript, "video_url": video_url, "video_type": video_type, "cefr_level": cefr, "source_stem": source_path.stem}

        return None

    def _render_handout(self, quiz_obj, template_name, audio_url=None, language=None):
        """Renders HTML templates with injected data."""
        template_path = Path(__file__).parent / "templates" / f"{template_name}.html"
        
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

    @staticmethod
    def _build_vocab_prose_draft_prompt(v_prompt: str, count: int) -> str:
        """Turn 1 prompt for the vocabulary Prose-to-JSON pipeline.

        Appends a strict FORMAT MANDATE draft layout instruction to the vocabulary prompt.
        Follows the strict FORMAT MANDATE architecture established in quiz and grammar prose generation,
        banning pre-analysis, stream-of-consciousness monologues, and repetitive drafts to prevent
        smaller models from over-generating tens of thousands of tokens.
        """
        count_str = str(count) if count is not None else "the target"
        return (
            f"{v_prompt}\n\n"
            "### GENERATION FORMAT MANDATE (VOCABULARY EXTRACTION DRAFT):\n"
            f"Output all authentic academic vocabulary words directly and consecutively using this clean, structured text format (up to {count_str} items).\n"
            "🚫 DO NOT include pre-analysis, stream-of-consciousness deliberations, self-correction monologues, or repetitive drafts. "
            "Begin immediately with 'Item 1:' and write out the items cleanly:\n\n"
            "Item 1:\n"
            "- Audit: AUDIT: [Surface Word in Text] -> [Base Lemma Headword] -> [PoS] -> [CEFR] -> [VERBATIM_CONFIRMED]\n"
            "- Context Sentence: \"[exact 100% verbatim sentence copied directly from the passage without ANY alteration or rewriting]\"\n"
            "- Word: [single-word base lemma headword derived from the audit above, strictly ONE word]\n"
            "- Part of Speech: [noun / verb / adjective / adverb / preposition / conjunction / interjection]\n"
            "- Definition: [concise, context-specific English meaning]\n"
            "- Example Usage: [original, high-quality academic illustrative sentence in a different context]\n"
            "- CEFR: [B1 / B2 / C1 / C2]\n\n"
            "Item 2:\n"
            "...\n\n"
            "Ensure all items follow this exact item layout consecutively without extra commentary, markdown tables, or code fences."
        )

    @staticmethod
    def _parse_vocab_prose(text, source_text: str = None) -> list:
        """Deterministically parse the Turn-1 prose draft into a list of vocabulary item dicts.

        The draft is a sequence of blocks, each with lines like:
            - Word: X
            - Context Sentence: "Y"
            - Part of Speech: Z
            - Definition: D
            - Example Usage: E
            - CEFR: L
            - Audit: A
        """
        import re
        if not text or not isinstance(text, str):
            return []
        # Split blocks by 'Item \d+' or 'Audit:' or 'Word:'
        item_split = re.compile(r"(?im)^[ \t]*(?:Item\s+\d+[:：]?|[-*]?[ \t]*(?:Design\s+Audit|Audit|Word)[ \t]*[:：])")
        boundaries = [m.start() for m in item_split.finditer(text)]
        if not boundaries:
            return []

        patterns = {
            "word": re.compile(r"(?im)^[ \t]*[-*]?[ \t]*Word[ \t]*[:：][ \t]*(.*)$"),
            "quoted_sentence": re.compile(r"(?im)^[ \t]*[-*]?[ \t]*(?:Context[ \t]+Sentence|Quote)[ \t]*[:：][ \t]*(.*)$"),
            "part_of_speech": re.compile(r"(?im)^[ \t]*[-*]?[ \t]*Part[ \t]+of[ \t]+Speech[ \t]*[:：][ \t]*(.*)$"),
            "definition": re.compile(r"(?im)^[ \t]*[-*]?[ \t]*Definition[ \t]*[:：][ \t]*(.*)$"),
            "example_usage": re.compile(r"(?im)^[ \t]*[-*]?[ \t]*Example[ \t]+Usage[ \t]*[:：][ \t]*(.*)$"),
            "word_cefr_level": re.compile(r"(?im)^[ \t]*[-*]?[ \t]*CEFR[ \t]*[:：][ \t]*(.*)$"),
            "design_audit": re.compile(r"(?im)^[ \t]*[-*]?[ \t]*(?:Design[ \t]+Audit|Audit)[ \t]*[:：][ \t]*(.*)$"),
        }
        valid_cefr = ("B1", "B2", "C1", "C2")
        valid_pos = ("noun", "verb", "adjective", "adverb", "preposition", "conjunction", "interjection")

        def grab(block, key):
            m = patterns[key].search(block)
            return m.group(1).strip() if m else ""

        normalized_source = ""
        if source_text:
            normalized_source = re.sub(r'\s+', ' ', source_text).replace('“', '"').replace('”', '"').replace("‘", "'").replace("’", "'").lower()

        items = []
        for i, start in enumerate(boundaries):
            end = boundaries[i + 1] if i + 1 < len(boundaries) else len(text)
            block = text[start:end]

            word = grab(block, "word").strip().strip("[]").strip()
            for pos in valid_pos:
                if word.lower().endswith(" " + pos):
                    word = word[: -(len(pos) + 1)].strip()
                    break

            quote = grab(block, "quoted_sentence")
            if len(quote) >= 2 and quote[0] in "\"'" and quote[-1] in "\"'":
                quote = quote[1:-1].strip()

            cefr = grab(block, "word_cefr_level").upper().strip()
            if cefr not in valid_cefr:
                cefr = "B2"
            pos = grab(block, "part_of_speech").lower().strip()
            if pos not in valid_pos:
                pos = "noun"

            if not word:
                continue
            lowered = word.lower()
            if any(p in lowered for p in (
                "headword", "lemma", "part of speech", "part-of-speech",
                "context sentence", "example usage", "placeholder",
                "surface word", "audit: ",
            )):
                continue

            # Deterministic source verification if source_text is provided
            if normalized_source and quote:
                cleaned_q = re.sub(r'\s+', ' ', quote).replace('“', '"').replace('”', '"').replace("‘", "'").replace("’", "'").strip('."\' ').lower()
                if cleaned_q and cleaned_q not in normalized_source:
                    continue

            items.append({
                "word": word,
                "quoted_sentence": quote,
                "part_of_speech": pos,
                "definition": grab(block, "definition"),
                "example_usage": grab(block, "example_usage"),
                "word_cefr_level": cefr,
                "design_audit": grab(block, "design_audit"),
            })
        return items

    def _run_vocab_prose_pipeline(self, v_prompt, v_kwargs, v_schema, file_stem):
        """Two-Turn Prose-to-JSON pipeline for vocabulary extraction.

        Turn 1 (Prose Drafting): Unconstrained generation with full native thinking,
        producing clean, academic, single-word vocabulary items with authentic citations.
        Turn 2 (Deterministic Packaging): Temperature 0 structured serialization with
        full source text anchor, strict QA retry loop, and Deterministic Code Gate filtering.
        """
        count = v_kwargs.get("count", 20)
        source_content = v_kwargs.get("content", "")
        draft_prompt = self._build_vocab_prose_draft_prompt(v_prompt, count)

        logger.info("🧠 Turn 1: prose vocabulary selection (native reasoning, json_format=False)...")
        prose_draft = llm.chat(
            [{"role": "user", "content": draft_prompt}],
            json_format=False,
            task_name=f"extract_vocabulary_{file_stem}_turn1_prose",
        )

        if prose_draft:
            logger.info("📦 Turn 2: packaging Turn 1 vocabulary prose draft into structured JSON Schema...")
            source_content = v_kwargs.get("content", "")
            packaging_prompt = (
                "You are a deterministic lexicographical data converter.\n"
                "Faithfully serialize every vocabulary item from the Turn 1 draft into the required VocabularyExtraction JSON schema format.\n"
                "MANDATES:\n"
                "- ⚠️ FAITHFUL 1:1 PACKAGING MANDATE: Convert all vocabulary items from the draft directly and consecutively into 'vocabulary'. Do NOT add, hallucinate, rewrite, or drop any items.\n"
                "- Map 'Audit' to 'design_audit'.\n"
                "- Map 'Context Sentence' to 'quoted_sentence': 🛑 CRITICAL: STRIP ALL SURROUNDING QUOTATION MARKS (\"...\") and escaped slashes. The 'quoted_sentence' field MUST contain only the raw sentence text without enclosing quotes.\n"
                "- Map 'Word' to 'word' (strictly a single dictionary base lemma).\n"
                "- Map 'Part of Speech' to 'part_of_speech' (strictly one of: noun, verb, adjective, adverb, preposition, conjunction, interjection).\n"
                "- Map 'Definition' to 'definition'.\n"
                "- Map 'CEFR' to 'word_cefr_level' (B1, B2, C1, or C2).\n"
                "- Map 'Example Usage' to 'example_usage'.\n"
                "- Set 'title' to 'Vocabulary'.\n"
                "- Derive 'overall_cefr_level' from the most frequent CEFR level of the vocabulary.\n"
                "- Clean any meta-tokens, code fences, or extraneous remarks.\n\n"
                f"### VOCABULARY DRAFT ###\n{prose_draft}"
            )

            try:
                vocab_obj = llm.chat(
                    [{"role": "user", "content": packaging_prompt}],
                    schema=v_schema,
                    temperature=0.0,
                    task_name=f"extract_vocabulary_{file_stem}_turn2_package",
                )
                if vocab_obj:
                    # Clean surrounding quotes on items while preserving all items faithfully
                    raw_items = getattr(vocab_obj, "vocabulary", []) if hasattr(vocab_obj, "vocabulary") else vocab_obj.get("vocabulary", [])
                    for it in raw_items:
                        q = it.quoted_sentence if hasattr(it, "quoted_sentence") else it.get("quoted_sentence", "")
                        q_clean = q.strip().strip('"\'“”‘’').strip()
                        if hasattr(it, "quoted_sentence"):
                            it.quoted_sentence = q_clean
                        elif isinstance(it, dict):
                            it["quoted_sentence"] = q_clean

                    logger.info("✅ Successfully packaged Turn 1 draft into Vocabulary JSON Schema via Turn 2.")
                    return vocab_obj
            except Exception as pkg_err:
                logger.warning(f"⚠️ Turn 2 packaging encountered error: {pkg_err}. Engaging deterministic parser fallback...")

            # Fallback to deterministic code parsing if Turn 2 call fails
            items = self._parse_vocab_prose(prose_draft, source_text=source_content)
            if items:
                cefr_counts = {}
                for it in items:
                    cefr_counts[it["word_cefr_level"]] = cefr_counts.get(it["word_cefr_level"], 0) + 1
                overall = max(cefr_counts, key=cefr_counts.get) if cefr_counts else "B2"
                logger.info(f"✅ Parsed {len(items)} vocabulary items from prose draft via deterministic parser.")
                return validate_and_map(VocabularyExtraction, {
                    "title": "Vocabulary",
                    "overall_cefr_level": overall,
                    "vocabulary": items,
                })

        # Final Fallback: prose draft unparsable -> original one-shot JSON extraction.
        logger.warning("⚠️ Vocabulary prose draft was unparsable; falling back to one-shot JSON extraction.")
        data = llm.chat(
            [{"role": "user", "content": v_prompt}],
            schema=v_schema,
            task_name=f"extract_vocabulary_{file_stem}_json_fallback",
        )
        return data

    def _build_grammar_prose_draft_prompt(self, g_prompt: str, count: int) -> str:
        """Appends a clear plain-text draft layout instruction to the grammar prompt.

        Follows the strict FORMAT MANDATE architecture established in quiz prose generation,
        banning pre-analysis, monologues, and repetitive outlines to prevent smaller models
        from over-generating tens of thousands of stream-of-consciousness tokens.
        """
        count_str = str(count) if count is not None else "the target"
        return (
            f"{g_prompt}\n\n"
            "### GENERATION FORMAT MANDATE (GRAMMAR EXTRACTION DRAFT):\n"
            f"Output all authentic grammar patterns directly and consecutively using this clean, structured text format (at most {count_str} items).\n"
            "MANDATES:\n"
            "- Extract ONLY genuine structures present in the passage. If only 2 or 3 genuine patterns exist, output ONLY 2 or 3. NEVER force-fit or fabricate weak items!\n"
            "- Each item MUST be anchored to a distinct, unique verbatim sentence. NEVER reuse the same sentence for multiple items.\n"
            "- 🚫 DO NOT include pre-analysis, stream-of-consciousness deliberations, self-correction monologues, or repetitive drafts. "
            "Begin immediately with 'Item 1:' and write out the items cleanly:\n\n"
            "Item 1:\n"
            "- Quote: \"[exact 100% verbatim sentence copied directly from the text without ANY alteration or rewriting]\"\n"
            "- Pattern Formula: [COBUILD algebraic slot formula, e.g. plain text anchors + [NP]/[VP]/[adj]/[to-V]]\n"
            "- Pedagogical Function: [concise academic explanation of how this structure enhances formality or rhetorical nuance]\n"
            "- Design Audit: AUDIT: [Physical Anchor in quote] -> [Formula] -> [Syntactic Function] -> [Allocated Category]\n"
            "- Category: [strictly one of: Rhetoric & Emphasis, Cohesion & Framing, Information Packaging, Logic & Stance - aligned with the Audit & Pedagogical Function above]\n"
            "- Imitation Example: [high-quality academic model sentence illustrating this pattern in a different domain]\n"
            "- Common Mistakes: [typical ESL learner errors with this pattern]\n"
            "- CEFR: [B1 / B2 / C1 / C2]\n\n"
            "Item 2:\n"
            "...\n\n"
            f"Ensure all items follow this exact item layout consecutively without extra commentary, markdown tables, or code fences."
        )

    @classmethod
    def _parse_grammar_prose(cls, text: str, source_text: str = None) -> list:
        """Deterministically parse the Turn-1 grammar prose draft into a list of grammar item dicts.

        The draft is a sequence of blank-line-separated blocks, each with lines like:
            - Category: X
            - Quote: "Y"
            - Pattern Formula: F
            - Pedagogical Function: P
            - Imitation Example: E
            - Common Mistakes: M
            - CEFR: L
            - Audit: A
        Deterministic code parsing avoids model token stalls, token truncation, and hallucinated keys.
        If `source_text` is provided, quotes are verified against the source to reject hallucinated or rewritten quotes.
        """
        if not text or not isinstance(text, str):
            return []
        text = text.replace("\r\n", "\n")
        # Split blocks by 'Item \d+' or 'Design Audit:' or 'Quote:' or 'Category:'
        item_boundary = re.compile(r"(?im)^[ \t]*(?:Item[ \t]+\d+|[-*]?[ \t]*(?:Design\s+Audit|Audit|Quote|Category)[ \t]*[:：])")
        boundaries = [m.start() for m in item_boundary.finditer(text)]
        if not boundaries:
            # Fallback to category line
            category_line = re.compile(r"(?im)^[ \t]*[-*]?[ \t]*Category[ \t]*[:：][ \t]*(.*)$")
            boundaries = [m.start() for m in category_line.finditer(text)]
            if not boundaries:
                return []

        patterns = {
            "category": re.compile(r"(?im)^[ \t]*[-*]?[ \t]*Category[ \t]*[:：][ \t]*(.*)$"),
            "quote": re.compile(r"(?im)^[ \t]*[-*]?[ \t]*Quote[ \t]*[:：][ \t]*(.*)$"),
            "pattern_formula": re.compile(r"(?im)^[ \t]*[-*]?[ \t]*Pattern[ \t]+Formula[ \t]*[:：][ \t]*(.*)$"),
            "pedagogical_function": re.compile(r"(?im)^[ \t]*[-*]?[ \t]*Pedagogical[ \t]+Function[ \t]*[:：][ \t]*(.*)$"),
            "imitation_example": re.compile(r"(?im)^[ \t]*[-*]?[ \t]*Imitation[ \t]+Example[ \t]*[:：][ \t]*(.*)$"),
            "common_mistakes": re.compile(r"(?im)^[ \t]*[-*]?[ \t]*Common[ \t]+Mistakes[ \t]*[:：][ \t]*(.*)$"),
            "cefr_level": re.compile(r"(?im)^[ \t]*[-*]?[ \t]*CEFR[ \t]*[:：][ \t]*(.*)$"),
            "design_audit": re.compile(r"(?im)^[ \t]*[-*]?[ \t]*(?:Design\s+Audit|Audit)[ \t]*[:：][ \t]*(.*)$"),
        }
        valid_cefr = ("B1", "B2", "C1", "C2")

        from .schemas import GRAMMAR_CATEGORIES, normalize_enum_value
        allowed_cats = get_args(GRAMMAR_CATEGORIES) if 'get_args' in globals() else (
            "Concessive clauses", "Conditional clauses", "Participial clauses",
            "Inversion", "Cleft sentences", "Nominalization", "Abstract frames",
            "Rhetorical parallelism", "Non-finite structures", "Hedging devices",
            "Anaphoric and cataphoric nouns", "Evaluative It-frameworks"
        )

        # Pre-clean source text for fuzzy-normalised presence check (collapse whitespace and normalize quotes)
        normalized_source = None
        if source_text:
            normalized_source = re.sub(r'\s+', ' ', source_text).replace('“', '"').replace('”', '"').replace("‘", "'").replace("’", "'").lower()

        def grab(block, key):
            m = patterns[key].search(block)
            return m.group(1).strip() if m else ""

        items = []
        for i, start in enumerate(boundaries):
            end = boundaries[i + 1] if i + 1 < len(boundaries) else len(text)
            block = text[start:end]

            raw_cat = grab(block, "category").strip("[]\"'").strip()
            norm_cat = normalize_enum_value(GRAMMAR_CATEGORIES, raw_cat) if raw_cat else "Information Packaging"

            quote = grab(block, "quote")
            if len(quote) >= 2 and quote[0] in "\"'" and quote[-1] in "\"'":
                quote = quote[1:-1].strip()

            formula = grab(block, "pattern_formula")
            if formula:
                formula = cls.normalize_grammar_formula(formula)

            cefr = grab(block, "cefr_level").upper().strip()
            if cefr not in valid_cefr:
                cefr = "B2"

            if not quote or not formula:
                continue

            # Deterministic Verbatim Gate: If source text is available, verify quote authenticity
            if normalized_source:
                cleaned_quote = re.sub(r'\s+', ' ', quote).replace('“', '"').replace('”', '"').replace("‘", "'").replace("’", "'").strip('."\' ').lower()
                # Check if core quote exists in text (at least 20 chars substring or exact word tokens)
                if cleaned_quote and cleaned_quote not in normalized_source:
                    logger.warning(
                        f"🛡️ Deterministic Code Gate: Dropping fabricated/tampered quote '{quote[:50]}...' "
                        f"in category '{norm_cat}' (not found in source passage)."
                    )
                    continue

            items.append({
                "category": norm_cat,
                "pattern_formula": formula,
                "quote": quote,
                "pedagogical_function": grab(block, "pedagogical_function"),
                "imitation_example": grab(block, "imitation_example"),
                "common_mistakes": grab(block, "common_mistakes"),
                "cefr_level": cefr,
                "design_audit": grab(block, "design_audit"),
            })
        return items

    def _run_grammar_prose_pipeline(self, g_prompt, g_kwargs, g_schema, file_stem):
        """Prose-to-JSON pipeline for grammar extraction.

        Turn 1 (json_format=False) gives the model unconstrained reasoning to analyze
        sentence syntax, verify physical markers, and formulate COBUILD rules without JSON syntax overhead.
        Turn 2 (packaging prompt with schema and temperature=0.0) deterministically packages
        the Turn 1 prose draft into the strict GrammarExtraction JSON schema.
        A deterministic code parser acts as a robust zero-failure fallback if Turn 2 is unavailable.
        """
        from .schemas import GrammarExtraction
        count = g_kwargs.get("count", 5)
        draft_prompt = self._build_grammar_prose_draft_prompt(g_prompt, count)
        logger.info("🧠 Turn 1: prose grammar extraction (unconstrained syntactic reasoning, json_format=False)...")
        prose_draft = llm.chat(
            [{"role": "user", "content": draft_prompt}],
            json_format=False,
            task_name=f"extract_grammar_{file_stem}_turn1_prose",
        )

        if prose_draft:
            logger.info("📦 Turn 2: packaging Turn 1 grammar prose draft into structured JSON Schema...")
            source_content = g_kwargs.get("content", "")
            packaging_prompt = (
                "You are a deterministic grammatical data converter.\n"
                "Faithfully serialize every grammar pattern from the Turn 1 draft into the required GrammarExtraction JSON schema format.\n"
                "MANDATES:\n"
                "- ⚠️ FAITHFUL 1:1 PACKAGING MANDATE: Convert all grammar patterns presented in the draft directly and consecutively into 'grammar_patterns'. Do NOT add, hallucinate, rewrite, or drop any items.\n"
                "- Map 'Design Audit' (or 'Audit') to 'design_audit'.\n"
                "- Map 'Quote' to 'quote': 🛑 CRITICAL: STRIP ALL SURROUNDING QUOTATION MARKS (\"...\") and escaped slashes. The 'quote' field MUST contain only the raw sentence text without enclosing quotes.\n"
                "- Map 'Category' to 'category': 🛑 MUST be strictly one of these exact 4 strings (NO extra words, NO parentheses, NO custom variants): 'Rhetoric & Emphasis', 'Cohesion & Framing', 'Information Packaging', 'Logic & Stance'.\n"
                "- Map 'Pattern Formula' to 'pattern_formula'.\n"
                "- Map 'Pedagogical Function' to 'pedagogical_function'.\n"
                "- Map 'Imitation Example' to 'imitation_example'.\n"
                "- Map 'Common Mistakes' to 'common_mistakes'.\n"
                "- Map 'CEFR' to 'cefr_level' (B1, B2, C1, or C2).\n"
                "- Set 'title' to 'Grammar'.\n"
                "- Derive 'overall_cefr_level' from the most frequent CEFR level of the patterns.\n"
                "- Clean any meta-tokens, code fences, or extraneous remarks.\n"
                "- 🛑 VERBATIM FIDELITY CHECK: every 'quote' MUST be copied word-for-word from the SOURCE TEXT at the end of this prompt (no rewording, no reordering, no splicing across sentences, no formula tokens).\n\n"
                f"### GRAMMAR PATTERNS DRAFT ###\n{prose_draft}"
            )
            if source_content:
                packaging_prompt += f"\n\n### SOURCE TEXT ###\n{source_content}"

            try:
                grammar_obj = llm.chat(
                    [{"role": "user", "content": packaging_prompt}],
                    schema=g_schema,
                    temperature=0.0,
                    task_name=f"extract_grammar_{file_stem}_turn2_package",
                )
                if grammar_obj:
                    # Clean surrounding quotes on items while preserving all items faithfully
                    raw_patterns = getattr(grammar_obj, "grammar_patterns", []) if hasattr(grammar_obj, "grammar_patterns") else grammar_obj.get("grammar_patterns", [])
                    for p in raw_patterns:
                        q = p.quote if hasattr(p, "quote") else p.get("quote", "")
                        q_clean = q.strip().strip('"\'“”‘’').strip()
                        if hasattr(p, "quote"):
                            p.quote = q_clean
                        elif isinstance(p, dict):
                            p["quote"] = q_clean

                    logger.info("✅ Successfully packaged Turn 1 draft into Grammar JSON Schema via Turn 2.")
                    return grammar_obj
            except Exception as pkg_err:
                logger.warning(f"⚠️ Turn 2 packaging encountered error: {pkg_err}. Engaging deterministic parser fallback...")

            # Fallback to deterministic code parsing if Turn 2 call fails
            items = self._parse_grammar_prose(prose_draft, source_text=source_content)
            if items:
                cefr_counts = {}
                for it in items:
                    cefr_counts[it["cefr_level"]] = cefr_counts.get(it["cefr_level"], 0) + 1
                overall = max(cefr_counts, key=cefr_counts.get) if cefr_counts else "B2"
                logger.info(f"✅ Parsed {len(items)} grammar patterns from prose draft via deterministic parser.")
                return validate_and_map(GrammarExtraction, {
                    "title": "Grammar",
                    "overall_cefr_level": overall,
                    "grammar_patterns": items,
                })

        logger.warning("⚠️ Grammar prose draft was unparsable; falling back to one-shot JSON extraction.")
        data = llm.chat(
            [{"role": "user", "content": g_prompt}],
            schema=g_schema,
            task_name=f"extract_grammar_{file_stem}_json_fallback",
        )
        return data

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

    def _render_mindmap(self, mindmap_dict):
        """Renders the horizontal Mind Map HTML page with injected data."""
        template_path = Path(__file__).parent / "templates" / "mindmap.html"
            
        if not template_path.exists():
            return f"<html><body><pre>{json.dumps(mindmap_dict, indent=2)}</pre></body></html>"

        with open(template_path, "r", encoding="utf-8") as f:
            html = f.read()
            
        json_data = json.dumps(mindmap_dict, ensure_ascii=False)
        
        if "const mindmapData =" in html:
            return re.sub(r'const mindmapData = .*?;', lambda _: f'const mindmapData = {json_data};', html, flags=re.DOTALL)
        return html.replace("</body>", f"<script>const mindmapData = {json_data};</script></body>")

    def _get_wiki_inventory(self):
        """Scans the wiki and raw directories recursively."""
        inventory = []
        wiki_dir = self.config.wiki_content_path
        if wiki_dir.exists():
            for f in wiki_dir.rglob("*.md"):
                if f.is_file():
                    if "_vocabulary" in f.name:
                        inventory.append({"name": f.stem, "type": "wiki_vocabulary", "path": f})
                    elif "_grammar" in f.name:
                        inventory.append({"name": f.stem, "type": "wiki_grammar", "path": f})
                    elif "_summary" in f.name:
                        inventory.append({"name": f.stem, "type": "wiki_summaries", "path": f})
                    else:
                        inventory.append({"name": f.stem, "type": "wiki_concepts", "path": f})
        
        # Add unit-specific sources to inventory as raw_source
        if wiki_dir.exists():
            for f in wiki_dir.rglob("sources/*.*"):
                if f.is_file() and f.suffix in [".md", ".txt"]:
                    try:
                        rel_name = str(f.relative_to(wiki_dir)).replace("\\", "/")
                        stem = f.stem
                        inventory.append({"name": rel_name, "type": "raw_source", "path": f})
                        inventory.append({"name": stem, "type": "raw_source", "path": f})
                    except Exception:
                        pass
        return inventory

processor = WikiProcessor()
