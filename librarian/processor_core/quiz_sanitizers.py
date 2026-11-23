"""Syllabus and extraction sanitization for quiz prompt construction."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple


class QuizSanitizerMixin:
    """Mixin providing syllabus sanitization for vocabulary and grammar patterns."""

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

