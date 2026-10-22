"""
Deterministic sentence indexing and vocabulary, multi-word expression, and COBUILD syntactic grammar extraction.
"""

"""
Linguistic Engine for Lexis Wiki.
Integrates spaCy (computational dependency syntax) and WordNet (lexical relations)
to achieve deterministic sentence indexing, macro grammar domain classification,
boundary-accurate phrase extraction, canonical lemmatization, and zero-collision distractors.
"""

import datetime
import hashlib
import json
import os
import re
import sqlite3
import threading
import zlib
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union



class ExtractionPipelineMixin:

    # -------------------------------------------------------------------------
    # 0. Pre-Extraction Source Hygiene & Idempotent Persistence
    # -------------------------------------------------------------------------
    _BYLINE_META_RE = re.compile(
        r"^\s*\(\s*(?:posted|updated|published|written|reported|edited|archived)\s+"
        r"(?:on|by|in|at)\b[^)]*\)\s*\.?\s*$",
        re.IGNORECASE,
    )

    @classmethod
    def _is_byline_or_meta_line(cls, line_or_para: str) -> bool:
        """
        Detects editorial bylines / timestamp metadata lines such as:
        '(Posted on January 22, 2015 by Karen)' or '(Updated on March 3, 2020 by Staff)'.
        """
        s = (line_or_para or "").strip()
        if not s:
            return False
        return bool(cls._BYLINE_META_RE.match(s))

    @classmethod
    def _normalize_inline_quotes_and_dashes(cls, s: str) -> str:
        """
        Normalizes OCR / copy-paste typography defects on a single prose line:
          - Repairs mojibake em-dash ('\\ufffdC') -> '—'
          - Normalizes spaced ellipsis ('. . .') -> '...'
          - Normalizes curly quote spacing ('“ word ”' -> '“word”')
          - When a line has balanced straight double quotes ('"'), uses quote parity
            (even = opening quote, odd = closing quote) to deterministically fix
            stuck quotes ('me."Pose' -> 'me. "Pose') and inverted quote spaces
            ('ached." I' -> 'ached. "I', 'responded," The' -> 'responded, "The',
            '" upright "' -> '"upright"').
        """
        if not s:
            return s
        # 1. Mojibake GBK em-dash repair and spaced ellipsis normalization
        s = s.replace("\ufffdC", "—")
        s = re.sub(r"\.\s+\.\s+\.", "...", s)

        # 2. Curly quote spacing normalization
        s = re.sub(r"([“«])\s+", r"\1", s)
        s = re.sub(r"\s+([”»])", r"\1", s)
        s = re.sub(r"(\w|[.!?,:;])([“«])", r"\1 \2", s)
        s = re.sub(r"([”»])(\w)", r"\1 \2", s)

        # 3. Parity-based straight double-quote ('"') spacing repair (only when '"' count is even > 0)
        q_count = s.count('"')
        if q_count >= 2 and q_count % 2 == 0:
            parts = s.split('"')
            # parts[0] is outside, parts[1] is inside quote 1, parts[2] is outside, parts[3] is inside quote 2, ...
            for i in range(1, len(parts), 2):
                # Strip accidental inner leading/trailing spaces inside quoted span: `" upright "` -> `"upright"`
                parts[i] = parts[i].strip()
            for i in range(0, len(parts), 2):
                # Outside span before an opening quote (if i < len(parts) - 1):
                if i < len(parts) - 1:
                    prev_out = parts[i].rstrip()
                    if prev_out and prev_out[-1] not in ("(", "[", "{", "—", "–", "/", "\n"):
                        parts[i] = prev_out + " "
                    else:
                        parts[i] = prev_out
                # Outside span after a closing quote (if i > 0):
                if i > 0:
                    cur_out = parts[i].lstrip()
                    if cur_out and cur_out[0] not in (".", ",", "!", "?", ";", ":", ")", "]", "}", "—", "–", "'", "’"):
                        parts[i] = " " + cur_out
                    else:
                        parts[i] = cur_out
            s = '"'.join(parts)

        return re.sub(r" {2,}", " ", s).strip()

    @classmethod
    def sanitize_source_markdown(
        cls,
        text: str,
        source_path: Optional[Union[str, Path]] = None,
    ) -> str:
        """
        Deterministic, idempotent pre-extraction source text sanitizer.
        Cleans obvious OCR / copy-paste formatting defects in body prose while leaving
        YAML frontmatter and `## Syllabus Vocabulary` / `## Appendix` untouched:
          1. Collapses multiple consecutive horizontal spaces (` {2,}`, `\t`, `\u00a0`) to a single space.
          2. Repairs missing or inverted spaces around paired quotation marks (`an"upright"message`,
             `me."Pose`, `ached." I`, `responded," The`, `" upright "`), mojibake em-dashes (`\\ufffdC`),
             and spaced ellipses (`. . .`).
          3. Appends a missing terminal period `.` to multi-word declarative prose paragraphs
             (>= 6 words, starts with a capital letter/quote, contains a finite verb/auxiliary,
             not a heading/list marker/byline/title, and lacks terminal punctuation).
        If `source_path` is provided and the sanitized text differs from disk, persists the
        cleaned text back to `source_path` so `sources/<Unit>.md`, `sentence_pool`, and
        Verbatim Quote Gate checks remain 100% synchronized.
        """
        if not text:
            return text

        fm_match = re.match(r"^(---\s*\n.*?\n---\s*\n)(.*)", text, re.DOTALL)
        frontmatter = fm_match.group(1) if fm_match else ""
        body = fm_match.group(2) if fm_match else text

        syllabus_suffix = ""
        m_appendix = re.search(r"(?mi)^(?:#+\s*(?:Syllabus\s+Vocabulary|Appendix)\b.*)$", body)
        if m_appendix:
            syllabus_suffix = body[m_appendix.start():]
            body = body[:m_appendix.start()]

        nlp = cls.get_spacy()
        raw_lines = body.split("\n")
        cleaned_lines: List[str] = []

        for idx, line in enumerate(raw_lines):
            # Normalize non-breaking spaces and tabs, collapse multiple inline spaces
            s = line.replace("\u00a0", " ").replace("\t", " ")
            s = re.sub(r" {2,}", " ", s).strip()
            if not s:
                cleaned_lines.append("")
                continue

            # Skip headings and horizontal rules
            if s.startswith("#") or re.match(r"^[-*_]{3,}\s*$", s):
                cleaned_lines.append(s)
                continue

            # Repair quote spacing, mojibake em-dashes, and spaced ellipses
            s = cls._normalize_inline_quotes_and_dashes(s)

            # Check if a standalone prose paragraph line is missing a terminal period:
            # Must not be a byline, heading, short subheading/caption, or already end with punctuation.
            next_non_empty = ""
            for nxt in raw_lines[idx + 1:]:
                if nxt.strip():
                    next_non_empty = nxt.strip()
                    break
            is_para_end = (idx == len(raw_lines) - 1) or (not raw_lines[idx + 1].strip()) or next_non_empty.startswith("#")

            if (
                is_para_end
                and not cls._is_byline_or_meta_line(s)
                and not s.endswith((".", "!", "?", '"', "”", "’", "'", ":", ";", ")", "]", "—", "–", "…"))
                and re.match(r'^[A-Z“"‘\']', s)
            ):
                words = s.split()
                # Exclude Title-Case headings (where >65% of 4+ letter words are capitalized)
                alpha_4 = [re.sub(r"^[^a-zA-Z]+|[^a-zA-Z]+$", "", w) for w in words]
                alpha_4 = [w for w in alpha_4 if len(w) >= 4 and w.isalpha()]
                cap_ratio = (
                    sum(1 for w in alpha_4 if w[0].isupper()) / len(alpha_4)
                    if alpha_4 else 1.0
                )
                if len(words) >= 6 and cap_ratio < 0.60:
                    doc_line = nlp(s)
                    has_finite_verb = any(
                        t.pos_ in ("VERB", "AUX") and t.tag_ in ("VBD", "VBP", "VBZ", "MD", "VB")
                        for t in doc_line
                    )
                    if has_finite_verb:
                        s = s + "."

            cleaned_lines.append(s)

        new_body = "\n".join(cleaned_lines)
        reconstructed = frontmatter + new_body + syllabus_suffix

        if source_path is not None and reconstructed != text:
            try:
                p = Path(source_path)
                if p.is_file():
                    p.write_text(reconstructed, encoding="utf-8")
            except Exception:
                pass

        return reconstructed

    # -------------------------------------------------------------------------
    # 1. Sentence Boundary Tokenization & Indexing ([S-1], [S-2])
    # -------------------------------------------------------------------------
    @classmethod
    def tokenize_and_index_sentences(cls, text: str) -> Tuple[str, Dict[str, str]]:
        """
        Pre-tokenizes raw source text into an indexed, complete-sentence pool.
        Returns:
            indexed_text: Text formatted with '[S-1] ... [S-2] ...' for LLM prompts.
            sentence_pool: Dict mapping 'S-1' -> pristine full sentence text.
        """
        text = cls.sanitize_source_markdown(text)
        nlp = cls.get_spacy()
        # Cleanly separate frontmatter if present so spaCy only tokenizes authentic body prose
        fm_match = re.match(r"^(---\s*\n.*?\n---\s*\n)(.*)", text, re.DOTALL)
        frontmatter = fm_match.group(1) if fm_match else ""
        body = fm_match.group(2) if fm_match else text

        # Strip trailing curriculum appendix sections (e.g. '## Syllabus Vocabulary', '## Appendix')
        # so bare vocabulary lists are never tokenized as passage evidence sentences.
        syllabus_suffix = ""
        m_appendix = re.search(r"(?mi)^(?:#+\s*(?:Syllabus\s+Vocabulary|Appendix)\b.*)$", body)
        if m_appendix:
            syllabus_suffix = body[m_appendix.start():]
            body = body[:m_appendix.start()]

        sentence_pool: Dict[str, str] = {}
        indexed_paragraphs: List[str] = []
        counter = 1

        # Normalize single-newline bullet/numbered lists so list items become independent paragraphs
        body_norm = re.sub(
            r'\n(?=[\s\u2022\u00b7\u25aa\u25ab\*\-]+|(?:\(?\d+[\.\)]\s+)|(?:\(?[a-zA-Z][\.\)]\s+))',
            r'\n\n',
            body
        )

        # Process paragraph-by-paragraph to preserve natural paragraph boundaries while streaming sentences within paragraphs
        raw_paragraphs = re.split(r'\n{2,}', body_norm.strip())
        for para in raw_paragraphs:
            para = para.strip()
            if not para:
                continue

            # If the paragraph is a pure markdown header (e.g. '## Text A', '### Section 1')
            lines = [l.strip() for l in para.split("\n") if l.strip()]
            if all(l.startswith("#") for l in lines):
                indexed_paragraphs.append(para)
                continue

            # Drop editorial bylines / timestamp metadata lines from content lines so they are never indexed as S-k
            lines = [l for l in lines if not cls._is_byline_or_meta_line(l)]
            if not lines:
                continue

            # Separate optional leading heading lines in this paragraph
            prefix_lines = []
            content_lines = []
            for l in lines:
                if not content_lines and l.startswith("#"):
                    prefix_lines.append(l)
                else:
                    content_lines.append(l)

            prefix_str = "\n\n".join(prefix_lines) + "\n\n" if prefix_lines else ""
            para_raw_text = " ".join(content_lines).strip()
            if not para_raw_text or cls._is_byline_or_meta_line(para_raw_text):
                if prefix_lines:
                    indexed_paragraphs.append("\n\n".join(prefix_lines))
                continue

            # Strip leading bullet/numbered list marker (e.g. '•', '-', '*', '1.', '(1)', 'a.')
            bullet_match = re.match(
                r"^([\s\u2022\u00b7\u25aa\u25ab\*\-]+|(?:\(?\d+[\.\)]\s*)|(?:\(?[a-zA-Z][\.\)]\s+))(.*)",
                para_raw_text
            )
            list_marker = bullet_match.group(1).strip() if bullet_match else ""
            para_text = bullet_match.group(2).strip() if bullet_match else para_raw_text

            # Protect internal punctuation inside quotation marks while allowing true sentence breaks
            # when a quote closes with '.!?' and is followed by a new capitalized sentence.
            def _mask_quoted_punct(m: re.Match) -> str:
                quote_open = m.group(1) or m.group(4) or m.group(7)
                quote_body = m.group(2) or m.group(5) or m.group(8)
                quote_close = m.group(3) or m.group(6) or m.group(9)
                after_str = m.string[m.end():]

                # Check if the quote ends with terminal punctuation (.!?) and is followed by
                # end-of-paragraph or a new independent sentence (whitespace + capital letter / opening quote),
                # rather than a lowercase continuation or speech attribution ('said ...', 'added ...').
                body_r = quote_body.rstrip()
                trailing_punct = body_r[-1] if body_r and body_r[-1] in ".!?" else ""
                after_clean = after_str.lstrip()
                is_sentence_end_quote = False
                if trailing_punct:
                    if not after_clean:
                        is_sentence_end_quote = True
                    elif re.match(r'^[“"‘\'\(\[]?[A-Z]', after_clean):
                        # Make sure it's not followed by a proper-noun attribution verb ('..." Mary said.')
                        m_attr = re.match(
                            r'^([A-Z][a-z]+)\s+(said|asked|replied|added|noted|wrote|explained|recalled|whispered|shouted|cried|argued|warned|insisted|continued|thought)\b',
                            after_clean
                        )
                        if not m_attr:
                            is_sentence_end_quote = True

                core_body = body_r[:-1] if (is_sentence_end_quote and trailing_punct) else quote_body

                # Protect abbreviations (Mr., Ms., Dr., etc.) and ellipses inside quotes so they don't get split
                protected_body = core_body.replace("...", "§ELLIPSIS§")
                protected_body = re.sub(
                    r'\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|e\.g|i\.e)\.',
                    lambda x: x.group(1) + '§ABBRDOT§',
                    protected_body,
                    flags=re.IGNORECASE
                )
                masked_body = (
                    protected_body.replace(".", " §DOT§ ")
                                  .replace("!", " §EXCL§ ")
                                  .replace("?", " §QUES§ ")
                                  .replace('§ABBRDOT§', '.')
                )
                if is_sentence_end_quote and trailing_punct:
                    p_tok = {".": "§QFINAL_DOT§", "!": "§QFINAL_EXCL§", "?": "§QFINAL_QUES§"}[trailing_punct]
                    return f"{quote_open}{masked_body}{p_tok}{quote_close} §QBREAK§ "
                return f"{quote_open}{masked_body}{quote_close}"

            # Match curly and straight quotes cleanly (curly pairs first, then straight pairs)
            masked_para_text = re.sub(
                r'(?:(“)([^”]+?)(”)|(")([^"]+? )(")|(«)([^»]+?)(»))'.replace("? )", "?)"),
                _mask_quoted_punct,
                para_text
            )

            para_doc = nlp(masked_para_text)
            raw_sents: List[str] = []
            for sent in para_doc.sents:
                for sub_sent in sent.text.split("§QBREAK§"):
                    raw_sent = sub_sent.strip()
                    if not raw_sent:
                        continue
                    # Restore quote-final punctuation before closing quote mark
                    raw_sent = raw_sent.replace("§QFINAL_DOT§", ".").replace("§QFINAL_EXCL§", "!").replace("§QFINAL_QUES§", "?")
                    raw_sent = re.sub(r'\s*§DOT§\s*', '. ', raw_sent)
                    raw_sent = re.sub(r'\s*§EXCL§\s*', '! ', raw_sent)
                    raw_sent = re.sub(r'\s*§QUES§\s*', '? ', raw_sent)
                    raw_sent = raw_sent.replace("§ELLIPSIS§", "...")
                    raw_sent = re.sub(r'([,:;])(["“«])(?=\w)', r'\1 \2', raw_sent)
                    raw_sent = re.sub(r'([.!?,\u2026])\s+(["”»])(?=\s|$|[)\].,:;!?])', r'\1\2', raw_sent)
                    raw_sent = re.sub(r'\s+([”»])', r'\1', raw_sent)
                    raw_sent = re.sub(r'([“«])\s+', r'\1', raw_sent)
                    raw_sent = re.sub(r'\s{2,}', ' ', raw_sent).strip()
                    if raw_sent:
                        raw_sents.append(raw_sent)

            # Merge any sentences where spaCy split across an unclosed quotation mark
            # or left an orphan attribution/punctuation fragment (e.g. `”, added Tom.`)
            merged_sents: List[str] = []
            for seg in raw_sents:
                if merged_sents:
                    prev = merged_sents[-1]
                    unclosed_curly = prev.count("“") > prev.count("”")
                    unclosed_straight = (prev.count('"') % 2) == 1
                    starts_with_close_or_punct = bool(re.match(r'^[”».,:;!?\)]', seg)) or (
                        unclosed_straight and seg.startswith('"')
                    )
                    if unclosed_curly or unclosed_straight or starts_with_close_or_punct:
                        sep = "" if starts_with_close_or_punct else " "
                        merged_sents[-1] = re.sub(r'\s{2,}', ' ', f"{prev}{sep}{seg}").strip()
                        continue
                merged_sents.append(seg)

            para_sent_parts = []
            for raw_sent in merged_sents:
                # Clean enclosing markdown bold/italic formatting from pristine sentence text
                sent_clean = re.sub(r'^\*+|\*+$', '', raw_sent).strip()
                if cls._is_byline_or_meta_line(sent_clean):
                    continue
                words = [t for t in nlp(sent_clean) if t.is_alpha]
                if len(words) < 2 and not sent_clean.endswith((".", "?", "!")):
                    para_sent_parts.append(raw_sent)
                    continue

                sid = f"S-{counter}"
                sentence_pool[sid] = sent_clean
                para_sent_parts.append(f"[{sid}] {sent_clean}")
                counter += 1

            marker_str = f"{list_marker} " if list_marker else ""
            indexed_para = prefix_str + marker_str + " ".join(para_sent_parts)
            indexed_paragraphs.append(indexed_para.strip())

        if frontmatter:
            indexed_text = frontmatter + "\n\n" + "\n\n".join(indexed_paragraphs)
        else:
            indexed_text = "\n\n".join(indexed_paragraphs)
        if syllabus_suffix:
            indexed_text = indexed_text.rstrip() + "\n\n" + syllabus_suffix.strip()
        return indexed_text, sentence_pool

    @classmethod
    def snap_to_sentence_pool(cls, raw_quote_or_id: str, sentence_pool: Dict[str, str]) -> Optional[str]:
        """
        Deterministically snaps a model-provided sentence ID (e.g. '[S-3]', 'S-3') or a
        partial quote with ellipsis back to the pristine, authentic source sentence in the pool.
        Eliminates trailing ellipses, copy-paste hallucinations, and verbatim mismatches.
        """
        if not raw_quote_or_id or not sentence_pool:
            return None

        clean_input = re.sub(r"^\s*\[?\bS-\d+\b\]?\s*[:\-]??\s*", "", raw_quote_or_id.strip(), flags=re.IGNORECASE).strip()

        # 1. Direct ID lookup (e.g. '[S-3]' or 'S-3' or 'Sentence 3')
        id_match = re.search(r"\bS-(\d+)\b", raw_quote_or_id, re.IGNORECASE)
        if id_match:
            sid = f"S-{id_match.group(1)}"
            if sid in sentence_pool:
                return re.sub(r"^\s*\[?\bS-\d+\b\]?\s*[:\-]??\s*", "", sentence_pool[sid], flags=re.IGNORECASE).strip()

        # 2. Exact match in sentence pool
        for sid, sent in sentence_pool.items():
            sent_clean = re.sub(r"^\s*\[?\bS-\d+\b\]?\s*[:\-]??\s*", "", sent, flags=re.IGNORECASE).strip()
            if clean_input.lower() == sent_clean.lower():
                return sent_clean

        # 3. Substring containment: if quote is a fragment of a pool sentence
        clean_target = re.sub(r"\.{3,}|…", "", clean_input).strip().lower()
        if len(clean_target) >= 15:
            for sid, sent in sentence_pool.items():
                sent_clean = re.sub(r"^\s*\[?\bS-\d+\b\]?\s*[:\-]??\s*", "", sent, flags=re.IGNORECASE).strip()
                if clean_target in sent_clean.lower():
                    return sent_clean

        # 4. High overlap fallback
        clean_tokens = set(re.findall(r"\w{3,}", clean_target))
        if clean_tokens:
            best_sent = None
            max_overlap = 0.0
            for sid, sent in sentence_pool.items():
                sent_clean = re.sub(r"^\s*\[?\bS-\d+\b\]?\s*[:\-]??\s*", "", sent, flags=re.IGNORECASE).strip()
                sent_tokens = set(re.findall(r"\w{3,}", sent_clean.lower()))
                if not sent_tokens:
                    continue
                overlap = len(clean_tokens & sent_tokens) / len(clean_tokens)
                if overlap > max_overlap and overlap >= 0.75:
                    max_overlap = overlap
                    best_sent = sent_clean
            if best_sent:
                return best_sent

    @classmethod
    def get_sentence_id(cls, raw_quote_or_id: str, sentence_pool: Dict[str, str]) -> Optional[str]:
        """
        Returns the canonical sentence identifier (e.g. 'S-3') for a given quote or ID.
        """
        if not raw_quote_or_id or not sentence_pool:
            return None

        clean_input = raw_quote_or_id.strip()

        # 1. Direct ID lookup
        id_match = re.search(r"\bS-(\d+)\b", clean_input, re.IGNORECASE)
        if id_match:
            sid = f"S-{id_match.group(1)}"
            if sid in sentence_pool:
                return sid

        # 2. Exact match
        for sid, sent in sentence_pool.items():
            if clean_input.lower() == sent.lower():
                return sid

        # 3. Substring containment
        clean_target = re.sub(r"\.{3,}|…", "", clean_input).strip().lower()
        if len(clean_target) >= 15:
            for sid, sent in sentence_pool.items():
                if clean_target in sent.lower():
                    return sid

        # 4. Token overlap fallback
        clean_tokens = set(re.findall(r"\w{3,}", clean_target))
        if clean_tokens:
            best_sid = None
            max_overlap = 0.0
            for sid, sent in sentence_pool.items():
                sent_tokens = set(re.findall(r"\w{3,}", sent.lower()))
                if not sent_tokens:
                    continue
                overlap = len(clean_tokens & sent_tokens) / len(clean_tokens)
                if overlap > max_overlap and overlap >= 0.75:
                    max_overlap = overlap
                    best_sid = sid
            if best_sid:
                return best_sid

        return None

    # -------------------------------------------------------------------------
    # 3. Dependency-Based Macro Domain Classification Gate
    # -------------------------------------------------------------------------
    @classmethod
    def classify_grammar_dependency(cls, sentence: str) -> Optional[str]:
        """
        Classifies complex sentence into one of the Four Macro Functional Domains
        using deterministic spaCy dependency trees and syntactic relations:
          - 'Rhetoric & Emphasis'
          - 'Cohesion & Framing'
          - 'Information Packaging'
          - 'Logic & Stance'
        Returns None if no unambiguous structural pattern dominates.
        """
        sentence_norm = sentence.replace('\u2019', "'").replace('\u2018', "'")
        nlp = cls.get_spacy()
        doc = nlp(sentence_norm)

        # Declarative DependencyMatcher Engine Pass (High-Precision DSL)
        try:
            from librarian.grammar_patterns import GrammarPatternEngine
            gp_engine = GrammarPatternEngine.get_engine(nlp)
            matched_patterns = gp_engine.match_sentence(doc)
            if matched_patterns:
                return matched_patterns[0]["category"]
        except Exception:
            pass

        # ---------------------------------------------------------------------
        # 1. Rhetoric & Emphasis:
        #    a. Fronted Inversion: aux or ROOT precedes nsubj in linear word order
        #    b. Cleft Focus: expl ('it') + 'is/was' + focal constituent + relcl/that...
        #    c. Antithesis: 'not... but...' parallel coordinating balance
        # ---------------------------------------------------------------------
        # Inversion check
        for token in doc:
            if token.dep_ == "nsubj":
                # Check if auxiliary or root verb precedes nsubj (excluding questions)
                head = token.head
                if head.pos_ == "VERB" or head.pos_ == "AUX":
                    aux_children = [c for c in head.children if c.dep_ in ("aux", "auxpass")]
                    for aux in aux_children:
                        if aux.i < token.i and not sentence.strip().endswith("?"):
                            # Check if fronted negative/restrictive adverbial starts sentence
                            first_token = doc[0].lemma_.lower()
                            if first_token in ("hardly", "scarcely", "seldom", "never", "rarely", "barely", "only", "not", "no"):
                                return "Rhetoric & Emphasis"

        # Cleft vs Dummy-it check
        for token in doc:
            if token.text.lower() == "it" and token.dep_ in ("expl", "nsubj"):
                # Check copula verb
                head = token.head
                if head.lemma_ in ("be", "seem"):
                    if any(c.dep_ == "attr" and c.lemma_.lower() == "time" for c in head.children):
                        continue
                    # 1. Relcl under focal constituent with relative pronoun
                    has_relcl = False
                    for child in head.children:
                        if child.dep_ == "relcl":
                            if any(gc.text.lower() in ("that", "who", "whom", "which", "whose") for gc in child.children):
                                has_relcl = True
                        for gc in child.children:
                            if gc.dep_ == "relcl":
                                if any(ggc.text.lower() in ("that", "who", "whom", "which", "whose") for ggc in gc.children):
                                    has_relcl = True
                    if has_relcl:
                        return "Rhetoric & Emphasis"

                    # 2. Prepositional or adverbial cleft (e.g. It's in December that we tend...)
                    for child in head.children:
                        if child.dep_ in ("ccomp", "advcl"):
                            has_marker = any(
                                gc.dep_ in ("mark", "nsubj", "advmod", "dobj", "pobj")
                                and gc.text.lower() in ("that", "who", "whom", "where", "when")
                                for gc in child.children
                            )
                            if has_marker:
                                # Ensure focal element is not an evaluative adjective or extraposition noun (dummy-it)
                                _EXTRAPOSITION_NOUNS = {
                                    "surprise", "wonder", "secret", "doubt", "coincidence", "accident",
                                    "pity", "shame", "mystery", "fact", "truth", "news", "use", "good",
                                    "point", "time", "question", "matter"
                                }
                                has_eval_adj = any(
                                    (c.dep_ == "acomp" and c.pos_ == "ADJ")
                                    or (c.dep_ == "attr" and c.lemma_.lower() in _EXTRAPOSITION_NOUNS)
                                    for c in head.children
                                )
                                if not has_eval_adj:
                                    return "Rhetoric & Emphasis"

        # Antithesis / Correlative Parallelism check (not... but..., not only... but also)
        text_lower = sentence_norm.lower()
        if re.search(r"\bnot\s+only\b.*?\bbut\s+also\b", text_lower):
            return "Rhetoric & Emphasis"

        # Resultative Degree Constructions (so + Adj/Adv + that / such + NP + that)
        if re.search(r"\bso\s+[a-z-]{3,}(?:\s+[a-z-]{3,})?\s+that\b", text_lower) or re.search(r"\bsuch\s+(?:a\s+|an\s+)?[a-z-]{3,}(?:\s+[a-z-]{3,}){0,2}\s+that\b", text_lower):
            return "Rhetoric & Emphasis"

        # ---------------------------------------------------------------------
        # 2. Logic & Stance:
        #    Conditionals (if/unless/as long as/provided that) or Concessives (although/even though/while/whereas/in spite of)
        #    Adversative transitions (however/nevertheless/nonetheless)
        #    Dual-stance coordinate contrast (..., but [S] also...)
        # ---------------------------------------------------------------------
        # Multi-word subordinating connectives
        if re.search(r"\b(as long as|so long as|provided that|providing that|in case|on condition that)\b", text_lower):
            return "Logic & Stance"

        subordinating_conjs = {"although", "though", "while", "whereas", "even though", "if", "unless", "provided", "providing"}
        for token in doc:
            if token.dep_ == "mark" and token.lemma_.lower() in subordinating_conjs:
                if token.lemma_.lower() == "if" and token.head.dep_ != "advcl":
                    continue
                # Subjectless 'If/While/Though/Although + V-ing/V-ed/Adj' is a reduced non-finite adjunct, not a finite clause
                if not any(c.dep_ in ("nsubj", "nsubjpass", "expl", "csubj", "csubjpass") for c in token.head.children):
                    continue
                return "Logic & Stance"

        # Stance / Adversative transition (However, / Nevertheless, ...)
        if re.match(r"^(however|nevertheless|nonetheless|conversely|in contrast)\b", text_lower):
            return "Logic & Stance"

        # Stance / Contrastive coordinate balance (..., but [S] also...)
        if re.search(r"\b,\s*but\s+[a-z]+\s+(also|still|yet)\b", text_lower):
            return "Logic & Stance"

        # ---------------------------------------------------------------------
        # 3. Cohesion & Framing:
        #    a. Propositional Encapsulation: , which + [interpretive verb] + that
        #    b. Shell Noun Frames: [shell noun] + that [clause]
        # ---------------------------------------------------------------------
        # Propositional encapsulation: which + interpretive verb + that
        interpretive_verbs = {"mean", "suggest", "indicate", "show", "demonstrate", "prove", "imply", "reveal"}
        for token in doc:
            if token.dep_ in ("relcl", "advcl") and token.lemma_.lower() in interpretive_verbs:
                # Check if subject is 'which'
                subj_children = [c for c in token.children if c.dep_ in ("nsubj", "nsubjpass") and c.text.lower() == "which"]
                if subj_children:
                    has_that_comp = any(child.dep_ == "ccomp" for child in token.children)
                    if has_that_comp:
                        return "Cohesion & Framing"

        # Shell noun patterns: fact, idea, conclusion, claim, belief, hypothesis + that
        shell_nouns = {"fact", "idea", "conclusion", "claim", "belief", "hypothesis", "notion", "argument", "evidence", "assumption"}
        for token in doc:
            if token.lemma_.lower() in shell_nouns:
                for child in token.children:
                    if child.dep_ in ("acl", "appos", "ccomp") and any(gc.dep_ == "mark" and gc.lower_ in ("that", "whether") for gc in child.children):
                        return "Cohesion & Framing"

        # ---------------------------------------------------------------------
        # 4. Information Packaging:
        #    a. Evaluative Dummy-It: It is + [adj/noun] + that/to [csubj/ccomp]
        #    b. Dense Object Complement: make/find/render/keep + Object + Adj
        #    c. Non-finite participial adjuncts: advcl with VerbForm=Part
        #    d. Non-finite subject nominalization (Gerund / Infinitive phrase as Subject)
        #    e. Correlative comparative: The more..., the more...
        #    f. Dense Prepositional Frame: Instead of / By / Thanks to + [V-ing/NP]
        #    g. Relative clauses: which/who/whom/whose + VP
        #    h. Elaborative non-restrictive relative clause: , which + VP
        # ---------------------------------------------------------------------
        # Correlative comparative: The more..., the more...
        COMPARATIVE_WORDS = r"(?:more|less|fewer|better|worse|[a-z]{2,}er)"
        if re.search(rf"\bthe\s+{COMPARATIVE_WORDS}\b(?!\s+hand\b).*?(?:,\s*|\band\s+the\s+{COMPARATIVE_WORDS}\b.*?,?\s*)\bthe\s+{COMPARATIVE_WORDS}\b", text_lower):
            return "Information Packaging"

        # Dense Prepositional Frame (Instead of, By doing, Due to, Thanks to + V-ing/NP, [Subject] + [VP])
        if re.match(r"^(instead of|by|through|despite|in spite of|thanks to|due to|owing to)\s+[a-z0-9\s-]+?,", text_lower):
            return "Information Packaging"

        # Non-finite subject nominalization (Gerund or Infinitive as subject)
        # Handles direct sentence subject or embedded clause subject (e.g. That is why [being on wheels] means...)
        for token in doc:
            if token.tag_ == "VBG" and token.dep_ in ("nsubj", "nsubjpass", "csubj"):
                return "Information Packaging"
            if token.tag_ == "VB" and token.dep_ == "csubj":
                return "Information Packaging"
            # Embedded gerund subject in why/that/wh-clauses where spaCy might label it advcl or pobj
            if token.tag_ == "VBG" and token.dep_ in ("advcl", "csubj") and token.head.pos_ in ("VERB", "AUX"):
                # Check if this VBG introduces a gerund clause following 'why' or copula
                has_wh = any(c.dep_ == "advmod" and c.lemma_.lower() in ("why", "how", "what", "where") for c in token.children)
                if has_wh:
                    return "Information Packaging"
                head_subjs = [c for c in token.head.children if c.dep_ in ("nsubj", "nsubjpass")]
                if not head_subjs and token.i < token.head.i:
                    return "Information Packaging"

        # Evaluative Dummy-It (spaCy tags 'It' as nsubj/expl and the complement clause as ccomp/csubj)
        _AMBIENT_PREDS = {
            "daylight", "dark", "night", "morning", "afternoon", "evening", "noon", "midnight",
            "dawn", "dusk", "winter", "summer", "spring", "autumn", "fall", "cold", "hot", "warm",
            "cool", "sunny", "rainy", "snowy", "windy", "cloudy", "late", "early", "o'clock"
        }
        for token in doc:
            if token.text.lower() == "it" and token.dep_ in ("expl", "nsubj"):
                head = token.head
                if head.lemma_ in ("be", "seem") and not any(c.dep_ == "advmod" and c.lower_ in ("when", "where") for c in head.children):
                    eval_children = [
                        child for child in head.children
                        if child.dep_ in ("acomp", "attr") and child.lemma_.lower() not in _AMBIENT_PREDS
                    ]
                    comp_children = [
                        child for child in head.children
                        if child.dep_ in ("ccomp", "csubj", "xcomp")
                        and child.tag_ not in ("VBG", "VBN")
                        and not any(gc.dep_ == "advmod" and gc.lower_ in ("when", "where") for gc in child.children)
                    ]
                    if eval_children and comp_children and comp_children[0].i > eval_children[0].i:
                        return "Information Packaging"

        # Evaluative Dummy-It Object (find/make/think it adj to-V)
        for t in doc:
            if t.lemma_ in ("find", "make", "think", "consider", "deem", "believe") and t.pos_ in ("VERB", "AUX"):
                for c in t.children:
                    if c.pos_ == "ADJ" and any(gc.text.lower() == "it" and gc.dep_ in ("nsubj", "dobj") for gc in c.children):
                        return "Information Packaging"

        # Dense Object Complement / Causative (make/find/render/keep + Object + Adj/Complement)
        for t in doc:
            if t.lemma_ in ("make", "find", "render", "keep") and t.pos_ in ("VERB", "AUX"):
                for c in t.children:
                    if c.dep_ in ("ccomp", "oprd") and c.pos_ == "ADJ":
                        return "Information Packaging"

        # Non-finite participial adjuncts (advcl with VBG or VBN)
        _OBJ_VING_VERBS = {"have", "find", "spend", "waste", "get", "catch", "keep", "leave", "start", "stop", "prevent", "busy"}
        for token in doc:
            if token.dep_ == "advcl" and token.tag_ in ("VBG", "VBN"):
                # Exclude 'have trouble/difficulty/time V-ing' and 'spend/waste time V-ing' complement frames
                if token.tag_ == "VBG" and token.head.lemma_.lower() in _OBJ_VING_VERBS and token.i > token.head.i:
                    if not any(t.text == "," for t in doc[token.head.i : token.i]):
                        continue
                # Ensure it has no subject of its own (non-finite) and not a preposition
                has_subj = any(c.dep_ in ("nsubj", "nsubjpass") for c in token.children)
                if not has_subj and token.lemma_.lower() not in ("include", "accord", "regard", "concern"):
                    return "Information Packaging"

        # Relative clauses (which, that, who, whom, whose)
        for token in doc:
            if token.dep_ == "relcl":
                if any(c.text.lower() in ("which", "that", "who", "whom", "whose") for c in token.children):
                    return "Information Packaging"

        # Passive voice (agentless or agentive passive foregrounding patient)
        _UNACCUSATIVE_STATE_VERBS = {"go", "come", "fall", "rise", "arrive", "depart", "vanish", "disappear", "die", "remain", "stay"}
        for token in doc:
            if token.dep_ in ("nsubjpass", "auxpass"):
                head = token.head
                if head.pos_ in ("VERB", "AUX") and head.lemma_.lower() not in _UNACCUSATIVE_STATE_VERBS:
                    return "Information Packaging"

        return None

    @classmethod
    def generate_cobuild_formula(cls, sentence: str, category: Optional[str] = None) -> str:
        """
        Deterministically derives standard algebraic COBUILD slot formula from sentence
        via spaCy dependency trees and syntactic pattern anchors.
        Replaces prone-to-hallucination LLM formula drafting with zero-token invariant logic.
        """
        sentence_norm = sentence.replace('\u2019', "'").replace('\u2018', "'")
        nlp = cls.get_spacy()
        doc = nlp(sentence_norm)
        cat = category or cls.classify_grammar_dependency(sentence_norm) or ""
        text_lower = sentence_norm.lower()

        # Declarative DependencyMatcher Engine Pass (High-Precision DSL)
        try:
            from librarian.grammar_patterns import GrammarPatternEngine
            gp_engine = GrammarPatternEngine.get_engine(nlp)
            matched_patterns = gp_engine.match_sentence(doc)
            if matched_patterns:
                if cat:
                    matched_cand = next((m for m in matched_patterns if m["category"].lower() == cat.lower() or cat.lower() in m["name"].lower() or cat.lower() in m["pattern_id"].lower()), None)
                else:
                    matched_cand = matched_patterns[0]
                if matched_cand and matched_cand.get("formula"):
                    return matched_cand["formula"]
        except Exception:
            pass

        # 1. Rhetoric & Emphasis
        if cat == "Rhetoric & Emphasis" or not cat:
            # Inversion
            for token in doc:
                if token.dep_ == "nsubj":
                    head = token.head
                    if head.pos_ in ("VERB", "AUX"):
                        aux_children = [c for c in head.children if c.dep_ in ("aux", "auxpass") and c.i < token.i]
                        if aux_children and not sentence.strip().endswith("?"):
                            first_tok = doc[0].lemma_.lower()
                            if first_tok in ("hardly", "scarcely", "seldom", "never", "rarely", "barely", "only", "not", "no"):
                                return f"{doc[0].text} + [aux/be] + [Subject] + [VP]"
            # Cleft
            _EXTRAPOSITION_NOUNS = {
                "surprise", "wonder", "secret", "doubt", "coincidence", "accident",
                "pity", "shame", "mystery", "fact", "truth", "news", "use", "good",
                "point", "time", "question", "matter"
            }
            for token in doc:
                if token.text.lower() == "it" and token.dep_ in ("expl", "nsubj") and token.head.lemma_ in ("be", "seem"):
                    if any(
                        (c.dep_ == "acomp" and c.pos_ == "ADJ")
                        or (c.dep_ == "attr" and c.lemma_.lower() in _EXTRAPOSITION_NOUNS)
                        for c in token.head.children
                    ):
                        continue
                    # Check relcl
                    has_relcl = any(
                        (c.dep_ == "relcl" and any(gc.text.lower() in ("that", "who", "whom", "which", "whose") for gc in c.children))
                        or any(gc.dep_ == "relcl" and any(ggc.text.lower() in ("that", "who", "whom", "which", "whose") for ggc in gc.children) for gc in c.children)
                        for c in token.head.children
                    )
                    # Check prepositional or adverbial cleft
                    has_cleft_marker = any(
                        c.dep_ in ("ccomp", "advcl")
                        and any(gc.dep_ in ("mark", "nsubj", "advmod", "dobj", "pobj") and gc.text.lower() in ("that", "who", "whom", "where", "when") for gc in c.children)
                        for c in token.head.children
                    )
                    if has_relcl or has_cleft_marker:
                        return "It + [be] + [Focal Element] + that/who + [Clause]"
            # Correlative Parallelism (with or without Fronted Negative Inversion)
            if re.search(r"\bnot\s+only\b.*?\bbut\s+(?:also\b|[a-z]+\s+also\b)", text_lower):
                if re.search(r"(?:^|[,;—–-]\s*)not\s+only\s+(?:can|could|will|would|shall|should|may|might|must|do|does|did|have|has|had|am|is|are|was|were)\b", text_lower):
                    return "Not only + [Aux] + [Subject] + [VP], but + [Subject] + [also] + [VP]"
                return "[Subject] + not only + [VP], but also + [VP]"
            # Resultative so/such ... that
            if re.search(r"\bsuch\s+(?:a\s+|an\s+)?[a-z-]{3,}(?:\s+[a-z-]{3,}){0,2}\s+that\b", text_lower):
                return "such + [NP] + that + [Result Clause]"
            if re.search(r"\bso\s+[a-z-]{3,}(?:\s+[a-z-]{3,})?\s+that\b", text_lower):
                return "so + [Adj/Adv] + that + [Result Clause]"

        # 2. Logic & Stance
        if cat == "Logic & Stance" or not cat:
            # Multi-word conditional & concessive connectives
            m_conn = re.search(r"\b(as long as|so long as|provided that|providing that|in case|on condition that|even though|even if)\b", text_lower)
            if m_conn:
                conn_name = m_conn.group(1).capitalize()
                return f"{conn_name} + [Clause], [Subject] + [VP]"
            for token in doc:
                if token.dep_ == "mark" and token.lemma_.lower() in {"although", "though", "while", "whereas", "even though", "if", "unless", "provided"}:
                    if token.lemma_.lower() == "if" and token.head.dep_ != "advcl":
                        continue
                    if not any(c.dep_ in ("nsubj", "nsubjpass", "expl", "csubj", "csubjpass") for c in token.head.children):
                        continue
                    prev_w = doc[token.i - 1].lower_ if token.i > 0 else ""
                    if prev_w == "even" and token.lower_ in ("if", "though"):
                        mark_word = f"Even {token.lower_}"
                    else:
                        mark_word = token.text.capitalize()
                    return f"{mark_word} + [Clause], [Subject] + [VP]"
            m_trans = re.match(r"^(however|nevertheless|nonetheless|conversely|in contrast)\b", text_lower)
            if m_trans:
                return f"{m_trans.group(1).capitalize()}, [Subject] + [Modal/VP]"
            if re.search(r"\b,\s*but\s+[a-z]+\s+(also|still|yet)\b", text_lower):
                return "[Subject] + [VP], but [Subject] + [also] + [VP]"

        # 3. Cohesion & Framing
        if cat == "Cohesion & Framing" or not cat:
            # Propositional Encapsulation: , which + [interpretive verb] + that
            interpretive_verbs = {"mean", "suggest", "indicate", "show", "demonstrate", "prove", "imply", "reveal"}
            for token in doc:
                if token.dep_ in ("relcl", "advcl") and token.lemma_.lower() in interpretive_verbs:
                    if any(c.dep_ in ("nsubj", "nsubjpass") and c.text.lower() == "which" for c in token.children):
                        return f", which + {token.lemma_.lower()}s + that + [Proposition Clause]"
            # Shell noun frame
            shell_nouns = {"fact", "idea", "conclusion", "claim", "belief", "hypothesis", "notion", "argument", "evidence", "assumption"}
            for token in doc:
                if token.lemma_.lower() in shell_nouns:
                    if any(
                        c.dep_ in ("acl", "appos", "ccomp")
                        and any(gc.dep_ == "mark" and gc.lower_ in ("that", "whether") for gc in c.children)
                        for c in token.children
                    ):
                        return f"The + {token.lemma_.lower()} + that/of + [Proposition Clause]"

        # 4. Information Packaging
        if cat == "Information Packaging" or not cat:
            # Correlative comparative: The more..., the more...
            COMPARATIVE_WORDS = r"(?:more|less|fewer|better|worse|[a-z]{2,}er)"
            if re.search(rf"\bthe\s+{COMPARATIVE_WORDS}\b(?!\s+hand\b).*?(?:,\s*|\band\s+the\s+{COMPARATIVE_WORDS}\b.*?,?\s*)\bthe\s+{COMPARATIVE_WORDS}\b", text_lower):
                return "The + [comparative] + [Clause], the + [comparative] + [Clause]"
            # Dense Prepositional Frame (Instead of, By doing, Thanks to...)
            m_prep = re.match(r"^(instead of|by|through|despite|in spite of|thanks to|due to|owing to)\b", text_lower)
            if m_prep:
                prep_head = m_prep.group(1).capitalize()
                return f"{prep_head} + [V-ing/NP], [Subject] + [VP]"
            # Non-finite Subject Nominalization (Gerund / Infinitive phrase as Subject)
            # Detects direct sentence subject or embedded predicate clause subject (e.g. That is why [being on wheels] means...)
            for token in doc:
                is_gerund_subj = (token.tag_ == "VBG" and token.dep_ in ("nsubj", "nsubjpass", "csubj"))
                is_infinitive_subj = (token.tag_ == "VB" and token.dep_ == "csubj")
                is_embedded_gerund = (
                    token.tag_ == "VBG" and token.dep_ in ("advcl", "csubj")
                    and (
                        any(c.dep_ == "advmod" and c.lemma_.lower() in ("why", "how", "what", "where") for c in token.children)
                        or (
                            token.head.pos_ in ("VERB", "AUX")
                            and not any(c.dep_ in ("nsubj", "nsubjpass") for c in token.head.children)
                            and token.i < token.head.i
                        )
                    )
                )
                if is_gerund_subj or is_embedded_gerund:
                    if re.search(r"\b(that|this)\s+(?:is|’s|'s)\s+why\b", text_lower):
                        return "[Subject] + is why + [V-ing / Gerund Phrase] + [VP]"
                    return "[V-ing / Gerund Phrase] + [Predicate Verb] + [Complement/Object]"
                if is_infinitive_subj:
                    return "To + [Infinitive Phrase] + [Predicate Verb] + [Complement/Object]"
            # Dummy-It Object Extraposition (find/make/think it adj to-V)
            for t in doc:
                if t.lemma_ in ("find", "make", "think", "consider", "deem", "believe") and t.pos_ in ("VERB", "AUX"):
                    for c in t.children:
                        if c.pos_ == "ADJ" and any(gc.text.lower() == "it" and gc.dep_ in ("nsubj", "dobj") for gc in c.children):
                            return f"[Subject] + {t.lemma_} + it + [{c.lemma_.capitalize()}] + to-V"
            # Dense Object Complement / Causative (make/find/render/keep + Object + Adj)
            for t in doc:
                if t.lemma_ in ("make", "find", "render", "keep") and t.pos_ in ("VERB", "AUX"):
                    for c in t.children:
                        if c.dep_ in ("ccomp", "oprd") and c.pos_ == "ADJ":
                            return f"[Subject] + {t.lemma_} + [Object] + [Adj]"
            # Dummy-It Subject Extraposition
            _AMBIENT_PREDS = {
                "daylight", "dark", "night", "morning", "afternoon", "evening", "noon", "midnight",
                "dawn", "dusk", "winter", "summer", "spring", "autumn", "fall", "cold", "hot", "warm",
                "cool", "sunny", "rainy", "snowy", "windy", "cloudy", "late", "early", "o'clock"
            }
            if any(t.text.lower() == "it" and t.dep_ in ("expl", "nsubj") for t in doc):
                for t in doc:
                    if t.text.lower() == "it" and t.head.lemma_ in ("be", "seem") and not any(c.dep_ == "advmod" and c.lower_ in ("when", "where") for c in t.head.children):
                        eval_children = [
                            c for c in t.head.children
                            if c.dep_ in ("acomp", "attr") and c.lemma_.lower() not in _AMBIENT_PREDS
                        ]
                        comp_children = [
                            c for c in t.head.children
                            if c.dep_ in ("ccomp", "csubj", "xcomp")
                            and c.tag_ not in ("VBG", "VBN")
                            and not any(gc.dep_ == "advmod" and gc.lower_ in ("when", "where") for gc in c.children)
                        ]
                        if eval_children and comp_children and comp_children[0].i > eval_children[0].i:
                            return "It + [be] + [Evaluative Adj] + to-V/that + [Clause]"
            # Participial adjunct (strictly requiring non-subject participle, comma boundary when post-positioned, verbal dependents, and actual main subject)
            for token in doc:
                if token.dep_ == "advcl" and token.tag_ in ("VBG", "VBN"):
                    if token.i > token.head.i and not any(t.text == "," for t in doc[token.head.i : token.i]):
                        continue
                    if token.i + 1 < len(doc) and doc[token.i + 1].pos_ in ("NOUN", "PROPN") and doc[token.i + 1].dep_ in ("dobj", "pobj", "nsubj", "attr"):
                        continue
                    if not any(c.dep_ in ("dobj", "prep", "advmod", "acomp", "xcomp", "ccomp", "prt", "agent", "npadvmod", "oprd") for c in token.children):
                        continue
                    if not any(c.dep_ in ("nsubj", "nsubjpass") for c in token.children) and token.lemma_.lower() not in ("include", "accord", "regard"):
                        # Ensure main verb actually has a subject, otherwise it might be a gerund subject misclassified as advcl
                        main_has_subj = any(c.dep_ in ("nsubj", "nsubjpass") for c in token.head.children)
                        if main_has_subj:
                            is_fronted = token.i < token.head.i
                            has_having = any(c.dep_ in ("aux", "auxpass") and c.lower_ == "having" for c in token.children)
                            has_being = any(c.dep_ in ("aux", "auxpass") and c.lower_ == "being" for c in token.children)
                            if has_having:
                                v_type = "Having + V-ed"
                            elif has_being:
                                v_type = "Being + V-ed"
                            elif token.tag_ == "VBG":
                                v_type = "V-ing"
                            else:
                                v_type = "V-ed"
                            if is_fronted:
                                return f"[{v_type} Phrase], [Subject] + [VP]"
                            else:
                                return f"[Subject] + [VP], [{v_type} Phrase]"
            # Relative clauses (which, that, who, whom, whose)
            for token in doc:
                if token.dep_ == "relcl":
                    rel_pron = next((c.text.lower() for c in token.children if c.text.lower() in ("which", "that", "who", "whom", "whose")), None)
                    if rel_pron:
                        return f"[NP] + {rel_pron} + [VP]"
            # Elaborative clause
            if re.search(r",\s*which\s+[a-z]+", text_lower):
                return "[Subject] + [VP], which + [VP]"

            # Passive voice
            for token in doc:
                if token.dep_ in ("nsubjpass", "auxpass") and token.head.lemma_.lower() not in {"go", "come", "fall", "rise", "arrive", "depart", "vanish", "disappear", "die", "remain", "stay"}:
                    return "[Subject] + [be/modal be] + [V-ed]"

        # Default fallback standard formula
        return "[Subject] + [VP] + [Clause]"

    @classmethod
    def mine_grammar_skeletons(
        cls,
        sentence_pool: Dict[str, str],
        target_count: int = 5,
        syllabus_grammar: Optional[List[str]] = None,
        cefr_band: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Deterministically mines genuine, non-trivial grammar structural skeletons from the sentence pool.
        Selects sentences exhibiting rich dependency structures across the Four Macro Domains,
        prioritized by the 3-band CEFR priority sequence (A1_A2, B1_B2, C1_C2) and intra-pattern complexity.
        Returns a list of dicts: [{'sid': 'S-9', 'quote': '...', 'category': '...', 'pattern_formula': '...'}]
        """
        raw_skeletons: List[Dict[str, Any]] = []
        band = cefr_band or cls.resolve_grammar_cefr_band(sentence_pool)

        nlp = cls.get_spacy()
        from librarian.grammar_patterns import GrammarPatternEngine
        gp_engine = GrammarPatternEngine.get_engine(nlp)

        for sid, sent in sentence_pool.items():
            sent_clean = sent.strip()
            # Skip very short sentences or editorial bylines
            if len(sent_clean.split()) < 7 or cls._is_byline_or_meta_line(sent_clean):
                continue
            # Skip passage title (S-1) when passage has multiple sentences, section headers, and syllabus word-lists
            if sid == "S-1" and len(sentence_pool) > 1:
                continue
            # If sentence does not end with standard sentence punctuation, skip (headings/titles)
            if not sent_clean.endswith((".", "!", "?", '"', "'", "”", "’")):
                continue
            # If sentence is a colon-separated headline/title (e.g. "Title: Subtitle") without terminal punctuation
            if ":" in sent_clean and not any(sent_clean.endswith(p) for p in (".", "!", "?")):
                continue

            sent_norm = sent_clean.replace('\u2019', "'").replace('\u2018', "'")
            doc = nlp(sent_norm)
            matched_patterns = gp_engine.match_sentence(doc)

            if matched_patterns:
                for match_item in matched_patterns:
                    cat = match_item["category"]
                    formula = match_item.get("formula", "")
                    if cat and formula and not formula.startswith("[Subject] + [VP] + [Clause]") and "[" in formula:
                        raw_skeletons.append({
                            "sid": sid,
                            "quote": sent_clean,
                            "category": cat,
                            "pattern_formula": formula,
                        })
            else:
                cat = cls.classify_grammar_dependency(sent_clean)
                formula = cls.generate_cobuild_formula(sent_clean, category=cat) if cat else ""
                if cat and formula and not formula.startswith("[Subject] + [VP] + [Clause]") and "[" in formula:
                    raw_skeletons.append({
                        "sid": sid,
                        "quote": sent_clean,
                        "category": cat,
                        "pattern_formula": formula,
                    })

        if not raw_skeletons:
            return []

        # Score each skeleton using 3-band CEFR priority + Intra-Pattern Syntactic Complexity
        def _score_skeleton(item: Dict[str, Any]) -> Tuple[float, int]:
            formula_str = item["pattern_formula"]
            quote_str = item["quote"]
            cat_str = item["category"]
            prof = cls.match_cobuild_grammar_profile(formula_str, quote_str, category=cat_str, cefr_band=band)
            base_prio = float(prof.get("priority", 70))
            prof_key = prof.get("profile_key", "")

            words = quote_str.split()
            w_cnt = len(words)
            q_low = quote_str.lower()
            f_low = formula_str.lower()

            bonus = 0.0
            # Perfective / passive participle or coordinated non-finite clauses boost
            if "having + v-ed" in f_low or "being + v-ed" in f_low or re.search(r"\bhaving\s+[a-z]+ed\b", q_low):
                bonus += 10.0
            # Optimal pedagogical sentence length boost (14-38 words)
            if 14 <= w_cnt <= 38:
                bonus += 6.0
            elif w_cnt < 10:
                bonus -= 8.0
            elif w_cnt > 48:
                bonus -= 5.0

            # In B1_B2 and C1_C2, penalize bare restrictive relative clauses (that/who/which without pied-piped prepositions)
            # so non-finite participial adjuncts, clefts, inversions, and degree results win the macro-domain slots
            if band in ("B1_B2", "C1_C2") and prof_key == "relative_clause_restrictive":
                has_pied_piping = bool(re.search(r"\b(?:in|on|at|from|to|with|by|through|for|of|under|beyond|within|upon)\s+(?:which|whom)\b", q_low))
                if has_pied_piping:
                    bonus += 12.0
                else:
                    bonus -= 15.0

            sid_m = re.search(r"\d+", item.get("sid", "999"))
            sid_num = int(sid_m.group(0)) if sid_m else 999
            return (-round(base_prio + bonus, 2), sid_num)

        ranked_skeletons = sorted(raw_skeletons, key=_score_skeleton)

        # Diverse selection algorithm across macro domains, distinct formulas, and strictly unique quotes
        def _norm_structure(formula_str: str) -> str:
            res = re.sub(
                r"\[NP\]\s*(?:\+|,\s*)\s*(?:who|which|that|whom|whose)\s*\+\s*\[VP\]",
                "[NP] + rel_pron + [VP]",
                formula_str,
                flags=re.IGNORECASE
            )
            res = re.sub(
                r"^(?:if|unless|as long as|provided that)\s*\+\s*\[Clause\]",
                "[Condition] + [Clause]",
                res,
                flags=re.IGNORECASE
            )
            res = re.sub(
                r"^\[(?:Having \+ V-ed|Being \+ V-ed|V-ing|V-ed) Phrase\],\s*\[Subject\] \+ \[VP\]",
                "[Participial Phrase], [Subject] + [VP]",
                res,
                flags=re.IGNORECASE
            )
            return res

        selected: List[Dict[str, str]] = []
        seen_cats: Set[str] = set()
        seen_structures: Set[str] = set()
        seen_profiles: Set[str] = set()
        seen_quotes: Set[str] = set()

        def _get_prof_key(item: Dict[str, str]) -> str:
            prof = cls.match_cobuild_grammar_profile(
                item["pattern_formula"], item["quote"], category=item["category"], cefr_band=band
            )
            return str(prof.get("profile_key") or "")

        # Pass 1: maximize category diversity across the 4 macro domains (highest priority first)
        for item in ranked_skeletons:
            cat = item["category"]
            quote = item["quote"]
            struct = _norm_structure(item["pattern_formula"])
            pkey = _get_prof_key(item)
            if quote not in seen_quotes and cat not in seen_cats and struct not in seen_structures and pkey not in seen_profiles:
                selected.append(item)
                seen_cats.add(cat)
                seen_structures.add(struct)
                if pkey:
                    seen_profiles.add(pkey)
                seen_quotes.add(quote)
            if len(selected) >= target_count:
                break

        # Pass 2: fill remaining slots with distinct structural formulas, distinct profiles, and unique quotes
        if len(selected) < target_count:
            for item in ranked_skeletons:
                quote = item["quote"]
                struct = _norm_structure(item["pattern_formula"])
                pkey = _get_prof_key(item)
                if quote not in seen_quotes and struct not in seen_structures and pkey not in seen_profiles:
                    selected.append(item)
                    seen_structures.add(struct)
                    if pkey:
                        seen_profiles.add(pkey)
                    seen_quotes.add(quote)
                if len(selected) >= target_count:
                    break

        # Pass 3: if still under target_count, append remaining distinct quotes (preferring unused profiles first)
        if len(selected) < target_count:
            for item in ranked_skeletons:
                quote = item["quote"]
                pkey = _get_prof_key(item)
                if quote not in seen_quotes and pkey not in seen_profiles:
                    selected.append(item)
                    seen_quotes.add(quote)
                    if pkey:
                        seen_profiles.add(pkey)
                if len(selected) >= target_count:
                    break
        if len(selected) < target_count:
            for item in ranked_skeletons:
                quote = item["quote"]
                if quote not in seen_quotes:
                    selected.append(item)
                    seen_quotes.add(quote)
                if len(selected) >= target_count:
                    break

        # Restore natural passage appearance order (by sentence ID) for coherent reading flow
        def _sid_order(item: Dict[str, str]) -> int:
            m = re.search(r"\d+", item.get("sid", "999"))
            return int(m.group(0)) if m else 999

        selected.sort(key=_sid_order)
        return selected[:target_count]

    # -------------------------------------------------------------------------
    # 3b. Deterministic Multi-Word Expression & Collocation Mining (ACL + spaCy)
    # -------------------------------------------------------------------------
    @classmethod
    def mine_expression_skeletons(
        cls,
        text: str,
        target_count: int = 8,
        syllabus_expressions: Optional[List[str]] = None
    ) -> List[Dict[str, str]]:
        """
        Deterministically extracts high-value multi-word expressions, phrasal verbs,
        and Academic Collocation List (ACL) items from the indexed sentence pool.
        Standardizes slotted formulas (e.g. [sb], [sth], [one's]) at zero token cost.
        If syllabus_expressions is provided, prioritizes resolving and standardizing
        those curriculum targets first before filling remaining slots with computational candidates.

        Returns list of dicts:
            [{
                "sid": "S-1",
                "quote": "full sentence text",
                "phrase": "have an adverse effect on",
                "type": "collocation",
                "pattern_formula": "have an adverse effect on [sth]"
            }, ...]
        """
        _, pool = cls.tokenize_and_index_sentences(text)
        if not pool:
            return []

        nlp = cls.get_spacy()

        COMMON_PARTICLES = frozenset({"away", "back", "down", "in", "off", "on", "out", "over", "round", "through", "up"})
        DEPENDENT_PREPS = frozenset({"on", "upon", "to", "with", "for", "from", "into", "against", "about", "of", "in"})

        raw_candidates: List[Dict[str, Any]] = []

        for sid, sentence in pool.items():
            sent_clean = sentence.strip()
            if len(sent_clean.split()) < 5:
                continue

            doc = nlp(sent_clean)
            text_lower = sent_clean.lower()

            # 1. Dependency Syntax Phrasal Verbs & Prepositional Combinations via Declarative Expression Pattern Engine
            from librarian.expression_patterns import ExpressionPatternEngine
            expr_engine = ExpressionPatternEngine.get_engine(nlp)
            for match in expr_engine.match_sentence(doc):
                m_type = match["type"]
                if m_type == "phrasal verb":
                    m_words = match["phrase"].lower().split()
                    if len(m_words) == 2 and m_words[1] not in COMMON_PARTICLES and not cls._unit_blocks(match["phrase"].lower()):
                        m_type = "collocation"
                raw_candidates.append({
                    "sid": sid,
                    "quote": sent_clean,
                    "phrase": match["phrase"],
                    "type": m_type,
                    "pattern_formula": match["pattern_formula"],
                    "pobj": match.get("pobj"),
                    "pobj_proper": match.get("pobj_proper", False),
                    "score": match.get("priority", 8),
                })


        if not raw_candidates:
            return []

        # PedLex attestation gate: a candidate ships only when PedLex itself states the unit.
        # The old gate asked the Oxford Collocations file whether it had a headword for the
        # phrase, which is a weaker test than it sounds - 'keep of' is a preposition lifted out
        # of 'keep somebody out of something' and no dictionary has it as a unit.
        # ldoce_phrase_evidence asks the real question: is this a phrasal-verb block, a PHRASES
        # item, a grammar pattern built on the headword, or a COLLOCATIONS box item?  Evidence
        # that amounts to 'some example sentence contains these words in this order' does not
        # ship, because an example is exactly where a stray preposition looks most like a phrase.
        verified_candidates: List[Dict[str, Any]] = []

        for item in raw_candidates:
            phrase = item.get("phrase", "").lower().strip()
            item_type = item.get("type")
            words = phrase.split()
            if not words:
                continue

            # Phrasal verbs and prepositional verbs (e.g. 'rely on', 'long for', 'worry about')
            if item_type == "phrasal verb":
                attested = cls.is_attested_phrase(phrase)
                if not attested and "touch" in words and any(v in words for v in ("keep", "get", "stay", "lose")):
                    # PedLex states it as 'be/keep/stay etc in touch (with something)'
                    attested = cls.is_attested_phrase(" ".join(w for w in words if w != "with"))
                if attested:
                    # 'keep in' is a real phrasal verb, but PedLex's frame for it is 'keep
                    # somebody in', so a parse of 'keep in WeChat Moments' - or of 'keep in
                    # touch' - has read a locative 'in ...' as the particle and is not that
                    # frame.  The check runs for any object, not only a proper noun, because
                    # the frame says where the object sits regardless of what it is.
                    pobj_word = item.get("pobj")
                    if pobj_word and not cls.ldoce_phrase_object_fit(phrase, pobj_word):
                        continue
                    verified_candidates.append(item)
                continue

            # Anything else the dictionary states as a unit (e.g. 'account for', 'participate in')
            if cls.is_attested_phrase(phrase):
                verified_candidates.append(item)
                continue

            # Verbal idioms (e.g. 'take into consideration') that PedLex does not list as a
            # PHRASES item: the head noun's own grammar patterns must show the verb governing it.
            if item_type == "idiom":
                v_lemma, head_noun = words[0], words[-1]
                n_entry = cls.get_ldoce_entry(head_noun)
                if n_entry:
                    idiom_rx = re.compile(
                        r"(?<!\w)" + re.escape(v_lemma) + r"\w*(?:\s+\w+){0,4}\s"
                        + re.escape(head_noun) + r"(?!\w)", re.IGNORECASE)
                    if any(idiom_rx.search(cls._entry_text_normalizer(pat))
                           for pat in (n_entry.get("patterns") or [])):
                        verified_candidates.append(item)
                        continue

            # Collocations (ACL items, possessive frames like 'hear [one\'s] voice')
            elif item_type == "collocation":
                # ACL items or verified possessive collocations
                verified_candidates.append(item)
                continue

            # Default: if none of the specific validation branches match, drop unverified candidate

        raw_candidates = verified_candidates
        if not raw_candidates:
            return []

        selected: List[Dict[str, str]] = []
        seen_phrases: Set[str] = set()
        seen_quotes: Set[str] = set()

        # Phase 1: If syllabus_expressions is provided, prioritize and standardize those curriculum targets
        if syllabus_expressions:
            syllabus_candidates: List[Dict[str, Any]] = []
            for raw_expr in syllabus_expressions:
                raw_norm = re.sub(r"\s+", " ", raw_expr.strip())
                raw_norm = raw_norm.replace("\u2019", "'").replace("\u2018", "'").replace("…", "...")
                raw_norm = re.sub(r"'\s+s\b", "'s", raw_norm)
                # Strip trailing ', etc.' or 'etc.'
                raw_norm = re.sub(r",?\s*\betc\.?(?=\s+to\b|\s+for\b|\s+in\b|\s+on\b|\s+with\b|$)", "", raw_norm, flags=re.IGNORECASE)
                # Preserve parenthesized metasyntax slots: (sb) -> [sb], (sth) -> [sth], (oneself) -> [oneself]
                raw_norm = re.sub(r"(?:\(\s*|\[\s*)+(?:sb\.?|somebody|someone)(?:\s*\)|\s*\])+", "[sb]", raw_norm, flags=re.IGNORECASE)
                raw_norm = re.sub(r"(?:\(\s*|\[\s*)+(?:sth\.?|something)(?:\s*\)|\s*\])+", "[sth]", raw_norm, flags=re.IGNORECASE)
                raw_norm = re.sub(r"(?:\(\s*|\[\s*)+(?:sb/sth|sth/sb)(?:\s*\)|\s*\])+", "[sth/sb]", raw_norm, flags=re.IGNORECASE)
                raw_norm = re.sub(r"(?:\(\s*|\[\s*)+oneself(?:\s*\)|\s*\])+", "[oneself]", raw_norm, flags=re.IGNORECASE)
                # Strip remaining optional parentheticals (e.g. 'no exaggeration (to say (sth) / that (sth))', 'protect [sth] (from)')
                while re.search(r"\([^()]*\)", raw_norm):
                    raw_norm = re.sub(r"\([^()]*\)", "", raw_norm).strip()
                raw_norm = re.sub(r"\s+", " ", raw_norm).strip()
                # Normalize metasyntax placeholders:
                # 1. Possessive placeholders: one's / ones / sb.'s / sb's / somebody's / someone's -> [one's]
                raw_norm = re.sub(r"\b(?:one's|ones|sb\.?'s|somebody's|someone's)\b", "[one's]", raw_norm, flags=re.IGNORECASE)
                # 2. Reflexive placeholder: oneself -> [oneself]
                raw_norm = re.sub(r"\boneself\b", "[oneself]", raw_norm, flags=re.IGNORECASE)
                # 3. Object placeholders: sb. / sb / somebody / someone -> [sb]; sth. / sth / something / ... -> [sth]
                raw_norm = re.sub(r"\b(?:sb|somebody|someone)\b\.?", "[sb]", raw_norm, flags=re.IGNORECASE)
                raw_norm = re.sub(r"\b(?:sth|something)\b\.?", "[sth]", raw_norm, flags=re.IGNORECASE)
                raw_norm = re.sub(r"\.{2,}", "[sth]", raw_norm)
                raw_norm = re.sub(r"\[+\s*(one's|oneself|sth/sb|sb/sth|sth|sb)\s*\]+", r"[\1]", raw_norm, flags=re.IGNORECASE)
                raw_norm = re.sub(r"\s+", " ", raw_norm).strip()

                # Pre-expand multi-word slash alternatives (e.g. 'have a edge over / an edge on',
                # 'not have the heart to do [sth] / for [sth]', 'open the door to [sth] / to doing [sth]',
                # 'derive from / be derived from')
                raw_norm = re.sub(r"\s*/\s*to\s+doing\s+\[sth\]$", "", raw_norm, flags=re.IGNORECASE)
                multi_slash_seeds = [raw_norm]
                if " / " in raw_norm:
                    m_tail = re.match(r"^(.*?)\s+(to\s+do\s+\[sth\]|for\s+\[sth\]|an?\s+edge\s+(?:over|on))\s*/\s*(.+)$", raw_norm, flags=re.IGNORECASE)
                    if m_tail:
                        prefix_p, alt1, alt2 = m_tail.group(1).strip(), m_tail.group(2).strip(), m_tail.group(3).strip()
                        alt1 = re.sub(r"^a\s+edge\b", "an edge", alt1, flags=re.IGNORECASE)
                        alt2 = re.sub(r"^a\s+edge\b", "an edge", alt2, flags=re.IGNORECASE)
                        multi_slash_seeds = [f"{prefix_p} {alt1}".strip(), f"{prefix_p} {alt2}".strip()]
                    else:
                        slash_parts = [p.strip() for p in raw_norm.split(" / ") if p.strip()]
                        if len(slash_parts) >= 2 and all(len(p.split()) >= 2 for p in slash_parts):
                            multi_slash_seeds = slash_parts

                # Expand single-token slash alternatives (e.g. 'put [sth] into the atmosphere/air' ->
                # ['put [sth] into the atmosphere', 'put [sth] into the air'])
                branch_variants: List[str] = []
                for seed in multi_slash_seeds:
                    for bv in cls._slash_variants(seed):
                        if bv not in branch_variants:
                            branch_variants.append(bv)
                matched_branch_expr = None
                matched_sid = None
                matched_sent = None

                for branch_expr in branch_variants:
                    expr_clean = re.sub(r"\s+", " ", branch_expr.strip())
                    words = expr_clean.split()

                    # Filter proper nouns / TV trivia entities (e.g. 'lone ranger' with kind == 'entity')
                    ldoce_ent = cls.get_ldoce_entry(expr_clean)
                    if ldoce_ent and ldoce_ent.get("kind") == "entity":
                        continue

                    # Canonicalize finite forms of 'be' ('is/was/were/are/been') at the start of a syllabus expression to 'be',
                    # preserving 'be' intact per dictionary & syllabus standard ('be exhausted from', 'be obsessed with', 'be featured in').
                    if len(words) > 1 and words[0].lower() in ("is", "was", "were", "are", "been"):
                        words = ["be"] + words[1:]
                        expr_clean = " ".join(words)

                    # Normalize over-extended prepositional prefixes (e.g. 'toward the end of' -> 'the end of')
                    m_prep = re.match(r"^(?:toward|towards|near)\s+(the\s+end\s+of\b.*)", expr_clean, re.IGNORECASE)
                    if m_prep:
                        expr_clean = m_prep.group(1).strip()
                        words = expr_clean.split()

                    # Normalize over-extended determiners/quantifiers for category frames (e.g. 'any type of' -> 'type of')
                    m_det = re.match(r"^(?:any|each|every|some|a|an)\s+((?:type|kind|sort)\s+of\b.*)", expr_clean, re.IGNORECASE)
                    if m_det:
                        expr_clean = m_det.group(1).strip()
                        words = expr_clean.split()

                    # Build regex for placeholder-bearing or standard contiguous expressions
                    has_placeholder = bool(re.search(r"\[(?:one's|oneself|sth|sb|sth/sb)\]", expr_clean, re.IGNORECASE))
                    search_core = re.sub(r"(?:\s+\[(?:sth|sb|sth/sb)\])+$", "", expr_clean, flags=re.IGNORECASE).strip() or expr_clean
                    search_words = search_core.split()
                    # When a syllabus expression starts with copular/passive 'be' (e.g. 'be admitted to', 'be exhausted from'),
                    # the passage sentence may use a reduced participial clause without an explicit 'be' auxiliary ('admitted to the university').
                    search_words_no_be = search_words[1:] if (len(search_words) >= 3 and search_words[0].lower() == "be") else None
                    placeholder_rx = None
                    if has_placeholder:
                        def _build_placeholder_rx(s_words: List[str]):
                            rx_chunks = []
                            for tok in s_words:
                                tl = tok.lower()
                                if tl == "[one's]":
                                    rx_chunks.append(r"(?:my|your|his|her|its|our|their|one's|ones|[a-z]+'s)")
                                elif tl == "[oneself]":
                                    rx_chunks.append(r"(?:myself|yourself|himself|herself|itself|ourselves|yourselves|themselves|oneself)")
                                elif tl in ("[sth]", "[sb]", "[sth/sb]"):
                                    rx_chunks.append(r"(?:[a-z0-9'\-]+(?:\s+[a-z0-9'\-]+){0,5})")
                                elif tl == "not":
                                    rx_chunks.append(r"(?:not|n't|never|hardly|barely|scarcely)")
                                else:
                                    forms = sorted(cls.inflected_forms(tl) | {tl}, key=len, reverse=True)
                                    rx_chunks.append(r"(?:" + "|".join(re.escape(f) for f in forms if f) + r")")
                            return re.compile(r"\b" + r"\s+".join(rx_chunks) + r"\b", re.IGNORECASE)
                        placeholder_rx = _build_placeholder_rx(search_words)
                        placeholder_rx_no_be = _build_placeholder_rx(search_words_no_be) if search_words_no_be else None
                    else:
                        placeholder_rx_no_be = None

                    phrase_rx = None
                    phrase_rx_no_be = None
                    tokens = cls._phrase_tokens(search_core)
                    if len(tokens) >= 2:
                        rx_pat = cls._phrase_regex(tokens, 2, allow_modifier=False)
                        if rx_pat:
                            phrase_rx = re.compile(rx_pat, re.IGNORECASE)
                    if search_words_no_be and len(search_words_no_be) >= 2:
                        tokens_nb = cls._phrase_tokens(" ".join(search_words_no_be))
                        if len(tokens_nb) >= 2:
                            rx_pat_nb = cls._phrase_regex(tokens_nb, 2, allow_modifier=False)
                            if rx_pat_nb:
                                phrase_rx_no_be = re.compile(rx_pat_nb, re.IGNORECASE)

                    content_words = [w for w in search_words if not (w.startswith("[") and w.endswith("]")) and w.lower() not in ("not", "be")]

                    def _verify_verb_placeholder_match(sent_str: str) -> bool:
                        if not (has_placeholder and len(content_words) == 1):
                            return True
                        v_word = content_words[0].lower()
                        v_forms = cls.inflected_forms(v_word) | {v_word}
                        s_doc = nlp(sent_str)
                        for t in s_doc:
                            if t.text.lower() in v_forms or t.lemma_.lower() == v_word:
                                if t.pos_ in ("VERB", "AUX") and not any(c.dep_ in ("det", "poss") for c in t.children):
                                    return True
                        return False

                    # Two-pass sentence search: Pass A requires anchor_evidence_ok(s);
                    # Pass B falls back to all sentences in pool (recovering subheadings like "Hats Off to Volunteers" / "Take Corporate Action")
                    for allow_short_headings in (False, True):
                        for sid, s in pool.items():
                            if not allow_short_headings and not cls.anchor_evidence_ok(s):
                                continue
                            s_lower = s.lower()
                            e_lower = search_core.lower()
                            if placeholder_rx and (placeholder_rx.search(s) or (placeholder_rx_no_be and placeholder_rx_no_be.search(s))) and _verify_verb_placeholder_match(s):
                                matched_sid = sid
                                matched_sent = s
                                break
                            elif not has_placeholder and re.search(r"\b" + re.escape(e_lower.rstrip("?.!")) + r"\b", s_lower):
                                matched_sid = sid
                                matched_sent = s
                                break
                            elif phrase_rx and (phrase_rx.search(cls._entry_text_normalizer(s)) or (phrase_rx_no_be and phrase_rx_no_be.search(cls._entry_text_normalizer(s)))):
                                matched_sid = sid
                                matched_sent = s
                                break

                        if not matched_sid and content_words and not expr_clean.endswith("?"):
                            for sid, s in pool.items():
                                if not allow_short_headings and not cls.anchor_evidence_ok(s):
                                    continue
                                if not _verify_verb_placeholder_match(s):
                                    continue
                                s_lower = s.lower()
                                if all(re.search(r"\b" + re.escape(w.lower().rstrip("?.!")) + r"\b", s_lower) for w in content_words):
                                    matched_sid = sid
                                    matched_sent = s
                                    break
                                else:
                                    s_doc = nlp(s)
                                    s_lemmas = [t.lemma_.lower() for t in s_doc]
                                    s_tokens = [t.text.lower() for t in s_doc]
                                    if all(w.lower().rstrip("?.!") in s_lemmas or w.lower().rstrip("?.!") in s_tokens for w in content_words):
                                        matched_sid = sid
                                        matched_sent = s
                                        break

                        if matched_sid:
                            break

                    if matched_sid and matched_sent:
                        matched_branch_expr = expr_clean
                        break

                if not matched_sid or not matched_sent or not matched_branch_expr:
                    continue

                expr_clean = matched_branch_expr
                words = expr_clean.split()
                has_placeholder = bool(re.search(r"\[(?:one's|oneself|sth|sb|sth/sb)\]", expr_clean, re.IGNORECASE))
                e_lower = expr_clean.lower()
                cand_item = None

                # 1. Specialized High-Frequency Idioms (figurative, non-compositional units)
                KNOWN_IDIOMS = {
                    "hats off", "hat off", "pat on the back", 
                    "spill the beans", "break the ice", "piece of cake", 
                    "bite the bullet", "call it a day", "under the weather"
                }
                # 2. Specialized Closed-Paradigm Set Phrases (discourse connectors, fixed correlatives/framing units)
                KNOWN_SET_PHRASES = {
                    "as long as", "as well as", "as soon as", "so far as",
                    "in order to", "so as to", "in spite of", "due to", "owing to",
                    "on the other hand", "on the contrary", "in addition to",
                    "as a result", "for instance", "for example", "at least",
                    "at most", "at last", "upside down", "side by side",
                    "peace of mind", "pros and cons", "no exaggeration",
                    "not up to par", "up to par", "how was it?", "how was it"
                }

                if any(idiom in e_lower for idiom in KNOWN_IDIOMS):
                    formula = "hats off to [sb]" if e_lower in ("hat off", "hats off") and "hats off to" in matched_sent.lower() else expr_clean
                    cand_item = {
                        "sid": matched_sid,
                        "quote": matched_sent,
                        "phrase": expr_clean,
                        "type": "idiom",
                        "pattern_formula": formula,
                        "score": 100,
                    }
                elif any(sp in e_lower for sp in KNOWN_SET_PHRASES):
                    cand_item = {
                        "sid": matched_sid,
                        "quote": matched_sent,
                        "phrase": expr_clean,
                        "type": "set phrase",
                        "pattern_formula": expr_clean,
                        "score": 75,
                    }
                elif "touch" in e_lower and any(v in e_lower for v in ("keep", "stay", "get", "lose", "be")):
                    formula = f"{e_lower} [sb]" if "with" in e_lower else f"{e_lower}"
                    cand_item = {
                        "sid": matched_sid,
                        "quote": matched_sent,
                        "phrase": expr_clean,
                        "type": "idiom",
                        "pattern_formula": formula,
                        "score": 100,
                    }
                elif "make" in words and "possible" in words:
                    cand_item = {
                        "sid": matched_sid,
                        "quote": matched_sent,
                        "phrase": "make [sth] possible",
                        "type": "collocation",
                        "pattern_formula": "make [sth] possible",
                        "score": 85,
                    }
                elif has_placeholder:
                    # Syllabus item already declares an explicit slot formula ('do [one\'s] part', 'put [sth] into the atmosphere', 'make the most of [sth]')
                    lookup_phrase = re.sub(r"\s+", " ", re.sub(r"\[(?:one's|oneself|sth|sb|sth/sb|sb/sth)\]", "", expr_clean, flags=re.IGNORECASE)).strip() or expr_clean
                    cand_type = cls.classify_expression_type(lookup_phrase, matched_sent) or "collocation"
                    cand_item = {
                        "sid": matched_sid,
                        "quote": matched_sent,
                        "phrase": lookup_phrase,
                        "type": cand_type,
                        "pattern_formula": expr_clean,
                        "score": 85,
                    }
                else:
                    doc = nlp(matched_sent)
                    # Find the contiguous token span of expr_clean in matched_sent to avoid grabbing disjoint tokens
                    expr_tokens = []
                    m_span = re.search(r'\b' + re.escape(expr_clean) + r'\b', matched_sent, re.IGNORECASE)
                    if not m_span and len(words) > 1:
                        # Fallback: match words joined by flexible whitespace/punctuation
                        pat = r'\b' + r'\s+'.join(re.escape(w) for w in words) + r'\b'
                        m_span = re.search(pat, matched_sent, re.IGNORECASE)

                    if m_span:
                        sp_span = doc.char_span(m_span.start(), m_span.end())
                        if sp_span:
                            expr_tokens = list(sp_span)

                    if not expr_tokens:
                        # Find best ordered token sequence matching words by text or lemma
                        words_lower = [w.lower() for w in words]
                        best_tokens = None
                        min_dist = 9999
                        for start_t in doc:
                            if start_t.lemma_.lower() == words_lower[0] or start_t.text.lower() == words_lower[0]:
                                curr = [start_t]
                                for w in words_lower[1:]:
                                    cands = [t for t in doc if 0 < (t.i - curr[-1].i) <= 5 and (t.lemma_.lower() == w or t.text.lower() == w)]
                                    if not cands:
                                        break
                                    curr.append(cands[0])
                                if len(curr) == len(words_lower):
                                    dist = curr[-1].i - curr[0].i
                                    if dist < min_dist:
                                        min_dist = dist
                                        best_tokens = curr
                        if best_tokens:
                            expr_tokens = best_tokens
                        else:
                            dedup_tokens = []
                            for w_item in words_lower:
                                matched_t = next(
                                    (
                                        t for t in doc
                                        if (t.text.lower() == w_item or t.lemma_.lower() == w_item)
                                        and t not in dedup_tokens
                                        and (not dedup_tokens or 0 < (t.i - dedup_tokens[-1].i) <= 5)
                                    ),
                                    None
                                )
                                if matched_t is not None:
                                    dedup_tokens.append(matched_t)
                            expr_tokens = dedup_tokens

                    first_tok = expr_tokens[0] if expr_tokens else None
                    is_participial_amod = bool(first_tok and first_tok.dep_ == "amod" and first_tok.head.pos_ in ("NOUN", "PROPN"))
                    is_copular_be_frame = bool(len(words) >= 3 and words[0].lower() in ("be", "feel", "get"))

                    if is_copular_be_frame or (first_tok and first_tok.pos_ in ("VERB", "AUX") and not is_participial_amod):
                        v_lemma = words[0].lower() if is_copular_be_frame else first_tok.lemma_.lower()
                        last_tok = expr_tokens[-1] if expr_tokens else None
                        last_word_low = words[-1].lower()
                        
                        # Syntax-aware detection: check if 'to' is an infinitive marker rather than a preposition
                        is_infinitive_to = False
                        if last_tok is not None and last_tok.text.lower() == "to":
                            if last_tok.pos_ == "PART" or last_tok.dep_ == "aux":
                                is_infinitive_to = True
                            elif last_tok.dep_ != "prep" and last_tok.head.pos_ == "VERB" and last_tok.head.dep_ in ("xcomp", "advcl", "purpcl") and last_tok.i < last_tok.head.i:
                                is_infinitive_to = True

                        if is_infinitive_to and len(words) == 2:
                            # Genuine infinitive construction (e.g. tend to do [sth], fail to do [sth])
                            formula = f"{v_lemma} to do [sth]"
                            cand_type = "collocation"
                        elif (last_tok is not None and (last_tok.pos_ in ("ADP", "PART") or last_tok.dep_ in ("prep", "prt"))) or (is_copular_be_frame and last_word_low in cls._PHRASAL_TAILS):
                            prep_word = last_tok.text.lower() if last_tok is not None else last_word_low
                            # Dynamic Transitivity Audit via Sentence Dependency Tree & Valence
                            # Check whether the verb or particle takes an authentic nominal object in the context sentence
                            verb_tok = expr_tokens[0] if expr_tokens else None
                            has_contextual_obj = False
                            if last_tok is not None:
                                def _is_valid_obj(c_tok, anchor_tok) -> bool:
                                    if c_tok.pos_ not in ("NOUN", "PROPN", "PRON"):
                                        return False
                                    span_start = min(c_tok.i, anchor_tok.i)
                                    span_end = max(c_tok.i, anchor_tok.i)
                                    if any(t.text in (",", ";", ":", "\"", "'", "“", "”", "‘", "’") for t in doc[span_start : span_end + 1]):
                                        return False
                                    return True

                                # 1. Verb direct object (dobj / obj) — only for separable phrasal verb particles (prt)
                                verb_objs = [c for c in verb_tok.children if c.dep_ in ("dobj", "obj") and _is_valid_obj(c, last_tok)] if (verb_tok is not None and last_tok.dep_ == "prt") else []
                                # 2. Particle / prepositional object (pobj / obj)
                                prep_objs = [c for c in last_tok.children if c.dep_ in ("pobj", "obj", "dobj") and _is_valid_obj(c, last_tok)]
                                # 3. Passive voice extraction (e.g. 'the lights were turned off')
                                is_passive = any(c.dep_ in ("auxpass", "nsubjpass") for c in verb_tok.children) if verb_tok is not None else False
                                if verb_objs or prep_objs or is_passive:
                                    has_contextual_obj = True
                            elif is_copular_be_frame:
                                has_contextual_obj = True

                            # If the verb+particle has no object in context (e.g. 'show up', 'go on'),
                            # it is an authentic intransitive phrasal verb: generate clean formula without [sth/sb]
                            is_intransitive = not has_contextual_obj

                            has_internal_noun_or_adj = (
                                is_copular_be_frame
                                or (
                                    len(words) >= 3
                                    and any(t.pos_ in ("NOUN", "PROPN", "ADJ") for t in expr_tokens[1:-1])
                                )
                            )
                            if len(words) == 2:
                                pair_str = f"{v_lemma} {prep_word}"
                                v_ent = cls.get_ldoce_entry(v_lemma) or {}
                                v_verb_senses = [
                                    s for s in (v_ent.get("senses") or [])
                                    if cls._pos_code(s.get("pos")) == "v" or not s.get("pos")
                                ]
                                # A 2-word verb+prep is a prepositional phrasal verb when:
                                # - particle is in COMMON_PARTICLES, OR
                                # - it has a dedicated unit/phrasal_verb block, OR
                                # - prep_word is in DEPENDENT_PREPS and the primary verb sense itself governs prep_word
                                #   (as in 'long for', 'worry about', 'rely on'), unlike secondary/optional prepositional
                                #   collocations (such as 'train for', 'dock at').
                                primary_governs_prep = False
                                if prep_word in DEPENDENT_PREPS and v_verb_senses:
                                    s0 = v_verb_senses[0]
                                    s0_pats = list(s0.get("patterns") or []) + [
                                        pb.get("pattern") for pb in (s0.get("pattern_bundles") or [])
                                        if isinstance(pb, dict) and pb.get("pattern")
                                    ]
                                    if len(v_verb_senses) == 1 or any(
                                        prep_word in re.findall(r"[a-z]+", str(p).lower()) for p in s0_pats
                                    ):
                                        primary_governs_prep = True
                                is_true_pv = (
                                    prep_word in COMMON_PARTICLES
                                    or bool(cls._unit_blocks(pair_str))
                                    or bool(cls._phrasal_verb_blocks(pair_str, v_lemma))
                                    or primary_governs_prep
                                    or cls.classify_expression_type(pair_str, matched_sent) == "phrasal verb"
                                )
                                if v_lemma == "be" and prep_word == "upon":
                                    formula = "be upon [sb/sth]"
                                    cand_type = "set phrase"
                                elif is_intransitive:
                                    formula = f"{v_lemma} {prep_word}"
                                    cand_type = "phrasal verb" if is_true_pv else "collocation"
                                else:
                                    slot_tag = "[sth/sb]"
                                    if prep_word in ("at", "into", "in", "on", "under", "within", "throughout") and prep_objs:
                                        if all(
                                            po.ent_type_ != "PERSON"
                                            and po.lemma_.lower() not in ("person", "people", "man", "woman", "child", "friend", "someone", "somebody", "who", "whom")
                                            for po in prep_objs
                                        ):
                                            slot_tag = "[sth]"
                                    formula = f"{v_lemma} {prep_word} {slot_tag}"
                                    cand_type = "phrasal verb" if is_true_pv else "collocation"
                            else:
                                core = " ".join([v_lemma] + [w.lower() for w in words[1:]])
                                if "importance to" in core:
                                    formula = f"{core} [sth]"
                                elif any(core.startswith(w) for w in ("show thanks", "give thanks", "pay tribute")):
                                    formula = f"{core} [sb]"
                                elif is_intransitive:
                                    formula = core
                                else:
                                    formula = f"{core} [sth/sb]"
                                cand_type = "collocation" if has_internal_noun_or_adj else (cls.classify_expression_type(expr_clean, matched_sent) or "phrasal verb")
                        else:
                            # Verb + object / adj (e.g. make smart choices, keep silent, have a try, take action)
                            # Use expr_clean words with lemmatized head verb so intervening sentence modifiers
                            # (e.g. 'Corporate' in 'Take Corporate Action') are not injected into the canonical formula.
                            core_words = [v_lemma] + [w.lower() for w in words[1:]]
                            core = " ".join(core_words)
                            formula = core
                            cand_type = "collocation"

                        cand_item = {
                            "sid": matched_sid,
                            "quote": matched_sent,
                            "phrase": expr_clean,
                            "type": cand_type,
                            "pattern_formula": formula,
                            "score": 90 if cand_type == "phrasal verb" else 80,
                        }
                    elif words[0].lower() in ("in", "on", "at", "by", "for", "with", "from", "to", "under", "over", "toward", "towards"):
                        # Prepositional expressions (e.g. toward the end of [sth], in touch with [sb])
                        formula = expr_clean
                        if expr_clean.endswith(" of") or expr_clean.endswith(" for") or expr_clean.endswith(" with"):
                            formula = f"{expr_clean} [sth]"
                        elif expr_clean.endswith(" to"):
                            formula = f"{expr_clean} [sth/sb]"
                        cand_item = {
                            "sid": matched_sid,
                            "quote": matched_sent,
                            "phrase": expr_clean,
                            "type": "set phrase",
                            "pattern_formula": formula,
                            "score": 50,
                        }
                    else:
                        # Nominal/quantifier/partitive/idiomatic expressions (e.g. a piece of [sth], any type of [sth], show thanks to [sb], hats off to [sb])
                        formula = expr_clean
                        if expr_clean.endswith(" of") or expr_clean.endswith(" for") or expr_clean.endswith(" with"):
                            formula = f"{expr_clean} [sth]"
                        elif expr_clean.endswith(" to"):
                            # If expression relates to importance, attention, or priority, default to [sth]
                            if any(k in expr_clean.lower() for k in ("importance", "attention", "priority", "regard")):
                                formula = f"{expr_clean} [sth]"
                            elif any(expr_clean.lower().startswith(w) for w in ("show thanks", "give thanks", "hats off", "hat off", "pay tribute")):
                                formula = f"{expr_clean} [sb]"
                            else:
                                formula = f"{expr_clean} [sth/sb]"
                        elif expr_clean.lower() in ("hat off", "hats off"):
                            formula = "hats off to [sb]"

                        cand_type = "idiom" if expr_clean.lower() in ("hat off", "hats off", "pat on the back") else "collocation"
                        cand_item = {
                            "sid": matched_sid,
                            "quote": matched_sent,
                            "phrase": expr_clean,
                            "type": cand_type,
                            "pattern_formula": formula,
                            "score": 75 if cand_type == "idiom" else 60,
                        }

                if cand_item:
                    syllabus_candidates.append(cand_item)

            # Sort syllabus candidates by pedagogical score (idioms & phrasal verbs first)
            syllabus_candidates.sort(key=lambda x: x["score"], reverse=True)

            # Pass 1: maximize sentence quote diversity among syllabus expressions
            for item in syllabus_candidates:
                p_clean = item["phrase"].lower()
                if p_clean not in seen_phrases and item["quote"] not in seen_quotes:
                    selected.append({
                        "sid": item["sid"],
                        "quote": item["quote"],
                        "phrase": item["phrase"],
                        "type": item["type"],
                        "pattern_formula": item["pattern_formula"],
                    })
                    seen_phrases.add(p_clean)
                    seen_quotes.add(item["quote"])
                if len(selected) >= target_count:
                    break

            # Pass 2: if more syllabus slots needed, allow reusing quote for distinct syllabus phrase
            if len(selected) < target_count:
                for item in syllabus_candidates:
                    p_clean = item["phrase"].lower()
                    if p_clean not in seen_phrases:
                        selected.append({
                            "sid": item["sid"],
                            "quote": item["quote"],
                            "phrase": item["phrase"],
                            "type": item["type"],
                            "pattern_formula": item["pattern_formula"],
                        })
                        seen_phrases.add(p_clean)
                    if len(selected) >= target_count:
                        break

        # Phase 2: If no syllabus provided and still under target_count, fill from computational NLP/ACL/OCD candidates
        if not syllabus_expressions and len(selected) < target_count:
            # Sort raw_candidates by score descending
            raw_candidates.sort(key=lambda x: x["score"], reverse=True)

            # Pass 3: balance across types with sentence quote diversity
            for item in raw_candidates:
                p_clean = item["phrase"].lower()
                if p_clean not in seen_phrases and item["quote"] not in seen_quotes:
                    selected.append({
                        "sid": item["sid"],
                        "quote": item["quote"],
                        "phrase": item["phrase"],
                        "type": item["type"],
                        "pattern_formula": item["pattern_formula"],
                    })
                    seen_phrases.add(p_clean)
                    seen_quotes.add(item["quote"])
                if len(selected) >= target_count:
                    break

            # Pass 4: if still under target_count, allow reusing quote if phrase is distinct
            if len(selected) < target_count:
                for item in raw_candidates:
                    p_clean = item["phrase"].lower()
                    if p_clean not in seen_phrases:
                        selected.append({
                            "sid": item["sid"],
                            "quote": item["quote"],
                            "phrase": item["phrase"],
                            "type": item["type"],
                            "pattern_formula": item["pattern_formula"],
                        })
                        seen_phrases.add(p_clean)
                    if len(selected) >= target_count:
                        break

        return selected[:target_count]

    # -------------------------------------------------------------------------
    # 3c. Deterministic Academic Vocabulary Mining (AWL + Lemmatization)
    # -------------------------------------------------------------------------
    @classmethod
    def mine_vocabulary_skeletons(cls, text: str, target_count: int = 20) -> List[Dict[str, Any]]:
        """
        Deterministically mines genuine academic vocabulary targets from the indexed sentence pool.
        Uses spaCy context-aware lemmatization to extract base dictionary headwords,
        prioritizes the Academic Word List (AWL 560), and filters out low-value everyday words.

        Returns list of dicts:
            [{
                "sid": "S-1",
                "quote": "full sentence text",
                "word": "autonomy",
                "part_of_speech": "noun",
                "is_awl": True
            }, ...]
        """
        _, pool = cls.tokenize_and_index_sentences(text)
        if not pool:
            return []

        nlp = cls.get_spacy()

        # Closed whitelist of parts of speech accepted by schema
        VALID_POS_MAP = {
            "NOUN": "noun",
            "VERB": "verb",
            "ADJ": "adjective",
            "ADV": "adverb",
        }

        # Exclude trivial grammaticalized or ultra-common verbs/words
        TRIVIAL_WORDS = frozenset({
            "be", "have", "do", "say", "get", "make", "go", "know", "take", "see",
            "come", "think", "look", "want", "give", "use", "find", "tell", "ask",
            "work", "seem", "feel", "try", "leave", "call", "good", "new", "first",
            "last", "long", "great", "little", "own", "other", "old", "right", "big",
            "high", "different", "small", "large", "next", "early", "young", "important",
            "few", "public", "bad", "same", "able", "man", "woman", "person", "people",
            "child", "time", "year", "day", "way", "thing", "life", "hand", "part",
            "eye", "place", "case", "week", "company", "system", "program", "question",
            "government", "number", "night", "point", "home", "water", "room", "mother",
            "area", "money", "story", "fact", "month", "lot", "right", "study", "book",
            "word", "business", "issue", "side", "kind", "head", "house", "service",
            "friend", "father", "power", "hour", "game", "line", "end", "member", "law",
            "car", "city", "community", "name", "president", "team", "minute", "idea",
            "kid", "body", "information", "back", "parent", "face", "others", "level",
            "office", "door", "health", "person", "art", "war", "history", "party",
            "result", "change", "morning", "reason", "research", "girl", "guy", "moment",
            "air", "teacher", "force", "education"
        })

        # Dependent adverbial connectives: these heads are grammatically incomplete
        # without their obligatory prepositional complement. They are only valid as
        # the fixed 'head + of' phrase (POS = preposition), never as a bare adverb —
        # a bare 'regardless' is not a testable single-blank target and, being absent
        # from the Oxford/ACL collocation model, would yield a broken distractor space.
        # This is a small, closed, stable set of 'of'-licensing connective heads
        # (deliberately NOT an open-ended valency map): any connective whose bare
        # form is incomplete on its own belongs here.
        DEPENDENT_CONNECTIVES = {
            "regardless": "of",
            "irrespective": "of",
        }

        candidates: List[Dict[str, Any]] = []
        seen_lemmas: Set[str] = set()

        for sid, sentence in pool.items():
            sent_clean = sentence.strip()
            if len(sent_clean.split()) < 5:
                continue

            doc = nlp(cls._strip_lexical_quotes(sent_clean))
            # First pass: mine any attested hyphenated compounds in the sentence as intact units
            for m_hc in re.finditer(r"\b([a-zA-Z]+(?:-[a-zA-Z]+)+)\b", sent_clean):
                hc = m_hc.group(1).lower()
                if hc in seen_lemmas:
                    continue
                hc_entry = cls.get_ldoce_entry(hc)
                if hc_entry and hc_entry.get("senses"):
                    hc_pos_raw = (hc_entry.get("pos") or "adjective").lower()
                    hc_pos = "noun" if "noun" in hc_pos_raw else ("verb" if "verb" in hc_pos_raw else ("adverb" if "adverb" in hc_pos_raw else "adjective"))
                    candidates.append({
                        "sid": sid,
                        "quote": sent_clean,
                        "word": hc,
                        "part_of_speech": hc_pos,
                        "is_awl": False,
                        "score": 12 + min(len(hc), 12)
                    })
                    seen_lemmas.add(hc)

            WH_AND_PRONOUN_STOPS = frozenset({
                "what", "whatever", "who", "whoever", "whom", "whose", "which", "whichever",
                "where", "wherever", "when", "whenever", "why", "how", "however"
            })
            for token in doc:
                # Never extract a severed sub-token that is attached to a hyphen (e.g. 'waste' or 'prone' in 'waste-prone')
                if (token.i > 0 and doc[token.i - 1].text == "-" and not doc[token.i - 1].whitespace_) or (
                    token.i + 1 < len(doc) and doc[token.i + 1].text == "-" and not token.whitespace_
                ):
                    continue
                lemma = token.lemma_.lower().strip()
                if len(lemma) < 3 or lemma in seen_lemmas or not lemma.isalpha() or lemma in WH_AND_PRONOUN_STOPS:
                    continue

                # 1. Academic Closed-Paradigm Function Word Gate:
                # If lemma belongs to our curated high-utility closed paradigms (e.g. 'whereas',
                # 'nonetheless', 'notwithstanding', 'albeit'), bypass spaCy's stopword filter and POS restriction!
                if cls.is_function_word(lemma):
                    # Filter out elementary/common prepositions and conjunctions so passage mining focuses on academic lexis
                    ULTRA_BASIC_FUNC_WORDS = frozenset({
                        "in", "at", "on", "to", "for", "with", "from", "by", "of",
                        "and", "but", "or", "nor", "so", "for", "yet",
                        "if", "when", "as", "than", "that", "because",
                        "without", "within", "upon", "onto", "into", "until", "till",
                        "since", "while", "though", "although", "despite", "throughout",
                        "during", "whether", "before", "after", "under", "above", "below",
                        "between", "among", "around", "against", "towards", "toward",
                        "through", "across", "behind", "beyond", "besides", "except",
                        "unless", "once", "where", "wherever", "whenever"
                    })
                    if lemma in ULTRA_BASIC_FUNC_WORDS:
                        continue

                    f_idx = cls._get_function_word_index()
                    _fam, stype = f_idx[lemma][0]
                    # Map syntactic complement type to schema-supported POS
                    assigned_pos = "preposition" if stype == "prepositional" else ("conjunction" if stype == "clausal" else "adverb")
                    w_cefr = cls.get_word_cefr(lemma)
                    is_awl = w_cefr in ("B2", "C1", "C2")
                    score = (14 if is_awl else 6) + min(len(lemma), 8)

                    candidates.append({
                        "sid": sid,
                        "quote": sent_clean,
                        "word": lemma,
                        "part_of_speech": assigned_pos,
                        "is_awl": is_awl,
                        "is_connective": True,
                        "score": score
                    })
                    seen_lemmas.add(lemma)
                    continue

                # 2. Open-Class Content Words (Noun, Verb, Adjective, Adverb)
                if token.pos_ not in VALID_POS_MAP:
                    continue
                if token.is_stop:
                    continue
                # Reject mid-sentence capitalized proper nouns / acronym expansions (e.g., 'Placement' in 'AP (Advanced Placement)')
                if token.text and token.text[0].isupper() and token.i > 0 and doc[token.i - 1].text not in (".", "!", "?", ":", ";", '"', "\u201c", "'", "\u2018", "\u2014", "--"):
                    if token.pos_ == "PROPN" or doc[token.i - 1].text == "(" or (token.i > 0 and doc[token.i - 1].text[0].isupper()):
                        continue
                if not cls.is_pedagogical_target(lemma, ent_type=token.ent_type_):
                    continue

                w_cefr = cls.get_word_cefr(lemma)
                is_awl = w_cefr in ("B2", "C1", "C2")
                is_b1 = (w_cefr == "B1")

                # If not B1/B2/C1/C2 and is trivial or very short, skip
                if not is_awl and (lemma in TRIVIAL_WORDS or len(lemma) < 5):
                    continue

                # Dependent-connective gate (extraction-stage quality, Pillar 0).
                # A connective head like 'regardless' is only valid as its fixed
                # 'head + of' form. When the obligatory preposition is present in the
                # parsed sentence, complete the head to the fixed phrase and re-label
                # it as a preposition; otherwise drop it — a bare dependent connective
                # is never a valid single-blank target and must not leak out as an
                # adverb (which is the root cause of the 'regardless of' double-key).
                if token.pos_ == "ADV" and lemma in DEPENDENT_CONNECTIVES:
                    required_prep = DEPENDENT_CONNECTIVES[lemma]
                    has_prep = (
                        any(ch.dep_ == "prep" and ch.text.lower() == required_prep for ch in token.children)
                        or (token.i + 1 < len(doc) and doc[token.i + 1].text.lower() == required_prep)
                    )
                    if not has_prep:
                        seen_lemmas.add(lemma)  # gate: never leak a bare dependent connective
                        continue
                    phrase = f"{lemma} {required_prep}"
                    score = (18 if is_awl else 8) + min(len(phrase), 10)
                    candidates.append({
                        "sid": sid,
                        "quote": sent_clean,
                        "word": phrase,
                        "part_of_speech": "preposition",
                        "is_awl": is_awl,
                        "is_connective": True,
                        "score": score
                    })
                    seen_lemmas.add(lemma)
                    seen_lemmas.add(phrase)
                    continue

                # Scoring: B2/C1/C2 headwords get top priority, followed by B1 headwords, then word length
                cefr_bonus = 25 if is_awl else (15 if is_b1 else 0)
                score = cefr_bonus + min(len(lemma), 12)

                ctx_pos = cls.determine_contextual_pos(lemma, sent_clean) or VALID_POS_MAP[token.pos_]
                candidates.append({
                    "sid": sid,
                    "quote": sent_clean,
                    "word": lemma,
                    "part_of_speech": ctx_pos,
                    "is_awl": is_awl,
                    "score": score
                })
                seen_lemmas.add(lemma)

        if not candidates:
            return []

        # Sort descending by score
        candidates.sort(key=lambda x: x["score"], reverse=True)

        selected: List[Dict[str, Any]] = []
        selected_words: Set[str] = set()
        selected_sids: Set[str] = set()

        def _get_sid_index(sid_str: str) -> int:
            m = re.search(r'\d+', sid_str)
            return int(m.group(0)) if m else -999

        # Pass 1: Select diverse items across different sentences with word-family and sentence stride gating
        for item in candidates:
            w = item["word"]
            sid = item["sid"]
            sid_num = _get_sid_index(sid)

            # 1. Morphological Word Family Gate: strictly reject items from an already selected word family
            if any(cls.are_same_word_family(w, sw) for sw in selected_words):
                continue

            # 2. Hard Anti-Clustering Gate (Pass 1): prefer 1 target word per sentence
            if sid in selected_sids:
                continue

            # 3. Soft Anti-Clustering Stride Gate: avoid immediately adjacent sentences (|sid - selected_sid| <= 1)
            # if we have enough distinct sentences to reach target_count
            has_adjacent_clash = any(
                abs(sid_num - _get_sid_index(ssid)) <= 1
                for ssid in selected_sids
            )
            if has_adjacent_clash:
                distinct_available_sids = {c["sid"] for c in candidates if not any(cls.are_same_word_family(c["word"], sw) for sw in selected_words)}
                if len(selected) + len(distinct_available_sids - selected_sids) >= target_count:
                    continue

            selected.append(item)
            selected_words.add(w)
            selected_sids.add(sid)
            if len(selected) >= target_count:
                break

        # Pass 2: If target_count not reached, fill from remaining distinct sentences (still 1 per sentence)
        if len(selected) < target_count:
            for item in candidates:
                w = item["word"]
                sid = item["sid"]
                if w in selected_words or any(cls.are_same_word_family(w, sw) for sw in selected_words):
                    continue
                if sid in selected_sids:
                    continue
                selected.append(item)
                selected_words.add(w)
                selected_sids.add(sid)
                if len(selected) >= target_count:
                    break

        # Pass 3: If target_count still not reached (short passages with few sentences), fill remaining slots
        # while strictly maintaining the Morphological Word Family Gate
        if len(selected) < target_count:
            for item in candidates:
                w = item["word"]
                if w in selected_words or any(cls.are_same_word_family(w, sw) for sw in selected_words):
                    continue
                selected.append(item)
                selected_words.add(w)
                if len(selected) >= target_count:
                    break

        return selected[:target_count]

    @classmethod
    def build_top_k_vocab_candidates(
        cls,
        word: str,
        quote: str,
        pos: Optional[str] = None,
        k: int = 4,
        discourse_context: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Pure Symbolic Candidate Selection (NO Embedding Model).
        Retrieves top-k (k <= 4) candidate senses from PedLex using:
        1. POS gating
        2. Syntactic valency / frame / particle agreement
        3. Simplified Lesk overlap, 2nd-order collocate projection, and discourse window cue scoring
        4. PedLex natural frequency order fallback
        """
        entry = cls.get_pedlex_entry(word)
        raw_senses = entry.get("senses", []) if entry else []

        # Filter out morphological pointer senses ("the past participle of...", "the plural of...")
        # whenever substantive semantic definitions exist in the entry
        _morph_ptr_re = re.compile(
            r"^(?:the\s+)?(?:past\s+participle|past\s+tense|present\s+participle|plural|comparative|superlative)\s+of\b",
            re.IGNORECASE
        )
        non_ptr_senses = [
            s for s in raw_senses
            if not _morph_ptr_re.match((s.get("definition") or "").strip())
        ]
        senses = non_ptr_senses if non_ptr_senses else raw_senses

        target_pos = pos or (entry.get("pos") if entry else None)
        target_p = cls._pos_code(target_pos)
        q_low = (quote or "").lower()
        has_pos_match = bool(target_p and any(cls._sense_matches_pos(s, target_p) for s in senses))

        if not senses:
            candidates = []
        elif len(senses) == 1 and (not target_p or not senses[0].get("pos") or cls._sense_matches_pos(senses[0], target_p)):
            s0 = senses[0]
            candidates = [{
                "definition": cls.format_ldoce_definition(s0) or s0.get("definition", ""),
                "pos": pos or s0.get("pos", "noun"),
                "sense_idx": 0,
                "signpost": s0.get("signpost", ""),
                "patterns": s0.get("patterns", []),
                "examples": s0.get("examples", []),
                "raw_sense": s0,
                "score": 10.0,
                "margin": 0.0,
            }]
        else:
            entry_filtered = dict(entry, senses=senses) if senses is not raw_senses else entry
            res = cls._lock_sense(
                entry_filtered,
                quote=quote,
                target_pos=target_pos,
                return_scores=True,
                discourse_context=discourse_context
            )

            def _is_unattested_phrase_unit_sense(s_obj: Dict[str, Any]) -> bool:
                u_list = s_obj.get("units") or []
                if not u_list:
                    return False
                w_low = word.strip().lower()
                s_defn_low = str(s_obj.get("definition") or "").strip().lower()
                # If a verb sense explicitly declares [transitive], its prepositional unit (e.g. 'tend to sb/sth')
                # is an optional prepositional variant, not an obligatory multi-word idiom
                s_gram_low = str(s_obj.get("grammar") or "").lower()
                if target_p == "v" and re.search(r"(?<!in)transitive", s_gram_low):
                    return False
                # Check if any unit phrase is attested or shares its grammatical preposition/frame with the quote
                all_multi_unattested = True
                def _unit_tok_matches(q_text: str, u_tok: str) -> bool:
                    if cls.text_contains_form(q_text, u_tok):
                        return True
                    if u_tok == "be":
                        return bool(re.search(r"\b(?:be|am|is|are|was|were|been|being|'m|'s|'re)\b", q_text))
                    if u_tok == "have":
                        return bool(re.search(r"\b(?:have|has|had|having|'ve|'d)\b", q_text))
                    if u_tok == "get":
                        return bool(re.search(r"\b(?:get|gets|got|gotten|getting)\b", q_text))
                    if u_tok == "do":
                        return bool(re.search(r"\b(?:do|does|did|done|doing)\b", q_text))
                    return False

                for u_it in u_list:
                    u_phr = (u_it.get("phrase") if isinstance(u_it, dict) else str(u_it or "")).strip()
                    u_clean = re.sub(r'\[.*?\]|\(.*?\)', '', u_phr).strip().lower()
                    if " " not in w_low and u_clean.startswith(("not ", "no ", "never ")) and s_defn_low.startswith(("no ", "not ", "never ")):
                        continue
                    u_clean_no_slot = re.sub(r"\b(?:somebody's|someone's|something's|somebody|someone|something|sb's|sb|sth|one's|oneself|etc)\b\.?", "", u_clean).strip()
                    u_toks = [t for t in re.findall(r"[a-z]+", u_clean_no_slot) if t not in ("a", "an", "the", "some", "any", "doing")]
                    # When extracting a single-word headword (e.g. 'likelihood'), do NOT allow a prepositional/adverbial idiom unit
                    # starting with another word (e.g. 'in all likelihood') to define the bare noun!
                    if " " not in w_low and len(u_toks) >= 2 and not cls.form_matches(u_toks[0], w_low):
                        continue
                    if cls.match_typed_slot_pattern(u_phr, quote, headword=w_low):
                        all_multi_unattested = False
                        break
                    if len(u_toks) <= 1:
                        all_multi_unattested = False
                        break
                    if " " not in w_low and u_toks[0] in ("not", "no", "never") and s_defn_low.startswith(("no ", "not ", "never ")):
                        continue
                    if all(_unit_tok_matches(q_low, t) for t in u_toks):
                        all_multi_unattested = False
                        break
                    # Preserve unit sense if it is a <headword> + <preposition> frame (e.g. 'an embarrassment of riches')
                    # and the quote contains <headword> + <preposition> (e.g. 'an embarrassment of possibilities')
                    if len(u_toks) >= 2 and u_toks[0] == w_low and u_toks[1] in ("of", "to", "for", "with", "in", "on", "from", "about"):
                        if f"{w_low} {u_toks[1]}" in q_low:
                            all_multi_unattested = False
                            break
                    # Preserve unit sense if it is a 2-word nominal compound <headword> + <noun> (e.g. 'comic book')
                    # whose definition defines the standalone headword rather than an opaque idiom
                    if len(u_toks) == 2 and u_toks[0] == w_low and target_p == "n":
                        all_multi_unattested = False
                        break
                if not all_multi_unattested:
                    return False
                return True

            if not res or len(res) < 5:
                matched = [
                    s for s in senses
                    if (not has_pos_match or cls._sense_matches_pos(s, target_p))
                    and not _is_unattested_phrase_unit_sense(s)
                ] or [s for s in senses if not has_pos_match or cls._sense_matches_pos(s, target_p)] or senses
                candidates = [{
                    "definition": cls.format_ldoce_definition(s) or s.get("definition", ""),
                    "pos": (pos if has_pos_match else (s.get("pos") or pos)) or s.get("pos", target_pos),
                    "sense_idx": i,
                    "signpost": s.get("signpost", ""),
                    "patterns": list(dict.fromkeys(list(s.get("patterns") or []) + list(s.get("units") or []))),
                    "examples": s.get("examples", []),
                    "raw_sense": s,
                } for i, s in enumerate(matched[:k])]
            else:
                idx, score, margin, conf, score_trace = res
                # Enforce POS hard gate and filter unattested multi-word idiom senses when standalone senses exist
                pos_filtered_trace = [
                    (s_i, sc) for s_i, sc in score_trace
                    if (not has_pos_match or cls._sense_matches_pos(senses[s_i], target_p))
                ] or score_trace
                non_idiom_trace = [
                    (s_i, sc) for s_i, sc in pos_filtered_trace
                    if not _is_unattested_phrase_unit_sense(senses[s_i])
                ] or pos_filtered_trace

                # Symbolic Hard Pruning (Grammar / Number / Valency / Register / Domain / Phrasal Particle):
                # Eliminate senses that violate hard syntactic/morphological invariants in the quote
                # whenever at least one grammatically compatible sense survives.
                q_seq_prune = [w.lower() for w in re.findall(r"\b[a-zA-Z']+\b", quote or "")]
                target_w_clean_p = (entry.get("word") or word or "").strip().lower()
                headword_tokens_p = set(re.findall(r"\b[a-z]+\b", target_w_clean_p))

                pv_sense_map_p: Dict[int, List[str]] = {}
                for pv in entry.get("phrasal_verbs") or []:
                    phr = pv.get("phrase", "")
                    p_words = [w for w in phr.lower().split() if w != target_w_clean_p]
                    if p_words:
                        for pvs in pv.get("senses") or []:
                            p_defn = pvs.get("definition")
                            for s_idx_p, s_cand_p in enumerate(senses):
                                if s_cand_p.get("definition") == p_defn:
                                    pv_sense_map_p[s_idx_p] = p_words

                q_morph_num: List[str] = []
                q_has_count_det = False
                q_is_pred = False
                q_is_attr = False
                q_has_dobj = False
                q_is_passive = False
                q_has_adv_or_prep_comp = False
                if quote and len(non_idiom_trace) > 1:
                    try:
                        nlp_p = cls.get_spacy()
                        doc_p = nlp_p(cls._strip_lexical_quotes(quote))
                        m_toks = [
                            t for t in doc_p
                            if t.lemma_.lower() == target_w_clean_p or t.text.lower() == target_w_clean_p
                            or cls.form_matches(t.text.lower(), target_w_clean_p)
                            or cls.form_matches(t.lemma_.lower(), target_w_clean_p)
                            or (headword_tokens_p and (t.lemma_.lower() in headword_tokens_p or t.text.lower() in headword_tokens_p))
                        ]
                        if m_toks:
                            mt = m_toks[0]
                            if mt.pos_ == "NOUN":
                                q_morph_num = list(mt.morph.get("Number") or [])
                                q_has_count_det = any(
                                    ch.dep_ in ("det", "nummod")
                                    and ch.lemma_.lower() in ("a", "an", "one", "two", "three", "many", "few", "several", "each", "every", "another", "both")
                                    for ch in mt.children
                                )
                            if mt.dep_ in ("acomp", "oprd") or (
                                mt.head != mt
                                and mt.head.lemma_.lower() in ("be", "become", "seem", "feel", "get", "remain", "stay", "grow", "turn", "appear", "look", "sound")
                                and mt.dep_ != "amod"
                            ):
                                q_is_pred = True
                            elif mt.dep_ == "amod" and mt.head.pos_ in ("NOUN", "PROPN") and mt.i < mt.head.i:
                                q_is_attr = True
                            if mt.pos_ == "VERB" or target_p == "v":
                                q_is_passive = any(c.dep_ in ("auxpass", "aux:pass", "nsubjpass", "nsubj:pass") for c in mt.children)
                                q_has_dobj = any(c.dep_ in ("dobj", "obj") for c in mt.children) or q_is_passive
                                q_has_adv_or_prep_comp = any(
                                    c.dep_ in ("prep", "prt", "npadvmod")
                                    or (c.dep_ == "advmod" and c.i > mt.i)
                                    for c in mt.children
                                )
                    except Exception:
                        pass

                def _is_grammatically_hard_pruned_sense(s_idx_c: int, sc_c: float) -> bool:
                    if sc_c < 0:
                        return True
                    s_c = senses[s_idx_c]
                    s_gram_c = (s_c.get("grammar") or "").lower()
                    s_reg_c = (s_c.get("register") or "").lower()
                    s_sp_c = (s_c.get("signpost") or "").lower()
                    s_defn_c = (s_c.get("definition") or "").lower()

                    # 1. Missing mandatory phrasal verb particle
                    if s_idx_c in pv_sense_map_p:
                        req_ps = pv_sense_map_p[s_idx_c]
                        if not all(any(cls.form_matches(tok_q, rp) for tok_q in q_seq_prune) for rp in req_ps):
                            return True

                    # 2. Noun countability & number hard clash
                    if ("Plur" in q_morph_num or q_has_count_det) and "[uncountable]" in s_gram_c and "[countable]" not in s_gram_c:
                        return True
                    if "Plur" in q_morph_num and "[singular]" in s_gram_c and "[countable]" not in s_gram_c and "[plural]" not in s_gram_c:
                        return True
                    if ("Sing" in q_morph_num or q_has_count_det) and "[plural]" in s_gram_c and "[singular]" not in s_gram_c and "[countable]" not in s_gram_c:
                        return True

                    # 3. Adjective syntactic slot hard clash
                    if q_is_pred and "[only before noun]" in s_gram_c:
                        return True
                    if q_is_attr and ("[not before noun]" in s_gram_c or "[never before noun]" in s_gram_c):
                        return True

                    # 4. Verb transitivity & mandatory locative/prepositional complement clash
                    if q_has_dobj and "[intransitive]" in s_gram_c and not re.search(r"(?<!in)transitive", s_gram_c):
                        return True
                    if "always + adverb/preposition" in s_gram_c and not q_has_adv_or_prep_comp:
                        return True

                    # 5. Spoken register gate in non-dialogue narrative
                    if "spoken" in s_reg_c or "[spoken]" in s_gram_c:
                        is_dialogue = any(qc in (quote or "") for qc in ('"', "'", "“", "”", "‘", "’"))
                        if not is_dialogue:
                            return True

                    # 6. Specialized domain gate (medical / law / technical) without domain triggers
                    is_med = s_reg_c == "medical" or "[medical]" in s_gram_c or s_sp_c in ("illness", "medical", "disease")
                    if is_med and not any(w in q_low for w in (
                        "illness", "disease", "symptom", "patient", "doctor", "hospital", "medicine",
                        "syndrome", "fever", "pain", "clinic", "clinical", "infection", "treatment",
                        "suffer", "suffered", "suffering", "mild", "severe", "fatal", "heart", "brain",
                        "blood", "attack", "paralyz", "recover", "health"
                    )):
                        return True
                    is_law = s_reg_c in ("law", "legal") or "[law]" in s_gram_c or "[legal]" in s_gram_c or s_sp_c in ("law", "legal", "court")
                    if is_law and not any(w in q_low for w in (
                        "court", "judge", "jury", "trial", "lawyer", "attorney", "lawsuit", "legal", "statute", "prosecution", "defendant", "verdict"
                    )):
                        return True
                    is_tech = (
                        s_reg_c in ("chemistry", "physics", "biology", "computing")
                        or (s_reg_c == "technical" and s_idx_c > 0)
                        or any(tag in s_gram_c for tag in ("[chemistry]", "[physics]", "[computing]"))
                        or ("[technical]" in s_gram_c and s_idx_c > 0)
                        or s_sp_c in ("chemistry", "physics", "technical", "chemical")
                    )
                    if is_tech and not any(w in q_low for w in (
                        "chemical", "acid", "substance", "atom", "molecule", "reaction", "physics",
                        "laboratory", "voltage", "circuit", "alkali", "gas", "metal", "compound",
                        "technology", "technological", "scientific", "science", "engineering"
                    )):
                        return True

                    # 6b. Weather/elements domain restriction in WordNet/PedLex glosses ('(of the elements) ...')
                    if s_defn_c.strip().startswith(("(of the elements)", "(of weather)", "(of wind")) and not any(
                        w in q_low for w in ("storm", "wind", "rain", "wave", "sea", "weather", "element", "sky", "tempest", "gale")
                    ):
                        return True

                    # 7. Cross-reference inflection stub
                    if re.match(r"^(?:the\s+)?(?:past\s+participle|past\s+tense|present\s+participle|plural|comparative|superlative)\s+of\b", s_defn_c.strip()):
                        return True

                    return False

                pruned_trace = [
                    (s_i, sc) for s_i, sc in non_idiom_trace
                    if not _is_grammatically_hard_pruned_sense(s_i, sc)
                ] or non_idiom_trace

                sorted_trace = sorted(pruned_trace, key=lambda x: -x[1])
                valid_idx_set = {s_i for s_i, _ in pruned_trace}
                top_indices = [s_i for s_i, _ in sorted_trace[:k]]
                if idx is not None and idx in valid_idx_set:
                    top_indices = ([idx] + [s_i for s_i in top_indices if s_i != idx])[:k]
                elif not top_indices:
                    top_indices = [0]

                candidates = []
                for s_i in top_indices:
                    s = senses[s_i]
                    s_exs = list(s.get("examples") or [])
                    if quote and s.get("pattern_bundles"):
                        for pb in (s.get("pattern_bundles") or []):
                            if isinstance(pb, dict) and pb.get("pattern") and pb.get("example"):
                                pb_pat = str(pb["pattern"]).strip()
                                pb_ex = str(pb["example"]).strip()
                                if pb_ex and cls.match_typed_slot_pattern(pb_pat, quote, headword=(entry.get("word") or word)):
                                    s_exs = [pb_ex] + [e for e in s_exs if e != pb_ex]
                                    break
                    candidates.append({
                        "definition": cls.format_ldoce_definition(s) or s.get("definition", ""),
                        "pos": (pos if has_pos_match else (s.get("pos") or pos)) or s.get("pos", target_pos),
                        "sense_idx": s_i,
                        "signpost": s.get("signpost", ""),
                        "patterns": list(dict.fromkeys(list(s.get("patterns") or []) + list(s.get("units") or []))),
                        "examples": s_exs,
                        "raw_sense": s,
                        "score": next((sc for s_idx, sc in score_trace if s_idx == s_i), 0.0),
                        "margin": float(margin or 0.0),
                    })

        # Cross-Lexicon Pool Fusion (WordNet oewn:2025+):
        # 1. Trigger when PedLex has NO senses matching the target POS.
        # 2. Also trigger when PedLex has only 1 sense for the target POS AND that sole sense is pragmatically
        #    restricted ([disapproving], [spoken], [informal], [humorous], [old-fashioned]) or a narrow physical
        #    conditional subject ('if a piece of ...'), so neutral/academic senses attested in WordNet with
        #    authentic examples can enter the pool for Stage 2 1-token LLM arbitration.
        sole_pedlex_restricted = False
        if len(candidates) == 1 and has_pos_match:
            c0_def = str(candidates[0].get("definition") or "").lower()
            c0_raw = candidates[0].get("raw_sense") if isinstance(candidates[0].get("raw_sense"), dict) else {}
            c0_reg = str(c0_raw.get("register") or "").lower()
            c0_gram = str(c0_raw.get("grammar") or "").lower()
            if (
                any(tag in c0_def or tag in c0_reg or tag in c0_gram for tag in ("disapproving", "spoken", "informal", "humorous", "old-fashioned"))
                or c0_def.startswith(("if a piece of ", "if a chemical substance "))
            ):
                sole_pedlex_restricted = True

        needs_wn_fallback = (
            not candidates
            or not has_pos_match
            or sole_pedlex_restricted
        )
        if needs_wn_fallback:
            try:
                wn = cls.get_wordnet()
                if wn:
                    wn_pos_map = {"n": ("n",), "v": ("v",), "a": ("a", "s"), "adj": ("a", "s"), "r": ("r",), "adv": ("r",)}
                    wn_poses = wn_pos_map.get(target_p) if target_p else None
                    syns = [s for p_code in wn_poses for s in (wn.synsets(word, pos=p_code) or [])] if wn_poses else wn.synsets(word)
                    lem_w = cls._surface_lemma(word)
                    if not syns and lem_w and lem_w != word:
                        syns = [s for p_code in wn_poses for s in (wn.synsets(lem_w, pos=p_code) or [])] if wn_poses else wn.synsets(lem_w)
                    if syns:
                        seen_norm = {re.sub(r"[^a-z0-9]+", " ", (c.get("definition") or "").lower()).strip() for c in candidates}
                        q_words = set(re.findall(r"[a-z]{3,}", q_low))
                        scored_syns = []
                        for syn in syns:
                            s_def = (syn.definition() or "").strip()
                            if not s_def:
                                continue
                            s_norm = re.sub(r"[^a-z0-9]+", " ", s_def.lower()).strip()
                            if s_norm in seen_norm:
                                continue
                            s_exs = [str(e).strip().strip('"“”') for e in (syn.examples() or []) if e]
                            # Only include WordNet synsets that either have examples or when PedLex has 0 POS-matched candidates
                            if not s_exs and candidates and has_pos_match:
                                continue
                            seen_norm.add(s_norm)
                            s_bag = set(re.findall(r"[a-z]{3,}", (s_def + " " + " ".join(s_exs)).lower()))
                            overlap = len(q_words & s_bag)
                            if f"{word.lower()} of " in q_low and any(kw in s_def.lower() for kw in ("quantity", "excess", "number", "amount", "multitude", "group", "competitors")):
                                overlap += 3
                            scored_syns.append((overlap, syn, s_def, s_exs, s_norm))
                        scored_syns.sort(key=lambda x: -x[0])
                        if not has_pos_match and scored_syns:
                            candidates = []
                        wn_slots = min(2, len(scored_syns)) if (candidates and has_pos_match) else min(k, len(scored_syns))
                        if candidates and has_pos_match and len(candidates) + wn_slots > k:
                            candidates = candidates[:max(1, k - wn_slots)]
                        for _ov, _syn, s_def, s_exs, s_norm in scored_syns[:wn_slots]:
                            if len(candidates) >= k:
                                break
                            if target_p == "v" and not s_def.lower().startswith("to "):
                                s_def = f"to {s_def}"
                            candidates.append({
                                "definition": s_def,
                                "pos": target_pos or pos or "noun",
                                "sense_idx": -1,
                                "signpost": "",
                                "patterns": [],
                                "examples": s_exs,
                                "raw_sense": {"definition": s_def, "pos": target_pos or pos or "noun", "examples": s_exs},
                                "score": 0.0,
                                "margin": 0.0,
                            })
            except Exception:
                pass

        return candidates[:k]

    @classmethod
    def build_top_k_expression_candidates(
        cls,
        expr_word: str,
        quote: str,
        k: int = 4,
        expr_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Pure Symbolic Candidate Selection for Multi-Word Expressions (NO Embedding Model):
        1. Phrasal verb blocks & direct multi-word PedLex headword entries
        2. PedLex phrase index candidates (exact, prefix, or slash-expanded unit_key match)
        3. Strictly attested phrase units on constituent headwords (and WordNet/constituent fallbacks ONLY if unattested)
        """
        clean = re.sub(r'\[.*?\]|\(.*?\)', '', expr_word).strip().lower()
        clean = re.sub(r"\b(?:sb's|sb|sth|one's|oneself)\b", "", clean).strip()
        clean = re.sub(r"\s+", " ", clean).strip()
        tokens = clean.split()
        lem_0 = cls.lemmatize_headword(tokens[0]) if tokens else ""
        if tokens and lem_0 == tokens[0] and tokens[0].endswith("ed") and len(tokens[0]) > 4:
            stem_ed = tokens[0][:-2]
            if cls.get_pedlex_entry(stem_ed):
                lem_0 = stem_ed
        lem_tokens = ([lem_0] + tokens[1:]) if tokens else []
        candidates = []
        seen_defs = set()
        q_low = (quote or "").lower()
        locked_expr_type = (
            expr_type
            if (expr_type in cls.EXPRESSION_TYPE_LABELS)
            else (cls.classify_expression_type(expr_word, quote) or "collocation")
        )

        def _add_cand(df_str: str, pos_str: str, src_str: str, ex_list: List[str], raw_s: Optional[Dict[str, Any]], is_attested: bool = True):
            if not df_str:
                return
            norm_d = re.sub(r"[^a-z0-9]+", " ", df_str.lower()).strip()
            if not norm_d or norm_d in seen_defs:
                return
            # Guard against 'be there (for somebody)' when 'for' is absent from both formula and sentence
            if clean == "be there" and " for " not in f" {q_low} ":
                has_for_sb = "for somebody" in df_str.lower() or "help them" in df_str.lower() or any(
                    "for somebody" in str(u.get("phrase") if isinstance(u, dict) else u).lower()
                    for u in ((raw_s.get("units") if isinstance(raw_s, dict) else None) or [])
                )
                if has_for_sb:
                    return
            # Guard against [spoken] sympathy formula for 'think of' ('I'm thinking of you') when used in narrative recall
            if clean in ("think of", "think about") and "used to say that you care about and feel sympathy" in df_str.lower():
                if not any(p in q_low for p in ("thinking of you", "think of you", "thoughts are with")):
                    return
            # Guard against stative 'to seem similar to someone or something else' for 'remind of' when used in passive voice ('be reminded of')
            if "remind" in clean and "to seem similar to" in df_str.lower():
                if re.search(r"\b(?:am|is|are|was|were|be|been|being|get|gets|got)\s+(?:often\s+|always\s+|suddenly\s+|immediately\s+|constantly\s+)?reminded\b", q_low):
                    return
            # Guard against narrow [technical] chemical reaction sense ('if a chemical substance reacts') in non-chemical contexts
            if "if a chemical substance reacts" in df_str.lower():
                _chem_words = {"chemical", "substance", "acid", "metal", "gas", "oxygen", "water", "element", "compound", "solution", "calcium", "sulphur", "hydrogen", "carbon", "alkali", "molecule", "atom", "chlorine", "iron"}
                if not (set(re.findall(r"[a-z]{3,}", q_low)) & _chem_words):
                    return
            # Guard against literal vehicle backing-up sense ('if a vehicle or its driver reverses') or reciprocal plural-order sense ('the usual things two people do') when 'reverse' takes a singular reflexive pronoun ('reverse oneself/myself')
            if "if a vehicle or its driver reverses" in df_str.lower() or "the usual things two people do" in df_str.lower():
                if "oneself" in expr_word.lower() or re.search(r"\breverse[ds]?\s+(?:myself|yourself|himself|herself|itself|oneself)\b", q_low):
                    return
            seen_defs.add(norm_d)
            cleaned_exs = [str(e).strip().strip('"“”') for e in (ex_list or []) if e and str(e).strip()]
            candidates.append({
                'definition': df_str,
                'pos': locked_expr_type,
                'source': src_str,
                'examples': cleaned_exs,
                'raw_sense': raw_s,
                'attested': is_attested,
            })

        # 1. Phrasal verb blocks & direct multi-word PedLex headword entries (e.g. 'stuffed animal', 'comic book')
        u_key = cls._unit_key(clean)
        lem_u_key = cls._unit_key(" ".join(lem_tokens)) if lem_tokens else u_key
        unit_keys = []
        for uk_cand in (u_key, lem_u_key, clean, " ".join(lem_tokens)):
            if uk_cand and uk_cand not in unit_keys:
                unit_keys.append(uk_cand)
            uk_parts = uk_cand.split() if uk_cand else []
            if (
                len(uk_parts) == 3
                and uk_parts[1] in {"away", "back", "down", "in", "off", "on", "out", "over", "round", "through", "up", "around", "aside", "apart", "ahead", "forth", "together"}
                and uk_parts[2] in {"to", "into", "onto", "from", "for", "with", "of", "on", "in", "at", "by"}
            ):
                vp2 = f"{uk_parts[0]} {uk_parts[1]}"
                if vp2 not in unit_keys:
                    unit_keys.append(vp2)

        def _unit_key_matches(cand_raw_or_uk: str, uk_target: str, allow_prefix: bool = False) -> bool:
            cand_uk = cls._unit_key(cand_raw_or_uk)
            if not cand_uk or not uk_target or len(cand_uk.split()) < 2:
                return False
            if cand_uk == uk_target:
                return True
            if allow_prefix and (cand_uk.startswith(uk_target + " ") or uk_target.startswith(cand_uk + " ")):
                return True
            # Check _declared_unit_spellings (which properly expands token-level slashes like 'open the door/way to something')
            for sp in cls._declared_unit_spellings(cand_raw_or_uk):
                sp_uk = cls._unit_key(re.sub(r"\bjust\b", "", sp).strip())
                if len(sp_uk.split()) < 2:
                    continue
                if sp_uk == uk_target:
                    return True
                if allow_prefix and (sp_uk.startswith(uk_target + " ") or uk_target.startswith(sp_uk + " ")):
                    return True
            # Expand slash alternatives and optional words ('just') in unit phrases like 'may/might/could (just) as well'
            for br in cls.clean_and_expand_pattern_slashes(cand_raw_or_uk):
                br_uk = cls._unit_key(re.sub(r"\bjust\b", "", br).strip())
                if len(br_uk.split()) < 2:
                    continue
                if br_uk == uk_target:
                    return True
                if allow_prefix and (br_uk.startswith(uk_target + " ") or uk_target.startswith(br_uk + " ")):
                    return True
            cand_no_just = re.sub(r"\s+", " ", re.sub(r"\bjust\b", "", cand_uk)).strip()
            if len(cand_no_just.split()) >= 2 and cand_no_just == uk_target:
                return True
            return False

        if len(tokens) >= 2:
            is_verb_expr = (
                locked_expr_type == "phrasal verb"
                or (tokens and cls.determine_contextual_pos(tokens[0], quote) == "verb")
            )
            # Check phrasal verb blocks FIRST so verbal expressions ('wind up') never get hijacked by hyphenated noun entries ('wind-up')
            for uk in unit_keys:
                for block in cls._unit_blocks(uk):
                    b_senses = cls._unit_senses(block, uk)
                    if len(b_senses) >= 2 and quote:
                        base_v = uk.split()[0]
                        part_w = uk.split()[-1]
                        # Filter out speech-act retraction senses ('admit that you were wrong', 'something that you said')
                        # when the quote has a separable noun object between verb and particle ('take her child back') or 'back to <NP>'
                        has_sep_or_dir = bool(
                            re.search(rf"\b{re.escape(base_v)}[a-z]*\s+(?:[a-z]+\s+){{1,4}}{re.escape(part_w)}\b", q_low)
                            or re.search(rf"\b{re.escape(part_w)}\s+to\s+(?:the|a|an|his|her|their|our|my|your|[a-z]+)\b", q_low)
                        )
                        if has_sep_or_dir and not any(sw in q_low for sw in ("said", "say", "words", "statement", "remark", "comment", "apolog", "admit", "wrong")):
                            non_speech = [
                                s for s in b_senses
                                if "admit that you were wrong" not in (s.get("definition") or "").lower()
                                and "something that you said" not in (s.get("definition") or "").lower()
                            ]
                            if non_speech:
                                b_senses = non_speech
                        prep_b = [dict(s, patterns=list(s.get("patterns") or []) + ([uk] if uk not in (s.get("patterns") or []) else [])) for s in b_senses]
                        res_b = cls._lock_sense({"word": base_v, "senses": prep_b}, quote=quote, target_pos="verb", return_scores=True)
                        if res_b and len(res_b) >= 5 and res_b[4]:
                            ranked_idxs = [s_i for s_i, _ in sorted(res_b[4], key=lambda x: (-x[1], x[0]))]
                            b_senses = [b_senses[s_i] for s_i in ranked_idxs if 0 <= s_i < len(b_senses)]
                    for s in b_senses:
                        df = cls.format_ldoce_definition(s)
                        _add_cand(df, locked_expr_type, block.get('phrase') or block.get('headword') or uk, s.get('examples', []), s, True)
            for mw_cand in unit_keys:
                mw_entry = cls.get_pedlex_entry(mw_cand)
                if mw_entry and mw_entry.get("senses"):
                    for s_i, s in enumerate(mw_entry.get("senses") or []):
                        if is_verb_expr and cls._pos_code(s.get("pos")) in ("n", "adj"):
                            continue
                        df = cls.format_ldoce_definition(s)
                        _add_cand(df, locked_expr_type, f"{mw_cand} #{s_i}", s.get("examples", []), s, True)

        is_passive_frame = len(tokens) >= 3 and tokens[0] in ("be", "feel", "get") and tokens[1].endswith("ed")

        # 2. PedLex phrase index candidates (exact or slash-expanded unit_key match first, prefix fallback if empty)
        query_list = []
        for q_s in [clean, u_key, " ".join(lem_tokens), lem_u_key]:
            if q_s and q_s not in query_list:
                query_list.append(q_s)
        for allow_pref in (False, True):
            if allow_pref and candidates:
                break
            p_matched_senses = []
            for query_str in query_list:
                phrase_cands = cls.lookup_ldoce_phrase_candidates(query_str)
                for host, s_idx, tier, raw_p in phrase_cands:
                    raw_p_low = str(raw_p).lower()
                    if "hospital" in raw_p_low and quote and not any(hw in q_low for hw in ("hospital", "patient", "ward", "doctor", "ill", "sick", "injured", "emergency")):
                        continue
                    if is_passive_frame and re.search(rf"\b{re.escape(tokens[-1])}\s+(?:doing\s+|somebody\b|sb\b)", re.sub(r"[()\[\]]+", " ", raw_p_low)):
                        continue
                    if not any(_unit_key_matches(raw_p, uk, allow_prefix=allow_pref) for uk in unit_keys):
                        continue
                    h_entry = cls.get_pedlex_entry(host)
                    if h_entry:
                        senses = h_entry.get('senses', [])
                        if s_idx is not None and s_idx < len(senses):
                            s = senses[s_idx]
                            exact_uk_prio = 1 if any(cls._unit_key(raw_p) == uk for uk in unit_keys) else 0
                            p_matched_senses.append((exact_uk_prio, host, s_idx, s))
            if len(p_matched_senses) >= 2 and quote:
                mock_ps = [item[3] for item in p_matched_senses]
                res_p = cls._lock_sense({"word": p_matched_senses[0][1], "senses": mock_ps}, quote=quote, return_scores=True)
                sc_p = dict(res_p[4]) if (res_p and len(res_p) >= 5 and res_p[4]) else {}
                p_matched_senses.sort(key=lambda x: (-x[0], -sc_p.get(mock_ps.index(x[3]), 0.0), x[2]))
            for _prio, host, s_idx, s in p_matched_senses:
                df = cls._compose_collocation_definition(u_key, None, host, s, cls.format_ldoce_definition(s), quote)
                _add_cand(df, locked_expr_type, f"{host} #{s_idx}", s.get('examples', []), s, True)

        # 3. Constituent headword senses:
        # Pass 3A-1 scans all senses of each constituent token strictly for attested multi-word `units` matches.
        stop_func = {
            "a", "an", "the", "to", "of", "in", "on", "at", "by", "for", "with", "from",
            "into", "onto", "upon", "over", "under", "up", "down", "off", "away",
            "not", "no", "be", "do", "have", "one", "ones", "it", "that", "this", "as", "or", "and"
        }
        LIGHT_OR_MODAL_TOKS = {
            "take", "make", "come", "go", "get", "give", "have", "do", "be", "put", "keep",
            "fall", "turn", "set", "run", "hold", "bring", "find", "might", "can", "will",
            "may", "would", "should", "could", "must", "there", "here"
        }
        raw_toks = []
        for t in (lem_tokens + tokens):
            if (len(t) >= 3 and (t not in stop_func or (t == "out" and clean == "out of"))) and t not in raw_toks:
                raw_toks.append(t)
            # When a copular/passive frame has a -ed participle at position 1 ('be obsessed with', 'be featured in', 'be measured in'),
            # also include its base verb lemma so Pass 3A-1/3A-2 can locate units/patterns stored on the base verb entry.
            if t.endswith("ed") and len(t) > 4:
                for v_stem in ([cls._surface_lemma(t)] + cls._headword_candidates(t)):
                    if v_stem and v_stem != t and v_stem not in raw_toks and cls.get_pedlex_entry(v_stem):
                        raw_toks.append(v_stem)
        # Prioritize specific open-class content words ahead of light verbs/modals
        content_toks = [t for t in raw_toks if t not in LIGHT_OR_MODAL_TOKS] + [t for t in raw_toks if t in LIGHT_OR_MODAL_TOKS]

        def _sense_matches_unit_strictly(s_obj: Dict[str, Any], allow_sub: bool = False) -> Tuple[bool, List[str]]:
            s_units = s_obj.get("units") or []
            for u_obj in s_units:
                u_raw = u_obj.get("phrase") if isinstance(u_obj, dict) else str(u_obj or "")
                u_p = cls._unit_key(u_raw or "")
                if not u_p or len(u_p.split()) < 2:
                    continue
                for uk in unit_keys:
                    if _unit_key_matches(u_raw or u_p, uk, allow_prefix=allow_sub):
                        u_ex = (u_obj.get("examples") or []) if isinstance(u_obj, dict) else []
                        return True, u_ex
                    if allow_sub and len(u_p.split()) >= 2 and len(uk.split()) >= 2 and (f" {u_p} " in f" {uk} " or f" {uk} " in f" {u_p} "):
                        u_ex = (u_obj.get("examples") or []) if isinstance(u_obj, dict) else []
                        return True, u_ex
            return False, []

        head_candidates = {tokens[0], lem_0} if tokens else set()
        if len(tokens) >= 2 and tokens[0] in ("be", "feel", "get", "make", "keep", "find", "leave", "hold", "remain", "stay", "become", "turn", "fall"):
            head_candidates.add(tokens[1])
            for v_stem_1 in ([cls._surface_lemma(tokens[1])] + cls._headword_candidates(tokens[1])):
                if v_stem_1:
                    head_candidates.add(v_stem_1)

        is_passive_frame = len(tokens) >= 3 and tokens[0] in ("be", "feel", "get") and tokens[1].endswith("ed")

        def _sense_matches_head_pattern(c_tok: str, s_obj: Dict[str, Any]) -> bool:
            if tokens and c_tok in head_candidates:
                if is_passive_frame and cls._pos_code(s_obj.get("pos")) == "n":
                    return False
                excl_toks = {c_tok, "be", "feel", "get"}
                if tokens[0] in ("be", "feel", "get") and len(tokens) >= 2 and c_tok != tokens[1]:
                    excl_toks.add(tokens[1])
                    excl_toks.update(cls._headword_candidates(tokens[1]))
                    excl_toks.add(cls._surface_lemma(tokens[1]))
                other_content = [t for t in (lem_tokens + tokens) if t not in excl_toks and len(t) >= 3 and t not in stop_func]
                other_toks = other_content if other_content else [t for t in (lem_tokens + tokens) if t not in excl_toks and len(t) >= 2]
                if other_toks:
                    pats = []
                    for p in (s_obj.get("patterns") or []):
                        pats.extend(v.lower() for v in cls._slash_variants(str(p)))
                    for pb in (s_obj.get("pattern_bundles") or []):
                        if isinstance(pb, dict) and pb.get("pattern"):
                            pats.extend(v.lower() for v in cls._slash_variants(str(pb["pattern"])))
                    other_tok_set = set(other_toks)
                    for pat_str in pats:
                        if "hospital" in pat_str and quote and not any(hw in q_low for hw in ("hospital", "patient", "ward", "doctor", "ill", "sick", "injured", "emergency")):
                            continue
                        pat_words = re.findall(r"[a-z]+", pat_str)
                        pat_word_set = set(pat_words) | {cls._surface_lemma(w) for w in pat_words}
                        if all(ot in pat_word_set for ot in other_tok_set):
                            if is_passive_frame and len(other_tok_set) == 1:
                                p_prep = next(iter(other_tok_set))
                                head_forms = {c_tok, tokens[1]} | set(cls._headword_candidates(tokens[1]))
                                head_pos = [i for i, w in enumerate(pat_words) if w in head_forms]
                                prep_pos = [i for i, w in enumerate(pat_words) if w == p_prep]
                                if head_pos and prep_pos and max(prep_pos) < min(head_pos):
                                    continue
                                # Active-only speech/gerund patterns ('admit to somebody', 'admit to doing sth')
                                # do not passivize into 'be <V-ed> to [organization/place]'
                                pat_str_flat = re.sub(r"[()\[\]]+", " ", pat_str)
                                if re.search(rf"\b(?:{'|'.join(re.escape(hf) for hf in head_forms)})\s+{re.escape(p_prep)}\s+(?:doing\s+|somebody\b|sb\b)", pat_str_flat):
                                    continue
                            return True
                        # Directional preposition open-class pattern (e.g. 'move away/out/to/towards etc' covers 'move into')
                        if not other_content and len(other_tok_set) == 1 and next(iter(other_tok_set)) in ("into", "onto", "towards", "toward", "across", "along", "through"):
                            if "etc" in pat_words and any(dp in pat_words for dp in ("to", "towards", "toward", "away", "out", "in", "into", "down", "up")):
                                return True
                    if not other_content and s_obj.get("examples"):
                        uk_list = [uk for uk in unit_keys if len(uk.split()) >= 2]
                        for uk in list(uk_list):
                            if uk.startswith(("be ", "feel ", "get ")) and len(uk.split()) >= 3:
                                uk_no_cop = " ".join(uk.split()[1:])
                                if uk_no_cop not in uk_list:
                                    uk_list.append(uk_no_cop)
                                p1 = uk_no_cop.split()[0]
                                if p1.endswith("led") and not p1.endswith("lled"):
                                    uk_br = p1[:-3] + "lled " + " ".join(uk_no_cop.split()[1:])
                                    if uk_br not in uk_list:
                                        uk_list.append(uk_br)
                        if any(cls._unit_example(uk, list(s_obj.get("examples") or [])) for uk in uk_list):
                            return True
            return False

        # Pass 3A-1: Strictly attested multi-word `units` across all senses of each constituent token
        for c_tok in content_toks:
            c_entry = cls.get_pedlex_entry(c_tok)
            if not c_entry:
                continue
            for s_i, s in enumerate(c_entry.get("senses") or []):
                is_match, matched_unit_ex = _sense_matches_unit_strictly(s)
                if not is_match:
                    continue
                raw_df = cls.format_ldoce_definition(s)
                if not raw_df:
                    continue
                df = cls._compose_collocation_definition(u_key, None, c_tok, s, raw_df, quote) or raw_df
                _add_cand(df, locked_expr_type, f"{c_tok} #{s_i}", matched_unit_ex or s.get('examples', []), s, True)

        # Pass 3A-2: Headword `patterns` / `pattern_bundles` match when no multi-word unit/phrase entry exists,
        # or when a verb + preposition frame (e.g. 'remind of') has competing main-entry pattern senses alongside phrasal-verb blocks.
        has_second_content = any(t not in head_candidates for t in content_toks) if tokens else False
        if not candidates or (not has_second_content and len(candidates) < k) or len(candidates) < 2:
            pat_matched_senses = []
            for c_tok in content_toks:
                c_entry = cls.get_pedlex_entry(c_tok)
                if not c_entry:
                    continue
                for s_i, s in enumerate(c_entry.get("senses") or []):
                    if not _sense_matches_head_pattern(c_tok, s):
                        continue
                    raw_df = cls.format_ldoce_definition(s)
                    if not raw_df:
                        continue
                    # For passive frames ('be admitted to', 'be featured in', 'be measured in'), prioritize senses with
                    # explicit passive patterns ('be <V-ed> <prep>') or transitive object-before-prep patterns ('<V> sb/sth <prep>')
                    rank_prio = 0
                    all_p_strs = [str(p).lower() for p in (s.get("patterns") or [])] + [
                        str(pb.get("pattern") or "").lower() for pb in (s.get("pattern_bundles") or []) if isinstance(pb, dict)
                    ]
                    if is_passive_frame:
                        p_word = tokens[-1]
                        v_stems_1 = [c for c in ([cls._surface_lemma(tokens[1])] + cls._headword_candidates(tokens[1])) if c and c != tokens[1] and cls.get_ldoce_entry(c)]
                        v_stem_1 = v_stems_1[0] if v_stems_1 else cls._surface_lemma(tokens[1])
                        if any(tokens[1] in ps and p_word in ps for ps in all_p_strs):
                            rank_prio = 2
                        elif any(re.search(rf"\b{re.escape(v_stem_1)}\s+(?:somebody|something|sb|sth)\b.*\b{re.escape(p_word)}\b", ps) for ps in all_p_strs):
                            rank_prio = 1
                    elif len(tokens) == 2 and tokens[0] in ("make", "keep", "find", "leave", "remain", "stay", "fall") and c_tok == tokens[1]:
                        rank_prio = 2
                    elif any(cls._unit_key(ps) in unit_keys for ps in all_p_strs):
                        rank_prio = 1
                    pat_matched_senses.append((rank_prio, c_tok, s_i, s, raw_df))
            if len(pat_matched_senses) >= 2 and quote:
                mock_s = [item[3] for item in pat_matched_senses]
                res_pm = cls._lock_sense({"word": pat_matched_senses[0][1], "senses": mock_s}, quote=quote, target_pos="verb" if is_passive_frame else None, return_scores=True)
                sc_map = dict(res_pm[4]) if (res_pm and len(res_pm) >= 5 and res_pm[4]) else {}
                pat_matched_senses.sort(key=lambda x: (-x[0], -sc_map.get(mock_s.index(x[3]), 0.0), x[2]))
            else:
                pat_matched_senses.sort(key=lambda x: (-x[0], x[2]))
            for _prio, c_tok, s_i, s, raw_df in pat_matched_senses:
                df = cls._compose_collocation_definition(u_key, None, c_tok, s, raw_df, quote) or raw_df
                _add_cand(df, locked_expr_type, f"{c_tok} #{s_i}", s.get('examples', []), s, True)

        # Pass 3B: WordNet (oewn:2025+) Multi-Word Synsets ONLY when PedLex has 0 attested phrase candidates
        if not candidates and len(tokens) >= 2:
            try:
                wn = cls.get_wordnet()
                if wn:
                    for mw_q in (clean, " ".join(lem_tokens), u_key):
                        if not mw_q or len(mw_q.split()) < 2:
                            continue
                        for syn in (wn.synsets(mw_q) or wn.synsets(mw_q.replace(" ", "_")) or []):
                            s_def = (syn.definition() or "").strip()
                            if not s_def:
                                continue
                            s_pos = getattr(syn, "pos", lambda: "")() if callable(getattr(syn, "pos", None)) else getattr(syn, "pos", "")
                            if s_pos == "v" and not s_def.lower().startswith("to "):
                                s_def = f"to {s_def}"
                            s_exs = [str(e).strip().strip('"“”') for e in (syn.examples() or []) if e]
                            _add_cand(s_def, locked_expr_type, f"wn:{mw_q}", s_exs, {"definition": s_def, "examples": s_exs}, True)
            except Exception:
                pass

        # Pass 3C: Constituent single-word fallback ONLY if NO phrase-attested candidates were found above
        if not candidates:
            is_verb_slot_frame = bool(re.search(r"[\[\(](?:sb|sth|oneself|one's)[\]\)]", expr_word, re.IGNORECASE)) and len(tokens) >= 1
            non_light = [t for t in content_toks if t not in LIGHT_OR_MODAL_TOKS] or content_toks
            expect_verb = is_verb_slot_frame and tokens and non_light and non_light[0] in (tokens[0], lem_0)
            for c_tok in non_light:
                c_entry = cls.get_pedlex_entry(c_tok)
                if c_entry:
                    for s_i, s in enumerate(c_entry.get("senses") or []):
                        if expect_verb and c_tok in (tokens[0], lem_0) and cls._pos_code(s.get("pos")) != "v":
                            continue
                        # Skip unrelated multi-word unit senses (e.g. 'call somebody names', 'fall victim/prey to')
                        s_units = s.get("units") or []
                        if s_units:
                            is_m, _ = _sense_matches_unit_strictly(s)
                            if not is_m:
                                continue
                        raw_df = cls.format_ldoce_definition(s)
                        if not raw_df:
                            continue
                        df = cls._compose_collocation_definition(u_key, None, c_tok, s, raw_df, quote) or raw_df
                        _add_cand(df, locked_expr_type, f"{c_tok} #{s_i}", s.get('examples', []), s, False)
                        if len(candidates) >= k - 2:
                            break
                # Also include WordNet verb synsets for verb-slot expressions (e.g. 'lift [sb] out of [sth]')
                if expect_verb and c_tok in (tokens[0], lem_0) and len(candidates) < k:
                    try:
                        wn = cls.get_wordnet()
                        if wn:
                            q_words = set(re.findall(r"[a-z]{3,}", q_low))
                            wn_syns = wn.synsets(c_tok, pos="v") or []
                            scored_v_syns = []
                            for syn in wn_syns:
                                s_def = (syn.definition() or "").strip()
                                if not s_def:
                                    continue
                                s_exs = [str(e).strip().strip('"“”') for e in (syn.examples() or []) if e]
                                s_bag = set(re.findall(r"[a-z]{3,}", (s_def + " " + " ".join(s_exs)).lower()))
                                ov = len(q_words & s_bag)
                                if ("out of" in clean or "from" in clean) and any(w in (s_def + " " + " ".join(s_exs)).lower() for w in ("condition", "rank", "poverty", "state", "from")):
                                    ov += 3
                                scored_v_syns.append((ov, s_def, s_exs))
                            scored_v_syns.sort(key=lambda x: -x[0])
                            for _ov, s_def, s_exs in scored_v_syns[:4]:
                                if not s_def.lower().startswith("to "):
                                    s_def = f"to {s_def}"
                                _add_cand(s_def, locked_expr_type, f"wn:{c_tok}", s_exs, {"definition": s_def, "examples": s_exs}, False)
                                if len(candidates) >= k:
                                    break
                    except Exception:
                        pass
                if len(candidates) >= k:
                    break

        return candidates[:k]

    @classmethod
    def ensure_distinct_example(
        cls,
        word_or_formula: str,
        pos: str,
        current_example: str,
        quoted_sentence: str,
        sense: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Guarantees that Example Usage is NEVER a verbatim duplicate of Quoted Sentence,
        and ALWAYS demonstrates the target word or expression in an authentic dictionary sentence
        from its locked sense (or from monosemous corpus entries). Returns '' if no valid example exists.
        """
        clean_target = re.sub(r'\[.*?\]|\(.*?\)', '', str(word_or_formula or '')).strip()
        clean_target = re.sub(r"\b(?:sb's|sb|sth|one's|oneself)\b", "", clean_target).strip()
        clean_target = re.sub(r"\s+", " ", clean_target).strip()
        clean_low = clean_target.lower()
        u_key = cls._unit_key(clean_target)
        u_key_no_cop = " ".join(u_key.split()[1:]) if (len(u_key.split()) >= 3 and u_key.split()[0] in ("be", "feel", "get")) else ""
        is_passive_cop = bool(u_key_no_cop and u_key_no_cop.split()[0].endswith("ed"))
        is_multi = len(u_key.split()) >= 2 and "-" not in clean_low
        q_norm = re.sub(r'\s+', ' ', str(quoted_sentence or '').strip()).lower()
        pos_low = str(pos or '').strip().lower()

        def _is_valid_distinct(ex_str: str, check_pos: bool = True) -> bool:
            if not ex_str or not str(ex_str).strip():
                return False
            ex_clean = str(ex_str).strip().strip('"“”')
            e_norm = re.sub(r'\s+', ' ', ex_clean).lower()
            if e_norm == q_norm or (q_norm and (e_norm in q_norm or q_norm in e_norm) and abs(len(e_norm) - len(q_norm)) <= 12):
                return False
            if is_multi:
                if is_passive_cop:
                    part_tok = u_key_no_cop.split()[0]
                    v_stems = [c for c in ([cls._surface_lemma(part_tok)] + cls._headword_candidates(part_tok)) if c and c != part_tok and cls.get_ldoce_entry(c)]
                    v_stem = v_stems[0] if v_stems else cls._surface_lemma(part_tok)
                    # Reject singular noun false positives (e.g. 'a standard feature in' for 'be featured in')
                    if v_stem and v_stem != part_tok and re.search(rf"\b(?:a|an|the)\s+(?:\w+\s+)?{re.escape(v_stem)}\b", e_norm) and part_tok not in e_norm:
                        return False
                if cls._unit_example(u_key, [ex_clean], allow_modifier=True):
                    return True
                if u_key_no_cop:
                    if cls._unit_example(u_key_no_cop, [ex_clean], allow_modifier=True):
                        return True
                    # British/American doubled-consonant participle equivalence ('modeled on' <-> 'modelled on')
                    p0 = u_key_no_cop.split()[0]
                    if p0.endswith("led") and not p0.endswith("lled"):
                        uk_br = p0[:-3] + "lled " + " ".join(u_key_no_cop.split()[1:])
                        if cls._unit_example(uk_br, [ex_clean], allow_modifier=True):
                            return True
                    # Transitive active / slash-preposition pattern equivalence on the locked verb sense (e.g. 'measure something in something' for 'be measured in', 'admit somebody to/into something' for 'be admitted to')
                    if is_passive_cop and sense and isinstance(sense, dict):
                        p_word = u_key_no_cop.split()[-1]
                        v_stems = [c for c in ([cls._surface_lemma(p0)] + cls._headword_candidates(p0)) if c and c != p0 and cls.get_ldoce_entry(c)]
                        v_stem = v_stems[0] if v_stems else cls._surface_lemma(p0)
                        for pb in (sense.get("pattern_bundles") or []):
                            if isinstance(pb, dict) and pb.get("pattern") and pb.get("example"):
                                if str(pb["example"]).strip().strip('"“”') == ex_clean:
                                    for p_var in cls._slash_variants(str(pb["pattern"])):
                                        pat_words = re.findall(r"[a-z]+", p_var.lower())
                                        if p_word in pat_words and (p0 in pat_words or (v_stem and v_stem in pat_words)):
                                            return True
                        all_s_pats = list(sense.get("patterns") or []) + [
                            pb.get("pattern") for pb in (sense.get("pattern_bundles") or [])
                            if isinstance(pb, dict) and pb.get("pattern")
                        ]
                        for pat in all_s_pats:
                            for p_var in cls._slash_variants(str(pat)):
                                pat_l = p_var.lower()
                                if p_word in re.findall(r"[a-z]+", pat_l) and (p0 in pat_l or (v_stem and v_stem in pat_l)):
                                    if cls.match_typed_slot_pattern(str(pat), ex_clean):
                                        return True
                norm_clean = re.sub(r"[^a-z0-9]+", " ", clean_low).strip()
                norm_ex = re.sub(r"[^a-z0-9]+", " ", e_norm).strip()
                return bool(norm_clean and norm_clean in norm_ex)
            if not cls.text_contains_form(ex_clean, clean_target):
                return False
            if check_pos and pos_low in ("noun", "verb", "adjective", "adverb"):
                ex_pos = cls.determine_contextual_pos(clean_target, ex_clean)
                if ex_pos and ex_pos in ("noun", "verb", "adjective", "adverb") and ex_pos != pos_low:
                    return False
            return True

        # 1. Check locked sense examples, units, and pattern_bundles first (Strict Sense-Isolation)
        cand_pool: List[str] = []
        if sense and isinstance(sense, dict):
            # When u_key_no_cop has a specific preposition (e.g. 'featured in', 'measured in'), promote pattern_bundles matching participle+prep first, then stem+prep
            target_prep = u_key_no_cop.split()[-1] if u_key_no_cop else ""
            part_tok = u_key_no_cop.split()[0] if u_key_no_cop else ""
            part_stems = [c for c in ([cls._surface_lemma(part_tok)] + cls._headword_candidates(part_tok)) if c and c != part_tok and cls.get_ldoce_entry(c)] if part_tok else []
            part_stem = part_stems[0] if part_stems else (cls._surface_lemma(part_tok) if part_tok else "")
            exact_pb_exs = []
            matching_pb_exs = []
            other_pb_exs = []
            for pb in (sense.get("pattern_bundles") or []):
                if isinstance(pb, dict) and pb.get("example"):
                    pb_ex = str(pb["example"]).strip()
                    pb_pat = str(pb.get("pattern") or "").lower()
                    pb_words = re.findall(r"[a-z]+", pb_pat)
                    if target_prep and target_prep in pb_words and part_tok and part_tok in pb_words:
                        exact_pb_exs.append(pb_ex)
                    elif target_prep and target_prep in pb_words and ((part_stem and part_stem in pb_words) or not is_passive_cop):
                        matching_pb_exs.append(pb_ex)
                    else:
                        other_pb_exs.append(pb_ex)
            cand_pool.extend(exact_pb_exs)
            cand_pool.extend(matching_pb_exs)
            cand_pool.extend(other_pb_exs)
            for u_obj in (sense.get("units") or []):
                if isinstance(u_obj, dict):
                    for ex in (u_obj.get("examples") or []):
                        if ex:
                            cand_pool.append(str(ex).strip())
            for ex in (sense.get("examples") or []):
                if ex:
                    cand_pool.append(str(ex).strip())
            # If multi-word expression has an extended preposition pattern attested in quoted_sentence (e.g. 'get by on'),
            # prefer an example from the locked sense that demonstrates that exact extended pattern.
            if is_multi and q_norm and cand_pool:
                for pat in (sense.get("patterns") or []):
                    for p_var in cls._slash_variants(str(pat)):
                        p_k = cls._unit_key(p_var)
                        if p_k and len(p_k.split()) > len(u_key.split()) and cls._unit_example(p_k, [quoted_sentence], allow_modifier=True):
                            for cand_ex in cand_pool:
                                if _is_valid_distinct(cand_ex, check_pos=False) and cls._unit_example(p_k, [cand_ex], allow_modifier=True):
                                    return cand_ex.strip().strip('"“”')
            for exact_ex in (exact_pb_exs + matching_pb_exs):
                if _is_valid_distinct(exact_ex, check_pos=False):
                    return exact_ex.strip().strip('"“”')

        if _is_valid_distinct(current_example, check_pos=(sense is None)):
            return str(current_example).strip().strip('"“”')

        if cand_pool:
            for cand_ex in cand_pool:
                if _is_valid_distinct(cand_ex, check_pos=False):
                    return cand_ex.strip().strip('"“”')

        # Determine whether the entry (or multi-word unit) is monosemous under the target POS
        entry = cls.get_ldoce_entry(clean_target)
        target_p = cls._pos_code(pos_low)
        entry_senses = (entry.get("senses") or []) if entry else []
        pos_senses = [s for s in entry_senses if not target_p or cls._sense_matches_pos(s, target_p)]
        is_derived_projected = bool(
            entry and entry.get("kind") == "derived"
            and pos_senses
            and all(s.get("is_derived_projected") or s.get("gloss_source") in ("wordnet", "derivational") for s in pos_senses)
        )
        is_monosemous = len(pos_senses) <= 1 or (sense is None and is_derived_projected)

        u_blocks = cls._unit_blocks(u_key) if is_multi else []
        u_rows = cls._unit_phrase_rows(u_key) if is_multi else []
        has_dedicated_unit = bool(u_rows or u_blocks or (entry and len(entry_senses) == 1))
        has_poly_block = any(len(cls._unit_senses(b, u_key)) > 1 for b in u_blocks)
        is_monosemous_multi = is_multi and has_dedicated_unit and not has_poly_block and len(u_rows) <= 1
        # Check if a copular/prepositional multi-word expression ('be exhausted from', 'be smart about')
        # is grounded in a host entry where only 1 sense owns that headword + preposition pattern.
        if is_multi and not is_monosemous_multi and sense and isinstance(sense, dict):
            uk_check = u_key_no_cop or u_key
            uk_toks = uk_check.split()
            if len(uk_toks) == 2:
                h_word, p_word = uk_toks[0], uk_toks[1]
                for h_cand in (h_word, cls._surface_lemma(h_word)):
                    if not h_cand:
                        continue
                    h_ent = cls.get_ldoce_entry(h_cand)
                    if not h_ent:
                        continue
                    matching_pat_senses = []
                    for hs in (h_ent.get("senses") or []):
                        hs_pats = []
                        for hp in (hs.get("patterns") or []):
                            hs_pats.extend(v.lower() for v in cls._slash_variants(str(hp)))
                        for hpb in (hs.get("pattern_bundles") or []):
                            if isinstance(hpb, dict) and hpb.get("pattern"):
                                hs_pats.extend(v.lower() for v in cls._slash_variants(str(hpb["pattern"])))
                        if any(p_word in re.findall(r"[a-z]+", p_str) for p_str in hs_pats):
                            matching_pat_senses.append(hs)
                    if len(matching_pat_senses) == 1 and matching_pat_senses[0].get("definition") == sense.get("definition"):
                        is_monosemous_multi = True
                        break

        # If sense was locked to a WordNet fallback sense while PedLex already has senses for this POS,
        # do NOT steal examples from PedLex's different senses.
        if sense and isinstance(sense, dict):
            if is_multi:
                if not is_monosemous_multi:
                    return ""
            else:
                s_def_norm = re.sub(r"[^a-z0-9]+", " ", str(sense.get("definition") or "").lower()).strip()
                in_pedlex = any(
                    ps is sense
                    or re.sub(r"[^a-z0-9]+", " ", str(cls.format_ldoce_definition(ps) or "").lower()).strip() == s_def_norm
                    or re.sub(r"[^a-z0-9]+", " ", str(ps.get("definition") or "").lower()).strip() == s_def_norm
                    for ps in pos_senses
                )
                if pos_senses and not in_pedlex:
                    return ""
                if not is_monosemous:
                    # If all senses on a derived entry are synthesized/projected ('wordnet' or 'derivational'),
                    # allow corpus/WordNet example lookup for the derived headword itself if Sense-Anchored Rescue doesn't find a sense-specific hit.
                    is_derived_projected = bool(
                        entry and entry.get("kind") == "derived"
                        and all(s.get("is_derived_projected") or s.get("gloss_source") in ("wordnet", "derivational") for s in pos_senses)
                    )
                    # Sense-Anchored Corpus Rescue: when a locked PedLex sense has 0 built-in examples (e.g. 'tablet' Sense 4),
                    # search PedLex corpus examples ONLY if the example matches at least one distinctive content word
                    # from this sense's definition/signpost AND has zero overlap with competing senses' distinctive keywords.
                    _stop_ex = {
                        "the", "and", "for", "with", "that", "this", "from", "into", "over", "under", "about",
                        "used", "make", "made", "something", "somebody", "someone", "person", "people", "thing",
                        "things", "small", "large", "piece", "type", "kind", "form", "flat", "round", "hard",
                        "have", "having", "which", "when", "where", "your", "their", "other", "such", "especially"
                    }
                    s_kws = set(re.findall(r"[a-z]{4,}", (str(sense.get("definition") or "") + " " + str(sense.get("signpost") or "")).lower())) - _stop_ex
                    other_kws: Set[str] = set()
                    for ps in pos_senses:
                        if ps is not sense and ps.get("definition") != sense.get("definition"):
                            other_kws.update(re.findall(r"[a-z]{4,}", (str(ps.get("definition") or "") + " " + str(ps.get("signpost") or "")).lower()))
                    s_distinct_kws = s_kws - other_kws
                    other_distinct_kws = (other_kws - s_kws) - _stop_ex
                    if s_distinct_kws:
                        try:
                            for ex in (cls.search_corpus_examples(clean_target, limit=30) or []):
                                ex_s = str(ex).strip()
                                if not ex_s or not _is_valid_distinct(ex_s):
                                    continue
                                ex_toks = set(re.findall(r"[a-z]{4,}", ex_s.lower()))
                                if (ex_toks & s_distinct_kws) and not (ex_toks & other_distinct_kws):
                                    return ex_s.strip('"“”')
                        except Exception:
                            pass
                    if is_derived_projected:
                        is_monosemous = True
                    else:
                        # Polysemous word whose locked sense has 0 valid examples: do NOT steal from another sense
                        return ""

        # 2. Monosemous or unlocked entry-level & cross-entry corpus examples
        if entry and is_monosemous:
            for s in pos_senses:
                for pb in (s.get("pattern_bundles") or []):
                    if isinstance(pb, dict) and pb.get("example"):
                        cand_pool.append(str(pb["example"]).strip())
                for ex in (s.get("examples") or []):
                    if ex:
                        cand_pool.append(str(ex).strip())
            for cc in (entry.get("cross_collocations") or []):
                if isinstance(cc, dict) and cc.get("example"):
                    cand_pool.append(str(cc["example"]).strip())

        # 3. For multiword units, check _unit_phrase_rows and _unit_blocks when not polysemously conflicting
        if is_multi and (not sense or is_monosemous_multi):
            for _h, row in u_rows:
                for ex in (row.get("examples") or []):
                    if ex:
                        cand_pool.append(str(ex).strip())
            for block in u_blocks:
                u_senses = cls._unit_senses(block, u_key)
                if not sense or len(u_senses) <= 1:
                    for s in u_senses:
                        for ex in (s.get("examples") or []):
                            if ex:
                                cand_pool.append(str(ex).strip())

        for cand_ex in cand_pool:
            if _is_valid_distinct(cand_ex):
                return cand_ex.strip().strip('"“”')

        # 4. Search corpus examples in PedLex SQLite (only for monosemous words or monosemous dedicated multi-word units)
        if (not is_multi and is_monosemous) or (is_multi and (is_monosemous_multi or (not sense and not has_poly_block))):
            try:
                search_queries = [u_key] if is_multi else [clean_target]
                if is_multi and u_key_no_cop and u_key_no_cop not in search_queries:
                    search_queries.append(u_key_no_cop)
                for sq in search_queries:
                    for ex in (cls.search_corpus_examples(sq, limit=15) or []):
                        if ex and _is_valid_distinct(str(ex).strip()):
                            return str(ex).strip().strip('"“”')
            except Exception:
                pass

        # 5. Check WordNet synset examples (only when monosemous in both PedLex and WordNet for target POS)
        if not is_multi and is_monosemous:
            try:
                wn = cls.get_wordnet()
                if wn:
                    wn_pos_map = {"noun": ("n",), "verb": ("v",), "adjective": ("a", "s"), "adverb": ("r",)}
                    wn_poses = wn_pos_map.get(pos_low)
                    synsets = [s for p_code in wn_poses for s in (wn.synsets(clean_target.replace(" ", "_"), pos=p_code) or [])] if wn_poses else wn.synsets(clean_target.replace(" ", "_"))
                    if len(synsets) <= 1:
                        for syn in synsets:
                            for ex in (syn.examples() or []):
                                if not ex:
                                    continue
                                ex_s = re.sub(r"\s*\((?:or|e\.g\.)\s+[^)]+\)", "", str(ex)).strip()
                                if ex_s and _is_valid_distinct(ex_s):
                                    return ex_s.strip().strip('"“”')
            except Exception:
                pass

        return ""

    @classmethod
    def extract_deterministic_vocabulary(
        cls,
        text: str,
        syllabus_vocab: Optional[List[str]] = None,
        target_count: int = 20,
        enable_llm_disambiguation: bool = True,
        exclude_words: Optional[Set[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Pure Computational Linguistic & Lexicographic Extractor (0 Tokens, 100% Deterministic).
        Extracts academic target vocabulary, locates authentic quoted sentences with Sentence IDs,
        tags contextual Part of Speech via spaCy, and hydrates authoritative definitions & authentic
        corpus examples from PedLex 6th Edition.
        """
        with cls.wsd_log_session("vocabulary"):
            _, pool = cls.tokenize_and_index_sentences(text)
            if not pool:
                return []

            excl_set: Set[str] = {
                re.sub(r'\[.*?\]|\(.*?\)', '', str(ew)).strip().lower()
                for ew in (exclude_words or set())
                if ew
            }
            excl_set.discard("")

            # 1. Determine target words (Syllabus primary + Passage reserve pool for Gate 2 backfill)
            target_candidates: List[Dict[str, Any]] = []
            if syllabus_vocab:
                cleaned_syllabus: List[str] = []
                seen_clean = set(excl_set)
                for w in syllabus_vocab:
                    w_clean = re.sub(r'\[.*?\]|\(.*?\)', '', str(w)).strip().lower()
                    if w_clean and w_clean not in seen_clean:
                        seen_clean.add(w_clean)
                        cleaned_syllabus.append(w_clean)

                # Verbatim Compound Resolution (Passage = Ground Truth):
                # If a bare syllabus word (e.g. 'indulgent', 'confidence', 'prone', 'reflective', 'balance')
                # has zero standalone occurrences in the passage, check if it only appears as a sub-token
                # of a hyphenated compound in the passage (e.g. 'self-indulgent', 'self-confidence', 'waste-prone').
                # Promote the candidate to the verbatim passage compound so Gate 1 & Gate 2 evaluate the true
                # passage word (accepting attested dictionary compounds like 'self-indulgent' / 'self-confidence'
                # and cleanly dropping unattested coinages like 'waste-prone' / 'self-reflective' / 'self-balance').
                if cleaned_syllabus and pool:
                    passage_hyphen_compounds: List[str] = []
                    seen_hyph: Set[str] = set()
                    for _sid, sent in pool.items():
                        for m_hc in re.finditer(r"\b([a-zA-Z]+(?:-[a-zA-Z]+)+)\b", sent):
                            hc = m_hc.group(1).lower()
                            if hc not in seen_hyph:
                                seen_hyph.add(hc)
                                passage_hyphen_compounds.append(hc)
                    if passage_hyphen_compounds:
                        resolved_syllabus: List[str] = []
                        seen_resolved: Set[str] = set()
                        for sw in cleaned_syllabus:
                            resolved_w = sw
                            if "-" not in sw and not any(cls.text_contains_form(sent, sw) for sent in pool.values()):
                                for hc in passage_hyphen_compounds:
                                    parts = hc.split("-")
                                    if any(cls.form_matches(p, sw) for p in parts):
                                        # Normalize plural hyphenated noun head to singular if attested in dictionary
                                        hc_norm = hc
                                        if not cls.get_ldoce_entry(hc) and len(parts) >= 2:
                                            last_lem = cls._surface_lemma(parts[-1])
                                            if last_lem and last_lem != parts[-1]:
                                                cand_sing = "-".join(parts[:-1] + [last_lem])
                                                if cls.get_ldoce_entry(cand_sing):
                                                    hc_norm = cand_sing
                                        resolved_w = hc_norm
                                        break
                            if resolved_w not in seen_resolved:
                                seen_resolved.add(resolved_w)
                                resolved_syllabus.append(resolved_w)
                        cleaned_syllabus = resolved_syllabus
                        seen_clean = set(cleaned_syllabus) | excl_set

                # Pre-filter same-word-family syllabus pairs (e.g. 'intrigue' + 'intrigued', 'predominant' + 'predominantly'):
                # If a syllabus word has a same-family sibling in the syllabus, drop any member that:
                #   (a) is completely unattested via text_contains_form while a sibling IS attested (e.g. 'predominant' vs 'predominantly'), OR
                #   (b) only matches passage sentences where the exact surface token is another syllabus sibling (e.g. 'intrigue' in "feeling intrigued")
                #       AND that sibling has a valid dictionary sense + example (otherwise retain the base headword).
                if len(cleaned_syllabus) > 1 and pool:
                    syl_shown_map: Dict[str, List[Tuple[str, str]]] = {
                        sw: [(sid, sent) for sid, sent in pool.items() if cls.text_contains_form(sent, sw)]
                        for sw in cleaned_syllabus
                    }
                    filtered_syllabus: List[str] = []
                    for sw in cleaned_syllabus:
                        siblings = [
                            other for other in cleaned_syllabus
                            if other != sw and cls.are_same_word_family(sw, other)
                        ]
                        if not siblings:
                            filtered_syllabus.append(sw)
                            continue
                        sw_shown = syl_shown_map.get(sw, [])
                        # Case (a): sw is not attested by any valid morphological form, while a sibling is attested
                        if not sw_shown and any(syl_shown_map.get(sib) for sib in siblings):
                            continue
                        # Case (b): sw has no exact surface match in its attested sentences, while a longer/derived sibling
                        # appears verbatim in those exact sentences AND has complete lexicon evidence (sense + example)
                        sw_has_exact = any(
                            bool(re.search(rf"\b{re.escape(sw)}\b", sent, flags=re.IGNORECASE))
                            for _sid, sent in sw_shown
                        )
                        if not sw_has_exact and sw_shown:
                            superseded = False
                            for sib in siblings:
                                sib_exact_sents = [
                                    sent for _sid, sent in sw_shown
                                    if re.search(rf"\b{re.escape(sib)}\b", sent, flags=re.IGNORECASE)
                                ]
                                if sib_exact_sents:
                                    sib_cands = cls.build_top_k_vocab_candidates(sib, sib_exact_sents[0], k=2)
                                    sib_ok = False
                                    if sib_cands:
                                        top_c = sib_cands[0]
                                        sib_def = top_c.get("definition", "")
                                        sib_ex = (top_c.get("examples") or [""])[0]
                                        sib_ex = cls.ensure_distinct_example(
                                            sib, str(top_c.get("pos") or "adjective"), sib_ex,
                                            sib_exact_sents[0], sense=top_c.get("raw_sense")
                                        )
                                        if sib_def and not cls.is_boilerplate_definition(sib_def) and sib_ex:
                                            sib_ok = True
                                    if sib_ok:
                                        superseded = True
                                        break
                            if superseded:
                                continue
                        filtered_syllabus.append(sw)
                    cleaned_syllabus = filtered_syllabus

                for w in cleaned_syllabus:
                    target_candidates.append({"word": w, "sid": "", "quote": "", "is_syllabus": True})
                # Reserve passage candidates to replace any syllabus item dropped by Gate 2 (Evidence-Completeness)
                reserve_skels = cls.mine_vocabulary_skeletons(text, target_count=max(target_count, 20) + 15)
                for sk in reserve_skels:
                    rw = sk.get("word", "").strip().lower()
                    if rw and rw not in seen_clean:
                        seen_clean.add(rw)
                        sk_copy = dict(sk)
                        sk_copy["is_syllabus"] = False
                        target_candidates.append(sk_copy)
            else:
                skels = cls.mine_vocabulary_skeletons(text, target_count=target_count + 15)
                target_candidates = skels

            results: List[Dict[str, Any]] = []
            seen_words: Set[str] = set()
            seen_word_quotes: List[Tuple[str, str]] = []

            for cand in target_candidates:
                if len(results) >= target_count:
                    break
                w = cand["word"].strip().lower()
                if not w or w in seen_words or w in excl_set:
                    continue
                if not cls.is_pedagogical_target(w):
                    continue
                if any(cls.are_same_word_family(w, sw) for sw in seen_words):
                    continue

                matching_sent = cand.get("quote", "")
                matching_sid = cand.get("sid", "")

                # If sentence not already matched, search in sentence pool
                anchor_quality = "quote" if matching_sent else "syllabus"
                if not matching_sent and pool:
                    w_clean = re.sub(r'[^a-z0-9]', '', w)
                    shown = [(sid, sent) for sid, sent in pool.items() if cls.text_contains_form(sent, w)]
                    # Filter out sentences where a lowercase common-noun/adj target ONLY appears as a mid-sentence capitalized proper noun / acronym expansion (e.g., 'Placement' in 'AP (Advanced Placement)')
                    if shown and w.islower():
                        non_propn_shown = []
                        for sid_s, sent_s in shown:
                            # Fast check: if any lowercase occurrence exists, keep immediately
                            if any(re.search(rf"(?<![a-zA-Z-]){re.escape(f)}(?![a-zA-Z-])", sent_s) for f in ({w} | cls.inflected_forms(w))):
                                non_propn_shown.append((sid_s, sent_s))
                                continue
                            # Otherwise check via spaCy whether all occurrences are mid-sentence PROPN / capitalized title-case compounds
                            _nlp = cls.get_spacy()
                            doc_s = _nlp(cls._strip_lexical_quotes(sent_s))
                            has_valid_tok = False
                            for t_s in doc_s:
                                if cls.form_matches(t_s.text, w) or cls.form_matches(t_s.lemma_, w):
                                    is_mid_cap = (
                                        t_s.text and t_s.text[0].isupper() and t_s.i > 0
                                        and doc_s[t_s.i - 1].text not in (".", "!", "?", ":", ";", '"', "\u201c", "'", "\u2018", "\u2014", "--")
                                        and (t_s.pos_ == "PROPN" or doc_s[t_s.i - 1].text == "(" or doc_s[t_s.i - 1].text[0].isupper())
                                    )
                                    if not is_mid_cap:
                                        has_valid_tok = True
                                        break
                            if has_valid_tok:
                                non_propn_shown.append((sid_s, sent_s))
                        shown = non_propn_shown
                    # Avoid sentences already claimed by a same-word-family item
                    if len(shown) > 1 and seen_word_quotes:
                        non_clash_shown = [
                            (sid, sent) for sid, sent in shown
                            if not any(cls.are_same_word_family(w, sw) and sent == sq for sw, sq in seen_word_quotes)
                        ]
                        if non_clash_shown:
                            shown = non_clash_shown
                    # For deverbal prepositions/participial adjectives (e.g. 'given', 'provided', 'considering'),
                    # prefer sentences where the word is NOT preceded by an auxiliary ('have/has/had/be/been')
                    if w in ("given", "provided", "considering", "regarding", "concerning", "following", "including", "granted") and len(shown) > 1:
                        non_aux_shown = [
                            (sid, sent) for sid, sent in shown
                            if not re.search(rf"\b(?:have|has|had|am|is|are|was|were|be|been|being)\s+(?:\w+\s+)?{re.escape(w)}\b", sent, flags=re.IGNORECASE)
                        ]
                        if non_aux_shown:
                            shown = non_aux_shown
                    valid_shown = [(sid, sent) for sid, sent in shown if cls.anchor_evidence_ok(sent, w)]
                    best_sent, _best_rich = cls.pick_anchor_sentence(w, [s for _sid, s in shown])
                    if best_sent:
                        matching_sent = best_sent
                        matching_sid = cls.get_sentence_id(best_sent, pool) or matching_sid
                        anchor_quality = "clause"
                    elif valid_shown:
                        matching_sid, matching_sent = valid_shown[0]
                        anchor_quality = "unqualified"
                    elif shown:
                        matching_sid, matching_sent = shown[0]
                        anchor_quality = "attested"

                # Ground-Truth Passage Gate: Never extract a word that is unattested as an independent word/inflection
                # in the passage (no blind substring hijacking of hyphenated compounds like 'waste-prone' -> 'prone'
                # or closed compounds like 'toothpaste' -> 'paste', 'microorganisms' -> 'organism').
                if not matching_sent:
                    continue

                if any(cls.are_same_word_family(w, sw) and matching_sent == sq for sw, sq in seen_word_quotes):
                    continue

                # Passage-First Derived Promotion:
                # If a base headword (e.g. 'enlighten', 'persevere') does NOT appear verbatim in matching_sent,
                # check the actual surface token in matching_sent (e.g. 'enlightening', 'persevering').
                # When the surface token's contextual POS in matching_sent (e.g. 'adjective' in
                # "an enlightening publication" or "They are more persevering") is absent from the base
                # headword's PedLex POS list while the verbatim surface token has its own PedLex entry
                # for that POS, promote `w` to the verbatim passage word so Gate 1 & Gate 2 evaluate the true passage form.
                if matching_sent and " " not in w and not re.search(rf"\b{re.escape(w)}\b", matching_sent, flags=re.IGNORECASE):
                    w_ent = cls.get_ldoce_entry(w)
                    w_poses = {pc for s in (w_ent.get("senses") or []) for pc in cls._sense_pos_codes(s)} if w_ent else set()
                    for surf_m in re.finditer(r"\b([a-zA-Z]{4,})\b", matching_sent):
                        surf_low = surf_m.group(1).lower()
                        if surf_low != w and cls.form_matches(surf_low, w):
                            surf_ctx_pos = cls.determine_contextual_pos(surf_low, matching_sent)
                            surf_p_code = cls._pos_code(surf_ctx_pos)
                            if surf_p_code and surf_p_code not in w_poses:
                                surf_ent = cls.get_ldoce_entry(surf_low)
                                surf_poses = {pc for s in (surf_ent.get("senses") or []) for pc in cls._sense_pos_codes(s)} if surf_ent else set()
                                if surf_p_code in surf_poses and surf_low not in seen_words:
                                    w = surf_low
                                    cand["part_of_speech"] = surf_ctx_pos
                                    break

                # Infer contextual POS
                pos = cand.get("part_of_speech")
                if (not pos or pos == "function_word") and matching_sent:
                    ctx_pos = cls.determine_contextual_pos(w, matching_sent)
                    if ctx_pos and ctx_pos != "function_word":
                        pos = ctx_pos
                if not pos or pos == "function_word":
                    ld_w = cls.get_ldoce_entry(w)
                    pos = (ld_w.get("pos") if ld_w else "") or "adverb"

                # Gate 1: Hydrate authoritative PedLex definition and authentic example via Top-K Candidate Pool + WSD
                definition = ""
                example = ""
                lock_confidence = "high"
                lock_score = 0.0
                lock_margin = 0.0
                locked_raw_sense = None

                discourse_ctx = ""
                if matching_sid and pool:
                    m_sid_num = re.match(r"S-(\d+)", str(matching_sid))
                    if m_sid_num:
                        s_num = int(m_sid_num.group(1))
                        nb_sents = [pool[f"S-{s_num + d}"] for d in (-2, -1, 1, 2) if f"S-{s_num + d}" in pool]
                        discourse_ctx = " ".join(nb_sents)

                cand_pool = cls.build_top_k_vocab_candidates(
                    w, matching_sent, pos=pos, k=4, discourse_context=discourse_ctx
                ) if matching_sent else []
                if cand_pool:
                    if len(cand_pool) > 1 and enable_llm_disambiguation:
                        chosen_idx = cls.disambiguate_via_llm(w, matching_sent, cand_pool, discourse_context=discourse_ctx)
                        chosen = cand_pool[chosen_idx]
                    else:
                        chosen = cand_pool[0]
                    definition = chosen.get("definition", "")
                    locked_raw_sense = chosen.get("raw_sense")
                    if not pos:
                        chosen_pos = str(chosen.get("pos") or "").strip().lower()
                        if chosen_pos:
                            for k, v in (("adv", "adverb"), ("noun", "noun"), ("verb", "verb"), ("adj", "adjective"), ("prep", "preposition"), ("conj", "conjunction")):
                                if k in chosen_pos:
                                    pos = v
                                    break
                    exs = chosen.get("examples") or []
                    if exs:
                        example = exs[0]
                    lock_score = chosen.get("score", 0.0)

                if not definition:
                    definition, example = cls.get_ldoce_definition_and_example(
                        w, target_pos=pos, context_sentence=matching_sent,
                        require_sense_confidence=False
                    )
                if not definition:
                    # Fallback to WordNet definition if PedLex headword missing
                    try:
                        best_s = cls.disambiguate_wordnet_synset(w, pos=pos, context_sentence=matching_sent)
                        if best_s:
                            definition = best_s.definition()
                    except Exception:
                        pass

                if pos == "adjective" and definition and re.match(r"^to\s+be\s+", definition.strip(), flags=re.IGNORECASE):
                    definition = re.sub(r"^to\s+be\s+", "", definition.strip(), flags=re.IGNORECASE)
                if definition and re.match(r"^to\s+[a-z]+\b", definition.strip().lower()) and pos in ("noun", "adjective", "adverb", "function_word"):
                    pos = "verb"
                if pos == "function_word":
                    pos = "adverb"
                # Dictionary-First Canonical Headword Normalization:
                # If a participial/inflected word (e.g. 'absorbed', 'condensed') has no standalone adjective entry
                # in the dictionary and resolves to its base verb entry (pos == 'verb'), normalize the headword
                # to the dictionary's canonical base verb ('absorb', 'condense') so headword and verb POS/definition agree.
                if pos == "verb" and " " not in w and "-" not in w:
                    ld_e = cls.get_ldoce_entry(w)
                    if ld_e:
                        dict_hw = str(ld_e.get("word") or "").strip().lower()
                        if dict_hw and dict_hw != w and cls.are_same_word_family(w, dict_hw):
                            w = dict_hw
                    else:
                        lem_v = cls._surface_lemma(w)
                        if lem_v and lem_v != w:
                            wn = cls.get_wordnet()
                            if wn and wn.synsets(lem_v, pos="v"):
                                w = lem_v
                if w in seen_words:
                    continue
                if not example and not locked_raw_sense:
                    example = cls.ldoce_sense_example(w, pos=pos, quote=matching_sent)
                example = cls.ensure_distinct_example(w, pos, example, matching_sent, sense=locked_raw_sense)

                # Gate 2 (Evidence-Completeness Gate - 宁缺勿滥):
                # Drop any candidate lacking a valid definition OR an authentic, non-duplicate dictionary example
                # for its locked sense, allowing a fully attested reserve candidate to take its slot.
                if not definition or cls.is_boilerplate_definition(definition) or not example:
                    continue

                audit_sid = f"[{matching_sid}]" if matching_sid else "[S-1]"
                results.append({
                    "word": w,
                    "part_of_speech": pos,
                    "definition": definition,
                    "quoted_sentence": matching_sent,
                    "example_usage": example,
                    "anchor_quality": anchor_quality,
                    "sense_confidence": lock_confidence,
                    "sense_lock_score": lock_score,
                    "sense_lock_margin": lock_margin,
                    "design_audit": f"AUDIT: {audit_sid} -> [{w}] -> [VERBATIM_CONFIRMED]"
                })
                seen_words.add(w)
                if matching_sent:
                    seen_word_quotes.append((w, matching_sent))

            return results

    @classmethod
    def extract_deterministic_expressions(
        cls,
        text: str,
        target_count: int = 5,
        syllabus_expressions: Optional[List[str]] = None,
        enable_llm_disambiguation: bool = True,
        exclude_words: Optional[Set[str]] = None,
        exclude_definitions: Optional[Set[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Extracts genuine multi-word academic expressions and collocations from text
        in 100% deterministic fashion (0 Tokens).
        Combines mine_expression_skeletons with get_expression_definition_and_example.
        Returns a list of dicts conforming to VocabularyItem / ExpressionItem:
            {
                "word": canonical formula (e.g. 'shut [sb/sth] down'),
                "part_of_speech": 'collocation' | 'phrasal verb' | ...,
                "definition": authoritative PedLex / WordNet definition,
                "quoted_sentence": verbatim passage sentence,
                "example_usage": authentic PedLex example or passage sentence,
                "design_audit": "AUDIT: [S-ID] -> [Formula] -> [VERBATIM_CONFIRMED]"
            }
        """
        with cls.wsd_log_session("expressions"):
            sent_pool = cls.tokenize_and_index_sentences(text)[1] if text else {}
            skeletons = cls.mine_expression_skeletons(
                text,
                target_count=max(target_count, len(syllabus_expressions)) if syllabus_expressions else target_count + 8,
                syllabus_expressions=syllabus_expressions
            )
            for sk in skeletons:
                sk["_is_syllabus"] = bool(syllabus_expressions)
            if syllabus_expressions:
                seen_ph = {re.sub(r'\[.*?\]|\(.*?\)', '', s.get("phrase", "")).strip().lower() for s in skeletons}
                reserve_exprs = cls.mine_expression_skeletons(
                    text,
                    target_count=target_count + 10,
                    syllabus_expressions=None
                )
                for rs in reserve_exprs:
                    rph = re.sub(r'\[.*?\]|\(.*?\)', '', rs.get("phrase", "")).strip().lower()
                    if rph and rph not in seen_ph:
                        seen_ph.add(rph)
                        rs["_is_syllabus"] = False
                        skeletons.append(rs)
            if not skeletons:
                return []

            excl_w_low = {str(w).strip().lower() for w in (exclude_words or set()) if w}
            excl_d_norm = {re.sub(r"[^a-z0-9]+", " ", str(d).lower()).strip() for d in (exclude_definitions or set()) if d}
            _expr_func_stop = {
                "a", "an", "the", "to", "of", "in", "on", "at", "by", "for", "with", "from", "into", "onto", "upon",
                "over", "under", "up", "down", "off", "away", "out", "back", "about", "around", "through", "along",
                "across", "against", "between", "among", "be", "have", "do", "get", "make", "take", "go", "come"
            }

            attested_sources = ("phrase_row", "unit_sense", "headword_entry", "phrasal_verb_block", "sense_pattern")
            results: List[Dict[str, Any]] = []
            seen_formulas: Set[str] = set()
            for s in skeletons:
                if len(results) >= target_count:
                    break
                phrase = s.get("phrase", "")
                formula = s.get("pattern_formula", phrase)
                norm_form = re.sub(r'\[.*?\]|\(.*?\)', '', formula).strip().lower()
                if not norm_form or norm_form in seen_formulas:
                    continue
                # Prevent verb+preposition expressions (e.g. 'conform to [sth/sb]') from duplicating
                # a single-word headword ('conform') already extracted in the same unit.
                if excl_w_low:
                    f_content_toks = [t for t in norm_form.split() if t not in _expr_func_stop]
                    if len(f_content_toks) == 1 and (
                        f_content_toks[0] in excl_w_low
                        or cls._surface_lemma(f_content_toks[0]) in excl_w_low
                    ):
                        continue
                sid = s.get("sid", "S-1")
                quote = s.get("quote", "")
                expr_type = s.get("type", "collocation")
                is_syllabus_item = bool(s.get("_is_syllabus", False))

                discourse_ctx = ""
                if sid and sent_pool:
                    m_sid_num = re.match(r"S-(\d+)", str(sid))
                    if m_sid_num:
                        s_num = int(m_sid_num.group(1))
                        nb_sents = [sent_pool[f"S-{s_num + d}"] for d in (-2, -1, 1, 2) if f"S-{s_num + d}" in sent_pool]
                        discourse_ctx = " ".join(nb_sents)

                # Primary: Ground multi-word expressions in authoritative PedLex expression evidence tiers
                # (Disable inner LLM disambiguation here since Gate 1 below arbitrates across the full unified pool)
                defn, ex, def_source = cls.expression_definition_evidence(
                    phrase=phrase,
                    expr_type=expr_type,
                    context_sentence=quote,
                    enable_llm_disambiguation=False
                )

                # Gate 1: Neuro-Symbolic Arbitration across Top-K (k=4) Expression Candidates
                locked_expr_sense = None
                if quote:
                    pool = cls.build_top_k_expression_candidates(formula, quote, k=4, expr_type=expr_type)
                    has_attested_pool = any(c.get("attested") for c in pool)
                    if not is_syllabus_item and not has_attested_pool and def_source not in attested_sources:
                        continue
                    if defn and not has_attested_pool:
                        norm_defn = re.sub(r'[^a-z0-9]+', ' ', defn.lower()).strip()
                        if not any(re.sub(r'[^a-z0-9]+', ' ', c.get("definition", "").lower()).strip() == norm_defn for c in pool):
                            pool.insert(0, {
                                "definition": defn,
                                "pos": expr_type,
                                "examples": [ex] if ex else [],
                                "score": 30.0,
                                "margin": 0.0,
                                "attested": def_source in attested_sources,
                            })
                    pool = pool[:4]
                    if pool:
                        if len(pool) > 1 and enable_llm_disambiguation:
                            chosen_idx = cls.disambiguate_via_llm(formula, quote, pool, discourse_context=discourse_ctx)
                            chosen = pool[chosen_idx] if 0 <= chosen_idx < len(pool) else pool[0]
                            if chosen.get("definition"):
                                defn = chosen["definition"]
                                def_source = def_source or "phrase_row"
                            locked_expr_sense = chosen.get("raw_sense")
                            if chosen.get("examples"):
                                ex = chosen["examples"][0]
                        elif not defn or (has_attested_pool and def_source not in attested_sources):
                            chosen = pool[0]
                            if chosen.get("definition"):
                                defn = chosen["definition"]
                                def_source = def_source or "phrase_row"
                            locked_expr_sense = chosen.get("raw_sense")
                            if chosen.get("examples"):
                                ex = chosen["examples"][0]
                        elif locked_expr_sense is None:
                            locked_expr_sense = pool[0].get("raw_sense")

                low_confidence = not def_source
                ex = cls.ensure_distinct_example(formula, expr_type, ex, quote, sense=locked_expr_sense)

                # Gate 2 (Evidence-Completeness Gate - 宁缺勿滥):
                # Drop any multi-word expression lacking an authoritative definition or distinct corpus example.
                if not defn or cls.is_boilerplate_definition(defn) or not ex:
                    continue
                if excl_d_norm and re.sub(r"[^a-z0-9]+", " ", defn.lower()).strip() in excl_d_norm:
                    continue

                seen_formulas.add(norm_form)
                results.append({
                    "word": formula,
                    "part_of_speech": expr_type,
                    "definition": defn,
                    "quoted_sentence": quote,
                    "example_usage": ex,
                    "definition_source": def_source,
                    "low_confidence": low_confidence,
                    "design_audit": f"AUDIT: [{sid}] -> [{formula}] -> [VERBATIM_CONFIRMED]"
                })

            return results

    # =========================================================================
    # DECLARATIVE 3-BAND COBUILD GRAMMAR PROFILES & HOT-RELOAD LOADER
    # Grounded in librarian/data/grammar_profiles.json (A1_A2 / B1_B2 / C1_C2)
    # =========================================================================
    _grammar_profiles_cache: Optional[Dict[str, Any]] = None
    _grammar_profiles_mtime: float = 0.0

    @classmethod
    def get_grammar_profiles_data(cls) -> Dict[str, Any]:
        """
        Loads and hot-reloads `librarian/data/grammar_profiles.json` using `st_mtime`.
        Decouples all 3-band priority weights, pedagogical functions, imitation examples,
        and common mistakes from Python source code.
        """
        json_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "grammar_profiles.json")
        try:
            mtime = os.stat(json_path).st_mtime
            if cls._grammar_profiles_cache is not None and mtime == cls._grammar_profiles_mtime:
                return cls._grammar_profiles_cache
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            cls._grammar_profiles_cache = data
            cls._grammar_profiles_mtime = mtime
            return data
        except Exception as e:
            logger.warning(f"Failed to load grammar_profiles.json ({e}); using cached or empty fallback.")
            return cls._grammar_profiles_cache or {"profiles": {}, "fallbacks": {}}

    @classmethod
    def resolve_grammar_cefr_band(cls, text_or_pool: Union[str, Dict[str, str], List[str]]) -> str:
        """
        Deterministically resolves a passage's grammar pedagogical band ('A1_A2', 'B1_B2', or 'C1_C2')
        using `calculate_text_cefr` and lexical sophistication metrics (`b2_plus_ratio`, `c1_plus_ratio`, `avg_cefr`).
        """
        if isinstance(text_or_pool, dict):
            full_text = " ".join(text_or_pool.values())
        elif isinstance(text_or_pool, list):
            full_text = " ".join(text_or_pool)
        else:
            full_text = str(text_or_pool or "")

        if not full_text.strip():
            return "B1_B2"

        cefr = cls.calculate_text_cefr(full_text)
        if cefr in ("A1", "A2"):
            return "A1_A2"
        if cefr in ("C1", "C2"):
            return "C1_C2"

        try:
            analyzer = cls.get_cefr_analyzer()
            words = re.findall(r"[a-zA-Z]+", full_text.lower())
            levels: List[float] = []
            for w in words:
                lvl = analyzer.get_average_word_level_float(w)
                if lvl is not None:
                    levels.append(float(lvl))
            if levels:
                n = len(levels)
                avg_cefr = sum(levels) / n
                b2_plus_ratio = sum(1 for x in levels if x >= 3.5) / n
                c1_plus_ratio = sum(1 for x in levels if x >= 4.5) / n
                if cefr == "B1" and avg_cefr < 1.75 and b2_plus_ratio < 0.088:
                    return "A1_A2"
                if cefr == "B2" and (avg_cefr >= 1.93 or (c1_plus_ratio >= 0.055 and avg_cefr >= 1.90)):
                    return "C1_C2"
        except Exception:
            pass

        return "B1_B2"

    @classmethod
    def match_cobuild_grammar_profile(
        cls,
        formula: str,
        quote: str,
        category: str = "",
        cefr_band: str = "B1_B2"
    ) -> Dict[str, Any]:
        """
        Deterministically matches a mined syntax formula and quote to the most specific
        COBUILD Grammar Profile from `librarian/data/grammar_profiles.json` and resolves
        the target CEFR band's (`A1_A2`, `B1_B2`, `C1_C2`) priority, pedagogical function,
        imitation example, and common mistakes (0 Token Cost).
        """
        band = cefr_band if cefr_band in ("A1_A2", "B1_B2", "C1_C2") else "B1_B2"
        data = cls.get_grammar_profiles_data()
        profiles = data.get("profiles", {})
        fallbacks = data.get("fallbacks", {})

        def _resolve_profile(profile_key: str, fallback_cat: str = "") -> Dict[str, Any]:
            raw_prof = profiles.get(profile_key)
            if not raw_prof:
                return _resolve_fallback(fallback_cat or category)
            tiers = raw_prof.get("tiers", {})
            tier_data = tiers.get(band) or tiers.get("B1_B2") or tiers.get("A1_A2") or {}
            prio_map = raw_prof.get("priority", {})
            prio_val = int(prio_map.get(band, prio_map.get("B1_B2", 70)))
            return {
                "profile_key": profile_key,
                "name": raw_prof.get("name", "Syntactic Pattern"),
                "category": raw_prof.get("category", category or "Information Packaging"),
                "pattern_formula": raw_prof.get("pattern_formula", formula),
                "priority": prio_val,
                "pedagogical_function": tier_data.get("pedagogical_function", ""),
                "imitation_example": tier_data.get("imitation_example", quote),
                "common_mistakes": tier_data.get("common_mistakes", ""),
                "variants": tier_data.get("variants", [])
            }

        def _resolve_fallback(cat_name: str) -> Dict[str, Any]:
            c_low = (cat_name or "").lower()
            if c_low == "rhetoric & emphasis":
                fb_key = "Rhetoric & Emphasis"
            elif c_low == "logic & stance":
                fb_key = "Logic & Stance"
            elif c_low == "cohesion & framing":
                fb_key = "Cohesion & Framing"
            else:
                fb_key = "Information Packaging"
            raw_fb = fallbacks.get(fb_key, {})
            tiers = raw_fb.get("tiers", {})
            tier_data = tiers.get(band) or tiers.get("B1_B2") or tiers.get("A1_A2") or {}
            prio_map = raw_fb.get("priority", {})
            prio_val = int(prio_map.get(band, 45))
            return {
                "profile_key": f"fallback_{fb_key.lower().replace(' ', '_')}",
                "name": raw_fb.get("name", f"{fb_key} Pattern"),
                "category": fb_key,
                "pattern_formula": formula,
                "priority": prio_val,
                "pedagogical_function": tier_data.get("pedagogical_function", f"Synthesizes discourse via {fb_key.lower()} syntax."),
                "imitation_example": tier_data.get("imitation_example", quote),
                "common_mistakes": tier_data.get("common_mistakes", "Grammar Warning (Syntax): Learners frequently misapply structural boundaries."),
                "variants": tier_data.get("variants", [])
            }

        f_lower = (formula or "").lower()
        q_lower = (quote or "").lower()
        c_lower = (category or "").lower()

        # 0. Explicit formula-first routing so formula and profile never diverge when a quote contains multiple clauses
        if any(f_lower.startswith(p) for p in ("had + [subject]", "were + [subject]", "should + [subject]")) or re.match(r"^(?:had|were|should)\s+(?:i|you|he|she|it|we|they|the\b|a\b|an\b|this\b|that\b|these\b|those\b|[a-z]+)\s+(?:not\s+)?(?:been|had|known|done|to\b|[a-z]+ed\b)", q_lower.strip()):
            if "inverted_conditional_subjunctive" in profiles:
                return _resolve_profile("inverted_conditional_subjunctive")
        if "negative/restrictive" in f_lower or "inversion" in f_lower or any(f_lower.startswith(p) for p in ("never +", "rarely +", "seldom +", "little +", "only +", "not only + [aux", "hardly +", "scarcely +", "no sooner +", "under no circumstances", "in no way", "at no time", "on no account")):
            return _resolve_profile("fronted_negative_inversion")
        if f_lower.startswith("with + [np]"):
            return _resolve_profile("with_absolute_adjunct")
        if f_lower.startswith("[np] + [v-ing") or f_lower.startswith("[np] + [v-ed") or "nominative absolute" in f_lower:
            if "nominative_absolute" in profiles:
                return _resolve_profile("nominative_absolute")
        if any(f_lower.startswith(p) for p in ("although +", "though +", "even though +", "even if +", "while +", "whereas +", "despite +", "in spite of +", "much as +", "albeit +", "notwithstanding +")):
            return _resolve_profile("subordinate_concessive")
        if any(f_lower.startswith(p) for p in ("as long as +", "provided that +", "on condition that +", "so long as +")):
            return _resolve_profile("conditional_clause_as_long_as")
        if any(f_lower.startswith(p) for p in ("if +", "once +", "unless +", "if/unless +")):
            return _resolve_profile("conditional_clause_if")
        if ("so + [" in f_lower or "such + [" in f_lower) and "that" in f_lower:
            return _resolve_profile("resultative_so_such_that")
        if "the + [comparative]" in f_lower:
            return _resolve_profile("proportional_comparative")
        if "as + [adj/adv] + as" in f_lower or ("as + [" in f_lower and "+ as" in f_lower):
            return _resolve_profile("correlative_comparative_as_as")
        if "by + [v-ing" in f_lower:
            return _resolve_profile("prepositional_means_by_ving")
        if "[v-ing / gerund phrase]" in f_lower or "[v-ing phrase] + [v]" in f_lower:
            return _resolve_profile("gerund_subject_nominalization")
        if "[subject] + [vp], [" in f_lower and ("v-ing" in f_lower or "having +" in f_lower):
            return _resolve_profile("participial_adjunct_post_ving")
        if "[subject] + [vp], [" in f_lower and ("v-ed" in f_lower or "being + v-ed" in f_lower):
            return _resolve_profile("participial_adjunct_post_ved")
        if any(p in f_lower for p in ("[v-ing phrase]", "[v-ed phrase]", "[having + v-ed phrase]", "[being + v-ed phrase]")):
            return _resolve_profile("participial_adjunct")
        if "passive" in f_lower or ("[be/modal be]" in f_lower and "[v-ed]" in f_lower):
            return _resolve_profile("passive_voice_agentless")
        if any(p in f_lower for p in ("in order to +", "so as to +", "to + [vp], [subject]")):
            return _resolve_profile("purpose_infinitive_adjunct")
        if f_lower.startswith("to + [vp] + [v"):
            if "infinitive_subject_nominalization" in profiles:
                return _resolve_profile("infinitive_subject_nominalization")
        if "which + [interpretive" in f_lower or f_lower.startswith(", which +"):
            if "propositional_encapsulation_which" in profiles:
                return _resolve_profile("propositional_encapsulation_which")
            return _resolve_profile("relative_clause_elaborative")
        if "[np], which" in f_lower:
            return _resolve_profile("relative_clause_elaborative")
        if "shell noun" in f_lower or "that/of + [proposition" in f_lower or "fact + that" in f_lower or "[abstract np] + that" in f_lower:
            return _resolve_profile("appositive_complement_that")
        if "focal element" in f_lower or "cleft" in f_lower:
            return _resolve_profile("cleft_focus")
        if "evaluative adj" in f_lower or "it + [be] + [evaluative" in f_lower or "it is + [adj" in f_lower or "it was + [adj" in f_lower or "dummy_it" in f_lower:
            return _resolve_profile("dummy_it_extraposition")
        if any(p in f_lower for p in ("[np] + which", "[np] + that", "[np] + who", "[np] + whom", "[np] + whose", "[np], who", "[np] , which", "[np] , who", "[np] + [prep] + which", "[np] + [prep] + whom")):
            return _resolve_profile("relative_clause_restrictive")

        # 1. Cleft Focus vs Dummy-It Extraposition (check extraposition predicates FIRST even if formula says Focal Element)
        if re.search(r"\bit(?:'s|\s+(?:was|is|has\s+been|seemed|seems|appeared|appears|turned\s+out|proved|would\s+be|could\s+be))\s+(?:no\s+|not\s+a\s+|little\s+|hardly\s+a\s+)?(?:clear|obvious|important|necessary|essential|vital|critical|imperative|crucial|possible|impossible|likely|unlikely|certain|true|evident|apparent|plain|notable|remarkable|worth\s+noting|fair|safe|reasonable|natural|normal|common|rare|strange|odd|ironic|paradoxical|surprising|unsurprising|astonishing|shocking|sad|tragic|unfortunate|regrettable|fortunate|lucky|good|better|best|wise|foolish|easy|hard|difficult|simple|complex|challenging|said|reported|believed|thought|assumed|estimated|expected|known|understood|agreed|accepted|argued|suggested|claimed|noted|observed|found|shown|proven|established|demonstrated|revealed|discovered|decided|arranged|recommended|proposed|demanded|required|requested|urged|advised|surprise|wonder|secret|doubt|coincidence|accident|pity|shame|mystery|miracle|comfort|relief|consolation|honor|honour|privilege|pleasure|joy|delight|tragedy|disgrace|scandal|fact|matter|question|issue|problem|challenge|duty|responsibility|task|job|custom|tradition|habit|rule|policy|practice|mistake|error|fault|crime|sin|blessing|curse|waste|use|good|point)\b", q_lower):
            return _resolve_profile("dummy_it_extraposition")
        if "focal element" in f_lower or "cleft" in f_lower:
            return _resolve_profile("cleft_focus")
        if re.search(r"\bit(?:'s|\s+(?:was|is))\s+.*?\b(?:that|who)\b", q_lower):
            return _resolve_profile("cleft_focus")

        # 2. Correlative Parallelism: not only... but also
        if "not only" in f_lower or ("not only" in q_lower and ("but also" in q_lower or "but " in q_lower)):
            if re.search(r"(?:^|[,;:]\s*)not\s+only\s+(?:did|do|does|had|have|has|was|were|is|are|can|could|will|would|should|must|may|might)\b", q_lower.strip()):
                return _resolve_profile("fronted_negative_inversion")
            return _resolve_profile("correlative_parallelism")

        # 2b. Antithesis Coordination: not... but... / instead of / rather than
        if "instead of +" in f_lower or "rather than +" in f_lower:
            if "prepositional_contrast_instead_of" in profiles:
                return _resolve_profile("prepositional_contrast_instead_of")
        if "not + [prepp/np], but" in f_lower or re.search(r",\s*not\b.*?\bbut\b", q_lower) or ("not + [" in f_lower and "but +" in f_lower):
            return _resolve_profile("antithesis_parallelism")

        # 3a. Causative bare infinitive: make/have/let + Object + [Bare Infinitive]
        if "bare infinitive" in f_lower:
            return _resolve_profile("causative_bare_infinitive")

        # 3b. Dummy-It Object Extraposition: make/find/think/consider + it + [Adj/NP] + to/that
        if "+ it + [" in f_lower and ("to + [vp]" in f_lower or "that + [clause]" in f_lower):
            if "dummy_it_object" in profiles:
                return _resolve_profile("dummy_it_object")

        # 3c. Object Complement / Complex Transitive (Noun Complement): make + Object + [Noun/NP]
        if ("make +" in f_lower or "find +" in f_lower or "appoint +" in f_lower or "consider +" in f_lower) and "[noun/np]" in f_lower:
            return _resolve_profile("object_complement_noun")

        # 3d. Object Complement / Complex Transitive (Adjective Complement): make/find/keep + Object + [Adj]
        if ("make +" in f_lower or "render +" in f_lower or "find +" in f_lower or "keep +" in f_lower or "leave +" in f_lower or "deem +" in f_lower) and ("[adj]" in f_lower or " + it + [" in f_lower or "[object]" in f_lower):
            return _resolve_profile("object_complement_adj")
        if re.search(r"\bmake\b.*?\b(?:possible|impossible|easy|clear|accessible|redundant)\b", q_lower):
            return _resolve_profile("object_complement_adj")

        # 4. Causative bare infinitive fallback: make/have/let + sb + do
        if "make + [" in f_lower or "let + [" in f_lower:
            return _resolve_profile("causative_bare_infinitive")

        # 5. Shell Noun / Appositive That-Clause: [Shell Noun] + that + [Clause]
        if "shell noun" in f_lower or "fact + that" in f_lower or "[abstract np] + that" in f_lower or re.search(r"\bthe\s+(?:fact|idea|belief|notion|view|possibility|chance|hope|fear|news|evidence|proof|assumption|hypothesis|premise|paradox|irony|conviction|realization|consensus|contention|proposition|myth|delusion|impression)\s+that\b", q_lower):
            return _resolve_profile("appositive_complement_that")

        # 5b. Participial Adjuncts (differentiating post-positioned V-ed vs V-ing, checked before relative clauses)
        if "[subject] + [vp], [" in f_lower and ("v-ing" in f_lower or "having +" in f_lower):
            return _resolve_profile("participial_adjunct_post_ving")
        if "[subject] + [vp], [" in f_lower and ("v-ed" in f_lower or "being + v-ed" in f_lower):
            return _resolve_profile("participial_adjunct_post_ved")
        if any(p in f_lower for p in ("[v-ing phrase]", "[v-ed phrase]", "[having + v-ed phrase]", "[being + v-ed phrase]")) or "particip" in c_lower:
            return _resolve_profile("participial_adjunct")

        # 6. Non-restrictive / Sentential Relative Clauses
        if "which + [interpretive" in f_lower or re.search(r",\s*which\s+(?:means|suggests|indicates|shows|implies|demonstrates|proves|reveals|reflects|underscores|highlights|explains|makes|leaves|renders|allows|enables|prevents|leads|results|contributes)\b", q_lower):
            if "propositional_encapsulation_which" in profiles:
                return _resolve_profile("propositional_encapsulation_which")
            return _resolve_profile("relative_clause_elaborative")
        if (", which" in q_lower and "fact + that" not in f_lower and "[shell noun]" not in f_lower) or "[np], which" in f_lower:
            return _resolve_profile("relative_clause_elaborative")

        # 7. Restrictive & Human Relative Clauses: [NP] + which/that/who + [VP] or Prep + which/whom
        if any(p in f_lower for p in ("[np] + which", "[np] + that", "[np] + who", "[np] + whom", "[np] + whose", "[np], who", "[np] , which", "[np] , who", "[np] + [prep] + which", "[np] + [prep] + whom")):
            return _resolve_profile("relative_clause_restrictive")

        # 8. Proportional Comparative & Equative Comparative (as ... as)
        if "the + [comparative]" in f_lower:
            return _resolve_profile("proportional_comparative")
        if "as + [adj/adv] + as" in f_lower or ("as + [" in f_lower and "+ as" in f_lower):
            return _resolve_profile("correlative_comparative_as_as")

        # 8b. Resultative Degree Constructions (so ... that / such ... that)
        if ("so + [" in f_lower or "such + [" in f_lower) and "that" in f_lower:
            return _resolve_profile("resultative_so_such_that")

        # 9. Dummy-It Extraposition: It is adj to/that
        if "it + [be] + [adj" in f_lower or "it is + [adj" in f_lower or "dummy" in f_lower:
            return _resolve_profile("dummy_it_extraposition")

        # 10. Purpose Infinitive Adjuncts (In order to / So as to / To + [VP]) & Infinitive Subject
        if any(p in f_lower for p in ("in order to +", "so as to +", "to + [vp], [subject]")):
            return _resolve_profile("purpose_infinitive_adjunct")
        if f_lower.startswith("to + [vp] + [v"):
            if "infinitive_subject_nominalization" in profiles:
                return _resolve_profile("infinitive_subject_nominalization")

        # 12. Gerund Subject Nominalization & Prepositional Means
        if "by + [v-ing" in f_lower or q_lower.startswith("by "):
            return _resolve_profile("prepositional_means_by_ving")
        if "[v-ing / gerund phrase]" in f_lower or "[v-ing phrase] + [v]" in f_lower:
            return _resolve_profile("gerund_subject_nominalization")

        # 13. Adversative & Concession
        if "but [subject] + [also]" in f_lower or ", but" in q_lower:
            return _resolve_profile("adversative_concession")
        if ", while + " in f_lower or ", whereas + " in f_lower or (any(w in q_lower for w in (", while ", "; while ", ", whereas ", "; whereas ")) and not q_lower.startswith(("while ", "whereas ", "but while "))):
            return _resolve_profile("subordinate_contrastive_while")
        if any(w in f_lower for w in ("although", "though", "while", "even if", "whereas")) or any(q_lower.startswith(w) for w in ("although ", "though ", "even though ", "even if ", "while ", "whereas ")):
            return _resolve_profile("subordinate_concessive")

        # 14. Discourse transition: However / Nevertheless / Nonetheless / Conversely / Instead
        if any(q_lower.startswith(w) or w in f_lower for w in ("however", "nevertheless", "nonetheless", "conversely", "in contrast", "instead,")):
            return _resolve_profile("discourse_transition_however")

        # 15. Causal & Explanatory Clauses (because / since / as / now that)
        if any(w in f_lower for w in ("because", "since + [clause]", "now that", "[causal clause]", "given that", "in that", "seeing that")):
            return _resolve_profile("explanatory_causal_clause")

        # 16. Comparative Manner & Analogy Clauses (just as / as if / as though)
        if any(w in f_lower for w in ("just as", "as if", "as though", "as + [clause]")):
            return _resolve_profile("manner_similarity_clause")

        # 17. Epistemic Stance & Hedging (suggest / indicate / show / argue that)
        if "stance" in f_lower or "epistemic" in f_lower or ("[verb] + that + [clause]" in f_lower and c_lower == "logic & stance"):
            return _resolve_profile("epistemic_stance_hedging")

        # 18. Passive voice
        if "passive" in f_lower or "passive" in c_lower or ("[be/modal be]" in f_lower and "[v-ed]" in f_lower):
            return _resolve_profile("passive_voice_agentless")

        # 19. Conditional clauses (if / as long as / unless / once)
        if "as long as" in f_lower or "as long as" in q_lower or "provided that" in f_lower or "on condition that" in f_lower or "so long as" in f_lower:
            return _resolve_profile("conditional_clause_as_long_as")
        if f_lower.startswith("if +") or f_lower.startswith("once +") or f_lower.startswith("unless +") or "if/unless" in f_lower or (q_lower.startswith("if ") or ", if " in q_lower or "so, if " in q_lower or "similarly, if " in q_lower):
            return _resolve_profile("conditional_clause_if")

        # Dynamic Category-Aware Fallback from grammar_profiles.json (3-Band Calibrated)
        return _resolve_fallback(category)

    @classmethod
    def _select_profile_variant(
        cls,
        profile: Dict[str, Any],
        quote: str,
        full_text: str,
        used_imitations: Set[str],
        used_mistakes: Set[str],
        formula: str = ""
    ) -> Tuple[str, str]:
        """
        Deterministically selects the best contextually-aligned (imitation_example, common_mistakes)
        variant from a profile's thematic bank based on lexical overlap with the quote and passage text,
        and structural alignment with the pattern formula's literal anchors.
        Guarantees band-calibrated examples and prevents duplicate templates.
        """
        default_imit = profile.get("imitation_example", quote)
        default_mist = profile.get("common_mistakes", "Grammar Warning (Syntax): Learners frequently misapply structural boundaries.")
        variants = profile.get("variants")

        # Extract literal anchor groups from formula (e.g. 'Although' or 'that/to')
        anchor_groups: List[List[str]] = []
        if formula:
            cobuild_pos = {
                "np", "vp", "v", "be", "aux", "v-ed", "v3", "v-ing", "to-v", "to-inf", "inf",
                "adj", "adv", "det", "prep", "conj", "clause", "s", "n", "pron", "modal"
            }
            raw_lits = re.findall(r"[a-zA-Z0-9_\'/]+", re.sub(r"\[.*?\]|\(.*?\)", " ", formula))
            for lit in raw_lits:
                alts = [a.strip().lower() for a in lit.split("/") if a.strip()]
                valid_alts = [a for a in alts if len(a) >= 2 and a not in cobuild_pos]
                if valid_alts:
                    anchor_groups.append(valid_alts)

        f_low = (formula or "").lower()
        requires_ving = ("[v-ing]" in f_low or "[v-ing phrase]" in f_low or "[v-ing / gerund phrase]" in f_low) and ("/" not in f_low.split("[v-ing")[1].split("]")[0] or "gerund" in f_low) if "[v-ing" in f_low else False

        def _instantiates_anchors(imit_text: str) -> bool:
            if not imit_text:
                return True
            imit_low = imit_text.lower()
            if requires_ving and not re.search(r"\b(?:[a-z]{3,}ing|being|doing|going)\b", imit_low):
                return False
            if not anchor_groups:
                return True
            imit_toks = set(re.findall(r"[a-z0-9']+", imit_low))
            for alts in anchor_groups:
                if any(a in imit_toks or cls.text_contains_form(imit_low, a) for a in alts):
                    continue
                # Closed-paradigm equivalence matching evaluator.py
                if any(re.match(r"^(?:mean[st]?|suggest(?:s|ed)?|indicat(?:es|ed)|show(?:s|ed|n)?|demonstrat(?:es|ed)|prov(?:es|ed|en)|impl(?:ies|ied)|reveal(?:s|ed)?|underscor(?:es|ed))$", a) for a in alts):
                    if re.search(r"\b(?:mean[st]?|suggest(?:s|ed)?|indicat(?:es|ed)|show(?:s|ed|n)?|demonstrat(?:es|ed)|prov(?:es|ed|en)|impl(?:ies|ied)|reveal(?:s|ed)?|underscor(?:es|ed))\b", imit_low):
                        continue
                if any(a in ("although", "though", "even", "while", "whereas") for a in alts):
                    if re.search(r"\b(?:although|though|even though|even if|while|whereas)\b", imit_low):
                        continue
                if any(a in ("fact", "idea", "belief", "assumption", "notion", "claim", "possibility", "view", "evidence", "news", "impression", "argument", "thought", "hope", "fear", "doubt", "contention", "recognition", "realization", "premise", "hypothesis", "consensus", "conclusion") for a in alts):
                    if re.search(r"\b(?:fact|idea|belief|assumption|notion|claim|possibility|view|evidence|news|impression|argument|thought|hope|fear|doubt|contention|recognition|realization|premise|hypothesis|consensus|conclusion)\b", imit_low):
                        continue
                if any(a in ("however", "nevertheless", "nonetheless", "conversely") for a in alts):
                    if re.search(r"\b(?:however|nevertheless|nonetheless|conversely)\b", imit_low):
                        continue
                return False
            return True

        if not variants:
            return default_imit, default_mist

        q_words = set(re.findall(r"[a-z]{3,}", (quote or "").lower()))
        t_words = set(re.findall(r"[a-z]{3,}", (full_text or "").lower()))

        best_var = None
        best_score = -1000
        for var in variants:
            kws = set(var.get("keywords", ()))
            # Quote keyword match carries 5x weight; passage theme match carries 1x weight
            score = len(kws & q_words) * 5 + len(kws & t_words)
            v_imit = var.get("imitation_example", default_imit)
            v_mist = var.get("common_mistakes", default_mist)
            if _instantiates_anchors(v_imit):
                score += 50
            else:
                score -= 200
            if v_imit in used_imitations or v_mist in used_mistakes:
                score -= 300
            if score > best_score:
                best_score = score
                best_var = var

        default_already_used = (default_imit in used_imitations) or (default_mist in used_mistakes)
        default_instantiates = _instantiates_anchors(default_imit)
        if best_var:
            v_imit_best = best_var.get("imitation_example", default_imit)
            v_mist_best = best_var.get("common_mistakes", default_mist)
            var_instantiates = _instantiates_anchors(v_imit_best)
            var_unused = (v_imit_best not in used_imitations) and (v_mist_best not in used_mistakes)
            if (
                best_score >= 0
                or (default_already_used and var_unused and var_instantiates)
                or (not default_instantiates and var_instantiates)
            ):
                return v_imit_best, v_mist_best
        return default_imit, default_mist

    @classmethod
    def extract_deterministic_grammar(
        cls,
        text: str,
        sentence_pool: Optional[Dict[str, str]] = None,
        target_count: int = 5,
        syllabus_grammar: Optional[List[str]] = None,
        cefr_band: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Deterministically extracts grammar patterns from the text at 0 token cost,
        calibrating pattern priority, pedagogical functions, imitation examples, and
        empirical common mistakes to the passage's 3-Band CEFR level ('A1_A2', 'B1_B2', 'C1_C2').
        """
        if not sentence_pool:
            _, sentence_pool = cls.tokenize_and_index_sentences(text)

        resolved_band = cefr_band or cls.resolve_grammar_cefr_band(text if text else sentence_pool)

        skeletons = cls.mine_grammar_skeletons(
            sentence_pool,
            target_count=target_count,
            syllabus_grammar=syllabus_grammar,
            cefr_band=resolved_band
        )
        if not skeletons:
            return []

        results: List[Dict[str, Any]] = []
        used_imitations: Set[str] = set()
        used_mistakes: Set[str] = set()
        for s in skeletons:
            sid = s.get("sid", "S-1")
            quote = s.get("quote", "")
            cat = s.get("category", "Information Packaging")
            formula = s.get("pattern_formula", "[Subject] + [VP]")

            profile = cls.match_cobuild_grammar_profile(
                formula,
                quote,
                category=cat,
                cefr_band=resolved_band
            )

            prof_name = profile.get("name", "Syntactic Pattern")
            prof_cat = profile.get("category", cat)
            if profile.get("profile_key") == "fronted_negative_inversion" and "aux" not in formula.lower():
                formula = "Not only + [Aux] + [Subject] + [VP], but + [Subject] + [also] + [VP]"
            elif profile.get("profile_key") == "dummy_it_extraposition":
                formula = profile.get("pattern_formula", formula)
            ped_func = profile.get("pedagogical_function", f"Synthesizes discourse via {prof_cat.lower()} syntax.")

            imitation, clean_mistake = cls._select_profile_variant(
                profile,
                quote=quote,
                full_text=text,
                used_imitations=used_imitations,
                used_mistakes=used_mistakes,
                formula=formula
            )

            # PedLex Authentic Grammar Alert & Don't Say diagnostics ONLY for generic fallback profiles
            try:
                STOP_WORDS_FOR_ALERTS = {"that", "not", "also", "it", "which", "this", "what", "there", "long", "like", "well", "such"}
                GENERIC_PROFILES = {
                    "Syntactic Pattern",
                    "Rhetorical Emphasis & Contrast",
                    "Logical Transition & Epistemic Stance",
                    "Discursive Cohesion & Structural Framing",
                    "Information Packaging & Syntactic Integration"
                }

                is_specific_prof = prof_name not in GENERIC_PROFILES and bool(clean_mistake)

                if not is_specific_prof:
                    formula_clean = re.sub(r'\[.*?\]|\(.*?\)', ' ', formula)
                    formula_lits = [tok.lower() for tok in re.findall(r"\b[a-zA-Z]{3,}\b", formula_clean) if tok.lower() not in STOP_WORDS_FOR_ALERTS]

                    quote_lits: List[str] = []
                    if not formula_lits:
                        quote_lits = [tok.lower() for tok in re.findall(r"\b[a-zA-Z]{3,}\b", quote)
                                      if tok.lower() in ("make", "want", "enjoy", "too", "enough", "such", "need", "own", "used", "help", "prefer", "suggest", "while", "if", "because", "unless", "since", "both", "neither")]

                    ordered_candidates = []
                    seen_cands = set()
                    for tok in formula_lits + quote_lits:
                        if tok not in seen_cands:
                            seen_cands.add(tok)
                            ordered_candidates.append(tok)

                    for aw in ordered_candidates:
                        alert = cls.get_ldoce_grammar_alert(aw, quote=quote)
                        if alert and ("Don't say" in alert.get("content", "") or "✗" in alert.get("content", "")):
                            raw_content = alert.get("content", "")
                            parts = [p.strip() for p in re.split(r'[•\n]+', raw_content) if p.strip()]
                            pitfalls = [p for p in parts if ("Don't say" in p or "✗" in p)]
                            for pit in pitfalls:
                                chosen = re.sub(r'\s+', ' ', pit).strip()
                                chosen = re.sub(r'^GRAMMAR:?\s*', '', chosen, flags=re.IGNORECASE).strip()
                                chosen = re.sub(r'\s*Grammar guide\s*[-‒–—].*$', '', chosen, flags=re.IGNORECASE).strip()
                                cand_text = f"Grammar Warning ({alert.get('word', '').upper()}): {chosen}"
                                if cand_text not in used_mistakes:
                                    clean_mistake = cand_text
                                    break
                        if clean_mistake not in used_mistakes:
                            break

                used_imitations.add(imitation)
                used_mistakes.add(clean_mistake)
            except Exception:
                used_imitations.add(imitation)
                used_mistakes.add(clean_mistake)

            results.append({
                "quote": quote,
                "pattern_formula": formula,
                "category": prof_cat,
                "syntax_topic": prof_name,
                "pedagogical_function": ped_func,
                "imitation_example": imitation,
                "common_mistakes": clean_mistake,
                "cefr_band": resolved_band,
                "design_audit": f"AUDIT: [{sid}] -> [{formula}] -> [BAND:{resolved_band}] -> [VERBATIM_CONFIRMED]"
            })

        return results
