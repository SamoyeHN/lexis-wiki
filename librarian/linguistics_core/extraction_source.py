"""
Deterministic source text hygiene, inline quote normalization, sentence tokenization, and sentence indexing pool.
"""

import datetime
import hashlib
import json
import logging
import os
import re
import sqlite3
import threading
import zlib
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

logger = logging.getLogger("librarian.linguistics")


class ExtractionSourceMixin:
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
          - Normalizes ellipsis ('…', '. . .') -> '...'
          - Converts full-width CJK punctuation ('！', '？', '，', '；', '：', '（', '）', '［', '］')
            to standard ASCII punctuation with canonical English spacing
          - Normalizes curly double quotes ('“', '”', '„', '‟', '＂', '«', '»') -> '"'
            and curly single quotes / apostrophes ('‘', '’', '‚', '‛', '′', '＇') -> "'"
          - When a line has balanced straight double quotes ('"'), uses quote parity
            (even = opening quote, odd = closing quote) to deterministically fix
            stuck quotes ('me."Pose' -> 'me. "Pose') and inverted quote spaces
            ('ached." I' -> 'ached. "I', 'responded," The' -> 'responded, "The',
            '" upright "' -> '"upright"').
        """
        if not s:
            return s
        # 0. Invisible / full-width whitespace normalization
        s = s.replace("\u3000", " ").replace("\u00a0", " ").replace("\u200b", "").replace("\ufeff", "")

        # 1. Mojibake GBK em-dash repair and ellipsis normalization
        s = s.replace("\ufffdC", "—")
        s = s.replace("…", "...")
        s = re.sub(r"\.\s+\.\s+\.", "...", s)

        # 2. Full-width punctuation -> ASCII punctuation with canonical spacing
        if any(ch in s for ch in ("！", "？", "，", "；", "：", "（", "）", "［", "］", "。")):
            s = re.sub(r"\s*（\s*", " (", s)
            s = re.sub(r"\s*）\s*", ") ", s)
            s = re.sub(r"\s*［\s*", " [", s)
            s = re.sub(r"\s*］\s*", "] ", s)
            s = re.sub(r"\s*！\s*", "! ", s)
            s = re.sub(r"\s*？\s*", "? ", s)
            s = re.sub(r"\s*，\s*", ", ", s)
            s = re.sub(r"\s*；\s*", "; ", s)
            s = re.sub(r"\s*：\s*", ": ", s)
            s = re.sub(r"([A-Za-z0-9'\"”’\)\]])\s*。\s*", r"\1. ", s)
            # Clean up space before closing punctuation or between closing paren and punctuation
            s = re.sub(r"\)\s+([.,!?;:])", r")\1", s)
            s = re.sub(r"\]\s+([.,!?;:])", r"]\1", s)

        # 3. Curly quote spacing normalization (while directional open/close distinction is still intact)
        s = re.sub(r"([“«„‟])\s+", r"\1", s)
        s = re.sub(r"\s+([”»])", r"\1", s)
        s = re.sub(r"(\w|[.!?,:;])([“«„‟])", r"\1 \2", s)
        s = re.sub(r"([”»])(\w)", r"\1 \2", s)
        s = re.sub(r"(^|[\s(\[{])‘\s+", r"\1‘", s)
        s = re.sub(r"\s+’(?=[\s.,!?;:)\]}]|$)", r"’", s)

        # 4. Unify curly quotes / apostrophes to straight ASCII '"' and "'"
        for ch in ("“", "”", "„", "‟", "″", "＂", "«", "»"):
            if ch in s:
                s = s.replace(ch, '"')
        for ch in ("‘", "’", "‚", "‛", "′", "＇"):
            if ch in s:
                s = s.replace(ch, "'")

        # 5. Parity-based straight double-quote ('"') spacing repair (only when '"' count is even > 0)
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
                    if cur_out and cur_out[0] not in (".", ",", "!", "?", ";", ":", ")", "]", "}", "—", "–", "'"):
                        parts[i] = " " + cur_out
                    else:
                        parts[i] = cur_out
            s = '"'.join(parts)

        # 6. OCR / copy-paste glued number-unit tokens ('15minute' -> '15-minute', '30minutes' -> '30 minutes')
        s = re.sub(
            r"\b(\d+)(minute|second|hour|day|week|month|year|page|word|step|foot|inch|mile)\b",
            r"\1-\2",
            s,
        )
        s = re.sub(
            r"\b(\d+)(minutes|seconds|hours|days|weeks|months|years|pages|words|steps|feet|inches|miles)\b",
            r"\1 \2",
            s,
        )

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

        text_norm = text.replace("\r\n", "\n").replace("\r", "\n")
        fm_match = re.match(r"^(---\s*\n.*?\n---\s*\n)(.*)", text_norm, re.DOTALL)
        frontmatter = fm_match.group(1) if fm_match else ""
        body = fm_match.group(2) if fm_match else text_norm

        syllabus_suffix = ""
        m_appendix = re.search(r"(?mi)^(?:#+\s*(?:Syllabus\s+Vocabulary|Appendix)\b.*)$", body)
        if m_appendix:
            syllabus_suffix = body[m_appendix.start():]
            body = body[:m_appendix.start()]

        nlp = cls.get_spacy()
        raw_lines = body.split("\n")
        norm_lines: List[str] = []

        for line in raw_lines:
            s = line.replace("\u00a0", " ").replace("\t", " ")
            s = re.sub(r" {2,}", " ", s).strip()
            if not s:
                norm_lines.append("")
                continue
            if s.startswith("#"):
                m_h = re.match(r"^(#+\s*)(.*)", s)
                if m_h:
                    s = m_h.group(1) + cls._normalize_inline_quotes_and_dashes(m_h.group(2))
            elif not re.match(r"^[-*_]{3,}\s*$", s):
                s = cls._normalize_inline_quotes_and_dashes(s)
            norm_lines.append(s)

        def _is_structural_line(line_str: str) -> bool:
            if not line_str:
                return True
            if line_str.startswith("#") or re.match(r"^[-*_]{3,}\s*$", line_str):
                return True
            if re.match(r"^(?:[\u2022\u00b7\u25aa\u25ab\*\-]\s+|\(?\d+[\.\)]\s+|\(?[a-zA-Z][\.\)]\s+)", line_str):
                return True
            if cls._is_byline_or_meta_line(line_str):
                return True
            return False

        def _is_standalone_title_line(line_str: str, prev_is_break: bool, next_str: str) -> bool:
            if not prev_is_break or not line_str or _is_structural_line(line_str):
                return False
            if re.match(
                r"^(?:Paragraph\s+|Para\.?\s+|Section\s+)?[\(\[]?(?:[A-Z]|\d{1,2})[\)\]\.]?$",
                line_str,
                re.IGNORECASE,
            ):
                return False
            if line_str.endswith((".", "!", "?", ",", ";", ":", "—", "–", "-", "…", '"', "”", "’", "'")):
                return False
            words = line_str.split()
            if not (1 <= len(words) <= 10 and len(line_str) <= 75):
                return False
            if next_str and not re.match(r'^[“"‘\'\(\[]?[A-Z0-9]', next_str):
                return False
            alpha_4 = [re.sub(r"^[^a-zA-Z]+|[^a-zA-Z]+$", "", w) for w in words]
            alpha_4 = [w for w in alpha_4 if len(w) >= 4 and w.isalpha()]
            cap_ratio = (
                sum(1 for w in alpha_4 if w[0].isupper()) / len(alpha_4)
                if alpha_4 else 1.0
            )
            if cap_ratio >= 0.60:
                return True
            doc_t = nlp(line_str)
            has_fv = any(
                t.pos_ in ("VERB", "AUX") and t.tag_ in ("VBD", "VBP", "VBZ", "MD", "VB")
                for t in doc_t
            )
            return not has_fv

        def _ends_with_sentence_terminal(line_str: str) -> bool:
            s_strip = line_str.strip()
            if not re.search(r'(?:\.|\!|\?|\.\.\.)["\'”’\)\]]*$', s_strip):
                return False
            if re.search(r'\b(?:Mr|Mrs|Ms|Dr|Prof|St|Jr|Sr|vs|etc|e\.g|i\.e|Fig|No)\.$', s_strip, re.IGNORECASE):
                return False
            return True

        def _is_valid_merged_word(w_low: str) -> bool:
            if cls.get_ldoce_entry(w_low):
                return True
            try:
                from wordfreq import zipf_frequency
                if zipf_frequency(w_low, "en") >= 1.5:
                    return True
            except Exception:
                pass
            try:
                wn = cls.get_wordnet()
                if wn and wn.words(w_low):
                    return True
            except Exception:
                pass
            return False

        def _join_wrapped_lines(prev_text: str, next_text: str) -> str:
            if re.search(r"[a-zA-Z]-$", prev_text) and re.match(r"^[a-z]", next_text):
                m_w1 = re.search(r"([a-zA-Z]+)-$", prev_text)
                m_w2 = re.match(r"^([a-z]+)", next_text)
                if m_w1 and m_w2:
                    merged_word = (m_w1.group(1) + m_w2.group(1)).lower()
                    if _is_valid_merged_word(merged_word):
                        return prev_text[:-1] + next_text
            m_pref = re.search(
                r"(?:^|\s)(dis|un|pre|mis|sub|inter|counter|anti|non|semi|trans|hyper|ultra|infra|intra|multi|micro|macro|auto)$",
                prev_text,
                re.IGNORECASE,
            )
            if m_pref and re.match(r"^[a-z]{3,}", next_text):
                m_w2 = re.match(r"^([a-z]+)", next_text)
                if m_w2:
                    merged_word = (m_pref.group(1) + m_w2.group(1)).lower()
                    if _is_valid_merged_word(merged_word):
                        return prev_text + next_text
            return f"{prev_text} {next_text}"

        # Heal inline broken prefix splits (e.g. 'dis orientate' -> 'disorientate') produced by prior unwraps
        _pref_rx = re.compile(
            r"\b(dis|un|pre|mis|sub|inter|counter|anti|non|semi|trans|hyper|ultra|infra|intra|multi|micro|macro|auto)\s+([a-z]{4,})\b"
        )
        for idx_p, s_p in enumerate(norm_lines):
            if not s_p or _is_structural_line(s_p):
                continue
            def _heal_pref_match(m: re.Match) -> str:
                merged_w = (m.group(1) + m.group(2)).lower()
                if _is_valid_merged_word(merged_w):
                    return m.group(1) + m.group(2)
                return m.group(0)
            norm_lines[idx_p] = _pref_rx.sub(_heal_pref_match, s_p)

        # Pre-scan body prose lines (after any leading headings/standalone title) to detect
        # column-wrap width and whether explicit blank lines already separate paragraphs.
        first_prose_idx = 0
        for idx_scan, s_scan in enumerate(norm_lines):
            if not s_scan or _is_structural_line(s_scan):
                continue
            nxt_scan = next((x for x in norm_lines[idx_scan + 1:] if x), "")
            if idx_scan == 0 or all(not x or _is_structural_line(x) for x in norm_lines[:idx_scan]):
                if _is_standalone_title_line(s_scan, True, nxt_scan):
                    first_prose_idx = idx_scan + 1
                    break
            first_prose_idx = idx_scan
            break

        # Recover partially-unwrapped inline sequential paragraph labels (e.g., when paragraphs 1..K were
        # previously stripped of 'A'..'C' and paragraph K still contains inline '. D A major...' and '. E The...').
        prose_para_indices = [
            i for i in range(first_prose_idx, len(norm_lines))
            if norm_lines[i] and not _is_structural_line(norm_lines[i])
            and (i == first_prose_idx or not norm_lines[i - 1])
        ]
        if len(prose_para_indices) >= 2:
            last_p_idx = prose_para_indices[-1]
            expected_next_ord = len(prose_para_indices) + 1
            if expected_next_ord <= 26:
                cur_line = norm_lines[last_p_idx]
                split_segments: List[str] = []
                next_ord = expected_next_ord
                while next_ord <= 26:
                    next_letter = chr(ord("A") + next_ord - 1)
                    m_mid = re.search(
                        rf'([.!?]["\'”’\)\]]*)\s+{next_letter}\s+((?:[A-Z][a-z]+|[A-Z]{{2,}}|[AI]\s+[a-zA-Z]|[\'"“‘\(\[][A-Z0-9]|\d+[a-zA-Z0-9,\.]*).*)',
                        cur_line,
                    )
                    if not m_mid:
                        break
                    before_part = cur_line[:m_mid.end(1)].strip()
                    after_part = m_mid.group(2).strip()
                    if before_part and after_part:
                        split_segments.append(before_part)
                        cur_line = after_part
                        next_ord += 1
                    else:
                        break
                if split_segments:
                    split_segments.append(cur_line)
                    expanded_tail: List[str] = []
                    for seg_i, seg_txt in enumerate(split_segments):
                        if seg_i > 0:
                            expanded_tail.append("")
                        expanded_tail.append(seg_txt)
                    norm_lines = norm_lines[:last_p_idx] + expanded_tail + norm_lines[last_p_idx + 1:]

        # Detect and strip sequential paragraph labels ('A', 'B', 'C'... or '1', '2', '3'...)
        # across candidate paragraph-start positions (inline 'A Memory...', '1. Memory...', '(A) Memory...'
        # or standalone lines 'A', '1', 'Paragraph A', 'Paragraph 1').
        def _match_para_marker(line_str: str) -> Optional[Tuple[str, int, str]]:
            m_stand = re.match(
                r"^(?:Paragraph\s+|Para\.?\s+|Section\s+)?[\(\[]?([A-Z]|\d{1,2})[\)\]\.:]?$",
                line_str,
                re.IGNORECASE,
            )
            if m_stand:
                tok = m_stand.group(1)
                if tok.isdigit():
                    val = int(tok)
                    if 1 <= val <= 50:
                        return ("num", val, "")
                elif len(tok) == 1 and tok.isupper():
                    return ("alpha", ord(tok) - ord("A") + 1, "")
            m_inl = re.match(
                r"^(?:(?:Paragraph|Para\.?|Section)\s+([A-Z]|\d{1,2})[\.\):\-]?\s+|[\(\[]([A-Z]|\d{1,2})[\)\]]\s+|([A-Z]|\d{1,2})[\.\)]\s+|([A-Z]|\d{1,2})\s+)((?:[A-Z][a-z]+|[A-Z]{2,}|[AI]\s+[a-zA-Z]|['\"“‘\(\[][A-Z0-9]|\d+[a-zA-Z0-9,\.]*).*)$",
                line_str,
            )
            if m_inl:
                tok = m_inl.group(1) or m_inl.group(2) or m_inl.group(3) or m_inl.group(4)
                rem = m_inl.group(5).strip()
                if tok.isdigit():
                    val = int(tok)
                    if 1 <= val <= 50:
                        return ("num", val, rem)
                elif len(tok) == 1 and tok.isupper():
                    return ("alpha", ord(tok) - ord("A") + 1, rem)
            return None

        cand_markers: List[Tuple[int, str, int, str]] = []
        for idx_m in range(first_prose_idx, len(norm_lines)):
            s_m = norm_lines[idx_m]
            if not s_m:
                continue
            prev_non_empty_idx = -1
            for p_idx in range(idx_m - 1, first_prose_idx - 1, -1):
                if norm_lines[p_idx]:
                    prev_non_empty_idx = p_idx
                    break
            is_para_start_cand = (
                prev_non_empty_idx == -1
                or (prev_non_empty_idx < idx_m - 1)
                or _ends_with_sentence_terminal(norm_lines[prev_non_empty_idx])
                or _is_structural_line(norm_lines[prev_non_empty_idx])
                or bool(
                    re.match(
                        r"^(?:Paragraph\s+|Para\.?\s+|Section\s+)?[\(\[]?(?:[A-Z]|\d{1,2})[\)\]\.:]?$",
                        norm_lines[prev_non_empty_idx],
                        re.IGNORECASE,
                    )
                )
            )
            if not is_para_start_cand:
                continue
            parsed_m = _match_para_marker(s_m)
            if parsed_m:
                cand_markers.append((idx_m, parsed_m[0], parsed_m[1], parsed_m[2]))

        labeled_para_starts: Set[int] = set()
        for scheme in ("alpha", "num"):
            seq_chain: List[Tuple[int, str]] = []
            expected_ord = 1
            for idx_m, sc, ord_val, rem in cand_markers:
                if sc == scheme and ord_val == expected_ord:
                    seq_chain.append((idx_m, rem))
                    expected_ord += 1
            if len(seq_chain) >= 2:
                for idx_m, rem in seq_chain:
                    norm_lines[idx_m] = rem
                    if rem:
                        labeled_para_starts.add(idx_m)
                    else:
                        for nxt_i in range(idx_m + 1, len(norm_lines)):
                            if norm_lines[nxt_i]:
                                labeled_para_starts.add(nxt_i)
                                break
                break

        body_prose_Slice = norm_lines[first_prose_idx:]
        body_blank_breaks = 0
        seen_prose_in_body = False
        in_blank_run = False
        for s_scan in body_prose_Slice:
            if not s_scan:
                if seen_prose_in_body:
                    in_blank_run = True
            elif not _is_structural_line(s_scan):
                if in_blank_run:
                    body_blank_breaks += 1
                    in_blank_run = False
                seen_prose_in_body = True

        non_term_lens = [
            len(s_scan)
            for s_scan in body_prose_Slice
            if s_scan and not _is_structural_line(s_scan) and not _ends_with_sentence_terminal(s_scan) and len(s_scan) >= 40
        ]
        global_wrap = sorted(non_term_lens)[int(len(non_term_lens) * 0.5)] if non_term_lens else 80

        # Unwrap manual mid-sentence line breaks within paragraphs while preserving real paragraph breaks
        unwrapped_lines: List[str] = []
        para_buf: List[str] = []

        def _flush_para_buf() -> None:
            if not para_buf:
                return
            merged = para_buf[0]
            for seg in para_buf[1:]:
                merged = _join_wrapped_lines(merged, seg)
            merged = cls._normalize_inline_quotes_and_dashes(merged)
            # Check if an accidental blank line split a single sentence in half
            if (
                len(unwrapped_lines) >= 2
                and unwrapped_lines[-1] == ""
                and unwrapped_lines[-2]
                and not _is_structural_line(unwrapped_lines[-2])
                and not unwrapped_lines[-2].endswith((".", "!", "?", '"', "”", "’", "'", ":", ";", ")", "]", "…"))
                and (unwrapped_lines[-2].endswith(",") or re.match(r"^[a-z]", merged))
            ):
                prev_para = unwrapped_lines[-2]
                unwrapped_lines.pop()  # remove blank line
                unwrapped_lines[-1] = cls._normalize_inline_quotes_and_dashes(
                    _join_wrapped_lines(prev_para, merged)
                )
            else:
                unwrapped_lines.append(merged)
            para_buf.clear()

        for idx, s in enumerate(norm_lines):
            if not s:
                _flush_para_buf()
                if not unwrapped_lines or unwrapped_lines[-1] != "":
                    unwrapped_lines.append("")
                continue

            next_non_empty = ""
            for nxt in norm_lines[idx + 1:]:
                if nxt:
                    next_non_empty = nxt
                    break
            prev_is_break = (not para_buf) and (not unwrapped_lines or unwrapped_lines[-1] == "" or _is_structural_line(unwrapped_lines[-1]))

            if _is_structural_line(s):
                _flush_para_buf()
                unwrapped_lines.append(s)
                continue

            if _is_standalone_title_line(s, prev_is_break, next_non_empty):
                _flush_para_buf()
                unwrapped_lines.append(s)
                if idx + 1 < len(norm_lines) and norm_lines[idx + 1] != "":
                    unwrapped_lines.append("")
                continue

            if idx in labeled_para_starts and para_buf:
                _flush_para_buf()
                if unwrapped_lines and unwrapped_lines[-1] != "":
                    unwrapped_lines.append("")

            para_buf.append(s)

            # Check if this terminal-punctuated line ends a paragraph in a document without blank lines
            if (
                idx + 1 < len(norm_lines)
                and norm_lines[idx + 1] != ""
                and next_non_empty
                and not _is_structural_line(next_non_empty)
                and _ends_with_sentence_terminal(s)
                and re.match(r'^[\'"“‘\(\[]?[A-Z0-9]', next_non_empty)
            ):
                if labeled_para_starts:
                    if (idx + 1) in labeled_para_starts:
                        _flush_para_buf()
                        unwrapped_lines.append("")
                elif not non_term_lens:
                    _flush_para_buf()
                elif body_blank_breaks == 0:
                    nxt_words = next_non_empty.split()
                    first_chunk = nxt_words[0]
                    if len(first_chunk) == 1 and len(nxt_words) > 1:
                        first_chunk = nxt_words[0] + " " + nxt_words[1]
                    buf_non_term = [len(x) for x in para_buf[:-1] if len(x) >= 40]
                    local_wrap = sorted(buf_non_term)[int(len(buf_non_term) * 0.5)] if buf_non_term else global_wrap
                    if len(s) + 1 + len(first_chunk) <= local_wrap:
                        _flush_para_buf()
                        unwrapped_lines.append("")

        _flush_para_buf()

        cleaned_lines: List[str] = []
        for idx, s in enumerate(unwrapped_lines):
            if not s or _is_structural_line(s):
                cleaned_lines.append(s)
                continue

            next_non_empty = ""
            for nxt in unwrapped_lines[idx + 1:]:
                if nxt.strip():
                    next_non_empty = nxt.strip()
                    break
            is_para_end = (idx == len(unwrapped_lines) - 1) or (not unwrapped_lines[idx + 1].strip()) or next_non_empty.startswith("#")

            if (
                is_para_end
                and not cls._is_byline_or_meta_line(s)
                and not s.endswith((".", "!", "?", '"', "”", "’", "'", ":", ";", ")", "]", "—", "–", "…"))
                and re.match(r'^[A-Z“"‘\']', s)
            ):
                words = s.split()
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
