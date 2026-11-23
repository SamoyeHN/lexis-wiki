"""Frontmatter parsing and series/edition extraction helpers for dashboard."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Tuple


def get_raw_file_stem(filepath: Path) -> str:
    return filepath.stem


_ORDINAL_EDITION_MAP = {
    "first": "1e", "1st": "1e", "1": "1e", "one": "1e",
    "second": "2e", "2nd": "2e", "2": "2e", "two": "2e",
    "third": "3e", "3rd": "3e", "3": "3e", "three": "3e",
    "fourth": "4e", "4th": "4e", "4": "4e", "four": "4e",
    "fifth": "5e", "5th": "5e", "5": "5e", "five": "5e",
    "sixth": "6e", "6th": "6e", "6": "6e", "six": "6e",
    "seventh": "7e", "7th": "7e", "7": "7e", "seven": "7e",
    "eighth": "8e", "8th": "8e", "8": "8e", "eight": "8e",
    "ninth": "9e", "9th": "9e", "9": "9e", "nine": "9e",
    "tenth": "10e", "10th": "10e", "10": "10e", "ten": "10e",
    "revised": "Rev", "rev": "Rev", "updated": "Upd",
}


def extract_series_edition(
    raw_series: Optional[str],
    raw_edition: Optional[str] = None,
    raw_abbr: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Cleans series title, extracts normalized edition tag (e.g. '4e', '2e', 'Rev'),
    and derives a friendly short name and combined display tag.
    Returns: (clean_series, edition_tag, series_abbr, series_display)
    """
    if not raw_series:
        return None, None, None, None

    clean_series = raw_series.strip()
    edition_tag = None

    if raw_edition is not None:
        ed_str = str(raw_edition).strip().lower()
        m = re.search(
            r"\b(\d+|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|1st|2nd|3rd|4th|5th|6th|7th|8th|9th|10th|revised|rev)\b",
            ed_str,
        )
        if m:
            edition_tag = _ORDINAL_EDITION_MAP.get(m.group(1), f"{m.group(1)}e")
        else:
            edition_tag = ed_str

    # Pattern 1: Parens containing Edition, e.g. "(Fourth Edition)", "(4th Ed.)", "(Revised Edition)"
    m_paren = re.search(
        r"\s*\(\s*(?:the\s+)?([A-Za-z0-9]+)\s+(?:edition|ed\.?|version)\s*\)",
        clean_series,
        re.IGNORECASE,
    )
    if m_paren:
        if not edition_tag:
            key = m_paren.group(1).lower()
            edition_tag = _ORDINAL_EDITION_MAP.get(key, f"{key}e")
        clean_series = re.sub(
            r"\s*\(\s*(?:the\s+)?([A-Za-z0-9]+)\s+(?:edition|ed\.?|version)\s*\)",
            "",
            clean_series,
            flags=re.IGNORECASE,
        ).strip()

    # Pattern 2: Trailing unparenthesized e.g. "4th Edition", "4th Ed.", "Fourth Edition"
    m_trail = re.search(
        r"\b(?:the\s+)?([A-Za-z0-9]+)\s+(?:edition|ed\.?)\b\.?",
        clean_series,
        re.IGNORECASE,
    )
    if m_trail:
        if not edition_tag:
            key = m_trail.group(1).lower()
            edition_tag = _ORDINAL_EDITION_MAP.get(key, f"{key}e")
        clean_series = re.sub(
            r"\b(?:the\s+)?([A-Za-z0-9]+)\s+(?:edition|ed\.?)\b\.?",
            "",
            clean_series,
            flags=re.IGNORECASE,
        ).strip()

    # Pattern 3: e.g. "(Rev.)" or "(Revised)"
    m_rev = re.search(r"\s*\(\s*(revised|rev\.?|updated)\s*\)", clean_series, re.IGNORECASE)
    if m_rev:
        if not edition_tag:
            edition_tag = _ORDINAL_EDITION_MAP.get(m_rev.group(1).lower().rstrip("."), "Rev")
        clean_series = re.sub(
            r"\s*\(\s*(revised|rev\.?|updated)\s*\)",
            "",
            clean_series,
            flags=re.IGNORECASE,
        ).strip()

    clean_series = re.sub(r"[,\.\-\s]+$", "", clean_series)
    clean_series = re.sub(r"\s+", " ", clean_series).strip()

    # Derive short abbreviation name
    series_abbr = raw_abbr.strip() if raw_abbr else None
    if not series_abbr:
        if "New Horizon College English" in clean_series:
            series_abbr = "New Horizon"
        elif "New Practical English" in clean_series:
            series_abbr = "Practical English"
        elif "Contemporary College English" in clean_series:
            series_abbr = "Contemporary"
        elif "An Integrated Skills Course" in clean_series:
            series_abbr = "Integrated Skills"
        elif len(clean_series) <= 16:
            series_abbr = clean_series
        else:
            series_abbr = clean_series[:14] + "…"

    # Compose series_display
    if edition_tag:
        series_display = f"{series_abbr} · {edition_tag}"
    else:
        series_display = series_abbr

    return clean_series, edition_tag, series_abbr, series_display


def inject_or_lock_frontmatter(
    text: str,
    target_series: Optional[str] = None,
    target_book: Optional[str] = None,
) -> str:
    """Smart Frontmatter Infill & Lock:
    - If frontmatter exists, locks/preserves any existing `series` or `book` fields.
      Only appends missing `series` or `book` fields if provided by upload context.
    - If frontmatter does not exist, synthesizes and prepends a clean YAML frontmatter
      block if target_series or target_book is specified.
    """
    if not target_series and not target_book:
        return text

    # Check for existing YAML frontmatter: ^---\s*\r?\n(.*?)\r?\n---\s*\r?\n
    fm_pattern = re.compile(r"^---\s*\r?\n(.*?)\r?\n---\s*\r?\n", re.DOTALL)
    m = fm_pattern.match(text)

    if m:
        fm_block = m.group(1)
        injected = []
        if target_series and not re.search(r"^\s*series:\s*", fm_block, re.MULTILINE):
            injected.append(f'series: "{target_series}"')
        if target_book and not re.search(r"^\s*book:\s*", fm_block, re.MULTILINE):
            try:
                b_num = int(target_book)
                injected.append(f"book: {b_num}")
            except ValueError:
                injected.append(f'book: "{target_book}"')
            if not re.search(r"^\s*book_name:\s*", fm_block, re.MULTILINE):
                injected.append(f'book_name: "Book {target_book}"')

        if injected:
            new_fm = fm_block.rstrip() + "\n" + "\n".join(injected) + "\n"
            return f"---\n{new_fm}---\n" + text[m.end():]
        return text
    else:
        # No frontmatter, synthesize new block
        injected = []
        if target_series:
            injected.append(f'series: "{target_series}"')
        if target_book:
            try:
                b_num = int(target_book)
                injected.append(f"book: {b_num}")
            except ValueError:
                injected.append(f'book: "{target_book}"')
            injected.append(f'book_name: "Book {target_book}"')

        if injected:
            header = "---\n" + "\n".join(injected) + "\n---\n\n"
            return header + text.lstrip()
        return text
