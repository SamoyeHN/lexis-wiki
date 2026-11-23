"""
Standalone HTML renderer for Lexis Wiki markdown documents with KaTeX, Mermaid, and typography.
"""

import html
import re
from pathlib import Path
from typing import Dict, List, Optional, Union

from ..config import config
from .template_loader import get_template, inline_asset_css


def format_inline_lexis_markdown(text: str) -> str:
    """Renders inline Lexis Markdown tokens into HTML."""
    escaped = html.escape(text)

    # Grammatical bracket pills: `[tag]`
    escaped = re.sub(
        r'`\[([^\]]+)\]`',
        lambda m: " ".join(
            f'<span class="lex-pill lex-pill-grammar">{html.escape(t.strip())}</span>'
            for t in m.group(1).split(",")
            if t.strip()
        ),
        escaped,
    )

    # Register / Attitude labels
    escaped = re.sub(
        r'(?<!\[)\[((?:(?:especially\s+)?(?:British|American|Australian)\s+English|(?:formal|informal|spoken|written|literary|humorous|old-fashioned|technical|law|medical|business|computing)(?:\s+(?:formal|informal|spoken|written|literary|humorous))?))\](?!\])',
        r'<span class="lex-pill lex-pill-register">\1</span>',
        escaped,
        flags=re.IGNORECASE,
    )

    # Standard Markdown inline formatting
    escaped = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', escaped)
    escaped = re.sub(r'\*(.*?)\*', r'<em>\1</em>', escaped)
    escaped = re.sub(r'`([^`]+)`', r'<code class="lex-inline-code">\1</code>', escaped)
    escaped = re.sub(r'\[\[(.*?)\]\]', r'<span class="lex-headword-link">\1</span>', escaped)

    return escaped


def render_markdown_to_standalone_html(
    md_content: str, fallback_title: str = "Lexis Wiki Handout", category: str = "handout"
) -> str:
    """
    Renders pedagogical Markdown extractions into a standalone, beautiful HTML document.
    Embeds all CSS styles for 100% offline portability.
    """
    body_content = md_content or ""
    meta = {
        "title": fallback_title,
        "cefr": "",
        "item_count": "",
        "category_label": (category or "Handout").capitalize(),
    }

    # Extract YAML frontmatter
    fm_match = re.match(r"^---\r?\n([\s\S]*?)\r?\n---", body_content)
    if fm_match:
        body_content = body_content[len(fm_match.group(0)) :].strip()
        fm_text = fm_match.group(1)
        tm = re.search(r'^title:\s*["\']?([^"\'\r\n]+)["\']?', fm_text, re.MULTILINE)
        cefr_m = re.search(r'^overall_cefr_level:\s*["\']?([^"\'\r\n]+)["\']?', fm_text, re.MULTILINE)
        count_m = re.search(r"^item_count:\s*(\d+)", fm_text, re.MULTILINE)
        if tm and tm.group(1).strip():
            meta["title"] = tm.group(1).strip()
        if cefr_m and cefr_m.group(1).strip():
            meta["cefr"] = cefr_m.group(1).strip().upper()
        if count_m and count_m.group(1).strip():
            meta["item_count"] = count_m.group(1).strip()

    lines = body_content.splitlines()
    doc_heading = ""
    sections: List[str] = []
    current_card: Optional[Dict[str, Union[str, List[str]]]] = None
    intro_lines: List[str] = []

    def flush_card():
        nonlocal current_card
        if current_card:
            body_html = "\n".join(current_card["body_lines"])
            sections.append(
                f'<section class="lex-entry-card">{current_card["header_html"]}<div class="lex-entry-body">{body_html}</div></section>'
            )
            current_card = None

    for raw_line in lines:
        trimmed = raw_line.strip()
        leading_spaces = len(re.match(r"^(\s*)", raw_line).group(1).replace("\t", "    "))
        indent_rem = min(leading_spaces * 0.4, 5.0) if leading_spaces > 0 else 0
        indent_style = f' style="padding-left: {indent_rem:.2f}rem;"' if indent_rem > 0 else ""

        if trimmed.startswith("# "):
            if not doc_heading:
                doc_heading = trimmed[2:].strip()
            else:
                flush_card()
                intro_lines.append(
                    f'<h2 class="lex-section-title">{format_inline_lexis_markdown(trimmed[2:].strip())}</h2>'
                )
            continue

        if trimmed.startswith("## "):
            flush_card()
            h2_body = trimmed[3:].strip()
            hw_match = re.match(r"^\[\[(.*?)\]\](.*)$", h2_body)
            if hw_match:
                hw = html.escape(hw_match.group(1))
                tail = hw_match.group(2).strip()
                attr_parts = []
                if tail:
                    gram_tags = []

                    def extract_gram(m):
                        for t in m.group(1).split(","):
                            if t.strip():
                                gram_tags.append(
                                    f'<span class="lex-pill lex-pill-grammar">{html.escape(t.strip())}</span>'
                                )
                        return ""

                    tail_no_gram = re.sub(r"`\[([^\]]+)\]`", extract_gram, tail).strip()
                    cefr_badge = ""

                    def extract_cefr(m):
                        nonlocal cefr_badge
                        cefr_badge = f'<span class="lex-pill lex-pill-cefr">{m.group(1)}</span>'
                        return ""

                    tail_no_cefr = re.sub(r"\b(A1|A2|B1|B2|C1|C2)\b", extract_cefr, tail_no_gram).strip()
                    pos_badge = (
                        f'<span class="lex-pill lex-pill-pos">{html.escape(tail_no_cefr)}</span>'
                        if tail_no_cefr
                        else ""
                    )
                    attr_parts = [p for p in [pos_badge, cefr_badge] + gram_tags if p]

                badges_html = (
                    f' <span class="lex-badges">{" ".join(attr_parts)}</span>' if attr_parts else ""
                )
                header_html = f'<h2 class="lex-entry-title"><span class="lex-headword">{hw}</span>{badges_html}</h2>'
            else:
                header_html = f'<h2 class="lex-entry-title"><span class="lex-headword">{format_inline_lexis_markdown(h2_body)}</span></h2>'

            current_card = {"header_html": header_html, "body_lines": []}
            continue

        if not trimmed or trimmed == "&nbsp;":
            continue

        if trimmed.startswith("### "):
            rendered_line = (
                f'<h3 class="lex-subheading">{format_inline_lexis_markdown(trimmed[4:].strip())}</h3>'
            )
        elif trimmed.startswith("- ") or trimmed.startswith("* "):
            bullet_style = (
                f' style="padding-left: {0.25 + indent_rem:.2f}rem;"' if indent_rem > 0 else ""
            )
            rendered_line = f'<div class="lex-bullet-row"{bullet_style}><span class="lex-bullet-dot">&bull;</span><div class="lex-bullet-text">{format_inline_lexis_markdown(trimmed[2:].strip())}</div></div>'
        elif trimmed.startswith("✗"):
            rendered_line = f'<div class="lex-contrast-row lex-wrong"{indent_style}><span class="lex-mark-wrong">&#10007;</span><span>{format_inline_lexis_markdown(trimmed[1:].strip())}</span></div>'
        elif trimmed.startswith("✓"):
            rendered_line = f'<div class="lex-contrast-row lex-right"{indent_style}><span class="lex-mark-right">&#10003;</span><span>{format_inline_lexis_markdown(trimmed[1:].strip())}</span></div>'
        else:
            rendered_line = f'<p class="lex-paragraph"{indent_style}>{format_inline_lexis_markdown(trimmed)}</p>'

        if current_card:
            current_card["body_lines"].append(rendered_line)
        else:
            intro_lines.append(rendered_line)

    flush_card()

    display_heading = doc_heading or f"{meta['category_label']}: {meta['title']}"
    meta_badges = [f'<span class="meta-badge meta-cat">{html.escape(meta["category_label"])}</span>']
    if meta["cefr"]:
        meta_badges.append(f'<span class="meta-badge meta-cefr">CEFR {html.escape(meta["cefr"])}</span>')
    if meta["item_count"]:
        meta_badges.append(f'<span class="meta-badge meta-count">{html.escape(meta["item_count"])} Items</span>')

    intro_body = "\n".join(intro_lines)
    intro_card_html = (
        f'<section class="lex-entry-card lex-prose-card"><div class="lex-entry-body">{intro_body}</div></section>'
        if intro_lines
        else ""
    )

    body_sections_html = "\n".join(sections)

    template = get_template("handout.html")
    # Inline handout.css to preserve the 100% self-contained offline delivery invariant
    html_content = inline_asset_css(
        template,
        injection_marker="<!-- INJECT:HANDOUT_CSS -->",
        link_tag='<link rel="stylesheet" href="../assets/handout.css">',
        css_name="handout.css",
    )

    replacements = {
        "{{ DISPLAY_HEADING }}": html.escape(display_heading),
        "{{ META_BADGES }}": "".join(meta_badges),
        "{{ INTRO_CARD }}": intro_card_html,
        "{{ BODY_SECTIONS }}": body_sections_html,
    }
    for placeholder, val in replacements.items():
        html_content = html_content.replace(placeholder, val)

    return html_content

