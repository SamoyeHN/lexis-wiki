"""
Standalone HTML renderer for Lexis Wiki markdown documents with KaTeX, Mermaid, and typography.
"""

import html
import re
from pathlib import Path
from typing import Dict, List, Optional, Union

from ..config import config


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

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html.escape(display_heading)}</title>
    <style>
        :root {{
            --bg-page: #f8fafc;
            --bg-card: #ffffff;
            --text-main: #0f172a;
            --text-secondary: #334155;
            --text-muted: #64748b;
            --border-card: #e2e8f0;
            --accent-primary: #0284c7;
            --accent-navy: #0369a1;
            --font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            --font-mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        }}
        * {{ box-sizing: border-box; }}
        body {{
            margin: 0;
            padding: 1.5rem 1.25rem 3.5rem;
            background-color: var(--bg-page);
            color: var(--text-main);
            font-family: var(--font-sans);
            line-height: 1.6;
            -webkit-font-smoothing: antialiased;
        }}
        .handout-container {{
            max-width: 880px;
            margin: 0 auto;
        }}
        .back-nav {{
            margin-bottom: 1rem;
        }}
        .back-link {{
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            color: var(--accent-primary);
            text-decoration: none;
            font-size: 0.82rem;
            font-weight: 600;
            padding: 0.3rem 0.75rem;
            border-radius: 6px;
            background: #ffffff;
            border: 1px solid var(--border-card);
            transition: all 0.2s ease;
        }}
        .back-link:hover {{
            background: #f1f5f9;
            color: var(--accent-navy);
            text-decoration: none;
        }}
        .handout-header {{
            background: linear-gradient(135deg, #ffffff 0%, #f1f5f9 100%);
            border: 1px solid var(--border-card);
            border-left: 5px solid var(--accent-primary);
            border-radius: 12px;
            padding: 1.25rem 1.6rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            justify-content: space-between;
            gap: 0.75rem;
        }}
        .handout-title {{
            margin: 0;
            font-size: 1.35rem;
            font-weight: 700;
            color: var(--text-main);
            letter-spacing: -0.015em;
        }}
        .handout-meta {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
            flex-wrap: wrap;
        }}
        .meta-badge {{
            display: inline-flex;
            align-items: center;
            padding: 0.2rem 0.65rem;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.02em;
        }}
        .meta-cat {{
            background: rgba(2, 132, 199, 0.1);
            color: #0284c7;
            border: 1px solid rgba(2, 132, 199, 0.25);
        }}
        .meta-cefr {{
            background: rgba(5, 150, 105, 0.1);
            color: #059669;
            border: 1px solid rgba(5, 150, 105, 0.25);
            font-family: var(--font-mono);
        }}
        .meta-count {{
            background: #f1f5f9;
            color: #475569;
            border: 1px solid #cbd5e1;
        }}
        .lex-entry-card {{
            background: var(--bg-card);
            border: 1px solid var(--border-card);
            border-radius: 10px;
            padding: 1.15rem 1.45rem;
            margin-bottom: 0.9rem;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.03);
            break-inside: avoid;
            page-break-inside: avoid;
        }}
        .lex-prose-card {{
            padding: 1.5rem 1.75rem;
        }}
        .lex-entry-title {{
            margin: 0 0 0.65rem 0;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid #f1f5f9;
            font-size: 1.12rem;
            font-weight: 700;
            color: var(--accent-navy);
            display: flex;
            align-items: center;
            flex-wrap: wrap;
            gap: 0.45rem;
        }}
        .lex-headword {{
            color: var(--accent-navy);
        }}
        .lex-badges {{
            display: inline-flex;
            align-items: center;
            flex-wrap: wrap;
            gap: 0.35rem;
        }}
        .lex-pill {{
            display: inline-flex;
            align-items: center;
            padding: 0.1rem 0.52rem;
            border-radius: 999px;
            font-size: 0.72rem;
            font-weight: 600;
            line-height: 1.35;
            white-space: nowrap;
            vertical-align: baseline;
        }}
        .lex-pill-pos {{
            background: rgba(2, 132, 199, 0.09);
            color: #0284c7;
            border: 1px solid rgba(2, 132, 199, 0.25);
            font-style: italic;
        }}
        .lex-pill-cefr {{
            background: rgba(5, 150, 105, 0.1);
            color: #059669;
            border: 1px solid rgba(5, 150, 105, 0.25);
            font-family: var(--font-mono);
            font-size: 0.68rem;
        }}
        .lex-pill-grammar {{
            background: rgba(109, 40, 217, 0.08);
            color: #6d28d9;
            border: 1px solid rgba(109, 40, 217, 0.24);
        }}
        .lex-pill-register {{
            background: rgba(217, 119, 6, 0.1);
            color: #b45309;
            border: 1px solid rgba(217, 119, 6, 0.28);
            font-size: 0.68rem;
            margin-right: 0.25rem;
        }}
        .lex-inline-code {{
            font-family: var(--font-mono);
            font-size: 0.8rem;
            padding: 0.08rem 0.38rem;
            border-radius: 4px;
            background: #f1f5f9;
            color: #0369a1;
            border: 1px solid #e2e8f0;
        }}
        .lex-headword-link {{
            color: #0284c7;
            font-weight: 600;
        }}
        .lex-subheading {{
            margin: 0.75rem 0 0.35rem;
            font-size: 0.95rem;
            font-weight: 700;
            color: #0f766e;
        }}
        .lex-section-title {{
            margin: 1.5rem 0 0.75rem;
            font-size: 1.15rem;
            font-weight: 700;
            color: var(--text-main);
            border-bottom: 2px solid var(--border-card);
            padding-bottom: 0.35rem;
        }}
        .lex-bullet-row {{
            display: flex;
            align-items: flex-start;
            gap: 0.45rem;
            margin-bottom: 0.32rem;
            font-size: 0.92rem;
            color: var(--text-secondary);
        }}
        .lex-bullet-dot {{
            color: var(--accent-primary);
            font-weight: 700;
            line-height: 1.5;
            user-select: none;
        }}
        .lex-bullet-text {{
            flex: 1;
        }}
        .lex-bullet-text strong, .lex-paragraph strong {{
            color: var(--text-main);
            font-weight: 600;
        }}
        .lex-contrast-row {{
            display: flex;
            align-items: flex-start;
            gap: 0.5rem;
            margin: 0.35rem 0;
            padding: 0.4rem 0.7rem;
            border-radius: 6px;
            font-size: 0.9rem;
        }}
        .lex-wrong {{
            background: #fef2f2;
            border-left: 3px solid #ef4444;
            color: #991b1b;
        }}
        .lex-right {{
            background: #f0fdf4;
            border-left: 3px solid #22c55e;
            color: #166534;
        }}
        .lex-mark-wrong {{
            color: #ef4444;
            font-weight: bold;
        }}
        .lex-mark-right {{
            color: #22c55e;
            font-weight: bold;
        }}
        .lex-paragraph {{
            margin: 0.4rem 0;
            color: var(--text-secondary);
            font-size: 0.93rem;
            line-height: 1.65;
        }}
    </style>
</head>
<body>
    <div class="handout-container">
        <div class="back-nav">
            <a href="../index.html" class="back-link">&#8592; Back to Unit Index</a>
        </div>
        <header class="handout-header">
            <h1 class="handout-title">{html.escape(display_heading)}</h1>
            <div class="handout-meta">
                {"".join(meta_badges)}
            </div>
        </header>

        {intro_card_html}
        {body_sections_html}
    </div>
</body>
</html>
"""

