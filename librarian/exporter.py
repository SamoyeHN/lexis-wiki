import html
import os
import platform
import re
import shutil
import subprocess
import urllib.parse
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union
from .config import config


def detect_video_path_from_html(html_content: str, html_path: Path):
    """
    Accurately locates the local media file referenced by a video quiz HTML handout.
    Prioritizes:
      1. Exact `video_url` inside `const quizData = {...}`
      2. Stem match inside `<unit_dir>/sources/media/`
      3. Any media file inside `<unit_dir>/sources/media/`
    """
    html_path = Path(html_path)
    unit_dir = html_path.parent.parent
    media_dir = unit_dir / "sources" / "media"
    media_exts = (".mp4", ".webm", ".ogg", ".mp3", ".wav", ".mkv", ".mov", ".avi")

    # 1. Check explicit video_url in quizData JSON
    m_vurl = re.search(r'"video_url"\s*:\s*"([^"]+)"', html_content)
    if m_vurl:
        raw_url = urllib.parse.unquote(m_vurl.group(1).strip())
        if raw_url and not raw_url.startswith(("http://", "https://")):
            rel_candidate = Path(config.project_root) / raw_url.lstrip("/")
            if rel_candidate.exists() and rel_candidate.is_file():
                return rel_candidate
            base_name = os.path.basename(raw_url)
            if media_dir.exists():
                cand = media_dir / base_name
                if cand.exists() and cand.is_file():
                    return cand

    # 2. Match by handout filename stem (e.g. <VideoStem>_video_quiz.html -> <VideoStem>.mp4)
    clean_stem = re.sub(r"_(?:video_quiz|quiz|handout)$", "", html_path.stem, flags=re.IGNORECASE)
    if media_dir.exists() and media_dir.is_dir():
        for ext in media_exts:
            cand = media_dir / f"{clean_stem}{ext}"
            if cand.exists() and cand.is_file():
                return cand
        for f in sorted(media_dir.iterdir()):
            if f.is_file() and f.suffix.lower() in media_exts:
                if clean_stem.lower() in f.stem.lower() or f.stem.lower() in clean_stem.lower():
                    return f
        # 3. Fallback: first valid media file in unit's sources/media/
        for f in sorted(media_dir.iterdir()):
            if f.is_file() and f.suffix.lower() in media_exts:
                return f

    return None


def make_portable_html_content(html_content: str, media_filename: str = None) -> str:
    """
    Rewrites `video_url` in the exported HTML copy to a relative same-folder filename
    so opening the extracted `.html` directly via `file://` on Windows/macOS works immediately.
    """
    if not media_filename:
        return html_content
    escaped_name = media_filename.replace("\\", "\\\\").replace('"', '\\"')
    return re.sub(
        r'("video_url"\s*:\s*)"[^"]*"',
        lambda m: f'{m.group(1)}"{escaped_name}"',
        html_content,
        count=1,
    )


def export_portable_bundle(html_path, video_path=None, output_path=None):
    """
    Packages an interactive video HTML handout and its companion local video/audio file
    into a portable `.zip` archive (or copies them into a target folder if output_path is a directory).
    """
    path_str = str(html_path).replace("\\", "/")
    if "wiki/" in path_str:
        relative_part = path_str.split("wiki/", 1)[1]
        resolved_path = Path(config.project_root) / "wiki" / relative_part
        html_path = resolved_path if resolved_path.exists() else Path(html_path)
    else:
        if path_str.startswith("/"):
            path_str = path_str[1:]
        resolved_path = Path(config.project_root) / path_str
        html_path = resolved_path if resolved_path.exists() else Path(html_path)

    if not html_path.exists():
        print(f"Error: HTML handout not found at '{html_path}'")
        return False

    html_content = html_path.read_text(encoding="utf-8")

    # Resolve companion video file
    resolved_video = None
    if video_path:
        cand = Path(video_path)
        if cand.exists() and cand.is_file():
            resolved_video = cand
    if not resolved_video:
        resolved_video = detect_video_path_from_html(html_content, html_path)

    media_filename = resolved_video.name if resolved_video else None
    portable_html = make_portable_html_content(html_content, media_filename)

    html_stem = html_path.stem
    if output_path:
        out_p = Path(output_path)
        # If output_path is an existing directory or has no suffix, copy files directly into that folder
        if out_p.is_dir() or not out_p.suffix:
            out_p.mkdir(parents=True, exist_ok=True)
            dst_html = out_p / html_path.name
            dst_html.write_text(portable_html, encoding="utf-8")
            if resolved_video and resolved_video.exists():
                shutil.copy2(resolved_video, out_p / resolved_video.name)
            return dst_html
        target_zip_path = out_p if out_p.suffix.lower() == ".zip" else out_p.with_suffix(".zip")
        target_zip_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        target_output_dir = html_path.parent
        target_output_dir.mkdir(parents=True, exist_ok=True)
        target_zip_path = target_output_dir / f"{html_stem}_bundle.zip"

    # Write portable HTML + companion video into ZIP archive
    with zipfile.ZipFile(target_zip_path, "w") as zf:
        zf.writestr(
            zipfile.ZipInfo(html_path.name),
            portable_html.encode("utf-8"),
            compress_type=zipfile.ZIP_DEFLATED,
        )
        if resolved_video and resolved_video.exists():
            # Video files are already compressed; ZIP_STORED packages 50MB in <0.05s with zero CPU overhead
            zf.write(resolved_video, arcname=resolved_video.name, compress_type=zipfile.ZIP_STORED)

    return target_zip_path


# Backward-compatible alias used by dashboard and CLI
export_standalone_exe = export_portable_bundle


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

    intro_card_html = (
        f'<section class="lex-entry-card lex-prose-card"><div class="lex-entry-body">{"\n".join(intro_lines)}</div></section>'
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


def render_index_portal_html(
    unit_name: str,
    display_title: str,
    cefr_level: str,
    extractions: List[Dict[str, str]],
    quizzes: List[Dict[str, str]],
) -> str:
    """
    Renders an elegant, self-contained Bento-grid navigation portal (`index.html`)
    for an exported unit in a clean, modern Light academic theme.
    """
    unit_title_clean = html.escape(display_title or unit_name.replace("_", " "))
    cefr_badge = (
        f'<span class="portal-badge badge-cefr">CEFR {html.escape(cefr_level)}</span>'
        if cefr_level
        else ""
    )

    # Render extraction cards
    extraction_cards_html = []
    category_icons = {
        "source": "📄",
        "vocabulary": "📚",
        "grammar": "🧩",
        "summary": "📝",
        "mindmap": "🧠",
    }
    category_tags = {
        "source": ("SOURCE", "tag-source"),
        "vocabulary": ("VOCAB", "tag-vocab"),
        "grammar": ("SYNTAX", "tag-syntax"),
        "summary": ("DISCOURSE", "tag-discourse"),
        "mindmap": ("MINDMAP", "tag-mindmap"),
    }
    for item in extractions:
        cat = item.get("category", "handout")
        icon = category_icons.get(cat, "📄")
        tag_text, tag_cls = category_tags.get(cat, (cat.upper(), "tag-source"))
        extraction_cards_html.append(
            f"""
            <a href="{html.escape(item['rel_path'])}" target="_blank" class="portal-card">
                <div class="portal-card-icon">{icon}</div>
                <div class="portal-card-content">
                    <div class="portal-card-header">
                        <span class="portal-card-title">{html.escape(item['title'])}</span>
                        <span class="portal-card-tag {tag_cls}">{html.escape(tag_text)}</span>
                    </div>
                    <p class="portal-card-desc">{html.escape(item.get('description', ''))}</p>
                </div>
                <div class="portal-card-arrow">&#8594;</div>
            </a>
            """
        )

    # Render quiz cards
    quiz_cards_html = []
    quiz_specs = {
        "vocabulary": ("🎯", "MCQ", "tag-mcq"),
        "cloze": ("✏️", "OPEN CLOZE", "tag-cloze"),
        "reading": ("📖", "READING", "tag-reading"),
        "translation": ("🌐", "TRANSLATION", "tag-translation"),
        "listening": ("🎧", "AUDIO", "tag-audio"),
        "video": ("🎬", "VIDEO", "tag-video"),
    }
    for item in quizzes:
        fn_lower = item["filename"].lower()
        q_type = "quiz"
        for k in quiz_specs:
            if k in fn_lower:
                q_type = k
                break
        icon, tag_text, tag_cls = quiz_specs.get(q_type, ("🎯", "ASSESSMENT", "tag-quiz"))
        quiz_cards_html.append(
            f"""
            <a href="{html.escape(item['rel_path'])}" target="_blank" class="portal-card">
                <div class="portal-card-icon">{icon}</div>
                <div class="portal-card-content">
                    <div class="portal-card-header">
                        <span class="portal-card-title">{html.escape(item['title'])}</span>
                        <span class="portal-card-tag {tag_cls}">{html.escape(tag_text)}</span>
                    </div>
                    <p class="portal-card-desc">{html.escape(item.get('description', ''))}</p>
                </div>
                <div class="portal-card-arrow">&#8594;</div>
            </a>
            """
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{unit_title_clean} — Study Suite</title>
    <style>
        :root {{
            --bg-page: #f8fafc;
            --bg-card: #ffffff;
            --bg-card-hover: #f1f5f9;
            --text-main: #0f172a;
            --text-secondary: #475569;
            --text-muted: #64748b;
            --border-card: #e2e8f0;
            --border-card-hover: #cbd5e1;
            --accent-primary: #2563eb;
            --accent-green: #059669;
            --accent-green-bg: #ecfdf5;
            --shadow-sm: 0 1px 3px rgba(15, 23, 42, 0.04);
            --shadow-md: 0 4px 12px -2px rgba(15, 23, 42, 0.08), 0 2px 6px -1px rgba(15, 23, 42, 0.04);
            --font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            --font-mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        }}
        * {{ box-sizing: border-box; }}
        body {{
            margin: 0;
            padding: 2.25rem 1.5rem 3.5rem;
            background-color: var(--bg-page);
            color: var(--text-main);
            font-family: var(--font-sans);
            line-height: 1.6;
            -webkit-font-smoothing: antialiased;
        }}
        .portal-container {{
            max-width: 960px;
            margin: 0 auto;
        }}
        .hero-banner {{
            background: #ffffff;
            border: 1px solid var(--border-card);
            border-left: 5px solid var(--accent-primary);
            border-radius: 12px;
            padding: 1.5rem 1.75rem;
            margin-bottom: 2rem;
            box-shadow: var(--shadow-sm);
        }}
        .hero-top {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 0.75rem;
            margin-bottom: 0.5rem;
        }}
        .hero-badges {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
            flex-wrap: wrap;
        }}
        .portal-badge {{
            display: inline-flex;
            align-items: center;
            padding: 0.2rem 0.6rem;
            border-radius: 999px;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.02em;
        }}
        .badge-unit {{
            background: #f1f5f9;
            color: #334155;
            border: 1px solid #cbd5e1;
            font-family: var(--font-mono);
        }}
        .badge-cefr {{
            background: var(--accent-green-bg);
            color: var(--accent-green);
            border: 1px solid rgba(5, 150, 105, 0.25);
            font-family: var(--font-mono);
        }}
        .hero-title {{
            margin: 0 0 0.5rem 0;
            font-size: 1.6rem;
            font-weight: 800;
            color: var(--text-main);
            letter-spacing: -0.02em;
        }}
        .hero-stats {{
            display: flex;
            align-items: center;
            gap: 0.6rem;
            font-size: 0.82rem;
            color: var(--text-muted);
        }}
        .hero-stat-pill strong {{
            color: var(--text-main);
            font-weight: 700;
        }}
        .hero-stat-divider {{
            color: var(--border-card);
        }}
        .section-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin: 0 0 0.85rem 0;
            padding-bottom: 0.4rem;
            border-bottom: 1px solid var(--border-card);
        }}
        .section-title {{
            font-size: 1.05rem;
            font-weight: 700;
            color: var(--text-main);
            display: flex;
            align-items: center;
            gap: 0.45rem;
            margin: 0;
            letter-spacing: -0.01em;
        }}
        .section-count {{
            font-size: 0.72rem;
            font-weight: 600;
            color: var(--text-muted);
            background: #ffffff;
            border: 1px solid var(--border-card);
            padding: 0.12rem 0.55rem;
            border-radius: 999px;
            font-family: var(--font-mono);
        }}
        .portal-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 0.85rem;
        }}
        .portal-card {{
            display: flex;
            align-items: center;
            gap: 0.85rem;
            background: var(--bg-card);
            border: 1px solid var(--border-card);
            border-radius: 10px;
            padding: 0.95rem 1.15rem;
            text-decoration: none;
            color: inherit;
            transition: all 0.15s ease;
            box-shadow: var(--shadow-sm);
        }}
        .portal-card:hover {{
            background: #ffffff;
            border-color: var(--accent-primary);
            transform: translateY(-1px);
            box-shadow: var(--shadow-md);
            text-decoration: none;
        }}
        .portal-card-icon {{
            font-size: 1.45rem;
            line-height: 1;
            flex-shrink: 0;
            width: 2.25rem;
            height: 2.25rem;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 8px;
            background: #f8fafc;
            border: 1px solid var(--border-card);
        }}
        .portal-card-content {{
            flex: 1;
            min-width: 0;
        }}
        .portal-card-header {{
            display: flex;
            align-items: center;
            gap: 0.4rem;
            margin-bottom: 0.2rem;
            flex-wrap: wrap;
        }}
        .portal-card-title {{
            font-size: 0.92rem;
            font-weight: 700;
            color: var(--text-main);
        }}
        .portal-card-tag {{
            font-size: 0.62rem;
            font-weight: 700;
            padding: 0.08rem 0.4rem;
            border-radius: 4px;
            letter-spacing: 0.03em;
            font-family: var(--font-mono);
        }}
        .tag-source {{ background: #f1f5f9; color: #475569; border: 1px solid #cbd5e1; }}
        .tag-vocab {{ background: #eff6ff; color: #1d4ed8; border: 1px solid #bfdbfe; }}
        .tag-syntax {{ background: #f5f3ff; color: #6d28d9; border: 1px solid #ddd6fe; }}
        .tag-discourse {{ background: #fff7ed; color: #c2410c; border: 1px solid #fed7aa; }}
        .tag-mindmap {{ background: #fdf2f8; color: #be185d; border: 1px solid #fbcfe8; }}
        
        .tag-mcq {{ background: #eff6ff; color: #1d4ed8; border: 1px solid #bfdbfe; }}
        .tag-cloze {{ background: #ecfdf5; color: #047857; border: 1px solid #a7f3d0; }}
        .tag-reading {{ background: #f0fdf4; color: #15803d; border: 1px solid #bbf7d0; }}
        .tag-translation {{ background: #fffbeb; color: #b45309; border: 1px solid #fde68a; }}
        .tag-audio {{ background: #faf5ff; color: #7e22ce; border: 1px solid #e9d5ff; }}
        .tag-video {{ background: #fef2f2; color: #b91c1c; border: 1px solid #fecaca; }}
        .tag-quiz {{ background: #f1f5f9; color: #475569; border: 1px solid #cbd5e1; }}

        .portal-card-desc {{
            margin: 0;
            font-size: 0.76rem;
            color: var(--text-secondary);
            line-height: 1.35;
        }}
        .portal-card-arrow {{
            color: var(--text-muted);
            font-size: 1rem;
            transition: transform 0.15s ease, color 0.15s ease;
            flex-shrink: 0;
        }}
        .portal-card:hover .portal-card-arrow {{
            color: var(--accent-primary);
            transform: translateX(3px);
        }}
        .portal-footer {{
            margin-top: 3rem;
            text-align: center;
            font-size: 0.75rem;
            color: var(--text-muted);
            border-top: 1px solid var(--border-card);
            padding-top: 1.25rem;
        }}
    </style>
</head>
<body>
    <div class="portal-container">
        <header class="hero-banner">
            <div class="hero-top">
                <div class="hero-badges">
                    <span class="portal-badge badge-unit">{html.escape(unit_name)}</span>
                    {cefr_badge}
                </div>
            </div>
            <h1 class="hero-title">{unit_title_clean}</h1>
            <div class="hero-stats">
                <span class="hero-stat-pill"><strong>{len(extractions)}</strong> Knowledge Nodes</span>
                <span class="hero-stat-divider">&bull;</span>
                <span class="hero-stat-pill"><strong>{len(quizzes)}</strong> Interactive Assessments</span>
            </div>
        </header>

        <section>
            <div class="section-header">
                <h2 class="section-title"><span>📚</span> Knowledge Nodes</h2>
                <span class="section-count">{len(extractions)} items</span>
            </div>
            <div class="portal-grid">
                {"".join(extraction_cards_html)}
            </div>
        </section>

        <section style="margin-top: 1.75rem;">
            <div class="section-header">
                <h2 class="section-title"><span>🎯</span> Interactive Assessments</h2>
                <span class="section-count">{len(quizzes)} assessments</span>
            </div>
            <div class="portal-grid">
                {"".join(quiz_cards_html)}
            </div>
        </section>

        <footer class="portal-footer">
            Generated by <strong>Lexis Wiki</strong> &bull; Neuro-Symbolic Educational Knowledge Base
        </footer>
    </div>
</body>
</html>
"""


def normalize_export_target_dir(
    target_dir_str: Optional[Union[str, Path]], unit_name: str
) -> Path:
    """
    Resolves custom export directory with cross-platform and WSL Windows drive compatibility.
    Supports Windows paths (e.g. C:/Users/..., D:\\Exports) inside WSL.
    """
    raw = str(target_dir_str or "").strip()
    if not raw:
        cfg_export = str(config.data.get("export_dir", "") or "").strip()
        if cfg_export:
            raw = cfg_export

    clean_unit = re.sub(r"\.[^/.]+$", "", unit_name)
    default_dir = Path(config.project_root) / "wiki" / clean_unit / "exports"

    if not raw:
        return default_dir

    # Support Windows drive letter syntax in WSL (e.g., C:\..., C:/..., D:/...)
    if len(raw) >= 2 and raw[1] == ":":
        wslpath = shutil.which("wslpath")
        if wslpath:
            try:
                res = subprocess.run([wslpath, "-u", raw], capture_output=True, text=True, check=True)
                converted = res.stdout.strip()
                if converted:
                    raw = converted
            except Exception:
                pass
        if len(raw) >= 2 and raw[1] == ":":
            drive = raw[0].lower()
            rest = raw[2:].replace("\\", "/").lstrip("/")
            raw = f"/mnt/{drive}/{rest}"

    p = Path(raw)
    if not p.is_absolute():
        p = (Path(config.project_root) / p).resolve()

    # If the user specified a general export folder (e.g. C:/Exports), nest the unit folder inside it
    if p.name != clean_unit and not p.name.endswith("exports"):
        p = p / clean_unit

    return p.resolve()


def to_windows_display_path(p: Path) -> str:
    """
    Returns a Windows-friendly UNC or drive path if running inside WSL, or native path.
    """
    is_wsl = bool(
        os.environ.get("WSL_DISTRO_NAME")
        or (Path("/proc/version").exists() and "microsoft" in Path("/proc/version").read_text().lower())
    )
    if is_wsl:
        wslpath = shutil.which("wslpath")
        if wslpath:
            try:
                res = subprocess.run([wslpath, "-w", str(p)], capture_output=True, text=True, check=True)
                win_path = res.stdout.strip()
                if win_path:
                    return win_path
            except Exception:
                pass
        s = str(p)
        if s.startswith("/mnt/") and len(s) > 6 and s[6] == "/":
            drive = s[5].upper()
            rest = s[7:].replace("/", "\\")
            return f"{drive}:\\{rest}"
    return str(p)


def export_unit_to_directory(unit_name: str, custom_target_dir: Optional[Union[str, Path]] = None) -> dict:
    """
    Exports all verified extractions, companion quizzes, and an index portal
    for a specific unit into `wiki/<UnitName>/exports/` or a custom user-defined directory.
    """
    unit_name = str(unit_name).strip()
    clean_unit = re.sub(r"\.[^/.]+$", "", unit_name)
    unit_dir = Path(config.project_root) / "wiki" / clean_unit

    if not unit_dir.exists() or not unit_dir.is_dir():
        return {
            "success": False,
            "error": f"Unit directory not found: wiki/{clean_unit}",
        }

    export_dir = normalize_export_target_dir(custom_target_dir, clean_unit)
    export_extractions_dir = export_dir / "extractions"
    export_handouts_dir = export_dir / "handouts"

    export_extractions_dir.mkdir(parents=True, exist_ok=True)
    export_handouts_dir.mkdir(parents=True, exist_ok=True)

    extracted_files: List[Dict[str, str]] = []
    quiz_files: List[Dict[str, str]] = []

    # 1. Process sources/<UnitName>.md -> exports/extractions/<UnitName>_source.html
    sources_dir = unit_dir / "sources"
    primary_file = None
    display_title = clean_unit.replace("_", " ")
    cefr_level = ""

    if sources_dir.exists():
        cand = sources_dir / f"{clean_unit}.md"
        if cand.exists():
            primary_file = cand
        else:
            for f in sorted(sources_dir.iterdir()):
                if f.is_file() and f.suffix in [".md", ".txt"]:
                    primary_file = f
                    break

    if primary_file and primary_file.exists():
        try:
            raw_text = primary_file.read_text(encoding="utf-8")
            tm = re.search(
                r'^(?:title|text_title|unit_title):\s*["\']?(.*?)["\']?\s*$',
                raw_text,
                re.MULTILINE,
            )
            if tm and tm.group(1).strip():
                display_title = tm.group(1).strip()
            clean_source = re.sub(
                r"<syllabus>[\s\S]*?</syllabus>", "", raw_text, flags=re.IGNORECASE
            ).strip()
            source_html = render_markdown_to_standalone_html(
                clean_source,
                fallback_title=f"{display_title} — Source Reading",
                category="source",
            )
            out_source_html = export_extractions_dir / f"{clean_unit}_source.html"
            out_source_html.write_text(source_html, encoding="utf-8")
            extracted_files.append(
                {
                    "title": "Source Text",
                    "filename": f"{clean_unit}_source.html",
                    "rel_path": f"extractions/{clean_unit}_source.html",
                    "category": "source",
                    "description": "Original passage with clean pedagogical typography",
                }
            )
        except Exception as e:
            print(f"Warning: Failed to export source text for {clean_unit}: {e}")

    # 2. Process extractions/
    extractions_dir = unit_dir / "extractions"
    if extractions_dir.exists():
        for f in sorted(extractions_dir.iterdir()):
            if not f.is_file():
                continue
            if f.suffix == ".md":
                try:
                    md_text = f.read_text(encoding="utf-8")
                    cat = "handout"
                    desc = "Pedagogical extraction node"
                    card_title = f.stem.replace(clean_unit, "").replace("_", " ").strip().title()
                    if "_vocabulary" in f.name:
                        cat = "vocabulary"
                        card_title = "Vocabulary Focus"
                        desc = "Sense-locked headwords, valency cascades & patterns"
                        cefr_m = re.search(
                            r'overall_cefr_level:\s*["\']?(.*?)["\']?\s*$', md_text, re.MULTILINE
                        )
                        if cefr_m and cefr_m.group(1).strip():
                            cefr_level = cefr_m.group(1).strip().upper()
                    elif "_grammar" in f.name:
                        cat = "grammar"
                        card_title = "Grammar Patterns"
                        desc = "Syntactic frames, constructions & SLA contrast pairs"
                    elif "_summary" in f.name:
                        cat = "summary"
                        card_title = "Discourse Summary"
                        desc = "Discourse organization, macro-propositions & outline"

                    html_doc = render_markdown_to_standalone_html(
                        md_text,
                        fallback_title=f"{display_title} — {card_title}",
                        category=cat,
                    )
                    out_html = export_extractions_dir / f"{f.stem}.html"
                    out_html.write_text(html_doc, encoding="utf-8")
                    extracted_files.append(
                        {
                            "title": card_title,
                            "filename": f"{f.stem}.html",
                            "rel_path": f"extractions/{f.stem}.html",
                            "category": cat,
                            "description": desc,
                        }
                    )
                except Exception as e:
                    print(f"Warning: Failed to export extraction {f.name}: {e}")
            elif f.suffix == ".html" and "_mindmap" in f.name:
                try:
                    out_mm = export_extractions_dir / f.name
                    shutil.copy2(f, out_mm)
                    json_cand = f.with_suffix(".json")
                    if json_cand.exists():
                        shutil.copy2(json_cand, export_extractions_dir / json_cand.name)
                    extracted_files.append(
                        {
                            "title": "Knowledge Mind Map",
                            "filename": f.name,
                            "rel_path": f"extractions/{f.name}",
                            "category": "mindmap",
                            "description": "Hierarchical concept graph & structural visualization",
                        }
                    )
                except Exception as e:
                    print(f"Warning: Failed to copy mindmap {f.name}: {e}")

    # 3. Process handouts/
    handouts_dir = unit_dir / "handouts"
    if handouts_dir.exists():
        for f in sorted(handouts_dir.iterdir()):
            if not f.is_file() or f.suffix != ".html" or "_quiz" not in f.name:
                continue
            try:
                html_content = f.read_text(encoding="utf-8")
                resolved_video = detect_video_path_from_html(html_content, f)
                media_filename = None
                if resolved_video and resolved_video.exists():
                    media_filename = resolved_video.name
                    shutil.copy2(resolved_video, export_handouts_dir / media_filename)

                portable_html = make_portable_html_content(html_content, media_filename)
                out_quiz = export_handouts_dir / f.name
                out_quiz.write_text(portable_html, encoding="utf-8")

                q_name = f.name
                q_title = "Assessment"
                q_desc = "Interactive assessment"
                if "vocabulary" in q_name:
                    q_title = "Vocabulary Quiz"
                    q_desc = "Single-fit context clozes with diagnostic lexicographic feedback"
                elif "cloze" in q_name:
                    q_title = "Discourse Cloze"
                    q_desc = "Pure open cloze assessment with acceptable-word scoring"
                elif "reading" in q_name:
                    q_title = "Reading Comprehension"
                    q_desc = "Discourse reasoning and fallacy-audited reading comprehension"
                elif "translation" in q_name:
                    q_title = "Translation Practice"
                    q_desc = "SLA minimal-pair error injection & structural contrast"
                elif "listening" in q_name:
                    q_title = "Listening Comprehension"
                    q_desc = "Conversation analysis multi-turn dialogues with authentic native audio"
                elif "video" in q_name:
                    q_title = "Video Comprehension"
                    q_desc = "Timestamp-anchored interactive multimodal video comprehension"

                quiz_files.append(
                    {
                        "title": q_title,
                        "filename": f.name,
                        "rel_path": f"handouts/{f.name}",
                        "description": q_desc,
                    }
                )
            except Exception as e:
                print(f"Warning: Failed to export quiz {f.name}: {e}")

    # 4. Generate exports/index.html navigation portal
    index_html = render_index_portal_html(
        unit_name=clean_unit,
        display_title=display_title,
        cefr_level=cefr_level,
        extractions=extracted_files,
        quizzes=quiz_files,
    )
    (export_dir / "index.html").write_text(index_html, encoding="utf-8")

    total_count = 1 + len(extracted_files) + len(quiz_files)
    win_path = to_windows_display_path(export_dir)
    try:
        rel_path = str(export_dir.relative_to(config.project_root))
    except Exception:
        rel_path = win_path

    return {
        "success": True,
        "unit_name": clean_unit,
        "display_title": display_title,
        "export_path": str(export_dir.resolve()),
        "windows_path": win_path,
        "relative_path": rel_path,
        "file_count": total_count,
        "extractions_count": len(extracted_files),
        "quizzes_count": len(quiz_files),
    }


def export_unit_to_zip(
    unit_name: str, custom_target_dir: Optional[Union[str, Path]] = None
) -> dict:
    """
    Exports all extractions and companion quizzes into a directory, then packages
    the complete unit bundle into a portable ZIP archive for direct browser download.
    """
    dir_res = export_unit_to_directory(unit_name, custom_target_dir)
    if not dir_res.get("success"):
        return dir_res

    export_dir = Path(dir_res["export_path"])
    clean_unit = dir_res["unit_name"]
    zip_path = (
        export_dir.parent / f"{clean_unit}_handouts.zip"
        if export_dir.name == "exports"
        else export_dir / f"{clean_unit}_handouts.zip"
    )

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in export_dir.rglob("*"):
            if file_path.is_file() and file_path != zip_path:
                arcname = file_path.relative_to(export_dir)
                if file_path.suffix.lower() in [".mp4", ".webm", ".ogg", ".mp3", ".wav", ".mkv", ".mov"]:
                    zf.write(file_path, arcname=str(arcname), compress_type=zipfile.ZIP_STORED)
                else:
                    zf.write(file_path, arcname=str(arcname), compress_type=zipfile.ZIP_DEFLATED)

    win_zip_path = to_windows_display_path(zip_path)
    return {
        "success": True,
        "unit_name": clean_unit,
        "zip_path": str(zip_path.resolve()),
        "windows_path": win_zip_path,
        "file_name": zip_path.name,
        "file_size": zip_path.stat().st_size,
    }


def open_directory_in_explorer(dir_path: Union[str, Path]) -> bool:
    """
    Opens the target directory in the host OS native file manager (cross-platform, with full WSL support).
    """
    p = Path(dir_path).resolve()
    if not p.exists():
        return False
    sys_name = platform.system()
    try:
        if sys_name == "Windows":
            os.startfile(str(p))
            return True
        elif sys_name == "Darwin":
            subprocess.Popen(["open", str(p)])
            return True
        else:  # Linux / Unix / WSL
            # 1. Detect Windows Subsystem for Linux (WSL)
            is_wsl = bool(
                os.environ.get("WSL_DISTRO_NAME")
                or os.environ.get("WSL_INTEROP")
                or (Path("/proc/version").exists() and "microsoft" in Path("/proc/version").read_text().lower())
            )
            explorer_cmd = shutil.which("explorer.exe")
            if not explorer_cmd:
                for cand in ["/mnt/c/Windows/explorer.exe", "/mnt/c/WINDOWS/explorer.exe"]:
                    if Path(cand).exists():
                        explorer_cmd = cand
                        break

            if is_wsl and explorer_cmd:
                target = str(p)
                wslpath_cmd = shutil.which("wslpath")
                if wslpath_cmd:
                    try:
                        res = subprocess.run([wslpath_cmd, "-w", str(p)], capture_output=True, text=True, check=True)
                        win_path = res.stdout.strip()
                        if win_path:
                            target = win_path
                    except Exception as e:
                        print(f"Warning: wslpath failed for {p}: {e}")
                subprocess.Popen([explorer_cmd, target])
                return True

            # 2. Standard Linux desktop environment
            if shutil.which("xdg-open"):
                subprocess.Popen(["xdg-open", str(p)])
                return True
            return False
    except Exception as e:
        print(f"Warning: Failed to open directory {p} in file manager: {e}")
        return False
