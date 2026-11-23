"""
Index portal HTML page generation with modern glassmorphism UI for browsing unit resources.
"""

import html
import urllib.parse
from typing import Dict, List, Optional


_BI_ICONS_SVG: Dict[str, str] = {
    "file-earmark-text": """<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" fill="currentColor" viewBox="0 0 16 16"><path d="M5.5 7a.5.5 0 0 0 0 1h5a.5.5 0 0 0 0-1zM5 9.5a.5.5 0 0 1 .5-.5h5a.5.5 0 0 1 0 1h-5a.5.5 0 0 1-.5-.5m0 2a.5.5 0 0 1 .5-.5h2a.5.5 0 0 1 0 1h-2a.5.5 0 0 1-.5-.5"/><path d="M9.5 0H4a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h8a2 2 0 0 0 2-2V4.5zm0 1v2A1.5 1.5 0 0 0 11 4.5h2V14a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V2a1 1 0 0 1 1-1z"/></svg>""",
    "type": """<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" fill="currentColor" viewBox="0 0 16 16"><path d="m2.244 13.081.943-2.803H6.66l.944 2.803H8.86L5.54 3.75H4.322L1 13.081zm3.82-3.818H3.593L4.9 5.215zM10 3.75v1.02h1.832V13.08h1.228V4.77H14.89V3.75z"/></svg>""",
    "diagram-2": """<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" fill="currentColor" viewBox="0 0 16 16"><path fill-rule="evenodd" d="M6 3.5A1.5 1.5 0 0 1 7.5 2h1A1.5 1.5 0 0 1 10 3.5v1A1.5 1.5 0 0 1 8.5 6v1H11a.5.5 0 0 1 .5.5v1a.5.5 0 0 1-1 0V8h-5v.5a.5.5 0 0 1-1 0v-1A.5.5 0 0 1 5 7h2.5V6A1.5 1.5 0 0 1 6 4.5zM8.5 5a.5.5 0 0 0 .5-.5v-1a.5.5 0 0 0-.5-.5h-1a.5.5 0 0 0-.5.5v1a.5.5 0 0 0 .5.5zM3 11.5A1.5 1.5 0 0 1 4.5 10h1A1.5 1.5 0 0 1 7 11.5v1A1.5 1.5 0 0 1 5.5 14h-1A1.5 1.5 0 0 1 3 12.5zm2.5.5a.5.5 0 0 0 .5-.5v-1a.5.5 0 0 0-.5-.5h-1a.5.5 0 0 0-.5.5v1a.5.5 0 0 0 .5.5zm4-2A1.5 1.5 0 0 0 8 11.5v1a1.5 1.5 0 0 0 1.5 1.5h1a1.5 1.5 0 0 0 1.5-1.5v-1a1.5 1.5 0 0 0-1.5-1.5zm1.5 2a.5.5 0 0 1-.5.5h-1a.5.5 0 0 1-.5-.5v-1a.5.5 0 0 1 .5-.5h1a.5.5 0 0 1 .5.5z"/></svg>""",
    "list-stars": """<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" fill="currentColor" viewBox="0 0 16 16"><path fill-rule="evenodd" d="M5 11.5a.5.5 0 0 1 .5-.5h9a.5.5 0 0 1 0 1h-9a.5.5 0 0 1-.5-.5m0-4a.5.5 0 0 1 .5-.5h9a.5.5 0 0 1 0 1h-9a.5.5 0 0 1-.5-.5m0-4a.5.5 0 0 1 .5-.5h9a.5.5 0 0 1 0 1h-9a.5.5 0 0 1-.5-.5m-3 1a1 1 0 0 0 0-2 1 1 0 0 0 0 2m0 4a1 1 0 0 0 0-2 1 1 0 0 0 0 2m0 4a1 1 0 0 0 0-2 1 1 0 0 0 0 2"/><path d="M2.866 14.85c-.078.444-.36.791-.74 1.041a2 2 0 0 1-1.085.35c-.328 0-.64-.092-.907-.267a1.6 1.6 0 0 1-.568-.724c-.13-.338-.13-.71 0-1.049.13-.337.362-.622.66-.822l.142-.093a2 2 0 0 1 .374-.187c.12-.047.247-.078.375-.094a1.8 1.8 0 0 1 .74.072 1.6 1.6 0 0 1 .66.425c.196.21.32.472.361.76l-.012.088zm-1.8-1.02a.8.8 0 0 0-.25.412.7.7 0 0 0 0 .432c.05.132.14.24.256.31.116.07.248.106.384.106.136 0 .268-.036.384-.106a.7.7 0 0 0 .256-.31.8.8 0 0 0 0-.432.8.8 0 0 0-.25-.412 1 1 0 0 0-.39-.17 1 1 0 0 0-.39.17M1.5 0a.5.5 0 0 1 .5.5v1a.5.5 0 0 1-1 0v-1A.5.5 0 0 1 1.5 0m0 6a.5.5 0 0 1 .5.5v1a.5.5 0 0 1-1 0v-1A.5.5 0 0 1 1.5 6"/></svg>""",
    "diagram-3": """<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" fill="currentColor" viewBox="0 0 16 16"><path fill-rule="evenodd" d="M6 3.5A1.5 1.5 0 0 1 7.5 2h1A1.5 1.5 0 0 1 10 3.5v1A1.5 1.5 0 0 1 8.5 6v1H14a.5.5 0 0 1 .5.5v1a.5.5 0 0 1-1 0V8h-5v.5a.5.5 0 0 1-1 0V8h-5v.5a.5.5 0 0 1-1 0v-1A.5.5 0 0 1 2 7h5.5V6A1.5 1.5 0 0 1 6 4.5zM8.5 5a.5.5 0 0 0 .5-.5v-1a.5.5 0 0 0-.5-.5h-1a.5.5 0 0 0-.5.5v1a.5.5 0 0 0 .5.5zM0 11.5A1.5 1.5 0 0 1 1.5 10h1A1.5 1.5 0 0 1 4 11.5v1A1.5 1.5 0 0 1 2.5 14h-1A1.5 1.5 0 0 1 0 12.5zm1.5-.5a.5.5 0 0 0-.5.5v1a.5.5 0 0 0 .5.5h1a.5.5 0 0 0 .5-.5v-1a.5.5 0 0 0-.5-.5zm4.5.5A1.5 1.5 0 0 1 7.5 10h1a1.5 1.5 0 0 1 1.5 1.5v1A1.5 1.5 0 0 1 8.5 14h-1A1.5 1.5 0 0 1 6 12.5zm1.5-.5a.5.5 0 0 0-.5.5v1a.5.5 0 0 0 .5.5h1a.5.5 0 0 0 .5-.5v-1a.5.5 0 0 0-.5-.5zm4.5.5a1.5 1.5 0 0 1 1.5-1.5h1a1.5 1.5 0 0 1 1.5 1.5v1a1.5 1.5 0 0 1-1.5 1.5h-1a1.5 1.5 0 0 1-1.5-1.5zm1.5-.5a.5.5 0 0 0-.5.5v1a.5.5 0 0 0 .5.5h1a.5.5 0 0 0 .5-.5v-1a.5.5 0 0 0-.5-.5z"/></svg>""",
    "spellcheck": """<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" fill="currentColor" viewBox="0 0 16 16"><path d="M8.217 11.068c1.216 0 1.948-.869 1.948-2.31v-.702c0-1.44-.727-2.305-1.929-2.305-.742 0-1.328.317-1.611.832h-.03V4.288h-.947v6.621h.889v-.815h.03c.273.537.864.832 1.616.832h.034zm-.542-.815c-.933 0-1.299-.786-1.299-1.636v-.42c0-.85.366-1.636 1.299-1.636.884 0 1.25.742 1.25 1.636v.42c0 .894-.366 1.636-1.25 1.636M2.44 10.91V4.288H1.5v6.621zM11.5 11.068c.884 0 1.514-.527 1.694-1.284h-.889c-.112.356-.425.59-.805.59-.576 0-.967-.43-.967-1.123V7.22h1.772v-.688H10.53V5.112h-.889v1.421h-.717V7.22h.717v2.124c0 1.147.742 1.724 1.859 1.724"/><path d="M12.646 1.646a.5.5 0 0 1 .708 0l2 2a.5.5 0 0 1 0 .708l-9 9a.5.5 0 0 1-.708 0l-4-4a.5.5 0 0 1 .708-.708L5.5 12.293l8.646-8.647z"/></svg>""",
    "input-cursor-text": """<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" fill="currentColor" viewBox="0 0 16 16"><path fill-rule="evenodd" d="M5 2a.5.5 0 0 1 .5-.5c.862 0 1.573.287 2.06.566.174.099.321.199.44.294V.5a.5.5 0 0 1 1 0v1.86c.119-.095.266-.195.44-.294.487-.279 1.198-.566 2.06-.566a.5.5 0 0 1 0 1c-.741 0-1.272.217-1.637.426a4 4 0 0 0-.863.66v7.828c.31.256.611.488.863.66.365.209.896.426 1.637.426a.5.5 0 0 1 0 1c-.862 0-1.573-.287-2.06-.566a4 4 0 0 1-.44-.294V15.5a.5.5 0 0 1-1 0v-1.86a4 4 0 0 1-.44.294c-.487.279-1.198.566-2.06.566a.5.5 0 0 1 0-1c.741 0 1.272-.217 1.637-.426.252-.144.553-.376.863-.66V3.86a4 4 0 0 1-.863-.66C5.772 2.99 5.241 2.773 4.5 2.773A.5.5 0 0 1 5 2"/></svg>""",
    "book": """<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" fill="currentColor" viewBox="0 0 16 16"><path d="M1 2.828c.885-.37 2.154-.769 3.388-.893 1.33-.134 2.458.063 3.112.752v9.746c-.935-.53-2.12-.603-3.213-.493-1.18.12-2.37.461-3.287.811zm7.5-.141c.654-.689 1.782-.886 3.112-.752 1.234.124 2.503.523 3.388.893v9.923c-.918-.35-2.107-.692-3.287-.81-1.094-.111-2.278-.039-3.213.492zM8 1.783C7.015.936 5.587.81 4.287.94c-1.514.153-3.042.672-3.994 1.105A.5.5 0 0 0 0 2.5v11a.5.5 0 0 0 .707.455c.882-.4 2.303-.881 3.68-1.02 1.409-.142 2.59.087 3.223.877a.5.5 0 0 0 .78 0c.633-.79 1.814-1.019 3.222-.877 1.378.139 2.8.62 3.681 1.02A.5.5 0 0 0 16 13.5v-11a.5.5 0 0 0-.293-.455c-.952-.433-2.48-.952-3.994-1.105C10.413.809 8.985.936 8 1.783"/></svg>""",
    "translate": """<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" fill="currentColor" viewBox="0 0 16 16"><path d="M4.545 6.714 4.11 8H3l1.862-5h1.284L8 8H6.833l-.435-1.286zm1.634-.736L5.5 3.956h-.049l-.679 2.022z"/><path d="M0 2a2 2 0 0 1 2-2h7a2 2 0 0 1 2 2v3h3a2 2 0 0 1 2 2v7a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2v-3H2a2 2 0 0 1-2-2zm2-1a1 1 0 0 0-1 1v7a1 1 0 0 0 1 1h7a1 1 0 0 0 1-1V2a1 1 0 0 0-1-1zm7.138 9.995q.289.451.63.846c-.24.258-.52.464-.829.616v-.95q.1-.218.199-.512m2.246.852q-.355-.392-.623-.852q.124-.344.208-.718h1.28q-.21.848-.865 1.57m-.814-2.222h-.785a5 5 0 0 1-.234-.85h1.201q-.083.421-.182.85"/></svg>""",
    "headphones": """<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" fill="currentColor" viewBox="0 0 16 16"><path d="M8 3a5 5 0 0 0-5 5v1h1a1 1 0 0 1 1 1v3a1 1 0 0 1-1 1H3a1 1 0 0 1-1-1V8a6 6 0 1 1 12 0v5a1 1 0 0 1-1 1h-1a1 1 0 0 1-1-1v-3a1 1 0 0 1 1-1h1V8a5 5 0 0 0-5-5"/></svg>""",
    "film": """<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" fill="currentColor" viewBox="0 0 16 16"><path d="M0 1a1 1 0 0 1 1-1h14a1 1 0 0 1 1 1v14a1 1 0 0 1-1 1H1a1 1 0 0 1-1-1zm4 0v6h8V1zm8 8H4v6h8zM1 1v2h2V1zm2 3H1v2h2zM1 7v2h2V7zm2 3H1v2h2zm-2 3v2h2v-2zM15 1h-2v2h2zm-2 3v2h2V4zm2 3h-2v2h2zm-2 3v2h2v-2zm2 3h-2v2h2z"/></svg>""",
}


def render_index_portal_html(
    unit_name: str,
    display_title: str,
    cefr_level: str,
    extractions: List[Dict[str, str]],
    quizzes: List[Dict[str, str]],
) -> str:
    """
    Renders an elegant, self-contained Bento-grid navigation portal (`index.html`)
    for an exported unit in a clean, modern Light academic theme with Bootstrap Icons.
    """
    unit_title_clean = html.escape(display_title or unit_name.replace("_", " "))
    cefr_badge = (
        f'<span class="portal-badge badge-cefr">CEFR {html.escape(cefr_level)}</span>'
        if cefr_level
        else ""
    )

    # Render extraction cards (using official Bootstrap Icons matching dashboard style)
    category_icon_keys = {
        "source": "file-earmark-text",
        "vocabulary": "type",
        "grammar": "diagram-2",
        "summary": "list-stars",
        "mindmap": "diagram-3",
    }
    extraction_cards_html = []
    for item in extractions:
        cat = item.get("category", "handout")
        icon_key = category_icon_keys.get(cat, "file-earmark-text")
        icon_svg = _BI_ICONS_SVG.get(icon_key, "")
        extraction_cards_html.append(
            f"""
            <a href="{html.escape(item['rel_path'])}" target="_blank" class="portal-card card-node">
                <div class="portal-card-icon">{icon_svg}</div>
                <div class="portal-card-content">
                    <span class="portal-card-title">{html.escape(item['title'])}</span>
                    <p class="portal-card-desc">{html.escape(item.get('description', ''))}</p>
                </div>
                <div class="portal-card-arrow">&#8594;</div>
            </a>
            """
        )

    # Render quiz cards (using official Bootstrap Icons matching dashboard Column 3)
    quiz_icon_keys = {
        "vocabulary": "spellcheck",
        "cloze": "input-cursor-text",
        "reading": "book",
        "translation": "translate",
        "listening": "headphones",
        "video": "film",
    }
    quiz_cards_html = []
    for item in quizzes:
        fn_lower = item["filename"].lower()
        q_type = "quiz"
        for k in quiz_icon_keys:
            if k in fn_lower:
                q_type = k
                break
        icon_key = quiz_icon_keys.get(q_type, "spellcheck")
        icon_svg = _BI_ICONS_SVG.get(icon_key, "")
        quiz_cards_html.append(
            f"""
            <a href="{html.escape(item['rel_path'])}" target="_blank" class="portal-card card-quiz">
                <div class="portal-card-icon">{icon_svg}</div>
                <div class="portal-card-content">
                    <span class="portal-card-title">{html.escape(item['title'])}</span>
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

            /* Scheme A: Knowledge Nodes (Indigo / Deep Blue) */
            --accent-node: #1d4ed8;
            --accent-node-dark: #1e3a8a;
            --accent-node-bg: #eff6ff;
            --accent-node-border: #bfdbfe;

            /* Scheme A: Interactive Assessments (Royal Violet / Purple) */
            --accent-quiz: #7c3aed;
            --accent-quiz-dark: #581c87;
            --accent-quiz-bg: #faf5ff;
            --accent-quiz-border: #e9d5ff;

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
            font-weight: 700;
        }}
        .hero-stat-pill strong.stat-node {{
            color: var(--accent-node-dark);
        }}
        .hero-stat-pill strong.stat-quiz {{
            color: var(--accent-quiz-dark);
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
            margin: 0;
            letter-spacing: -0.01em;
        }}
        .section-nodes .section-title {{
            color: var(--accent-node-dark);
        }}
        .section-nodes .section-count {{
            color: var(--accent-node);
            background: var(--accent-node-bg);
            border: 1px solid var(--accent-node-border);
        }}
        .section-quizzes .section-title {{
            color: var(--accent-quiz-dark);
        }}
        .section-quizzes .section-count {{
            color: var(--accent-quiz);
            background: var(--accent-quiz-bg);
            border: 1px solid var(--accent-quiz-border);
        }}
        .section-count {{
            font-size: 0.72rem;
            font-weight: 600;
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
            transform: translateY(-1px);
            box-shadow: var(--shadow-md);
            text-decoration: none;
        }}
        .card-node:hover {{
            border-color: var(--accent-node-border);
        }}
        .card-quiz:hover {{
            border-color: var(--accent-quiz-border);
        }}
        .portal-card-icon {{
            flex-shrink: 0;
            width: 2.25rem;
            height: 2.25rem;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 8px;
            background: #f1f5f9;
            border: 1px solid var(--border-card);
            transition: all 0.15s ease;
        }}
        .card-node .portal-card-icon {{
            color: var(--accent-node);
        }}
        .card-node:hover .portal-card-icon {{
            background: var(--accent-node-bg);
            border-color: var(--accent-node-border);
            color: var(--accent-node-dark);
        }}
        .card-quiz .portal-card-icon {{
            color: var(--accent-quiz);
        }}
        .card-quiz:hover .portal-card-icon {{
            background: var(--accent-quiz-bg);
            border-color: var(--accent-quiz-border);
            color: var(--accent-quiz-dark);
        }}
        .portal-card-content {{
            flex: 1;
            min-width: 0;
        }}
        .portal-card-title {{
            display: block;
            font-size: 0.92rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }}
        .card-node .portal-card-title {{
            color: var(--accent-node-dark);
        }}
        .card-quiz .portal-card-title {{
            color: var(--accent-quiz-dark);
        }}
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
        .card-node:hover .portal-card-arrow {{
            color: var(--accent-node);
            transform: translateX(3px);
        }}
        .card-quiz:hover .portal-card-arrow {{
            color: var(--accent-quiz);
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
                <span class="hero-stat-pill"><strong class="stat-node">{len(extractions)}</strong> Knowledge Nodes</span>
                <span class="hero-stat-divider">&bull;</span>
                <span class="hero-stat-pill"><strong class="stat-quiz">{len(quizzes)}</strong> Interactive Assessments</span>
            </div>
        </header>

        <section class="section-nodes">
            <div class="section-header">
                <h2 class="section-title">Knowledge Nodes</h2>
                <span class="section-count">{len(extractions)} items</span>
            </div>
            <div class="portal-grid">
                {"".join(extraction_cards_html)}
            </div>
        </section>

        <section class="section-quizzes" style="margin-top: 1.75rem;">
            <div class="section-header">
                <h2 class="section-title">Interactive Assessments</h2>
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

