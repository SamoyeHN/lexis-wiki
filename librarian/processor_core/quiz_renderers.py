"""HTML handout rendering, shared template asset inlining, and schema interpolation."""

from __future__ import annotations

import dataclasses
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("librarian.processor.renderers")


class QuizRendererMixin:
    """Mixin providing HTML template rendering with asset inlining and JSON injection."""

    def _get_shared_template_asset(self, asset_name: str) -> str:
        """Loads and caches shared template assets (CSS/JS) for inline embedding."""
        if not hasattr(self, "_template_asset_cache"):
            self._template_asset_cache = {}
        if asset_name in self._template_asset_cache:
            return self._template_asset_cache[asset_name]
        asset_file = Path(__file__).resolve().parent.parent / "templates" / "assets" / asset_name
        if asset_file.exists():
            content = asset_file.read_text(encoding="utf-8")
            self._template_asset_cache[asset_name] = content
            return content
        return ""

    def _render_handout(self, quiz_obj, template_name, audio_url=None, language=None):
        """Renders HTML templates with injected data."""
        tmpl_file = f"{template_name}.html"
        template_path = Path(__file__).resolve().parent.parent / "templates" / tmpl_file
        
        # 1. Convert to dict if it's a dataclass
        if dataclasses.is_dataclass(quiz_obj):
            data_dict = dataclasses.asdict(quiz_obj)
            for extra_attr in ("_expert_audit", "passage", "transcript", "video_url", "video_type", "cefr_level", "vocabulary"):
                if hasattr(quiz_obj, extra_attr) and getattr(quiz_obj, extra_attr) is not None:
                    data_dict[extra_attr] = getattr(quiz_obj, extra_attr)
        else:
            data_dict = quiz_obj # It's already a dict from a JSON-file schema
            
        if not template_path.exists():
            return f"<html><body><pre>{json.dumps(data_dict, indent=2)}</pre></body></html>"

        with open(template_path, "r", encoding="utf-8") as f:
            html = f.read()

        # Inlining shared template assets for self-contained zero-dependency offline delivery
        if "<!-- INJECT:QUIZ_BASE_CSS -->" in html:
            base_css = self._get_shared_template_asset("quiz-base.css")
            if base_css:
                html = html.replace("<!-- INJECT:QUIZ_BASE_CSS -->", f"<style>\n{base_css}\n</style>")
            html = html.replace('<link rel="stylesheet" href="assets/quiz-base.css">', "")
        elif '<link rel="stylesheet" href="assets/quiz-base.css">' in html:
            base_css = self._get_shared_template_asset("quiz-base.css")
            if base_css:
                html = html.replace('<link rel="stylesheet" href="assets/quiz-base.css">', f"<style>\n{base_css}\n</style>")

        if "<!-- INJECT:QUIZ_CORE_JS -->" in html:
            core_js = self._get_shared_template_asset("quiz-core.js")
            if core_js:
                html = html.replace("<!-- INJECT:QUIZ_CORE_JS -->", f"<script>\n{core_js}\n</script>")
            html = html.replace('<script src="assets/quiz-core.js"></script>', "")
        elif '<script src="assets/quiz-core.js"></script>' in html:
            core_js = self._get_shared_template_asset("quiz-core.js")
            if core_js:
                html = html.replace('<script src="assets/quiz-core.js"></script>', f"<script>\n{core_js}\n</script>")
        
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
        if template_name == "translation" and isinstance(cleaned_data, dict):
            for q_item in cleaned_data.get("questions", []):
                if not isinstance(q_item, dict):
                    continue
                opts = q_item.get("options")
                c_idx = q_item.get("correct_answer_index")
                if isinstance(opts, list) and len(opts) == 2 and c_idx in (0, 1):
                    q_item["idiomatic_translation"] = opts[c_idx]
                    q_item["flawed_translation"] = opts[1 - c_idx]
                q_item.pop("correct_english_answer", None)
                if q_item.get("explanation") and not q_item.get("diagnostic_critique"):
                    q_item["diagnostic_critique"] = q_item["explanation"]
                elif q_item.get("diagnostic_critique") and not q_item.get("explanation"):
                    q_item["explanation"] = q_item["diagnostic_critique"]
            if isinstance(cleaned_data.get("_expert_audit"), dict):
                cleaned_data["_expert_audit"] = {
                    k: v for k, v in cleaned_data["_expert_audit"].items() if k != "questions"
                }

        json_data = json.dumps(cleaned_data, ensure_ascii=False)
        

        if "const quizData =" in html:
            return re.sub(r'const quizData = .*?;', lambda _: f'const quizData = {json_data};', html, flags=re.DOTALL)
        return html.replace("</body>", f"<script>const quizData = {json_data};</script></body>")

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
