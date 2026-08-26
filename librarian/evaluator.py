import re
import json
import unicodedata
from pathlib import Path
from typing import Dict, List, Any, Optional

from .config import config

# --- Dimension weights (max points per dimension). Single source of truth. ---
W_SCHEMA = 25.0
W_VERBATIM = 30.0
W_PEDAGOGY = 25.0
W_UNIQUENESS = 20.0

# Top-level arrays that may hold gradeable items, in priority order.
ITEM_KEYS = ["vocabulary", "expressions", "grammar_patterns", "questions", "concepts", "branches"]


def _normalize_text(value: Any) -> str:
    """Normalize unicode, drop decorative quotes/punctuation, collapse whitespace."""
    if not isinstance(value, str):
        return ""
    text = unicodedata.normalize("NFKC", value).lower()
    text = re.sub(r"[“”‘’\"'«»]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _token_coverage(quote: str, source: str) -> float:
    """Fraction of the quote's tokens that appear in the source text word-set."""
    words = quote.split()
    if not words or not source:
        return 0.0
    source_words = set(source.split())
    return sum(1 for w in words if w in source_words) / len(words)


def _extract_source_content(user_prompt: str) -> str:
    """Extracts isolated source text from user prompt (under CONTENT:) or falls back to whole prompt."""
    if not user_prompt:
        return ""
    match = re.search(r"CONTENT:\s*\n(.*)", user_prompt, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return user_prompt


def _detect_task_type(task: str, parsed: Any) -> str:
    """Detect the logical task type from JSON structure first, task name as fallback."""
    if isinstance(parsed, dict):
        if isinstance(parsed.get("questions"), list):
            return "quiz"
        if isinstance(parsed.get("grammar_patterns"), list):
            return "grammar"
        if isinstance(parsed.get("expressions"), list):
            return "expressions"
        if isinstance(parsed.get("vocabulary"), list):
            return "vocabulary"
        if isinstance(parsed.get("concepts"), list):
            return "summary"
        if isinstance(parsed.get("branches"), list):
            return "mindmap"
    t = (task or "").lower()
    if "extract_grammar" in t:
        return "grammar"
    if "extract_expressions" in t:
        return "expressions"
    if "quiz" in t:
        return "quiz"
    if "extract_summary" in t or "summary" in t:
        return "summary"
    if "extract_mindmap" in t or "mindmap" in t:
        return "mindmap"
    if "vocabulary" in t:
        return "vocabulary"
    return "unknown"


def _extract_items(parsed: Any, task_type: str) -> List[Dict[str, Any]]:
    """Return the list of dict-items for the given task type (structure-aware)."""
    if not isinstance(parsed, dict):
        return []
    key_by_type = {
        "quiz": "questions",
        "grammar": "grammar_patterns",
        "expressions": "expressions",
        "vocabulary": "vocabulary",
        "summary": "concepts",
        "mindmap": "branches",
    }
    preferred = key_by_type.get(task_type)
    candidates = ([preferred] if preferred else []) + [k for k in ITEM_KEYS if k != preferred]
    for key in candidates:
        value = parsed.get(key)
        if isinstance(value, list):
            return [it for it in value if isinstance(it, dict)]
    return []


def _extract_json(text: Any) -> Any:
    """Best-effort JSON extraction: direct, fenced ```json, or first balanced brace."""
    if not text or not isinstance(text, str):
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    fenced = re.search(r"```json\s*\n(.*?)\n```", text, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except Exception:
            pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except Exception:
                continue
    return None


def _score_schema(parsed: Any, raw_response: str = ""):
    """Dimension 1 (0–25). Always applicable."""
    if isinstance(parsed, dict):
        if len(parsed) == 0:
            return 15.0, ["⚠️ Valid JSON but empty object"]
        
        # Check if raw response was truncated or required brace balancing
        flags = []
        deduction = 0.0
        if raw_response and isinstance(raw_response, str):
            trimmed = raw_response.strip()
            # If closing brace was missing or raw string didn't end properly
            if trimmed.count("{") > trimmed.count("}") or trimmed.count("[") > trimmed.count("]"):
                deduction += 5.0
                flags.append("⚠️ Structural repair: unclosed braces/brackets in raw output")
        
        score = max(0.0, W_SCHEMA - deduction)
        return score, flags
    return 0.0, ["❌ Invalid or missing JSON output"]


def _score_verbatim(items: List[Dict[str, Any]], task_type: str, user_prompt: str):
    """Dimension 2 (0–30). Applies to extraction tasks only; N/A (no penalty) otherwise."""
    if task_type not in ("vocabulary", "expressions", "grammar"):
        return W_VERBATIM, []  # N/A -> no penalty (e.g. quizzes generate questions)
    flags: List[str] = []
    source = _extract_source_content(user_prompt)
    norm_src = _normalize_text(source)
    checks = matches = 0
    for item in items:
        quote = item.get("quoted_sentence") or item.get("quote")
        if not (quote and isinstance(quote, str)):
            continue
        checks += 1
        norm_quote = _normalize_text(quote)
        if norm_quote and (norm_quote in norm_src or _token_coverage(norm_quote, norm_src) >= 0.8):
            matches += 1
        else:
            flags.append(f"⚠️ Non-verbatim quote detected: '{quote[:40]}...'")
    if checks == 0:
        return W_VERBATIM, flags  # nothing checkable -> no penalty
    return round((matches / checks) * W_VERBATIM, 1), flags


def _score_pedagogy(items: List[Dict[str, Any]], task_type: str):
    """Dimension 3 (0–25). Evaluates pedagogical quality across extraction and assessment types."""
    if task_type not in ("vocabulary", "expressions", "grammar", "quiz", "summary", "mindmap"):
        return W_PEDAGOGY, []
    if not items:
        return W_PEDAGOGY, []
    flags: List[str] = []
    checks = passes = 0
    for item in items:
        if task_type == "vocabulary":
            word = str(item.get("word", "")).strip()
            pos = str(item.get("part_of_speech", "")).strip().lower()
            definition = str(item.get("definition", "")).strip()
            example = str(item.get("example_usage", "")).strip()
            quote = str(item.get("quoted_sentence", "")).strip()
            checks += 1
            valid_word = bool(re.search(r"^[A-Za-z\s\-']+$", word) and len(word) >= 2)
            valid_pos = pos in ("noun", "verb", "adjective", "adverb", "preposition", "conjunction")
            original_example = bool(example and example.lower() != quote.lower())
            if valid_word and valid_pos and definition and original_example:
                passes += 1
            else:
                reasons = []
                if not valid_word: reasons.append("invalid headword")
                if not valid_pos: reasons.append("invalid PoS")
                if not definition: reasons.append("missing definition")
                if not original_example: reasons.append("non-original example usage")
                flags.append(f"⚠️ Vocabulary item '{word}' failed pedagogy check: {', '.join(reasons)}")
        elif task_type == "expressions":
            word = str(item.get("word", "")).strip()
            checks += 1
            is_multiword = bool(re.search(r"\[.+?\]|one's", word, re.IGNORECASE) or len(word.split()) > 1)
            is_trivial = word.lower() in ("talk", "listen", "turn", "watch", "sit down", "talk to", "listen to", "look at")
            if is_multiword and not is_trivial:
                passes += 1
            else:
                flags.append(f"⚠️ Expression lacks multi-word/slot form or is too basic: '{word}'")
        elif task_type == "grammar":
            pattern = str(item.get("pattern_formula", ""))
            audit = str(item.get("design_audit", "")).strip()
            checks += 1
            if re.search(r"\[.+?\]", pattern) and audit:
                passes += 1
            else:
                flags.append("⚠️ Grammar pattern missing slot formula or design audit")
        elif task_type == "quiz":
            options = item.get("options", [])
            idx = item.get("correct_answer_index")
            explanation = str(item.get("explanation", "")).strip()
            question = str(item.get("question", item.get("translated_sentence", ""))).strip()
            checks += 1
            valid_options = isinstance(options, list) and len(options) == 4 and len(set(options)) == 4
            valid_idx = isinstance(idx, int) and 0 <= idx <= 3
            if valid_options and valid_idx and explanation and question:
                passes += 1
            else:
                reasons = []
                if not valid_options: reasons.append("options not 4 distinct items")
                if not valid_idx: reasons.append("invalid answer index")
                if not explanation: reasons.append("missing explanation")
                if not question: reasons.append("missing question text")
                flags.append(f"⚠️ Quiz question failed pedagogy check: {', '.join(reasons)}")
        elif task_type == "summary":
            name = str(item.get("concept_name", "")).strip()
            sig = str(item.get("educational_significance", "")).strip()
            details = item.get("key_details", [])
            checks += 1
            if name and sig and isinstance(details, list) and len(details) > 0:
                passes += 1
            else:
                flags.append(f"⚠️ Concept '{name}' missing educational significance or key details")
        elif task_type == "mindmap":
            name = str(item.get("branch_name", "")).strip()
            checks += 1
            if name:
                passes += 1
            else:
                flags.append("⚠️ MindMap branch missing branch name")
    if checks == 0:
        if task_type in ("vocabulary", "expressions", "grammar", "quiz", "summary", "mindmap"):
            return 0.0, [f"⚠️ Output list for '{task_type}' is empty."]
        return W_PEDAGOGY, []
    return round((passes / checks) * W_PEDAGOGY, 1), flags


def _score_uniqueness(items: List[Dict[str, Any]], task_type: str):
    """Dimension 4 (0–20). Deduplicate by the task's primary identifying field."""
    key_by_type = {
        "vocabulary": ("word",),
        "expressions": ("word",),
        "grammar": ("pattern_formula", "quote"),
        "quiz": ("question", "translated_sentence", "target_word", "correct_english_answer"),
        "summary": ("concept_name",),
        "mindmap": ("branch_name",),
    }
    keys = key_by_type.get(task_type, ("word", "quote", "question"))
    headwords = []
    for item in items:
        for key in keys:
            value = item.get(key)
            if value:
                headwords.append(str(value).strip().lower())
                break
    if not headwords:
        return W_UNIQUENESS, []
    unique = len(set(headwords))
    if unique < len(headwords):
        dup = len(headwords) - unique
        return round((unique / len(headwords)) * W_UNIQUENESS, 1), [f"❌ Found {dup} duplicate item(s)"]
    return W_UNIQUENESS, []


class LogEvaluator:
    """
    Evaluates LLM execution logs across 4 pedagogical quality dimensions:
    1. Schema & Structural Adherence (25%)
    2. Source Faithfulness & Verbatim Verification (30%)
    3. Pedagogical & Slot-Filling Quality (25%)
    4. Uniqueness & Deduplication (20%)

    Scoring convention: a dimension that does not apply to a given task type
    contributes its full weight (no penalty), so the composite stays in 0–100
    and the output shape stays stable for all consumers (CLI, dashboard).
    """

    # Cache for repeated audits (the dashboard polls /api/hero-board on every load).
    _AUDIT_CACHE: Dict[str, Any] = {"key": None, "value": None}

    @classmethod
    def parse_log_file(cls, log_path: Path) -> Optional[Dict[str, Any]]:
        """Parses a single .log file and extracts metadata, prompt content, and JSON response."""
        try:
            content = log_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return None

        task_match = re.search(r"=== TASK:\s*(.+?)\s*===", content)
        model_match = re.search(r"=== MODEL:\s*(.+?)\s*===", content)
        time_match = re.search(r"=== TIMESTAMP:\s*(.+?)\s*===", content)

        if not task_match or not model_match:
            return None

        task = task_match.group(1).strip()
        model = model_match.group(1).strip()
        timestamp = time_match.group(1).strip() if time_match else ""

        # Extract USER PROMPT content block
        user_prompt = ""
        user_prompt_match = re.search(r"--- USER PROMPT ---\n(.*?)(?=\n--- RAW RESPONSE ---|\n===|\Z)", content, re.DOTALL)
        if user_prompt_match:
            user_prompt = user_prompt_match.group(1).strip()

        # Extract RAW RESPONSE
        raw_response = ""
        response_match = re.search(r"--- RAW RESPONSE ---\n(.*?)(?=\n===|\Z)", content, re.DOTALL)
        if response_match:
            raw_response = response_match.group(1).strip()

        # Parse JSON (direct, fenced ```json, or first balanced brace)
        parsed_json = _extract_json(raw_response)

        return {
            "log_name": log_path.name,
            "task": task,
            "model": model,
            "timestamp": timestamp,
            "user_prompt": user_prompt,
            "raw_response": raw_response,
            "parsed_json": parsed_json,
        }

    @classmethod
    def evaluate_log(cls, log_data: Dict[str, Any]) -> Dict[str, Any]:
        """Runs the 4-dimension audit on a parsed log entry."""
        log_name = log_data["log_name"]
        task = log_data["task"]
        model = log_data["model"]
        user_prompt = log_data["user_prompt"]
        parsed = log_data["parsed_json"]

        raw_response = log_data.get("raw_response", "")
        task_type = _detect_task_type(task, parsed)
        items = _extract_items(parsed, task_type)

        flags: List[str] = []
        schema_score, f1 = _score_schema(parsed, raw_response)
        verbatim_score, f2 = _score_verbatim(items, task_type, user_prompt)
        pedagogy_score, f3 = _score_pedagogy(items, task_type)
        uniqueness_score, f4 = _score_uniqueness(items, task_type)
        flags.extend(f1 + f2 + f3 + f4)

        # Total Composite Score (0–100)
        composite_score = round(
            schema_score + verbatim_score + pedagogy_score + uniqueness_score, 1
        )

        return {
            "log_name": log_name,
            "task": task,
            "model": model,
            "composite_score": composite_score,
            "scores": {
                "schema_adherence": schema_score,
                "verbatim_faithfulness": verbatim_score,
                "pedagogical_quality": pedagogy_score,
                "uniqueness": uniqueness_score,
            },
            "flags": flags,
        }

    @classmethod
    def audit_all_logs(cls, logs_dir: Path = None, use_cache: bool = True) -> Dict[str, Any]:
        """
        Scans and audits all .log files in the logs/ directory.
        Returns the overall Hero Board leaderboard and individual log audits.
        Results are cached by the set of (name, mtime, size) so repeated calls
        (e.g. the dashboard /api/hero-board endpoint) stay cheap.
        """
        if logs_dir is None:
            logs_dir = config.project_root / "logs"
        logs_dir = Path(logs_dir)

        if not logs_dir.exists():
            return {"hero_board": [], "audits": []}

        entries = []
        for path in logs_dir.glob("*.log"):
            try:
                st = path.stat()
            except OSError:
                continue
            entries.append((path, st.st_mtime, st.st_size))

        if not entries:
            return {"hero_board": [], "audits": []}

        cache_key = [logs_dir.as_posix()] + [[p.name, mt, sz] for (p, mt, sz) in entries]
        if use_cache and cls._AUDIT_CACHE["key"] == cache_key and cls._AUDIT_CACHE["value"] is not None:
            return cls._AUDIT_CACHE["value"]

        # Newest first
        entries.sort(key=lambda e: e[1], reverse=True)

        audits: List[Dict[str, Any]] = []
        model_stats: Dict[str, Dict[str, Any]] = {}

        for log_file, _mtime, _size in entries:
            parsed = cls.parse_log_file(log_file)
            if not parsed:
                continue

            evaluation = cls.evaluate_log(parsed)
            audits.append(evaluation)

            model = evaluation["model"]
            if model not in model_stats:
                model_stats[model] = {
                    "model": model,
                    "runs": 0,
                    "total_score": 0.0,
                    "schema_sum": 0.0,
                    "verbatim_sum": 0.0,
                    "pedagogy_sum": 0.0,
                    "uniqueness_sum": 0.0,
                }

            s = model_stats[model]
            s["runs"] += 1
            s["total_score"] += evaluation["composite_score"]
            s["schema_sum"] += evaluation["scores"]["schema_adherence"]
            s["verbatim_sum"] += evaluation["scores"]["verbatim_faithfulness"]
            s["pedagogy_sum"] += evaluation["scores"]["pedagogical_quality"]
            s["uniqueness_sum"] += evaluation["scores"]["uniqueness"]

        # Build Hero Board Leaderboard
        hero_board = []
        for model, s in model_stats.items():
            runs = s["runs"]
            hero_board.append({
                "model": model,
                "runs": runs,
                "composite_score": round(s["total_score"] / runs, 1),
                "schema_adherence_avg": round((s["schema_sum"] / (runs * W_SCHEMA)) * 100.0, 1),
                "verbatim_faithfulness_avg": round((s["verbatim_sum"] / (runs * W_VERBATIM)) * 100.0, 1),
                "pedagogical_quality_avg": round((s["pedagogy_sum"] / (runs * W_PEDAGOGY)) * 100.0, 1),
                "uniqueness_avg": round((s["uniqueness_sum"] / (runs * W_UNIQUENESS)) * 100.0, 1),
            })

        # Sort Leaderboard descending by composite score
        hero_board.sort(key=lambda x: x["composite_score"], reverse=True)

        result = {"hero_board": hero_board, "audits": audits}
        cls._AUDIT_CACHE["key"] = cache_key
        cls._AUDIT_CACHE["value"] = result
        return result
