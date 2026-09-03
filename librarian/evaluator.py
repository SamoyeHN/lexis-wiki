import re
import json
import unicodedata
import threading
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, get_args

from .config import config
from .schemas import PARTS_OF_SPEECH

# --- Dimension weights (max points per dimension). Single source of truth. ---
W_SCHEMA = 25.0
W_VERBATIM = 30.0
W_PEDAGOGY = 25.0
W_UNIQUENESS = 20.0

# Valid Part of Speech enum set derived directly from schemas.py
VALID_POS_SET = set(get_args(PARTS_OF_SPEECH))

# Top-level arrays that may hold gradeable items, in priority order.
ITEM_KEYS = ["vocabulary", "expressions", "grammar_patterns", "questions", "concepts", "branches"]


def _normalize_text(value: Any) -> str:
    """Normalize unicode, strip markdown formatting, brackets, ellipses, quotes, and whitespace."""
    if not isinstance(value, str):
        return ""
    text = unicodedata.normalize("NFKC", value).lower()
    # Remove markdown bold/italics/code/strikethrough markers
    text = re.sub(r"[\*\_`~]+", " ", text)
    # Remove ellipses and dots
    text = re.sub(r"(\.{2,}|…)", " ", text)
    # Remove quotes and apostrophes directly to keep words intact
    text = re.sub(r"[“”‘’\"'«»]+", "", text)
    # Remove structural brackets/parentheses with whitespace separation
    text = re.sub(r"[\[\]\(\)\{\}\<\>]+", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _clean_core(text: str) -> str:
    """Strip all punctuation and non-alphanumeric chars (retaining spaces)."""
    norm = _normalize_text(text)
    return re.sub(r"[^\w\s]", "", norm).strip()


def _ngram_coverage(quote_clean: str, source_clean: str, n: int = 3) -> float:
    """Calculates n-gram overlap between cleaned quote and source text."""
    q_words = quote_clean.split()
    s_words = source_clean.split()
    if not q_words or not s_words:
        return 0.0
    if len(q_words) < n:
        if quote_clean in source_clean:
            return 1.0
        s_set = set(s_words)
        return sum(1 for w in q_words if w in s_set) / len(q_words)
    q_ngrams = set(tuple(q_words[i:i + n]) for i in range(len(q_words) - n + 1))
    s_ngrams = set(tuple(s_words[i:i + n]) for i in range(len(s_words) - n + 1))
    if not q_ngrams:
        return 0.0
    return len(q_ngrams.intersection(s_ngrams)) / len(q_ngrams)


def _is_hallucinated_quote(quote: str) -> bool:
    """Detect if model explicitly notes quote is inferred or missing from text."""
    q_lower = quote.lower()
    indicators = [
        "not present in text",
        "not in text",
        "not found in text",
        "not in source",
        "inferred from",
        "implied by",
        "constructed from",
        "not explicitly mentioned",
    ]
    return any(ind in q_lower for ind in indicators)


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
    # Strip <think>...</think> reasoning blocks if present
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
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


def _score_schema(parsed: Any, raw_response: str = "") -> Tuple[Optional[float], List[str]]:
    """Dimension 1 (0–25). Always applicable."""
    if isinstance(parsed, dict):
        if len(parsed) == 0:
            return 0.0, ["❌ Valid JSON but empty object"]
        
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


def _score_verbatim(items: List[Dict[str, Any]], task_type: str, user_prompt: str) -> Tuple[Optional[float], List[str]]:
    """Dimension 2 (0–30). Applies to extraction tasks only; returns (None, []) for non-extraction tasks."""
    if task_type not in ("vocabulary", "expressions", "grammar"):
        return None, []  # N/A -> normalized out of composite score
    if not items:
        return 0.0, ["❌ No items to evaluate for source faithfulness"]
    
    flags: List[str] = []
    source = _extract_source_content(user_prompt)
    core_src = _clean_core(source)
    checks = matches = 0
    
    for item in items:
        quote = item.get("quoted_sentence") or item.get("quote")
        word = str(item.get("word") or item.get("pattern_formula") or "").strip()
        if not (quote and isinstance(quote, str)):
            continue
            
        checks += 1
        
        # Check 1: Explicit hallucination acknowledgment
        if _is_hallucinated_quote(quote):
            flags.append(f"❌ Hallucinated quote (explicitly inferred/absent): '{quote[:50]}...'")
            continue

        # Check 2: Target word must be present in the quoted sentence (for vocabulary & expressions)
        if word and task_type in ("vocabulary", "expressions"):
            clean_word_no_slots = re.sub(r"\[.*?\]|\(.*?\)", " ", word)
            clean_word = _clean_core(clean_word_no_slots)
            clean_quote = _clean_core(quote)
            stop_slots = {"something", "somebody", "ones", "someone", "sb", "sth", "entity", "field", "area", "role", "object", "domain", "type", "situation", "goal"}
            word_tokens = [w for w in clean_word.split() if w and w not in stop_slots]
            
            if word_tokens:
                # Match token or stem/inflection (e.g. degrade -> degradation, took -> take, pose -> poses)
                matched_count = sum(1 for w in word_tokens if (w in clean_quote or (len(w) >= 4 and w[:4] in clean_quote)))
                min_needed = max(1, len(word_tokens) // 2 + (1 if len(word_tokens) % 2 == 1 else 0))
                word_in_quote = (matched_count >= min_needed)
            else:
                word_in_quote = (clean_word in clean_quote)
                
            if not word_in_quote:
                flags.append(f"⚠️ Target word '{word}' does not appear in quoted sentence: '{quote[:40]}...'")
                continue

        # Check 3: Cleaned quote in source or high n-gram coverage
        core_quote = _clean_core(quote)
        if core_quote and (core_quote in core_src or _ngram_coverage(core_quote, core_src, n=3) >= 0.85):
            matches += 1
        else:
            flags.append(f"⚠️ Non-verbatim quote detected: '{quote[:40]}...'")
            
    if checks == 0:
        return W_VERBATIM, flags
    return round((matches / checks) * W_VERBATIM, 1), flags


def _score_pedagogy(items: List[Dict[str, Any]], task_type: str) -> Tuple[Optional[float], List[str]]:
    """Dimension 3 (0–25). Evaluates pedagogical quality across extraction and assessment types."""
    if task_type not in ("vocabulary", "expressions", "grammar", "quiz", "summary", "mindmap"):
        return None, []
    if not items:
        return 0.0, [f"⚠️ Output list for '{task_type}' is empty."]
        
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
            
            # Allow words/phrases with slots, hyphens, brackets, parentheses, apostrophes
            valid_word = bool(re.search(r"^[A-Za-z\s\-'\[\]\(\)]+$", word) and len(word) >= 2)
            valid_pos = pos in VALID_POS_SET
            original_example = bool(example and _clean_core(example) != _clean_core(quote))
            
            if valid_word and valid_pos and definition and original_example:
                passes += 1
            else:
                reasons = []
                if not valid_word: reasons.append(f"invalid headword '{word}'")
                if not valid_pos: reasons.append(f"invalid PoS '{pos}'")
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
        return 0.0, [f"⚠️ Output list for '{task_type}' is empty."]
    return round((passes / checks) * W_PEDAGOGY, 1), flags


def _score_uniqueness(items: List[Dict[str, Any]], task_type: str) -> Tuple[Optional[float], List[str]]:
    """Dimension 4 (0–20). Deduplicate by the task's primary identifying field."""
    if not items:
        return 0.0, ["❌ No items to evaluate for uniqueness"]
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
        return 0.0, ["⚠️ Missing identifying keys for uniqueness check"]
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

    Scoring convention:
    - Dimensions returning None (N/A) are excluded from composite calculation.
    - Composite Score is strictly normalized to 0–100 based on applicable dimensions.
    """

    # Cache & thread-lock for repeated audits (dashboard polls /api/hero-board).
    _AUDIT_LOCK = threading.Lock()
    _AUDIT_CACHE: Dict[str, Any] = {"key": None, "value": None}

    @classmethod
    def parse_log_file(cls, log_path: Path) -> Optional[Dict[str, Any]]:
        """Parses a single .log file and extracts metadata, status, prompt content, and JSON response."""
        try:
            content = log_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return None

        task_match = re.search(r"=== TASK:\s*(.+?)\s*===", content)
        model_match = re.search(r"=== MODEL:\s*(.+?)\s*===", content)
        time_match = re.search(r"=== TIMESTAMP:\s*(.+?)\s*===", content)
        status_match = re.search(r"=== STATUS:\s*(.+?)\s*===", content)
        fail_cat_match = re.search(r"=== FAILURE_CATEGORY:\s*(.+?)\s*===", content)

        if not task_match or not model_match:
            return None

        task = task_match.group(1).strip()
        model = model_match.group(1).strip()
        timestamp = time_match.group(1).strip() if time_match else ""
        status = status_match.group(1).strip() if status_match else "SUCCESS"
        failure_category = fail_cat_match.group(1).strip() if fail_cat_match else None

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
            "status": status,
            "failure_category": failure_category,
            "timestamp": timestamp,
            "user_prompt": user_prompt,
            "raw_response": raw_response,
            "parsed_json": parsed_json,
        }

    @classmethod
    def evaluate_log(cls, log_data: Dict[str, Any]) -> Dict[str, Any]:
        """Runs the 4-dimension audit on a parsed log entry."""
        log_name = log_data.get("log_name", "")
        task = log_data.get("task", "")
        model = log_data.get("model", "")
        status = log_data.get("status", "SUCCESS")
        failure_category = log_data.get("failure_category")
        user_prompt = log_data.get("user_prompt", "")
        parsed = log_data.get("parsed_json")

        raw_response = log_data.get("raw_response", "")
        task_type = _detect_task_type(task, parsed)
        items = _extract_items(parsed, task_type)

        flags: List[str] = []
        schema_score, f1 = _score_schema(parsed, raw_response)
        verbatim_score, f2 = _score_verbatim(items, task_type, user_prompt)
        pedagogy_score, f3 = _score_pedagogy(items, task_type)
        uniqueness_score, f4 = _score_uniqueness(items, task_type)
        flags.extend(f1 + f2 + f3 + f4)

        # Dimension scores & weights map
        dim_results = [
            ("schema_adherence", schema_score, W_SCHEMA),
            ("verbatim_faithfulness", verbatim_score, W_VERBATIM),
            ("pedagogical_quality", pedagogy_score, W_PEDAGOGY),
            ("uniqueness", uniqueness_score, W_UNIQUENESS),
        ]

        applicable_weight = sum(w for _, s, w in dim_results if s is not None)
        earned_score = sum(s for _, s, _ in dim_results if s is not None)

        if applicable_weight > 0:
            composite_score = round((earned_score / applicable_weight) * 100.0, 1)
        else:
            composite_score = 0.0

        # Sort flags: critical errors (❌) first, then warnings (⚠️)
        flags.sort(key=lambda x: (0 if x.startswith("❌") else 1))

        return {
            "log_name": log_name,
            "task": task,
            "task_type": task_type,
            "model": model,
            "status": status,
            "failure_category": failure_category,
            "n_items": len(items),
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
        Results are cached by MD5 hash of entries' metadata so repeated calls stay fast and thread-safe.
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

        # Build compact hash-based cache key
        sig_data = f"{logs_dir.as_posix()}:{len(entries)}:" + ":".join(f"{p.name}:{mt}:{sz}" for (p, mt, sz) in entries)
        cache_key = hashlib.md5(sig_data.encode("utf-8")).hexdigest()

        with cls._AUDIT_LOCK:
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
                    "success_runs": 0,
                    "failed_runs": 0,
                    "total_score": 0.0,
                    "schema_sum": 0.0,
                    "schema_runs": 0,
                    "verbatim_sum": 0.0,
                    "verbatim_runs": 0,
                    "pedagogy_sum": 0.0,
                    "pedagogy_runs": 0,
                    "uniqueness_sum": 0.0,
                    "uniqueness_runs": 0,
                }

            s = model_stats[model]
            s["runs"] += 1

            # Exclude FAILED status logs from the quality leaderboard composite score
            if evaluation.get("status") == "FAILED":
                s["failed_runs"] += 1
                continue

            s["success_runs"] += 1
            s["total_score"] += evaluation["composite_score"]

            sc = evaluation["scores"]
            if sc["schema_adherence"] is not None:
                s["schema_sum"] += sc["schema_adherence"]
                s["schema_runs"] += 1
            if sc["verbatim_faithfulness"] is not None:
                s["verbatim_sum"] += sc["verbatim_faithfulness"]
                s["verbatim_runs"] += 1
            if sc["pedagogical_quality"] is not None:
                s["pedagogy_sum"] += sc["pedagogical_quality"]
                s["pedagogy_runs"] += 1
            if sc["uniqueness"] is not None:
                s["uniqueness_sum"] += sc["uniqueness"]
                s["uniqueness_runs"] += 1

        # Build Hero Board Leaderboard
        hero_board = []
        for model, s in model_stats.items():
            succ = s["success_runs"]
            if succ == 0:
                comp = 0.0
                sch_avg = vbt_avg = ped_avg = unq_avg = 0.0
            else:
                comp = round(s["total_score"] / succ, 1)
                sch_avg = round((s["schema_sum"] / (s["schema_runs"] * W_SCHEMA)) * 100.0, 1) if s["schema_runs"] else 100.0
                vbt_avg = round((s["verbatim_sum"] / (s["verbatim_runs"] * W_VERBATIM)) * 100.0, 1) if s["verbatim_runs"] else 100.0
                ped_avg = round((s["pedagogy_sum"] / (s["pedagogy_runs"] * W_PEDAGOGY)) * 100.0, 1) if s["pedagogy_runs"] else 100.0
                unq_avg = round((s["uniqueness_sum"] / (s["uniqueness_runs"] * W_UNIQUENESS)) * 100.0, 1) if s["uniqueness_runs"] else 100.0

            hero_board.append({
                "model": model,
                "runs": s["runs"],
                "success_runs": succ,
                "failed_runs": s["failed_runs"],
                "composite_score": comp,
                "schema_adherence_avg": sch_avg,
                "verbatim_faithfulness_avg": vbt_avg,
                "pedagogical_quality_avg": ped_avg,
                "uniqueness_avg": unq_avg,
            })

        # Sort Leaderboard descending by composite score
        hero_board.sort(key=lambda x: x["composite_score"], reverse=True)

        result = {"hero_board": hero_board, "audits": audits}
        with cls._AUDIT_LOCK:
            cls._AUDIT_CACHE["key"] = cache_key
            cls._AUDIT_CACHE["value"] = result
        return result
