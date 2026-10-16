import os
import datetime
import json
import re
from pathlib import Path
from .config import config

def log_task(task_name, system_prompt, user_prompt, response_text, schema=None, status="SUCCESS", failure_category=None, mode=None, api_constraint=None, duration=None, start_time=None, end_time=None, model=None, context_prompt=None):
    """
    Logs an LLM task to the logs directory with structured status, true API constraints, timing metrics, and failure categorization.
    """
    # Ensure we use an absolute path for logs
    logs_dir = Path(config.project_root).resolve() / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    finish_dt = end_time if isinstance(end_time, datetime.datetime) else datetime.datetime.now()
    timestamp = finish_dt.strftime("%Y%m%d_%H%M%S_%f")[:-3]
    # Sanitize task_name for filesystem compatibility (eliminate Windows NTFS stream colons ':', slashes, etc.)
    safe_task_name = re.sub(r'[\\/:*?"<>|\r\n]+', '_', str(task_name)).strip('_')
    filename = f"{timestamp}_{safe_task_name}.log"
    log_path = logs_dir / filename

    actual_model = model or config.get('model') or 'unknown'

    content = []
    content.append(f"=== TASK: {task_name} ===")
    content.append(f"=== MODEL: {actual_model} ===")
    content.append(f"=== TIMESTAMP: {finish_dt.ctime()} ===")
    if duration is not None:
        start_str = start_time.strftime("%H:%M:%S") if isinstance(start_time, datetime.datetime) else "N/A"
        end_str = finish_dt.strftime("%H:%M:%S")
        content.append(f"=== DURATION: {duration:.1f}s (Started: {start_str}, Finished: {end_str}) ===")
    content.append(f"=== STATUS: {status} ===")
    if mode:
        content.append(f"=== MODE: {mode} ===")
    if failure_category:
        content.append(f"=== FAILURE_CATEGORY: {failure_category} ===")

    # Add Level 1 deterministic code gate evaluation score breakdown
    try:
        from .evaluator import LogEvaluator, _extract_json
        parsed = _extract_json(response_text)
        if parsed:
            simulated = {
                "log_name": filename,
                "task": task_name,
                "model": actual_model,
                "user_prompt": user_prompt,
                "system_prompt": system_prompt,
                "context_prompt": context_prompt or (f"{system_prompt}\n{user_prompt}".strip() if system_prompt else None),
                "raw_response": response_text,
                "parsed_json": parsed,
                "status": status,
                "failure_category": failure_category,
            }
            audit_res = LogEvaluator.evaluate_log(simulated)
            if audit_res:
                task_t = audit_res.get("task_type", "")
                is_audit_task = (task_t == "expert_audit" or "expert_audit" in (task_name or "").lower())
                stage_label = "L2_EXPERT_AUDIT" if is_audit_task else "L1_DETERMINISTIC_GATE"
                score_label = "L2_QUALITY_SCORE" if is_audit_task else "L1_STRUCTURAL_SCORE"
                content.append(f"=== EVALUATION_STAGE: {stage_label} ===")
                content.append(f"=== {score_label}: {audit_res.get('composite_score')}% ===")
                content.append(f"=== COMPOSITE_SCORE: {audit_res.get('composite_score')}% ===")
                scores = audit_res.get("scores", {})
                content.append("=== DIMENSION_SCORES: " + json.dumps(scores, ensure_ascii=False) + " ===")
                if audit_res.get("flags"):
                    flags_str = "\n".join(f"  {f}" for f in audit_res["flags"])
                    content.append(f"=== QUALITY_FLAGS ===\n{flags_str}")
    except Exception:
        pass

    content.append("")
    
    # 1. Truthful representation of wire-level API constraints
    # - In STRICT_SCHEMA: The schema is sent as an API payload constraint (token-level masking)
    # - In JSON_MODE: The API constraint is format: "json", and the schema was injected into prompt
    if mode == "STRICT_SCHEMA" or (api_constraint and isinstance(api_constraint, dict)):
        content.append("--- API SCHEMA CONSTRAINT ---")
        constraint_dict = api_constraint if isinstance(api_constraint, dict) else schema
        if constraint_dict:
            content.append(json.dumps(constraint_dict, indent=2, ensure_ascii=False))
            content.append("")
    elif mode == "JSON_MODE" or api_constraint == "json":
        content.append("--- API FORMAT CONSTRAINT: \"json\" ---")
        content.append("")
    elif schema and not mode:
        # Fallback for backward compatibility
        content.append("--- JSON SCHEMA ---")
        content.append(json.dumps(schema, indent=2, ensure_ascii=False))
        content.append("")

    if system_prompt:
        content.append("--- SYSTEM PROMPT ---")
        content.append(system_prompt)
        content.append("")

    content.append("--- USER PROMPT ---")
    content.append(user_prompt)
    content.append("")

    content.append("--- RAW RESPONSE ---")
    # Pretty-print JSON for human readability if valid JSON
    formatted_response = response_text
    if response_text and isinstance(response_text, str):
        trimmed = response_text.strip()
        try:
            data = json.loads(trimmed)
            formatted_response = json.dumps(data, indent=2, ensure_ascii=False)
        except Exception:
            # Handle markdown codeblock ```json ... ```
            if trimmed.startswith("```"):
                lines = trimmed.splitlines()
                if len(lines) >= 3 and lines[-1].strip() == "```":
                    inner = "\n".join(lines[1:-1])
                    try:
                        data = json.loads(inner)
                        formatted_response = f"```json\n{json.dumps(data, indent=2, ensure_ascii=False)}\n```"
                    except Exception:
                        pass
    content.append(formatted_response)
    content.append("")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(content))

    return str(log_path)


def log_wsd_session(unit_name, model, records, duration=None, start_time=None, end_time=None):
    """
    Writes a single consolidated WSD arbitration log for a unit extraction session.
    """
    if not records:
        return None

    logs_dir = Path(config.project_root).resolve() / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    finish_dt = end_time if isinstance(end_time, datetime.datetime) else datetime.datetime.now()
    timestamp = finish_dt.strftime("%Y%m%d_%H%M%S_%f")[:-3]
    safe_unit = re.sub(r'[\\/:*?"<>|\r\n]+', '_', str(unit_name or "unit")).strip('_') or "unit"
    task_name = f"wsd_arbitration_{safe_unit}"
    filename = f"{timestamp}_{task_name}.log"
    log_path = logs_dir / filename

    actual_model = model or config.get('model') or 'unknown'
    n_total = len(records)
    n_cached = sum(1 for r in records if r.get("cached"))
    n_llm = n_total - n_cached

    content = []
    content.append(f"=== TASK: {task_name} ===")
    content.append(f"=== MODEL: {actual_model} ===")
    content.append(f"=== TIMESTAMP: {finish_dt.ctime()} ===")
    if duration is not None:
        start_str = start_time.strftime("%H:%M:%S") if isinstance(start_time, datetime.datetime) else "N/A"
        end_str = finish_dt.strftime("%H:%M:%S")
        content.append(f"=== DURATION: {duration:.1f}s (Started: {start_str}, Finished: {end_str}) ===")
    content.append("=== STATUS: SUCCESS ===")
    content.append("=== MODE: WSD_ARBITRATION ===")
    content.append(f"=== SUMMARY: {n_total} items arbitrated ({n_llm} LLM calls, {n_cached} cache hits) ===")
    content.append("")
    content.append("--- USER PROMPT ---")
    content.append(f"Consolidated Top-K Sense Disambiguation for unit '{safe_unit}' ({n_total} items):")
    content.append("")

    all_letters = ["A", "B", "C", "D", "E", "F", "G", "H"]
    for idx, rec in enumerate(records, 1):
        item_name = rec.get("item_name", "")
        sentence = rec.get("sentence", "")
        cands = rec.get("candidates", [])
        chosen_idx = rec.get("chosen_idx", 0)
        letters = all_letters[:max(len(cands), 1)]
        chosen_letter = letters[chosen_idx] if 0 <= chosen_idx < len(letters) else "A"
        chosen_cand = cands[chosen_idx] if 0 <= chosen_idx < len(cands) else {}
        chosen_def = chosen_cand.get("definition", "")
        chosen_pos = chosen_cand.get("pos", "")
        cached_tag = "CACHED (0.00s)" if rec.get("cached") else f"LLM ({rec.get('duration', 0.0):.2f}s, raw={rec.get('raw_reply', '')!r})"

        content.append(f"[{idx:02d}/{n_total:02d}] Target: '{item_name}' | Status: {cached_tag}")
        content.append(f"  Sentence: \"{sentence}\"")
        for c_i, c in enumerate(cands):
            let = letters[c_i] if c_i < len(letters) else str(c_i)
            mark = "👉" if c_i == chosen_idx else "  "
            p_str = f"({c.get('pos', '')}) " if c.get('pos') else ""
            raw_s = c.get("raw_sense") if isinstance(c.get("raw_sense"), dict) else {}
            gram_raw = str(raw_s.get("grammar") or "").strip()
            d_str = c.get('definition', '').strip()
            g_str = f"{gram_raw} " if gram_raw in ("[plural]", "[singular]", "[only plural]") and gram_raw.lower() not in d_str.lower() else ""
            content.append(f"  {mark} {let}. {p_str}{g_str}{d_str}")
        content.append(f"  Selected: {chosen_letter} -> ({chosen_pos}) {chosen_def}")
        content.append("")

    content.append("--- RAW RESPONSE ---")
    summary_payload = {
        "unit": safe_unit,
        "total_arbitrated": n_total,
        "llm_calls": n_llm,
        "cache_hits": n_cached,
        "decisions": [
            {
                "item": r.get("item_name", ""),
                "chosen_letter": all_letters[r.get("chosen_idx", 0)] if 0 <= r.get("chosen_idx", 0) < len(all_letters) else "A",
                "chosen_pos": (r.get("candidates", [{}])[r.get("chosen_idx", 0)].get("pos", "") if r.get("candidates") and 0 <= r.get("chosen_idx", 0) < len(r.get("candidates", [])) else ""),
                "chosen_definition": (r.get("candidates", [{}])[r.get("chosen_idx", 0)].get("definition", "") if r.get("candidates") and 0 <= r.get("chosen_idx", 0) < len(r.get("candidates", [])) else ""),
                "cached": bool(r.get("cached", False)),
            }
            for r in records
        ],
    }
    content.append(json.dumps(summary_payload, indent=2, ensure_ascii=False))
    content.append("")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(content))

    return str(log_path)

