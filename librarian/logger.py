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
        if parsed or status == "FAILED":
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
    # - In STRICT_SCHEMA: The schema is sent as an API constraint (token-level masking)
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
    Follows the canonical log naming convention: `<timestamp>_extract_wsd_<Unit>_TEXT_MODE.log`.
    Records the exact, verbatim assembled WSD prompt for each item alongside the raw LLM reply and selected sense.
    """
    if not records:
        return None

    logs_dir = Path(config.project_root).resolve() / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    finish_dt = end_time if isinstance(end_time, datetime.datetime) else datetime.datetime.now()
    timestamp = finish_dt.strftime("%Y%m%d_%H%M%S_%f")[:-3]
    safe_unit = re.sub(r'[\\/:*?"<>|\r\n]+', '_', str(unit_name or "unit")).strip('_') or "unit"
    task_name = f"extract_wsd_{safe_unit}_TEXT_MODE"
    filename = f"{timestamp}_{task_name}.log"
    log_path = logs_dir / filename

    actual_model = model or config.get('model') or 'unknown'
    n_total = len(records)
    n_cached = sum(1 for r in records if r.get("cached"))
    n_llm = n_total - n_cached
    n_errors = sum(1 for r in records if r.get("error") or r.get("chosen_idx") is None)

    first_err = next((str(r.get("error")) for r in records if r.get("error")), "")
    is_timeout = "timeout" in first_err.lower() or "timed out" in first_err.lower()
    status_str = "FAILED" if (n_errors > 0 and (n_errors == n_total or is_timeout)) else "SUCCESS"
    fail_cat = None
    if status_str == "FAILED":
        fail_cat = "CONNECTION_TIMEOUT" if is_timeout else "API_ERROR"

    content = []
    content.append(f"=== TASK: {task_name} ===")
    content.append(f"=== MODEL: {actual_model} ===")
    content.append(f"=== TIMESTAMP: {finish_dt.ctime()} ===")
    if duration is not None:
        start_str = start_time.strftime("%H:%M:%S") if isinstance(start_time, datetime.datetime) else "N/A"
        end_str = finish_dt.strftime("%H:%M:%S")
        content.append(f"=== DURATION: {duration:.1f}s (Started: {start_str}, Finished: {end_str}) ===")
    content.append(f"=== STATUS: {status_str} ===")
    content.append("=== MODE: WSD_ARBITRATION ===")
    if fail_cat:
        content.append(f"=== FAILURE_CATEGORY: {fail_cat} ===")
    content.append(
        f"=== SUMMARY: {n_total} items arbitrated ({n_llm} LLM calls, {n_cached} cache hits, {n_errors} errors) ==="
    )
    content.append("")
    content.append("--- USER PROMPT ---")
    content.append(f"Consolidated Top-K Sense Disambiguation for unit '{safe_unit}' ({n_total} items):")
    content.append("")

    all_letters = [chr(ord("A") + i) for i in range(20)]
    for idx, rec in enumerate(records, 1):
        item_name = rec.get("item_name", "")
        sentence = rec.get("sentence", "")
        cands = rec.get("candidates", [])
        chosen_idx = rec.get("chosen_idx")
        letters = all_letters[:max(len(cands), 1)]
        err_msg = rec.get("error")
        if chosen_idx is not None and 0 <= chosen_idx < len(cands) and not err_msg:
            chosen_letter = letters[chosen_idx] if chosen_idx < len(letters) else str(chosen_idx)
            chosen_cand = cands[chosen_idx]
            chosen_def = chosen_cand.get("definition", "")
            chosen_pos = chosen_cand.get("pos", "")
        else:
            chosen_letter = None
            chosen_def = ""
            chosen_pos = ""
        if rec.get("cached"):
            status_tag = f"CACHED (0.00s, raw={rec.get('raw_reply', '')!r})"
        elif err_msg or chosen_letter is None:
            status_tag = f"SKIPPED ({rec.get('duration', 0.0):.2f}s, next_candidate_fallback, err={err_msg!r})"
        else:
            status_tag = f"LLM ({rec.get('duration', 0.0):.2f}s, raw={rec.get('raw_reply', '')!r})"

        content.append(f"{'=' * 76}")
        content.append(f"[{idx:02d}/{n_total:02d}] Target: '{item_name}' | Status: {status_tag}")
        content.append(f"{'-' * 76}")

        verbatim_prompt = str(rec.get("prompt") or "").strip()
        if verbatim_prompt:
            content.append(verbatim_prompt)
        else:
            ctx_clean = str(rec.get("discourse_context") or "").strip()
            if ctx_clean and ">>" in ctx_clean and "<<" in ctx_clean:
                content.append(f"Select the single best dictionary sense for '{item_name}'.")
                content.append("")
                content.append("Passage Context (target sentence in >>...<<):")
                content.append(f"\"{ctx_clean}\"")
            else:
                content.append(f"Select the single best dictionary sense for '{item_name}' as used in this sentence:")
                content.append(f"\"{sentence}\"")
                if ctx_clean:
                    content.append("")
                    content.append("Surrounding Context:")
                    content.append(f"\"{ctx_clean}\"")
            content.append("")
            content.append("Candidate Senses:")
            opts = rec.get("options_text") or []
            if opts:
                for opt_line in opts:
                    content.append(opt_line)
            else:
                for c_i, c in enumerate(cands):
                    let = letters[c_i] if c_i < len(letters) else str(c_i)
                    p_str = f"({c.get('pos', '')}) " if c.get('pos') else ""
                    d_str = c.get('definition', '').strip()
                    content.append(f"{let}. {p_str}{d_str}")

        content.append(f"{'-' * 76}")
        if chosen_letter is not None:
            content.append(f"👉 Selected: {chosen_letter} -> ({chosen_pos}) {chosen_def}")
        else:
            content.append(f"⚠️ Selected: NONE (Candidate Rejected/Skipped — Advancing to Next Candidate: {err_msg})")
        content.append("")

    content.append("--- RAW RESPONSE ---")
    summary_payload = {
        "unit": safe_unit,
        "status": status_str,
        "total_arbitrated": n_total,
        "llm_calls": n_llm,
        "cache_hits": n_cached,
        "errors": n_errors,
        "decisions": [
            {
                "item": r.get("item_name", ""),
                "chosen_letter": (
                    all_letters[r.get("chosen_idx")]
                    if isinstance(r.get("chosen_idx"), int) and 0 <= r.get("chosen_idx") < len(all_letters) and not r.get("error")
                    else None
                ),
                "chosen_pos": (
                    r.get("candidates", [{}])[r.get("chosen_idx")].get("pos", "")
                    if isinstance(r.get("chosen_idx"), int) and r.get("candidates") and 0 <= r.get("chosen_idx") < len(r.get("candidates", [])) and not r.get("error")
                    else ""
                ),
                "chosen_definition": (
                    r.get("candidates", [{}])[r.get("chosen_idx")].get("definition", "")
                    if isinstance(r.get("chosen_idx"), int) and r.get("candidates") and 0 <= r.get("chosen_idx") < len(r.get("candidates", [])) and not r.get("error")
                    else ""
                ),
                "raw_reply": r.get("raw_reply", ""),
                "cached": bool(r.get("cached", False)),
                **({"error": r.get("error")} if r.get("error") else {}),
            }
            for r in records
        ],
    }
    content.append(json.dumps(summary_payload, indent=2, ensure_ascii=False))
    content.append("")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(content))

    return str(log_path)
