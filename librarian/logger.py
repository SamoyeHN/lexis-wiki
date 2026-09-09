import os
import datetime
import json
import re
from pathlib import Path
from .config import config

def log_task(task_name, system_prompt, user_prompt, response_text, schema=None, status="SUCCESS", failure_category=None, mode=None, api_constraint=None, duration=None, start_time=None, end_time=None, model=None):
    """
    Logs an LLM task to the logs directory with structured status, true API constraints, timing metrics, and failure categorization.
    """
    # Ensure we use an absolute path for logs
    logs_dir = Path(config.project_root).resolve() / "logs"
    logs_dir.mkdir(exist_ok=True)

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
                "raw_response": response_text,
                "parsed_json": parsed,
                "status": status,
                "failure_category": failure_category,
            }
            audit_res = LogEvaluator.evaluate_log(simulated)
            if audit_res:
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
