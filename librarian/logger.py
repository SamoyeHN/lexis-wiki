import os
import datetime
import json
from pathlib import Path
from .config import config

def log_task(task_name, system_prompt, user_prompt, response_text, schema=None, status="SUCCESS", failure_category=None, mode=None, api_constraint=None):
    """
    Logs an LLM task to the logs directory with structured status, true API constraints, and failure categorization.
    """
    # Ensure we use an absolute path for logs
    logs_dir = Path(config.project_root).resolve() / "logs"
    logs_dir.mkdir(exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    filename = f"{timestamp}_{task_name}.log"
    log_path = logs_dir / filename

    content = []
    content.append(f"=== TASK: {task_name} ===")
    content.append(f"=== MODEL: {config.get('model')} ===")
    content.append(f"=== TIMESTAMP: {datetime.datetime.now().ctime()} ===")
    content.append(f"=== STATUS: {status} ===")
    if mode:
        content.append(f"=== MODE: {mode} ===")
    if failure_category:
        content.append(f"=== FAILURE_CATEGORY: {failure_category} ===")
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
