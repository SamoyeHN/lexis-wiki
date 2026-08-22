import os
import datetime
import json
from pathlib import Path
from .config import config

def log_task(task_name, system_prompt, user_prompt, response_text, schema=None):
    """
    Logs an LLM task to the logs directory.
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
    content.append("")
    
    if system_prompt:
        content.append("--- SYSTEM PROMPT ---")
        content.append(system_prompt)
        content.append("")

    if schema:
        content.append("--- JSON SCHEMA ---")
        content.append(json.dumps(schema, indent=2, ensure_ascii=False))
        content.append("")

    content.append("--- USER PROMPT ---")
    content.append(user_prompt)
    content.append("")

    content.append("--- RAW RESPONSE ---")
    content.append(response_text)
    content.append("")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(content))

    return str(log_path)
