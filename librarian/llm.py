import requests
import json
import re
import dataclasses
from pathlib import Path
from typing import List, Dict, Any, Type, Union, Optional
from .config import config
from .schemas import get_json_schema, validate_and_map

class LLMError(Exception):
    """Base exception for LLM related errors."""
    pass

def get_model_profile(model_name: str) -> dict:
    """
    Intelligently determines optimal execution profile for a model.
    Merges:
    1. Built-in model architecture heuristics (Qwen, Llama, Mistral, Gemma, DeepSeek, etc.)
    2. Explicit user overrides in wiki_config.json['model_options']
    3. Global wiki_config.json settings
    """
    m_lower = (model_name or "").lower()
    
    # Architectural Defaults
    strict_gbnf_families = ("qwen", "llama", "mistral", "mixtral", "codestral", "muse", "hermes")
    prompt_json_families = ("gemma", "nemotron", "phi")
    reasoning_families = ("ornith", "granite", "deepseek-r1", "qwq")
    
    inferred_gbnf = None
    if any(k in m_lower for k in strict_gbnf_families):
        inferred_gbnf = True
    elif any(k in m_lower for k in prompt_json_families):
        inferred_gbnf = False
        
    inferred_think = None
    if any(k in m_lower for k in reasoning_families):
        inferred_think = False
        
    # User Config Overrides (Highest Precedence)
    model_opts = config.get("model_options", {}).get(model_name, {})
    global_gbnf = config.get("enforce_gbnf", False)
    
    if "enforce_gbnf" in model_opts:
        resolved_gbnf = model_opts["enforce_gbnf"]
    elif inferred_gbnf is not None:
        resolved_gbnf = inferred_gbnf
    else:
        resolved_gbnf = global_gbnf
        
    if "think" in model_opts:
        resolved_think = model_opts["think"]
    elif inferred_think is not None:
        resolved_think = inferred_think
    else:
        resolved_think = None
        
    profile = dict(model_opts)
    profile["enforce_gbnf"] = resolved_gbnf
    if resolved_think is not None:
        profile["think"] = resolved_think
        
    return profile

class LLMClient:
    def __init__(self):
        self._refresh_config()
        self.last_raw_response = None
        self.last_done_reason = None

    def _refresh_config(self):
        """Refreshes configuration from the config object."""
        self.api_type = config.get("api_type") or "ollama"
        self.api_url = (config.get("api_url") or config.get("ollama_url") or "http://localhost:11434").rstrip('/')
        self.api_key = config.get("api_key") or "ollama"
        self.model = config.get("model")

    def list_models(self):
        """Fetches models. Only supported for Ollama for now."""
        self._refresh_config()
        if self.api_type != "ollama":
            return []
            
        url = f"{self.api_url}/api/tags"
        try:
            response = requests.get(url, timeout=2)
            response.raise_for_status()
            data = response.json()
            return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []

    def chat(self, messages, stream=False, json_format=True, schema=None, task_name=None, **kwargs):
        """
        Main chat interface. 
        Supports both streaming and non-streaming responses.
        If schema is provided, returns an instance of the schema dataclass.
        """
        self._refresh_config()
        self.last_raw_response = None
        self.last_done_reason = None

        # 1. OPTIMIZATION: Move Personas to System Role
        if messages and messages[0]["role"] == "user":
            content = messages[0]["content"]
            
            # 1.1 Priority: Explicit Tagged Blocks
            if "### SYSTEM ###" in content and "### USER ###" in content:
                parts = content.split("### SYSTEM ###", 1)
                # content before ### SYSTEM ### is ignored
                remaining = parts[1].strip()
                sys_text, user_text = remaining.split("### USER ###", 1)
                messages[0]["content"] = user_text.strip()
                messages.insert(0, {"role": "system", "content": sys_text.strip()})
            
            # 1.2 Secondary: Legacy '---' delimiter (only if no tags)
            elif "---" in content:
                parts = content.split("---", 1)
                messages[0]["content"] = parts[1].strip()
                messages.insert(0, {"role": "system", "content": parts[0].strip()})
            
            # 1.3 Tertiary: Fallback to 'You are a' pattern
            elif content.startswith("You are a"):
                parts = re.split(r'\n+', content, maxsplit=1)
                if len(parts) > 1:
                    messages[0]["content"] = parts[1]
                    messages.insert(0, {"role": "system", "content": parts[0]})

        # 2. OPTIMIZATION: Move SCHEMA GUIDANCE to System Role
        schema_guidance = ""
        for msg in messages:
            if msg["role"] == "user" and "### SCHEMA GUIDANCE ###" in msg["content"]:
                parts = msg["content"].split("### SCHEMA GUIDANCE ###")
                msg["content"] = parts[0].strip()
                schema_guidance = "### SCHEMA GUIDANCE ###\n" + parts[1].strip()
                break
        
        if schema_guidance:
            system_msg = next((m for m in messages if m["role"] == "system"), None)
            if system_msg:
                system_msg["content"] += "\n\n" + schema_guidance
            else:
                messages.insert(0, {"role": "system", "content": schema_guidance})

        # 3. Prompt-Guided JSON Mode: Inject JSON schema if GBNF is disabled
        profile = get_model_profile(self.model)
        use_gbnf = profile.get("enforce_gbnf", False)
        if schema and not use_gbnf:
            schema_dict = schema if isinstance(schema, dict) else get_json_schema(schema, include_descriptions=False)
            schema_json_str = json.dumps(schema_dict, indent=2, ensure_ascii=False)
            schema_prompt = f"### JSON SCHEMA REQUIREMENT ###\nRespond strictly with a valid JSON object matching this schema definition:\n```json\n{schema_json_str}\n```"
            system_msg = next((m for m in messages if m["role"] == "system"), None)
            if system_msg:
                if "### JSON SCHEMA REQUIREMENT ###" not in system_msg["content"]:
                    system_msg["content"] += f"\n\n{schema_prompt}"
            else:
                messages.insert(0, {"role": "system", "content": schema_prompt})

        # 3.1 Gemma-Family Compatibility: Flatten ALL system messages into user prompt
        # Google Gemma tokenizer only supports user and model turns. Multiple turns break alignment.
        is_gemma_family = "gemma" in (self.model or "").lower()
        if is_gemma_family:
            sys_msgs = [m["content"] for m in messages if m["role"] == "system"]
            user_msg = next((m for m in messages if m["role"] == "user"), None)
            if sys_msgs and user_msg:
                combined_sys = "\n\n".join(sys_msgs)
                user_msg["content"] = f"{combined_sys}\n\n{user_msg['content']}"
                messages = [m for m in messages if m["role"] != "system"]

        # 4. Handle Constraints
        schema_for_api = schema if use_gbnf else None
        force_json_mode = json_format or bool(schema and not use_gbnf)

        # 5. Call API
        if self.api_type == "openai":
            content = self._chat_openai(messages, stream, force_json_mode, schema_for_api, **kwargs)
        else:
            content = self._chat_ollama(messages, stream, force_json_mode, schema_for_api, **kwargs)

        # Prompts for logging
        mode_str = "STRICT_SCHEMA" if (schema and use_gbnf) else ("JSON_MODE" if force_json_mode else "TEXT_MODE")
        system_prompt = "\n".join([m["content"] for m in messages if m["role"] == "system"])
        user_prompt = "\n".join([m["content"] for m in messages if m["role"] == "user"])
        t_name = task_name or "chat"
        schema_dict = schema if isinstance(schema, dict) else (get_json_schema(schema, include_descriptions=False) if schema else None)

        # 6. Post-Processing & Parsing
        if schema and not stream:
            failure_cat = None
            try:
                # 6.0. Truncation check
                if getattr(self, "last_done_reason", None) == "length":
                    failure_cat = "TRUNCATED"
                    import logging
                    logging.getLogger("librarian").warning(
                        f"[TRUNCATION DETECTED] Generation hit context/token limit (done_reason='length') for task '{task_name or 'chat'}' on model '{self.model}'."
                    )

                # 6.1. CLEANING: Extract JSON from markdown or clutter
                json_str = content.strip()
                if not json_str or json_str in ("{}", "[]", "null"):
                    failure_cat = "EMPTY_RESPONSE"
                    raise LLMError(f"LLM returned empty/trivial response for task '{task_name or 'chat'}'.")
                
                # Remove ```json ... ``` blocks
                if "```" in json_str:
                    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', json_str, re.DOTALL)
                    if match:
                        json_str = match.group(1)
                    else:
                        # Fallback: remove backticks
                        json_str = re.sub(r'```[a-z]*\n?', '', json_str).replace('```', '')

                # Find outermost { and }
                start = json_str.find('{')
                end = json_str.rfind('}')
                if start != -1 and end != -1:
                    json_str = json_str[start:end+1]
                elif start != -1 and end == -1:
                    # Cut off before closing brace -> Truncated
                    failure_cat = "TRUNCATED"
                
                # 6.2. HEALING
                final_json = self._heal_json(json_str)
                was_healed = (final_json != json_str)
                
                try:
                    data = json.loads(final_json)
                except json.JSONDecodeError as jde:
                    # Light cleaning of unescaped newlines in values
                    final_json = re.sub(r'\n(?!\s*[, "\}\]\{\[0-9tfn\-\:])', r'\\n', final_json)
                    was_healed = True
                    try:
                        data = json.loads(final_json)
                    except json.JSONDecodeError:
                        failure_cat = failure_cat or "INVALID_JSON"
                        raise jde
                
                if was_healed:
                    import logging
                # Auto-sync slotted form from design_audit to word for expressions if needed
                if isinstance(data, dict) and "expressions" in data and isinstance(data["expressions"], list):
                    for expr_item in data["expressions"]:
                        if isinstance(expr_item, dict):
                            cur_word = expr_item.get("word", "").strip()
                            cur_audit = expr_item.get("design_audit", "")
                            if ("[" not in cur_word and "one's" not in cur_word) and ("[" in cur_audit or "one's" in cur_audit):
                                parts = [p.strip() for p in cur_audit.replace("->", "➔").split("➔")]
                                for p in parts:
                                    if ("[" in p or "one's" in p):
                                        cand = p[5:].strip().lstrip(':').strip() if p.upper().startswith("DRAFT") else p
                                        candidate = cand.split(" -")[0].split(" (")[0].strip()
                                        cand_tokens = re.findall(r'[a-zA-Z]+', candidate.replace("[", "").replace("]", ""))
                                        word_tokens = re.findall(r'[a-zA-Z]+', cur_word)
                                        if cand_tokens and word_tokens and cand_tokens[0].lower() == word_tokens[0].lower():
                                            expr_item["word"] = candidate
                                            break

                # 6.3. MAPPING
                if isinstance(schema, dict):
                    if isinstance(data, list):
                        props = schema.get("properties", {})
                        list_field = next((k for k, v in props.items() if v.get("type") == "array"), "items")
                        data = {list_field: data}
                        if "title" in props: data["title"] = task_name or "Untitled"
                try:
                    result_obj = validate_and_map(schema, data) if not isinstance(schema, dict) else data
                except Exception as map_err:
                    failure_cat = "SCHEMA_MISMATCH"
                    raise map_err

                # 6.4. QA EVALUATION & RETRY LOOP
                retry_count = kwargs.pop("_qa_retry_count", 0)
                if retry_count == 0 and not kwargs.get("_disable_qa_retry", False):
                    try:
                        from .evaluator import LogEvaluator
                        dict_to_eval = data if isinstance(data, dict) else (dataclasses.asdict(result_obj) if dataclasses.is_dataclass(result_obj) else None)
                        if dict_to_eval:
                            simulated_log = {
                                "log_name": f"{t_name}.log",
                                "task": t_name,
                                "model": self.model or "unknown",
                                "user_prompt": user_prompt,
                                "raw_response": final_json,
                                "parsed_json": dict_to_eval,
                            }
                            audit = LogEvaluator.evaluate_log(simulated_log)
                            composite = audit.get("composite_score", 100.0)
                            
                            if composite < 80.0:
                                flags = audit.get("flags", [])
                                scores = audit.get("scores", {})
                                lowest_dim = min(scores.keys(), key=lambda k: scores[k]) if scores else "pedagogy"
                                feedback_note = (
                                    f"\n\n[QUALITY AUDIT RETRY: Previous attempt scored {composite}/100. "
                                    f"Lowest dimension: {lowest_dim}. "
                                    f"Please address the following issues carefully: {'; '.join(flags[:3])}]"
                                )
                                import logging
                                logging.getLogger("librarian").info(
                                    f"QA score {composite}/100 < 80% for {t_name}. Retrying once with feedback..."
                                )
                                # Log failed attempt with QA_LOW_SCORE
                                from .logger import log_task
                                log_task(f"{t_name}_{mode_str}", system_prompt, user_prompt, final_json, schema=schema_dict, status="FAILED", failure_category="QA_LOW_SCORE")

                                retry_messages = [dict(m) for m in messages]
                                sys_msg = next((m for m in retry_messages if m["role"] == "system"), None)
                                if sys_msg:
                                    sys_msg["content"] += feedback_note
                                else:
                                    user_target = next((m for m in retry_messages if m["role"] == "user"), None)
                                    if user_target:
                                        user_target["content"] += feedback_note
                                    else:
                                        retry_messages.insert(0, {"role": "system", "content": feedback_note})
                                
                                return self.chat(
                                    retry_messages,
                                    stream=stream,
                                    json_format=json_format,
                                    schema=schema,
                                    task_name=task_name,
                                    _qa_retry_count=1,
                                    **kwargs
                                )
                    except Exception as eval_err:
                        import logging
                        logging.getLogger("librarian").warning(f"Evaluator check skipped due to error: {eval_err}")

                # 6.5. LOG FINAL HEALED & VALIDATED RESPONSE
                from .logger import log_task
                log_task(f"{t_name}_{mode_str}", system_prompt, user_prompt, final_json, schema=schema_dict, status="SUCCESS")

                return result_obj

            except Exception as e:
                import logging
                logging.getLogger("librarian").error(f"JSON Parsing / Schema Validation Failed for {task_name or 'chat'}: {e}")
                # Categorize failure if not already set
                if not failure_cat:
                    if not content or not content.strip():
                        failure_cat = "EMPTY_RESPONSE"
                    elif "JSONDecodeError" in type(e).__name__ or "json" in str(e).lower():
                        failure_cat = "INVALID_JSON"
                    else:
                        failure_cat = "SCHEMA_MISMATCH"

                # Log even on parse failure with failure categorization so it is auditable
                from .logger import log_task
                log_task(f"{t_name}_{mode_str}", system_prompt, user_prompt, content, schema=schema_dict, status="FAILED", failure_category=failure_cat)
                raise LLMError(f"Failed to generate structured data matching schema: {e}")
        else:
            if not stream:
                from .logger import log_task
                log_task(f"{t_name}_{mode_str}", system_prompt, user_prompt, content, schema=schema_dict, status="SUCCESS")
            return content


    def _heal_json(self, json_str):
        """Attempts to fix common LLM structural errors with minimal intrusion."""
        if not json_str:
            return "{}"

        # 0. Strip <think>...</think> reasoning blocks if present in text output
        json_str = re.sub(r"<think>.*?</think>", "", json_str, flags=re.DOTALL).strip()

        # 1. Normalize whitespace (tabs to spaces)
        json_str = json_str.replace('\t', ' ')

        # 2. Fix trailing commas (e.g., [1, 2, ] -> [1, 2])
        json_str = re.sub(r',\s*([\]\}])', r'\1', json_str)

        # 3. Fix unescaped newlines within values
        # This is a bit risky but common: "value": "line1\nline2"
        # We only escape newlines that are NOT followed by a potential key or object close
        # json_str = re.sub(r'\n(?!\s*["\}\]])', r'\\n', json_str)

        # 4. Ensure balanced braces (String-aware brace tracker)
        depth = 0
        in_string = False
        escaped = False
        for ch in json_str:
            if in_string:
                if escaped:
                    escaped = False
                elif ch == '\\':
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
        if depth > 0:
            json_str += '}' * depth

        return json_str

    def _chat_ollama(self, messages, stream, json_format, schema, **kwargs):
        url = f"{self.api_url}/api/chat"
        options = {
            "temperature": kwargs.pop("temperature", 0.2), # Deterministic temperature for schema extraction
            "num_predict": 16384,
            "num_ctx": 32768,
            "repeat_penalty": 1.1
        }
        provided_options = kwargs.pop("options", {})
        options.update(provided_options)
        
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "options": options,
            **kwargs
        }
        
        # Smart profile resolution (GBNF, think, etc.)
        profile = get_model_profile(self.model)
        model_lower = (self.model or "").lower()

        if "think" in kwargs:
            payload["think"] = kwargs.pop("think")
        elif "think" in profile:
            payload["think"] = profile["think"]
        
        use_gbnf = profile.get("enforce_gbnf", False)
        is_gemma_family = "gemma" in model_lower
        if schema and use_gbnf:
            payload["format"] = get_json_schema(schema, include_descriptions=False)
        elif (json_format or schema) and not is_gemma_family:
            payload["format"] = "json"
            
        try:
            response = requests.post(url, json=payload, timeout=(5, 600) if not stream else (5, None), stream=stream)
            response.raise_for_status()
            if stream: return self._iterate_ollama(response)
            data = response.json()
            if "error" in data: raise LLMError(f"Ollama API Error: {data['error']}")
            
            self.last_done_reason = data.get("done_reason")
            content = data.get("message", {}).get("content", "")
            
            # Bidirectional automatic fallback for empty/trivial response
            is_empty_or_trivial = not content.strip() or content.strip() in ("{}", "[]", "null")
            if schema and is_empty_or_trivial:
                import logging
                if use_gbnf:
                    logging.getLogger("librarian").warning(
                        f"Model '{self.model}' returned empty/trivial content ('{content.strip()}') under strict schema constraint. Automatically falling back to JSON mode..."
                    )
                    fallback_payload = dict(payload)
                    fallback_payload["format"] = "json"
                else:
                    logging.getLogger("librarian").warning(
                        f"Model '{self.model}' returned empty/trivial content ('{content.strip()}') in prompt-guided mode. Automatically retrying with strict JSON schema constraint..."
                    )
                    fallback_payload = dict(payload)
                    fallback_payload["format"] = get_json_schema(schema, include_descriptions=False)

                fb_response = requests.post(url, json=fallback_payload, timeout=(5, 600))
                fb_response.raise_for_status()
                fb_data = fb_response.json()
                if "error" in fb_data: raise LLMError(f"Ollama API Error: {fb_data['error']}")
                self.last_done_reason = fb_data.get("done_reason")
                content = fb_data.get("message", {}).get("content", "")

            self.last_raw_response = content
            return content
        except Exception as e:
            raise LLMError(f"Ollama Communication Error: {e}")

    def _iterate_ollama(self, response):
        """Generator for Ollama streaming responses."""
        try:
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    if "error" in chunk: raise LLMError(f"Ollama streaming error: {chunk['error']}")
                    content = chunk.get("message", {}).get("content", "")
                    if chunk.get("done"):
                        self.last_done_reason = chunk.get("done_reason")
                    if content: yield content
                    if chunk.get("done"): break
        except Exception as e:
            raise LLMError(f"Error during Ollama streaming: {e}")

    def _chat_openai(self, messages, stream, json_format, schema, **kwargs):
        url = f"{self.api_url}/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "temperature": kwargs.pop("temperature", 0.2), # Deterministic temperature for schema extraction
            "max_tokens": 4096,
            **kwargs
        }
        use_gbnf = config.get("enforce_gbnf", False)
        if schema and use_gbnf:
            schema_name = getattr(schema, "__name__", "ResponseSchema") if not isinstance(schema, dict) else "ResponseSchema"
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "strict": True, "schema": get_json_schema(schema, include_descriptions=False)}
            }
        elif json_format or schema:
            payload["response_format"] = {"type": "json_object"}
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=(5, 600) if not stream else (5, None), stream=stream)
            response.raise_for_status()
            if stream: return self._iterate_openai(response)
            data = response.json()
            choice = data.get("choices", [{}])[0]
            self.last_done_reason = choice.get("finish_reason")
            content = choice.get("message", {}).get("content", "")
            self.last_raw_response = content
            return content
        except Exception as e:
            raise LLMError(f"OpenAI Communication Error: {e}")

    def _iterate_openai(self, response):
        """Generator for OpenAI streaming responses."""
        try:
            full_content = ""
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]": break
                        chunk = json.loads(data_str)
                        content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        if content:
                            full_content += content
                            yield content
            self.last_raw_response = full_content
        except Exception as e:
            raise LLMError(f"Error during OpenAI streaming: {e}")

llm = LLMClient()
