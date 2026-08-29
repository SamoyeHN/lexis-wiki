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

class LLMClient:
    def __init__(self):
        self._refresh_config()
        self.last_raw_response = None

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

        # 3. Prompt-Guided JSON Mode: Inject JSON schema into System message if GBNF is disabled
        use_gbnf = config.get("enforce_gbnf", False)
        if schema and not use_gbnf:
            schema_dict = schema if isinstance(schema, dict) else get_json_schema(schema, include_descriptions=True)
            schema_json_str = json.dumps(schema_dict, indent=2, ensure_ascii=False)
            schema_prompt = f"### JSON SCHEMA REQUIREMENT ###\nRespond strictly with a valid JSON object matching this schema definition:\n```json\n{schema_json_str}\n```"
            system_msg = next((m for m in messages if m["role"] == "system"), None)
            if system_msg:
                if "### JSON SCHEMA REQUIREMENT ###" not in system_msg["content"]:
                    system_msg["content"] += f"\n\n{schema_prompt}"
            else:
                messages.insert(0, {"role": "system", "content": schema_prompt})

        # 4. Handle Constraints
        schema_for_api = schema
        force_json_mode = json_format or bool(schema and not use_gbnf)

        # 5. Call API
        if self.api_type == "openai":
            content = self._chat_openai(messages, stream, force_json_mode, schema_for_api, **kwargs)
        else:
            content = self._chat_ollama(messages, stream, force_json_mode, schema_for_api, **kwargs)

        # 5. Logging
        if not stream:
            from .logger import log_task
            mode_str = "STRICT_SCHEMA" if schema_for_api else ("JSON_MODE" if force_json_mode else "TEXT_MODE")
            system_prompt = "\n".join([m["content"] for m in messages if m["role"] == "system"])
            user_prompt = "\n".join([m["content"] for m in messages if m["role"] == "user"])
            
            # Schema logging: handle dict or dataclass
            schema_dict = None
            if isinstance(schema, dict):
                schema_dict = schema
            elif schema:
                schema_dict = get_json_schema(schema, include_descriptions=True)
                
            t_name = task_name or "chat"
            log_task(f"{t_name}_{mode_str}", system_prompt, user_prompt, content, schema=schema_dict)

        # 6. Post-Processing & Parsing
        if schema and not stream:
            try:
                # 6.1. CLEANING: Extract JSON from markdown or clutter
                json_str = content.strip()
                if not json_str:
                    raise LLMError(f"LLM returned empty response for task '{task_name or 'chat'}'.")
                
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
                
                # 6.2. HEALING
                final_json = self._heal_json(json_str)
                was_healed = (final_json != json_str)
                
                try:
                    data = json.loads(final_json)
                except json.JSONDecodeError:
                    # Light cleaning of unescaped newlines in values
                    # IMPROVED REGEX: Ignore newlines followed by valid JSON structural tokens
                    # Valid token starts: { [ } ] " , - 0-9 t f n :
                    final_json = re.sub(r'\n(?!\s*[, "\}\]\{\[0-9tfn\-\:])', r'\\n', final_json)
                    was_healed = True
                    data = json.loads(final_json)
                
                if was_healed:
                    import logging
                    logging.getLogger("librarian").warning(f"LLM JSON response for {task_name or 'chat'} required structural healing.")
                
                # 6.3. MAPPING
                if isinstance(schema, dict):
                    # If schema is a dict, return the data dict directly
                    # But we'll add a helper property if it's a list for auto-wrapping
                    if isinstance(data, list):
                        props = schema.get("properties", {})
                        list_field = next((k for k, v in props.items() if v.get("type") == "array"), "items")
                        data = {list_field: data}
                        if "title" in props: data["title"] = task_name or "Untitled"
                result_obj = validate_and_map(schema, data) if not isinstance(schema, dict) else data

                # 6.4. QA EVALUATION & RETRY LOOP
                # GEMINI.md mandate: retry loop at <80% quality threshold (one retry max)
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
                                "raw_response": content,
                                "parsed_json": dict_to_eval,
                            }
                            audit = LogEvaluator.evaluate_log(simulated_log)
                            composite = audit.get("composite_score", 100.0)
                            
                            if composite < 80.0:
                                flags = audit.get("flags", [])
                                scores = audit.get("scores", {})
                                # Identify lowest scoring dimension
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
                                retry_messages = [dict(m) for m in messages]
                                sys_msg = next((m for m in retry_messages if m["role"] == "system"), None)
                                if sys_msg:
                                    sys_msg["content"] += feedback_note
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

                return result_obj
            except LLMError:
                raise
            except Exception as e:
                snippet = content[:200].replace('\n', ' ')
                raise LLMError(f"Structured parsing failure: {e}. Snippet: {snippet}...")
        
        return content

    def _heal_json(self, json_str):
        """Attempts to fix common LLM structural errors with minimal intrusion."""
        if not json_str:
            return "{}"

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
            "temperature": kwargs.pop("temperature", 0.7), # Schema-first deterministic output
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
        
        use_gbnf = config.get("enforce_gbnf", False)
        if schema and use_gbnf:
            payload["format"] = get_json_schema(schema, include_descriptions=True)
        elif json_format or schema:
            payload["format"] = "json"
            
        try:
            response = requests.post(url, json=payload, timeout=(5, 600) if not stream else (5, None), stream=stream)
            response.raise_for_status()
            if stream: return self._iterate_ollama(response)
            data = response.json()
            if "error" in data: raise LLMError(f"Ollama API Error: {data['error']}")
            content = data.get("message", {}).get("content", "")
            
            # Automatic fallback for empty response under strict GBNF schema format
            if schema and use_gbnf and not content.strip():
                import logging
                logging.getLogger("librarian").warning(
                    f"Model '{self.model}' returned empty content under GBNF schema constraint. Automatically falling back to JSON mode..."
                )
                fallback_payload = dict(payload)
                fallback_payload["format"] = "json"
                fb_response = requests.post(url, json=fallback_payload, timeout=(5, 600))
                fb_response.raise_for_status()
                fb_data = fb_response.json()
                if "error" in fb_data: raise LLMError(f"Ollama API Error: {fb_data['error']}")
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
            "temperature": kwargs.pop("temperature", 0.7), # Schema-first output with standard 0.7 temperature
            "max_tokens": 4096,
            **kwargs
        }
        use_gbnf = config.get("enforce_gbnf", False)
        if schema and use_gbnf:
            schema_name = getattr(schema, "__name__", "ResponseSchema") if not isinstance(schema, dict) else "ResponseSchema"
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "strict": True, "schema": get_json_schema(schema, include_descriptions=True)}
            }
        elif json_format or schema:
            payload["response_format"] = {"type": "json_object"}
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=(5, 600) if not stream else (5, None), stream=stream)
            response.raise_for_status()
            if stream: return self._iterate_openai(response)
            data = response.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
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
