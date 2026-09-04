import requests
import json
import re
import datetime
import dataclasses
from pathlib import Path
from typing import List, Dict, Any, Type, Union, Optional
from .config import config
from .schemas import get_json_schema, validate_and_map

class LLMError(Exception):
    """Base exception for LLM related errors."""
    pass

# Declarative architectural model capability profiles
MODEL_CAPABILITY_PROFILES = [
    {
        # Reasoning models: Native strict schema, disable reasoning tokens during structured JSON extraction
        "match": ["granite", "deepseek-r1", "qwq", "ornith"],
        "enforce_gbnf": True,
        "think": False,
    },
    {
        # Tokenizer-constrained or large-vocab families (Gemma, Nemotron, Phi, Qwen): Prompt-guided JSON mode
        # Eliminates CPU-bound GBNF token-masking bottlenecks across large tokenizers (131k/248k) while maintaining 100% schema fidelity
        "match": ["gemma", "nemotron", "phi", "qwen"],
        "enforce_gbnf": False,
        "think": None,
    },
    {
        # Native GBNF / strict grammar standard families
        "match": ["llama", "mistral", "mixtral", "codestral", "muse", "hermes", "vicuna"],
        "enforce_gbnf": True,
        "think": None,
    },
]


def get_model_profile(model_name: str) -> dict:
    """
    Intelligently determines optimal execution profile for a model via a 3-tier precedence hierarchy:
    - Tier 1 (Base): Declarative architectural heuristics (Granite, Qwen, Gemma, etc.)
    - Tier 2 (Global): Global settings in wiki_config.json (e.g. global enforce_gbnf)
    - Tier 3 (Override): Explicit per-model overrides in wiki_config.json['model_options'][model_name]
    """
    m_lower = (model_name or "").lower()
    
    # Tier 1: Base architectural profile matching
    base_gbnf = False
    base_think = None
    for profile_rule in MODEL_CAPABILITY_PROFILES:
        if any(keyword in m_lower for keyword in profile_rule["match"]):
            base_gbnf = profile_rule["enforce_gbnf"]
            base_think = profile_rule["think"]
            break

    # Tier 2: Global config settings (if explicitly enforced)
    if config.data.get("enforce_gbnf") is True:
        base_gbnf = True

    # Tier 3: Per-model explicit overrides (Highest Precedence)
    model_opts = config.get("model_options", {}).get(model_name, {})
    resolved_gbnf = model_opts.get("enforce_gbnf", base_gbnf)
    resolved_think = model_opts.get("think", base_think)

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
        # Default read timeout: 1200 seconds (20 mins) to support deep reasoning and large parameter local models (27B-70B)
        self.timeout = config.get("request_timeout", 1200)

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

        # 3.1 Gemma compatibility: combine system instructions with the user message
        # because the Gemma chat template does not render a separate system turn.
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
        start_time = datetime.datetime.now()
        if self.api_type == "openai":
            content = self._chat_openai(messages, stream, force_json_mode, schema_for_api, **kwargs)
        else:
            content = self._chat_ollama(messages, stream, force_json_mode, schema_for_api, **kwargs)
        end_time = datetime.datetime.now()
        call_duration = (end_time - start_time).total_seconds()

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
                                        # Strip common prefixes like AUDIT:, DRAFT:, STEP:
                                        cand = re.sub(r'^(?:AUDIT|DRAFT|STEP\s*\d*)\s*:\s*', '', p, flags=re.IGNORECASE).strip()
                                        candidate = cand.split(" -")[0].split(" (")[0].strip()
                                        cand_tokens = [t.lower() for t in re.findall(r'[a-zA-Z]+', candidate.replace("[", "").replace("]", ""))]
                                        word_tokens = [t.lower() for t in re.findall(r'[a-zA-Z]+', cur_word)]
                                        
                                        # Match by exact token or stem/inflection (e.g., turn vs turned, lay vs laid, put vs putting)
                                        matched = False
                                        if cand_tokens and word_tokens:
                                            c0, w0 = cand_tokens[0], word_tokens[0]
                                            if c0 == w0:
                                                matched = True
                                            elif c0.startswith(w0[:3]) or w0.startswith(c0[:3]):
                                                matched = True
                                            elif any(tok in cand_tokens for tok in word_tokens if len(tok) >= 4):
                                                matched = True
                                                
                                        if matched:
                                            expr_item["word"] = candidate
                                            break

                # Auto-sync slotted pattern_formula from design_audit for grammar if missing slots
                if isinstance(data, dict) and "grammar_patterns" in data and isinstance(data["grammar_patterns"], list):
                    for g_item in data["grammar_patterns"]:
                        if isinstance(g_item, dict):
                            cur_formula = str(g_item.get("pattern_formula", "")).strip()
                            cur_audit = str(g_item.get("design_audit", "")).strip()
                            # If formula lacks brackets or is overly generic but design_audit derived a slotted formula
                            if ("[" not in cur_formula) and ("[" in cur_audit):
                                parts = [p.strip() for p in cur_audit.replace("->", "➔").split("➔")]
                                for p in reversed(parts):
                                    if "[" in p and "]" in p:
                                        cand_formula = re.sub(r'^(?:AUDIT|DRAFT|STEP\s*\d*)\s*:\s*', '', p, flags=re.IGNORECASE).strip()
                                        if "[" in cand_formula:
                                            g_item["pattern_formula"] = cand_formula
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

                # 6.4. QA EVALUATION & RETRY LOOP (Max 2 Retries with Structured Surgical Feedback)
                retry_count = kwargs.pop("_qa_retry_count", 0)
                max_qa_retries = 2
                if retry_count < max_qa_retries and not kwargs.get("_disable_qa_retry", False):
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
                            composite = audit.get("composite_score")
                            if composite is None:
                                composite = 100.0
                            
                            flags = audit.get("flags", [])
                            has_fatal_flags = any("does not appear in quoted sentence" in f or "duplicate" in f or "copy-pasted definition" in f for f in flags)

                            if composite < 80.0 or has_fatal_flags:
                                scores = {k: v for k, v in audit.get("scores", {}).items() if v is not None}
                                lowest_dim = min(scores.keys(), key=lambda k: scores[k]) if scores else "pedagogical_quality"
                                
                                # Format clear, surgical feedback for the model
                                issue_bullets = "\n".join([f"- {f}" for f in flags[:5]])
                                feedback_note = (
                                    f"\n\n### 🚨 [QUALITY AUDIT RETRY #{retry_count + 1}/{max_qa_retries} - Score: {composite}/100]\n"
                                    f"Lowest dimension: {lowest_dim}.\n"
                                    f"Please address and resolve these critical pedagogical issues:\n"
                                    f"{issue_bullets}\n\n"
                                    f"MANDATORY FIX RULES:\n"
                                    f"1. ZERO HALLUCINATION: All words and quoted sentences MUST physically exist verbatim in the source text.\n"
                                    f"2. Every quoted sentence MUST literally contain the target word/expression.\n"
                                    f"3. Eliminate duplicate items and ensure each definition is distinct and context-specific.\n"
                                    f"4. Quality > Quota: Do not pad with nonexistent words."
                                )
                                import logging
                                logging.getLogger("librarian").info(
                                    f"QA score {composite}/100 (<80% or fatal flags) for {t_name}. Retrying ({retry_count + 1}/{max_qa_retries}) with surgical feedback..."
                                )
                                # Log failed attempt with QA_LOW_SCORE
                                from .logger import log_task
                                log_task(
                                    f"{t_name}_{mode_str}",
                                    system_prompt,
                                    user_prompt,
                                    final_json,
                                    schema=schema_dict,
                                    status="FAILED",
                                    failure_category="QA_LOW_SCORE",
                                    mode=mode_str,
                                    api_constraint=schema_for_api if mode_str == "STRICT_SCHEMA" else ("json" if force_json_mode else None),
                                )

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
                                    _qa_retry_count=retry_count + 1,
                                    **kwargs
                                )
                    except Exception as eval_err:
                        import logging
                        logging.getLogger("librarian").warning(f"Evaluator check skipped due to error: {eval_err}")

                # 6.5. LOG FINAL HEALED & VALIDATED RESPONSE
                from .logger import log_task
                log_task(
                    f"{t_name}_{mode_str}",
                    system_prompt,
                    user_prompt,
                    final_json,
                    schema=schema_dict,
                    status="SUCCESS",
                    mode=mode_str,
                    api_constraint=schema_for_api if mode_str == "STRICT_SCHEMA" else ("json" if force_json_mode else None),
                    duration=call_duration,
                    start_time=start_time,
                    end_time=end_time,
                )

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
                log_task(
                    f"{t_name}_{mode_str}",
                    system_prompt,
                    user_prompt,
                    content,
                    schema=schema_dict,
                    status="FAILED",
                    failure_category=failure_cat,
                    mode=mode_str,
                    api_constraint=schema_for_api if mode_str == "STRICT_SCHEMA" else ("json" if force_json_mode else None),
                    duration=call_duration,
                    start_time=start_time,
                    end_time=end_time,
                )
                raise LLMError(f"Failed to generate structured data matching schema: {e}")
        else:
            if not stream:
                from .logger import log_task
                log_task(
                    f"{t_name}_{mode_str}",
                    system_prompt,
                    user_prompt,
                    content,
                    schema=schema_dict,
                    status="SUCCESS",
                    mode=mode_str,
                    api_constraint=schema_for_api if mode_str == "STRICT_SCHEMA" else ("json" if force_json_mode else None),
                    duration=call_duration,
                    start_time=start_time,
                    end_time=end_time,
                )
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

        # 2.1. Fix missing commas between key-value pairs or array elements
        # e.g., "imitation_example": "..."\n  "common_mistakes": "..."
        json_str = re.sub(r'("|\d+|true|false|null|\}|\])\s*\n(\s*")', r'\1,\n\2', json_str)
        json_str = re.sub(r'(\})\s*\n(\s*\{)', r'\1,\n\2', json_str)

        # 2.2. Strip dangling truncated elements at the end of arrays/objects
        # e.g., [..., "{\n] or [..., "abc\n] or [..., {\n] where generation was cut off
        json_str = re.sub(r',\s*["\']\{?["\']\s*([\]\}])', r'\1', json_str)
        json_str = re.sub(r',\s*\{?\s*([\]\}])', r'\1', json_str)

        # 3. Fix unescaped newlines within values
        # This is a bit risky but common: "value": "line1\nline2"
        # We only escape newlines that are NOT followed by a potential key or object close
        # json_str = re.sub(r'\n(?!\s*["\}\]])', r'\\n', json_str)

        # 4. Ensure balanced braces and brackets (String-aware tracker)
        brace_depth = 0
        bracket_depth = 0
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
                brace_depth += 1
            elif ch == '}':
                brace_depth = max(0, brace_depth - 1)
            elif ch == '[':
                bracket_depth += 1
            elif ch == ']':
                bracket_depth = max(0, bracket_depth - 1)

        # Close any dangling open string
        if in_string:
            json_str += '"'

        # Close open brackets first, then open braces
        if bracket_depth > 0:
            json_str += ']' * bracket_depth
        if brace_depth > 0:
            json_str += '}' * brace_depth

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
        if schema and use_gbnf:
            payload["format"] = get_json_schema(schema, include_descriptions=False)
        elif json_format or schema:
            payload["format"] = "json"
            
        timeout_val = (5, self.timeout) if not stream else (5, None)
        try:
            response = requests.post(url, json=payload, timeout=timeout_val, stream=stream)
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

                fb_response = requests.post(url, json=fallback_payload, timeout=timeout_val)
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
        profile = get_model_profile(self.model)
        use_gbnf = profile.get("enforce_gbnf", False)
        if schema and use_gbnf:
            schema_name = getattr(schema, "__name__", "ResponseSchema") if not isinstance(schema, dict) else "ResponseSchema"
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "strict": True, "schema": get_json_schema(schema, include_descriptions=False)}
            }
        elif json_format or schema:
            payload["response_format"] = {"type": "json_object"}
        try:
            timeout_val = (5, self.timeout) if not stream else (5, None)
            response = requests.post(url, json=payload, headers=headers, timeout=timeout_val, stream=stream)
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
