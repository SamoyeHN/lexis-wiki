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
        # Granite family (e.g. granite4.2:30b): Prompt-guided JSON mode with thinking disabled
        # Avoids GBNF grammar token-masking repetition loops and reasoning stalls while maintaining diverse, schema-compliant output
        "match": ["granite"],
        "enforce_gbnf": False,
        "think": False,
    },
    {
        # Dedicated reasoning models: Native strict schema, disable reasoning tokens during structured JSON extraction
        "match": ["deepseek-r1", "qwq", "ornith"],
        "enforce_gbnf": True,
        "think": False,
    },
    {
        # Specialized translation checkpoints (e.g. translategemma):
        # Setting API format: "json" causes translategemma to immediately emit empty '{}'
        # because its weights were trained specifically for direct translation pairs.
        # It follows prompt-injected JSON schema natively without the Ollama format constraint.
        "match": ["translategemma"],
        "enforce_gbnf": False,
        "enforce_json_format": False,
        "think": False,
    },
    {
        # Standard Gemma family (e.g. gemma2, gemma4):
        # Fully compatible with prompt-guided JSON mode (format: "json")
        "match": ["gemma"],
        "enforce_gbnf": False,
        "enforce_json_format": True,
        "think": False,
    },
    {
        # Tokenizer-constrained or large-vocab families (Nemotron, Phi, Qwen): Prompt-guided JSON mode
        # Eliminates CPU-bound GBNF token-masking bottlenecks across large tokenizers (131k/248k) while maintaining 100% schema fidelity
        "match": ["nemotron", "phi", "qwen"],
        "enforce_gbnf": False,
        "think": False,
    },
    {
        # Muse family (e.g., muse-glimmer): Prompt-guided JSON mode with thinking disabled
        # Avoids 300s+ hidden reasoning loops and prompt regurgitation while preserving in-schema design_audit.
        "match": ["muse"],
        "enforce_gbnf": False,
        "think": False,
    },
    {
        # Fragile / Prompt-based families (Hermes custom tags, Vicuna legacy attention):
        # Hermes uses native XML/markdown tags (<tool_call>, <thought>) which crash under rigid GBNF clamps.
        # Vicuna lacks modern attention calibration for strict character-level grammar masking.
        "match": ["hermes", "vicuna"],
        "enforce_gbnf": False,
        "think": False,
    },
    {
        # Native GBNF / strict grammar standard families (Llama 3+, Mistral, Mixtral, Codestral):
        # Perfectly compatible with context-free grammar parsers (llama.cpp, vLLM, SGLang).
        # Defaults to think: False for speed; dynamically sets think: True if running a reasoning/thinking variant.
        "match": ["llama", "mistral", "mixtral", "codestral"],
        "enforce_gbnf": True,
        "think": False,
    },
]


def get_model_profile(model_name: str) -> dict:
    """
    Intelligently determines optimal execution profile for a model via a 3-tier precedence hierarchy:
    - Tier 1 (Base): Declarative architectural heuristics (Granite, Qwen, Gemma, Llama, Hermes, etc.)
    - Tier 2 (Global): Global settings in wiki_config.json (e.g. global enforce_gbnf)
    - Tier 3 (Override): Explicit per-model overrides in wiki_config.json['model_options'][model_name]
    """
    m_lower = (model_name or "").lower()
    
    # Tier 1: Base architectural profile matching
    base_gbnf = False
    base_think = None
    base_json_format = True
    for profile_rule in MODEL_CAPABILITY_PROFILES:
        if any(keyword in m_lower for keyword in profile_rule["match"]):
            base_gbnf = profile_rule["enforce_gbnf"]
            base_think = profile_rule["think"]
            base_json_format = profile_rule.get("enforce_json_format", True)
            # Dynamic reasoning check for Native GBNF models:
            # If running a thinking/reasoning fine-tune (e.g. Llama-3-Thinking), allow thinking unconstrained
            # before the GBNF grammar clamp locks onto the final JSON output.
            if base_gbnf is True and ("thinking" in m_lower or "reasoning" in m_lower):
                base_think = True
            break

    # Tier 2: Global config settings (if explicitly enforced)
    if config.data.get("enforce_gbnf") is True:
        base_gbnf = True

    # Tier 3: Per-model explicit overrides (Highest Precedence)
    model_opts = config.get("model_options", {}).get(model_name, {})
    resolved_gbnf = model_opts.get("enforce_gbnf", base_gbnf)
    resolved_think = model_opts.get("think", base_think)
    resolved_json_format = model_opts.get("enforce_json_format", base_json_format)

    profile = dict(model_opts)
    profile["enforce_gbnf"] = resolved_gbnf
    profile["enforce_json_format"] = resolved_json_format
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

    def list_models(self, api_type=None, api_url=None, api_key=None):
        """Fetches models from Ollama (/api/tags) or OpenAI-compatible servers like llama-server (/v1/models).

        Optional overrides (api_type / api_url / api_key) allow previewing a different
        engine/endpoint than the one currently saved in config. The dashboard uses this
        when the user switches the API Engine dropdown before saving, so the model list
        reflects the selected engine instead of silently staying on the saved one.
        Falls back to the saved config when an override is not provided.
        """
        self._refresh_config()
        api_type = api_type or self.api_type
        api_url = (api_url or self.api_url).rstrip('/')
        api_key = api_key if api_key else self.api_key
        # Surface reachability so the dashboard can distinguish "server offline"
        # (request failed) from "server online but lists no models".
        self.last_models_ok = False
        self.last_models_error = None
        if api_type == "ollama":
            url = f"{api_url}/api/tags"
            try:
                response = requests.get(url, timeout=2)
                response.raise_for_status()
                data = response.json()
                self.last_models_ok = True
                return [m["name"] for m in data.get("models", [])]
            except Exception as e:
                self.last_models_error = str(e)
                return []
        elif api_type == "openai":
            url = f"{api_url}/v1/models"
            headers = {}
            if api_key and api_key != "ollama":
                headers["Authorization"] = f"Bearer {api_key}"
            try:
                response = requests.get(url, headers=headers, timeout=2)
                response.raise_for_status()
                data = response.json()
                # Support OpenAI standard {"data": [{"id": "model_name"}]} and llama-server {"models": [...]}
                models = []
                if "data" in data and isinstance(data["data"], list):
                    models.extend([m.get("id") for m in data["data"] if m.get("id")])
                elif "models" in data and isinstance(data["models"], list):
                    models.extend([m.get("name") or m.get("model") for m in data["models"] if (m.get("name") or m.get("model"))])
                self.last_models_ok = True
                return models
            except Exception as e:
                self.last_models_error = str(e)
                return []
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

                # Auto-align quoted_sentence for vocabulary & expressions if target word exists in source passage
                # Solves off-by-one sentence mismatches (e.g. model quoting an adjacent sentence) without masking hallucinations
                self._align_quoted_sentences(data, user_prompt)

                # 6.3. MAPPING
                # Auto-unwrap agentic wrappers (e.g. {"self": {...}}, {"response": {...}}, {"data": {...}})
                if isinstance(data, dict):
                    for wrapper_key in ("self", "response", "result", "data", "output", "content"):
                        if wrapper_key in data and isinstance(data[wrapper_key], dict) and len(data) == 1:
                            data = data[wrapper_key]
                            break

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
                                critique_prompt = (
                                    f"### 🚨 [QUALITY AUDIT REVIEW #{retry_count + 1}/{max_qa_retries} - Score: {composite:.1f}/100]\n"
                                    f"Lowest dimension: {lowest_dim}.\n"
                                    f"Your previous response had the following critical pedagogical issues:\n"
                                    f"{issue_bullets}\n\n"
                                    f"MANDATORY FIX RULES:\n"
                                    f"1. ZERO HALLUCINATION: All words and quoted sentences MUST physically exist verbatim in the source text.\n"
                                    f"2. Every quoted sentence MUST literally contain the target word/expression.\n"
                                    f"3. Eliminate duplicate items and ensure each definition is distinct and context-specific.\n"
                                    f"4. Quality > Quota: Do not pad with nonexistent words.\n\n"
                                    f"Please output the corrected, complete JSON object resolving these issues."
                                )
                                import logging
                                logging.getLogger("librarian").info(
                                    f"QA score {composite}/100 (<80% or fatal flags) for {t_name}. Retrying ({retry_count + 1}/{max_qa_retries}) via multi-turn self-correction..."
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
                                    duration=call_duration,
                                    start_time=start_time,
                                    end_time=end_time,
                                )

                                # Multi-turn self-correction:
                                # Retain base prompt, append model's prior output as assistant turn, and append reviewer critique
                                # If this is a subsequent retry (retry_count > 0), prune prior critique turns to keep context tidy:
                                # [Original Prompt] -> [Latest Assistant Output] -> [Latest Reviewer Critique]
                                base_messages = []
                                for m in messages:
                                    if m["role"] == "assistant" or (m["role"] == "user" and "### 🚨 [QUALITY AUDIT REVIEW" in m.get("content", "")):
                                        break
                                    base_messages.append(dict(m))

                                retry_messages = list(base_messages)
                                retry_messages.append({"role": "assistant", "content": final_json})
                                retry_messages.append({"role": "user", "content": critique_prompt})
                                
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

    def _align_quoted_sentences(self, data, user_prompt):
        """
        Auto-aligns quoted_sentence for vocabulary and expressions items if the model quoted
        an adjacent sentence in the passage instead of the exact sentence containing the word.
        STRICT GROUNDING RULE: If the target word is NOT physically in the source passage,
        this method leaves it untouched so that QA evaluation flags it as an ungrounded hallucination.
        """
        if not isinstance(data, dict) or not user_prompt:
            return

        # Extract the passage content from user prompt
        source_content = user_prompt.split("CONTENT:", 1)[1].strip() if "CONTENT:" in user_prompt else user_prompt
        # Strip any trailing retry critique prompts from user_prompt
        source_content = source_content.split("### 🚨 [QUALITY AUDIT REVIEW", 1)[0].strip()
        if not source_content:
            return

        # Pre-segment source passage into sentences
        norm_source = re.sub(r'\s+', ' ', source_content).strip()
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', norm_source) if len(s.strip()) > 10]
        if not sentences:
            return

        for array_key in ("vocabulary", "expressions"):
            items = data.get(array_key)
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                target_word = str(item.get("word", "")).strip()
                cur_quote = str(item.get("quoted_sentence", "")).strip()
                audit_str = str(item.get("design_audit", "")).strip()

                if not target_word:
                    continue

                # Check if target word already exists in cur_quote
                clean_target = re.sub(r'\[.*?\]|\(.*?\)', '', target_word).strip().lower()
                clean_quote_core = re.sub(r'[^\w\s]', ' ', cur_quote.lower())
                quote_tokens = set(clean_quote_core.split())

                target_tokens = [t for t in clean_target.split() if t]
                already_in_quote = False
                if target_tokens:
                    matches = sum(1 for t in target_tokens if (t in quote_tokens or (len(t) >= 4 and any(qt.startswith(t[:4]) for qt in quote_tokens))))
                    if matches >= max(1, len(target_tokens) // 2 + (1 if len(target_tokens) % 2 == 1 else 0)):
                        already_in_quote = True

                if already_in_quote:
                    continue

                # Extract surface word from design_audit if available (e.g. AUDIT: [forethought] -> [forethought])
                surface_word = None
                if "[" in audit_str and "]" in audit_str:
                    bracket_m = re.search(r'\[(.*?)\]', audit_str)
                    if bracket_m:
                        surface_word = bracket_m.group(1).strip().lower()

                # Build matching logic to locate the genuine sentence in the source text:
                # 1. Multi-token expression matching (e.g. "take [something] for granted" -> "take .* for granted")
                # 2. Single-word vocabulary matching (with stems & surface word)
                matching_sentences = []

                # Clean tokens, ignoring stop slots like something/somebody/one's/sb/sth
                stop_slots = {"something", "somebody", "ones", "one's", "someone", "sb", "sth"}
                core_tokens = [t for t in re.sub(r'\[.*?\]|\(.*?\)', ' ', target_word).lower().split() if t and t not in stop_slots]

                if array_key == "expressions" and len(core_tokens) >= 2:
                    # Construct wildcard regex: \btake\W+(?:\w+\W+){0,6}for\W+(?:\w+\W+){0,6}granted\b
                    def _get_token_pattern(tok):
                        irreg = {
                            'take': 'take|took|taken|taking',
                            'bring': 'bring|brought|bringing',
                            'lay': 'lay|laid|laying',
                            'set': 'set|setting',
                            'put': 'put|putting',
                            'make': 'make|made|making',
                            'give': 'give|gave|given|giving',
                            'come': 'come|came|coming',
                            'go': 'go|went|gone|going',
                            'keep': 'keep|kept|keeping',
                            'hold': 'hold|held|holding',
                            'find': 'find|found|finding',
                        }
                        if tok in irreg:
                            return f"(?:{irreg[tok]})"
                        parts = [re.escape(tok)]
                        if tok.endswith('ing') and len(tok) > 5: parts.append(re.escape(tok[:-3]))
                        elif tok.endswith('ed') and len(tok) > 4: parts.append(re.escape(tok[:-2]))
                        elif tok.endswith('s') and len(tok) > 3: parts.append(re.escape(tok[:-1]))
                        return f"(?:{'|'.join(parts)})"

                    token_regexes = [_get_token_pattern(tok) for tok in core_tokens]
                    phrase_regex = r'\b' + r'\W+(?:\w+\W+){0,6}'.join(token_regexes) + r'\b'

                    for s in sentences:
                        if re.search(phrase_regex, s, re.IGNORECASE):
                            matching_sentences.append(s)
                else:
                    # Single word or fallback: require all core tokens if multiple, or stem match if single
                    candidates = set(core_tokens)
                    if surface_word:
                        candidates.add(surface_word)
                    for tok in list(candidates):
                        if tok.endswith('ing') and len(tok) > 5: candidates.add(tok[:-3])
                        if tok.endswith('ed') and len(tok) > 4: candidates.add(tok[:-2])
                        if tok.endswith('es') and len(tok) > 4: candidates.add(tok[:-2])
                        elif tok.endswith('s') and len(tok) > 3: candidates.add(tok[:-1])

                    for s in sentences:
                        s_tokens = set(re.sub(r'[^\w\s]', ' ', s.lower()).split())
                        # If single word, any candidate matches; if multi-word, at least 70% of core tokens must match
                        if len(core_tokens) <= 1:
                            if any(cand in s_tokens or (len(cand) >= 4 and any(st.startswith(cand[:4]) for st in s_tokens)) for cand in candidates):
                                matching_sentences.append(s)
                        else:
                            matched = sum(1 for tok in core_tokens if (tok in s_tokens or (len(tok) >= 4 and any(st.startswith(tok[:4]) for st in s_tokens))))
                            if matched == len(core_tokens):
                                matching_sentences.append(s)

                # Grounding check: if not in source text at all, DO NOT TOUCH (it is a true hallucination)
                if not matching_sentences:
                    continue

                # Select best matching sentence (closest in text or single match)
                chosen_sentence = None
                if len(matching_sentences) == 1:
                    chosen_sentence = matching_sentences[0]
                else:
                    # If multiple sentences contain candidate, pick the one closest to cur_quote position in source
                    cur_idx = norm_source.find(cur_quote[:30]) if cur_quote else -1
                    if cur_idx != -1:
                        chosen_sentence = min(matching_sentences, key=lambda s: abs(norm_source.find(s[:30]) - cur_idx))
                    else:
                        chosen_sentence = matching_sentences[0]

                if chosen_sentence and chosen_sentence != cur_quote:
                    import logging
                    logging.getLogger("librarian").info(
                        f"Auto-aligned quoted_sentence for '{target_word}' from adjacent quote to genuine source sentence: '{chosen_sentence[:60]}...'"
                    )
                    item["quoted_sentence"] = chosen_sentence

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
        enforce_json = profile.get("enforce_json_format", True)
        if schema and use_gbnf:
            payload["format"] = get_json_schema(schema, include_descriptions=False)
        elif (json_format or schema) and enforce_json:
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
            "max_tokens": 16384,
            **kwargs
        }
        profile = get_model_profile(self.model)
        think_setting = kwargs.pop("think", profile.get("think"))
        if think_setting is False:
            # Standard OpenAI / llama-server / vLLM parameters to suppress CoT thinking
            payload["reasoning_effort"] = "none"
            payload["chat_template_kwargs"] = {"thinking": False}
        elif think_setting is True:
            # Enable CoT thinking for models that require reasoning traces (e.g. muse-glimmer)
            payload["reasoning_effort"] = "high"
            payload["chat_template_kwargs"] = {"thinking": True}

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
