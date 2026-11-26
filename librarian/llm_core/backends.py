# -*- coding: utf-8 -*-
"""HTTP communication and streaming backends for Ollama and OpenAI-compatible endpoints."""

import json
import logging
from typing import Any, Dict, List, Optional
import requests

from librarian.config import config
from librarian.schemas import get_json_schema
from .model_profiles import LLMError, get_model_profile

logger = logging.getLogger("librarian.llm")


class LLMBackendsMixin:
    """Mixin providing Ollama and OpenAI API transport and streaming handlers."""

    def _chat_ollama(self, messages, stream, json_format, schema, **kwargs):
        url = f"{self.api_url}/api/chat"
        profile = get_model_profile(self.model)
        cfg_num_ctx = config.get("num_ctx", 16384)
        cfg_max_tokens = config.get("max_tokens", 8192)

        options = {
            "num_predict": cfg_max_tokens,
            "num_ctx": cfg_num_ctx,
            "repeat_penalty": 1.1,  # Safeguard against catastrophic token degeneration loops
        }
        
        # Check if sampling parameters should remain at model default:
        # 1. Explicitly requested via use_default_params=True (e.g., expert audit)
        # 2. Unconstrained prose generation (not json_format and no schema)
        is_default_sampling = kwargs.pop("use_default_params", False) or (not schema and not json_format)
        
        if not is_default_sampling:
            options["temperature"] = kwargs.pop("temperature", 0.2) # Deterministic temperature for schema extraction
        elif "temperature" in kwargs:
            options["temperature"] = kwargs.pop("temperature")

        for opt_k in ("temperature", "num_ctx", "num_predict", "max_tokens", "top_p", "top_k", "repeat_penalty", "seed"):
            if opt_k in profile:
                target_k = "num_predict" if opt_k == "max_tokens" else opt_k
                options[target_k] = profile[opt_k]

        provided_options = kwargs.pop("options", {})
        if "max_tokens" in provided_options and "num_predict" not in provided_options:
            provided_options["num_predict"] = provided_options.pop("max_tokens")
        options.update(provided_options)

        # Support keep_alive from profile, kwargs, or wiki_config.json (e.g. keep_alive: 0 or "0m" to unload immediately)
        keep_alive = kwargs.pop("keep_alive", profile.get("keep_alive", config.data.get("keep_alive")))
        
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "options": options,
            **kwargs
        }
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive
        
        # Smart profile resolution (GBNF, think, etc.)
        model_lower = (self.model or "").lower()

        # Enforce profile["think"] (defaults to False for stability and preventing token exhaustion)
        # Can be overridden by kwargs or wiki_config.json['model_options'][model]['think']
        if "think" in kwargs:
            payload["think"] = kwargs.pop("think")
        elif "think" in profile and profile["think"] is not None:
            payload["think"] = profile["think"]
        else:
            payload["think"] = False
        
        use_gbnf = profile.get("enforce_gbnf", False)
        enforce_json = profile.get("enforce_json_format", True)
        if schema and use_gbnf:
            payload["format"] = get_json_schema(schema, include_descriptions=False)
        elif (json_format or schema) and enforce_json:
            payload["format"] = "json"
            
        timeout_val = (15, self.timeout) if not stream else (15, None)
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
        profile = get_model_profile(self.model)

        cfg_max_tokens = config.get("max_tokens", 8192)
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "max_tokens": cfg_max_tokens,
            **kwargs
        }
        is_default_sampling = kwargs.pop("use_default_params", False) or (not schema and not json_format)
        if not is_default_sampling:
            payload["temperature"] = kwargs.pop("temperature", 0.2) # Deterministic temperature for schema extraction
        elif "temperature" in kwargs:
            payload["temperature"] = kwargs.pop("temperature")

        for opt_k in ("temperature", "max_tokens", "num_predict", "top_p", "seed"):
            if opt_k in profile:
                target_k = "max_tokens" if opt_k == "num_predict" else opt_k
                payload[target_k] = profile[opt_k]
        if "num_predict" in payload and "max_tokens" not in payload:
            payload["max_tokens"] = payload.pop("num_predict")
        think_setting = kwargs.pop("think", profile.get("think"))
        if think_setting is False or think_setting == "none":
            # Standard OpenAI / llama-server / vLLM parameters to suppress CoT thinking
            payload["reasoning_effort"] = "none"
            payload["chat_template_kwargs"] = {"thinking": False}
        elif think_setting is True:
            # Enable CoT thinking for models that require reasoning traces (e.g. muse-glimmer)
            payload["reasoning_effort"] = "high"
            payload["chat_template_kwargs"] = {"thinking": True}
        elif isinstance(think_setting, str) and think_setting.lower() in ("low", "medium", "high", "max"):
            payload["reasoning_effort"] = think_setting.lower()
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
            timeout_val = (15, self.timeout) if not stream else (15, None)
            response = requests.post(url, json=payload, headers=headers, timeout=timeout_val, stream=stream)
            
            # Robust self-healing for LM Studio / local servers that do not accept {"type": "json_object"}
            # e.g., LM Studio errors with: "'response_format.type' must be 'json_schema' or 'text'"
            if response.status_code == 400 and payload.get("response_format", {}).get("type") == "json_object":
                err_text = response.text.lower()
                if "response_format" in err_text or "json_object" in err_text:
                    if schema:
                        # Auto-promote to json_schema mode
                        schema_name = getattr(schema, "__name__", "ResponseSchema") if not isinstance(schema, dict) else "ResponseSchema"
                        payload["response_format"] = {
                            "type": "json_schema",
                            "json_schema": {"name": schema_name, "strict": True, "schema": get_json_schema(schema, include_descriptions=False)}
                        }
                    else:
                        # Server only accepts 'text' or doesn't support json_object
                        payload.pop("response_format", None)
                    response = requests.post(url, json=payload, headers=headers, timeout=timeout_val, stream=stream)

            response.raise_for_status()
            if stream: return self._iterate_openai(response)
            data = response.json()
            choice = data.get("choices", [{}])[0]
            self.last_done_reason = choice.get("finish_reason")
            content = choice.get("message", {}).get("content", "")
            self.last_raw_response = content
            return content
        except requests.exceptions.HTTPError as e:
            err_body = ""
            try:
                err_body = f" - Server Response: {response.text}"
            except Exception:
                pass
            raise LLMError(f"OpenAI Communication Error: {e}{err_body}")
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

